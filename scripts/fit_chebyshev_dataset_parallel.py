#!/usr/bin/env python3
"""Fit Chebyshev coefficients for every mesh in a dataset, in parallel.

The source dataset is never modified.  Coefficients are written to a separate,
sharded and resumable dataset with an incremental manifest and diagnostics.
"""

from __future__ import annotations

import os

# One BLAS thread per worker prevents process x BLAS oversubscription.  Set
# PLATEAU_BLAS_THREADS before launch only if deliberate nested parallelism is wanted.
_BLAS_THREADS = os.environ.get("PLATEAU_BLAS_THREADS", "1")
for _VARIABLE in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_VARIABLE] = _BLAS_THREADS

import argparse
from dataclasses import dataclass
import hashlib
import json
from math import comb
import re
import sys
import tempfile
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from minsurf_nn.spectral import coefficient_energy, fit_triangulation  # noqa: E402


FORMAT_VERSION = 1
OUTPUT_RE = re.compile(r"^sample_(\d+)_chebyshev_p(\d+)\.npz$")


@dataclass(frozen=True)
class SourceSample:
    index: int
    sample_id: str
    split: str
    seed: int
    path: str


def _default_workers() -> int:
    return max(1, min(6, (os.cpu_count() or 2) - 1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="source contour/mesh manifest.jsonl")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("dataset/plateau-100k-chebyshev-p10"))
    parser.add_argument("--degree", type=int, default=10)
    parser.add_argument("--surface-samples", type=int, default=4096)
    parser.add_argument("--boundary-samples", type=int, default=1024)
    parser.add_argument("--offset-distance", type=float, default=0.03)
    parser.add_argument("--boundary-weight", type=float, default=10.0)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--rcond", type=float, default=None)
    parser.add_argument("--seed", type=int, default=700_000)
    parser.add_argument("--workers", type=int, default=0,
                        help="worker processes; zero chooses up to six automatically")
    parser.add_argument("--queue-factor", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-condition", type=float, default=1e8,
                        help="reject fits with a larger least-squares condition number")
    parser.add_argument("--allow-rank-deficient", action="store_true")
    parser.add_argument("--limit", type=int, default=0,
                        help="process only the first N records; zero means all")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.degree < 1:
        raise ValueError("degree must be positive")
    if args.surface_samples < 1 or args.boundary_samples < 1:
        raise ValueError("surface-samples and boundary-samples must be positive")
    if not 0.0 < args.offset_distance < 1.0:
        raise ValueError("offset-distance must lie between zero and one")
    if args.boundary_weight <= 0.0 or args.ridge < 0.0:
        raise ValueError("boundary-weight must be positive and ridge non-negative")
    if args.rcond is not None and args.rcond <= 0.0:
        raise ValueError("rcond must be positive when specified")
    if args.seed < 0 or args.workers < 0 or args.queue_factor < 1:
        raise ValueError("seed/workers must be non-negative and queue-factor positive")
    if args.shard_size < 0 or args.retries < 0 or args.progress_every < 1:
        raise ValueError("shard-size/retries must be non-negative and progress-every positive")
    if args.max_condition <= 0.0 or args.limit < 0:
        raise ValueError("max-condition must be positive and limit non-negative")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source_manifest(path: Path, limit: int) -> list[SourceSample]:
    samples = []
    seen_indices = set()
    with path.open(encoding="utf-8") as stream:
        for position, line in enumerate(stream):
            if not line.strip():
                continue
            record = json.loads(line)
            index = int(record.get("index", position))
            if index < 0 or index in seen_indices:
                raise ValueError(f"source manifest contains an invalid/duplicate index {index}")
            relative = record.get("path", record.get("file"))
            if not isinstance(relative, str):
                raise ValueError(f"source record {index} has no NPZ path")
            source_path = (path.parent / relative).resolve()
            samples.append(SourceSample(
                index=index,
                sample_id=str(record.get("sample_id", f"sample_{index:06d}")),
                split=str(record.get("split", "train")),
                seed=int(record.get("seed", index)),
                path=str(source_path),
            ))
            seen_indices.add(index)
            if limit and len(samples) >= limit:
                break
    if not samples:
        raise ValueError("source manifest is empty")
    samples.sort(key=lambda sample: sample.index)
    return samples


def _material_config(args: argparse.Namespace, manifest: Path,
                     manifest_hash: str, sample_count: int) -> Dict[str, object]:
    return {
        "format_version": FORMAT_VERSION,
        "algorithm": "local_signed_distance_total_degree_chebyshev",
        "source_manifest": str(manifest),
        "source_manifest_sha256": manifest_hash,
        "source_samples": sample_count,
        "limit": int(args.limit),
        "degree": int(args.degree),
        "coefficient_count": comb(args.degree + 3, 3),
        "surface_samples": int(args.surface_samples),
        "boundary_samples": int(args.boundary_samples),
        "offset_distance": float(args.offset_distance),
        "boundary_weight": float(args.boundary_weight),
        "ridge": float(args.ridge),
        "rcond": args.rcond,
        "seed": int(args.seed),
        "shard_size": int(args.shard_size),
        "max_condition": float(args.max_condition),
        "allow_rank_deficient": bool(args.allow_rank_deficient),
    }


def _config_fingerprint(config: Mapping[str, object]) -> str:
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{path.stem}.", suffix=".tmp",
                dir=path.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _ensure_config(output_dir: Path, config: Mapping[str, object]) -> None:
    path = output_dir / "fitting_config.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            changed = sorted(key for key in set(existing) | set(config)
                             if existing.get(key) != config.get(key))
            detail = ", ".join(
                f"{key}: {existing.get(key)!r} -> {config.get(key)!r}" for key in changed)
            raise RuntimeError(
                f"output directory has a different fitting configuration ({detail}); "
                "choose another --output-dir")
        return
    occupied = [entry for entry in output_dir.iterdir() if entry.name != path.name]
    if occupied:
        raise RuntimeError("output directory is non-empty but has no fitting_config.json")
    _atomic_write_text(path, json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _shard_dir(output_dir: Path, index: int, shard_size: int) -> Path:
    return output_dir if shard_size == 0 else output_dir / f"shard-{index // shard_size:05d}"


def _output_name(index: int, degree: int) -> str:
    return f"sample_{index:06d}_chebyshev_p{degree}.npz"


def _fit_one(sample: SourceSample, output_dir_string: str,
             config: Mapping[str, object], fingerprint: str) -> Dict[str, object]:
    output_dir = Path(output_dir_string)
    source_path = Path(sample.path)
    if not source_path.is_file():
        raise FileNotFoundError(f"source NPZ is missing: {source_path}")
    with np.load(source_path, allow_pickle=False) as source:
        vertices = np.asarray(source["surface_vertices"], dtype=np.float64)
        faces = np.asarray(source["faces"], dtype=np.int32)

    fit_seed = int(config["seed"]) + sample.index
    result = fit_triangulation(
        vertices, faces,
        degree=int(config["degree"]),
        surface_samples=int(config["surface_samples"]),
        boundary_samples=int(config["boundary_samples"]),
        offset_distance=float(config["offset_distance"]),
        boundary_weight=float(config["boundary_weight"]),
        ridge=float(config["ridge"]),
        rcond=config["rcond"],
        seed=fit_seed,
    )
    expected_rank = comb(int(config["degree"]) + 3, 3)
    rank = int(result.diagnostics["rank"])
    condition = float(result.diagnostics["condition_number"])
    if rank != expected_rank and not bool(config["allow_rank_deficient"]):
        raise RuntimeError(f"rank-deficient fit: {rank}/{expected_rank}")
    if not np.isfinite(condition) or condition > float(config["max_condition"]):
        raise RuntimeError(
            f"fit condition number {condition:.6g} exceeds {float(config['max_condition']):.6g}")

    sample_dir = _shard_dir(output_dir, sample.index, int(config["shard_size"]))
    output_path = sample_dir / _output_name(sample.index, int(config["degree"]))
    source_relative = os.path.relpath(source_path, output_dir)
    diagnostics = {key: (int(value) if isinstance(value, (int, np.integer)) else float(value))
                   for key, value in result.diagnostics.items()}
    metadata = {
        "format_version": FORMAT_VERSION,
        "representation": "total_degree_chebyshev",
        "domain": "[-1,1]^3",
        "source_sample_id": sample.sample_id,
        "source_index": sample.index,
        "source_path": source_relative,
        "source_seed": sample.seed,
        "fit_seed": fit_seed,
        "config_fingerprint": fingerprint,
        "degree": int(config["degree"]),
        "surface_samples": int(config["surface_samples"]),
        "boundary_samples": int(config["boundary_samples"]),
        "offset_distance": float(config["offset_distance"]),
        "boundary_weight": float(config["boundary_weight"]),
        "ridge": float(config["ridge"]),
        "rcond": config["rcond"],
        "diagnostics": diagnostics,
    }
    _atomic_savez(
        output_path,
        coefficients=result.coefficients.astype(np.float64),
        indices=result.indices.astype(np.int16),
        degree=np.asarray(result.degree, dtype=np.int16),
        coefficient_energy=coefficient_energy(
            result.coefficients, result.indices).astype(np.float64),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )
    return {
        "index": sample.index,
        "sample_id": sample.sample_id,
        "group_id": sample.sample_id,
        "split": sample.split,
        "source_path": source_relative,
        "coefficient_path": output_path.relative_to(output_dir).as_posix(),
        "path": output_path.relative_to(output_dir).as_posix(),
        "degree": int(config["degree"]),
        "coefficient_count": expected_rank,
        "source_seed": sample.seed,
        "fit_seed": fit_seed,
        "diagnostics": diagnostics,
    }


def _record_index(record: Mapping[str, object]) -> Optional[int]:
    value = record.get("index")
    return value if isinstance(value, int) else None


def _read_output_manifest(path: Path, output_dir: Path,
                          source_by_index: Mapping[int, SourceSample]) -> Dict[int, Dict[str, object]]:
    records = {}
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            print(f"warning: ignoring incomplete output manifest line {line_number}", file=sys.stderr)
            continue
        index = _record_index(record)
        target = output_dir / str(record.get("coefficient_path", record.get("path", "")))
        if index in source_by_index and target.is_file():
            records[index] = record
    return records


def _recover_record(path: Path, output_dir: Path, degree: int, fingerprint: str,
                    source_by_index: Mapping[int, SourceSample]) -> Optional[Tuple[int, Dict[str, object]]]:
    match = OUTPUT_RE.match(path.name)
    if not match or int(match.group(2)) != degree:
        return None
    index = int(match.group(1))
    source = source_by_index.get(index)
    if source is None:
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            coefficients = np.asarray(data["coefficients"])
            indices = np.asarray(data["indices"])
            metadata = json.loads(str(data["metadata"]))
        if (coefficients.shape != (comb(degree + 3, 3),) or
                indices.shape != (len(coefficients), 3) or
                metadata.get("config_fingerprint") != fingerprint or
                metadata.get("source_index") != index):
            return None
        diagnostics = dict(metadata["diagnostics"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    relative = path.relative_to(output_dir).as_posix()
    record = {
        "index": index,
        "sample_id": source.sample_id,
        "group_id": source.sample_id,
        "split": source.split,
        "source_path": os.path.relpath(source.path, output_dir),
        "coefficient_path": relative,
        "path": relative,
        "degree": degree,
        "coefficient_count": len(coefficients),
        "source_seed": source.seed,
        "fit_seed": int(metadata["fit_seed"]),
        "diagnostics": diagnostics,
    }
    return index, record


def _recover_orphans(output_dir: Path, records: Dict[int, Dict[str, object]],
                     degree: int, fingerprint: str,
                     source_by_index: Mapping[int, SourceSample]) -> int:
    recovered = 0
    for path in output_dir.rglob(f"sample_*_chebyshev_p{degree}.npz"):
        match = OUTPUT_RE.match(path.name)
        if not match or int(match.group(1)) in records:
            continue
        result = _recover_record(path, output_dir, degree, fingerprint, source_by_index)
        if result is not None:
            index, record = result
            records[index] = record
            recovered += 1
    return recovered


def _manifest_text(records: Mapping[int, Mapping[str, object]]) -> str:
    return "".join(json.dumps(records[index], ensure_ascii=False, sort_keys=True) + "\n"
                   for index in sorted(records))


def _append_json(stream, value: Mapping[str, object]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {seconds:02d}s"


def _print_progress(done: int, total: int, generated: int, failures: int,
                    started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = generated / elapsed
    eta = (total - done) / rate if rate > 0.0 else float("inf")
    print(f"progress {done}/{total} ({100.0 * done / total:.2f}%): "
          f"generated={generated}, failures={failures}, {rate:.2f} fits/s, "
          f"ETA {_format_duration(eta)}", flush=True)


def _metric_summary(values) -> Dict[str, Optional[float]]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(finite):
        return {"mean": None, "median": None, "p90": None, "max": None}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
    }


def _write_summary(output_dir: Path, records: Mapping[int, Mapping[str, object]],
                   total: int, elapsed: float, generated: int, failures: int) -> None:
    split_counts = {name: 0 for name in ("train", "val", "test")}
    metric_names = ("condition_number", "surface_rmse", "offset_rmse", "boundary_rmse",
                    "maximum_abs_coefficient", "minimum_offset", "maximum_offset")
    metrics = {name: [] for name in metric_names}
    full_rank = 0
    for record in records.values():
        split = str(record.get("split", ""))
        split_counts[split] = split_counts.get(split, 0) + 1
        diagnostics = record.get("diagnostics", {})
        if isinstance(diagnostics, Mapping):
            full_rank += int(diagnostics.get("rank", -1) == record.get("coefficient_count"))
            for name in metric_names:
                if name in diagnostics:
                    metrics[name].append(float(diagnostics[name]))
    summary = {
        "complete": len(records) == total,
        "requested_samples": total,
        "completed_samples": len(records),
        "missing_samples": total - len(records),
        "generated_this_run": generated,
        "failures_this_run": failures,
        "elapsed_seconds_this_run": elapsed,
        "fits_per_second_this_run": generated / elapsed if elapsed > 0.0 else None,
        "split_counts": split_counts,
        "full_rank_samples": full_rank,
        "diagnostics": {name: _metric_summary(values) for name, values in metrics.items()},
    }
    _atomic_write_text(
        output_dir / "fitting_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _run_pool(args: argparse.Namespace, config: Mapping[str, object], fingerprint: str,
              output_dir: Path, source_by_index: Mapping[int, SourceSample],
              records: Dict[int, Dict[str, object]]) -> int:
    missing = [source for index, source in sorted(source_by_index.items()) if index not in records]
    total = len(source_by_index)
    if not missing:
        _write_summary(output_dir, records, total, 0.0, 0, 0)
        print(f"coefficient dataset is already complete: {len(records)} samples")
        return 0

    workers = args.workers or _default_workers()
    queue_limit = max(workers, workers * args.queue_factor)
    manifest_path = output_dir / "manifest.jsonl"
    failure_path = output_dir / "failures.jsonl"
    generated = failures = terminal = 0
    started = time.monotonic()
    pending: Dict[Future, Tuple[SourceSample, int]] = {}
    missing_iter = iter(missing)
    interrupted = False
    print(f"starting {len(missing)} missing fits with {workers} worker processes "
          f"and {_BLAS_THREADS} BLAS thread(s) per worker", flush=True)
    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        with manifest_path.open("a", encoding="utf-8") as manifest_stream, \
                failure_path.open("a", encoding="utf-8") as failure_stream:
            def submit_next() -> bool:
                try:
                    source = next(missing_iter)
                except StopIteration:
                    return False
                future = executor.submit(_fit_one, source, str(output_dir), config, fingerprint)
                pending[future] = (source, 0)
                return True

            while len(pending) < queue_limit and submit_next():
                pass
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    source, attempt = pending.pop(future)
                    try:
                        record = future.result()
                    except Exception as error:
                        if attempt < args.retries:
                            retry = executor.submit(
                                _fit_one, source, str(output_dir), config, fingerprint)
                            pending[retry] = (source, attempt + 1)
                            continue
                        failures += 1
                        terminal += 1
                        _append_json(failure_stream, {
                            "index": source.index,
                            "sample_id": source.sample_id,
                            "attempts": attempt + 1,
                            "error": repr(error),
                            "traceback": "".join(traceback.format_exception(
                                type(error), error, error.__traceback__)),
                            "time_unix": time.time(),
                        })
                    else:
                        records[source.index] = record
                        _append_json(manifest_stream, record)
                        generated += 1
                        terminal += 1
                    while len(pending) < queue_limit and submit_next():
                        pass
                    if terminal % args.progress_every == 0 or len(records) + failures == total:
                        _print_progress(len(records), total, generated, failures, started)
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted; completed coefficient files were preserved", file=sys.stderr)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    _atomic_write_text(manifest_path, _manifest_text(records))
    elapsed = time.monotonic() - started
    _write_summary(output_dir, records, total, elapsed, generated, failures)
    if interrupted:
        return 130
    _print_progress(len(records), total, generated, failures, started)
    if failures:
        print(f"{failures} fits failed; rerun the same command to retry missing samples. "
              f"Details: {failure_path}", file=sys.stderr)
        return 2
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"source manifest does not exist: {manifest}")
    manifest_hash = _sha256(manifest)
    samples = _read_source_manifest(manifest, args.limit)
    source_by_index = {sample.index: sample for sample in samples}
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _material_config(args, manifest, manifest_hash, len(samples))
    _ensure_config(output_dir, config)
    fingerprint = _config_fingerprint(config)

    manifest_path = output_dir / "manifest.jsonl"
    records = _read_output_manifest(manifest_path, output_dir, source_by_index)
    recovered = _recover_orphans(
        output_dir, records, args.degree, fingerprint, source_by_index)
    if recovered:
        print(f"recovered {recovered} coefficient files missing from the output manifest")
    _atomic_write_text(manifest_path, _manifest_text(records))

    workers = args.workers or _default_workers()
    print(f"source={manifest}\noutput={output_dir}\ndegree={args.degree}\n"
          f"samples={len(samples)}\nexisting={len(records)}\nworkers={workers}\n"
          f"surface_samples={args.surface_samples}\nboundary_samples={args.boundary_samples}")
    if args.dry_run:
        print("dry run complete; no fitting jobs were started")
        return 0
    return _run_pool(args, config, fingerprint, output_dir, source_by_index, records)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"parallel Chebyshev fitting failed: {error}")
