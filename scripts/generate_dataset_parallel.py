#!/usr/bin/env python3
"""Generate a large contour-to-Plateau-surface dataset in parallel.

The run is resumable: completed samples are recorded incrementally, orphaned
NPZ files left after an interruption are recovered, and existing samples are
never recomputed when the same configuration is used again.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from generate_dataset import _split_for, find_solver, generate_sample
except ImportError:  # Allow ``python -m scripts.generate_dataset_parallel``.
    from scripts.generate_dataset import _split_for, find_solver, generate_sample


CONFIG_VERSION = 2
SAMPLE_RE = re.compile(r"^sample_(\d+)\.npz$")


def _default_workers() -> int:
    return max(1, min(6, (os.cpu_count() or 2) - 1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/plateau-100k"))
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=0,
                        help="worker processes; 0 chooses up to six automatically")
    parser.add_argument("--queue-factor", type=int, default=2,
                        help="maximum queued jobs per worker")
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--boundary-size", type=int, default=64)
    parser.add_argument("--raw-contour-size", type=int, default=128)
    parser.add_argument("--fourier-modes", type=int, default=5)
    parser.add_argument("--fourier-decay", type=float, default=2.5)
    parser.add_argument("--xy-variation", type=float, default=0.22)
    parser.add_argument("--nonplanarity", type=float, default=0.32)
    parser.add_argument("--min-separation", type=float, default=0.025)
    parser.add_argument("--max-curvature", type=float, default=25.0)
    parser.add_argument("--queries", type=int, default=0,
                        help="SDF samples per mesh; zero stores only contour and surface")
    parser.add_argument("--solver-mode", choices=("graph", "spatial"), default="graph")
    parser.add_argument("--refine", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--remesh-passes", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--near-fraction", type=float, default=0.5)
    parser.add_argument("--query-margin", type=float, default=0.35)
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--shard-size", type=int, default=1000,
                        help="NPZ files per subdirectory; zero disables sharding")
    parser.add_argument("--retries", type=int, default=1,
                        help="retries after a worker/sample failure")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--allow-unconverged", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and show the plan without starting workers")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.samples < 1:
        raise ValueError("samples must be positive")
    if args.boundary_size < 8 or args.raw_contour_size < 8:
        raise ValueError("boundary-size and raw-contour-size must be at least 8")
    if args.queries < 0:
        raise ValueError("queries must be non-negative")
    if (args.fourier_modes < 2 or args.fourier_decay <= 1.0 or
            args.xy_variation < 0.0 or args.nonplanarity < 0.0 or
            args.min_separation < 0.0 or args.max_curvature <= 0.0):
        raise ValueError("invalid Fourier contour or contour-quality parameters")
    if args.workers < 0 or args.queue_factor < 1:
        raise ValueError("workers must be non-negative and queue-factor must be positive")
    if args.shard_size < 0 or args.retries < 0 or args.progress_every < 1:
        raise ValueError("shard-size and retries must be non-negative; progress-every must be positive")
    if not 0.0 <= args.near_fraction <= 1.0:
        raise ValueError("near-fraction must be between 0 and 1")
    if args.refine < 0 or args.iterations < 1 or args.tolerance <= 0.0:
        raise ValueError("refine must be non-negative; iterations and tolerance must be positive")
    if args.remesh_passes is None:
        args.remesh_passes = 0 if args.solver_mode == "graph" else 3
    if args.solver_mode == "graph" and args.remesh_passes != 0:
        raise ValueError("remesh-passes must be zero with solver-mode graph")


def _material_config(args: argparse.Namespace, solver: Path) -> Dict[str, object]:
    return {
        "config_version": CONFIG_VERSION,
        "samples": int(args.samples),
        "seed": int(args.seed),
        "boundary_size": int(args.boundary_size),
        "raw_contour_size": int(args.raw_contour_size),
        "fourier_modes": int(args.fourier_modes),
        "fourier_decay": float(args.fourier_decay),
        "xy_variation": float(args.xy_variation),
        "nonplanarity": float(args.nonplanarity),
        "min_separation": float(args.min_separation),
        "max_curvature": float(args.max_curvature),
        "queries": int(args.queries),
        "solver_mode": args.solver_mode,
        "refine": int(args.refine),
        "iterations": int(args.iterations),
        "remesh_passes": int(args.remesh_passes),
        "tolerance": float(args.tolerance),
        "near_fraction": float(args.near_fraction),
        "query_margin": float(args.query_margin),
        "solver": str(solver),
        "shard_size": int(args.shard_size),
        "allow_unconverged": bool(args.allow_unconverged),
        "keep_intermediates": bool(args.keep_intermediates),
    }


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


def _ensure_config(output_dir: Path, config: Mapping[str, object]) -> None:
    path = output_dir / "generation_config.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            changed = sorted(key for key in set(existing) | set(config)
                             if existing.get(key) != config.get(key))
            details = ", ".join(
                f"{key}: {existing.get(key)!r} -> {config.get(key)!r}" for key in changed)
            raise RuntimeError(
                "output directory belongs to a different generation configuration ("
                f"{details}). Choose a new --output-dir.")
        return
    _atomic_write_text(path, json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _sample_index(record: Mapping[str, object]) -> Optional[int]:
    value = record.get("index")
    if isinstance(value, int):
        return value
    sample_id = str(record.get("sample_id", ""))
    match = re.fullmatch(r"sample_(\d+)", sample_id)
    return int(match.group(1)) if match else None


def _read_manifest(path: Path, output_dir: Path, total: int) -> Dict[int, Dict[str, object]]:
    records: Dict[int, Dict[str, object]] = {}
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            print(f"warning: ignoring incomplete manifest line {line_number}", file=sys.stderr)
            continue
        index = _sample_index(record)
        sample_path = output_dir / str(record.get("path", ""))
        if index is None or not 0 <= index < total or not sample_path.is_file():
            continue
        record["index"] = index
        records[index] = record
    return records


def _recover_record(path: Path, output_dir: Path, total: int,
                    base_seed: int) -> Optional[Tuple[int, Dict[str, object]]]:
    match = SAMPLE_RE.match(path.name)
    if not match:
        return None
    index = int(match.group(1))
    if not 0 <= index < total:
        return None
    try:
        with np.load(path, allow_pickle=False) as sample:
            metadata = json.loads(str(sample["metadata"]))
            counts = dict(metadata["counts"])
            seed = int(metadata["seed"])
            sample_id = str(metadata["sample_id"])
            solver = dict(metadata["solver"])
            scale = float(metadata["normalization"]["scale"])
            family = str(metadata["contour"]["family"])
            if sample_id != f"sample_{index:06d}" or seed != base_seed + index:
                return None
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    record: Dict[str, object] = {
        "path": path.relative_to(output_dir).as_posix(),
        "file": path.name,
        "index": index,
        "sample_id": sample_id,
        "group_id": sample_id,
        "split": _split_for(index, total, base_seed),
        "seed": seed,
        "counts": counts,
        "normalization_scale": scale,
        "solver": solver,
        "contour_family": family,
    }
    return index, record


def _recover_orphans(output_dir: Path, records: Dict[int, Dict[str, object]],
                     total: int, base_seed: int) -> int:
    recovered = 0
    for path in output_dir.rglob("sample_*.npz"):
        match = SAMPLE_RE.match(path.name)
        if not match or int(match.group(1)) in records:
            continue
        result = _recover_record(path, output_dir, total, base_seed)
        if result is not None:
            index, record = result
            records[index] = record
            recovered += 1
    return recovered


def _manifest_text(records: Mapping[int, Mapping[str, object]]) -> str:
    return "".join(json.dumps(records[index], ensure_ascii=False, sort_keys=True) + "\n"
                   for index in sorted(records))


def _shard_dir(output_dir: Path, index: int, shard_size: int) -> Path:
    if shard_size == 0:
        return output_dir
    return output_dir / f"shard-{index // shard_size:05d}"


def _generate_one(index: int, config: Mapping[str, object], output_dir_string: str,
                  contour_variant: int = 0) -> Dict[str, object]:
    output_dir = Path(output_dir_string)
    sample_dir = _shard_dir(output_dir, index, int(config["shard_size"]))
    sample_dir.mkdir(parents=True, exist_ok=True)
    record = generate_sample(
        index=index,
        total=int(config["samples"]),
        base_seed=int(config["seed"]),
        output_dir=sample_dir,
        solver=Path(str(config["solver"])),
        boundary_count=int(config["boundary_size"]),
        query_count=int(config["queries"]),
        raw_contour_count=int(config["raw_contour_size"]),
        solver_mode=str(config["solver_mode"]),
        refine=int(config["refine"]),
        iterations=int(config["iterations"]),
        remesh_passes=int(config["remesh_passes"]),
        tolerance=float(config["tolerance"]),
        near_fraction=float(config["near_fraction"]),
        query_margin=float(config["query_margin"]),
        keep_intermediates=bool(config["keep_intermediates"]),
        allow_unconverged=bool(config["allow_unconverged"]),
        fourier_modes=int(config["fourier_modes"]),
        fourier_decay=float(config["fourier_decay"]),
        xy_variation=float(config["xy_variation"]),
        nonplanarity=float(config["nonplanarity"]),
        min_separation=float(config["min_separation"]),
        max_curvature=float(config["max_curvature"]),
        contour_variant=contour_variant,
    )
    target = sample_dir / str(record["path"])
    record["path"] = target.relative_to(output_dir).as_posix()
    return record


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
    remaining = total - done
    eta = remaining / rate if rate > 0.0 else float("inf")
    print(f"progress {done}/{total} ({100.0 * done / total:.2f}%): "
          f"generated={generated}, failures={failures}, {rate:.2f} samples/s, "
          f"ETA {_format_duration(eta)}", flush=True)


def _append_json(stream, value: Mapping[str, object]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()


def _write_summary(output_dir: Path, records: Mapping[int, Mapping[str, object]],
                   total: int, elapsed_seconds: float, generated_this_run: int,
                   failures_this_run: int) -> None:
    split_counts = {name: 0 for name in ("train", "val", "test")}
    vertices = []
    faces = []
    converged = 0
    for record in records.values():
        split = str(record.get("split", ""))
        split_counts[split] = split_counts.get(split, 0) + 1
        counts = record.get("counts", {})
        if isinstance(counts, Mapping):
            vertices.append(int(counts.get("surface_vertices", 0)))
            faces.append(int(counts.get("faces", 0)))
        solver = record.get("solver", {})
        if isinstance(solver, Mapping) and solver.get("converged") is True:
            converged += 1

    def statistics(values):
        if not values:
            return {"min": None, "mean": None, "max": None}
        return {"min": min(values), "mean": float(np.mean(values)), "max": max(values)}

    summary = {
        "complete": len(records) == total,
        "requested_samples": total,
        "completed_samples": len(records),
        "missing_samples": total - len(records),
        "generated_this_run": generated_this_run,
        "failures_this_run": failures_this_run,
        "elapsed_seconds_this_run": elapsed_seconds,
        "samples_per_second_this_run": (
            generated_this_run / elapsed_seconds if elapsed_seconds > 0.0 else None),
        "split_counts": split_counts,
        "converged_samples": converged,
        "surface_vertices": statistics(vertices),
        "faces": statistics(faces),
    }
    _atomic_write_text(
        output_dir / "generation_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _run_pool(args: argparse.Namespace, config: Mapping[str, object], output_dir: Path,
              records: Dict[int, Dict[str, object]]) -> int:
    missing = [index for index in range(args.samples) if index not in records]
    if not missing:
        _write_summary(output_dir, records, args.samples, 0.0, 0, 0)
        print(f"dataset is already complete: {len(records)} samples")
        return 0

    workers = args.workers or _default_workers()
    queue_limit = max(workers, workers * args.queue_factor)
    manifest_path = output_dir / "manifest.jsonl"
    failure_path = output_dir / "failures.jsonl"
    generated = 0
    failures = 0
    terminal_this_run = 0
    started = time.monotonic()
    index_iter = iter(missing)
    pending: Dict[Future, Tuple[int, int]] = {}
    interrupted = False

    print(f"starting {len(missing)} missing samples with {workers} workers "
          f"(bounded queue: {queue_limit})", flush=True)
    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        with manifest_path.open("a", encoding="utf-8") as manifest_stream, \
                failure_path.open("a", encoding="utf-8") as failure_stream:
            def submit_next() -> bool:
                try:
                    index = next(index_iter)
                except StopIteration:
                    return False
                future = executor.submit(_generate_one, index, config, str(output_dir), 0)
                pending[future] = (index, 0)
                return True

            while len(pending) < queue_limit and submit_next():
                pass
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    index, attempt = pending.pop(future)
                    try:
                        record = future.result()
                    except Exception as error:  # The full traceback is retained for later diagnosis.
                        if attempt < args.retries:
                            retry = executor.submit(
                                _generate_one, index, config, str(output_dir), attempt + 1)
                            pending[retry] = (index, attempt + 1)
                            continue
                        failures += 1
                        terminal_this_run += 1
                        _append_json(failure_stream, {
                            "index": index,
                            "sample_id": f"sample_{index:06d}",
                            "attempts": attempt + 1,
                            "error": repr(error),
                            "traceback": "".join(traceback.format_exception(
                                type(error), error, error.__traceback__)),
                            "time_unix": time.time(),
                        })
                    else:
                        records[index] = record
                        _append_json(manifest_stream, record)
                        generated += 1
                        terminal_this_run += 1
                    while len(pending) < queue_limit and submit_next():
                        pass
                    if (terminal_this_run % args.progress_every == 0 or
                            len(records) + failures == args.samples):
                        _print_progress(len(records), args.samples, generated, failures, started)
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted; completed NPZ files and manifest entries were preserved", file=sys.stderr)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    _atomic_write_text(manifest_path, _manifest_text(records))
    elapsed = time.monotonic() - started
    _write_summary(output_dir, records, args.samples, elapsed, generated, failures)
    if interrupted:
        return 130
    _print_progress(len(records), args.samples, generated, failures, started)
    if failures:
        print(f"{failures} samples failed; rerun the same command to retry missing indices. "
              f"Details: {failure_path}", file=sys.stderr)
        return 2
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    solver = find_solver(args.solver)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _material_config(args, solver)
    _ensure_config(output_dir, config)

    manifest_path = output_dir / "manifest.jsonl"
    records = _read_manifest(manifest_path, output_dir, args.samples)
    recovered = _recover_orphans(output_dir, records, args.samples, args.seed)
    if recovered:
        print(f"recovered {recovered} completed samples not present in the manifest")
    _atomic_write_text(manifest_path, _manifest_text(records))

    workers = args.workers or _default_workers()
    print(f"output={output_dir}\nsamples={args.samples}\nexisting={len(records)}\n"
          f"workers={workers}\nshard_size={args.shard_size}\nqueries={args.queries}")
    if args.dry_run:
        print("dry run complete; no solver jobs were started")
        return 0
    return _run_pool(args, config, output_dir, records)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"parallel dataset generation failed: {error}")
