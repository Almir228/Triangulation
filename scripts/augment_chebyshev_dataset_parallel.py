#!/usr/bin/env python3
"""Materialize compact SO(2) augmentation packs for a Chebyshev dataset."""

from __future__ import annotations

import os

# Prevent process x BLAS oversubscription.  The p=10 block transforms use BLAS.
_BLAS_THREADS = os.environ.get("PLATEAU_BLAS_THREADS", "1")
for _VARIABLE in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_VARIABLE] = _BLAS_THREADS

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import json
from math import pi
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import traceback
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from minsurf_nn.rotations import (  # noqa: E402
    BUNDLED_CHANGE_OF_BASIS,
    rotate_surface_coefficients_many,
    rotate_xy_points_many,
)
from minsurf_nn.spectral import evaluate_polynomial  # noqa: E402

try:  # The generator lives beside this script and is intentionally reusable.
    from generate_dataset import prepare_canonical_contour
except ImportError:  # pragma: no cover - module execution from the repository root
    from scripts.generate_dataset import prepare_canonical_contour


FORMAT_VERSION = 1


@dataclass(frozen=True)
class SourcePair:
    index: int
    sample_id: str
    group_id: str
    split: str
    source_path: str
    coefficient_path: str
    degree: int


def _default_workers() -> int:
    return max(1, min(6, (os.cpu_count() or 2) - 1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="fitted Chebyshev manifest.jsonl")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("dataset/plateau-100k-chebyshev-p10-rot8"),
    )
    parser.add_argument(
        "--copies", type=int, default=8,
        help="copies per source, including the unrotated copy (default: %(default)s)",
    )
    parser.add_argument(
        "--exclude-original", action="store_true",
        help="sample all angles randomly instead of fixing the first copy at zero",
    )
    parser.add_argument(
        "--contour-source", choices=("auto", "regenerate", "npz"), default="auto",
        help="reproduce deterministic contours from generation_config or read source NPZ",
    )
    parser.add_argument("--seed", type=int, default=810_000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--queue-factor", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--hydrate-icloud", action="store_true",
        help="on macOS, request download of dataless iCloud input files on demand",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--verification-points", type=int, default=8,
        help="points used to verify every rotated polynomial",
    )
    parser.add_argument(
        "--verification-tolerance", type=float, default=5e-10,
        help="maximum |F_rot(Rx)-F(x)| allowed",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.copies < 1:
        raise ValueError("--copies must be positive")
    if args.seed < 0 or args.workers < 0 or args.queue_factor < 1:
        raise ValueError("seed/workers must be non-negative and queue-factor positive")
    if args.shard_size < 0 or args.retries < 0 or args.progress_every < 1:
        raise ValueError("shard-size/retries must be non-negative and progress-every positive")
    if args.verification_points < 1 or args.verification_tolerance <= 0.0:
        raise ValueError("verification count/tolerance must be positive")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source_manifest(path: Path, limit: int) -> list[SourcePair]:
    pairs: list[SourcePair] = []
    seen: set[int] = set()
    with path.open(encoding="utf-8") as stream:
        for position, line in enumerate(stream):
            if not line.strip():
                continue
            record = json.loads(line)
            index = int(record.get("index", position))
            if index < 0 or index in seen:
                raise ValueError(f"manifest contains an invalid/duplicate index {index}")
            source_name = record.get("source_path")
            coefficient_name = record.get("coefficient_path", record.get("path"))
            if not isinstance(source_name, str) or not isinstance(coefficient_name, str):
                raise ValueError(f"record {index} has no source/coefficient path")
            degree = int(record.get("degree", -1))
            if degree < 1:
                raise ValueError(f"record {index} has no valid polynomial degree")
            pairs.append(SourcePair(
                index=index,
                sample_id=str(record.get("sample_id", f"sample_{index:06d}")),
                group_id=str(record.get("group_id", record.get("sample_id",
                                                                f"sample_{index:06d}"))),
                split=str(record.get("split", "train")),
                source_path=str((path.parent / source_name).resolve()),
                coefficient_path=str((path.parent / coefficient_name).resolve()),
                degree=degree,
            ))
            seen.add(index)
            if limit and len(pairs) >= limit:
                break
    if not pairs:
        raise ValueError("source manifest is empty")
    pairs.sort(key=lambda pair: pair.index)
    degrees = {pair.degree for pair in pairs}
    if len(degrees) != 1:
        raise ValueError(f"mixed polynomial degrees are unsupported: {sorted(degrees)}")
    return pairs


def _find_generation_config(manifest: Path, pairs: Sequence[SourcePair]
                            ) -> Tuple[Optional[Path], Optional[dict[str, object]]]:
    candidates = []
    fitting_config = manifest.parent / "fitting_config.json"
    if fitting_config.is_file():
        fitted = json.loads(fitting_config.read_text(encoding="utf-8"))
        source_manifest = fitted.get("source_manifest")
        if isinstance(source_manifest, str):
            candidates.append(Path(source_manifest).expanduser().resolve().parent /
                              "generation_config.json")
    source_path = Path(pairs[0].source_path)
    candidates.extend(parent / "generation_config.json" for parent in source_path.parents[:3])
    required = {
        "samples", "seed", "boundary_size", "raw_contour_size", "fourier_modes",
        "fourier_decay", "xy_variation", "nonplanarity", "min_separation",
        "max_curvature",
    }
    for candidate in candidates:
        if not candidate.is_file():
            continue
        generation = json.loads(candidate.read_text(encoding="utf-8"))
        if required.issubset(generation):
            return candidate, generation
    return None, None


def _material_config(args: argparse.Namespace, manifest: Path, manifest_hash: str,
                     pairs: Sequence[SourcePair], generation_path: Optional[Path],
                     generation: Optional[Mapping[str, object]]) -> dict[str, object]:
    degree = pairs[0].degree
    matrix_path = BUNDLED_CHANGE_OF_BASIS if degree == 10 else None
    if args.contour_source == "regenerate" and generation is None:
        raise ValueError("--contour-source regenerate requires a compatible generation_config.json")
    contour_source = ("regenerate" if args.contour_source == "auto" and generation is not None
                      else "npz" if args.contour_source == "auto" else args.contour_source)
    generation_material = None
    generation_sha = None
    if contour_source == "regenerate":
        assert generation is not None and generation_path is not None
        keys = (
            "samples", "seed", "boundary_size", "raw_contour_size", "fourier_modes",
            "fourier_decay", "xy_variation", "nonplanarity", "min_separation",
            "max_curvature",
        )
        generation_material = {key: generation[key] for key in keys}
        generation_sha = _sha256(generation_path)
    return {
        "format_version": FORMAT_VERSION,
        "algorithm": "active_so2_chebyshev_rotation_packs",
        "source_manifest": str(manifest),
        "source_manifest_sha256": manifest_hash,
        "source_pairs": len(pairs),
        "degree": degree,
        "copies_per_source": int(args.copies),
        "include_original": not args.exclude_original,
        "angle_distribution": "uniform_[0,2pi)",
        "contour_source": contour_source,
        "generation_config": generation_material,
        "generation_config_sha256": generation_sha,
        "seed": int(args.seed),
        "shard_size": int(args.shard_size),
        "verification_points": int(args.verification_points),
        "verification_tolerance": float(args.verification_tolerance),
        "bundled_matrix_path": str(matrix_path.resolve()) if matrix_path else None,
        "bundled_matrix_sha256": _sha256(matrix_path) if matrix_path else None,
        "stored_arrays": [
            "boundary_points", "coefficients", "coefficient_energy", "angles",
            "indices", "degree", "source_index", "metadata",
        ],
        "mesh_storage": "referenced_and_rotated_lazily",
    }


def _fingerprint(config: Mapping[str, object]) -> str:
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
    path = output_dir / "augmentation_config.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            changed = sorted(key for key in set(existing) | set(config)
                             if existing.get(key) != config.get(key))
            detail = ", ".join(
                f"{key}: {existing.get(key)!r} -> {config.get(key)!r}" for key in changed)
            raise RuntimeError(
                f"output directory has a different augmentation configuration ({detail}); "
                "choose another --output-dir")
        return
    occupied = [entry for entry in output_dir.iterdir() if entry.name != path.name]
    if occupied:
        raise RuntimeError("output directory is non-empty but has no augmentation_config.json")
    _atomic_write_text(path, json.dumps(config, ensure_ascii=False, indent=2,
                                        sort_keys=True) + "\n")


def _shard_dir(output_dir: Path, index: int, shard_size: int) -> Path:
    return output_dir if shard_size == 0 else output_dir / f"shard-{index // shard_size:05d}"


def _output_name(index: int, degree: int) -> str:
    return f"sample_{index:06d}_rotations_p{degree}.npz"


def _angles(pair: SourcePair, config: Mapping[str, object]) -> np.ndarray:
    seed = np.random.SeedSequence([int(config["seed"]), pair.index])
    values = np.random.default_rng(seed).uniform(0.0, 2.0 * pi,
                                                  size=int(config["copies_per_source"]))
    if bool(config["include_original"]):
        values[0] = 0.0
    return values


def _coefficient_energy_batch(coefficients: np.ndarray,
                              indices: np.ndarray, degree: int) -> np.ndarray:
    orders = indices.sum(axis=1)
    result = np.empty((len(coefficients), degree + 1), dtype=np.float64)
    for order in range(degree + 1):
        result[:, order] = np.sqrt(np.sum(coefficients[:, orders == order] ** 2, axis=1))
    return result


def _load_npz_arrays(path: Path, names: Sequence[str],
                     hydrate_icloud: bool) -> list[np.ndarray]:
    """Load selected arrays, optionally materializing an iCloud placeholder."""

    try:
        with np.load(path, allow_pickle=False) as data:
            return [np.asarray(data[name]) for name in names]
    except EOFError:
        if not hydrate_icloud or sys.platform != "darwin":
            raise EOFError(
                f"{path} is an unreadable/dataless file; on macOS rerun with "
                "--hydrate-icloud") from None
    command = Path("/usr/bin/brctl")
    if not command.is_file():
        raise EOFError(f"cannot hydrate {path}: /usr/bin/brctl is unavailable")
    completed = subprocess.run(
        [str(command), "download", str(path)], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise EOFError(f"iCloud download request failed for {path}: {completed.stderr.strip()}")
    deadline = time.monotonic() + 60.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with np.load(path, allow_pickle=False) as data:
                return [np.asarray(data[name]) for name in names]
        except EOFError as error:
            last_error = error
            time.sleep(0.25)
    raise EOFError(f"iCloud did not materialize {path} within 60 seconds") from last_error


def _augment_one(pair: SourcePair, output_dir_string: str,
                 config: Mapping[str, object], fingerprint: str,
                 hydrate_icloud: bool) -> dict[str, object]:
    source_path = Path(pair.source_path)
    coefficient_path = Path(pair.coefficient_path)
    if not coefficient_path.is_file():
        raise FileNotFoundError(f"missing coefficient NPZ: {coefficient_path}")
    if config["contour_source"] == "npz" and not source_path.is_file():
        raise FileNotFoundError(f"missing source NPZ: {source_path}")

    if config["contour_source"] == "regenerate":
        generation = config["generation_config"]
        assert isinstance(generation, Mapping)
        try:
            boundary, _, _, _, _, _ = prepare_canonical_contour(
                index=pair.index,
                base_seed=int(generation["seed"]),
                boundary_count=int(generation["boundary_size"]),
                raw_contour_count=int(generation["raw_contour_size"]),
                fourier_modes=int(generation["fourier_modes"]),
                fourier_decay=float(generation["fourier_decay"]),
                xy_variation=float(generation["xy_variation"]),
                nonplanarity=float(generation["nonplanarity"]),
                min_separation=float(generation["min_separation"]),
                max_curvature=float(generation["max_curvature"]),
            )
            verification_cloud = None
            contour_origin = "deterministically_regenerated"
        except (RuntimeError, ValueError):
            # A handful of historical samples were accepted only after the
            # generator's numerical retry path.  Their stored contour remains
            # authoritative and lets a resumed augmentation finish exactly.
            boundary, surface_vertices = _load_npz_arrays(
                source_path, ("boundary_points", "surface_vertices"), hydrate_icloud)
            verification_cloud = np.asarray(surface_vertices, dtype=np.float64)
            contour_origin = "source_npz_fallback"
    else:
        boundary, surface_vertices = _load_npz_arrays(
            source_path, ("boundary_points", "surface_vertices"), hydrate_icloud)
        verification_cloud = np.asarray(surface_vertices, dtype=np.float64)
        contour_origin = "source_npz"
    coefficients, indices, degree_array = _load_npz_arrays(
        coefficient_path, ("coefficients", "indices", "degree"), hydrate_icloud)
    boundary = np.asarray(boundary, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int16)
    degree = int(degree_array)
    if degree != pair.degree:
        raise ValueError(f"degree mismatch for source index {pair.index}")

    angles = _angles(pair, config)
    rotated_boundary = rotate_xy_points_many(boundary, angles)
    rotated_coefficients = rotate_surface_coefficients_many(coefficients, degree, angles)
    radial_maximum = float(np.max(np.linalg.norm(boundary[:, :2], axis=1)))
    cube_maximum = float(np.max(np.abs(rotated_boundary)))
    if cube_maximum > 1.0 + 1e-12:
        raise RuntimeError(
            f"rotation leaves Chebyshev cube for index {pair.index}: max={cube_maximum}")

    rng = np.random.default_rng(np.random.SeedSequence(
        [int(config["seed"]), pair.index, 0xBEEF]))
    if verification_cloud is None:
        verification_count = int(config["verification_points"])
        points = rng.uniform(-0.8, 0.8, size=(verification_count, 3))
        verification_domain = "deterministic_uniform_cube_points"
    else:
        verification_count = min(int(config["verification_points"]),
                                 len(verification_cloud))
        chosen = rng.choice(len(verification_cloud), verification_count, replace=False)
        points = verification_cloud[chosen]
        verification_domain = "source_surface_vertices"
    original_values = evaluate_polynomial(points, coefficients, indices)
    rotated_points = rotate_xy_points_many(points, angles)
    identity_error = 0.0
    for copy in range(len(angles)):
        values = evaluate_polynomial(
            rotated_points[copy], rotated_coefficients[copy], indices)
        identity_error = max(identity_error,
                             float(np.max(np.abs(values - original_values))))
    if identity_error > float(config["verification_tolerance"]):
        raise RuntimeError(
            f"rotation identity failed for index {pair.index}: {identity_error:.6g}")
    if bool(config["include_original"]):
        original_error = float(np.max(np.abs(rotated_coefficients[0] - coefficients)))
        if original_error > float(config["verification_tolerance"]):
            raise RuntimeError(
                f"zero-angle coefficient identity failed for index {pair.index}: "
                f"{original_error:.6g}")

    output_dir = Path(output_dir_string)
    output_path = (_shard_dir(output_dir, pair.index, int(config["shard_size"])) /
                   _output_name(pair.index, degree))
    source_relative = os.path.relpath(source_path, output_dir)
    coefficient_relative = os.path.relpath(coefficient_path, output_dir)
    metadata = {
        "format_version": FORMAT_VERSION,
        "representation": "total_degree_chebyshev_so2_pack",
        "source_index": pair.index,
        "source_sample_id": pair.sample_id,
        "source_path": source_relative,
        "coefficient_path": coefficient_relative,
        "config_fingerprint": fingerprint,
        "active_rotation": "counterclockwise_about_positive_z",
        "coefficient_convention": "F_rot(R_phi*x)=F(x)",
        "copies": len(angles),
        "degree": degree,
        "maximum_rotation_identity_error": identity_error,
        "verification_domain": verification_domain,
        "contour_origin": contour_origin,
        "boundary_xy_radius_maximum": radial_maximum,
        "rotated_boundary_cube_maximum": cube_maximum,
    }
    _atomic_savez(
        output_path,
        boundary_points=rotated_boundary.astype(np.float32),
        coefficients=rotated_coefficients.astype(np.float64),
        coefficient_energy=_coefficient_energy_batch(
            rotated_coefficients, indices, degree).astype(np.float64),
        angles=angles.astype(np.float64),
        indices=indices.astype(np.int16),
        degree=np.asarray(degree, dtype=np.int16),
        source_index=np.asarray(pair.index, dtype=np.int64),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )
    relative = output_path.relative_to(output_dir).as_posix()
    return {
        "index": pair.index,
        "sample_id": pair.sample_id,
        "group_id": pair.group_id,
        "split": pair.split,
        "path": relative,
        "source_path": source_relative,
        "coefficient_path": coefficient_relative,
        "degree": degree,
        "copies": len(angles),
        "augmented_samples": len(angles),
        "seed": int(config["seed"]),
        "config_fingerprint": fingerprint,
        "maximum_rotation_identity_error": identity_error,
        "boundary_xy_radius_maximum": radial_maximum,
        "rotated_boundary_cube_maximum": cube_maximum,
    }


def _read_output_manifest(path: Path, output_dir: Path,
                          source_indices: set[int], fingerprint: str) -> dict[int, dict]:
    records: dict[int, dict] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                index = int(record["index"])
                target = output_dir / record["path"]
                if (index in source_indices and target.is_file() and
                        record.get("config_fingerprint") == fingerprint):
                    records[index] = record
            except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError):
                print(f"warning: ignoring invalid output manifest line {line_number}",
                      file=sys.stderr)
    return records


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
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s"


def _print_progress(done: int, total: int, generated: int, failures: int,
                    copies: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = generated / elapsed
    eta = (total - done) / rate if rate else float("inf")
    print(f"progress {done}/{total} ({100.0 * done / total:.2f}%): "
          f"packs={generated}, virtual_samples={done * copies}, failures={failures}, "
          f"{rate:.2f} packs/s, ETA {_format_duration(eta)}", flush=True)


def _write_summary(output_dir: Path, records: Mapping[int, Mapping[str, object]],
                   total: int, copies: int, elapsed: float,
                   generated: int, failures: int) -> None:
    errors = np.asarray([
        float(record.get("maximum_rotation_identity_error", float("nan")))
        for record in records.values()
    ])
    errors = errors[np.isfinite(errors)]
    split_packs: dict[str, int] = {}
    for record in records.values():
        split = str(record.get("split", "unknown"))
        split_packs[split] = split_packs.get(split, 0) + 1
    summary = {
        "complete": len(records) == total,
        "requested_source_pairs": total,
        "completed_source_pairs": len(records),
        "copies_per_source": copies,
        "augmented_samples": len(records) * copies,
        "missing_source_pairs": total - len(records),
        "generated_this_run": generated,
        "failures_this_run": failures,
        "elapsed_seconds_this_run": elapsed,
        "packs_per_second_this_run": generated / elapsed if elapsed > 0.0 else None,
        "virtual_samples_per_second_this_run": (
            generated * copies / elapsed if elapsed > 0.0 else None),
        "split_pack_counts": split_packs,
        "split_augmented_sample_counts": {
            split: count * copies for split, count in split_packs.items()
        },
        "maximum_rotation_identity_error": float(errors.max()) if len(errors) else None,
    }
    _atomic_write_text(output_dir / "augmentation_summary.json",
                       json.dumps(summary, ensure_ascii=False, indent=2,
                                  sort_keys=True) + "\n")


def _run_pool(args: argparse.Namespace, config: Mapping[str, object], fingerprint: str,
              output_dir: Path, pairs: Sequence[SourcePair],
              records: dict[int, dict]) -> int:
    missing = [pair for pair in pairs if pair.index not in records]
    total = len(pairs)
    copies = int(config["copies_per_source"])
    if not missing:
        _write_summary(output_dir, records, total, copies, 0.0, 0, 0)
        print(f"augmentation dataset is already complete: {len(records)} packs, "
              f"{len(records) * copies} virtual samples")
        return 0

    workers = args.workers or _default_workers()
    queue_limit = max(workers, workers * args.queue_factor)
    manifest_path = output_dir / "manifest.jsonl"
    failure_path = output_dir / "failures.jsonl"
    generated = failures = terminal = 0
    started = time.monotonic()
    pending: dict[Future, Tuple[SourcePair, int]] = {}
    missing_iter = iter(missing)
    interrupted = False
    print(f"starting {len(missing)} missing rotation packs with {workers} workers; "
          f"{copies} copies/source and {_BLAS_THREADS} BLAS thread(s)/worker", flush=True)
    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        with manifest_path.open("a", encoding="utf-8") as manifest_stream, \
                failure_path.open("a", encoding="utf-8") as failure_stream:
            def submit_next() -> bool:
                try:
                    pair = next(missing_iter)
                except StopIteration:
                    return False
                future = executor.submit(
                    _augment_one, pair, str(output_dir), config, fingerprint,
                    args.hydrate_icloud)
                pending[future] = (pair, 0)
                return True

            while len(pending) < queue_limit and submit_next():
                pass
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    pair, attempt = pending.pop(future)
                    try:
                        record = future.result()
                    except Exception as error:
                        if attempt < args.retries:
                            retry = executor.submit(
                                _augment_one, pair, str(output_dir), config, fingerprint,
                                args.hydrate_icloud)
                            pending[retry] = (pair, attempt + 1)
                            continue
                        failures += 1
                        terminal += 1
                        _append_json(failure_stream, {
                            "index": pair.index,
                            "sample_id": pair.sample_id,
                            "attempts": attempt + 1,
                            "error": repr(error),
                            "traceback": "".join(traceback.format_exception(
                                type(error), error, error.__traceback__)),
                            "time_unix": time.time(),
                        })
                    else:
                        records[pair.index] = record
                        _append_json(manifest_stream, record)
                        generated += 1
                        terminal += 1
                    while len(pending) < queue_limit and submit_next():
                        pass
                    if terminal % args.progress_every == 0 or len(records) + failures == total:
                        _print_progress(len(records), total, generated, failures,
                                        copies, started)
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted; completed augmentation packs were preserved", file=sys.stderr)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    _atomic_write_text(manifest_path, _manifest_text(records))
    elapsed = time.monotonic() - started
    _write_summary(output_dir, records, total, copies, elapsed, generated, failures)
    if interrupted:
        return 130
    _print_progress(len(records), total, generated, failures, copies, started)
    if failures:
        print(f"{failures} packs failed; rerun the same command to retry. "
              f"Details: {failure_path}", file=sys.stderr)
        return 2
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"source manifest does not exist: {manifest}")
    pairs = _read_source_manifest(manifest, args.limit)
    generation_path, generation = _find_generation_config(manifest, pairs)
    config = _material_config(args, manifest, _sha256(manifest), pairs,
                              generation_path, generation)
    fingerprint = _fingerprint(config)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_config(output_dir, config)
    manifest_path = output_dir / "manifest.jsonl"
    records = _read_output_manifest(
        manifest_path, output_dir, {pair.index for pair in pairs}, fingerprint)
    if manifest_path.exists():
        _atomic_write_text(manifest_path, _manifest_text(records))

    workers = args.workers or _default_workers()
    print(f"source={manifest}\noutput={output_dir}\ndegree={pairs[0].degree}\n"
          f"source_pairs={len(pairs)}\ncopies_per_source={args.copies}\n"
          f"virtual_samples={len(pairs) * args.copies}\nexisting_packs={len(records)}\n"
          f"workers={workers}\ninclude_original={not args.exclude_original}\n"
          f"contour_source={config['contour_source']}\n"
          f"hydrate_icloud={args.hydrate_icloud}")
    if args.dry_run:
        print("dry run complete; no rotation jobs were started")
        return 0
    return _run_pool(args, config, fingerprint, output_dir, pairs, records)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EOFError, FileNotFoundError, OSError, RuntimeError, ValueError,
            json.JSONDecodeError) as error:
        raise SystemExit(f"parallel rotation augmentation failed: {error}")
