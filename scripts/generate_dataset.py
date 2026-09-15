#!/usr/bin/env python3
"""Generate training samples for a contour-conditioned implicit surface model.

Each sample is a compressed NPZ containing a fixed-size resampled boundary,
query locations and local signed distances, plus the solver's target mesh.
The script deliberately keeps the geometry generation deterministic for a
given seed and delegates surface optimization to the repository's C++ CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Sequence, Tuple, Optional

import numpy as np

try:
    from dataset_utils import (canonicalize_contour, is_strictly_convex_xy,
                               json_dumps, read_obj_triangles,
                               resample_closed_curve, signed_distance)
except ImportError:  # Allow ``python -m scripts.generate_dataset`` from repo root.
    from scripts.dataset_utils import (canonicalize_contour, is_strictly_convex_xy,
                                       json_dumps, read_obj_triangles,
                                       resample_closed_curve, signed_distance)


def _contour_quality(points: np.ndarray) -> Dict[str, float]:
    """Return scale-free sampled separation and curvature diagnostics."""

    count = len(points)
    centered = points - points.mean(axis=0)
    diameter = max(2.0 * float(np.linalg.norm(centered, axis=1).max()), 1e-12)
    differences = points[:, None, :] - points[None, :, :]
    squared_distances = np.sum(differences * differences, axis=2)
    indices = np.arange(count)
    cyclic_steps = np.abs(indices[:, None] - indices[None, :])
    cyclic_steps = np.minimum(cyclic_steps, count - cyclic_steps)
    # Pairs closer along the parameterized curve are not distinct branches.
    separated = cyclic_steps > max(2, count // 32)
    minimum_separation = float(np.sqrt(np.min(squared_distances[separated])))

    dt = 2.0 * math.pi / count
    first = (np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)) / (2.0 * dt)
    second = (np.roll(points, -1, axis=0) - 2.0 * points +
              np.roll(points, 1, axis=0)) / (dt * dt)
    speed = np.linalg.norm(first, axis=1)
    curvature = np.linalg.norm(np.cross(first, second), axis=1) / np.maximum(speed, 1e-9) ** 3
    return {
        "minimum_separation_over_diameter": minimum_separation / diameter,
        "maximum_curvature_times_diameter": float(np.max(curvature)) * diameter,
    }


def generate_contour(rng: np.random.Generator, count: int = 64,
                     fourier_modes: int = 5, fourier_decay: float = 2.5,
                     xy_variation: float = 0.22,
                     nonplanarity: float = 0.32) -> Tuple[np.ndarray, Dict[str, object]]:
    """Generate a smooth Fourier contour with a strictly convex XY projection.

    XY is constructed from the support function of a convex curve.  Positivity
    of ``h + h''`` is enforced before an affine stretch, so the generated
    projection stays convex while being substantially richer than an ellipse.
    A Fourier height profile supplies nonplanarity.  Its affine component is
    removed so the discrete oriented area vector already points along +Z.
    """

    if count < 8:
        raise ValueError("contour sample count must be at least 8")
    if fourier_modes < 2:
        raise ValueError("fourier_modes must be at least 2")
    if fourier_decay <= 1.0 or xy_variation < 0.0 or nonplanarity < 0.0:
        raise ValueError("fourier_decay must exceed 1; amplitudes must be non-negative")

    t = 2.0 * math.pi * np.arange(count, dtype=np.float64) / count
    xy_modes = np.arange(2, fourier_modes + 1, dtype=np.float64)
    xy_cos = rng.normal(size=len(xy_modes)) * xy_variation / xy_modes ** fourier_decay
    xy_sin = rng.normal(size=len(xy_modes)) * xy_variation / xy_modes ** fourier_decay

    angles = xy_modes[:, None] * t[None, :]
    curvature_radius_perturbation = np.sum(
        (1.0 - xy_modes[:, None] ** 2) *
        (xy_cos[:, None] * np.cos(angles) + xy_sin[:, None] * np.sin(angles)), axis=0)
    # Scale all non-circular modes together until the curvature radius has a
    # comfortable positive margin.  This is deterministic for the sample seed.
    minimum_radius = float(np.min(1.0 + curvature_radius_perturbation))
    convexity_scale = 1.0
    if minimum_radius < 0.25:
        convexity_scale = min(1.0, 0.75 / max(1.0 - minimum_radius, 1e-12))
        xy_cos *= convexity_scale
        xy_sin *= convexity_scale

    h = 1.0 + np.sum(
        xy_cos[:, None] * np.cos(angles) + xy_sin[:, None] * np.sin(angles), axis=0)
    h_prime = np.sum(
        xy_modes[:, None] * (-xy_cos[:, None] * np.sin(angles) +
                             xy_sin[:, None] * np.cos(angles)), axis=0)
    x0 = h * np.cos(t) - h_prime * np.sin(t)
    y0 = h * np.sin(t) + h_prime * np.cos(t)

    semi_major = float(rng.uniform(0.85, 1.35))
    semi_minor = float(rng.uniform(0.65, 1.05))
    rotation = float(rng.uniform(0.0, 2.0 * math.pi))
    stretched_x = semi_major * x0
    stretched_y = semi_minor * y0
    x = stretched_x * math.cos(rotation) - stretched_y * math.sin(rotation)
    y = stretched_x * math.sin(rotation) + stretched_y * math.cos(rotation)

    z_modes = np.arange(1, fourier_modes + 2, dtype=np.float64)
    z_cos = rng.normal(size=len(z_modes)) / z_modes ** fourier_decay
    z_sin = rng.normal(size=len(z_modes)) / z_modes ** fourier_decay
    z_angles = z_modes[:, None] * t[None, :]
    z_profile = np.sum(
        z_cos[:, None] * np.cos(z_angles) + z_sin[:, None] * np.sin(z_angles), axis=0)
    z_profile -= z_profile.mean()
    peak = max(float(np.max(np.abs(z_profile))), 1e-12)
    z_amplitude = float(rng.uniform(0.2, 1.0) * nonplanarity) if nonplanarity else 0.0
    z = z_profile * (z_amplitude / peak) + float(rng.uniform(-0.25, 0.25))

    # Remove the affine height component that tilts the vector area away from
    # +Z.  Unlike a post-hoc 3D rotation, this preserves the convex projection.
    provisional = np.column_stack((x, y, z))
    area = 0.5 * np.sum(np.cross(provisional, np.roll(provisional, -1, axis=0)), axis=0)
    if area[2] <= 1e-12:
        raise ValueError("generated contour has degenerate projected area")
    z = z + (area[0] / area[2]) * x + (area[1] / area[2]) * y
    points = np.column_stack((x, y, z)).astype(np.float64)
    corrected_area = 0.5 * np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)

    metadata: Dict[str, object] = {
        "family": "convex_support_fourier",
        "parameters": {
            "fourier_modes": int(fourier_modes),
            "fourier_decay": float(fourier_decay),
            "xy_variation": float(xy_variation),
            "nonplanarity_limit": float(nonplanarity),
            "z_amplitude": z_amplitude,
            "semi_major": semi_major,
            "semi_minor": semi_minor,
            "xy_rotation": rotation,
            "convexity_scale": float(convexity_scale),
            "xy_cos": xy_cos.tolist(),
            "xy_sin": xy_sin.tolist(),
            "z_cos": z_cos.tolist(),
            "z_sin": z_sin.tolist(),
        },
        "quality": _contour_quality(points),
        "generated_vector_area": corrected_area.tolist(),
        "raw_contour_count": int(count),
    }
    return points, metadata


def _write_contour(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for point in points:
            stream.write("{:.17g},{:.17g},{:.17g}\n".format(*point))


def find_solver(requested: Optional[str]) -> Path:
    if requested:
        candidate = Path(requested).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"solver executable does not exist: {candidate}")
        return candidate
    root = Path(__file__).resolve().parents[1]
    candidates = [root / "build" / "triangulation",
                  root / "cmake-build-debug" / "triangulation"]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate
    discovered = shutil.which("triangulation")
    if discovered:
        return Path(discovered).resolve()
    raise FileNotFoundError("could not find triangulation; build it or pass --solver PATH")


def _query_points(vertices: np.ndarray, faces: np.ndarray, rng: np.random.Generator,
                  count: int, near_fraction: float, margin: float) -> np.ndarray:
    if count < 0:
        raise ValueError("query count must be non-negative")
    if count == 0:
        return np.empty((0, 3), dtype=np.float32)
    low = vertices.min(axis=0) - margin
    high = vertices.max(axis=0) + margin
    uniform_count = int(round(count * (1.0 - near_fraction)))
    near_count = count - uniform_count
    uniform = rng.uniform(low, high, size=(uniform_count, 3))

    face_ids = rng.integers(0, len(faces), size=near_count)
    tri = vertices[faces[face_ids]]
    bary = rng.exponential(1.0, size=(near_count, 3))
    bary /= bary.sum(axis=1, keepdims=True)
    near = np.sum(tri * bary[:, :, None], axis=1)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norm, 1e-12)
    # Include both sides of each oriented face.  The small floor keeps points
    # numerically distinct from the exact mesh, which is useful for training.
    offsets = rng.normal(0.0, 0.045, size=(near_count, 1))
    offsets += rng.choice([-1.0, 1.0], size=(near_count, 1)) * 0.002
    near = near + offsets * normals
    queries = np.concatenate((uniform, near), axis=0)
    rng.shuffle(queries, axis=0)
    return queries.astype(np.float32)


def _split_for(index: int, total: int, base_seed: int = 0) -> str:
    if total <= 1:
        return "train"
    if total == 2:
        return "val" if index == 1 else "train"
    # Keep tiny smoke datasets useful and preserve their historical ordering.
    if total < 20 and index == total - 1:
        return "test"
    if total < 20 and index == total - 2:
        return "val"
    if total < 20:
        return "train"
    # A hash avoids putting one contiguous region of the deterministic contour
    # stream into validation/test.  The assignment is stable across processes.
    digest = hashlib.blake2b(f"{base_seed}:{index}".encode("ascii"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "little") % 10_000
    if bucket < 9_000:
        return "train"
    return "val" if bucket < 9_500 else "test"


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    """Write a compressed NPZ without exposing a partially written sample."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
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


def prepare_canonical_contour(index: int, base_seed: int, boundary_count: int,
                              raw_contour_count: int, fourier_modes: int = 5,
                              fourier_decay: float = 2.5, xy_variation: float = 0.22,
                              nonplanarity: float = 0.32, min_separation: float = 0.025,
                              max_curvature: float = 25.0, contour_variant: int = 0
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                         Dict[str, np.ndarray], Dict[str, object],
                                         np.random.Generator]:
    """Reproduce the canonical contour without running the Plateau solver.

    The returned generator retains the exact state following contour creation,
    which keeps subsequent query sampling in ``generate_sample`` unchanged.
    """

    seed = int(base_seed + index)
    if contour_variant < 0:
        raise ValueError("contour_variant must be non-negative")
    rng_seed = seed if contour_variant == 0 else np.random.SeedSequence([seed, contour_variant])
    rng = np.random.default_rng(rng_seed)
    for attempt in range(128):
        raw_contour, contour_meta = generate_contour(
            rng, raw_contour_count, fourier_modes=fourier_modes,
            fourier_decay=fourier_decay, xy_variation=xy_variation,
            nonplanarity=nonplanarity)
        quality = contour_meta["quality"]
        if (quality["minimum_separation_over_diameter"] < min_separation or
                quality["maximum_curvature_times_diameter"] > max_curvature):
            continue
        original_boundary = resample_closed_curve(
            raw_contour, boundary_count).astype(np.float64)
        boundary, transform = canonicalize_contour(original_boundary, margin=0.1)
        if is_strictly_convex_xy(boundary):
            break
    else:
        raise RuntimeError("could not generate a canonical contour with convex XY projection")
    contour_meta["generation_attempt"] = attempt + 1
    contour_meta["variant"] = int(contour_variant)
    contour_meta["canonical_vector_area"] = transform["area_after"].tolist()
    contour_meta["original_vector_area"] = transform["area_before"].tolist()
    dense_boundary = ((raw_contour - transform["center"]) @
                      transform["rotation"].T / float(transform["scale"]))
    return (boundary.astype(np.float32), dense_boundary.astype(np.float32),
            original_boundary, transform, contour_meta, rng)


def generate_sample(index: int, total: int, base_seed: int, output_dir: Path,
                    solver: Path, boundary_count: int, query_count: int,
                    raw_contour_count: int, solver_mode: str, refine: int, iterations: int,
                    remesh_passes: int, tolerance: float, near_fraction: float,
                    query_margin: float, keep_intermediates: bool = False,
                    allow_unconverged: bool = False, fourier_modes: int = 5,
                    fourier_decay: float = 2.5, xy_variation: float = 0.22,
                    nonplanarity: float = 0.32, min_separation: float = 0.025,
                    max_curvature: float = 25.0,
                    contour_variant: int = 0) -> Dict[str, object]:
    seed = int(base_seed + index)
    boundary, dense_boundary, original_boundary, transform, contour_meta, rng = (
        prepare_canonical_contour(
            index=index, base_seed=base_seed, boundary_count=boundary_count,
            raw_contour_count=raw_contour_count, fourier_modes=fourier_modes,
            fourier_decay=fourier_decay, xy_variation=xy_variation,
            nonplanarity=nonplanarity, min_separation=min_separation,
            max_curvature=max_curvature, contour_variant=contour_variant))
    sample_name = f"sample_{index:06d}"
    target_path = output_dir / f"{sample_name}.npz"
    with tempfile.TemporaryDirectory(prefix="triangulation_dataset_") as temporary:
        temporary_dir = Path(temporary)
        contour_path = temporary_dir / "contour.csv"
        prefix = temporary_dir / "surface"
        _write_contour(contour_path, boundary)
        # The actual input has already been centered, rotated and scaled.  The
        # explicit normal now agrees with its oriented area instead of merely
        # overriding the solver's projection heuristic.
        command = [str(solver), "--contour", str(contour_path), "--normal", "0", "0", "1",
                   "--mode", solver_mode,
                   "--refine", str(refine), "--iterations", str(iterations),
                   "--remesh-passes", str(remesh_passes), "--tolerance", str(tolerance),
                   "--output", str(prefix)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode not in (0, 2):
            raise RuntimeError("solver failed for sample {} (return code {}): {}".format(
                index, completed.returncode, completed.stderr.strip() or completed.stdout.strip()))
        if completed.returncode == 2 and not allow_unconverged:
            raise RuntimeError("solver did not converge for sample {}; increase --iterations or "
                               "use --allow-unconverged explicitly".format(index))
        obj_path = prefix.with_suffix(".obj")
        if not obj_path.is_file():
            raise RuntimeError("solver returned without writing an OBJ for sample {}".format(index))
        mesh_vertices, faces = read_obj_triangles(obj_path)
        norm_boundary = boundary.astype(np.float32)
        norm_vertices = mesh_vertices.astype(np.float32)
        if np.max(np.abs(norm_vertices)) > 1.0 + 1e-6:
            raise ValueError("solver surface escaped the canonical cube")
        queries = _query_points(norm_vertices, faces, rng, query_count, near_fraction, query_margin)
        distances = (signed_distance(queries, norm_vertices, faces)
                     if len(queries) else np.empty((0,), dtype=np.float32))
        solver_meta = {"mode": solver_mode, "refine": int(refine), "iterations": int(iterations),
                       "remesh_passes": int(remesh_passes), "tolerance": float(tolerance),
                       "returncode": int(completed.returncode),
                       "converged": bool(completed.returncode == 0)}
        metadata = {
            "format_version": 2,
            "sample_id": sample_name,
            "seed": seed,
            "contour": contour_meta,
            "solver": solver_meta,
            "query": {"count": int(query_count), "uniform_fraction": float(1.0 - near_fraction),
                       "near_surface_fraction": float(near_fraction), "margin": float(query_margin)},
            "signed_distance": {
                "units": "normalized_coordinate_units",
                "definition": "distance to nearest triangle; sign from nearest listed face normal",
                "scope": "local_for_open_surface",
            },
            "normalization": {
                "center": transform["center"].tolist(),
                "rotation": transform["rotation"].tolist(),
                "scale": float(transform["scale"]),
                "margin": float(transform["margin"]),
                "formula": "canonical=rotation@(world-center)/scale",
            },
            "counts": {"boundary": int(len(norm_boundary)),
                       "dense_boundary": int(len(dense_boundary)), "queries": int(len(queries)),
                       "surface_vertices": int(len(norm_vertices)), "faces": int(len(faces))},
        }
        _atomic_savez(
            target_path,
            boundary_points=norm_boundary.astype(np.float32),
            dense_boundary_points=dense_boundary,
            query_points=queries.astype(np.float32),
            signed_distance=distances.astype(np.float32),
            surface_vertices=norm_vertices.astype(np.float32),
            faces=faces.astype(np.int32),
            normalization_center=transform["center"].astype(np.float32),
            normalization_rotation=transform["rotation"].astype(np.float32),
            normalization_scale=np.asarray(transform["scale"], dtype=np.float32),
            metadata=np.asarray(json_dumps(metadata)),
        )
        if keep_intermediates:
            _write_contour(output_dir / f"{sample_name}_contour.csv", boundary)
            _write_contour(output_dir / f"{sample_name}_original_contour.csv", original_boundary)
    return {"path": target_path.relative_to(output_dir).as_posix(), "file": target_path.name,
            "index": int(index),
            "sample_id": sample_name, "group_id": sample_name,
            "split": _split_for(index, total, base_seed), "seed": seed,
            "counts": metadata["counts"],
            "normalization_scale": float(transform["scale"]),
            "solver": metadata["solver"], "contour_family": contour_meta["family"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"),
                        help="directory receiving NPZ files and manifest.jsonl")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--boundary-size", type=int, default=64)
    parser.add_argument("--queries", type=int, default=8192)
    parser.add_argument("--raw-contour-size", type=int, default=64)
    parser.add_argument("--fourier-modes", type=int, default=5)
    parser.add_argument("--fourier-decay", type=float, default=2.5)
    parser.add_argument("--xy-variation", type=float, default=0.22)
    parser.add_argument("--nonplanarity", type=float, default=0.32)
    parser.add_argument("--min-separation", type=float, default=0.025,
                        help="minimum sampled nonlocal distance divided by contour diameter")
    parser.add_argument("--max-curvature", type=float, default=25.0,
                        help="maximum sampled curvature times contour diameter")
    parser.add_argument("--solver-mode", choices=("graph", "spatial"), default="graph",
                        help="stable graph teacher, or experimental free-XYZ spatial teacher")
    parser.add_argument("--refine", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--remesh-passes", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--near-fraction", type=float, default=0.5)
    parser.add_argument("--query-margin", type=float, default=0.35)
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="generate exactly two small samples for validation")
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="also keep generated contour CSVs")
    parser.add_argument("--allow-unconverged", action="store_true",
                        help="retain solver status 2 samples (unsafe as training targets)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.samples = 2
        args.boundary_size = min(args.boundary_size, 32)
        args.queries = min(args.queries, 1024)
        args.raw_contour_size = min(args.raw_contour_size, 32)
        # The unrefined fan is a quick, stable smoke target.  Larger runs can
        # request refinement explicitly; difficult high-frequency contours may
        # need more iterations or the explicit --allow-unconverged escape hatch.
        args.refine = 0
        args.iterations = min(args.iterations, 80)
    if args.samples < 1 or args.boundary_size < 3 or args.queries < 0:
        raise SystemExit("samples and boundary-size must be positive; queries must be non-negative")
    if (args.fourier_modes < 2 or args.fourier_decay <= 1.0 or
            args.xy_variation < 0.0 or args.nonplanarity < 0.0 or
            args.min_separation < 0.0 or args.max_curvature <= 0.0):
        raise SystemExit("invalid Fourier contour or contour-quality parameters")
    if not 0.0 <= args.near_fraction <= 1.0:
        raise SystemExit("near-fraction must be between 0 and 1")
    if args.remesh_passes is None:
        args.remesh_passes = 0 if args.solver_mode == "graph" else 3
    if args.solver_mode == "graph" and args.remesh_passes != 0:
        raise SystemExit("remesh-passes must be 0 with solver-mode graph")
    solver = find_solver(args.solver)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(args.samples):
        record = generate_sample(index, args.samples, args.seed, output_dir, solver,
                                 args.boundary_size, args.queries, args.raw_contour_size,
                                 args.solver_mode, args.refine, args.iterations, args.remesh_passes,
                                 args.tolerance, args.near_fraction, args.query_margin,
                                 args.keep_intermediates, args.allow_unconverged,
                                 args.fourier_modes, args.fourier_decay, args.xy_variation,
                                 args.nonplanarity, args.min_separation, args.max_curvature)
        records.append(record)
        print(f"generated {record['path']} ({record['counts']['surface_vertices']} vertices, "
              f"{record['counts']['faces']} faces)")
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {manifest_path} ({len(records)} samples)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"dataset generation failed: {error}")
