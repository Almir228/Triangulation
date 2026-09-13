#!/usr/bin/env python3
"""Generate training samples for a contour-conditioned implicit surface model.

Each sample is a compressed NPZ containing a fixed-size resampled boundary,
query locations and local signed distances, plus the solver's target mesh.
The script deliberately keeps the geometry generation deterministic for a
given seed and delegates surface optimization to the repository's C++ CLI.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Sequence, Tuple, Optional

import numpy as np

try:
    from dataset_utils import json_dumps, read_obj_triangles, resample_closed_curve, signed_distance
except ImportError:  # Allow ``python -m scripts.generate_dataset`` from repo root.
    from scripts.dataset_utils import (json_dumps, read_obj_triangles,
                                       resample_closed_curve, signed_distance)


def _shape_parameters(rng: np.random.Generator) -> Dict[str, float]:
    return {
        "z_1": float(rng.uniform(-0.22, 0.22)),
        "z_2": float(rng.uniform(-0.18, 0.18)),
        "z_3": float(rng.uniform(-0.12, 0.12)),
        "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        "tilt_x": float(rng.uniform(-0.16, 0.16)),
        "tilt_y": float(rng.uniform(-0.16, 0.16)),
        "z_offset": float(rng.uniform(-0.25, 0.25)),
        "semi_major": float(rng.uniform(0.9, 1.25)),
        "semi_minor": float(rng.uniform(0.78, 1.0)),
        "xy_rotation": float(rng.uniform(0.0, 2.0 * math.pi)),
    }


def generate_contour(rng: np.random.Generator, count: int = 64) -> Tuple[np.ndarray, Dict[str, object]]:
    """Generate a smooth closed contour with a convex XY projection.

    An exact ellipse preserves the convex projection required by the current
    C++ fan triangulator.  Z harmonics and a tilt make the spatial boundary
    nonplanar while keeping the orientation fixed.
    """

    if count < 8:
        raise ValueError("contour sample count must be at least 8")
    params = _shape_parameters(rng)
    t = 2.0 * math.pi * np.arange(count, dtype=np.float64) / count
    phase = params["phase"]
    # An exact ellipse gives a strictly convex XY projection, which is the
    # input class accepted by the current fan triangulator.  The 3D Z profile
    # below still supplies several independent smooth contour families.
    x0 = params["semi_major"] * np.cos(t)
    y0 = params["semi_minor"] * np.sin(t)
    rotation = params["xy_rotation"]
    x = x0 * np.cos(rotation) - y0 * np.sin(rotation)
    y = x0 * np.sin(rotation) + y0 * np.cos(rotation)
    z = (params["z_offset"] + params["z_1"] * np.cos(t + phase) +
         params["z_2"] * np.sin(2 * t - phase) + params["z_3"] * np.cos(3 * t))
    # A linear tilt is still smooth and helps avoid an accidental symmetry.
    z = z + params["tilt_x"] * x + params["tilt_y"] * y
    points = np.column_stack((x, y, z)).astype(np.float64)
    metadata = {"family": "convex_ellipse_harmonic", "parameters": params,
                "raw_contour_count": int(count)}
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


def _normalize(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return ((np.asarray(points, dtype=np.float64) - center) / scale).astype(np.float32)


def _query_points(vertices: np.ndarray, faces: np.ndarray, rng: np.random.Generator,
                  count: int, near_fraction: float, margin: float) -> np.ndarray:
    if count < 1:
        raise ValueError("query count must be positive")
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


def _split_for(index: int, total: int) -> str:
    if total <= 1:
        return "train"
    if total == 2:
        return "val" if index == 1 else "train"
    # Keep the split deterministic and ensure tiny datasets still have a
    # useful validation sample when there are at least three examples.
    if total >= 3 and index == total - 1:
        return "test"
    if total >= 3 and index == total - 2:
        return "val"
    return "train"


def generate_sample(index: int, total: int, base_seed: int, output_dir: Path,
                    solver: Path, boundary_count: int, query_count: int,
                    raw_contour_count: int, solver_mode: str, refine: int, iterations: int,
                    remesh_passes: int, tolerance: float, near_fraction: float,
                    query_margin: float, keep_intermediates: bool = False,
                    allow_unconverged: bool = False) -> Dict[str, object]:
    seed = int(base_seed + index)
    rng = np.random.default_rng(seed)
    raw_contour, contour_meta = generate_contour(rng, raw_contour_count)
    boundary = resample_closed_curve(raw_contour, boundary_count).astype(np.float64)
    sample_name = f"sample_{index:06d}"
    target_path = output_dir / f"{sample_name}.npz"
    with tempfile.TemporaryDirectory(prefix="triangulation_dataset_") as temporary:
        temporary_dir = Path(temporary)
        contour_path = temporary_dir / "contour.csv"
        prefix = temporary_dir / "surface"
        _write_contour(contour_path, raw_contour)
        # The generated XY projection is counter-clockwise and the +Z normal
        # fixes one orientation for every sample.  Passing it explicitly also
        # prevents the solver's best-fit normal heuristic from changing the
        # projection plane when the boundary has a strong Z waviness.
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
        # Use the exact boundary stored in the NPZ.  Raw NPY inference follows
        # the same rule, so training and prediction share one normalization.
        center = boundary.mean(axis=0)
        scale = float(np.max(np.linalg.norm(boundary - center, axis=1)))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("generated contour has invalid normalization scale")
        norm_boundary = _normalize(boundary, center, scale)
        norm_vertices = _normalize(mesh_vertices, center, scale)
        queries = _query_points(norm_vertices, faces, rng, query_count, near_fraction, query_margin)
        distances = signed_distance(queries, norm_vertices, faces)
        solver_meta = {"mode": solver_mode, "refine": int(refine), "iterations": int(iterations),
                       "remesh_passes": int(remesh_passes), "tolerance": float(tolerance),
                       "returncode": int(completed.returncode),
                       "converged": bool(completed.returncode == 0)}
        metadata = {
            "format_version": 1,
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
            "normalization": {"center": center.tolist(), "scale": scale,
                               "formula": "normalized=(world-center)/scale"},
            "counts": {"boundary": int(len(norm_boundary)), "queries": int(len(queries)),
                       "surface_vertices": int(len(norm_vertices)), "faces": int(len(faces))},
        }
        np.savez_compressed(
            target_path,
            boundary_points=norm_boundary.astype(np.float32),
            query_points=queries.astype(np.float32),
            signed_distance=distances.astype(np.float32),
            surface_vertices=norm_vertices.astype(np.float32),
            faces=faces.astype(np.int32),
            normalization_center=center.astype(np.float32),
            normalization_scale=np.asarray(scale, dtype=np.float32),
            metadata=np.asarray(json_dumps(metadata)),
        )
        if keep_intermediates:
            _write_contour(output_dir / f"{sample_name}_contour.csv", raw_contour)
    return {"path": target_path.relative_to(output_dir).as_posix(), "file": target_path.name,
            "sample_id": sample_name, "group_id": sample_name,
            "split": _split_for(index, total), "seed": seed,
            "counts": metadata["counts"], "normalization_scale": scale,
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
    if args.samples < 1 or args.boundary_size < 3 or args.queries < 1:
        raise SystemExit("samples, boundary-size and queries must be positive")
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
                                 args.keep_intermediates, args.allow_unconverged)
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
