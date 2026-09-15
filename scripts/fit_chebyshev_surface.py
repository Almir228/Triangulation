#!/usr/bin/env python3
"""Fit a total-degree Chebyshev polynomial to an oriented OBJ/NPZ triangulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from minsurf_nn.spectral import fit_triangulation  # noqa: E402
from scripts.dataset_utils import read_obj_triangles  # noqa: E402


def _load_mesh(path: Path):
    if path.suffix.lower() == ".obj":
        return read_obj_triangles(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "surface_vertices" not in data or "faces" not in data:
                raise ValueError("NPZ must contain surface_vertices and faces")
            return (np.asarray(data["surface_vertices"], dtype=np.float64),
                    np.asarray(data["faces"], dtype=np.int32))
    raise ValueError("input must be an OBJ mesh or a dataset NPZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path, help="normalized OBJ or dataset NPZ")
    parser.add_argument("--output", type=Path, help="output coefficient NPZ")
    parser.add_argument("--degree", type=int, default=10)
    parser.add_argument("--surface-samples", type=int, default=4096)
    parser.add_argument("--boundary-samples", type=int, default=1024)
    parser.add_argument("--offset-distance", type=float, default=0.03)
    parser.add_argument("--boundary-weight", type=float, default=10.0)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--rcond", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="replace an existing output")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = args.output or args.mesh.with_name(
        f"{args.mesh.stem}_chebyshev_p{args.degree}.npz")
    if output.exists() and not args.force:
        parser.error(f"output already exists: {output}; pass --force to replace it")
    try:
        vertices, faces = _load_mesh(args.mesh)
        result = fit_triangulation(
            vertices, faces, degree=args.degree,
            surface_samples=args.surface_samples,
            boundary_samples=args.boundary_samples,
            offset_distance=args.offset_distance,
            boundary_weight=args.boundary_weight,
            ridge=args.ridge, rcond=args.rcond, seed=args.seed)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Chebyshev fitting failed: {exc}\n")

    metadata = {
        "format_version": 1,
        "representation": "total_degree_chebyshev",
        "domain": "[-1,1]^3",
        "degree": result.degree,
        "source_mesh": str(args.mesh),
        "offset_distance": args.offset_distance,
        "boundary_weight": args.boundary_weight,
        "ridge": args.ridge,
        "seed": args.seed,
        "diagnostics": result.diagnostics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        coefficients=result.coefficients.astype(np.float64),
        indices=result.indices.astype(np.int16),
        degree=np.asarray(result.degree, dtype=np.int32),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )
    print(json.dumps({"output": str(output), **result.diagnostics},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
