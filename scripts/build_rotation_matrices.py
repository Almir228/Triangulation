#!/usr/bin/env python3
"""Build and export exact-derived SO(2) Chebyshev change-of-basis matrices."""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from minsurf_nn.rotations import derive_total_degree_change_of_basis  # noqa: E402
from minsurf_nn.spectral import total_degree_indices  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("runs/rotation/degree-10.npz"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.degree < 0:
        parser.error("degree must be nonnegative")
    if args.output.exists() and not args.force:
        parser.error(f"output already exists: {args.output}; use --force to replace it")

    forward, inverse = derive_total_degree_change_of_basis(args.degree)
    identity_error = float(np.max(np.abs(forward @ inverse - np.eye(len(forward)))))
    metadata = {
        "format_version": 1,
        "degree": args.degree,
        "coefficients": comb(args.degree + 3, 3),
        "source_arithmetic": "exact SymPy Q(i)",
        "export_dtype": "complex128",
        "forward_convention": "d = S @ c",
        "rotation_convention": "c_prime = S_inverse D(phi) S c",
        "numerical_identity_error": identity_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                chebyshev_to_complex=forward,
                complex_to_chebyshev=inverse,
                indices=total_degree_indices(args.degree),
                metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
            )
        temporary.replace(args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({"output": str(args.output), **metadata}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
