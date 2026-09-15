#!/usr/bin/env python3
"""Compare ground-truth meshes for the same contours across refinement levels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.dataset_utils import signed_distance  # noqa: E402


def mesh_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    return float(0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0],
                 triangles[:, 2] - triangles[:, 0]), axis=1).sum())


def load_manifests(paths: List[Path]):
    grouped = {}
    for manifest in paths:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            sample_path = manifest.parent / str(record.get("path", record.get("file")))
            with np.load(sample_path, allow_pickle=False) as sample:
                vertices = np.asarray(sample["surface_vertices"], dtype=np.float64)
                faces = np.asarray(sample["faces"], dtype=np.int32)
                metadata = json.loads(str(sample["metadata"]))
            refine = int(metadata["solver"]["refine"])
            sample_id = str(record.get("sample_id", sample_path.stem))
            grouped.setdefault(sample_id, {})[refine] = (vertices, faces, sample_path)
    return grouped


def write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="replace experiment outputs")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.manifests) < 2:
        parser.error("provide at least two manifests from different refinement levels")
    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        parser.error("output directory is not empty; choose a fresh directory")
    args.output.mkdir(parents=True, exist_ok=True)
    grouped = load_manifests(args.manifests)
    if not grouped:
        parser.error("no samples found")
    expected_levels = sorted({level for samples in grouped.values() for level in samples})
    if len(expected_levels) < 2:
        parser.error("manifests contain only one refinement level")
    rows: List[Dict[str, object]] = []
    for sample_id, levels in sorted(grouped.items()):
        if sorted(levels) != expected_levels:
            raise ValueError(f"sample {sample_id} is missing a refinement level")
        reference_level = max(levels)
        reference_vertices, reference_faces, _ = levels[reference_level]
        reference_area = mesh_area(reference_vertices, reference_faces)
        for level in expected_levels:
            vertices, faces, sample_path = levels[level]
            area = mesh_area(vertices, faces)
            if level == reference_level:
                chamfer = 0.0
                hausdorff = 0.0
            else:
                coarse_to_fine = np.abs(signed_distance(
                    vertices, reference_vertices, reference_faces))
                fine_to_coarse = np.abs(signed_distance(
                    reference_vertices, vertices, faces))
                chamfer = float(np.sqrt(0.5 * (
                    np.mean(coarse_to_fine ** 2) + np.mean(fine_to_coarse ** 2))))
                hausdorff = float(max(coarse_to_fine.max(), fine_to_coarse.max()))
            rows.append({
                "sample_id": sample_id,
                "refine": level,
                "vertices": len(vertices),
                "faces": len(faces),
                "area": area,
                "relative_area_to_finest": abs(area - reference_area) / reference_area,
                "chamfer_to_finest": chamfer,
                "hausdorff_to_finest": hausdorff,
                "path": str(sample_path),
            })
    write_csv(args.output / "per_sample.csv", rows)

    summary = []
    for level in expected_levels:
        selected = [row for row in rows if row["refine"] == level]
        summary.append({
            "refine": level,
            "samples": len(selected),
            "vertices_median": float(np.median([row["vertices"] for row in selected])),
            "faces_median": float(np.median([row["faces"] for row in selected])),
            "relative_area_mean": float(np.mean(
                [row["relative_area_to_finest"] for row in selected])),
            "relative_area_median": float(np.median(
                [row["relative_area_to_finest"] for row in selected])),
            "chamfer_mean": float(np.mean([row["chamfer_to_finest"] for row in selected])),
            "chamfer_median": float(np.median(
                [row["chamfer_to_finest"] for row in selected])),
            "hausdorff_mean": float(np.mean(
                [row["hausdorff_to_finest"] for row in selected])),
            "hausdorff_median": float(np.median(
                [row["hausdorff_to_finest"] for row in selected])),
        })
    write_csv(args.output / "summary.csv", summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    compared_levels = expected_levels[:-1]
    compared_summary = summary[:-1]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].semilogy(compared_levels,
                     [row["relative_area_median"] for row in compared_summary], marker="o")
    axes[0].set_xlabel("refinement level")
    axes[0].set_ylabel("median relative area difference to finest")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].semilogy(compared_levels,
                     [row["chamfer_median"] for row in compared_summary], marker="o")
    axes[1].set_xlabel("refinement level")
    axes[1].set_ylabel("median Chamfer RMSE to finest")
    axes[1].grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(args.output / "refinement_convergence.png", dpi=180)
    plt.close(figure)

    lines = [
        "# Ground-truth mesh refinement",
        "",
        "| refinement | median vertices | median faces | area difference | Chamfer | Hausdorff |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['refine']} | {row['vertices_median']:.0f} | "
            f"{row['faces_median']:.0f} | {row['relative_area_median']:.3e} | "
            f"{row['chamfer_median']:.3e} | {row['hausdorff_median']:.3e} |")
    lines.extend([
        "",
        "Every row compares meshes for the same generated contours. The finest supplied "
        "level is the numerical reference, not a proof of convergence to the smooth solution.",
    ])
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote solver refinement report to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"solver refinement failed: {error}")
