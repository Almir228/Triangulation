#!/usr/bin/env python3
"""Ablate normal-offset distance and boundary weight for Chebyshev fitting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from minsurf_nn.spectral import (boundary_edges, extract_graph_zero_mesh,
                                 fit_triangulation)  # noqa: E402
from scripts.dataset_utils import signed_distance  # noqa: E402
from scripts.run_spectral_convergence import mesh_area, read_manifest  # noqa: E402


def positive_values(text: str):
    try:
        values = [float(part.strip()) for part in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--degree", type=int, default=10)
    parser.add_argument("--offset-distances", type=positive_values,
                        default=positive_values("0.015,0.03,0.06"))
    parser.add_argument("--boundary-weights", type=positive_values,
                        default=positive_values("1,10,100"))
    parser.add_argument("--surface-samples", type=int, default=4096)
    parser.add_argument("--boundary-samples", type=int, default=1024)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3030)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.degree < 1 or args.surface_samples < 1 or args.boundary_samples < 1:
        parser.error("degree and sample counts must be positive")
    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        parser.error("output directory is not empty; choose a fresh directory")
    args.output.mkdir(parents=True, exist_ok=True)
    records = read_manifest(args.manifest, args.limit)
    rows = []
    for offset in args.offset_distances:
        for boundary_weight in args.boundary_weights:
            for sample_index, record in enumerate(records):
                sample_path = args.manifest.parent / str(record.get("path", record.get("file")))
                with np.load(sample_path, allow_pickle=False) as sample:
                    vertices = np.asarray(sample["surface_vertices"], dtype=np.float64)
                    faces = np.asarray(sample["faces"], dtype=np.int32)
                fit = fit_triangulation(
                    vertices, faces, degree=args.degree,
                    surface_samples=args.surface_samples,
                    boundary_samples=args.boundary_samples,
                    offset_distance=offset, boundary_weight=boundary_weight,
                    ridge=args.ridge, seed=args.seed + sample_index)
                predicted, root = extract_graph_zero_mesh(vertices, faces, fit)
                target_to_predicted = np.abs(signed_distance(vertices, predicted, faces))
                predicted_to_target = np.abs(signed_distance(predicted, vertices, faces))
                boundary_ids = np.unique(boundary_edges(faces))
                target_area = mesh_area(vertices, faces)
                predicted_area = mesh_area(predicted, faces)
                rows.append({
                    "sample_id": record.get("sample_id", sample_path.stem),
                    "degree": args.degree,
                    "offset_distance": offset,
                    "boundary_weight": boundary_weight,
                    "surface_field_rmse": float(np.sqrt(np.mean(fit.evaluate(vertices) ** 2))),
                    "boundary_residual": float(np.sqrt(np.mean(
                        fit.evaluate(vertices[boundary_ids]) ** 2))),
                    "chamfer_rmse": float(np.sqrt(0.5 * (
                        np.mean(target_to_predicted ** 2) +
                        np.mean(predicted_to_target ** 2)))),
                    "hausdorff_approx": float(max(target_to_predicted.max(),
                                                   predicted_to_target.max())),
                    "relative_area_error": float(abs(predicted_area - target_area) / target_area),
                    "condition_number": fit.diagnostics["condition_number"],
                    "root_failure_fraction": root["root_failures"] / len(vertices),
                })
            print(f"finished offset={offset:g}, boundary_weight={boundary_weight:g}", flush=True)
    write_csv(args.output / "per_sample.csv", rows)

    summary = []
    for offset in args.offset_distances:
        for boundary_weight in args.boundary_weights:
            selected = [row for row in rows
                        if row["offset_distance"] == offset and
                        row["boundary_weight"] == boundary_weight]
            summary.append({
                "offset_distance": offset,
                "boundary_weight": boundary_weight,
                "samples": len(selected),
                "surface_field_rmse_median": float(np.median(
                    [row["surface_field_rmse"] for row in selected])),
                "boundary_residual_median": float(np.median(
                    [row["boundary_residual"] for row in selected])),
                "chamfer_rmse_median": float(np.median(
                    [row["chamfer_rmse"] for row in selected])),
                "hausdorff_approx_median": float(np.median(
                    [row["hausdorff_approx"] for row in selected])),
                "relative_area_error_median": float(np.median(
                    [row["relative_area_error"] for row in selected])),
                "condition_number_median": float(np.median(
                    [row["condition_number"] for row in selected])),
            })
    write_csv(args.output / "summary.csv", summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    specifications = (
        ("chamfer_rmse_median", "median Chamfer RMSE"),
        ("boundary_residual_median", "median boundary residual"),
        ("condition_number_median", "median condition number"),
    )
    for axis, (metric, title) in zip(axes, specifications):
        matrix = np.asarray([
            [next(row[metric] for row in summary
                  if row["offset_distance"] == offset and
                  row["boundary_weight"] == weight)
             for weight in args.boundary_weights]
            for offset in args.offset_distances])
        image = axis.imshow(np.log10(matrix), cmap="viridis", aspect="auto")
        for i in range(len(args.offset_distances)):
            for j in range(len(args.boundary_weights)):
                axis.text(j, i, f"{matrix[i, j]:.2e}", ha="center", va="center",
                          color="white" if np.log10(matrix[i, j]) < np.median(np.log10(matrix))
                          else "black", fontsize=8)
        axis.set_xticks(range(len(args.boundary_weights)),
                        [f"{value:g}" for value in args.boundary_weights])
        axis.set_yticks(range(len(args.offset_distances)),
                        [f"{value:g}" for value in args.offset_distances])
        axis.set_xlabel("boundary weight")
        axis.set_ylabel("offset distance")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.75, label="log10")
    figure.tight_layout()
    figure.savefig(args.output / "ablation_heatmaps.png", dpi=180)
    plt.close(figure)

    best = min(summary, key=lambda row: row["chamfer_rmse_median"])
    lines = [
        "# Chebyshev fitting ablation",
        "",
        f"Degree: **p={args.degree}**. Samples: **{len(records)}**.",
        "",
        "| offset | boundary weight | Chamfer | boundary | Hausdorff | area error | condition |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['offset_distance']:.3g} | {row['boundary_weight']:.3g} | "
            f"{row['chamfer_rmse_median']:.3e} | {row['boundary_residual_median']:.3e} | "
            f"{row['hausdorff_approx_median']:.3e} | "
            f"{row['relative_area_error_median']:.3e} | "
            f"{row['condition_number_median']:.3e} |")
    lines.extend([
        "",
        f"Lowest median Chamfer in this grid: offset **{best['offset_distance']:.3g}**, "
        f"boundary weight **{best['boundary_weight']:.3g}**.",
    ])
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote fitting ablation report to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"fitting ablation failed: {error}")
