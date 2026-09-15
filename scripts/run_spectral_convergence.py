#!/usr/bin/env python3
"""Run reproducible Chebyshev convergence studies on canonical dataset meshes."""

from __future__ import annotations

import argparse
import csv
import json
from math import comb
from pathlib import Path
import sys
from typing import Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from minsurf_nn.spectral import (boundary_edges, extract_graph_zero_mesh,
                                 fit_triangulation)  # noqa: E402
from scripts.dataset_utils import signed_distance  # noqa: E402


def mesh_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    return float(0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0],
                 triangles[:, 2] - triangles[:, 0]), axis=1).sum())


def parse_degrees(value: str) -> List[int]:
    try:
        degrees = sorted({int(part.strip()) for part in value.split(",")})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("degrees must be comma-separated integers") from exc
    if not degrees or degrees[0] < 1:
        raise argparse.ArgumentTypeError("degrees must be positive")
    return degrees


def read_manifest(path: Path, limit: int = 0) -> List[Dict[str, object]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if limit:
        records = records[:limit]
    if not records:
        raise ValueError("manifest contains no samples")
    return records


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, comment: str) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"# {comment}\n")
        for x, y, z in vertices:
            stream.write(f"v {x:.10g} {y:.10g} {z:.10g}\n")
        for a, b, c in faces + 1:
            stream.write(f"f {a} {b} {c}\n")


def summarize(rows: List[Dict[str, object]], degrees: List[int]) -> List[Dict[str, object]]:
    metrics = [
        "surface_field_rmse", "boundary_residual", "near_function_rmse",
        "chamfer_rmse", "hausdorff_approx", "relative_area_error",
        "root_failure_fraction",
    ]
    output = []
    for degree in degrees:
        selected = [row for row in rows if row["degree"] == degree]
        summary: Dict[str, object] = {
            "degree": degree,
            "coefficients": comb(degree + 3, 3),
            "samples": len(selected),
        }
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_median"] = float(np.median(values))
            summary[f"{metric}_p90"] = float(np.quantile(values, 0.9))
        output.append(summary)
    return output


def make_plots(output: Path, summary: List[Dict[str, object]], energy_rows,
               representative) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degrees = np.asarray([row["degree"] for row in summary])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for metric, label in (("surface_field_rmse", "surface |F|"),
                          ("boundary_residual", "boundary |F|"),
                          ("near_function_rmse", "near-band function")):
        axes[0].plot(degrees, [row[f"{metric}_median"] for row in summary],
                     marker="o", label=label)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("total degree p")
    axes[0].set_ylabel("median error")
    axes[0].set_title("Implicit fitting")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    for metric, label in (("chamfer_rmse", "Chamfer RMSE"),
                          ("hausdorff_approx", "approx. Hausdorff"),
                          ("relative_area_error", "relative area")):
        axes[1].plot(degrees, [row[f"{metric}_median"] for row in summary],
                     marker="o", label=label)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("total degree p")
    axes[1].set_ylabel("median error")
    axes[1].set_title("Geometric convergence")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "errors_vs_degree.png", dpi=180)
    plt.close(figure)

    by_order: Dict[int, List[float]] = {}
    for row in energy_rows:
        by_order.setdefault(int(row["order"]), []).append(float(row["energy"]))
    orders = np.asarray(sorted(by_order))
    median = np.asarray([np.median(by_order[int(order)]) for order in orders])
    low = np.asarray([np.quantile(by_order[int(order)], 0.25) for order in orders])
    high = np.asarray([np.quantile(by_order[int(order)], 0.75) for order in orders])
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.semilogy(orders, median, marker="o", label="median")
    axis.fill_between(orders, np.maximum(low, 1e-16), np.maximum(high, 1e-16),
                      alpha=0.25, label="IQR")
    axis.set_xlabel("total degree r")
    axis.set_ylabel(r"$E_r=(\sum_{i+j+k=r}c_{ijk}^2)^{1/2}$")
    axis.set_title("Chebyshev coefficient energy")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "coefficient_decay.png", dpi=180)
    plt.close(figure)

    if representative is None:
        return
    true_vertices, predicted_vertices, faces, boundary_ids, degree = representative
    errors = np.abs(predicted_vertices[:, 2] - true_vertices[:, 2])
    figure = plt.figure(figsize=(14, 4.5))
    for position, vertices, title in ((1, true_vertices, "Ground truth"),
                                      (2, predicted_vertices, f"Chebyshev p={degree}")):
        axis = figure.add_subplot(1, 3, position, projection="3d")
        axis.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                          triangles=faces, cmap="viridis", linewidth=0.08)
        axis.plot(true_vertices[boundary_ids, 0], true_vertices[boundary_ids, 1],
                  true_vertices[boundary_ids, 2], color="crimson", linewidth=1.2)
        axis.set_title(title)
        axis.set_box_aspect((1, 1, 0.55))
    axis = figure.add_subplot(1, 3, 3, projection="3d")
    plotted = axis.scatter(true_vertices[:, 0], true_vertices[:, 1], true_vertices[:, 2],
                           c=errors, cmap="magma", s=8)
    figure.colorbar(plotted, ax=axis, shrink=0.65, label="|predicted z - true z|")
    axis.set_title("Spatial error")
    axis.set_box_aspect((1, 1, 0.55))
    figure.tight_layout()
    figure.savefig(output / "representative_surface.png", dpi=180)
    plt.close(figure)


def make_report(path: Path, manifest: Path, summary, rho_values, degrees) -> None:
    lines = [
        "# Chebyshev spectral convergence pilot",
        "",
        f"Dataset: `{manifest}`.",
        "",
        "| p | K | surface | boundary | near field | Chamfer | Hausdorff | area error |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['degree']} | {row['coefficients']} | "
            f"{row['surface_field_rmse_median']:.3e} | "
            f"{row['boundary_residual_median']:.3e} | "
            f"{row['near_function_rmse_median']:.3e} | "
            f"{row['chamfer_rmse_median']:.3e} | "
            f"{row['hausdorff_approx_median']:.3e} | "
            f"{row['relative_area_error_median']:.3e} |")
    lines.extend(["", "Reported values are medians across contours.", ""])
    if rho_values:
        median_rho = float(np.median(rho_values))
        lines.append(f"Median fitted coefficient-decay rho at p={max(degrees)}: "
                     f"**{median_rho:.3f}**.")
        if median_rho <= 1.05:
            lines.append(
                "There is **no clear exponential-decay regime** in this pilot; the "
                "piecewise-flat signed-distance target and fitting setup dominate the tail.")
        else:
            lines.append("The fitted tail is consistent with coefficient decay (rho > 1).")
        lines.append("")
    by_degree = {row["degree"]: row for row in summary}
    if 10 in by_degree and max(degrees) > 10:
        base = by_degree[10]["chamfer_rmse_median"]
        final = by_degree[max(degrees)]["chamfer_rmse_median"]
        improvement = 100.0 * (base - final) / base if base else 0.0
        lines.append(
            f"From p=10 to p={max(degrees)}, median Chamfer RMSE changes by "
            f"**{improvement:.1f}%**. This is the main pilot indicator; it is not yet "
            "a proof for a broader contour distribution.")
    lines.extend([
        "",
        "The extracted graph uses the ground-truth XY mesh only to select the nearby "
        "zero-level branch. No ground-truth z values are copied into the prediction.",
        "",
        "Neural-network error is intentionally absent: this experiment measures only "
        "ground-truth triangulation to finite Chebyshev representation.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--degrees", type=parse_degrees, default=parse_degrees("4,6,8,10,12,15"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--surface-samples", type=int, default=4096)
    parser.add_argument("--boundary-samples", type=int, default=1024)
    parser.add_argument("--offset-distance", type=float, default=0.03)
    parser.add_argument("--boundary-weight", type=float, default=10.0)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--near-band", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true", help="replace experiment outputs")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit < 0 or args.surface_samples < 1 or args.boundary_samples < 1:
        parser.error("limit must be nonnegative and sample counts positive")
    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        parser.error("output directory is not empty; choose a fresh experiment directory")
    args.output.mkdir(parents=True, exist_ok=True)
    coefficient_dir = args.output / "coefficients"
    coefficient_dir.mkdir(exist_ok=True)
    records = read_manifest(args.manifest, args.limit)
    rows: List[Dict[str, object]] = []
    energy_rows = []
    rho_values = []
    representative = None
    representative_degree = 10 if 10 in args.degrees else args.degrees[len(args.degrees) // 2]

    for sample_index, record in enumerate(records):
        sample_path = args.manifest.parent / str(record.get("path", record.get("file")))
        with np.load(sample_path, allow_pickle=False) as sample:
            vertices = np.asarray(sample["surface_vertices"], dtype=np.float64)
            faces = np.asarray(sample["faces"], dtype=np.int32)
            boundary = np.asarray(sample["boundary_points"], dtype=np.float64)
            queries = np.asarray(sample["query_points"], dtype=np.float64)
            distances = np.asarray(sample["signed_distance"], dtype=np.float64)
        edges = boundary_edges(faces)
        boundary_ids = np.unique(edges)
        target_area = mesh_area(vertices, faces)

        for degree in args.degrees:
            fit = fit_triangulation(
                vertices, faces, degree=degree,
                surface_samples=args.surface_samples,
                boundary_samples=args.boundary_samples,
                offset_distance=args.offset_distance,
                boundary_weight=args.boundary_weight,
                ridge=args.ridge, seed=args.seed + sample_index)
            predicted, root = extract_graph_zero_mesh(vertices, faces, fit)
            target_to_predicted = np.abs(signed_distance(vertices, predicted, faces))
            predicted_to_target = np.abs(signed_distance(predicted, vertices, faces))
            predicted_area = mesh_area(predicted, faces)
            surface_values = fit.evaluate(vertices)
            boundary_values = fit.evaluate(vertices[boundary_ids])
            inside = np.max(np.abs(queries), axis=1) <= 1.0
            near = inside & (np.abs(distances) <= args.near_band)
            if not np.any(near):
                raise RuntimeError(f"sample {sample_path} has no near-band validation queries")
            query_residual = fit.evaluate(queries[near]) - distances[near]
            chamfer = float(np.sqrt(0.5 * (
                np.mean(target_to_predicted ** 2) + np.mean(predicted_to_target ** 2))))
            row: Dict[str, object] = {
                "sample_id": record.get("sample_id", sample_path.stem),
                "degree": degree,
                "coefficients": len(fit.coefficients),
                "surface_field_rmse": float(np.sqrt(np.mean(surface_values ** 2))),
                "boundary_residual": float(np.sqrt(np.mean(boundary_values ** 2))),
                "near_function_rmse": float(np.sqrt(np.mean(query_residual ** 2))),
                "chamfer_rmse": chamfer,
                "hausdorff_approx": float(max(target_to_predicted.max(),
                                               predicted_to_target.max())),
                "relative_area_error": float(abs(predicted_area - target_area) / target_area),
                "target_area": target_area,
                "predicted_area": predicted_area,
                "condition_number": fit.diagnostics["condition_number"],
                "root_failure_fraction": float(root["root_failures"] / len(vertices)),
                "root_residual_max": root["root_residual_max"],
            }
            rows.append(row)
            np.savez_compressed(
                coefficient_dir / f"{row['sample_id']}_p{degree}.npz",
                coefficients=fit.coefficients, indices=fit.indices,
                degree=np.asarray(degree, dtype=np.int32),
                diagnostics=np.asarray(json.dumps({**fit.diagnostics, **root}, sort_keys=True)))
            if degree == max(args.degrees):
                energy = fit.energy_by_degree()
                for order, value in enumerate(energy):
                    energy_rows.append({"sample_id": row["sample_id"],
                                        "order": order, "energy": float(value)})
                valid = (np.arange(len(energy)) >= max(2, degree // 3)) & (energy > 1e-14)
                if np.count_nonzero(valid) >= 3:
                    slope = float(np.polyfit(np.arange(len(energy))[valid],
                                             np.log(energy[valid]), 1)[0])
                    rho_values.append(float(np.exp(-slope)))
            if sample_index == 0 and degree == representative_degree:
                representative = (vertices, predicted, faces, boundary_ids,
                                  representative_degree)
                write_obj(args.output / "representative_ground_truth.obj", vertices, faces,
                          "canonical ground-truth triangulation")
                write_obj(args.output / f"representative_chebyshev_p{degree}.obj",
                          predicted, faces, "nearby Chebyshev zero-level graph")
            print(json.dumps(row, sort_keys=True), flush=True)

    summary = summarize(rows, args.degrees)
    write_csv(args.output / "per_sample.csv", rows)
    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "coefficient_energy.csv", energy_rows)
    make_plots(args.output, summary, energy_rows, representative)
    make_report(args.output / "REPORT.md", args.manifest, summary, rho_values, args.degrees)
    print(f"wrote spectral convergence report to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"spectral convergence failed: {error}")
