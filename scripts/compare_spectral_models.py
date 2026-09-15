#!/usr/bin/env python3
"""Compare teacher Plateau surfaces, fitted labels, and two neural operators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from minsurf_nn.data import resample_boundary  # noqa: E402
from minsurf_nn.spectral import (SpectralFit, evaluate_gradient,
                                 evaluate_polynomial, extract_graph_zero_mesh)  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("dataset/plateau-100k-chebyshev-p10/manifest.jsonl"),
    )
    parser.add_argument(
        "--old-checkpoint", type=Path,
        default=Path("runs/spectral-operator-p10-n64/best.pt"),
    )
    parser.add_argument(
        "--hybrid-checkpoint", type=Path,
        default=Path("runs/spectral-operator-p10-n64-hybrid-v2/best.pt"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("runs/spectral-model-comparison-teacher"),
    )
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--indices", type=int, nargs="*",
                        help="explicit source indices instead of a random validation sample")
    parser.add_argument("--boundary-count", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"),
                        default="auto")
    return parser


def choose_device(name, torch):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if name == "mps" and not (hasattr(torch.backends, "mps") and
                               torch.backends.mps.is_available()):
        raise ValueError("MPS requested but unavailable")
    return torch.device(name)


def synchronize(device, torch) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def read_records(path: Path) -> list[dict]:
    manifest = path.resolve()
    records = []
    with manifest.open(encoding="utf-8") as stream:
        for position, line in enumerate(stream):
            if not line.strip():
                continue
            record = json.loads(line)
            record["index"] = int(record.get("index", position))
            records.append(record)
    if not records:
        raise ValueError(f"empty manifest: {manifest}")
    return records


def resolve_path(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest.resolve().parent / path).resolve()


def load_model(path: Path, device, torch):
    from minsurf_nn.spectral_operator import ContourToChebyshev, SpectralOperatorConfig

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ContourToChebyshev(SpectralOperatorConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


def predict(model, boundary: np.ndarray, device, torch) -> tuple[np.ndarray, float]:
    tensor = torch.from_numpy(boundary.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(tensor)
        synchronize(device, torch)
        timings = []
        for _ in range(50):
            started = perf_counter()
            coefficients = model(tensor)
            synchronize(device, torch)
            timings.append((perf_counter() - started) * 1000.0)
    return coefficients[0].cpu().numpy().astype(np.float64), float(np.median(timings))


def project_boundary_to_zero(boundary: np.ndarray, fit: SpectralFit,
                             iterations: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Project contour xy positions to the nearby polynomial zero root in z."""

    projected = boundary.astype(np.float64).copy()
    for _ in range(iterations):
        values = fit.evaluate(projected)
        derivative = fit.gradient(projected)[:, 2]
        active = np.abs(derivative) > 1e-9
        step = np.zeros_like(values)
        step[active] = np.clip(values[active] / derivative[active], -0.1, 0.1)
        projected[:, 2] = np.clip(projected[:, 2] - step, -1.0, 1.0)
    return projected, np.abs(fit.evaluate(projected))


def l2_weights(indices: np.ndarray) -> np.ndarray:
    return np.where(indices == 0, 1.0, 0.5).prod(axis=1)


def evaluate_candidate(name: str, coefficients: np.ndarray, indices: np.ndarray,
                       vertices: np.ndarray, faces: np.ndarray,
                       boundary: np.ndarray, inference_ms: float | None) -> tuple[dict, np.ndarray]:
    fit = SpectralFit(int(indices.sum(axis=1).max()), indices, coefficients, {})
    surface, root = extract_graph_zero_mesh(vertices, faces, fit)
    boundary_surface, boundary_residual = project_boundary_to_zero(boundary, fit)
    z_error = surface[:, 2] - vertices[:, 2]
    boundary_z_error = boundary_surface[:, 2] - boundary[:, 2]
    field = evaluate_polynomial(boundary, coefficients, indices)
    gradient = evaluate_gradient(boundary, coefficients, indices)
    distance = np.abs(field) / np.maximum(np.linalg.norm(gradient, axis=1), 1e-9)
    metrics = {
        "name": name,
        "surface_z_rmse": float(np.sqrt(np.mean(z_error ** 2))),
        "surface_z_max": float(np.max(np.abs(z_error))),
        "boundary_z_rmse": float(np.sqrt(np.mean(boundary_z_error ** 2))),
        "boundary_z_max": float(np.max(np.abs(boundary_z_error))),
        "boundary_distance_rmse": float(np.sqrt(np.mean(distance ** 2))),
        "boundary_root_residual_max": float(boundary_residual.max()),
        "root_failures": int(root["root_failures"]),
        "inference_median_ms": inference_ms,
    }
    return metrics, surface


def add_mesh(axis, vertices: np.ndarray, faces: np.ndarray, color: str) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    collection = Poly3DCollection(
        vertices[faces], facecolors=color, edgecolors=(0.05, 0.05, 0.05, 0.08),
        linewidths=0.12, alpha=0.9,
    )
    axis.add_collection3d(collection)


def set_axes(axis, limits, elevation=25.0, azimuth=-58.0) -> None:
    low, high = limits
    center = 0.5 * (low + high)
    span = np.maximum(high - low, 1e-4)
    margin = 0.06
    axis.set_xlim(center[0] - span[0] * (0.5 + margin),
                  center[0] + span[0] * (0.5 + margin))
    axis.set_ylim(center[1] - span[1] * (0.5 + margin),
                  center[1] + span[1] * (0.5 + margin))
    axis.set_zlim(center[2] - span[2] * (0.5 + margin),
                  center[2] + span[2] * (0.5 + margin))
    axis.set_box_aspect((span[0], span[1], max(span[2], 0.35 * max(span[:2]))))
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_xlabel("x", fontsize=8)
    axis.set_ylabel("y", fontsize=8)
    axis.set_zlabel("z", fontsize=8)
    axis.tick_params(labelsize=6)


def plot(samples: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = (
        ("teacher", "Учитель: Plateau", "#4c78a8"),
        ("teacher_fit", "Метка: Chebyshev p=10", "#54a24b"),
        ("old", "Старая supervised NN", "#f58518"),
        ("hybrid", "Новая hybrid NN", "#b279a2"),
    )
    figure = plt.figure(figsize=(18, 4.6 * len(samples)), constrained_layout=True)
    for row, sample in enumerate(samples):
        all_vertices = np.vstack([sample[key]["vertices"] for key, _, _ in columns] +
                                 [sample["boundary"]])
        limits = (all_vertices.min(axis=0), all_vertices.max(axis=0))
        closed = np.vstack((sample["boundary"], sample["boundary"][0]))
        for column, (key, heading, color) in enumerate(columns):
            axis = figure.add_subplot(len(samples), len(columns),
                                      row * len(columns) + column + 1,
                                      projection="3d")
            candidate = sample[key]
            add_mesh(axis, candidate["vertices"], sample["faces"], color)
            axis.plot(*closed.T, color="black", linewidth=1.8)
            metrics = candidate.get("metrics")
            if metrics is None:
                subtitle = "ground truth mesh"
            else:
                subtitle = (f"surface RMSE={metrics['surface_z_rmse']:.3g}\n"
                            f"boundary RMSE={metrics['boundary_z_rmse']:.3g}")
            axis.set_title(f"{heading}\n{subtitle}", fontsize=10)
            set_axes(axis, limits)
            if column == 0:
                axis.text2D(0.01, 0.98, f"validation #{sample['index']}",
                            transform=axis.transAxes, va="top", fontsize=9)
    figure.suptitle(
        "Plateau teacher vs Chebyshev label vs neural operators\n"
        "Black curve is the same 64-point input contour in every panel",
        fontsize=15,
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.count < 1 or args.boundary_count < 3:
        raise SystemExit("count must be positive and boundary-count >= 3")
    import torch

    try:
        device = choose_device(args.device, torch)
        records = read_records(args.manifest)
        validation = [record for record in records
                      if record.get("split") in ("val", "validation")]
        by_index = {record["index"]: record for record in validation}
        if args.indices:
            missing = [index for index in args.indices if index not in by_index]
            if missing:
                raise ValueError(f"indices are absent from validation split: {missing}")
            selected = [by_index[index] for index in args.indices]
        else:
            if args.count > len(validation):
                raise ValueError("count exceeds validation split size")
            rng = np.random.default_rng(args.seed)
            chosen = rng.choice(len(validation), size=args.count, replace=False)
            selected = [validation[int(index)] for index in chosen]

        old_model, old_checkpoint = load_model(args.old_checkpoint, device, torch)
        hybrid_model, hybrid_checkpoint = load_model(
            args.hybrid_checkpoint, device, torch)
        if old_model.config.degree != hybrid_model.config.degree:
            raise ValueError("model degrees differ")

        samples = []
        report = {
            "manifest": str(args.manifest),
            "split": "validation",
            "seed": args.seed,
            "boundary_count": args.boundary_count,
            "device": str(device),
            "old_checkpoint": str(args.old_checkpoint),
            "old_epoch": int(old_checkpoint["epoch"]),
            "hybrid_checkpoint": str(args.hybrid_checkpoint),
            "hybrid_epoch": int(hybrid_checkpoint["epoch"]),
            "samples": [],
        }
        for record in selected:
            source_path = resolve_path(args.manifest, record["source_path"])
            coefficient_path = resolve_path(
                args.manifest, record.get("coefficient_path", record["path"]))
            with np.load(source_path, allow_pickle=False) as source:
                vertices = np.asarray(source["surface_vertices"], dtype=np.float64)
                faces = np.asarray(source["faces"], dtype=np.int32)
                boundary_key = ("dense_boundary_points"
                                if "dense_boundary_points" in source
                                else "boundary_points")
                boundary = resample_boundary(source[boundary_key], args.boundary_count)
            with np.load(coefficient_path, allow_pickle=False) as label:
                teacher_coefficients = np.asarray(label["coefficients"], dtype=np.float64)
                indices = np.asarray(label["indices"], dtype=np.int16)

            old_coefficients, old_ms = predict(
                old_model, boundary, device, torch)
            hybrid_coefficients, hybrid_ms = predict(
                hybrid_model, boundary, device, torch)
            teacher_metrics, teacher_surface = evaluate_candidate(
                "teacher_fit", teacher_coefficients, indices, vertices, faces,
                boundary, None)
            old_metrics, old_surface = evaluate_candidate(
                "old", old_coefficients, indices, vertices, faces, boundary, old_ms)
            hybrid_metrics, hybrid_surface = evaluate_candidate(
                "hybrid", hybrid_coefficients, indices, vertices, faces,
                boundary, hybrid_ms)
            weights = l2_weights(indices)
            teacher_power = float(np.sum(weights * teacher_coefficients ** 2))
            for metrics, coefficients in ((old_metrics, old_coefficients),
                                          (hybrid_metrics, hybrid_coefficients)):
                metrics["coefficient_relative_l2"] = float(np.sqrt(
                    np.sum(weights * (coefficients - teacher_coefficients) ** 2) /
                    max(teacher_power, 1e-30)))

            sample_report = {
                "index": record["index"], "sample_id": record.get("sample_id"),
                "source_path": str(record["source_path"]),
                "teacher_fit": teacher_metrics,
                "old": old_metrics, "hybrid": hybrid_metrics,
            }
            report["samples"].append(sample_report)
            samples.append({
                "index": record["index"], "boundary": boundary, "faces": faces,
                "teacher": {"vertices": vertices},
                "teacher_fit": {"vertices": teacher_surface,
                                "metrics": teacher_metrics},
                "old": {"vertices": old_surface, "metrics": old_metrics},
                "hybrid": {"vertices": hybrid_surface,
                           "metrics": hybrid_metrics},
            })
            print(json.dumps(sample_report, ensure_ascii=False), flush=True)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.output_dir / "metrics.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        image_path = args.output_dir / "comparison.png"
        plot(samples, image_path)
        print(json.dumps({"image": str(image_path), "metrics": str(report_path)},
                         ensure_ascii=False), flush=True)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
