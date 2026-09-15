#!/usr/bin/env python3
"""Generate fresh contours and compare neural inference with the Plateau solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataset_utils import read_obj_triangles  # noqa: E402
from generate_dataset import prepare_canonical_contour  # noqa: E402
from minsurf_nn.spectral import evaluate_polynomial  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path,
                        default=Path("runs/spectral-operator-p10-n64/best.pt"))
    result.add_argument("--solver", type=Path,
                        default=Path("build-benchmark/triangulation"))
    result.add_argument("--generation-config", type=Path,
                        default=Path("dataset/plateau-100k/generation_config.json"))
    result.add_argument("--output-dir", type=Path,
                        default=Path("runs/inference-vs-triangulation"))
    result.add_argument("--count", type=int, default=2)
    result.add_argument("--start-index", type=int, default=100_001)
    result.add_argument("--resolution", type=int, default=64)
    result.add_argument("--inference-repeats", type=int, default=200)
    result.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"),
                        default="auto")
    return result


def choose_device(name, torch):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def synchronize(device, torch) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def write_contour(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for x, y, z in points:
            stream.write(f"{x:.17g},{y:.17g},{z:.17g}\n")


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# Neural Chebyshev zero level clipped to the contour projection\n")
        for x, y, z in vertices:
            stream.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in faces + 1:
            stream.write(f"f {a} {b} {c}\n")


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    inside = np.zeros(len(points), dtype=bool)
    previous = polygon[-1]
    x, y = points[:, 0], points[:, 1]
    for current in polygon:
        crossing = (current[1] > y) != (previous[1] > y)
        at_x = ((previous[0] - current[0]) * (y - current[1]) /
                (previous[1] - current[1] + 1e-30) + current[0])
        inside ^= crossing & (x < at_x)
        previous = current
    return inside


def neural_zero_mesh(coefficients, indices, boundary, resolution):
    from scipy.spatial import Delaunay

    started = perf_counter()
    xy_axis = np.linspace(-1.0, 1.0, resolution, dtype=np.float64)
    xy_grid = np.stack(np.meshgrid(xy_axis, xy_axis, indexing="ij"), axis=-1).reshape(-1, 2)
    xy = np.vstack((xy_grid[points_in_polygon(xy_grid, boundary[:, :2])],
                    boundary[:, :2].astype(np.float64)))
    triangulation = Delaunay(xy)
    faces = triangulation.simplices.astype(np.int32)
    faces = faces[points_in_polygon(xy[faces].mean(axis=1), boundary[:, :2])]

    z_axis = np.linspace(-1.0, 1.0, 129, dtype=np.float64)
    values = np.empty((len(xy), len(z_axis)), dtype=np.float64)
    chunk = 256
    for start in range(0, len(xy), chunk):
        stop = min(start + chunk, len(xy))
        count = stop - start
        probes = np.empty((count, len(z_axis), 3), dtype=np.float64)
        probes[..., :2] = xy[start:stop, None, :]
        probes[..., 2] = z_axis[None, :]
        values[start:stop] = evaluate_polynomial(
            probes.reshape(-1, 3), coefficients, indices).reshape(count, -1)
    crossings = values[:, :-1] * values[:, 1:] <= 0.0
    midpoints = 0.5 * (z_axis[:-1] + z_axis[1:])
    crossing_index = np.argmin(np.where(crossings, np.abs(midpoints)[None, :], np.inf), axis=1)
    valid = crossings[np.arange(len(xy)), crossing_index]
    if np.count_nonzero(valid) < 3:
        raise RuntimeError(
            f"predicted polynomial has no graph-like zero sheet: "
            f"range=[{values.min():.4g}, {values.max():.4g}]")
    lower = z_axis[crossing_index].copy()
    upper = z_axis[crossing_index + 1].copy()
    lower_value = values[np.arange(len(xy)), crossing_index].copy()
    active = np.flatnonzero(valid)
    for _ in range(30):
        middle = 0.5 * (lower[active] + upper[active])
        probes = np.column_stack((xy[active], middle))
        middle_value = evaluate_polynomial(probes, coefficients, indices)
        left = lower_value[active] * middle_value <= 0.0
        upper[active[left]] = middle[left]
        lower[active[~left]] = middle[~left]
        lower_value[active[~left]] = middle_value[~left]
    z = 0.5 * (lower + upper)
    vertices = np.column_stack((xy, z))
    faces = faces[np.all(valid[faces], axis=1)]
    if not len(faces):
        raise RuntimeError("predicted roots do not form triangles inside the contour")
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    vertices = vertices[used]
    faces = inverse.reshape(-1, 3).astype(np.int32)
    return (vertices, faces, perf_counter() - started,
            (float(values.min()), float(values.max())))


def chamfer_distance(first: np.ndarray, second: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    forward = cKDTree(second).query(first, workers=-1)[0]
    backward = cKDTree(first).query(second, workers=-1)[0]
    return float(0.5 * (forward.mean() + backward.mean()))


def plot_results(samples, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(16, 7.6), constrained_layout=True)
    for row, sample in enumerate(samples):
        boundary = sample["boundary"]
        for column, title in enumerate(("Новый контур", "C++ Plateau", "Нейросеть p=10")):
            axis = figure.add_subplot(len(samples), 4, row * 4 + column + 1,
                                      projection="3d")
            axis.plot(*np.vstack((boundary, boundary[0])).T, color="black",
                      linewidth=1.5)
            if column == 1:
                vertices, faces = sample["solver_vertices"], sample["solver_faces"]
                axis.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                  triangles=faces, color="#4c78a8", alpha=0.82,
                                  linewidth=0.0)
            elif column == 2 and sample.get("neural_vertices") is not None:
                vertices, faces = sample["neural_vertices"], sample["neural_faces"]
                axis.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                  triangles=faces, color="#f58518", alpha=0.82,
                                  linewidth=0.0)
            elif column == 2:
                axis.text2D(0.12, 0.5, sample["neural_error"], transform=axis.transAxes,
                            color="crimson", wrap=True)
            axis.set_title(title if row == 0 else "")
            axis.set_xlim(-1, 1)
            axis.set_ylim(-1, 1)
            axis.set_zlim(-0.65, 0.65)
            axis.set_box_aspect((1, 1, 0.65))
            axis.view_init(elev=25, azim=-55)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.set_zlabel("z")
            axis.tick_params(labelsize=7)
        time_axis = figure.add_subplot(len(samples), 4, row * 4 + 4)
        values = [sample["neural_inference_ms"], sample["neural_total_ms"],
                  sample["solver_ms"]]
        labels = ["NN forward", "NN + zero mesh", "C++ Plateau"]
        bars = time_axis.barh(labels, values, color=["#f58518", "#eeca3b", "#4c78a8"])
        time_axis.set_xscale("log")
        time_axis.set_xlabel("Время, мс (логарифмическая шкала)")
        time_axis.set_title(f"Контур {row + 1}: время")
        for bar, value in zip(bars, values):
            time_axis.text(value * 1.06, bar.get_y() + bar.get_height() / 2,
                           f"{value:.3g} мс", va="center", fontsize=9)
        time_axis.grid(axis="x", alpha=0.25)
    figure.suptitle("Два новых контура: решение Плато и нейросетевая аппроксимация",
                    fontsize=15)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.count < 1 or args.resolution < 16 or args.inference_repeats < 1:
        raise SystemExit("count/repeats must be positive and resolution >= 16")
    import torch
    from minsurf_nn.spectral_operator import ContourToChebyshev, SpectralOperatorConfig

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.generation_config.read_text(encoding="utf-8"))
    device = choose_device(args.device, torch)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    load_started = perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ContourToChebyshev(SpectralOperatorConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    synchronize(device, torch)
    model_load_ms = (perf_counter() - load_started) * 1000.0

    samples = []
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "model_load_ms": model_load_ms,
        "surface_grid_resolution": args.resolution,
        "samples": [],
    }
    for offset in range(args.count):
        index = args.start_index + offset
        generated = perf_counter()
        boundary, *_ = prepare_canonical_contour(
            index=index, base_seed=int(config["seed"]),
            boundary_count=int(config["boundary_size"]),
            raw_contour_count=int(config["raw_contour_size"]),
            fourier_modes=int(config["fourier_modes"]),
            fourier_decay=float(config["fourier_decay"]),
            xy_variation=float(config["xy_variation"]),
            nonplanarity=float(config["nonplanarity"]),
            min_separation=float(config["min_separation"]),
            max_curvature=float(config["max_curvature"]),
        )
        generation_ms = (perf_counter() - generated) * 1000.0
        stem = f"contour_{offset + 1}"
        contour_path = args.output_dir / f"{stem}.csv"
        solver_prefix = args.output_dir / f"{stem}_plateau"
        write_contour(contour_path, boundary)
        command = [
            str(args.solver.resolve()), "--contour", str(contour_path.resolve()),
            "--normal", "0", "0", "1", "--mode", str(config["solver_mode"]),
            "--refine", str(config["refine"]), "--iterations", str(config["iterations"]),
            "--remesh-passes", str(config["remesh_passes"]),
            "--tolerance", str(config["tolerance"]), "--output", str(solver_prefix),
        ]
        solver_started = perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        solver_ms = (perf_counter() - solver_started) * 1000.0
        if completed.returncode not in (0, 2):
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        solver_vertices, solver_faces = read_obj_triangles(solver_prefix.with_suffix(".obj"))

        boundary_tensor = torch.from_numpy(boundary).unsqueeze(0).to(device)
        with torch.no_grad():
            for _ in range(20):
                model(boundary_tensor)
            synchronize(device, torch)
            timings = []
            for _ in range(args.inference_repeats):
                started = perf_counter()
                coefficients_tensor = model(boundary_tensor)
                synchronize(device, torch)
                timings.append((perf_counter() - started) * 1000.0)
        coefficients = coefficients_tensor[0].cpu().numpy().astype(np.float64)
        inference_ms = float(np.median(timings))
        inference_p95_ms = float(np.percentile(timings, 95))
        boundary_values = np.abs(evaluate_polynomial(
            boundary.astype(np.float64), coefficients, model.indices.cpu().numpy()))
        neural_vertices = neural_faces = None
        neural_error = None
        extraction_ms = float("nan")
        field_range = None
        chamfer = float("nan")
        try:
            neural_vertices, neural_faces, extraction_seconds, field_range = neural_zero_mesh(
                coefficients, model.indices.cpu().numpy(), boundary, args.resolution)
            extraction_ms = extraction_seconds * 1000.0
            write_obj(args.output_dir / f"{stem}_neural.obj",
                      neural_vertices, neural_faces)
            chamfer = chamfer_distance(solver_vertices, neural_vertices)
        except RuntimeError as error:
            neural_error = str(error)
        neural_total_ms = inference_ms + extraction_ms
        sample_report = {
            "sample": offset + 1, "generator_index": index,
            "generation_ms": generation_ms,
            "solver_ms": solver_ms, "solver_returncode": completed.returncode,
            "solver_vertices": len(solver_vertices), "solver_faces": len(solver_faces),
            "neural_inference_median_ms": inference_ms,
            "neural_inference_p95_ms": inference_p95_ms,
            "neural_surface_extraction_ms": extraction_ms,
            "neural_total_ms": neural_total_ms,
            "neural_vertices": 0 if neural_vertices is None else len(neural_vertices),
            "neural_faces": 0 if neural_faces is None else len(neural_faces),
            "boundary_abs_field_mean": float(boundary_values.mean()),
            "boundary_abs_field_max": float(boundary_values.max()),
            "vertex_chamfer": chamfer,
            "field_range": field_range, "neural_error": neural_error,
        }
        report["samples"].append(sample_report)
        samples.append({
            "boundary": boundary, "solver_vertices": solver_vertices,
            "solver_faces": solver_faces, "neural_vertices": neural_vertices,
            "neural_faces": neural_faces, "neural_error": neural_error,
            "neural_inference_ms": inference_ms,
            "neural_total_ms": neural_total_ms, "solver_ms": solver_ms,
        })
        print(json.dumps(sample_report, ensure_ascii=False), flush=True)

    (args.output_dir / "benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_results(samples, args.output_dir / "comparison.png")
    print(f"saved {args.output_dir / 'comparison.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
