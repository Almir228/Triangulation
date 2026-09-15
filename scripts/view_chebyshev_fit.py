#!/usr/bin/env python3
"""Interactively compare Plateau meshes with their Chebyshev zero surfaces."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from minsurf_nn.spectral import (  # noqa: E402
    SpectralFit,
    boundary_edges,
    extract_graph_zero_mesh,
)


DEFAULT_MANIFEST = Path("dataset/plateau-100k-chebyshev-p10/manifest.jsonl")


@dataclass(frozen=True)
class ManifestRecord:
    index: int
    sample_id: str
    split: str
    source_path: str
    coefficient_path: str
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class SurfaceView:
    record: ManifestRecord
    vertices: np.ndarray
    faces: np.ndarray
    boundary: np.ndarray
    predicted: np.ndarray
    degree: int
    coefficient_energy: np.ndarray
    root_diagnostics: Mapping[str, float]
    metrics: Mapping[str, float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST,
        help="Chebyshev dataset manifest (default: %(default)s)",
    )
    parser.add_argument(
        "--index", type=int, default=None,
        help="source sample index to open; by default the first manifest record",
    )
    parser.add_argument("--save", type=Path, help="also save the initial view as PNG/PDF")
    parser.add_argument(
        "--save-dir", type=Path, default=Path("runs/chebyshev-viewer"),
        help="directory used by the Save button (default: %(default)s)",
    )
    parser.add_argument("--no-show", action="store_true", help="render without opening a window")
    parser.add_argument("--cache-size", type=int, default=8, help="decoded samples kept in RAM")
    parser.add_argument("--elevation", type=float, default=24.0)
    parser.add_argument("--azimuth", type=float, default=-58.0)
    return parser


def read_manifest(path: Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    seen: set[int] = set()
    with path.open(encoding="utf-8") as stream:
        for position, line in enumerate(stream):
            if not line.strip():
                continue
            raw = json.loads(line)
            index = int(raw.get("index", position))
            if index < 0 or index in seen:
                raise ValueError(f"manifest contains an invalid/duplicate index {index}")
            coefficient_path = raw.get("coefficient_path", raw.get("path"))
            source_path = raw.get("source_path")
            if not isinstance(source_path, str) or not isinstance(coefficient_path, str):
                raise ValueError(f"manifest record {index} has no source/coefficient path")
            diagnostics = raw.get("diagnostics", {})
            records.append(ManifestRecord(
                index=index,
                sample_id=str(raw.get("sample_id", f"sample_{index:06d}")),
                split=str(raw.get("split", "unknown")),
                source_path=source_path,
                coefficient_path=coefficient_path,
                diagnostics=diagnostics if isinstance(diagnostics, dict) else {},
            ))
            seen.add(index)
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    records.sort(key=lambda record: record.index)
    return records


def resolve_record_path(manifest_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_dir / path).resolve()


def mesh_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    crosses = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    return float(0.5 * np.linalg.norm(crosses, axis=1).sum())


def _ordered_boundary(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Order a single disk boundary loop, independent of vertex numbering."""

    adjacency: dict[int, list[int]] = {}
    for first, second in np.asarray(edges, dtype=np.int64):
        adjacency.setdefault(int(first), []).append(int(second))
        adjacency.setdefault(int(second), []).append(int(first))
    if not adjacency or any(len(neighbours) != 2 for neighbours in adjacency.values()):
        return vertices[np.unique(edges)]

    start = min(adjacency)
    ordered = [start]
    previous = -1
    current = start
    while True:
        candidates = adjacency[current]
        following = candidates[0] if candidates[0] != previous else candidates[1]
        if following == start:
            break
        if following in ordered:
            return vertices[np.unique(edges)]
        ordered.append(following)
        previous, current = current, following
    return vertices[np.asarray(ordered, dtype=np.int64)]


class ChebyshevDatasetViewer:
    def __init__(self, manifest: Path, start_index: int | None, cache_size: int,
                 save_dir: Path, elevation: float, azimuth: float) -> None:
        if cache_size < 1:
            raise ValueError("--cache-size must be positive")
        self.manifest = manifest.resolve()
        self.records = read_manifest(self.manifest)
        self.position_by_index = {
            record.index: position for position, record in enumerate(self.records)
        }
        if start_index is None:
            self.position = 0
        elif start_index in self.position_by_index:
            self.position = self.position_by_index[start_index]
        else:
            raise ValueError(f"index {start_index} is absent from {self.manifest}")

        self.save_dir = save_dir
        self.elevation = elevation
        self.azimuth = azimuth
        self._load = lru_cache(maxsize=cache_size)(self._load_uncached)

        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        from matplotlib.widgets import Button, TextBox

        self.plt = plt
        self.Normalize = Normalize
        self.figure = plt.figure(figsize=(15.5, 7.4))
        grid = self.figure.add_gridspec(
            1, 3, left=0.025, right=0.91, bottom=0.24, top=0.88, wspace=0.04
        )
        self.axes = [self.figure.add_subplot(grid[0, column], projection="3d")
                     for column in range(3)]

        self.scalar_mappable = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="magma")
        self.scalar_mappable.set_array(np.asarray([0.0, 1.0]))
        color_axis = self.figure.add_axes([0.925, 0.29, 0.012, 0.42])
        self.colorbar = self.figure.colorbar(self.scalar_mappable, cax=color_axis)
        self.colorbar.set_label(r"$|z_{fit}-z_{true}|$")

        self.metrics_text = self.figure.text(0.025, 0.175, "", fontsize=9.5, va="top")
        self.status_text = self.figure.text(0.025, 0.014, "", fontsize=8.5, color="#444444")

        previous_axis = self.figure.add_axes([0.37, 0.055, 0.08, 0.052])
        random_axis = self.figure.add_axes([0.455, 0.055, 0.08, 0.052])
        next_axis = self.figure.add_axes([0.54, 0.055, 0.08, 0.052])
        index_axis = self.figure.add_axes([0.665, 0.055, 0.105, 0.052])
        save_axis = self.figure.add_axes([0.80, 0.055, 0.08, 0.052])
        self.previous_button = Button(previous_axis, "← Prev")
        self.random_button = Button(random_axis, "Random")
        self.next_button = Button(next_axis, "Next →")
        self.index_box = TextBox(index_axis, "Index ", initial=str(self.current.index))
        self.save_button = Button(save_axis, "Save PNG")
        self._updating_text_box = False

        self.previous_button.on_clicked(lambda _event: self.move(-1))
        self.random_button.on_clicked(lambda _event: self.random())
        self.next_button.on_clicked(lambda _event: self.move(1))
        self.index_box.on_submit(self.jump)
        self.save_button.on_clicked(lambda _event: self.save())
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.figure.canvas.mpl_connect("button_release_event", self.synchronise_camera)

    @property
    def current(self) -> ManifestRecord:
        return self.records[self.position]

    def _load_uncached(self, position: int) -> SurfaceView:
        record = self.records[position]
        manifest_dir = self.manifest.parent
        source_path = resolve_record_path(manifest_dir, record.source_path)
        coefficient_path = resolve_record_path(manifest_dir, record.coefficient_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"source NPZ is missing: {source_path}")
        if not coefficient_path.is_file():
            raise FileNotFoundError(f"coefficient NPZ is missing: {coefficient_path}")

        with np.load(source_path, allow_pickle=False) as source:
            vertices = np.asarray(source["surface_vertices"], dtype=np.float64)
            faces = np.asarray(source["faces"], dtype=np.int32)
            if "dense_boundary_points" in source:
                boundary = np.asarray(source["dense_boundary_points"], dtype=np.float64)
            elif "boundary_points" in source:
                boundary = np.asarray(source["boundary_points"], dtype=np.float64)
            else:
                boundary = _ordered_boundary(vertices, boundary_edges(faces))

        with np.load(coefficient_path, allow_pickle=False) as coefficient_file:
            coefficients = np.asarray(coefficient_file["coefficients"], dtype=np.float64)
            indices = np.asarray(coefficient_file["indices"], dtype=np.int16)
            degree = (int(coefficient_file["degree"])
                      if "degree" in coefficient_file else int(indices.sum(axis=1).max()))
            energy = (np.asarray(coefficient_file["coefficient_energy"], dtype=np.float64)
                      if "coefficient_energy" in coefficient_file else np.asarray([]))

        fit = SpectralFit(degree, indices, coefficients, dict(record.diagnostics))
        predicted, root_diagnostics = extract_graph_zero_mesh(vertices, faces, fit)
        differences = predicted[:, 2] - vertices[:, 2]
        target_area = mesh_area(vertices, faces)
        fit_area = mesh_area(predicted, faces)
        surface_values = fit.evaluate(vertices)
        boundary_values = fit.evaluate(boundary)
        metrics = {
            "vertical_rmse": float(np.sqrt(np.mean(differences ** 2))),
            "vertical_max": float(np.max(np.abs(differences))),
            "surface_field_rmse": float(np.sqrt(np.mean(surface_values ** 2))),
            "boundary_field_rmse": float(np.sqrt(np.mean(boundary_values ** 2))),
            "target_area": target_area,
            "fit_area": fit_area,
            "relative_area_error": float((fit_area - target_area) / target_area),
        }
        return SurfaceView(record, vertices, faces, boundary, predicted, degree, energy,
                           root_diagnostics, metrics)

    @staticmethod
    def _closed(points: np.ndarray) -> np.ndarray:
        return np.vstack((points, points[0])) if len(points) else points

    @staticmethod
    def _add_mesh(axis, vertices: np.ndarray, faces: np.ndarray, facecolors,
                  alpha: float, edgecolor=(0.1, 0.1, 0.1, 0.18)) -> None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        collection = Poly3DCollection(
            vertices[faces], facecolors=facecolors, edgecolors=edgecolor,
            linewidths=0.15, alpha=alpha,
        )
        axis.add_collection3d(collection)

    def _set_axes(self, axis, all_vertices: np.ndarray, title: str,
                  elevation: float, azimuth: float) -> None:
        low = all_vertices.min(axis=0)
        high = all_vertices.max(axis=0)
        span = np.maximum(high - low, 1e-5)
        center = 0.5 * (low + high)
        margin = 0.06
        axis.set_xlim(center[0] - span[0] * (0.5 + margin),
                      center[0] + span[0] * (0.5 + margin))
        axis.set_ylim(center[1] - span[1] * (0.5 + margin),
                      center[1] + span[1] * (0.5 + margin))
        axis.set_zlim(center[2] - span[2] * (0.5 + margin),
                      center[2] + span[2] * (0.5 + margin))
        axis.set_box_aspect((span[0], span[1], max(span[2], 0.35 * max(span[:2]))))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.set_title(title, pad=12)
        axis.view_init(elev=elevation, azim=azimuth)

    def render(self) -> SurfaceView:
        elevation = getattr(self.axes[0], "elev", self.elevation)
        azimuth = getattr(self.axes[0], "azim", self.azimuth)
        view = self._load(self.position)
        vertices, predicted, faces = view.vertices, view.predicted, view.faces
        closed_boundary = self._closed(view.boundary)
        errors = np.abs(predicted[:, 2] - vertices[:, 2])
        face_errors = errors[faces].mean(axis=1)
        maximum = max(float(errors.max()), 1e-12)
        norm = self.Normalize(0.0, maximum)
        self.scalar_mappable.set_norm(norm)
        self.scalar_mappable.set_array(errors)
        self.colorbar.update_normal(self.scalar_mappable)

        for axis in self.axes:
            axis.clear()

        self._add_mesh(self.axes[0], vertices, faces, "#69b3d0", 0.92)
        self.axes[0].plot(*closed_boundary.T, color="#b51f2e", linewidth=2.2)

        fit_colors = self.plt.get_cmap("magma")(norm(face_errors))
        self._add_mesh(self.axes[1], predicted, faces, fit_colors, 0.95)
        fitted_boundary = _ordered_boundary(predicted, boundary_edges(faces))
        self.axes[1].plot(*self._closed(fitted_boundary).T, color="#1b1b1b", linewidth=1.5)

        self._add_mesh(self.axes[2], vertices, faces, "#8b9199", 0.28,
                       edgecolor=(0.2, 0.2, 0.2, 0.08))
        self._add_mesh(self.axes[2], predicted, faces, "#15a6a6", 0.58,
                       edgecolor=(0.05, 0.2, 0.2, 0.12))
        self.axes[2].plot(*closed_boundary.T, color="#b51f2e", linewidth=2.0)

        all_vertices = np.vstack((vertices, predicted, view.boundary))
        titles = ("Plateau surface (ground truth)",
                  f"Chebyshev zero sheet, p={view.degree}",
                  "Overlay: gray truth / teal fit")
        for axis, title in zip(self.axes, titles):
            self._set_axes(axis, all_vertices, title, elevation, azimuth)

        diagnostics = view.record.diagnostics
        metrics = view.metrics
        condition = float(diagnostics.get("condition_number", float("nan")))
        self.figure.suptitle(
            f"{view.record.sample_id}   •   index {view.record.index}   •   "
            f"{self.position + 1:,}/{len(self.records):,}   •   split={view.record.split}",
            fontsize=13,
        )
        self.metrics_text.set_text(
            f"geometric z-error: RMSE {metrics['vertical_rmse']:.3e}, "
            f"max {metrics['vertical_max']:.3e}    |    "
            f"F(target) RMSE {metrics['surface_field_rmse']:.3e}, "
            f"F(boundary) RMSE {metrics['boundary_field_rmse']:.3e}    |    "
            f"area: {metrics['target_area']:.6f} → {metrics['fit_area']:.6f} "
            f"({metrics['relative_area_error']:+.3%})\n"
            f"fit condition {condition:.3e}    |    "
            f"zero-root residual max {view.root_diagnostics['root_residual_max']:.3e}, "
            f"failures {int(view.root_diagnostics['root_failures'])}    |    "
            "keys: ←/→, Home/End, R=random, S=save"
        )
        self.status_text.set_text(
            f"source: {resolve_record_path(self.manifest.parent, view.record.source_path)}"
        )
        self._set_index_text(str(view.record.index))
        self.figure.canvas.draw_idle()
        return view

    def _set_index_text(self, value: str) -> None:
        self._updating_text_box = True
        self.index_box.set_val(value)
        self._updating_text_box = False

    def move(self, offset: int) -> None:
        self.position = (self.position + offset) % len(self.records)
        self._render_safely()

    def random(self) -> None:
        if len(self.records) > 1:
            candidate = int(np.random.default_rng().integers(len(self.records) - 1))
            if candidate >= self.position:
                candidate += 1
            self.position = candidate
        self._render_safely()

    def jump(self, text: str) -> None:
        if self._updating_text_box:
            return
        try:
            source_index = int(text)
        except ValueError:
            self.status_text.set_text("Index must be an integer")
            self.figure.canvas.draw_idle()
            return
        position = self.position_by_index.get(source_index)
        if position is None:
            self.status_text.set_text(f"Index {source_index} is absent from the manifest")
            self.figure.canvas.draw_idle()
            return
        self.position = position
        self._render_safely()

    def on_key(self, event) -> None:
        if event.key in ("left", "down", "pageup"):
            self.move(-1)
        elif event.key in ("right", "up", "pagedown"):
            self.move(1)
        elif event.key == "home":
            self.position = 0
            self._render_safely()
        elif event.key == "end":
            self.position = len(self.records) - 1
            self._render_safely()
        elif event.key in ("r", "R"):
            self.random()
        elif event.key in ("s", "S"):
            self.save()

    def synchronise_camera(self, event) -> None:
        if event.inaxes not in self.axes:
            return
        source = event.inaxes
        for axis in self.axes:
            if axis is not source:
                axis.view_init(elev=source.elev, azim=source.azim)
        self.figure.canvas.draw_idle()

    def _render_safely(self) -> None:
        try:
            self.render()
        except Exception as error:  # callbacks must keep the window usable
            self.status_text.set_text(f"ERROR: {error}")
            self.figure.canvas.draw_idle()
            print(f"ERROR: {error}", file=sys.stderr)

    def save(self, path: Path | None = None) -> Path:
        if path is None:
            path = self.save_dir / f"sample_{self.current.index:06d}_p{self._load(self.position).degree}.png"
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(path, dpi=180, bbox_inches="tight")
        self.status_text.set_text(f"Saved: {path}")
        self.figure.canvas.draw_idle()
        print(f"saved {path}")
        return path


def main() -> int:
    args = build_parser().parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    try:
        viewer = ChebyshevDatasetViewer(
            args.manifest, args.index, args.cache_size, args.save_dir,
            args.elevation, args.azimuth,
        )
        view = viewer.render()
        print(json.dumps({
            "index": view.record.index,
            "sample_id": view.record.sample_id,
            "degree": view.degree,
            **view.metrics,
            **view.root_diagnostics,
        }, ensure_ascii=False, sort_keys=True))
        if args.save:
            viewer.save(args.save)
        if not args.no_show:
            viewer.plt.show()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
