#!/usr/bin/env python3
"""Compare an inferred OBJ with the target mesh stored in a dataset NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.dataset_utils import read_obj_triangles, signed_distance  # noqa: E402


def mesh_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    return float(0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0],
                 triangles[:, 2] - triangles[:, 0]), axis=1).sum())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="dataset NPZ with target mesh")
    parser.add_argument("prediction", type=Path, help="predicted OBJ in world coordinates")
    parser.add_argument("--output", type=Path, help="optional JSON metrics file")
    args = parser.parse_args(argv)

    with np.load(args.target, allow_pickle=False) as sample:
        target_vertices = np.asarray(sample["surface_vertices"], dtype=np.float64)
        target_faces = np.asarray(sample["faces"], dtype=np.int32)
        boundary = np.asarray(sample["boundary_points"], dtype=np.float64)
        center = np.asarray(sample["normalization_center"], dtype=np.float64)
        scale = float(sample["normalization_scale"])
    prediction_world, prediction_faces = read_obj_triangles(args.prediction)
    prediction = (prediction_world - center) / scale

    target_to_prediction = np.abs(signed_distance(
        target_vertices, prediction, prediction_faces))
    prediction_to_target = np.abs(signed_distance(
        prediction, target_vertices, target_faces))
    boundary_error = np.abs(signed_distance(boundary, prediction, prediction_faces))
    target_area = mesh_area(target_vertices, target_faces) * scale * scale
    prediction_area = mesh_area(prediction, prediction_faces) * scale * scale
    metrics = {
        "target": str(args.target),
        "prediction": str(args.prediction),
        "vertices": int(len(prediction)),
        "faces": int(len(prediction_faces)),
        "target_area_world": target_area,
        "prediction_area_world": prediction_area,
        "area_ratio": prediction_area / target_area,
        "surface_distance_mean_world": float(
            0.5 * (target_to_prediction.mean() + prediction_to_target.mean()) * scale),
        "surface_distance_max_world": float(
            max(target_to_prediction.max(), prediction_to_target.max()) * scale),
        "boundary_distance_mean_world": float(boundary_error.mean() * scale),
        "boundary_distance_max_world": float(boundary_error.max() * scale),
    }
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
