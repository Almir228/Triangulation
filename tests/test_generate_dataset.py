"""Checks for the contour dataset format and local signed-distance convention."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dataset_utils import resample_closed_curve, signed_distance  # noqa: E402
from scripts.evaluate_prediction import mesh_area  # noqa: E402


class DatasetGeometryTests(unittest.TestCase):
    def test_mesh_area(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                            dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        self.assertAlmostEqual(mesh_area(vertices, faces), 1.0)

    def test_closed_resampling_has_requested_size(self):
        t = np.linspace(0, 2 * np.pi, 17, endpoint=False)
        curve = np.column_stack((np.cos(t), np.sin(t), 0.2 * np.cos(2 * t)))
        sampled = resample_closed_curve(np.vstack((curve, curve[0])), 32)
        self.assertEqual(sampled.shape, (32, 3))
        self.assertFalse(np.allclose(sampled[0], sampled[-1]))

    def test_signed_distance_uses_listed_face_normal(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        points = np.array([[0.25, 0.25, 0.1], [0.25, 0.25, -0.1], [0.25, 0.25, 0]], dtype=np.float64)
        distances = signed_distance(points, vertices, faces)
        np.testing.assert_allclose(distances[:2], [0.1, -0.1], atol=1e-6)
        self.assertGreaterEqual(float(distances[2]), 0.0)


class DatasetSmokeTests(unittest.TestCase):
    def test_smoke_writes_two_samples_when_solver_is_built(self):
        solver = ROOT / "build" / "triangulation"
        if not solver.is_file():
            self.skipTest("build/triangulation is not available")
        with tempfile.TemporaryDirectory(prefix="triangulation_dataset_test_") as directory:
            output = Path(directory) / "samples"
            subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_dataset.py"),
                            "--smoke", "--solver", str(solver), "--output-dir", str(output)],
                           check=True, capture_output=True, text=True)
            manifest = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual(len(manifest), 2)
            self.assertEqual([record["split"] for record in manifest], ["train", "val"])
            for record in manifest:
                sample = np.load(output / record["path"])
                self.assertEqual(sample["boundary_points"].shape, (32, 3))
                self.assertEqual(sample["query_points"].shape, (1024, 3))
                self.assertEqual(sample["surface_vertices"].shape[1], 3)
                self.assertEqual(sample["faces"].shape[1], 3)
                self.assertEqual(json.loads(str(sample["metadata"]))["format_version"], 1)
                boundary = sample["boundary_points"]
                np.testing.assert_allclose(boundary.mean(axis=0), 0.0, atol=2e-6)
                np.testing.assert_allclose(np.linalg.norm(boundary, axis=1).max(), 1.0,
                                           atol=2e-6)


if __name__ == "__main__":
    unittest.main()
