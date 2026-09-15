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
from scripts.dataset_utils import (canonicalize_contour, is_strictly_convex_xy, oriented_area_vector,
                                   resample_closed_curve, signed_distance)  # noqa: E402
from scripts.evaluate_prediction import mesh_area  # noqa: E402
from scripts.generate_dataset import _query_points, _split_for, generate_contour  # noqa: E402


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

    def test_canonicalization_aligns_area_and_round_trips(self):
        t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        original = np.column_stack((
            np.cos(t), 0.8 * np.sin(t),
            0.25 * np.cos(t + 0.4) + 0.15 * np.sin(2 * t)))
        original += np.array([2.0, -3.0, 0.7])
        canonical, transform = canonicalize_contour(original, margin=0.1)
        area = oriented_area_vector(canonical)
        np.testing.assert_allclose(area[:2], 0.0, atol=1e-12)
        self.assertGreater(float(area[2]), 0.0)
        self.assertLessEqual(float(np.linalg.norm(canonical, axis=1).max()), 0.9 + 1e-12)
        recovered = (canonical * float(transform["scale"])) @ transform["rotation"]
        recovered += transform["center"]
        np.testing.assert_allclose(recovered, original, atol=1e-12)

    def test_zero_query_points_are_supported_for_mesh_only_datasets(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        points = _query_points(vertices, faces, np.random.default_rng(1), 0, 0.5, 0.35)
        self.assertEqual(points.shape, (0, 3))
        self.assertEqual(points.dtype, np.float32)

    def test_large_split_is_deterministic_and_non_contiguous(self):
        first = [_split_for(index, 1000, 123) for index in range(1000)]
        second = [_split_for(index, 1000, 123) for index in range(1000)]
        self.assertEqual(first, second)
        self.assertGreater(first.count("val"), 20)
        self.assertGreater(first.count("test"), 20)
        self.assertGreater(first.count("train"), 850)

    def test_fourier_contours_are_convex_and_area_aligned(self):
        contours = [generate_contour(np.random.default_rng(seed), 128)
                    for seed in range(20)]
        parameter_vectors = []
        for points, metadata in contours:
            self.assertTrue(is_strictly_convex_xy(points))
            area = oriented_area_vector(points)
            np.testing.assert_allclose(area[:2], 0.0, atol=2e-12)
            self.assertGreater(float(area[2]), 0.0)
            quality = metadata["quality"]
            self.assertGreater(quality["minimum_separation_over_diameter"], 0.0)
            self.assertLess(quality["maximum_curvature_times_diameter"], 100.0)
            parameter_vectors.append(metadata["parameters"]["xy_cos"])
        self.assertFalse(np.allclose(parameter_vectors[0], parameter_vectors[1]))


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
                self.assertEqual(sample["dense_boundary_points"].shape, (32, 3))
                self.assertEqual(sample["query_points"].shape, (1024, 3))
                self.assertEqual(sample["surface_vertices"].shape[1], 3)
                self.assertEqual(sample["faces"].shape[1], 3)
                self.assertEqual(json.loads(str(sample["metadata"]))["format_version"], 2)
                boundary = sample["boundary_points"]
                np.testing.assert_allclose(boundary.mean(axis=0), 0.0, atol=2e-6)
                np.testing.assert_allclose(np.linalg.norm(boundary, axis=1).max(), 0.9,
                                           atol=2e-6)
                area = oriented_area_vector(boundary)
                np.testing.assert_allclose(area[:2], 0.0, atol=2e-6)
                self.assertGreater(float(area[2]), 0.0)
                self.assertEqual(sample["normalization_rotation"].shape, (3, 3))

    def test_parallel_mesh_only_generation_resumes(self):
        solver = ROOT / "build" / "triangulation"
        if not solver.is_file():
            self.skipTest("build/triangulation is not available")
        with tempfile.TemporaryDirectory(prefix="triangulation_parallel_test_") as directory:
            output = Path(directory) / "samples"
            command = [
                sys.executable, str(ROOT / "scripts" / "generate_dataset_parallel.py"),
                "--samples", "4", "--workers", "2", "--queries", "0",
                "--boundary-size", "32", "--raw-contour-size", "32",
                "--refine", "0", "--iterations", "80", "--shard-size", "2",
                "--progress-every", "1", "--solver", str(solver),
                "--output-dir", str(output),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            manifest_path = output / "manifest.jsonl"
            first_manifest = manifest_path.read_text(encoding="utf-8")
            records = [json.loads(line) for line in first_manifest.splitlines()]
            self.assertEqual([record["index"] for record in records], list(range(4)))
            self.assertEqual(len(list(output.glob("shard-*/*.npz"))), 4)
            for record in records:
                with np.load(output / record["path"]) as sample:
                    self.assertEqual(sample["dense_boundary_points"].shape, (32, 3))
                    self.assertEqual(sample["query_points"].shape, (0, 3))
                    self.assertEqual(sample["signed_distance"].shape, (0,))

            manifest_path.write_text("\n".join(first_manifest.splitlines()[:-1]) + "\n",
                                     encoding="utf-8")
            recovered = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertIn("recovered 1", recovered.stdout)
            self.assertEqual(manifest_path.read_text(encoding="utf-8"), first_manifest)
            resumed = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertIn("already complete", resumed.stdout)
            summary = json.loads((output / "generation_summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["completed_samples"], 4)


if __name__ == "__main__":
    unittest.main()
