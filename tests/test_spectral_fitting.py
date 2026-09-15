"""Tests for total-degree Chebyshev fitting of normalized triangle meshes."""

import json
import subprocess
import sys
import tempfile
import unittest
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from minsurf_nn.spectral import (evaluate_gradient, evaluate_polynomial,
                                 extract_graph_zero_mesh, fit_triangulation,
                                 total_degree_indices, validate_mesh)  # noqa: E402


def plane_mesh():
    vertices = np.array([
        [-0.75, -0.75, 0.0], [0.0, -0.75, 0.0], [0.75, -0.75, 0.0],
        [-0.75, 0.0, 0.0], [0.0, 0.0, 0.0], [0.75, 0.0, 0.0],
        [-0.75, 0.75, 0.0], [0.0, 0.75, 0.0], [0.75, 0.75, 0.0],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
        [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7],
    ], dtype=np.int32)
    return vertices, faces


class SpectralBasisTests(unittest.TestCase):
    def test_total_degree_counts_and_order(self):
        for degree in (0, 1, 4, 10):
            indices = total_degree_indices(degree)
            self.assertEqual(len(indices), comb(degree + 3, 3))
            self.assertTrue(np.all(indices.sum(axis=1) <= degree))
            self.assertTrue(np.all(np.diff(indices.sum(axis=1)) >= 0))

    def test_known_polynomial_evaluation(self):
        indices = total_degree_indices(2)
        coefficients = np.zeros(len(indices))
        coefficients[np.flatnonzero(np.all(indices == (1, 0, 0), axis=1))[0]] = 2.0
        coefficients[np.flatnonzero(np.all(indices == (0, 2, 0), axis=1))[0]] = -0.5
        points = np.array([[0.2, -0.3, 0.4], [-0.7, 0.6, -0.1]])
        expected = 2.0 * points[:, 0] - 0.5 * (2.0 * points[:, 1] ** 2 - 1.0)
        np.testing.assert_allclose(
            evaluate_polynomial(points, coefficients, indices), expected, atol=1e-14)
        expected_gradient = np.column_stack((
            np.full(len(points), 2.0), -2.0 * points[:, 1], np.zeros(len(points))))
        np.testing.assert_allclose(
            evaluate_gradient(points, coefficients, indices), expected_gradient, atol=1e-14)


class SpectralFitTests(unittest.TestCase):
    def test_plane_recovers_signed_height(self):
        vertices, faces = plane_mesh()
        result = fit_triangulation(
            vertices, faces, degree=2, surface_samples=800, boundary_samples=300,
            offset_distance=0.08, boundary_weight=5.0, ridge=0.0, seed=17)
        rng = np.random.default_rng(18)
        points = rng.uniform(-0.7, 0.7, size=(200, 3))
        np.testing.assert_allclose(result.evaluate(points), points[:, 2], atol=2e-12)
        self.assertEqual(result.diagnostics["rank"], comb(5, 3))
        self.assertLess(result.diagnostics["boundary_rmse"], 1e-12)
        displaced = vertices.copy()
        displaced[:, 2] = 0.12
        extracted, diagnostics = extract_graph_zero_mesh(displaced, faces, result)
        np.testing.assert_allclose(extracted[:, 2], 0.0, atol=1e-10)
        self.assertEqual(diagnostics["root_failures"], 0)

    def test_rejects_wrong_vector_area(self):
        vertices, faces = plane_mesh()
        with self.assertRaisesRegex(ValueError, "toward \\+z"):
            validate_mesh(vertices, faces[:, ::-1])
        rotated = vertices[:, [2, 1, 0]]
        with self.assertRaises(ValueError):
            validate_mesh(rotated, faces[:, ::-1])

    def test_cli_writes_reusable_coefficients(self):
        vertices, faces = plane_mesh()
        with tempfile.TemporaryDirectory(prefix="spectral_fit_test_") as directory:
            source = Path(directory) / "surface.npz"
            output = Path(directory) / "fit.npz"
            np.savez(source, surface_vertices=vertices, faces=faces)
            completed = subprocess.run([
                sys.executable, str(ROOT / "scripts" / "fit_chebyshev_surface.py"),
                str(source), "--output", str(output), "--degree", "2",
                "--surface-samples", "300", "--boundary-samples", "100",
                "--offset-distance", "0.08", "--ridge", "0", "--seed", "4",
            ], check=True, capture_output=True, text=True)
            self.assertTrue(output.is_file(), completed.stdout)
            with np.load(output, allow_pickle=False) as fitted:
                self.assertEqual(fitted["coefficients"].shape, (10,))
                self.assertEqual(fitted["indices"].shape, (10, 3))
                metadata = json.loads(str(fitted["metadata"]))
                self.assertEqual(metadata["representation"], "total_degree_chebyshev")


if __name__ == "__main__":
    unittest.main()
