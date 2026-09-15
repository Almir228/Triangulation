"""Tests for the exact-derived SO(2) action on Chebyshev coefficients."""

import json
from math import comb, pi
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from minsurf_nn.rotations import (chebyshev_to_complex,  # noqa: E402
                                  BUNDLED_CHANGE_OF_BASIS,
                                  coordinate_rotation_matrix,
                                  exact_xy_change_of_basis,
                                  load_bundled_change_of_basis,
                                  rotate_coordinate_coefficients,
                                  rotate_surface_coefficients,
                                  rotate_surface_coefficients_many,
                                  rotate_xy_points,
                                  rotate_xy_points_many,
                                  total_degree_change_of_basis)
from minsurf_nn.spectral import (evaluate_polynomial,  # noqa: E402
                                 total_degree_indices)


class ExactChangeOfBasisTests(unittest.TestCase):
    def test_exact_matrices_are_two_sided_inverses(self):
        import sympy as sp

        forward, inverse = exact_xy_change_of_basis(5)
        identity = sp.eye(comb(7, 2))
        self.assertEqual(forward * inverse, identity)
        self.assertEqual(inverse * forward, identity)

    def test_degree_ten_export_is_complex128_and_invertible(self):
        self.assertTrue(BUNDLED_CHANGE_OF_BASIS.is_file())
        forward, inverse = total_degree_change_of_basis(10)
        self.assertEqual(forward.shape, (286, 286))
        self.assertEqual(forward.dtype, np.complex128)
        np.testing.assert_allclose(forward @ inverse, np.eye(286), rtol=0.0, atol=1e-13)

    def test_bundled_degree_ten_matrices_are_read_only_and_cached(self):
        first = load_bundled_change_of_basis(10)
        second = load_bundled_change_of_basis(10)
        self.assertIs(first, second)
        self.assertFalse(first[0].flags.writeable)
        self.assertFalse(first[1].flags.writeable)


class RotationActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.degree = 6
        cls.indices = total_degree_indices(cls.degree)
        cls.rng = np.random.default_rng(912)
        cls.coefficients = cls.rng.normal(size=len(cls.indices))
        cls.points = cls.rng.uniform(-0.6, 0.6, size=(300, 3))

    def test_coordinate_rotation_matches_direct_coordinate_evaluation(self):
        angle = 0.731
        rotated = rotate_coordinate_coefficients(self.coefficients, self.degree, angle)
        expected = evaluate_polynomial(
            rotate_xy_points(self.points, angle), self.coefficients, self.indices)
        actual = evaluate_polynomial(self.points, rotated, self.indices)
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)

    def test_active_surface_rotation_uses_opposite_angle(self):
        angle = -0.417
        rotated = rotate_surface_coefficients(self.coefficients, self.degree, angle)
        actual = evaluate_polynomial(
            rotate_xy_points(self.points, angle), rotated, self.indices)
        expected = evaluate_polynomial(self.points, self.coefficients, self.indices)
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)

    def test_group_law_and_two_pi_identity(self):
        first, second = 0.37, -1.11
        sequential = rotate_coordinate_coefficients(
            rotate_coordinate_coefficients(self.coefficients, self.degree, second),
            self.degree, first)
        combined = rotate_coordinate_coefficients(
            self.coefficients, self.degree, first + second)
        np.testing.assert_allclose(sequential, combined, rtol=2e-13, atol=2e-13)
        full_turn = rotate_coordinate_coefficients(self.coefficients, self.degree, 2.0 * pi)
        np.testing.assert_allclose(full_turn, self.coefficients, rtol=0.0, atol=2e-13)

    def test_diagonal_representation_preserves_norm_and_reality(self):
        angle = 1.234
        rotated = rotate_coordinate_coefficients(self.coefficients, self.degree, angle)
        self.assertFalse(np.iscomplexobj(rotated))
        before = chebyshev_to_complex(self.coefficients, self.degree)
        after = chebyshev_to_complex(rotated, self.degree)
        self.assertAlmostEqual(float(np.linalg.norm(before)), float(np.linalg.norm(after)), places=12)

    def test_matrix_and_blockwise_implementations_agree(self):
        angle = 0.29
        matrix = coordinate_rotation_matrix(self.degree, angle)
        expected = matrix @ self.coefficients
        actual = rotate_coordinate_coefficients(self.coefficients, self.degree, angle)
        np.testing.assert_allclose(actual, expected.real, rtol=2e-13, atol=2e-13)
        self.assertLess(float(np.max(np.abs(expected.imag))), 2e-13)

    def test_batched_coefficients_are_supported(self):
        batch = np.stack((self.coefficients, 2.0 * self.coefficients))
        rotated = rotate_coordinate_coefficients(batch, self.degree, 0.2)
        self.assertEqual(rotated.shape, batch.shape)
        np.testing.assert_allclose(rotated[1], 2.0 * rotated[0], rtol=1e-14, atol=1e-14)

    def test_multi_angle_rotation_matches_scalar_path(self):
        angles = np.array([0.0, 0.17, -0.93, 2.0 * pi])
        actual = rotate_surface_coefficients_many(
            self.coefficients, self.degree, angles)
        expected = np.stack([
            rotate_surface_coefficients(self.coefficients, self.degree, angle)
            for angle in angles
        ])
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)

        rotated_points = rotate_xy_points_many(self.points, angles)
        for position, angle in enumerate(angles):
            np.testing.assert_allclose(
                rotated_points[position], rotate_xy_points(self.points, angle),
                rtol=0.0, atol=2e-15)

    def test_integer_points_are_promoted_before_rotation(self):
        points = np.array([[1, 0, 2]], dtype=np.int32)
        rotated = rotate_xy_points(points, pi / 4)
        self.assertEqual(rotated.dtype, np.float64)
        np.testing.assert_allclose(rotated[0], [2 ** -0.5, 2 ** -0.5, 2.0], atol=1e-15)


class RotationCliTests(unittest.TestCase):
    def test_cli_exports_reusable_matrices(self):
        with tempfile.TemporaryDirectory(prefix="chebyshev_rotation_test_") as directory:
            output = Path(directory) / "rotation.npz"
            completed = subprocess.run([
                sys.executable, str(ROOT / "scripts" / "build_rotation_matrices.py"),
                "--degree", "3", "--output", str(output),
            ], check=True, capture_output=True, text=True)
            self.assertTrue(output.is_file(), completed.stdout)
            with np.load(output, allow_pickle=False) as data:
                self.assertEqual(data["chebyshev_to_complex"].shape, (20, 20))
                self.assertEqual(data["complex_to_chebyshev"].dtype, np.complex128)
                metadata = json.loads(str(data["metadata"]))
                self.assertEqual(metadata["source_arithmetic"], "exact SymPy Q(i)")
                self.assertEqual(metadata["numerical_identity_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
