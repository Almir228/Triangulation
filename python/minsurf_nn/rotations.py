"""Exact SO(2) action on total-degree Chebyshev coefficients.

For every fixed z degree ``k``, the two-dimensional Chebyshev block

    T_i(x) T_j(y),  i + j <= p - k

is converted exactly to the complex monomial basis ``w^m wbar^n`` with
``w=x+iy``.  In that basis an XY coordinate rotation is diagonal.  The module
uses the convention from the project specification:

    d = S c
    d' = diag(exp(i (m-n) phi)) d
    c' = S^{-1} d'

Thus ``F_{c'}(x,y,z) = F_c(R_phi(x,y), z)``.  An active +phi rotation of the
zero set uses the opposite angle and is exposed as ``rotate_surface_coefficients``.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb
import json
from pathlib import Path
from typing import Tuple

import numpy as np

from .spectral import total_degree_indices


BUNDLED_DEGREE = 10
BUNDLED_CHANGE_OF_BASIS = (
    Path(__file__).resolve().parent / "assets" / "chebyshev_so2_degree10.npz")


def xy_total_degree_indices(degree: int) -> np.ndarray:
    """Return ``(i,j)``, ordered by total degree and then increasing ``i``."""

    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    result = [(i, total - i) for total in range(degree + 1)
              for i in range(total + 1)]
    if len(result) != comb(degree + 2, 2):
        raise AssertionError("internal XY total-degree indexing error")
    return np.asarray(result, dtype=np.int16)


@lru_cache(maxsize=None)
def _power_to_chebyshev(power: int):
    """Exact coefficients of x**power in the first-kind Chebyshev basis."""

    import sympy as sp

    if power < 0:
        raise ValueError("power must be nonnegative")
    if power == 0:
        return ((0, sp.Integer(1)),)
    terms = [
        (power - 2 * j, sp.Rational(comb(power, j), 2 ** (power - 1)))
        for j in range((power - 1) // 2 + 1)
    ]
    if power % 2 == 0:
        terms.append((0, sp.Rational(comb(power, power // 2), 2 ** power)))
    return tuple(terms)


@lru_cache(maxsize=None)
def exact_xy_change_of_basis(degree: int):
    """Return exact ``(S, S_inverse)`` over Q(i) for one XY degree block.

    Columns of ``S`` are expansions of ``T_i(x)T_j(y)`` in the ordered
    ``w^m wbar^n`` basis.  ``S_inverse`` is constructed independently by
    expanding the complex monomials back into Chebyshev polynomials.  Both
    products are checked against the exact identity before the matrices are
    returned as SymPy ``ImmutableMatrix`` objects.
    """

    import sympy as sp

    indices = xy_total_degree_indices(degree)
    tuples = [tuple(map(int, row)) for row in indices]
    positions = {pair: position for position, pair in enumerate(tuples)}
    size = len(tuples)
    w, wbar = sp.symbols("w wbar")
    x, y = sp.symbols("x y")
    x_from_w = (w + wbar) / 2
    y_from_w = (w - wbar) / (2 * sp.I)

    forward = sp.zeros(size, size)
    for column, (i, j) in enumerate(tuples):
        expression = sp.expand(
            sp.chebyshevt(i, x_from_w) * sp.chebyshevt(j, y_from_w))
        polynomial = sp.Poly(expression, w, wbar, extension=sp.I)
        for (m, n), coefficient in polynomial.terms():
            forward[positions[(m, n)], column] = coefficient

    inverse = sp.zeros(size, size)
    for column, (m, n) in enumerate(tuples):
        expression = sp.expand((x + sp.I * y) ** m * (x - sp.I * y) ** n)
        polynomial = sp.Poly(expression, x, y, extension=sp.I)
        for (x_power, y_power), coefficient in polynomial.terms():
            for i, x_coefficient in _power_to_chebyshev(x_power):
                for j, y_coefficient in _power_to_chebyshev(y_power):
                    inverse[positions[(i, j)], column] += (
                        coefficient * x_coefficient * y_coefficient)

    identity = sp.eye(size)
    if forward * inverse != identity or inverse * forward != identity:
        raise AssertionError("exact Chebyshev/complex change of basis is not invertible")
    return sp.ImmutableMatrix(forward), sp.ImmutableMatrix(inverse)


def _as_complex128(matrix) -> np.ndarray:
    return np.asarray(matrix.tolist(), dtype=np.complex128)


@lru_cache(maxsize=None)
def numerical_xy_change_of_basis(degree: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return the exact matrices converted once to read-only ``complex128``."""

    forward_exact, inverse_exact = exact_xy_change_of_basis(degree)
    forward = _as_complex128(forward_exact)
    inverse = _as_complex128(inverse_exact)
    forward.setflags(write=False)
    inverse.setflags(write=False)
    return forward, inverse


@lru_cache(maxsize=None)
def _nested_numerical_xy_blocks(maximum_degree: int):
    """Build the maximum exact block once and extract all nested subspaces."""

    master_forward, master_inverse = numerical_xy_change_of_basis(maximum_degree)
    blocks = []
    for degree in range(maximum_degree + 1):
        size = comb(degree + 2, 2)
        forward = master_forward[:size, :size].copy()
        inverse = master_inverse[:size, :size].copy()
        if not np.array_equal(forward @ inverse, np.eye(size, dtype=np.complex128)):
            # Products of dyadic Q(i) entries are exact for the degrees used in
            # this project; keep an allclose fallback for larger future degrees.
            if not np.allclose(forward @ inverse, np.eye(size), rtol=0.0, atol=1e-13):
                raise RuntimeError("nested numerical change-of-basis block is inconsistent")
        forward.setflags(write=False)
        inverse.setflags(write=False)
        blocks.append((forward, inverse))
    return tuple(blocks)


def _validate_coefficients(coefficients: np.ndarray, degree: int) -> np.ndarray:
    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    values = np.asarray(coefficients)
    count = comb(int(degree) + 3, 3)
    if values.ndim < 1 or values.shape[-1] != count:
        raise ValueError(f"coefficients must have final dimension {count} for degree {degree}")
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise ValueError("coefficients must be finite numeric values")
    return values


def _z_block_positions(degree: int, z_degree: int) -> np.ndarray:
    indices = total_degree_indices(degree)
    return np.flatnonzero(indices[:, 2] == z_degree)


@lru_cache(maxsize=None)
def load_bundled_change_of_basis(degree: int = BUNDLED_DEGREE
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Load and validate the persistent degree-10 matrices shipped with the package."""

    if degree != BUNDLED_DEGREE:
        raise ValueError(f"only the bundled degree {BUNDLED_DEGREE} is available")
    if not BUNDLED_CHANGE_OF_BASIS.is_file():
        raise FileNotFoundError(f"bundled rotation matrices are missing: {BUNDLED_CHANGE_OF_BASIS}")
    with np.load(BUNDLED_CHANGE_OF_BASIS, allow_pickle=False) as data:
        forward = np.asarray(data["chebyshev_to_complex"], dtype=np.complex128)
        inverse = np.asarray(data["complex_to_chebyshev"], dtype=np.complex128)
        indices = np.asarray(data["indices"], dtype=np.int16)
        metadata = json.loads(str(data["metadata"]))
    count = comb(degree + 3, 3)
    if forward.shape != (count, count) or inverse.shape != (count, count):
        raise ValueError("bundled rotation matrices have invalid shapes")
    if not np.array_equal(indices, total_degree_indices(degree)):
        raise ValueError("bundled rotation matrices use an incompatible coefficient ordering")
    if (metadata.get("degree") != degree or
            metadata.get("source_arithmetic") != "exact SymPy Q(i)"):
        raise ValueError("bundled rotation matrix metadata is incompatible")
    if not np.all(np.isfinite(forward)) or not np.all(np.isfinite(inverse)):
        raise ValueError("bundled rotation matrices contain non-finite values")
    identity = np.eye(count, dtype=np.complex128)
    if not np.allclose(forward @ inverse, identity, rtol=0.0, atol=1e-13):
        raise ValueError("bundled rotation matrices failed the inverse check")
    forward.setflags(write=False)
    inverse.setflags(write=False)
    return forward, inverse


def chebyshev_to_complex(coefficients: np.ndarray, degree: int) -> np.ndarray:
    """Apply the block diagonal ``S_p`` to one vector or a batch of vectors."""

    values = _validate_coefficients(coefficients, degree).astype(np.complex128, copy=False)
    if degree == BUNDLED_DEGREE and BUNDLED_CHANGE_OF_BASIS.is_file():
        forward, _ = load_bundled_change_of_basis(degree)
        return values @ forward.T
    result = np.empty(values.shape, dtype=np.complex128)
    blocks = _nested_numerical_xy_blocks(degree)
    for z_degree in range(degree + 1):
        block_degree = degree - z_degree
        positions = _z_block_positions(degree, z_degree)
        forward, _ = blocks[block_degree]
        result[..., positions] = values[..., positions] @ forward.T
    return result


def complex_to_chebyshev(coefficients: np.ndarray, degree: int) -> np.ndarray:
    """Apply the exact block inverse ``S_p^{-1}`` in ``complex128`` arithmetic."""

    values = _validate_coefficients(coefficients, degree).astype(np.complex128, copy=False)
    if degree == BUNDLED_DEGREE and BUNDLED_CHANGE_OF_BASIS.is_file():
        _, inverse = load_bundled_change_of_basis(degree)
        return values @ inverse.T
    result = np.empty(values.shape, dtype=np.complex128)
    blocks = _nested_numerical_xy_blocks(degree)
    for z_degree in range(degree + 1):
        block_degree = degree - z_degree
        positions = _z_block_positions(degree, z_degree)
        _, inverse = blocks[block_degree]
        result[..., positions] = values[..., positions] @ inverse.T
    return result


def _assemble_total_degree_change_of_basis(degree: int, blocks
                                           ) -> Tuple[np.ndarray, np.ndarray]:
    count = comb(degree + 3, 3)
    forward = np.zeros((count, count), dtype=np.complex128)
    inverse = np.zeros((count, count), dtype=np.complex128)
    for z_degree in range(degree + 1):
        positions = _z_block_positions(degree, z_degree)
        block_forward, block_inverse = blocks[degree - z_degree]
        forward[np.ix_(positions, positions)] = block_forward
        inverse[np.ix_(positions, positions)] = block_inverse
    return forward, inverse


def derive_total_degree_change_of_basis(degree: int) -> Tuple[np.ndarray, np.ndarray]:
    """Derive numerical full matrices from a freshly checked exact SymPy block."""

    return _assemble_total_degree_change_of_basis(
        degree, _nested_numerical_xy_blocks(degree))


def total_degree_change_of_basis(degree: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load bundled p=10 matrices, or derive another degree in exact arithmetic."""

    if degree == BUNDLED_DEGREE and BUNDLED_CHANGE_OF_BASIS.is_file():
        return load_bundled_change_of_basis(degree)
    return derive_total_degree_change_of_basis(degree)


def coordinate_rotation_matrix(degree: int, angle: float) -> np.ndarray:
    """Return ``S_p^{-1} D_p(angle) S_p`` for a coordinate-system rotation."""

    if not np.isfinite(angle):
        raise ValueError("angle must be finite")
    indices = total_degree_indices(degree)
    phases = np.exp(1j * (indices[:, 0] - indices[:, 1]) * float(angle))
    forward, inverse = total_degree_change_of_basis(degree)
    return inverse @ (phases[:, None] * forward)


def _real_output_if_real_input(result: np.ndarray, original: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(original):
        return result
    scale = max(1.0, float(np.max(np.abs(result.real))))
    if float(np.max(np.abs(result.imag))) > 5e-11 * scale:
        raise RuntimeError("rotation produced an unexpectedly large imaginary residual")
    return result.real


def rotate_coordinate_coefficients(coefficients: np.ndarray, degree: int,
                                   angle: float) -> np.ndarray:
    """Transform coefficients under an XY coordinate rotation by ``angle``.

    The returned coefficients obey
    ``F_rotated(point) = F_original(R_angle(point))``.
    """

    original = _validate_coefficients(coefficients, degree)
    if not np.isfinite(angle):
        raise ValueError("angle must be finite")
    indices = total_degree_indices(degree)
    complex_coefficients = chebyshev_to_complex(original, degree)
    phases = np.exp(1j * (indices[:, 0] - indices[:, 1]) * float(angle))
    result = complex_to_chebyshev(complex_coefficients * phases, degree)
    return _real_output_if_real_input(result, original)


def rotate_surface_coefficients(coefficients: np.ndarray, degree: int,
                                angle: float) -> np.ndarray:
    """Actively rotate the represented zero level set by ``+angle`` around Z."""

    return rotate_coordinate_coefficients(coefficients, degree, -float(angle))


def rotate_surface_coefficients_many(coefficients: np.ndarray, degree: int,
                                     angles: np.ndarray) -> np.ndarray:
    """Actively rotate one real coefficient vector through several angles.

    This is the dataset-augmentation path.  It converts each fixed-z block to
    the complex basis only once and applies every diagonal phase in a batch.
    For degree 10 the blocks are sliced from the bundled ``S`` and ``S^-1``
    matrices, so SymPy is never invoked during dataset generation.
    """

    original = _validate_coefficients(coefficients, degree)
    if original.ndim != 1:
        raise ValueError("coefficients must be one-dimensional for multi-angle rotation")
    values = np.asarray(angles, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("angles must be a finite one-dimensional array")
    result = np.empty((len(values), len(original)), dtype=np.complex128)
    full_forward = full_inverse = None
    blocks = None
    if degree == BUNDLED_DEGREE and BUNDLED_CHANGE_OF_BASIS.is_file():
        full_forward, full_inverse = load_bundled_change_of_basis(degree)
    else:
        blocks = _nested_numerical_xy_blocks(degree)

    indices = total_degree_indices(degree)
    for z_degree in range(degree + 1):
        positions = _z_block_positions(degree, z_degree)
        if full_forward is not None and full_inverse is not None:
            forward = full_forward[np.ix_(positions, positions)]
            inverse = full_inverse[np.ix_(positions, positions)]
        else:
            assert blocks is not None
            forward, inverse = blocks[degree - z_degree]
        complex_block = forward @ original[positions]
        frequencies = indices[positions, 0] - indices[positions, 1]
        # An active surface rotation by +phi is the coordinate action by -phi.
        phases = np.exp(-1j * values[:, None] * frequencies[None, :])
        result[:, positions] = (phases * complex_block[None, :]) @ inverse.T
    return _real_output_if_real_input(result, original)


def rotate_xy_points(points: np.ndarray, angle: float) -> np.ndarray:
    """Actively rotate one ``[...,3]`` point array by ``+angle`` around Z."""

    values = np.asarray(points)
    if (values.ndim < 1 or values.shape[-1] != 3 or np.iscomplexobj(values) or
            not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values))):
        raise ValueError("points must have finite final dimension 3")
    if not np.isfinite(angle):
        raise ValueError("angle must be finite")
    values = values.astype(np.float64, copy=False)
    cosine, sine = np.cos(angle), np.sin(angle)
    result = values.copy()
    result[..., 0] = cosine * values[..., 0] - sine * values[..., 1]
    result[..., 1] = sine * values[..., 0] + cosine * values[..., 1]
    return result


def rotate_xy_points_many(points: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Actively rotate one ``[N,3]`` array for every angle, returning ``[A,N,3]``."""

    values = np.asarray(points)
    rotations = np.asarray(angles, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != 3 or np.iscomplexobj(values) or
            not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values))):
        raise ValueError("points must have finite shape [N,3]")
    if rotations.ndim != 1 or not np.all(np.isfinite(rotations)):
        raise ValueError("angles must be a finite one-dimensional array")
    values = values.astype(np.float64, copy=False)
    cosine = np.cos(rotations)[:, None]
    sine = np.sin(rotations)[:, None]
    result = np.broadcast_to(values, (len(rotations),) + values.shape).copy()
    result[:, :, 0] = cosine * values[None, :, 0] - sine * values[None, :, 1]
    result[:, :, 1] = sine * values[None, :, 0] + cosine * values[None, :, 1]
    return result
