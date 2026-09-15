"""Total-degree Chebyshev fitting for an oriented triangulated surface.

The mesh is assumed to live in ``[-1, 1]^3`` and to be an oriented topological
disk whose vector area points along ``+z``.  The fitted polynomial approximates
the local signed distance in a narrow band around the mesh.  Its zero level must
still be clipped to the surface domain: a single implicit equation cannot encode
the boundary of an open Plateau surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Dict, Optional, Tuple

import numpy as np


def total_degree_indices(degree: int) -> np.ndarray:
    """Return a stable graded ordering of ``(i, j, k)``, ``i+j+k <= degree``."""

    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    indices = []
    for total in range(degree + 1):
        for i in range(total + 1):
            for j in range(total - i + 1):
                indices.append((i, j, total - i - j))
    result = np.asarray(indices, dtype=np.int16)
    if len(result) != comb(degree + 3, 3):
        raise AssertionError("internal total-degree indexing error")
    return result


def chebyshev_table(values: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate ``T_0, ..., T_degree`` using the three-term recurrence."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite one-dimensional array")
    table = np.empty((len(values), degree + 1), dtype=np.float64)
    table[:, 0] = 1.0
    if degree >= 1:
        table[:, 1] = values
    for order in range(2, degree + 1):
        table[:, order] = 2.0 * values * table[:, order - 1] - table[:, order - 2]
    return table


def chebyshev_table_with_derivative(values: np.ndarray, degree: int
                                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate Chebyshev polynomials and their derivatives."""

    table = chebyshev_table(values, degree)
    derivative = np.zeros_like(table)
    if degree >= 1:
        derivative[:, 1] = 1.0
    for order in range(2, degree + 1):
        derivative[:, order] = (2.0 * table[:, order - 1] +
                                2.0 * values * derivative[:, order - 1] -
                                derivative[:, order - 2])
    return table, derivative


def design_matrix(points: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Build the total-degree tensor-product Chebyshev design matrix."""

    points = np.asarray(points, dtype=np.float64)
    indices = np.asarray(indices)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points must have finite shape [N, 3]")
    if indices.ndim != 2 or indices.shape[1] != 3 or len(indices) == 0:
        raise ValueError("indices must have shape [K, 3]")
    if np.any(indices < 0):
        raise ValueError("Chebyshev indices must be nonnegative")
    degree = int(indices.max())
    tx = chebyshev_table(points[:, 0], degree)
    ty = chebyshev_table(points[:, 1], degree)
    tz = chebyshev_table(points[:, 2], degree)
    return tx[:, indices[:, 0]] * ty[:, indices[:, 1]] * tz[:, indices[:, 2]]


def evaluate_polynomial(points: np.ndarray, coefficients: np.ndarray,
                        indices: np.ndarray) -> np.ndarray:
    """Evaluate a coefficient vector in the ordering from ``total_degree_indices``."""

    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 1 or len(coefficients) != len(indices):
        raise ValueError("coefficients must have shape [K] matching indices")
    return design_matrix(points, indices) @ coefficients


def evaluate_gradient(points: np.ndarray, coefficients: np.ndarray,
                      indices: np.ndarray) -> np.ndarray:
    """Evaluate the Cartesian gradient of a total-degree Chebyshev polynomial."""

    points = np.asarray(points, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    indices = np.asarray(indices)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if coefficients.ndim != 1 or len(coefficients) != len(indices):
        raise ValueError("coefficients must have shape [K] matching indices")
    degree = int(indices.max())
    tx, dx = chebyshev_table_with_derivative(points[:, 0], degree)
    ty, dy = chebyshev_table_with_derivative(points[:, 1], degree)
    tz, dz = chebyshev_table_with_derivative(points[:, 2], degree)
    result = np.empty_like(points)
    result[:, 0] = ((dx[:, indices[:, 0]] * ty[:, indices[:, 1]] *
                     tz[:, indices[:, 2]]) @ coefficients)
    result[:, 1] = ((tx[:, indices[:, 0]] * dy[:, indices[:, 1]] *
                     tz[:, indices[:, 2]]) @ coefficients)
    result[:, 2] = ((tx[:, indices[:, 0]] * ty[:, indices[:, 1]] *
                     dz[:, indices[:, 2]]) @ coefficients)
    return result


def coefficient_energy(coefficients: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Return ``sqrt(sum(c_ijk^2))`` for each total degree."""

    coefficients = np.asarray(coefficients, dtype=np.float64)
    indices = np.asarray(indices)
    if coefficients.ndim != 1 or len(coefficients) != len(indices):
        raise ValueError("coefficients must have shape [K] matching indices")
    total = indices.sum(axis=1)
    return np.sqrt(np.bincount(total, weights=coefficients ** 2,
                               minlength=int(total.max()) + 1))


def boundary_edges(faces: np.ndarray) -> np.ndarray:
    """Return the unoriented mesh edges incident to exactly one triangle."""

    faces = np.asarray(faces, dtype=np.int64)
    edges: Dict[Tuple[int, int], list[int]] = {}
    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (int(min(a, b)), int(max(a, b)))
            edges.setdefault(key, []).append(1 if a < b else -1)
    if any(len(uses) > 2 for uses in edges.values()):
        raise ValueError("mesh is non-manifold: an edge belongs to more than two faces")
    if any(len(uses) == 2 and uses[0] == uses[1] for uses in edges.values()):
        raise ValueError("adjacent faces have inconsistent orientation")
    result = np.asarray([edge for edge, uses in edges.items() if len(uses) == 1],
                        dtype=np.int32)
    if result.size == 0:
        raise ValueError("mesh has no boundary; an open triangulated disk is required")
    return result.reshape(-1, 2)


def validate_mesh(vertices: np.ndarray, faces: np.ndarray, cube_tolerance: float = 1e-10,
                  orientation_tolerance: float = 1e-7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the cube and ``+z`` vector-area assumptions and return face geometry."""

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("vertices must have shape [V, 3], V >= 3")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("vertices contain non-finite coordinates")
    if np.max(np.abs(vertices)) > 1.0 + cube_tolerance:
        raise ValueError("mesh vertices must lie inside [-1, 1]^3")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("faces must have shape [F, 3], F >= 1")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("face index is out of range")
    if np.any(faces[:, 0] == faces[:, 1]) or np.any(faces[:, 1] == faces[:, 2]) or \
            np.any(faces[:, 2] == faces[:, 0]):
        raise ValueError("face contains a repeated vertex")
    boundary_edges(faces)

    triangles = vertices[faces]
    area_vectors = np.cross(triangles[:, 1] - triangles[:, 0],
                            triangles[:, 2] - triangles[:, 0])
    twice_areas = np.linalg.norm(area_vectors, axis=1)
    if np.any(twice_areas <= 1e-14) or not np.all(np.isfinite(twice_areas)):
        raise ValueError("mesh contains a degenerate triangle")
    vector_area = 0.5 * area_vectors.sum(axis=0)
    if vector_area[2] <= 0:
        raise ValueError("oriented mesh area must point toward +z")
    transverse = float(np.linalg.norm(vector_area[:2]))
    if transverse > orientation_tolerance * float(vector_area[2]):
        raise ValueError("oriented mesh area is not parallel to the z axis")
    normals = area_vectors / twice_areas[:, None]
    return triangles, normals, twice_areas * 0.5


def _surface_samples(triangles: np.ndarray, normals: np.ndarray, areas: np.ndarray,
                     count: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    face_ids = rng.choice(len(triangles), size=count, p=areas / areas.sum())
    selected = triangles[face_ids]
    root = np.sqrt(rng.random(count))
    other = rng.random(count)
    barycentric = np.column_stack((1.0 - root, root * (1.0 - other), root * other))
    points = np.sum(selected * barycentric[:, :, None], axis=1)
    return points, normals[face_ids]


def _boundary_samples(vertices: np.ndarray, edges: np.ndarray, count: int,
                      rng: np.random.Generator) -> np.ndarray:
    segments = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    lengths = np.linalg.norm(segments, axis=1)
    edge_ids = rng.choice(len(edges), size=count, p=lengths / lengths.sum())
    fraction = rng.random(count)
    return vertices[edges[edge_ids, 0]] + fraction[:, None] * segments[edge_ids]


def _safe_offset_distances(points: np.ndarray, normals: np.ndarray, maximum: float,
                           rng: np.random.Generator) -> np.ndarray:
    """Choose symmetric offsets that stay strictly inside the Chebyshev cube."""

    room = 1.0 - np.abs(points)
    ratio = np.full_like(room, np.inf)
    active = np.abs(normals) > 1e-14
    ratio[active] = room[active] / np.abs(normals[active])
    cube_limit = np.min(ratio, axis=1)
    requested = maximum * rng.uniform(0.35, 1.0, size=len(points))
    return np.minimum(requested, 0.9 * np.maximum(cube_limit, 0.0))


@dataclass(frozen=True)
class SpectralFit:
    degree: int
    indices: np.ndarray
    coefficients: np.ndarray
    diagnostics: Dict[str, float]

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        return evaluate_polynomial(points, self.coefficients, self.indices)

    def gradient(self, points: np.ndarray) -> np.ndarray:
        return evaluate_gradient(points, self.coefficients, self.indices)

    def energy_by_degree(self) -> np.ndarray:
        return coefficient_energy(self.coefficients, self.indices)


def extract_graph_zero_mesh(reference_vertices: np.ndarray, faces: np.ndarray,
                            fit: SpectralFit, max_iterations: int = 30,
                            tolerance: float = 1e-9) -> Tuple[np.ndarray, Dict[str, float]]:
    """Extract the nearby zero sheet as ``z(x,y)`` on a reference mesh topology.

    This is an evaluation helper for the current graph-solver regime.  Newton's
    method starts from each ground-truth vertex only to select the corresponding
    local zero-sheet branch; the resulting z coordinate satisfies the polynomial.
    """

    reference_vertices = np.asarray(reference_vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("max_iterations and tolerance must be positive")
    validate_mesh(reference_vertices, faces)
    predicted = reference_vertices.copy()
    initial_z = predicted[:, 2].copy()
    for _ in range(max_iterations):
        values = fit.evaluate(predicted)
        if float(np.max(np.abs(values))) <= tolerance:
            break
        derivative = fit.gradient(predicted)[:, 2]
        active = np.abs(derivative) > 1e-10
        step = np.zeros_like(values)
        step[active] = np.clip(values[active] / derivative[active], -0.1, 0.1)
        predicted[:, 2] = np.clip(predicted[:, 2] - step, -1.0, 1.0)

    residual = np.abs(fit.evaluate(predicted))
    failed = residual > max(tolerance * 10.0, 1e-7)
    # A sampled bisection fallback handles a small derivative or an unlucky
    # Newton step while still choosing the root nearest the reference sheet.
    axis = np.linspace(-1.0, 1.0, 257)
    for vertex_id in np.flatnonzero(failed):
        probes = np.repeat(predicted[vertex_id:vertex_id + 1], len(axis), axis=0)
        probes[:, 2] = axis
        values = fit.evaluate(probes)
        crossings = np.flatnonzero(values[:-1] * values[1:] <= 0)
        if len(crossings) == 0:
            best = int(np.argmin(np.abs(values)))
            predicted[vertex_id, 2] = axis[best]
            continue
        midpoints = 0.5 * (axis[crossings] + axis[crossings + 1])
        crossing = int(crossings[np.argmin(np.abs(midpoints - initial_z[vertex_id]))])
        lower, upper = float(axis[crossing]), float(axis[crossing + 1])
        lower_value = float(values[crossing])
        for _ in range(50):
            middle = 0.5 * (lower + upper)
            probe = predicted[vertex_id:vertex_id + 1].copy()
            probe[0, 2] = middle
            middle_value = float(fit.evaluate(probe)[0])
            if lower_value * middle_value <= 0:
                upper = middle
            else:
                lower, lower_value = middle, middle_value
        predicted[vertex_id, 2] = 0.5 * (lower + upper)

    residual = np.abs(fit.evaluate(predicted))
    diagnostics = {
        "root_residual_mean": float(residual.mean()),
        "root_residual_max": float(residual.max()),
        "root_failures": int(np.count_nonzero(residual > max(tolerance * 10.0, 1e-7))),
        "vertical_displacement_rmse": float(np.sqrt(np.mean(
            (predicted[:, 2] - initial_z) ** 2))),
        "vertical_displacement_max": float(np.max(np.abs(predicted[:, 2] - initial_z))),
    }
    return predicted, diagnostics


def fit_triangulation(vertices: np.ndarray, faces: np.ndarray, degree: int = 10,
                      surface_samples: int = 4096, boundary_samples: int = 1024,
                      offset_distance: float = 0.03, boundary_weight: float = 10.0,
                      ridge: float = 1e-6, rcond: Optional[float] = None,
                      seed: int = 0) -> SpectralFit:
    """Fit a local signed-distance-like Chebyshev polynomial to a triangle mesh.

    The objective contains zero targets on the surface and boundary, plus targets
    ``+d`` and ``-d`` at symmetric normal offsets.  ``boundary_weight`` is the
    multiplier in the squared objective, not its square root.
    """

    if degree < 1:
        raise ValueError("degree must be positive")
    if surface_samples < 1 or boundary_samples < 1:
        raise ValueError("sample counts must be positive")
    if offset_distance <= 0 or offset_distance >= 1:
        raise ValueError("offset_distance must lie between zero and one")
    if boundary_weight <= 0 or ridge < 0:
        raise ValueError("boundary_weight must be positive and ridge nonnegative")

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    triangles, normals, areas = validate_mesh(vertices, faces)
    edges = boundary_edges(faces)
    rng = np.random.default_rng(seed)
    surface, sample_normals = _surface_samples(
        triangles, normals, areas, surface_samples, rng)
    distances = _safe_offset_distances(surface, sample_normals, offset_distance, rng)
    usable = distances > max(1e-8, offset_distance * 1e-4)
    if np.count_nonzero(usable) < max(10, surface_samples // 4):
        raise ValueError("too few surface samples admit symmetric offsets inside the cube")
    surface = surface[usable]
    sample_normals = sample_normals[usable]
    distances = distances[usable]
    positive = surface + distances[:, None] * sample_normals
    negative = surface - distances[:, None] * sample_normals
    boundary = _boundary_samples(vertices, edges, boundary_samples, rng)

    points = np.concatenate((surface, positive, negative, boundary), axis=0)
    targets = np.concatenate((np.zeros(len(surface)), distances, -distances,
                              np.zeros(len(boundary))))
    weights = np.concatenate((np.ones(3 * len(surface)),
                              np.full(len(boundary), boundary_weight)))
    indices = total_degree_indices(degree)
    matrix = design_matrix(points, indices)
    weighted_matrix = matrix * np.sqrt(weights)[:, None]
    weighted_targets = targets * np.sqrt(weights)

    if ridge > 0:
        order = indices.sum(axis=1).astype(np.float64)
        penalty = np.sqrt(ridge) * (1.0 + order)
        weighted_matrix = np.vstack((weighted_matrix, np.diag(penalty)))
        weighted_targets = np.concatenate((weighted_targets, np.zeros(len(indices))))
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        weighted_matrix, weighted_targets, rcond=rcond)
    if not np.all(np.isfinite(coefficients)):
        raise RuntimeError("least-squares fitting produced non-finite coefficients")

    surface_residual = evaluate_polynomial(surface, coefficients, indices)
    positive_residual = evaluate_polynomial(positive, coefficients, indices) - distances
    negative_residual = evaluate_polynomial(negative, coefficients, indices) + distances
    boundary_residual = evaluate_polynomial(boundary, coefficients, indices)
    condition = (float(singular_values[0] / singular_values[-1])
                 if len(singular_values) and singular_values[-1] > 0 else float("inf"))
    diagnostics = {
        "coefficients": int(len(indices)),
        "equations": int(len(points)),
        "rank": int(rank),
        "condition_number": condition,
        "surface_rmse": float(np.sqrt(np.mean(surface_residual ** 2))),
        "offset_rmse": float(np.sqrt(np.mean(np.concatenate(
            (positive_residual, negative_residual)) ** 2))),
        "boundary_rmse": float(np.sqrt(np.mean(boundary_residual ** 2))),
        "maximum_abs_coefficient": float(np.max(np.abs(coefficients))),
        "minimum_offset": float(np.min(distances)),
        "maximum_offset": float(np.max(distances)),
    }
    return SpectralFit(degree, indices, coefficients, diagnostics)
