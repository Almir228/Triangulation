"""Small, dependency-light geometry helpers used by the dataset generator.

All arrays in this module use ordinary Cartesian coordinates.  The caller is
responsible for applying a common normalization to the boundary, mesh and
queries before writing a sample.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


def oriented_area_vector(points: np.ndarray) -> np.ndarray:
    """Return ``0.5 * sum((p_i-c) x (p_{i+1}-c))`` for a closed polygon."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("points must have shape [N, 3], N >= 3")
    if not np.all(np.isfinite(points)):
        raise ValueError("points contain non-finite coordinates")
    if np.allclose(points[0], points[-1], rtol=0.0, atol=1e-14):
        points = points[:-1]
    centered = points - points.mean(axis=0)
    return 0.5 * np.cross(centered, np.roll(centered, -1, axis=0)).sum(axis=0)


def rotation_to_positive_z(vector: np.ndarray) -> np.ndarray:
    """Construct a proper rotation mapping a nonzero vector to ``+e_z``."""

    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("vector must be a finite 3-vector")
    length = float(np.linalg.norm(vector))
    if length <= 1e-14:
        raise ValueError("oriented area vector is degenerate")
    source = vector / length
    target = np.array([0.0, 0.0, 1.0])
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if cosine >= 1.0 - 1e-14:
        return np.eye(3)
    if cosine <= -1.0 + 1e-14:
        # A half turn around x maps -z to +z and has determinant +1.
        return np.diag([1.0, -1.0, -1.0])
    axis_sine = np.cross(source, target)
    skew = np.array([
        [0.0, -axis_sine[2], axis_sine[1]],
        [axis_sine[2], 0.0, -axis_sine[0]],
        [-axis_sine[1], axis_sine[0], 0.0],
    ])
    return np.eye(3) + skew + skew @ skew / (1.0 + cosine)


def canonicalize_contour(points: np.ndarray, margin: float = 0.1
                         ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Center, rotate vector area to ``+z`` and scale a contour into the cube."""

    points = np.asarray(points, dtype=np.float64)
    if not 0.0 <= margin < 1.0:
        raise ValueError("margin must lie in [0, 1)")
    center = points.mean(axis=0)
    area_before = oriented_area_vector(points)
    rotation = rotation_to_positive_z(area_before)
    rotated = (points - center) @ rotation.T
    radius = float(np.max(np.linalg.norm(rotated, axis=1)))
    if not np.isfinite(radius) or radius <= 1e-14:
        raise ValueError("contour has invalid scale")
    scale = radius / (1.0 - margin)
    canonical = rotated / scale
    area_after = oriented_area_vector(canonical)
    tolerance = 1e-10 * max(1.0, abs(float(area_after[2])))
    if area_after[2] <= 0 or np.linalg.norm(area_after[:2]) > tolerance:
        raise RuntimeError("failed to align contour vector area with +z")
    transform = {
        "center": center,
        "rotation": rotation,
        "scale": np.asarray(scale),
        "area_before": area_before,
        "area_after": area_after,
        "margin": np.asarray(margin),
    }
    return canonical.astype(np.float64), transform


def is_strictly_convex_xy(points: np.ndarray, tolerance: float = 1e-10) -> bool:
    """Whether an ordered polygon is counter-clockwise and strictly convex in XY."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        return False
    xy = points[:, :2]
    first = np.roll(xy, -1, axis=0) - xy
    second = np.roll(xy, -2, axis=0) - np.roll(xy, -1, axis=0)
    turns = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    return bool(np.all(turns > tolerance))


def resample_closed_curve(points: np.ndarray, count: int) -> np.ndarray:
    """Resample a closed polyline at equal arc-length intervals.

    The last input point is treated as a duplicate of the first when present;
    the returned curve never repeats its first point.
    """

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("points must have shape [N, 3], N >= 3")
    if count < 3:
        raise ValueError("count must be at least 3")
    if np.allclose(points[0], points[-1], rtol=0.0, atol=1e-14):
        points = points[:-1]
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 1e-14):
        raise ValueError("closed curve has a zero or non-finite edge")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = np.arange(count, dtype=np.float64) * cumulative[-1] / count
    indices = np.searchsorted(cumulative, targets, side="right") - 1
    indices %= len(points)
    local = (targets - cumulative[indices]) / lengths[indices]
    result = points[indices] + local[:, None] * edges[indices]
    return result.astype(np.float32)


def read_obj_triangles(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read the vertex and triangular-face subset emitted by the C++ solver."""

    vertices = []
    faces = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if fields[0] == "v":
            if len(fields) != 4:
                raise ValueError(f"OBJ line {line_number}: expected v x y z")
            vertices.append([float(value) for value in fields[1:]])
        elif fields[0] == "f":
            if len(fields) != 4:
                raise ValueError(f"OBJ line {line_number}: expected a triangle")
            face = []
            for token in fields[1:]:
                value = token.split("/", 1)[0]
                index = int(value)
                face.append(index - 1 if index > 0 else len(vertices) + index)
            faces.append(face)
    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int32)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
        raise ValueError("OBJ contains no 3D vertices")
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise ValueError("OBJ contains no triangular faces")
    if np.any(face_array < 0) or np.any(face_array >= len(vertex_array)):
        raise ValueError("OBJ face index is out of range")
    return vertex_array, face_array


def _closest_points_on_triangles(points: np.ndarray, vertices: np.ndarray,
                                 faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return closest points, triangle indices and oriented normals.

    This is Ericson's point-triangle region test, vectorized over a point
    chunk and all triangles.  Chunking in ``signed_distance`` keeps memory
    bounded for meshes with thousands of faces.
    """

    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    ab = b - a
    ac = c - a
    bc = c - b
    normals = np.cross(ab, ac)
    normal_lengths = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(normal_lengths[:, None], 1e-30)

    p = points[:, None, :]
    ap = p - a[None, :, :]
    bp = p - b[None, :, :]
    cp = p - c[None, :, :]
    d1 = np.sum(ab[None, :, :] * ap, axis=2)
    d2 = np.sum(ac[None, :, :] * ap, axis=2)
    d3 = np.sum(ab[None, :, :] * bp, axis=2)
    d4 = np.sum(ac[None, :, :] * bp, axis=2)
    d5 = np.sum(ab[None, :, :] * cp, axis=2)
    d6 = np.sum(ac[None, :, :] * cp, axis=2)

    q = np.empty((len(points), len(faces), 3), dtype=np.float64)
    assigned = np.zeros((len(points), len(faces)), dtype=bool)

    mask = (d1 <= 0) & (d2 <= 0)
    q[mask] = np.broadcast_to(a, q.shape)[mask]
    assigned |= mask
    mask = (~assigned) & (d3 >= 0) & (d4 <= d3)
    q[mask] = np.broadcast_to(b, q.shape)[mask]
    assigned |= mask
    mask = (~assigned) & (d6 >= 0) & (d5 <= d6)
    q[mask] = np.broadcast_to(c, q.shape)[mask]
    assigned |= mask

    vc = d1 * d4 - d3 * d2
    mask = (~assigned) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    denominator = np.where(np.abs(d1 - d3) > 1e-30, d1 - d3, 1.0)
    v = d1 / denominator
    q[mask] = (a[None, :, :] + v[:, :, None] * ab[None, :, :])[mask]
    assigned |= mask

    vb = d5 * d2 - d1 * d6
    mask = (~assigned) & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    denominator = np.where(np.abs(d2 - d6) > 1e-30, d2 - d6, 1.0)
    w = d2 / denominator
    q[mask] = (a[None, :, :] + w[:, :, None] * ac[None, :, :])[mask]
    assigned |= mask

    va = d3 * d6 - d5 * d4
    mask = (~assigned) & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    denominator = np.where(np.abs((d4 - d3) + (d5 - d6)) > 1e-30,
                           (d4 - d3) + (d5 - d6), 1.0)
    w = (d4 - d3) / denominator
    q[mask] = (b[None, :, :] + w[:, :, None] * bc[None, :, :])[mask]
    assigned |= mask

    # Remaining points project inside the triangle's plane.
    mask = ~assigned
    denominator = np.where(np.abs(va + vb + vc) > 1e-30, va + vb + vc, 1.0)
    v = vb / denominator
    w = vc / denominator
    face_q = (a[None, :, :] + v[:, :, None] * ab[None, :, :] +
              w[:, :, None] * ac[None, :, :])
    q[mask] = face_q[mask]

    distances = np.sum((points[:, None, :] - q) ** 2, axis=2)
    nearest = np.argmin(distances, axis=1)
    rows = np.arange(len(points))
    closest = q[rows, nearest]
    closest_normals = normals[nearest]
    return closest, nearest, closest_normals


def signed_distance(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray,
                    chunk_size: int = 256) -> np.ndarray:
    """Compute local signed distance to an oriented, generally open triangle mesh.

    The magnitude is the Euclidean distance to the closest triangle.  The sign
    is ``sign(dot(p - closest, triangle_normal))`` using that triangle's listed
    orientation.  It is intentionally a *local* sign: an open disk has no
    globally defined inside/outside, and points near different faces can use
    different sides.  A zero value is assigned a positive sign.
    """

    points = np.asarray(points, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [Q, 3]")
    if len(faces) == 0:
        raise ValueError("mesh has no faces")
    output = np.empty(len(points), dtype=np.float64)
    # The vectorized region test allocates several [points, faces] arrays.
    # Reduce the chunk for refined meshes so memory remains practical.
    effective_chunk = min(max(1, int(chunk_size)), max(1, 1_000_000 // len(faces)))
    for begin in range(0, len(points), effective_chunk):
        end = min(len(points), begin + effective_chunk)
        query = points[begin:end]
        closest, _, normals = _closest_points_on_triangles(query, vertices, faces)
        magnitude = np.linalg.norm(query - closest, axis=1)
        side = np.sum((query - closest) * normals, axis=1)
        output[begin:end] = np.copysign(magnitude, np.where(side < 0, -1.0, 1.0))
    return output.astype(np.float32)


def json_dumps(value: object) -> str:
    """Stable JSON representation suitable for a scalar NPZ metadata field."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
