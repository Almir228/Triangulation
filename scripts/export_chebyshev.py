#!/usr/bin/env python3
"""Export contour-conditioned Chebyshev coefficients as standalone Python."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


def render_formula(coefficients, center, scale, extent, predicted_area, boundary_world):
    coefficients = coefficients.tolist()
    boundary_xy = tuple((float(point[0]), float(point[1])) for point in boundary_world)
    return f'''"""Generated analytical approximation of F(x, y, z) = 0."""

CENTER = {tuple(float(x) for x in center)!r}
SCALE = {float(scale)!r}
EXTENT = {float(extent)!r}
PREDICTED_AREA = {float(predicted_area)!r}
COEFFICIENTS = {coefficients!r}
BOUNDARY_XY = {boundary_xy!r}


def _basis(value):
    value = value / EXTENT
    result = [1.0, value]
    for _ in range(2, len(COEFFICIENTS)):
        result.append(2.0 * value * result[-1] - result[-2])
    return result[:len(COEFFICIENTS)]


def F(x, y, z):
    """Evaluate the predicted field in world coordinates; the surface is F=0."""
    x = (x - CENTER[0]) / SCALE
    y = (y - CENTER[1]) / SCALE
    z = (z - CENTER[2]) / SCALE
    tx, ty, tz = _basis(x), _basis(y), _basis(z)
    return sum(COEFFICIENTS[i][j][k] * tx[i] * ty[j] * tz[k]
               for i in range(len(COEFFICIENTS))
               for j in range(len(COEFFICIENTS))
               for k in range(len(COEFFICIENTS)))


def inside_domain(x, y):
    """Whether (x,y) is inside the contour projection used to trim the sheet."""
    inside = False
    previous = BOUNDARY_XY[-1]
    for current in BOUNDARY_XY:
        if ((current[1] > y) != (previous[1] > y) and
                x < (previous[0] - current[0]) * (y - current[1]) /
                (previous[1] - current[1]) + current[0]):
            inside = not inside
        previous = current
    return inside


def surface_contains(x, y, z, tolerance=1e-6):
    """The exported open patch is F=0 restricted to the contour projection."""
    return inside_domain(x, y) and abs(F(x, y, z)) <= tolerance
'''


def points_in_polygon(points, polygon):
    """Vectorized even-odd test for XY points and an ordered XY polygon."""
    import numpy as np
    points = np.asarray(points)
    polygon = np.asarray(polygon)
    inside = np.zeros(len(points), dtype=bool)
    previous = polygon[-1]
    x, y = points[:, 0], points[:, 1]
    for current in polygon:
        crossing = ((current[1] > y) != (previous[1] > y))
        at_x = ((previous[0] - current[0]) * (y - current[1]) /
                (previous[1] - current[1] + 1e-30) + current[0])
        inside ^= crossing & (x < at_x)
        previous = current
    return inside


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("boundary", type=Path,
                   help="dataset NPZ or ordered world-coordinate Nx3 NPY contour")
    p.add_argument("--output", type=Path, default=Path("surface_formula.py"))
    p.add_argument("--coefficients", type=Path, help="optional coefficient NPZ")
    p.add_argument("--obj", type=Path, help="optional Marching Cubes preview")
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--grid-extent", type=float, default=1.2)
    p.add_argument("--prune-relative", type=float, default=0.0,
                   help="zero coefficients below fraction of maximum magnitude")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.resolution < 8 or args.grid_extent <= 0 or args.threads < 1:
        p.error("resolution >= 8, positive grid extent and threads required")
    if not 0 <= args.prune_relative < 1:
        p.error("prune-relative must be in [0,1)")
    try:
        import numpy as np
        import torch
        from minsurf_nn.chebyshev import ChebyshevConfig, ConditionalChebyshev
        from minsurf_nn.data import resample_boundary
    except ImportError as exc:
        p.exit(1, f"Missing ML dependency: {exc}. Install requirements-ml.txt\n")
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("representation") != "chebyshev":
        p.error("checkpoint is not a Chebyshev formula model")
    model = ConditionalChebyshev(ChebyshevConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    if args.boundary.suffix.lower() == ".npz":
        with np.load(args.boundary, allow_pickle=False) as sample:
            boundary = np.asarray(sample["boundary_points"], dtype=np.float32)
            center = np.asarray(sample["normalization_center"], dtype=np.float64)
            scale = float(sample["normalization_scale"])
    else:
        world_boundary = np.asarray(np.load(args.boundary, allow_pickle=False), dtype=np.float32)
        if (world_boundary.ndim != 2 or world_boundary.shape[1] != 3 or
                not np.isfinite(world_boundary).all()):
            p.error("NPY boundary must be finite Nx3")
        center = world_boundary.mean(axis=0).astype(np.float64)
        scale = float(np.linalg.norm(world_boundary - center, axis=1).max())
        if not np.isfinite(scale) or scale <= 0:
            p.error("boundary is degenerate")
        boundary = (world_boundary - center) / scale
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(scale) or scale <= 0:
        p.error("invalid boundary normalization")
    count = checkpoint["training_config"]["boundary_count"]
    boundary_tensor = torch.from_numpy(resample_boundary(boundary, count)).unsqueeze(0).to(device)
    with torch.no_grad():
        coefficients, normalized_area = model.predict(boundary_tensor)
    coefficients = coefficients[0].cpu().numpy()
    if args.prune_relative and coefficients.size:
        threshold = args.prune_relative * float(np.abs(coefficients).max())
        coefficients[np.abs(coefficients) < threshold] = 0.0
    predicted_area = float(normalized_area[0]) * scale * scale
    args.output.parent.mkdir(parents=True, exist_ok=True)
    boundary_world = boundary * scale + center
    args.output.write_text(render_formula(coefficients, center, scale,
                                          model.config.extent, predicted_area,
                                          boundary_world), encoding="utf-8")
    if args.coefficients:
        args.coefficients.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.coefficients, coefficients=coefficients,
                            normalization_center=center, normalization_scale=scale,
                            extent=model.config.extent, predicted_area=predicted_area)
    if args.obj:
        try:
            from skimage.measure import marching_cubes
        except ImportError as exc:
            p.exit(1, f"OBJ preview requires scikit-image: {exc}\n")
        axis = np.linspace(-args.grid_extent, args.grid_extent, args.resolution,
                           dtype=np.float32)
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
        queries = torch.from_numpy(grid.reshape(1, -1, 3)).to(device)
        kept = torch.from_numpy(coefficients).unsqueeze(0).to(device)
        from minsurf_nn.chebyshev import evaluate_chebyshev
        with torch.no_grad():
            field = evaluate_chebyshev(kept, queries, model.config.extent)
        field = field.cpu().numpy().reshape((args.resolution,) * 3)
        if not field.min() < 0 < field.max():
            p.error(f"no zero crossing in grid: [{field.min():.6g}, {field.max():.6g}]")
        spacing = float(axis[1] - axis[0])
        vertices, faces, _, _ = marching_cubes(field, level=0, spacing=(spacing,) * 3)
        centroids = vertices[faces].mean(axis=1) + float(axis[0])
        faces = faces[points_in_polygon(centroids[:, :2], boundary[:, :2])]
        if not len(faces):
            p.error("zero level does not intersect the projected contour domain")
        used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
        vertices = vertices[used]
        faces = inverse.reshape(-1, 3)
        vertices = (vertices + float(axis[0])) * scale + center
        args.obj.parent.mkdir(parents=True, exist_ok=True)
        with args.obj.open("w", encoding="utf-8") as stream:
            stream.write("# Chebyshev formula zero level, world coordinates\n")
            for x, y, z in vertices:
                stream.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
            for a, b, c in faces + 1:
                stream.write(f"f {a} {b} {c}\n")
        print(f"Saved {args.obj}: {len(vertices)} vertices, {len(faces)} faces")
    active = int(np.count_nonzero(coefficients))
    print(f"Saved {args.output}: {active}/{coefficients.size} coefficients, "
          f"predicted area {predicted_area:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
