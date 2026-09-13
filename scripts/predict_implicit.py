#!/usr/bin/env python3
"""Evaluate a conditional local SDF on a grid and export its zero level as OBJ."""
import argparse
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("boundary", type=Path, help="Dataset NPZ (normalized) or ordered Nx3 NPY contour (world coordinates)")
    p.add_argument("--output", type=Path, default=Path("prediction.obj"))
    p.add_argument("--field-output", type=Path, help="Optional compressed NPZ containing field, axis and normalization")
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--extent", type=float, default=1.2, help="Normalized grid spans [-extent, extent] on each axis")
    p.add_argument("--chunk-size", type=int, default=32768)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.resolution < 8 or args.extent <= 0 or min(args.chunk_size, args.threads) < 1:
        p.error("resolution >= 8, extent > 0, chunk-size/threads >= 1 required")
    try:
        import numpy as np
        import torch
        from skimage.measure import marching_cubes
        from minsurf_nn.data import resample_boundary
        from minsurf_nn.model import ConditionalSDF, ModelConfig
    except ImportError as exc:
        p.exit(1, f"Missing ML dependency: {exc}. Install requirements-ml.txt\n")
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        p.error("Unsupported checkpoint format")
    model = ConditionalSDF(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    if args.boundary.suffix.lower() == ".npz":
        with np.load(args.boundary, allow_pickle=False) as sample:
            boundary = np.asarray(sample["boundary_points"], dtype=np.float32)
            center = np.asarray(sample["normalization_center"], dtype=np.float32)
            scale = float(sample["normalization_scale"])
    else:
        boundary = np.asarray(np.load(args.boundary, allow_pickle=False), dtype=np.float32)
        if boundary.ndim != 2 or boundary.shape[1] != 3 or not np.isfinite(boundary).all():
            p.error("NPY boundary must be a finite ordered Nx3 contour")
        center = boundary.mean(axis=0)
        scale = float(np.linalg.norm(boundary - center, axis=1).max())
        if scale <= 0:
            p.error("Boundary is degenerate")
        boundary = (boundary - center) / scale
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(scale) or scale <= 0:
        p.error("Invalid normalization metadata")
    count = checkpoint["training_config"]["boundary_count"]
    boundary = torch.from_numpy(resample_boundary(boundary, count)).unsqueeze(0).to(device)
    warnings.warn("Open patches have no global inside/outside SDF. This local-sign prototype may produce extra sheets and may not preserve the contour; OBJ is an experimental zero level, not a certified minimal surface.", stacklevel=1)
    axis = np.linspace(-args.extent, args.extent, args.resolution, dtype=np.float32)
    field = np.empty(args.resolution ** 3, dtype=np.float32)
    with torch.no_grad():
        context = model.encode(boundary)
        for start in range(0, len(field), args.chunk_size):
            stop = min(len(field), start + args.chunk_size)
            flat = np.arange(start, stop)
            indices = np.stack(np.unravel_index(flat, (args.resolution,) * 3), axis=-1)
            points = torch.from_numpy(axis[indices]).unsqueeze(0).to(device)
            field[start:stop] = model.decode(context, points).cpu().numpy()[0]
    field = field.reshape((args.resolution,) * 3)
    if not np.isfinite(field).all():
        p.error("Model produced non-finite field values")
    if args.field_output:
        args.field_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.field_output, field=field, axis=axis,
                            normalization_center=center, normalization_scale=scale)
    if not field.min() < 0 < field.max():
        p.error(f"No zero crossing in grid: [{field.min():.6g}, {field.max():.6g}]; train longer or adjust extent")
    spacing = float(axis[1] - axis[0])
    vertices, faces, _, _ = marching_cubes(field, level=0, spacing=(spacing,) * 3)
    vertices = (vertices + float(axis[0])) * scale + center
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        stream.write("# Experimental conditional local-SDF zero level, world coordinates\n")
        for x, y, z in vertices:
            stream.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in faces + 1:
            stream.write(f"f {a} {b} {c}\n")
    print(f"Saved {args.output}: {len(vertices)} vertices, {len(faces)} faces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
