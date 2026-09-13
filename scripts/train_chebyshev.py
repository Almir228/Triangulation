#!/usr/bin/env python3
"""Train a contour-to-Chebyshev-formula model and an area prediction head."""

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--output", type=Path, default=Path("runs/chebyshev"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--boundary-count", type=int, default=64)
    p.add_argument("--query-count", type=int, default=1024)
    p.add_argument("--degree", type=int, default=4)
    p.add_argument("--extent", type=float, default=1.35)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--sdf-weight", type=float, default=1.0)
    p.add_argument("--boundary-weight", type=float, default=1.0)
    p.add_argument("--eikonal-weight", type=float, default=0.01)
    p.add_argument("--eikonal-samples", type=int, default=64)
    p.add_argument("--eikonal-band", type=float, default=0.15)
    p.add_argument("--minimal-weight", type=float, default=0.001)
    p.add_argument("--minimal-samples", type=int, default=16)
    p.add_argument("--minimal-band", type=float, default=0.08)
    p.add_argument("--area-weight", type=float, default=0.1)
    p.add_argument("--spectral-weight", type=float, default=1e-5)
    p.add_argument("--max-batches", type=int, default=None)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if min(args.epochs, args.batch_size, args.boundary_count, args.query_count,
           args.degree, args.latent_dim, args.hidden_dim, args.threads) < 1:
        p.error("counts, degree, dimensions and threads must be positive")
    if min(args.extent, args.learning_rate, args.eikonal_band, args.minimal_band) <= 0:
        p.error("extent, learning rate and bands must be positive")
    if min(args.sdf_weight, args.boundary_weight, args.eikonal_weight,
           args.minimal_weight, args.area_weight, args.spectral_weight) < 0:
        p.error("loss weights must be nonnegative")
    if args.workers < 0 or args.seed < 0 or (args.max_batches is not None and args.max_batches < 1):
        p.error("workers/seed must be nonnegative and max-batches positive")
    try:
        import numpy as np
        import torch
        from torch.nn import functional as F
        from torch.utils.data import DataLoader
        from minsurf_nn.chebyshev import (ChebyshevConfig, ConditionalChebyshev,
                                          coefficient_regularization)
        from minsurf_nn.data import SurfaceDataset, read_manifest, split_records
        from minsurf_nn.losses import surface_loss
    except ImportError as exc:
        p.exit(1, f"Missing ML dependency: {exc}. Install requirements-ml.txt\n")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    records = read_manifest(args.manifest)
    train_records, val_records = split_records(records, args.validation_fraction, args.seed)
    train_data = SurfaceDataset(train_records, args.boundary_count, args.query_count,
                                args.seed, return_area=True)
    val_data = SurfaceDataset(val_records, args.boundary_count, args.query_count,
                              args.seed + 1, return_area=True)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, generator=generator)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers)
    model = ConditionalChebyshev(ChebyshevConfig(
        args.degree, args.extent, args.latent_dim, args.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "last.pt").exists():
        p.error("Output already contains last.pt; choose a fresh directory")
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output / "split.json").write_text(json.dumps(
        {"train": train_records, "val": val_records}, indent=2), encoding="utf-8")
    surface_options = dict(
        sdf_weight=args.sdf_weight, boundary_weight=args.boundary_weight,
        eikonal_weight=args.eikonal_weight, eikonal_samples=args.eikonal_samples,
        eikonal_band=args.eikonal_band, minimal_weight=args.minimal_weight,
        minimal_samples=args.minimal_samples, minimal_band=args.minimal_band)
    print(json.dumps({"device": str(device), "train": len(train_data),
                      "val": len(val_data), "coefficients": (args.degree + 1) ** 3}), flush=True)
    best = float("inf")
    for epoch in range(args.epochs):
        train_data.set_epoch(epoch)
        result = {"epoch": epoch + 1}
        for name, loader in (("train", train_loader), ("val", val_loader)):
            training = name == "train"
            model.train(training)
            totals, samples = {}, 0
            with torch.enable_grad():
                for step, batch in enumerate(loader):
                    if args.max_batches is not None and step >= args.max_batches:
                        break
                    batch = {key: value.to(device) for key, value in batch.items()}
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    base, metrics = surface_loss(model, batch, training=training,
                                                 **surface_options)
                    context = model.encode(batch["boundary"])
                    coefficients = model.coefficients_from_context(context)
                    area_prediction = model.area_from_context(context)
                    area_loss = F.smooth_l1_loss(
                        area_prediction.log(), batch["area"].log(), beta=0.05)
                    spectral = coefficient_regularization(coefficients)
                    loss = base + args.area_weight * area_loss + args.spectral_weight * spectral
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite {name} loss at epoch {epoch + 1}")
                    if training:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                        optimizer.step()
                    metrics.update(loss=float(loss.detach()), area=float(area_loss.detach()),
                                   area_relative=float(((area_prediction - batch["area"]).abs() /
                                                       batch["area"]).mean().detach()),
                                   spectral=float(spectral.detach()))
                    size = batch["boundary"].shape[0]
                    samples += size
                    for key, value in metrics.items():
                        totals[key] = totals.get(key, 0.0) + size * value
            result[name] = {key: value / samples for key, value in totals.items()}
        checkpoint = {
            "format_version": 1, "representation": "chebyshev", "epoch": epoch + 1,
            "model_config": model.config_dict(), "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "training_config": config,
            "metrics": result,
        }
        temporary = args.output / "checkpoint.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(args.output / "last.pt")
        if result["val"]["loss"] < best:
            best = result["val"]["loss"]
            torch.save(checkpoint, temporary)
            temporary.replace(args.output / "best.pt")
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
