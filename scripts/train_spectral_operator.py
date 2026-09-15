#!/usr/bin/env python3
"""Hybrid training of a variable-size contour-to-Chebyshev operator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="rotation-pack manifest.jsonl")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/spectral-operator-p10-n64-hybrid-v2"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch-packs", type=int, default=32,
        help="NPZ packs per batch (effective batch = packs x rotations)",
    )
    parser.add_argument("--boundary-count", type=int, default=64)
    parser.add_argument("--point-width", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--supervised-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=10.0)
    parser.add_argument("--eikonal-weight", type=float, default=0.02)
    parser.add_argument("--curvature-weight", type=float, default=0.01)
    parser.add_argument("--graph-weight", type=float, default=0.1)
    parser.add_argument("--graph-margin", type=float, default=0.05)
    parser.add_argument("--collocation-samples", type=int, default=8)
    parser.add_argument(
        "--physics-contours-per-batch", type=int, default=32,
        help="one rotation per pack is enough for rotationally invariant physics",
    )
    parser.add_argument("--projection-steps", type=int, default=4)
    parser.add_argument("--curvature-warmup-epochs", type=int, default=5)
    parser.add_argument("--curvature-ramp-epochs", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"),
                        default="auto")
    parser.add_argument(
        "--train-pack-limit", type=int, default=0,
        help="debug/smoke-test limit; zero uses the full training split",
    )
    parser.add_argument(
        "--val-pack-limit", type=int, default=0,
        help="debug/smoke-test limit; zero uses the full validation split",
    )
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument(
        "--no-resume", action="store_true",
        help="fail instead of resuming when output-dir/last.pt exists",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = (args.epochs, args.batch_packs, args.boundary_count, args.point_width,
                args.latent_dim, args.hidden_dim, args.threads,
                args.collocation_samples, args.physics_contours_per_batch)
    if min(positive) < 1 or args.boundary_count < 3:
        parser.error("epochs/batch/dimensions/threads must be positive; boundary-count >= 3")
    if args.workers < 0 or args.seed < 0:
        parser.error("workers and seed must be non-negative")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("validation-fraction must lie in (0,1)")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("dropout must lie in [0,1)")
    if min(args.learning_rate, args.gradient_clip) <= 0.0 or args.weight_decay < 0.0:
        parser.error("learning-rate/gradient-clip must be positive and weight-decay non-negative")
    loss_weights = (args.supervised_weight, args.boundary_weight,
                    args.eikonal_weight, args.curvature_weight,
                    args.graph_weight, args.graph_margin)
    if min(loss_weights) < 0.0 or args.supervised_weight == 0.0:
        parser.error("loss weights/margin must be non-negative and supervised-weight positive")
    if min(args.projection_steps, args.curvature_warmup_epochs,
           args.curvature_ramp_epochs) < 0:
        parser.error("projection/warmup/ramp values must be non-negative")
    if min(args.train_pack_limit, args.val_pack_limit,
           args.max_train_batches, args.max_val_batches) < 0:
        parser.error("limits must be non-negative")


def _device(name, torch):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if name == "mps" and not (hasattr(torch.backends, "mps") and
                               torch.backends.mps.is_available()):
        raise ValueError("MPS requested but unavailable")
    return torch.device(name)


def _atomic_torch_save(value, path: Path, torch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _stats_path(output_dir: Path) -> Path:
    return output_dir / "coefficient_stats.npz"


def _compute_or_load_stats(records, output_dir: Path, expected_indices: np.ndarray):
    path = _stats_path(output_dir)
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            stats = {key: np.asarray(saved[key]) for key in saved.files}
        if not np.array_equal(stats["indices"], expected_indices):
            raise ValueError(f"stored coefficient ordering does not match model: {path}")
        return stats

    total = np.zeros(len(expected_indices), dtype=np.float64)
    total_squared = np.zeros_like(total)
    sample_count = 0
    degree = int(expected_indices.max())
    factors = np.where(expected_indices == 0, 1.0, 0.5)
    weights = factors.prod(axis=1).astype(np.float64)
    weighted_power_sum = 0.0
    started = time.monotonic()
    print(f"computing coefficient normalization from {len(records)} training packs", flush=True)
    for position, record in enumerate(records, 1):
        with np.load(record["path"], allow_pickle=False) as pack:
            coefficients = np.asarray(pack["coefficients"], dtype=np.float64)
            indices = np.asarray(pack["indices"], dtype=np.int16)
            pack_degree = int(pack["degree"])
        if (pack_degree != degree or not np.array_equal(indices, expected_indices) or
                coefficients.ndim != 2 or coefficients.shape[1] != len(expected_indices) or
                not np.isfinite(coefficients).all()):
            raise ValueError(f"incompatible coefficient pack: {record['path']}")
        total += coefficients.sum(axis=0)
        total_squared += np.square(coefficients).sum(axis=0)
        weighted_power_sum += float((np.square(coefficients) * weights).sum())
        sample_count += len(coefficients)
        if position % 10000 == 0:
            rate = position / max(time.monotonic() - started, 1e-9)
            print(f"stats {position}/{len(records)} packs ({rate:.1f} packs/s)", flush=True)
    mean = total / sample_count
    variance = np.maximum(total_squared / sample_count - np.square(mean), 0.0)
    raw_scale = np.sqrt(variance)
    scale_floor = max(1e-7, float(raw_scale.max()) * 1e-6)
    scale = np.maximum(raw_scale, scale_floor)
    target_power = weighted_power_sum / (sample_count * weights.sum())
    if not np.isfinite(target_power) or target_power <= 0.0:
        raise ValueError("training coefficients have zero/non-finite power")
    np.savez_compressed(
        path,
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        indices=expected_indices.astype(np.int16),
        l2_weights=weights.astype(np.float32),
        target_power=np.asarray(target_power, dtype=np.float64),
        sample_count=np.asarray(sample_count, dtype=np.int64),
    )
    return {
        "mean": mean.astype(np.float32), "scale": scale.astype(np.float32),
        "indices": expected_indices.astype(np.int16),
        "l2_weights": weights.astype(np.float32),
        "target_power": np.asarray(target_power, dtype=np.float64),
        "sample_count": np.asarray(sample_count, dtype=np.int64),
    }


def _sample_collocation(boundary, sample_count, torch):
    """Sample a star-shaped interior in xy and interpolate a useful z seed."""

    batch, count, _ = boundary.shape
    first_indices = torch.randint(count, (batch, sample_count),
                                  device=boundary.device)
    second_indices = torch.remainder(first_indices + 1, count)
    expanded_first = first_indices.unsqueeze(-1).expand(-1, -1, 3)
    expanded_second = second_indices.unsqueeze(-1).expand(-1, -1, 3)
    first = boundary.gather(1, expanded_first)
    second = boundary.gather(1, expanded_second)
    edge_fraction = torch.rand(batch, sample_count, 1, device=boundary.device)
    edge = first + edge_fraction * (second - first)
    center = boundary.mean(dim=1, keepdim=True)
    radius = torch.sqrt(torch.rand(batch, sample_count, 1,
                                   device=boundary.device))
    return center + radius * (edge - center)


def _curvature_multiplier(epoch, warmup, ramp):
    """Delay the stiff second-order equation until the network finds a sheet."""

    if epoch <= warmup:
        return 0.0
    if ramp == 0:
        return 1.0
    return min(1.0, (epoch - warmup) / ramp)


def _run_epoch(model, loader, optimizer, device, target_power, training,
               gradient_clip, maximum_batches, epoch, args, torch, losses):
    model.train(training)
    totals: dict[str, float] = {}
    examples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, batch in enumerate(loader):
            if maximum_batches and step >= maximum_batches:
                break
            boundary_pack = batch["boundary"].to(device, non_blocking=True)
            target_pack = batch["coefficients"].to(device, non_blocking=True)
            if boundary_pack.ndim != 4 or target_pack.ndim != 3:
                raise ValueError("pack batches must have shapes [B,C,N,3] and [B,C,K]")
            copies = boundary_pack.shape[1]
            boundary = boundary_pack.flatten(0, 1)
            target = target_pack.flatten(0, 1)
            if training:
                optimizer.zero_grad(set_to_none=True)
            normalized_prediction = model.normalized_coefficients(boundary)
            prediction = (model.coefficient_mean +
                          model.coefficient_scale * normalized_prediction)
            supervised, metrics = losses.normalized_coefficient_loss(
                normalized_prediction, target, model.coefficient_mean,
                model.coefficient_scale)
            _, physical_metrics = losses.weighted_coefficient_loss(
                prediction, target, model.l2_weights, target_power=target_power)
            physics_count = min(boundary_pack.shape[0],
                                args.physics_contours_per_batch)
            physics_boundary = boundary_pack[:physics_count, 0]
            physics_prediction = prediction[::copies][:physics_count]
            boundary_term = losses.boundary_distance_loss(
                physics_prediction, physics_boundary, model.indices)
            seeds = _sample_collocation(
                physics_boundary, args.collocation_samples, torch)
            surface_points = losses.project_to_graph_sheet(
                physics_prediction, seeds, model.indices,
                steps=args.projection_steps)
            geometry = losses.surface_geometry_losses(
                physics_prediction, surface_points, model.indices,
                graph_margin=args.graph_margin)
            curvature_multiplier = _curvature_multiplier(
                epoch, args.curvature_warmup_epochs,
                args.curvature_ramp_epochs)
            loss = (args.supervised_weight * supervised +
                    args.boundary_weight * boundary_term +
                    args.eikonal_weight * geometry["eikonal"] +
                    args.graph_weight * geometry["graph"] +
                    args.curvature_weight * curvature_multiplier *
                    geometry["curvature"])
            metrics.update({
                "loss": loss.detach(),
                "boundary_distance": boundary_term.detach(),
                "eikonal": geometry["eikonal"].detach(),
                "curvature": geometry["curvature"].detach(),
                "graph": geometry["graph"].detach(),
                "mean_abs_curvature": geometry["mean_abs_curvature"].detach(),
                "mean_abs_field_gradient":
                    geometry["mean_abs_field_gradient"].detach(),
                "curvature_multiplier": torch.as_tensor(
                    curvature_multiplier, device=device),
            })
            metrics.update({f"physical_{key}": value
                            for key, value in physical_metrics.items()
                            if key != "loss"})
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {'training' if training else 'validation'} loss")
            if training:
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip)
                optimizer.step()
                metrics["gradient_norm"] = gradient_norm.detach()
            count = len(boundary)
            examples += count
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + count * float(value)
    if not examples:
        raise RuntimeError("no batches were processed")
    return {key: value / examples for key, value in totals.items()}, examples


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    try:
        import torch
        from torch.utils.data import DataLoader
        from minsurf_nn.data import (ChebyshevRotationPackDataset, read_manifest,
                                     split_records)
        from minsurf_nn.spectral import total_degree_indices
        import minsurf_nn.spectral_losses as spectral_losses
        from minsurf_nn.spectral_operator import (ContourToChebyshev,
                                                  SpectralOperatorConfig)
    except ImportError as error:
        parser.exit(1, f"Missing ML dependency: {error}. Install requirements-ml.txt\n")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    try:
        device = _device(args.device, torch)
    except ValueError as error:
        parser.error(str(error))

    records = read_manifest(args.manifest)
    train_records, val_records = split_records(
        records, validation_fraction=args.validation_fraction, seed=args.seed)
    if args.train_pack_limit:
        train_records = train_records[:args.train_pack_limit]
    if args.val_pack_limit:
        val_records = val_records[:args.val_pack_limit]
    if not train_records or not val_records:
        parser.error("training and validation splits must remain nonempty after limits")
    degrees = {int(record["degree"]) for record in train_records + val_records}
    if len(degrees) != 1:
        parser.error(f"expected one polynomial degree, found {sorted(degrees)}")
    degree = degrees.pop()
    model_config = SpectralOperatorConfig(
        degree=degree, point_width=args.point_width, latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim, dropout=args.dropout)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = _compute_or_load_stats(
        train_records, args.output_dir, total_degree_indices(degree))
    model = ContourToChebyshev(model_config)
    model.set_output_normalization(stats["mean"], stats["scale"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs))

    start_epoch = 0
    best = float("inf")
    checkpoint_path = args.output_dir / "last.pt"
    if checkpoint_path.is_file():
        if args.no_resume:
            parser.error(f"checkpoint already exists: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("objective_version") != "hybrid_v2":
            parser.error("checkpoint was produced by a different training objective")
        if checkpoint.get("model_config") != model_config.to_dict():
            parser.error("checkpoint model configuration differs from requested architecture")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"])
        best = float(checkpoint.get("best_validation_loss", float("inf")))
        print(f"resumed {checkpoint_path} after epoch {start_epoch}", flush=True)
    if start_epoch >= args.epochs:
        print(f"already complete: checkpoint epoch {start_epoch} >= {args.epochs}", flush=True)
        return 0

    train_dataset = ChebyshevRotationPackDataset(
        train_records, boundary_count=args.boundary_count)
    val_dataset = ChebyshevRotationPackDataset(
        val_records, boundary_count=args.boundary_count)
    generator = torch.Generator().manual_seed(args.seed)
    common_loader = dict(
        batch_size=args.batch_packs, num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator,
                              **common_loader)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader)

    serializable_config = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    serializable_config.update(
        degree=degree, coefficient_count=model_config.coefficient_count,
        model_config=model_config.to_dict())
    (args.output_dir / "config.json").write_text(
        json.dumps(serializable_config, ensure_ascii=False, indent=2,
                   sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "split_summary.json").write_text(
        json.dumps({
            "train_packs": len(train_records), "validation_packs": len(val_records),
            "copies_per_pack": int(train_records[0].get("copies", 1)),
            "boundary_count": args.boundary_count,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    copies = int(train_records[0].get("copies", 1))
    print(json.dumps({
        "device": str(device), "degree": degree,
        "coefficients": model_config.coefficient_count,
        "parameters": parameter_count, "boundary_points": args.boundary_count,
        "train_packs": len(train_dataset), "validation_packs": len(val_dataset),
        "train_examples": len(train_dataset) * copies,
        "effective_batch_size": args.batch_packs * copies,
        "normalization_samples": int(stats["sample_count"]),
        "objective": "hybrid_v2",
    }), flush=True)

    target_power = float(stats["target_power"])
    metrics_path = args.output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        started = time.monotonic()
        train_metrics, train_examples = _run_epoch(
            model, train_loader, optimizer, device, target_power, True,
            args.gradient_clip, args.max_train_batches, epoch + 1, args, torch,
            spectral_losses)
        val_metrics, val_examples = _run_epoch(
            model, val_loader, optimizer, device, target_power, False,
            args.gradient_clip, args.max_val_batches, epoch + 1, args, torch,
            spectral_losses)
        scheduler.step()
        result = {
            "epoch": epoch + 1, "seconds": time.monotonic() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_examples": train_examples, "validation_examples": val_examples,
            "train": train_metrics, "validation": val_metrics,
        }
        improved = val_metrics["loss"] < best
        best = min(best, val_metrics["loss"])
        checkpoint = {
            "format_version": 1,
            "representation": "total_degree_chebyshev_contour_operator",
            "objective_version": "hybrid_v2",
            "epoch": epoch + 1,
            "best_validation_loss": best,
            "model_config": model_config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "training_config": serializable_config,
            "metrics": result,
        }
        _atomic_torch_save(checkpoint, checkpoint_path, torch)
        if improved:
            _atomic_torch_save(checkpoint, args.output_dir / "best.pt", torch)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
