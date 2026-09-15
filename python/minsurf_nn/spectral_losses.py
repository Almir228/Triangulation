"""Supervised and future geometry-only losses for the spectral operator."""

from __future__ import annotations

from typing import Dict

import torch

from .spectral_operator import (evaluate_total_degree_chebyshev,
                                evaluate_total_degree_chebyshev_geometry)


def normalized_coefficient_loss(
        predicted_normalized: torch.Tensor, target: torch.Tensor,
        mean: torch.Tensor, scale: torch.Tensor
        ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """MSE after per-coefficient standardization.

    Equalizing the target variance prevents the many small high-order modes
    from being ignored in favor of a dataset-mean surface.
    """

    if predicted_normalized.shape != target.shape or target.ndim != 2:
        raise ValueError("prediction and target must have equal shape [B,K]")
    if mean.shape != target.shape[1:] or scale.shape != target.shape[1:]:
        raise ValueError("mean and scale must have shape [K]")
    target_normalized = (target - mean) / scale
    error = predicted_normalized - target_normalized
    loss = error.square().mean()
    return loss, {
        "normalized_loss": loss.detach(),
        "normalized_rmse": torch.sqrt(loss.detach()),
        "normalized_mae": error.abs().mean().detach(),
    }


def weighted_coefficient_loss(prediction: torch.Tensor, target: torch.Tensor,
                              weights: torch.Tensor,
                              target_power: float | torch.Tensor = 1.0
                              ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Chebyshev-weighted coefficient error, normalized by dataset power."""

    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must have equal shape [B,K]")
    if weights.shape != (prediction.shape[1],):
        raise ValueError("weights must have shape [K]")
    power = torch.as_tensor(target_power, dtype=prediction.dtype,
                            device=prediction.device)
    if not torch.isfinite(power) or power <= 0:
        raise ValueError("target_power must be positive and finite")
    squared = (prediction - target).square()
    weighted_mse = (squared * weights).sum(dim=1).div(weights.sum()).mean()
    loss = weighted_mse / power
    metrics = {
        "loss": loss.detach(),
        "weighted_mse": weighted_mse.detach(),
        "relative_l2": torch.sqrt(loss.detach()),
        "coefficient_rmse": torch.sqrt(squared.mean()).detach(),
        "coefficient_mae": (prediction - target).abs().mean().detach(),
    }
    return loss, metrics


def boundary_zero_loss(coefficients: torch.Tensor, boundary: torch.Tensor,
                       indices: torch.Tensor) -> torch.Tensor:
    """Geometry-only constraint available for later unsupervised fine-tuning."""

    return evaluate_total_degree_chebyshev(
        coefficients, boundary, indices).square().mean()


def boundary_distance_loss(coefficients: torch.Tensor, boundary: torch.Tensor,
                           indices: torch.Tensor,
                           epsilon: float = 1e-6) -> torch.Tensor:
    """Scale-invariant first-order squared distance from boundary to ``F=0``."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    values, gradient, _ = evaluate_total_degree_chebyshev_geometry(
        coefficients, boundary, indices, with_hessian=False)
    return (values.square() / (gradient.square().sum(dim=-1) + epsilon)).mean()


def project_to_graph_sheet(coefficients: torch.Tensor, points: torch.Tensor,
                           indices: torch.Tensor, steps: int = 4,
                           maximum_step: float = 0.25,
                           minimum_slope: float = 1e-3) -> torch.Tensor:
    """Detach and Newton-project seed points onto the graph-like zero sheet."""

    if steps < 0 or maximum_step <= 0.0 or minimum_slope <= 0.0:
        raise ValueError("invalid projection parameters")
    projected = points.detach().clone()
    with torch.no_grad():
        detached = coefficients.detach()
        for _ in range(steps):
            values, gradient, _ = evaluate_total_degree_chebyshev_geometry(
                detached, projected, indices, with_hessian=False)
            slope = gradient[..., 2]
            safe_slope = torch.where(
                slope.abs() >= minimum_slope, slope,
                torch.where(slope >= 0.0, minimum_slope, -minimum_slope))
            delta = (values / safe_slope).clamp(-maximum_step, maximum_step)
            projected[..., 2] = (projected[..., 2] - delta).clamp(-1.0, 1.0)
    return projected


def surface_geometry_losses(
        coefficients: torch.Tensor, points: torch.Tensor, indices: torch.Tensor,
        graph_margin: float = 0.05, epsilon: float = 1e-6
        ) -> Dict[str, torch.Tensor]:
    """Mean-curvature, Eikonal and positive-z graph losses on surface points."""

    if graph_margin < 0.0 or epsilon <= 0.0:
        raise ValueError("graph_margin must be non-negative and epsilon positive")
    _, gradient, hessian = evaluate_total_degree_chebyshev_geometry(
        coefficients, points, indices)
    assert hessian is not None
    gradient_squared = gradient.square().sum(dim=-1)
    gradient_norm = torch.sqrt(gradient_squared + epsilon)
    trace = hessian.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    hessian_gradient = torch.einsum("bqij,bqj->bqi", hessian, gradient)
    quadratic = (gradient * hessian_gradient).sum(dim=-1)
    curvature = ((gradient_squared * trace - quadratic) /
                 (gradient_squared + epsilon).pow(1.5))
    curvature_squared = curvature.square()
    return {
        # Pseudo-Huber keeps the second-order residual useful near zero without
        # letting an occasional low-gradient Newton failure dominate a batch.
        "curvature": (torch.sqrt(1.0 + curvature_squared) - 1.0).mean(),
        "mean_abs_curvature": curvature.abs().mean(),
        "eikonal": (gradient_norm - 1.0).square().mean(),
        "graph": torch.relu(graph_margin - gradient[..., 2]).square().mean(),
        "mean_abs_field_gradient": gradient_norm.mean(),
    }


def eikonal_loss(coefficients: torch.Tensor, points: torch.Tensor,
                  indices: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Keep the defining function nontrivial and signed-distance-like."""

    points = points.requires_grad_(True)
    values = evaluate_total_degree_chebyshev(coefficients, points, indices)
    gradient = torch.autograd.grad(values.sum(), points, create_graph=create_graph)[0]
    return (gradient.norm(dim=-1) - 1.0).square().mean()


def minimal_surface_loss(coefficients: torch.Tensor, points: torch.Tensor,
                         indices: torch.Tensor, band: float = 0.05,
                         epsilon: float = 1e-6) -> torch.Tensor:
    """Narrow-band squared mean curvature for later label-free fine-tuning.

    The exponential band concentrates the implicit minimal-surface equation
    near ``F=0``. Combine this with boundary and Eikonal terms; by itself the
    zero polynomial is a degenerate solution.
    """

    if band <= 0.0 or epsilon <= 0.0:
        raise ValueError("band and epsilon must be positive")
    points = points.requires_grad_(True)
    values = evaluate_total_degree_chebyshev(coefficients, points, indices)
    gradient = torch.autograd.grad(values.sum(), points, create_graph=True)[0]
    norm = torch.sqrt(gradient.square().sum(dim=-1, keepdim=True) + epsilon ** 2)
    normal = gradient / norm
    divergence = torch.zeros_like(values)
    for axis in range(3):
        derivative = torch.autograd.grad(
            normal[..., axis].sum(), points, create_graph=True,
            retain_graph=True)[0][..., axis]
        divergence = divergence + derivative
    weight = torch.exp(-values.detach().square() / (band ** 2))
    return (weight * divergence.square()).sum() / weight.sum().clamp_min(epsilon)
