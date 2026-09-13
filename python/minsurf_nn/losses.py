"""Local-distance fitting; no claim of globally defined SDF for open patches."""
import torch
from torch.nn import functional as F


def _mean_curvature(model, context, query, training):
    """Divergence of the normalized field gradient at query points."""
    query = query.detach().requires_grad_(True)
    values = model.decode(context, query)
    gradient = torch.autograd.grad(values.sum(), query, create_graph=True,
                                   retain_graph=True)[0]
    rows = []
    for coordinate in range(3):
        component = gradient[..., coordinate]
        if not component.requires_grad:
            rows.append(torch.zeros_like(query))
            continue
        second = torch.autograd.grad(component.sum(), query, create_graph=training,
                                     retain_graph=True, allow_unused=True)[0]
        rows.append(torch.zeros_like(query) if second is None else second)
    hessian = torch.stack(rows, dim=-2)
    norm = gradient.norm(dim=-1).clamp_min(1e-6)
    trace = hessian.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    quadratic = torch.einsum("...i,...ij,...j->...", gradient, hessian, gradient)
    return trace / norm - quadratic / norm.pow(3)


def surface_loss(model, batch, sdf_weight=1.0, boundary_weight=1.0,
                 eikonal_weight=0.01, eikonal_samples=128, eikonal_band=0.15,
                 minimal_weight=0.001, minimal_samples=32, minimal_band=0.08,
                 training=True):
    if min(sdf_weight, boundary_weight, eikonal_weight, minimal_weight) < 0:
        raise ValueError("Loss weights must be nonnegative")
    if min(eikonal_samples, minimal_samples) < 1 or min(eikonal_band, minimal_band) <= 0:
        raise ValueError("Sample counts and bands must be positive")
    context = model.encode(batch["boundary"])
    prediction = model.decode(context, batch["query"])
    regression = F.smooth_l1_loss(prediction, batch["sdf"], beta=0.01)
    boundary = model.decode(context, batch["boundary"]).square().mean()
    eikonal = prediction.new_zeros(())
    if eikonal_weight:
        count = min(eikonal_samples, batch["query"].shape[1])
        query = batch["query"][:, :count].detach().requires_grad_(True)
        values = model.decode(context, query)
        gradient = torch.autograd.grad(values.sum(), query, create_graph=training, retain_graph=training)[0]
        mask = batch["sdf"][:, :count].abs() < eikonal_band
        if mask.any():
            eikonal = (gradient.norm(dim=-1)[mask] - 1).square().mean()
    minimality = prediction.new_zeros(())
    if minimal_weight:
        count = min(minimal_samples, batch["query"].shape[1])
        mask = batch["sdf"][:, :count].abs() < minimal_band
        if mask.any():
            curvature = _mean_curvature(model, context, batch["query"][:, :count], training)
            minimality = curvature[mask].square().mean()
    total = (sdf_weight * regression + boundary_weight * boundary +
             eikonal_weight * eikonal + minimal_weight * minimality)
    metrics = {"loss": float(total.detach()), "sdf": float(regression.detach()),
               "boundary": float(boundary.detach()), "eikonal": float(eikonal.detach()),
               "minimality": float(minimality.detach()),
               "mae": float((prediction - batch["sdf"]).detach().abs().mean())}
    return total, metrics
