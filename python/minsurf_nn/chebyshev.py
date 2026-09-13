"""Contour-conditioned finite Chebyshev field and direct area prediction."""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ChebyshevConfig:
    degree: int = 4
    extent: float = 1.35
    latent_dim: int = 128
    hidden_dim: int = 128

    def __post_init__(self):
        if self.degree < 1 or min(self.latent_dim, self.hidden_dim) < 1:
            raise ValueError("degree and dimensions must be positive")
        if self.extent <= 0:
            raise ValueError("extent must be positive")


def chebyshev_basis(values, degree, extent=1.0):
    """Return T_0..T_degree at values/extent without external libraries."""
    scaled = values / extent
    basis = [torch.ones_like(scaled), scaled]
    for _ in range(2, degree + 1):
        basis.append(2.0 * scaled * basis[-1] - basis[-2])
    return torch.stack(basis[:degree + 1], dim=-1)


def evaluate_chebyshev(coefficients, query, extent):
    if coefficients.ndim != 4 or query.ndim != 3 or query.shape[-1] != 3:
        raise ValueError("coefficients must be [B,D,D,D], query [B,Q,3]")
    degree = coefficients.shape[1] - 1
    if coefficients.shape[1:] != (degree + 1,) * 3 or coefficients.shape[0] != query.shape[0]:
        raise ValueError("inconsistent Chebyshev coefficient shape")
    x = chebyshev_basis(query[..., 0], degree, extent)
    y = chebyshev_basis(query[..., 1], degree, extent)
    z = chebyshev_basis(query[..., 2], degree, extent)
    return torch.einsum("bqi,bqj,bqk,bijk->bq", x, y, z, coefficients)


class ConditionalChebyshev(nn.Module):
    """Map a contour directly to finite-series coefficients and surface area."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ChebyshevConfig()
        c = self.config
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 64), nn.Softplus(), nn.Linear(64, 128), nn.Softplus(),
            nn.Linear(128, c.latent_dim),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(2 * c.latent_dim, c.hidden_dim), nn.Softplus(),
            nn.Linear(c.hidden_dim, c.latent_dim), nn.Softplus(),
        )
        count = (c.degree + 1) ** 3
        self.coefficient_head = nn.Linear(c.latent_dim, count)
        self.area_head = nn.Linear(c.latent_dim, 1)

    def encode(self, boundary):
        if boundary.ndim != 3 or boundary.shape[-1] != 3 or boundary.shape[1] < 3:
            raise ValueError("boundary must have shape [batch, points >= 3, 3]")
        features = self.point_encoder(boundary)
        pooled = torch.cat((features.amax(dim=1), features.mean(dim=1)), dim=-1)
        return self.context_encoder(pooled)

    def coefficients_from_context(self, context):
        size = self.config.degree + 1
        return self.coefficient_head(context).reshape(-1, size, size, size)

    def area_from_context(self, context):
        return F.softplus(self.area_head(context).squeeze(-1)) + 1e-6

    def decode(self, context, query):
        return evaluate_chebyshev(self.coefficients_from_context(context), query,
                                  self.config.extent)

    def forward(self, boundary, query):
        context = self.encode(boundary)
        return self.decode(context, query), self.area_from_context(context)

    def predict(self, boundary):
        context = self.encode(boundary)
        return self.coefficients_from_context(context), self.area_from_context(context)

    def config_dict(self):
        return asdict(self.config)


def coefficient_regularization(coefficients):
    """Penalize high orders more strongly to prefer a compact smooth formula."""
    size = coefficients.shape[1]
    order = torch.arange(size, device=coefficients.device, dtype=coefficients.dtype)
    weight = 1.0 + order[:, None, None] ** 2 + order[None, :, None] ** 2 + order[None, None, :] ** 2
    return (coefficients.square() * weight).mean()
