"""Variable-size contour encoder for total-degree Chebyshev coefficients."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import comb
from typing import Optional

import numpy as np
import torch
from torch import nn

from .spectral import total_degree_indices


@dataclass(frozen=True)
class SpectralOperatorConfig:
    degree: int = 10
    point_width: int = 128
    latent_dim: int = 256
    hidden_dim: int = 256
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError("degree must be positive")
        if min(self.point_width, self.latent_dim, self.hidden_dim) < 1:
            raise ValueError("network dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")

    @property
    def coefficient_count(self) -> int:
        return comb(self.degree + 3, 3)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def chebyshev_table_torch(values: torch.Tensor, degree: int) -> torch.Tensor:
    """Differentiably evaluate ``T_0,...,T_degree`` on an arbitrary tensor."""

    if degree < 0:
        raise ValueError("degree must be non-negative")
    table = [torch.ones_like(values)]
    if degree:
        table.append(values)
    for _ in range(2, degree + 1):
        table.append(2.0 * values * table[-1] - table[-2])
    return torch.stack(table, dim=-1)


def chebyshev_table_with_derivatives_torch(
        values: torch.Tensor, degree: int
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate first-kind Chebyshev polynomials and two derivatives."""

    if degree < 0:
        raise ValueError("degree must be non-negative")
    table = [torch.ones_like(values)]
    first = [torch.zeros_like(values)]
    second = [torch.zeros_like(values)]
    if degree:
        table.append(values)
        first.append(torch.ones_like(values))
        second.append(torch.zeros_like(values))
    for _ in range(2, degree + 1):
        table.append(2.0 * values * table[-1] - table[-2])
        first.append(2.0 * table[-2] + 2.0 * values * first[-1] - first[-2])
        second.append(4.0 * first[-2] + 2.0 * values * second[-1] - second[-2])
    return (torch.stack(table, dim=-1), torch.stack(first, dim=-1),
            torch.stack(second, dim=-1))


def evaluate_total_degree_chebyshev(coefficients: torch.Tensor,
                                    points: torch.Tensor,
                                    indices: torch.Tensor) -> torch.Tensor:
    """Evaluate batched total-degree polynomials at batched 3D points.

    ``coefficients`` has shape ``[B,K]``, ``points`` has shape ``[B,Q,3]`` and
    ``indices`` is the common stable ``[K,3]`` ordering.
    """

    if coefficients.ndim != 2:
        raise ValueError("coefficients must have shape [B,K]")
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B,Q,3]")
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("indices must have shape [K,3]")
    if coefficients.shape[0] != points.shape[0] or coefficients.shape[1] != len(indices):
        raise ValueError("batch/coefficient dimensions are inconsistent")
    degree = int(indices.max().item())
    tx = chebyshev_table_torch(points[..., 0], degree)
    ty = chebyshev_table_torch(points[..., 1], degree)
    tz = chebyshev_table_torch(points[..., 2], degree)
    basis = (tx[..., indices[:, 0]] * ty[..., indices[:, 1]] *
             tz[..., indices[:, 2]])
    return torch.einsum("bqk,bk->bq", basis, coefficients)


def evaluate_total_degree_chebyshev_geometry(
        coefficients: torch.Tensor, points: torch.Tensor,
        indices: torch.Tensor, with_hessian: bool = True
        ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Evaluate a polynomial together with its gradient and Hessian.

    The recurrences are fully differentiable with respect to coefficients, but
    avoid nested point-autograd.  Shapes are ``[B,Q]``, ``[B,Q,3]`` and
    ``[B,Q,3,3]`` respectively.  The last result is ``None`` when
    ``with_hessian=False``, which is useful for the cheaper boundary term.
    """

    if coefficients.ndim != 2:
        raise ValueError("coefficients must have shape [B,K]")
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B,Q,3]")
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("indices must have shape [K,3]")
    if coefficients.shape[0] != points.shape[0] or coefficients.shape[1] != len(indices):
        raise ValueError("batch/coefficient dimensions are inconsistent")

    degree = int(indices.max().item())
    tables = [chebyshev_table_with_derivatives_torch(
        points[..., axis], degree) for axis in range(3)]
    selected = [[part[..., indices[:, axis]] for part in tables[axis]]
                for axis in range(3)]

    def contract(a: torch.Tensor, b: torch.Tensor,
                 c: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bqk,bk->bq", a * b * c, coefficients)

    values = contract(selected[0][0], selected[1][0], selected[2][0])
    gradient = torch.stack([
        contract(selected[0][1], selected[1][0], selected[2][0]),
        contract(selected[0][0], selected[1][1], selected[2][0]),
        contract(selected[0][0], selected[1][0], selected[2][1]),
    ], dim=-1)
    if not with_hessian:
        return values, gradient, None
    xx = contract(selected[0][2], selected[1][0], selected[2][0])
    yy = contract(selected[0][0], selected[1][2], selected[2][0])
    zz = contract(selected[0][0], selected[1][0], selected[2][2])
    xy = contract(selected[0][1], selected[1][1], selected[2][0])
    xz = contract(selected[0][1], selected[1][0], selected[2][1])
    yz = contract(selected[0][0], selected[1][1], selected[2][1])
    hessian = torch.stack((
        torch.stack((xx, xy, xz), dim=-1),
        torch.stack((xy, yy, yz), dim=-1),
        torch.stack((xz, yz, zz), dim=-1),
    ), dim=-2)
    return values, gradient, hessian


def chebyshev_l2_weights(indices: torch.Tensor) -> torch.Tensor:
    """Diagonal weights for the first-kind Chebyshev product ``L2`` norm.

    Under the normalized weight ``dx/(pi*sqrt(1-x^2))``, ``||T_0||^2=1`` and
    ``||T_n||^2=1/2`` for ``n>0``.  Tensor-product basis functions therefore
    have weights equal to the product of these three factors.
    """

    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("indices must have shape [K,3]")
    factors = torch.where(indices == 0,
                          torch.ones_like(indices, dtype=torch.float32),
                          torch.full_like(indices, 0.5, dtype=torch.float32))
    return factors.prod(dim=1)


class ContourToChebyshev(nn.Module):
    """Map any sampled closed contour ``[B,N,3]`` to ``[B,K]`` coefficients.

    Local cyclic edge features preserve polygon connectivity and orientation;
    shared encoding plus mean/max pooling makes the result independent of the
    chosen first vertex and of ``N``.  A model first trained with ``N=64`` can
    consequently be fine-tuned with denser 128/252/256-point contours.
    """

    def __init__(self, config: Optional[SpectralOperatorConfig] = None) -> None:
        super().__init__()
        self.config = config or SpectralOperatorConfig()
        c = self.config
        middle = max(32, c.point_width // 2)
        self.point_encoder = nn.Sequential(
            nn.Linear(9, middle),
            nn.SiLU(),
            nn.Linear(middle, c.point_width),
            nn.SiLU(),
            nn.Linear(c.point_width, c.latent_dim),
            nn.SiLU(),
        )
        self.coefficient_head = nn.Sequential(
            nn.Linear(2 * c.latent_dim, c.hidden_dim),
            nn.LayerNorm(c.hidden_dim),
            nn.SiLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.SiLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim, c.coefficient_count),
        )
        nn.init.normal_(self.coefficient_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.coefficient_head[-1].bias)

        indices = torch.from_numpy(total_degree_indices(c.degree).astype(np.int64))
        self.register_buffer("indices", indices, persistent=True)
        self.register_buffer("coefficient_mean", torch.zeros(c.coefficient_count),
                             persistent=True)
        self.register_buffer("coefficient_scale", torch.ones(c.coefficient_count),
                             persistent=True)
        self.register_buffer("l2_weights", chebyshev_l2_weights(indices),
                             persistent=True)

    def set_output_normalization(self, mean: torch.Tensor,
                                 scale: torch.Tensor) -> None:
        mean = torch.as_tensor(mean, dtype=self.coefficient_mean.dtype,
                               device=self.coefficient_mean.device)
        scale = torch.as_tensor(scale, dtype=self.coefficient_scale.dtype,
                                device=self.coefficient_scale.device)
        expected = (self.config.coefficient_count,)
        if mean.shape != expected or scale.shape != expected:
            raise ValueError(f"normalization arrays must have shape {expected}")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("normalization arrays must be finite")
        if torch.any(scale <= 0):
            raise ValueError("normalization scale must be positive")
        self.coefficient_mean.copy_(mean)
        self.coefficient_scale.copy_(scale)

    def encode(self, boundary: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if boundary.ndim != 3 or boundary.shape[-1] != 3 or boundary.shape[1] < 3:
            raise ValueError("boundary must have shape [B,N>=3,3]")
        if mask is None:
            previous = torch.roll(boundary, shifts=1, dims=1)
            following = torch.roll(boundary, shifts=-1, dims=1)
            local = torch.cat((boundary, following - boundary,
                               boundary - previous), dim=-1)
            features = self.point_encoder(local)
            mean = features.mean(dim=1)
            maximum = features.amax(dim=1)
        else:
            if mask.shape != boundary.shape[:2]:
                raise ValueError("mask must have shape [B,N]")
            mask = mask.to(dtype=torch.bool, device=boundary.device)
            counts = mask.sum(dim=1, keepdim=True)
            if torch.any(counts < 3):
                raise ValueError("every masked contour must contain at least three points")
            positions = torch.arange(boundary.shape[1], device=boundary.device)[None, :]
            if not torch.equal(mask, positions < counts):
                raise ValueError("masked contour points must form a contiguous prefix")
            previous_index = torch.remainder(positions - 1, counts)
            following_index = torch.remainder(positions + 1, counts)
            previous = boundary.gather(
                1, previous_index.unsqueeze(-1).expand(-1, -1, 3))
            following = boundary.gather(
                1, following_index.unsqueeze(-1).expand(-1, -1, 3))
            local = torch.cat((boundary, following - boundary,
                               boundary - previous), dim=-1)
            features = self.point_encoder(local)
            expanded = mask.unsqueeze(-1)
            mean = (features * expanded).sum(dim=1) / counts.to(features.dtype)
            maximum = features.masked_fill(~expanded, -torch.inf).amax(dim=1)
        return torch.cat((mean, maximum), dim=-1)

    def normalized_coefficients(self, boundary: torch.Tensor,
                                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.coefficient_head(self.encode(boundary, mask))

    def forward(self, boundary: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        normalized = self.normalized_coefficients(boundary, mask)
        return self.coefficient_mean + self.coefficient_scale * normalized

    def field_from_coefficients(self, coefficients: torch.Tensor,
                                points: torch.Tensor) -> torch.Tensor:
        return evaluate_total_degree_chebyshev(coefficients, points, self.indices)

    def field(self, boundary: torch.Tensor, points: torch.Tensor,
              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.field_from_coefficients(self(boundary, mask), points)
