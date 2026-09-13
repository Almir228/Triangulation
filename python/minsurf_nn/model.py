"""Permutation-invariant contour encoder and coordinate-conditioned decoder."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass
class ModelConfig:
    latent_dim: int = 128
    hidden_dim: int = 128
    layers: int = 4
    frequencies: int = 5

    def __post_init__(self):
        if min(self.latent_dim, self.hidden_dim, self.layers) < 1 or self.frequencies < 0:
            raise ValueError("Model dimensions/layers must be positive; frequencies >= 0")


class ConditionalSDF(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 64), nn.Softplus(), nn.Linear(64, 128), nn.Softplus(),
            nn.Linear(128, c.latent_dim),
        )
        self.context_encoder = nn.Sequential(nn.Linear(2 * c.latent_dim, c.latent_dim), nn.Softplus())
        self.register_buffer("bands", math.pi * 2.0 ** torch.arange(c.frequencies))
        width = c.latent_dim + 3 * (1 + 2 * c.frequencies)
        layers = []
        for _ in range(c.layers):
            layers.extend((nn.Linear(width, c.hidden_dim), nn.Softplus(beta=10)))
            width = c.hidden_dim
        layers.append(nn.Linear(width, 1))
        self.decoder = nn.Sequential(*layers)

    def encode(self, boundary):
        if boundary.ndim != 3 or boundary.shape[-1] != 3 or boundary.shape[1] < 3:
            raise ValueError("boundary must have shape [batch, points >= 3, 3]")
        features = self.point_encoder(boundary)
        pooled = torch.cat((features.amax(dim=1), features.mean(dim=1)), dim=-1)
        return self.context_encoder(pooled)

    def decode(self, context, query):
        if query.ndim != 3 or query.shape[-1] != 3 or query.shape[0] != context.shape[0]:
            raise ValueError("query must have shape [batch, queries, 3]")
        phase = query.unsqueeze(-1) * self.bands
        embedded = torch.cat((query, phase.sin().flatten(-2), phase.cos().flatten(-2)), dim=-1)
        latent = context.unsqueeze(1).expand(-1, query.shape[1], -1)
        return self.decoder(torch.cat((embedded, latent), dim=-1)).squeeze(-1)

    def forward(self, boundary, query):
        return self.decode(self.encode(boundary), query)

    def config_dict(self):
        return asdict(self.config)
