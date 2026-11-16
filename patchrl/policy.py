"""Compact history encoder used by the budget-conditioned patch policy."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class QueryHistoryExtractor(BaseFeaturesExtractor):
    """Encode the patch program, explicit context, and recent black-box replies.

    Action masking is supplied by :class:`sb3_contrib.MaskablePPO`.  The GRU is
    evaluated over the fixed query-history window included in each observation,
    so it remains compatible with the masked feed-forward PPO implementation
    while representing ordered query-response evidence.
    """

    def __init__(self, observation_space, features_dim: int = 192):
        if not hasattr(observation_space, "spaces"):
            raise TypeError("QueryHistoryExtractor requires a Dict observation space")
        required = {"grid", "history", "context"}
        missing = required - set(observation_space.spaces)
        if missing:
            raise ValueError("observation space is missing: " + ", ".join(sorted(missing)))
        super().__init__(observation_space, features_dim=int(features_dim))

        grid_shape = tuple(observation_space.spaces["grid"].shape)
        history_shape = tuple(observation_space.spaces["history"].shape)
        context_shape = tuple(observation_space.spaces["context"].shape)
        if len(history_shape) != 2:
            raise ValueError("history observations must have shape (window, features)")

        grid_dim = 1
        for value in grid_shape:
            grid_dim *= int(value)
        context_dim = 1
        for value in context_shape:
            context_dim *= int(value)

        self.grid_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(grid_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 96),
            nn.SiLU(),
        )
        self.history_encoder = nn.GRU(
            input_size=int(history_shape[1]),
            hidden_size=64,
            batch_first=True,
        )
        self.context_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(context_dim, 48),
            nn.LayerNorm(48),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(96 + 64 + 48, int(features_dim)),
            nn.LayerNorm(int(features_dim)),
            nn.SiLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        grid = observations["grid"].float()
        history = observations["history"].float()
        context = observations["context"].float()
        grid_features = self.grid_encoder(grid)
        _, hidden = self.history_encoder(history)
        history_features = hidden[-1]
        context_features = self.context_encoder(context)
        return self.fusion(
            torch.cat((grid_features, history_features, context_features), dim=1)
        )


__all__ = ["QueryHistoryExtractor"]
