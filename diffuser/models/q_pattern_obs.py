from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransitionQ(nn.Module):
    """
    Transition critic for obs-trajectory diffusion + inverse dynamics:

        Q_tau(h_t, s_{t+1}) -> scalar

    where h_t is a history window of observations (including s_t).
    """

    def __init__(
        self,
        observation_dim: int,
        n_agents: int,
        history_horizon: int,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.n_agents = n_agents
        self.history_horizon = history_horizon

        # h_obs: [B, H+1, A, D] and s_tp1: [B, A, D]
        in_dim = (history_horizon + 1) * n_agents * observation_dim + n_agents * observation_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_obs: torch.Tensor, s_tp1: torch.Tensor) -> torch.Tensor:
        b = h_obs.shape[0]
        x = torch.cat([h_obs.reshape(b, -1), s_tp1.reshape(b, -1)], dim=-1)
        return self.net(x)


class PatternEncoderObs(nn.Module):
    """
    Encode the *full* obs diffusion chain (DDIM steps) into a pattern latent.

    Input:
        diffusion_chain_obs: [B, K, T, A, D_obs]
    Output:
        pattern_latent: [B, P]
    """

    def __init__(
        self,
        observation_dim: int,
        n_agents: int,
        horizon_total: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.n_agents = n_agents
        self.horizon_total = horizon_total
        self.latent_dim = latent_dim

        step_in = horizon_total * n_agents * observation_dim
        self.step_mlp = nn.Sequential(
            nn.Linear(step_in, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, diffusion_chain_obs: torch.Tensor) -> torch.Tensor:
        # diffusion_chain_obs: [B, K, T, A, D]
        b, k, t, a, d = diffusion_chain_obs.shape
        assert (
            t == self.horizon_total and a == self.n_agents and d == self.observation_dim
        ), f"Unexpected diffusion_chain_obs shape {diffusion_chain_obs.shape}"
        x = diffusion_chain_obs.reshape(b, k, -1)  # [B, K, T*A*D]
        z = self.step_mlp(x)  # [B, K, P]
        return z.mean(dim=1)  # [B, P]


class PatternQ(nn.Module):
    """
    Student critic operating on history + pattern latent:

        Q_pattern(h_t, z_pattern) -> scalar
    """

    def __init__(
        self,
        observation_dim: int,
        n_agents: int,
        history_horizon: int,
        pattern_latent_dim: int = 64,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.n_agents = n_agents
        self.history_horizon = history_horizon
        self.pattern_latent_dim = pattern_latent_dim

        in_dim = (history_horizon + 1) * n_agents * observation_dim + pattern_latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_obs: torch.Tensor, pattern_latent: torch.Tensor) -> torch.Tensor:
        b = h_obs.shape[0]
        x = torch.cat([h_obs.reshape(b, -1), pattern_latent], dim=-1)
        return self.net(x)

    @staticmethod
    def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


