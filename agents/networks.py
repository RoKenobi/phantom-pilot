"""Shared feature extractor plus the PPO and DQN heads.

The LiDAR scan is a spatially ordered angular signal, so it goes through a
small 1D convolution rather than being flattened into an MLP. Stacked frames
become convolution channels, which keeps the parameter count flat as the
stack grows.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    """Orthogonal weights, constant bias -- the standard PPO initialisation."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ScanEncoder(nn.Module):
    """1D CNN over the range readings.

    Padding is `replicate`, not `circular`: with a 270-degree FOV the first
    and last rays are 90 degrees apart across the blind zone behind the
    robot, so wrapping them would invent an adjacency that does not exist.
    """

    def __init__(self, n_rays: int = 24, in_channels: int = 1,
                 out_dim: int = 64, padding_mode: str = "replicate") -> None:
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv1d(in_channels, 32, 5, padding=2, padding_mode=padding_mode)),
            nn.ReLU(),
            layer_init(nn.Conv1d(32, 32, 3, padding=1, padding_mode=padding_mode)),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(32 * n_rays, out_dim)),
            nn.ReLU(),
        )

    def forward(self, scan: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(scan))


class ObsSplitter:
    """Split a flat observation into its scan and non-scan parts.

    With frame stacking the observation is `k` consecutive frames laid end to
    end, each `n_rays + n_extra` wide; the scans become `k` conv channels.
    """

    def __init__(self, n_rays: int = 24, n_extra: int = 5, frame_stack: int = 1) -> None:
        self.n_rays, self.n_extra, self.k = n_rays, n_extra, frame_stack
        self.frame_dim = n_rays + n_extra
        self.obs_dim = self.frame_dim * frame_stack

    def split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = obs.shape[0]
        frames = obs.view(b, self.k, self.frame_dim)
        scan = frames[:, :, : self.n_rays]
        extra = frames[:, :, self.n_rays:].reshape(b, self.k * self.n_extra)
        return scan, extra


class _Trunk(nn.Module):
    """Scan encoder + extras, fused into one feature vector."""

    def __init__(self, n_rays: int, n_extra: int, frame_stack: int,
                 hidden: int, feat_dim: int = 64) -> None:
        super().__init__()
        self.splitter = ObsSplitter(n_rays, n_extra, frame_stack)
        self.encoder = ScanEncoder(n_rays, in_channels=frame_stack, out_dim=feat_dim)
        self.fuse = nn.Sequential(
            layer_init(nn.Linear(feat_dim + n_extra * frame_stack, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )
        self.out_dim = hidden

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        scan, extra = self.splitter.split(obs)
        return self.fuse(torch.cat([self.encoder(scan), extra], dim=1))


class ActorCritic(nn.Module):
    """Gaussian policy with a state-independent log-std, plus a value head."""

    def __init__(self, n_rays: int, n_extra: int, frame_stack: int,
                 act_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.trunk = _Trunk(n_rays, n_extra, frame_stack, hidden)
        # Gain 0.01 keeps the initial policy near the centre of the action space.
        self.mu = layer_init(nn.Linear(hidden, act_dim), std=0.01)
        self.v = layer_init(nn.Linear(hidden, 1), std=1.0)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.mu(h), self.v(h).squeeze(-1)


class QNetwork(nn.Module):
    def __init__(self, n_rays: int, n_extra: int, frame_stack: int,
                 n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.trunk = _Trunk(n_rays, n_extra, frame_stack, hidden)
        self.q = layer_init(nn.Linear(hidden, n_actions), std=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q(self.trunk(obs))


class MLPActorCritic(nn.Module):
    """Plain-MLP actor-critic for reference tasks with unstructured
    observations (Pendulum, etc.). Same interface as ActorCritic."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
        )
        self.mu = layer_init(nn.Linear(hidden, act_dim), std=0.01)
        self.v = layer_init(nn.Linear(hidden, 1), std=1.0)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.mu(h), self.v(h).squeeze(-1)


class MLPQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.ReLU(),
            layer_init(nn.Linear(hidden, hidden)), nn.ReLU(),
            layer_init(nn.Linear(hidden, n_actions), std=1.0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
