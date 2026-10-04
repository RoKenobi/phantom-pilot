"""Simulated 2D LiDAR, modelled on the TurtleBot3's LDS-01.

`cast_rays` is fully vectorised: all rays are intersected against all wall
segments in a single NumPy broadcast. This is the throughput bottleneck for
training, so it must never become a Python loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


def cast_rays(
    origin: np.ndarray,
    angles: np.ndarray,
    segments: np.ndarray,
    max_range: float,
) -> np.ndarray:
    """Distance from `origin` to the nearest wall along each ray.

    Args:
        origin: (2,) ray origin in metres.
        angles: (N,) absolute world-frame ray bearings in radians.
        segments: (M, 4) wall segments as [x1, y1, x2, y2].
        max_range: returned where a ray hits nothing.

    Returns:
        (N,) distances in metres, clipped to `max_range`.
    """
    angles = np.asarray(angles, dtype=np.float64)
    n = angles.shape[0]
    if segments.size == 0:
        return np.full(n, max_range, dtype=np.float64)

    ox, oy = float(origin[0]), float(origin[1])
    dx = np.cos(angles)[:, None]          # (N, 1)
    dy = np.sin(angles)[:, None]

    x1, y1, x2, y2 = (segments[:, i][None, :] for i in range(4))  # (1, M)
    ex, ey = x2 - x1, y2 - y1             # segment direction

    # Solve  o + t*d = p + u*e  for t (along ray) and u (along segment).
    denom = dx * ey - dy * ex             # (N, M); zero when parallel
    parallel = np.abs(denom) < _EPS
    safe = np.where(parallel, 1.0, denom)

    px, py = x1 - ox, y1 - oy
    t = (px * ey - py * ex) / safe
    u = (px * dy - py * dx) / safe

    valid = (~parallel) & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
    t = np.where(valid, t, np.inf)

    nearest = t.min(axis=1)
    return np.minimum(nearest, max_range)


@dataclass
class LidarConfig:
    n_rays: int = 24
    fov: float = np.deg2rad(270.0)
    max_range: float = 3.5     # LDS-01
    min_range: float = 0.12    # LDS-01
    noise_std: float = 0.01    # fraction of the measured range
    dropout: float = 0.02      # probability a ray returns max_range


class Lidar:
    def __init__(self, cfg: LidarConfig | None = None) -> None:
        self.cfg = cfg or LidarConfig()

    @property
    def ray_angles(self) -> np.ndarray:
        """Ray bearings relative to the robot heading, spanning the FOV."""
        c = self.cfg
        return np.linspace(-c.fov / 2.0, c.fov / 2.0, c.n_rays)

    def scan(
        self, pose: np.ndarray, segments: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Noisy ranges in metres for a robot at pose (x, y, theta)."""
        c = self.cfg
        world_angles = self.ray_angles + float(pose[2])
        d = cast_rays(pose[:2], world_angles, segments, c.max_range)

        if c.noise_std > 0.0:
            d = d + rng.normal(0.0, c.noise_std * np.maximum(d, c.min_range))
        if c.dropout > 0.0:
            lost = rng.random(d.shape) < c.dropout
            d = np.where(lost, c.max_range, d)

        return np.clip(d, c.min_range, c.max_range)
