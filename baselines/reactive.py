"""Potential-field (virtual force field) local controller.

This is the classical reactive layer of a navigation stack -- our stand-in
for DWA. It sees exactly what the RL agent sees: the LiDAR scan and the goal
bearing. Its well-known weakness is local minima in concave obstacles, which
is precisely the comparison the report is built around.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baselines.base import twist_to_wheels
from env.robot_env import OMEGA_MAX, WHEEL_V_MAX, wrap_to_pi


@dataclass
class ReactiveConfig:
    k_att: float = 1.0        # pull toward the goal
    k_rep: float = 2.0        # push away from obstacles
    influence: float = 1.0    # metres; beyond this, obstacles are ignored
    k_omega: float = 1.5      # proportional heading gain
    v_gain: float = 1.0


class ReactiveController:
    def __init__(self, cfg: ReactiveConfig | None = None) -> None:
        self.cfg = cfg or ReactiveConfig()
        self._angles = None
        self._max_range = 3.5

    def reset(self, env) -> None:
        self._angles = env.lidar.ray_angles.copy()   # robot-frame bearings
        self._max_range = env.lidar.cfg.max_range

    def act(self, obs: np.ndarray):
        c = self.cfg
        scan = np.asarray(obs[: len(self._angles)], dtype=np.float64) * self._max_range
        sin_psi, cos_psi = float(obs[-4]), float(obs[-3])

        # Attractive force: unit vector toward the goal, in the robot frame.
        f = c.k_att * np.array([cos_psi, sin_psi])

        # Repulsive force: each close ray pushes back along its own bearing.
        near = scan < c.influence
        if near.any():
            d = np.maximum(scan[near], 1e-3)
            mag = c.k_rep * (1.0 / d - 1.0 / c.influence) / d
            a = self._angles[near]
            f -= np.array([(mag * np.cos(a)).sum(), (mag * np.sin(a)).sum()])

        heading_err = wrap_to_pi(float(np.arctan2(f[1], f[0])))
        omega = float(np.clip(c.k_omega * heading_err, -OMEGA_MAX, OMEGA_MAX))

        # Slow for misalignment and for a blocked path ahead.
        front = scan[len(scan) // 3: 2 * len(scan) // 3]
        clearance = float(np.clip(front.min() / c.influence, 0.0, 1.0)) if front.size else 1.0
        v = c.v_gain * WHEEL_V_MAX * max(np.cos(heading_err), 0.0) * clearance

        return twist_to_wheels(v, omega)
