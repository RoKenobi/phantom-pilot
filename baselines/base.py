"""Shared policy interface, so baselines and trained agents are
interchangeable inside the evaluation harness."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from env.robot_env import WHEEL_V_MAX, WHEELBASE


@runtime_checkable
class Policy(Protocol):
    def reset(self, env) -> None: ...
    def act(self, obs: np.ndarray): ...


def twist_to_wheels(v: float, omega: float) -> np.ndarray:
    """Body twist -> normalised wheel velocities. Inverse of
    `RobotNavEnv._decode_action`, clipped into the actuator's range."""
    v_l = v - omega * WHEELBASE / 2.0
    v_r = v + omega * WHEELBASE / 2.0
    return np.clip(np.array([v_l, v_r]) / WHEEL_V_MAX, -1.0, 1.0).astype(np.float32)
