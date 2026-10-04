"""Differential-drive robot navigating a 2D floor plan from LiDAR alone.

The agent never sees the map. Its entire view of the world is 24 noisy range
readings plus its own bearing and distance to the goal, which makes this a
POMDP rather than the fully-observed MDP a GridWorld would give.
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.dynamic import PedestrianCrowd
from env.lidar import Lidar, LidarConfig
from env.maps import FloorPlan, generate_floorplan, sample_start_goal
from env.rewards import RewardConfig, compute_reward

# TurtleBot3 Burger physical constants.
ROBOT_RADIUS = 0.105   # m
WHEELBASE = 0.16       # m
WHEEL_V_MAX = 0.22     # m/s, per wheel
DT = 0.1               # s

OBS_DIM = 29           # 24 ranges + d_goal + sin(psi) + cos(psi) + v_prev + w_prev
OMEGA_MAX = 2.0 * WHEEL_V_MAX / WHEELBASE   # 2.75 rad/s

# Normalised wheel-velocity pairs (v_L, v_R).
DISCRETE_ACTIONS = np.array([
    [1.0, 1.0],     # 0 forward
    [0.5, 1.0],     # 1 forward-left
    [1.0, 0.5],     # 2 forward-right
    [-1.0, 1.0],    # 3 spin left
    [1.0, -1.0],    # 4 spin right
    [0.0, 0.0],     # 5 stop
], dtype=np.float64)

_EPS = 1e-12


def min_distance_to_segments(point: np.ndarray, segments: np.ndarray) -> float:
    """Exact Euclidean distance from a point to the nearest wall segment.

    Collision detection uses this rather than the LiDAR scan, because the scan
    is noisy, angularly sparse, and can slip a wall between two rays.
    """
    if segments.size == 0:
        return float("inf")
    a, b = segments[:, :2], segments[:, 2:]
    ab = b - a
    ap = point[None, :] - a
    denom = (ab * ab).sum(axis=1)
    t = np.where(denom > _EPS, (ap * ab).sum(axis=1) / np.maximum(denom, _EPS), 0.0)
    closest = a + np.clip(t, 0.0, 1.0)[:, None] * ab
    return float(np.linalg.norm(point[None, :] - closest, axis=1).min())


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class EnvConfig:
    n_rays: int = 24
    max_steps: int = 500
    goal_radius: float = 0.25
    action_mode: str = "continuous"   # "continuous" | "discrete"
    min_separation: float = 3.0
    map_rows: int = 20
    map_cols: int = 20
    cell_size: float = 0.5
    obstacle_density: float = 0.15
    n_pedestrians: int = 0


class RobotNavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        env_cfg: EnvConfig | None = None,
        reward_cfg: RewardConfig | None = None,
        lidar_cfg: LidarConfig | None = None,
        floorplan: FloorPlan | None = None,
    ) -> None:
        self.cfg = env_cfg or EnvConfig()
        self.reward_cfg = reward_cfg or RewardConfig()
        self.lidar = Lidar(lidar_cfg or LidarConfig(n_rays=self.cfg.n_rays))
        self._fixed_floorplan = floorplan

        n = self.lidar.cfg.n_rays
        self.observation_space = spaces.Box(
            low=np.concatenate([np.zeros(n), [0.0, -1.0, -1.0, -1.0, -1.0]]).astype(np.float32),
            high=np.ones(n + 5, dtype=np.float32),
            dtype=np.float32,
        )
        if self.cfg.action_mode == "continuous":
            self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        elif self.cfg.action_mode == "discrete":
            self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))
        else:
            raise ValueError(f"unknown action_mode: {self.cfg.action_mode!r}")

        self.rng = np.random.default_rng(0)
        self.floorplan: FloorPlan | None = None
        self.pose = np.zeros(3)
        self.start = np.zeros(2)
        self.goal = np.zeros(2)
        self.scan = np.zeros(n)
        self.trajectory: list[np.ndarray] = []
        self.crowd: PedestrianCrowd | None = None
        self._prev_v = 0.0
        self._prev_omega = 0.0

    # -- helpers ----------------------------------------------------------

    @property
    def _diag(self) -> float:
        w, h = self.floorplan.extent
        return float(np.hypot(w, h))

    def _all_segments(self) -> np.ndarray:
        if self.crowd is None:
            return self.floorplan.segments
        return np.concatenate([self.floorplan.segments, self.crowd.segments()])

    def _dist_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal - self.pose[:2]))

    def _observation(self) -> np.ndarray:
        c = self.lidar.cfg
        bearing = np.arctan2(self.goal[1] - self.pose[1], self.goal[0] - self.pose[0])
        psi = wrap_to_pi(bearing - self.pose[2])
        obs = np.concatenate([
            self.scan / c.max_range,
            [
                np.clip(self._dist_to_goal() / self._diag, 0.0, 1.0),
                np.sin(psi),
                np.cos(psi),
                self._prev_v / WHEEL_V_MAX,
                self._prev_omega / OMEGA_MAX,
            ],
        ])
        return obs.astype(np.float32)

    def _decode_action(self, action) -> tuple[float, float]:
        """Map an action to body twist (v, omega) through wheel velocities."""
        if self.cfg.action_mode == "discrete":
            wheels = DISCRETE_ACTIONS[int(action)]
        else:
            wheels = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        v_l, v_r = wheels * WHEEL_V_MAX
        return (v_r + v_l) / 2.0, (v_r - v_l) / WHEELBASE

    # -- gymnasium API ----------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self._fixed_floorplan is not None:
            self.floorplan = self._fixed_floorplan
        else:
            self.floorplan = generate_floorplan(
                self.rng, self.cfg.map_rows, self.cfg.map_cols,
                self.cfg.cell_size, self.cfg.obstacle_density,
            )

        self.crowd = (PedestrianCrowd(self.cfg.n_pedestrians, self.floorplan, self.rng)
                      if self.cfg.n_pedestrians > 0 else None)

        start, goal = sample_start_goal(self.floorplan, self.rng, self.cfg.min_separation)
        self.start, self.goal = start.copy(), goal
        self.pose = np.array([start[0], start[1], self.rng.uniform(-np.pi, np.pi)])

        self._prev_v = self._prev_omega = 0.0
        self._steps = 0
        self._path_length = 0.0
        self._min_clearance = np.inf
        self._prev_dist = self._dist_to_goal()
        self.trajectory = [self.pose.copy()]
        self.scan = self.lidar.scan(self.pose, self._all_segments(), self.rng)

        return self._observation(), {"start": self.start.copy(), "goal": self.goal.copy()}

    def step(self, action):
        v, omega = self._decode_action(action)

        x, y, th = self.pose
        self.pose = np.array([
            x + v * np.cos(th) * DT,
            y + v * np.sin(th) * DT,
            wrap_to_pi(th + omega * DT),
        ])
        self.trajectory.append(self.pose.copy())

        self._steps += 1
        self._path_length += abs(v) * DT

        if self.crowd is not None:
            self.crowd.step(DT)
        clearance = min_distance_to_segments(self.pose[:2], self.floorplan.segments)
        if self.crowd is not None:
            clearance = min(clearance, self.crowd.min_distance_to(self.pose[:2]))
        self._min_clearance = min(self._min_clearance, clearance)
        self.scan = self.lidar.scan(self.pose, self._all_segments(), self.rng)

        dist = self._dist_to_goal()
        collided = clearance < ROBOT_RADIUS
        reached = (not collided) and dist < self.cfg.goal_radius

        terms = compute_reward(
            self.reward_cfg,
            prev_dist=self._prev_dist, dist=dist,
            min_scan=float(self.scan.min()),
            omega=omega, prev_omega=self._prev_omega,
            reached=reached, collided=collided,
        )

        terminated = bool(reached or collided)
        truncated = bool((not terminated) and self._steps >= self.cfg.max_steps)

        self._prev_dist = dist
        self._prev_v, self._prev_omega = v, omega

        info = {
            "success": bool(reached),
            "collision": bool(collided),
            "timeout": bool(truncated),
            "path_length": self._path_length,
            "min_clearance": float(self._min_clearance),
            "steps": self._steps,
            "dist_to_goal": dist,
            "reward_terms": terms.as_dict(),
        }
        return self._observation(), float(terms.total), terminated, truncated, info
