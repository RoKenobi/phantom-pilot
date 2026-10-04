"""Observation stacking and vector-env plumbing.

Frame stacking is the cheap answer to the partial observability that
LiDAR-only sensing introduces: a single scan cannot reveal the robot's
velocity, nor tell two identical corridors apart.
"""
from __future__ import annotations

from collections import deque
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.lidar import LidarConfig
from env.rewards import RewardConfig
from env.robot_env import EnvConfig, RobotNavEnv


class FrameStack(gym.Wrapper):
    """Concatenate the last `k` observations, oldest first."""

    def __init__(self, env: gym.Env, k: int = 4) -> None:
        super().__init__(env)
        self.k = int(k)
        self._frames: deque[np.ndarray] = deque(maxlen=self.k)
        self.observation_space = spaces.Box(
            low=np.tile(env.observation_space.low, self.k),
            high=np.tile(env.observation_space.high, self.k),
            dtype=np.float32,
        )

    def _stacked(self) -> np.ndarray:
        return np.concatenate(list(self._frames)).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self._frames.append(obs)
        return self._stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._stacked(), reward, terminated, truncated, info


def make_env(
    env_cfg: EnvConfig | None = None,
    reward_cfg: RewardConfig | None = None,
    lidar_cfg: LidarConfig | None = None,
    floorplan=None,
    frame_stack: int = 1,
    seed: int | None = None,
    normalize_reward: bool = False,
) -> Callable[[], gym.Env]:
    """Build a thunk that constructs one environment, for vector envs.

    `normalize_reward` is for TRAINING ONLY. Without it PPO does not learn:
    raw returns make the value loss two orders of magnitude larger than the
    policy loss, and because `clip_grad_norm_` clips the *global* norm, the
    critic's gradient consumes the whole budget and the policy barely moves
    (measured: policy grad 0.21 vs value grad 13.6, everything scaled by
    0.037). It must stay off during evaluation, or the reported episode
    returns are in normalised units and not comparable to the baselines.
    """

    def _thunk() -> gym.Env:
        env = RobotNavEnv(
            env_cfg=env_cfg or EnvConfig(),
            reward_cfg=reward_cfg,
            lidar_cfg=lidar_cfg,
            floorplan=floorplan,
        )
        if frame_stack > 1:
            env = FrameStack(env, frame_stack)
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=0.99)
            env = gym.wrappers.TransformReward(
                env, lambda r: float(np.clip(r, -10.0, 10.0))
            )
        if seed is not None:
            env.reset(seed=seed)
        return env

    return _thunk


def make_vec_env(n_envs: int, async_envs: bool = True,
                 **make_env_kwargs) -> gym.vector.VectorEnv:
    """`n_envs` parallel copies of the environment.

    `async_envs=False` keeps them in-process. That is the right choice when
    several training runs share a machine: AsyncVectorEnv spawns a process per
    environment, and with multi-threaded torch in each, a handful of lanes
    oversubscribe the CPU. Measured on this project: 767 steps/s in-process with
    `torch.set_num_threads(1)` against 199 steps/s with async envs.
    """
    base_seed = make_env_kwargs.pop("seed", None)
    thunks = [
        make_env(**make_env_kwargs, seed=None if base_seed is None else base_seed + i)
        for i in range(n_envs)
    ]
    if n_envs == 1 or not async_envs:
        return gym.vector.SyncVectorEnv(thunks)
    return gym.vector.AsyncVectorEnv(thunks)
