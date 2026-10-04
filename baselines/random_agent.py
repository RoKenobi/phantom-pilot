"""Uniform random actions -- the performance floor."""
from __future__ import annotations

import numpy as np


class RandomAgent:
    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._space = None

    def reset(self, env) -> None:
        self._space = env.action_space
        if self._seed is not None:
            self._space.seed(self._seed)

    def act(self, obs):
        return self._space.sample()
