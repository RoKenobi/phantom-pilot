"""Moving obstacles -- the problem every real service robot actually faces.

Each pedestrian is drawn into the wall-segment set as an octagon, so the
LiDAR sees it exactly as it sees a wall. The observation space is therefore
identical with and without a crowd, which lets a policy trained on static
maps be evaluated on dynamic ones.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from env.maps import FloorPlan, cell_to_xy

PED_RADIUS = 0.18
_N_SIDES = 8


@dataclass
class Pedestrian:
    pos: np.ndarray
    vel: np.ndarray
    radius: float = PED_RADIUS


def circle_to_segments(centre: np.ndarray, radius: float,
                       n_sides: int = _N_SIDES) -> np.ndarray:
    """Closed polygon approximation of a circle, as line segments."""
    a = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    pts = centre[None, :] + radius * np.stack([np.cos(a), np.sin(a)], axis=1)
    nxt = np.roll(pts, -1, axis=0)
    return np.concatenate([pts, nxt], axis=1)


class PedestrianCrowd:
    def __init__(self, n: int, floorplan: FloorPlan,
                 rng: np.random.Generator, speed: float = 0.15) -> None:
        self.fp = floorplan
        self.speed = speed
        self.n = int(n)
        self.pedestrians: list[Pedestrian] = []
        self.reset(rng)

    def _is_free(self, xy: np.ndarray) -> bool:
        r, c = int(xy[1] // self.fp.cell_size), int(xy[0] // self.fp.cell_size)
        h, w = self.fp.grid.shape
        return 0 <= r < h and 0 <= c < w and not self.fp.grid[r, c]

    def reset(self, rng: np.random.Generator) -> None:
        free = self.fp.free_cells
        self.pedestrians = []
        for _ in range(self.n):
            cell = free[rng.integers(0, len(free))]
            theta = rng.uniform(-np.pi, np.pi)
            self.pedestrians.append(Pedestrian(
                pos=cell_to_xy(cell, self.fp.cell_size),
                vel=self.speed * np.array([np.cos(theta), np.sin(theta)]),
            ))

    def step(self, dt: float) -> None:
        """Constant velocity, reflecting off anything it would walk into."""
        for p in self.pedestrians:
            nxt = p.pos + p.vel * dt
            if self._is_free(nxt):
                p.pos = nxt
                continue
            # Try each axis separately; reverse entirely if both are blocked.
            slide_x = p.pos + np.array([p.vel[0], 0.0]) * dt
            slide_y = p.pos + np.array([0.0, p.vel[1]]) * dt
            if self._is_free(slide_x):
                p.pos, p.vel = slide_x, np.array([p.vel[0], -p.vel[1]])
            elif self._is_free(slide_y):
                p.pos, p.vel = slide_y, np.array([-p.vel[0], p.vel[1]])
            else:
                p.vel = -p.vel

    def positions(self) -> np.ndarray:
        if not self.pedestrians:
            return np.zeros((0, 2))
        return np.stack([p.pos for p in self.pedestrians])

    def segments(self) -> np.ndarray:
        if not self.pedestrians:
            return np.zeros((0, 4))
        return np.concatenate(
            [circle_to_segments(p.pos, p.radius) for p in self.pedestrians]
        )

    def min_distance_to(self, point: np.ndarray) -> float:
        if not self.pedestrians:
            return float("inf")
        pos = self.positions()
        return float((np.linalg.norm(pos - point[None, :], axis=1)
                      - np.array([p.radius for p in self.pedestrians])).min())
