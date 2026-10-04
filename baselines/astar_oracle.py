"""A* on the occupancy grid -- the global planner, and the ceiling.

Two jobs. As a *metric source* it supplies the shortest feasible path length
for every evaluation episode, which is what turns "average steps" into SPL.
As a *policy* it plans once with full map knowledge and follows the waypoints,
showing what perfect global planning looks like on the same episodes.
"""
from __future__ import annotations

import heapq

import numpy as np

from baselines.base import twist_to_wheels
from env.maps import FloorPlan, cell_to_xy, xy_to_cell
from env.robot_env import OMEGA_MAX, WHEEL_V_MAX, wrap_to_pi

_SQRT2 = float(np.sqrt(2.0))
_NEIGHBOURS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2),
]


def _octile(a: tuple[int, int], b: tuple[int, int]) -> float:
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dr + dc) + (_SQRT2 - 2.0) * min(dr, dc)


def astar(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]):
    """Shortest 8-connected path through free cells, or None."""
    h, w = grid.shape
    start, goal = tuple(start), tuple(goal)
    if grid[start] or grid[goal]:
        return None

    open_heap = [(_octile(start, goal), 0.0, start)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start: 0.0}
    closed = set()

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while path[-1] in came:
                path.append(came[path[-1]])
            return path[::-1]
        closed.add(cur)

        r, c = cur
        for dr, dc, cost in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or grid[nr, nc]:
                continue
            # Forbid squeezing diagonally between two orthogonal walls.
            if dr and dc and (grid[r + dr, c] or grid[r, c + dc]):
                continue
            ng = g + cost
            nbr = (nr, nc)
            if ng < g_score.get(nbr, np.inf):
                g_score[nbr] = ng
                came[nbr] = cur
                heapq.heappush(open_heap, (ng + _octile(nbr, goal), ng, nbr))
    return None


def path_length_m(path, cell_size: float) -> float:
    if path is None or len(path) < 2:
        return 0.0
    p = np.array(path, dtype=np.float64)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() * cell_size)


def optimal_path_length(fp: FloorPlan, start_xy, goal_xy) -> float | None:
    """Shortest feasible distance in metres, or None if unreachable."""
    path = astar(fp.grid, xy_to_cell(start_xy, fp.cell_size),
                 xy_to_cell(goal_xy, fp.cell_size))
    return None if path is None else path_length_m(path, fp.cell_size)


class AStarAgent:
    """Plans once on reset, then pure-pursuits the waypoints.

    Privileged: it reads the map and the true pose. That is deliberate -- it
    is an upper bound, not a competitor under the same information.
    """

    def __init__(self, lookahead: float = 0.2, k_omega: float = 2.5) -> None:
        self.lookahead = lookahead
        self.k_omega = k_omega
        self._env = None
        self._waypoints = np.zeros((0, 2))
        self._idx = 0

    def reset(self, env) -> None:
        self._env = env
        fp = env.floorplan
        path = astar(fp.grid, xy_to_cell(env.pose[:2], fp.cell_size),
                     xy_to_cell(env.goal, fp.cell_size))
        pts = [] if path is None else [cell_to_xy(c, fp.cell_size) for c in path]
        pts.append(np.asarray(env.goal, dtype=np.float64))
        self._waypoints = np.array(pts)
        self._idx = 0

    def act(self, obs):
        env = self._env
        pos, theta = env.pose[:2], env.pose[2]

        # Advance past waypoints already within the lookahead radius.
        while (self._idx < len(self._waypoints) - 1
               and np.linalg.norm(self._waypoints[self._idx] - pos) < self.lookahead):
            self._idx += 1
        target = self._waypoints[self._idx]

        err = wrap_to_pi(float(np.arctan2(*(target - pos)[::-1]) - theta))
        omega = float(np.clip(self.k_omega * err, -OMEGA_MAX, OMEGA_MAX))
        v = WHEEL_V_MAX * max(np.cos(err), 0.0)
        return twist_to_wheels(v, omega)
