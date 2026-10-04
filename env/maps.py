"""Occupancy-grid floor plans and their wall-segment representation.

A plain GridWorld would be enough here. This module keeps the map a genuine
occupancy grid -- which the A* oracle plans on directly -- while exposing
the same geometry as line segments so the robot can move continuously
through it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class FloorPlan:
    grid: np.ndarray       # (H, W) bool, True = occupied
    segments: np.ndarray   # (M, 4) float64, [x1, y1, x2, y2] in metres
    cell_size: float

    @property
    def free_cells(self) -> np.ndarray:
        return np.argwhere(~self.grid)

    @property
    def extent(self) -> tuple[float, float]:
        h, w = self.grid.shape
        return w * self.cell_size, h * self.cell_size


def cell_to_xy(rc, cell_size: float) -> np.ndarray:
    """Centre of cell (row, col) in metres. x follows col, y follows row."""
    r, c = rc
    return np.array([(c + 0.5) * cell_size, (r + 0.5) * cell_size], dtype=np.float64)


def xy_to_cell(xy, cell_size: float) -> tuple[int, int]:
    x, y = float(xy[0]), float(xy[1])
    return int(y // cell_size), int(x // cell_size)


def grid_to_segments(grid: np.ndarray, cell_size: float) -> np.ndarray:
    """Exposed faces of occupied cells, as line segments.

    A face is emitted only where an occupied cell meets free space or the
    array edge; faces shared by two occupied cells are interior and skipped.
    """
    h, w = grid.shape
    segs: list[tuple[float, float, float, float]] = []
    occ = np.argwhere(grid)
    for r, c in occ:
        x0, y0 = c * cell_size, r * cell_size
        x1, y1 = x0 + cell_size, y0 + cell_size
        # (neighbour_row, neighbour_col, face endpoints)
        faces = (
            (r - 1, c, (x0, y0, x1, y0)),  # top
            (r + 1, c, (x0, y1, x1, y1)),  # bottom
            (r, c - 1, (x0, y0, x0, y1)),  # left
            (r, c + 1, (x1, y0, x1, y1)),  # right
        )
        for nr, nc, seg in faces:
            outside = not (0 <= nr < h and 0 <= nc < w)
            if outside or not grid[nr, nc]:
                segs.append(seg)
    return np.asarray(segs, dtype=np.float64).reshape(-1, 4)


def _largest_free_component(grid: np.ndarray) -> np.ndarray:
    """Return a copy of grid with every free cell outside the largest free
    component filled in, guaranteeing a single connected free space."""
    free = ~grid
    h, w = grid.shape
    seen = np.zeros_like(free)
    best: list[tuple[int, int]] = []
    for start in map(tuple, np.argwhere(free)):
        if seen[start]:
            continue
        comp, q = [], deque([start])
        seen[start] = True
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and free[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        if len(comp) > len(best):
            best = comp
    out = np.ones_like(grid)
    for r, c in best:
        out[r, c] = False
    return out


def generate_floorplan(
    rng: np.random.Generator,
    n_rows: int = 20,
    n_cols: int = 20,
    cell_size: float = 0.5,
    obstacle_density: float = 0.15,
) -> FloorPlan:
    """Random interior obstacles inside a closed border, pruned to one
    connected free component."""
    grid = np.zeros((n_rows, n_cols), dtype=bool)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True

    interior = (n_rows - 2) * (n_cols - 2)
    n_blocks = int(interior * obstacle_density)
    rows = rng.integers(1, n_rows - 1, size=n_blocks)
    cols = rng.integers(1, n_cols - 1, size=n_blocks)
    grid[rows, cols] = True

    grid = _largest_free_component(grid)
    return FloorPlan(grid=grid, segments=grid_to_segments(grid, cell_size), cell_size=cell_size)


def sample_start_goal(
    fp: FloorPlan, rng: np.random.Generator, min_separation: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Two free-cell centres at least `min_separation` metres apart.

    Falls back to the farthest available pair if the map is too small to
    satisfy the requested separation.
    """
    free = fp.free_cells
    pts = np.stack([cell_to_xy(rc, fp.cell_size) for rc in free])
    for _ in range(200):
        i, j = rng.integers(0, len(pts), size=2)
        if np.linalg.norm(pts[j] - pts[i]) >= min_separation:
            return pts[i].copy(), pts[j].copy()
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    return pts[i].copy(), pts[j].copy()


def _from_ascii(rows: list[str], cell_size: float = 0.5) -> FloorPlan:
    grid = np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)
    return FloorPlan(grid=grid, segments=grid_to_segments(grid, cell_size), cell_size=cell_size)


def _apartment() -> FloorPlan:
    return _from_ascii([
        "####################",
        "#........#.........#",
        "#........#.........#",
        "#........#.........#",
        "#...................",   # doorway in the right wall
        "#........#.........#",
        "####.#####.........#",
        "#........#.........#",
        "#........###.#######",
        "#..................#",
        "#..................#",
        "#####.##############",
        "#..................#",
        "#..........#.......#",
        "#..........#.......#",
        "#..........#.......#",
        "#..........#.......#",
        "#..........#.......#",
        "#..................#",
        "####################",
    ])


def _u_trap() -> FloorPlan:
    """A concave pocket. The reactive potential-field baseline gets stuck here."""
    return _from_ascii([
        "####################",
        "#..................#",
        "#..................#",
        "#.......######.....#",
        "#.......#....#.....#",
        "#.......#....#.....#",
        "#.......#....#.....#",
        "#.......#....#.....#",
        "#.......#....#.....#",
        "#..................#",
        "#..................#",
        "#.....##########...#",
        "#.....#........#...#",
        "#.....#........#...#",
        "#.....#........#...#",
        "#..................#",
        "#..................#",
        "#..................#",
        "#..................#",
        "####################",
    ])


def _corridor() -> FloorPlan:
    """Long narrow corridors -- the perceptual-aliasing stress test."""
    return _from_ascii([
        "####################",
        "#..................#",
        "####################".replace("#", "#", 1),
        "#..................#",
        "##################.#",
        "#..................#",
        "#.##################",
        "#..................#",
        "##################.#",
        "#..................#",
        "#.##################",
        "#..................#",
        "##################.#",
        "#..................#",
        "#.##################",
        "#..................#",
        "##################.#",
        "#..................#",
        "#..................#",
        "####################",
    ])


HAND_MAPS: dict[str, Callable[[], FloorPlan]] = {
    "apartment": _apartment,
    "u_trap": _u_trap,
    "corridor": _corridor,
}
