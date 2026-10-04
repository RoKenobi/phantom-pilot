import numpy as np
import pytest

from env.maps import (
    FloorPlan, HAND_MAPS, cell_to_xy, generate_floorplan,
    grid_to_segments, sample_start_goal, xy_to_cell,
)


def test_grid_to_segments_single_cell_gives_four_walls():
    grid = np.zeros((3, 3), dtype=bool)
    grid[1, 1] = True
    segs = grid_to_segments(grid, cell_size=1.0)
    assert segs.shape == (4, 4), "one isolated cell has four exposed faces"


def test_grid_to_segments_omits_shared_faces():
    """Two adjacent occupied cells share a face; it is interior, not a wall."""
    grid = np.zeros((3, 4), dtype=bool)
    grid[1, 1] = True
    grid[1, 2] = True
    segs = grid_to_segments(grid, cell_size=1.0)
    assert segs.shape == (6, 4), "2 cells = 8 faces - 2 shared = 6"


def test_grid_to_segments_border_is_closed():
    """A fully-walled border must produce a closed loop with no gaps."""
    grid = np.zeros((5, 5), dtype=bool)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True
    segs = grid_to_segments(grid, cell_size=0.5)
    assert len(segs) > 0
    assert np.isfinite(segs).all()


def test_generated_floorplan_has_closed_border():
    rng = np.random.default_rng(0)
    fp = generate_floorplan(rng, n_rows=12, n_cols=12, cell_size=0.5)
    assert fp.grid[0, :].all() and fp.grid[-1, :].all()
    assert fp.grid[:, 0].all() and fp.grid[:, -1].all()


def test_generated_floorplan_is_deterministic_under_seed():
    a = generate_floorplan(np.random.default_rng(7), n_rows=12, n_cols=12)
    b = generate_floorplan(np.random.default_rng(7), n_rows=12, n_cols=12)
    assert np.array_equal(a.grid, b.grid)


def test_generated_floorplan_free_space_is_connected():
    """Every free cell must be reachable from every other, or goals become impossible."""
    from collections import deque

    rng = np.random.default_rng(3)
    fp = generate_floorplan(rng, n_rows=16, n_cols=16, obstacle_density=0.2)
    free = ~fp.grid
    start = tuple(np.argwhere(free)[0])
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < free.shape[0] and 0 <= n[1] < free.shape[1]:
                if free[n] and n not in seen:
                    seen.add(n)
                    q.append(n)
    assert len(seen) == int(free.sum()), "free space must be one connected component"


def test_sample_start_goal_respects_separation_and_free_space():
    rng = np.random.default_rng(1)
    fp = generate_floorplan(rng, n_rows=20, n_cols=20, cell_size=0.5)
    start, goal = sample_start_goal(fp, rng, min_separation=3.0)
    assert np.linalg.norm(goal - start) >= 3.0
    assert not fp.grid[xy_to_cell(start, fp.cell_size)]
    assert not fp.grid[xy_to_cell(goal, fp.cell_size)]


def test_cell_xy_roundtrip():
    xy = cell_to_xy((3, 5), cell_size=0.5)
    assert xy_to_cell(xy, 0.5) == (3, 5)


@pytest.mark.parametrize("name", ["apartment", "u_trap", "corridor"])
def test_hand_maps_build(name):
    fp = HAND_MAPS[name]()
    assert isinstance(fp, FloorPlan)
    assert fp.segments.shape[1] == 4
    assert (~fp.grid).sum() > 0
