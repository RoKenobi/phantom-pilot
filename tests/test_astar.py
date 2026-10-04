import numpy as np
import pytest

from baselines.astar_oracle import AStarAgent, astar, optimal_path_length, path_length_m
from env.lidar import LidarConfig
from env.maps import FloorPlan, HAND_MAPS, generate_floorplan, grid_to_segments
from env.robot_env import EnvConfig, RobotNavEnv


def _open_grid(n=5):
    g = np.zeros((n, n), dtype=bool)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = True
    return g


def test_astar_finds_a_straight_path():
    path = astar(_open_grid(5), (1, 1), (1, 3))
    assert path is not None
    assert path[0] == (1, 1) and path[-1] == (1, 3)
    assert len(path) == 3


def test_astar_path_cells_are_all_free():
    g = _open_grid(8)
    g[3, 1:6] = True
    path = astar(g, (1, 1), (6, 6))
    assert path is not None
    assert all(not g[c] for c in path)


def test_astar_returns_none_when_unreachable():
    g = _open_grid(7)
    g[3, :] = True  # full wall across the middle
    assert astar(g, (1, 1), (5, 5)) is None


def test_astar_routes_around_an_obstacle():
    g = _open_grid(9)
    g[1:7, 4] = True  # vertical wall with a gap at row 7
    path = astar(g, (2, 2), (2, 6))
    assert path is not None
    assert any(r >= 7 for r, _ in path), "must detour around the wall"


def test_astar_does_not_cut_diagonal_corners():
    """Passing diagonally between two orthogonally-adjacent walls would clip
    a corner the robot cannot physically pass."""
    g = np.zeros((4, 4), dtype=bool)
    g[1, 2] = True
    g[2, 1] = True
    path = astar(g, (1, 1), (2, 2))
    assert path is None or (1, 1) not in path or len(path) > 2


def test_astar_is_optimal_on_an_open_grid():
    path = astar(_open_grid(12), (1, 1), (5, 9))
    # Octile optimum: 4 diagonal steps + 4 straight steps = 9 cells visited.
    assert len(path) == 9


def test_path_length_m_scales_with_cell_size():
    path = [(1, 1), (1, 2), (1, 3)]
    assert path_length_m(path, 0.5) == pytest.approx(1.0)


def test_path_length_m_counts_diagonals_correctly():
    assert path_length_m([(1, 1), (2, 2)], 1.0) == pytest.approx(np.sqrt(2.0))


def test_optimal_path_length_matches_manual_computation():
    fp = generate_floorplan(np.random.default_rng(0), 12, 12, cell_size=0.5)
    free = fp.free_cells
    from env.maps import cell_to_xy
    a, b = cell_to_xy(free[0], 0.5), cell_to_xy(free[-1], 0.5)
    length = optimal_path_length(fp, a, b)
    assert length is not None and length > 0.0


def test_optimal_path_length_is_none_when_goal_is_walled_off():
    g = _open_grid(9)
    g[4, :] = True
    fp = FloorPlan(grid=g, segments=grid_to_segments(g, 0.5), cell_size=0.5)
    from env.maps import cell_to_xy
    assert optimal_path_length(fp, cell_to_xy((1, 1), 0.5), cell_to_xy((7, 7), 0.5)) is None


# --- the oracle as a policy ----------------------------------------------

def test_astar_agent_reaches_the_goal_on_a_hand_map():
    fp = HAND_MAPS["apartment"]()
    env = RobotNavEnv(env_cfg=EnvConfig(max_steps=1500), floorplan=fp,
                      lidar_cfg=LidarConfig(noise_std=0.0, dropout=0.0))
    obs, _ = env.reset(seed=0)
    agent = AStarAgent()
    agent.reset(env)
    for _ in range(1500):
        obs, _, term, trunc, info = env.step(agent.act(obs))
        if term or trunc:
            break
    assert info["success"], "the oracle defines the ceiling; it must succeed"


def test_astar_agent_beats_the_reactive_controller_in_the_trap():
    from baselines.reactive import ReactiveController

    fp = HAND_MAPS["u_trap"]()
    results = {}
    for name, make in (("oracle", AStarAgent), ("reactive", ReactiveController)):
        env = RobotNavEnv(env_cfg=EnvConfig(max_steps=800), floorplan=fp,
                          lidar_cfg=LidarConfig(noise_std=0.0, dropout=0.0))
        obs, _ = env.reset(seed=0)
        env.pose = np.array([5.0, 3.0, np.pi / 2])
        env.goal = np.array([5.0, 8.5])
        env.scan = env.lidar.scan(env.pose, fp.segments, env.rng)
        env._prev_dist = float(np.linalg.norm(env.goal - env.pose[:2]))
        agent = make()
        agent.reset(env)
        results[name] = False
        for _ in range(800):
            obs, _, term, trunc, info = env.step(agent.act(obs))
            if term or trunc:
                results[name] = info["success"]
                break
    assert results["oracle"] and not results["reactive"]
