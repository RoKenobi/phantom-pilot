import numpy as np
import pytest

from env.dynamic import PED_RADIUS, Pedestrian, PedestrianCrowd, circle_to_segments
from env.lidar import LidarConfig
from env.maps import HAND_MAPS
from env.robot_env import DT, EnvConfig, RobotNavEnv

QUIET = LidarConfig(noise_std=0.0, dropout=0.0)


def test_circle_to_segments_forms_a_closed_polygon():
    segs = circle_to_segments(np.array([1.0, 1.0]), 0.2, n_sides=8)
    assert segs.shape == (8, 4)
    # Each segment's end is the next segment's start.
    assert np.allclose(segs[:-1, 2:], segs[1:, :2])
    assert np.allclose(segs[-1, 2:], segs[0, :2])


def test_circle_to_segments_vertices_lie_on_the_circle():
    c, r = np.array([2.0, 3.0]), 0.25
    segs = circle_to_segments(c, r, n_sides=12)
    assert np.allclose(np.linalg.norm(segs[:, :2] - c, axis=1), r)


def test_crowd_spawns_the_requested_number_in_free_space():
    fp = HAND_MAPS["apartment"]()
    crowd = PedestrianCrowd(5, fp, np.random.default_rng(0))
    assert len(crowd.positions()) == 5
    for p in crowd.positions():
        r, c = int(p[1] // fp.cell_size), int(p[0] // fp.cell_size)
        assert not fp.grid[r, c]


def test_crowd_segments_count_matches_population():
    crowd = PedestrianCrowd(3, HAND_MAPS["apartment"](), np.random.default_rng(0))
    assert crowd.segments().shape == (3 * 8, 4)


def test_crowd_of_zero_returns_an_empty_segment_array():
    crowd = PedestrianCrowd(0, HAND_MAPS["apartment"](), np.random.default_rng(0))
    assert crowd.segments().shape == (0, 4)


def test_pedestrians_move_when_stepped():
    crowd = PedestrianCrowd(4, HAND_MAPS["apartment"](), np.random.default_rng(0))
    before = crowd.positions().copy()
    for _ in range(10):
        crowd.step(DT)
    assert not np.allclose(before, crowd.positions())


def test_pedestrians_stay_inside_free_space():
    fp = HAND_MAPS["apartment"]()
    crowd = PedestrianCrowd(6, fp, np.random.default_rng(1))
    for _ in range(500):
        crowd.step(DT)
    for p in crowd.positions():
        r, c = int(p[1] // fp.cell_size), int(p[0] // fp.cell_size)
        assert 0 <= r < fp.grid.shape[0] and 0 <= c < fp.grid.shape[1]
        assert not fp.grid[r, c], "a pedestrian walked through a wall"


def test_crowd_is_deterministic_under_seed():
    fp = HAND_MAPS["apartment"]()
    a = PedestrianCrowd(4, fp, np.random.default_rng(3))
    b = PedestrianCrowd(4, fp, np.random.default_rng(3))
    for _ in range(50):
        a.step(DT); b.step(DT)
    assert np.allclose(a.positions(), b.positions())


def test_min_distance_to_measures_from_the_pedestrian_surface():
    crowd = PedestrianCrowd(0, HAND_MAPS["apartment"](), np.random.default_rng(0))
    crowd.pedestrians = [Pedestrian(np.array([2.0, 2.0]), np.zeros(2), PED_RADIUS)]
    assert crowd.min_distance_to(np.array([3.0, 2.0])) == pytest.approx(1.0 - PED_RADIUS)


# --- integration with the env --------------------------------------------

def test_env_with_zero_pedestrians_is_unchanged():
    a = RobotNavEnv(env_cfg=EnvConfig(n_pedestrians=0), lidar_cfg=QUIET)
    b = RobotNavEnv(env_cfg=EnvConfig(), lidar_cfg=QUIET)
    oa, _ = a.reset(seed=0)
    ob, _ = b.reset(seed=0)
    assert np.allclose(oa, ob)


def test_pedestrians_appear_in_the_lidar_scan():
    env = RobotNavEnv(env_cfg=EnvConfig(n_pedestrians=8, map_rows=12, map_cols=12),
                      lidar_cfg=QUIET)
    env.reset(seed=0)
    with_peds = env.scan.copy()
    env2 = RobotNavEnv(env_cfg=EnvConfig(n_pedestrians=0, map_rows=12, map_cols=12),
                       lidar_cfg=QUIET)
    env2.reset(seed=0)
    assert not np.allclose(with_peds, env2.scan), "moving obstacles must be visible"


def test_observation_dimension_is_unchanged_by_pedestrians():
    env = RobotNavEnv(env_cfg=EnvConfig(n_pedestrians=6), lidar_cfg=QUIET)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (29,), "a policy trained without crowds must still be runnable"


def test_colliding_with_a_pedestrian_ends_the_episode():
    env = RobotNavEnv(env_cfg=EnvConfig(n_pedestrians=1, map_rows=12, map_cols=12),
                      lidar_cfg=QUIET)
    env.reset(seed=0)
    env.crowd.pedestrians[0].pos = env.pose[:2] + np.array([0.02, 0.0])
    env.crowd.pedestrians[0].vel = np.zeros(2)
    _, _, terminated, _, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert terminated and info["collision"]
