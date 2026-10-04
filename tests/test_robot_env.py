import numpy as np
import pytest

from env.lidar import LidarConfig
from env.maps import HAND_MAPS
from env.rewards import REWARD_VARIANTS
from env.robot_env import (
    DISCRETE_ACTIONS, DT, OBS_DIM, ROBOT_RADIUS, WHEELBASE, WHEEL_V_MAX,
    EnvConfig, RobotNavEnv, min_distance_to_segments, wrap_to_pi,
)

QUIET = LidarConfig(noise_std=0.0, dropout=0.0)


def _env(**kw):
    return RobotNavEnv(env_cfg=EnvConfig(**kw), lidar_cfg=QUIET)


# --- geometry -------------------------------------------------------------

def test_min_distance_to_segments_perpendicular():
    segs = np.array([[0.0, 1.0, 4.0, 1.0]])
    assert min_distance_to_segments(np.array([2.0, 0.0]), segs) == pytest.approx(1.0)


def test_min_distance_to_segments_clamps_to_endpoint():
    segs = np.array([[0.0, 0.0, 1.0, 0.0]])
    assert min_distance_to_segments(np.array([3.0, 0.0]), segs) == pytest.approx(2.0)


def test_min_distance_to_segments_picks_nearest():
    segs = np.array([[0.0, 5.0, 4.0, 5.0], [0.0, 1.0, 4.0, 1.0]])
    assert min_distance_to_segments(np.array([2.0, 0.0]), segs) == pytest.approx(1.0)


# --- API conformance ------------------------------------------------------

def test_reset_returns_obs_and_info():
    obs, info = _env().reset(seed=0)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert isinstance(info, dict)


def test_observation_is_within_declared_space():
    env = _env()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_continuous_action_space_is_normalised_wheel_velocities():
    env = _env(action_mode="continuous")
    assert env.action_space.shape == (2,)
    assert np.allclose(env.action_space.low, -1.0)
    assert np.allclose(env.action_space.high, 1.0)


def test_discrete_action_space_has_six_actions():
    assert _env(action_mode="discrete").action_space.n == 6
    assert DISCRETE_ACTIONS.shape == (6, 2)


def test_step_returns_five_tuple():
    env = _env()
    env.reset(seed=0)
    obs, r, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert obs.shape == (OBS_DIM,)
    assert isinstance(r, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_same_seed_reproduces_identical_episode():
    a, b = _env(), _env()
    oa, _ = a.reset(seed=42)
    ob, _ = b.reset(seed=42)
    assert np.allclose(oa, ob)
    act = np.array([0.6, 0.4], dtype=np.float32)
    for _ in range(20):
        oa, ob = a.step(act)[0], b.step(act)[0]
    assert np.allclose(oa, ob)


def test_different_seeds_give_different_maps():
    env = _env()
    env.reset(seed=1)
    g1 = env.floorplan.grid.copy()
    env.reset(seed=2)
    assert not np.array_equal(g1, env.floorplan.grid)


def test_fixed_floorplan_is_not_regenerated():
    env = RobotNavEnv(env_cfg=EnvConfig(), floorplan=HAND_MAPS["u_trap"](), lidar_cfg=QUIET)
    env.reset(seed=1)
    g1 = env.floorplan.grid.copy()
    env.reset(seed=2)
    assert np.array_equal(g1, env.floorplan.grid)


# --- kinematics -----------------------------------------------------------

def test_equal_wheel_velocities_drive_straight():
    env = _env()
    env.reset(seed=0)
    env.pose = np.array([2.0, 2.0, 0.0])
    env._prev_v = env._prev_omega = 0.0
    env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert env.pose[0] == pytest.approx(2.0 + WHEEL_V_MAX * DT, abs=1e-9)
    assert env.pose[1] == pytest.approx(2.0, abs=1e-12)
    assert env.pose[2] == pytest.approx(0.0, abs=1e-12)


def test_opposite_wheel_velocities_spin_in_place():
    env = _env()
    env.reset(seed=0)
    env.pose = np.array([2.0, 2.0, 0.0])
    env._prev_v = env._prev_omega = 0.0
    env.step(np.array([-1.0, 1.0], dtype=np.float32))
    assert env.pose[0] == pytest.approx(2.0, abs=1e-12)
    assert env.pose[1] == pytest.approx(2.0, abs=1e-12)
    assert env.pose[2] == pytest.approx((2 * WHEEL_V_MAX / WHEELBASE) * DT, abs=1e-9)


def test_heading_stays_wrapped_to_pi():
    env = _env()
    env.reset(seed=0)
    env.pose = np.array([2.0, 2.0, 3.1])
    for _ in range(50):
        env.step(np.array([-1.0, 1.0], dtype=np.float32))
        assert -np.pi <= env.pose[2] <= np.pi


def test_zero_action_does_not_move_the_robot():
    env = _env()
    env.reset(seed=0)
    before = env.pose.copy()
    env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert np.allclose(env.pose, before)


def test_discrete_forward_action_moves_without_turning():
    env = _env(action_mode="discrete")
    env.reset(seed=0)
    env.pose = np.array([2.0, 2.0, 0.0])
    env._prev_v = env._prev_omega = 0.0
    env.step(0)
    assert env.pose[0] > 2.0
    assert env.pose[2] == pytest.approx(0.0, abs=1e-12)


# --- observation ----------------------------------------------------------

def test_observation_layout():
    env = _env()
    obs, _ = env.reset(seed=0)
    scan, rest = obs[:24], obs[24:]
    assert (scan >= 0.0).all() and (scan <= 1.0).all()
    assert rest.shape == (5,)
    d_goal, s, c, v_prev, w_prev = rest
    assert 0.0 <= d_goal <= 1.0
    assert s ** 2 + c ** 2 == pytest.approx(1.0, abs=1e-5)
    assert v_prev == pytest.approx(0.0) and w_prev == pytest.approx(0.0)


def test_heading_error_encoding_has_no_discontinuity():
    """sin/cos of the heading error must vary smoothly across the +-pi seam."""
    env = _env()
    env.reset(seed=0)
    env.goal = env.pose[:2] + np.array([1.0, 0.0])
    vals = []
    for theta in np.linspace(np.pi - 0.1, np.pi + 0.1, 9):
        env.pose[2] = wrap_to_pi(theta)      # sweep across the seam itself
        vals.append(env._observation()[25:27])
    steps = np.linalg.norm(np.diff(np.array(vals), axis=0), axis=1)
    assert steps.max() < 0.05, "encoding must not jump at the wrap point"


def test_previous_action_appears_in_next_observation():
    env = _env()
    env.reset(seed=0)
    obs = env.step(np.array([1.0, 1.0], dtype=np.float32))[0]
    assert obs[27] == pytest.approx(1.0, abs=1e-6)
    assert obs[28] == pytest.approx(0.0, abs=1e-6)


# --- termination ----------------------------------------------------------

def test_reaching_goal_terminates_with_success():
    env = _env()
    env.reset(seed=0)
    env.goal = env.pose[:2] + np.array([0.05, 0.0])
    _, r, terminated, truncated, info = env.step(np.array([0.1, 0.1], dtype=np.float32))
    assert terminated and not truncated
    assert info["success"] is True and r > 5.0


def test_collision_terminates_and_is_not_truncation():
    fp = HAND_MAPS["u_trap"]()
    env = RobotNavEnv(env_cfg=EnvConfig(), floorplan=fp, lidar_cfg=QUIET)
    env.reset(seed=0)
    env.pose = np.array([fp.cell_size + ROBOT_RADIUS * 0.5, 2.0, np.pi])
    _, r, terminated, truncated, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert terminated and not truncated
    assert info["collision"] is True and r < 0.0


def test_timeout_truncates_but_does_not_terminate():
    env = _env(max_steps=3)
    env.reset(seed=0)
    for _ in range(3):
        _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert truncated and not terminated
    assert info["timeout"] is True


def test_success_and_collision_are_mutually_exclusive():
    env = _env()
    env.reset(seed=0)
    for _ in range(200):
        _, _, term, trunc, info = env.step(env.action_space.sample())
        assert not (info["success"] and info["collision"])
        if term or trunc:
            break


# --- info bookkeeping -----------------------------------------------------

def test_path_length_accumulates_distance_travelled():
    env = _env()
    env.reset(seed=0)
    env.pose = np.array([2.0, 2.0, 0.0])
    total = 0.0
    for _ in range(5):
        _, _, term, trunc, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
        total = info["path_length"]
        if term or trunc:
            break
    assert total == pytest.approx(5 * WHEEL_V_MAX * DT, rel=1e-6)


def test_min_clearance_is_monotonically_non_increasing():
    env = _env()
    env.reset(seed=0)
    last = np.inf
    for _ in range(30):
        _, _, term, trunc, info = env.step(env.action_space.sample())
        assert info["min_clearance"] <= last + 1e-12
        last = info["min_clearance"]
        if term or trunc:
            break


def test_reward_terms_are_reported_in_info_and_sum_to_reward():
    env = _env()
    env.reset(seed=0)
    _, r, _, _, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert set(info["reward_terms"]) >= {
        "goal", "collision", "shaping", "clearance", "step", "jerk", "total"}
    assert info["reward_terms"]["total"] == pytest.approx(r)


def test_trajectory_is_recorded_for_rendering():
    env = _env()
    env.reset(seed=0)
    for _ in range(5):
        env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert len(env.trajectory) == 6, "initial pose plus one entry per step"


def test_sparse_reward_variant_is_silent_before_terminal():
    env = RobotNavEnv(env_cfg=EnvConfig(), reward_cfg=REWARD_VARIANTS["sparse"],
                      lidar_cfg=QUIET)
    env.reset(seed=0)
    _, r, term, _, _ = env.step(np.zeros(2, dtype=np.float32))
    if not term:
        assert r == pytest.approx(0.0)


def test_episode_always_ends():
    env = _env()
    env.reset(seed=3)
    for _ in range(600):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            break
    else:
        pytest.fail("episode never ended -- max_steps truncation is broken")
