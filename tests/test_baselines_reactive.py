import numpy as np
import pytest

from baselines.base import twist_to_wheels
from baselines.random_agent import RandomAgent
from baselines.reactive import ReactiveConfig, ReactiveController
from env.lidar import LidarConfig
from env.maps import HAND_MAPS
from env.robot_env import OMEGA_MAX, WHEEL_V_MAX, EnvConfig, RobotNavEnv

QUIET = LidarConfig(noise_std=0.0, dropout=0.0)


def _env(mode="continuous", fp=None):
    return RobotNavEnv(env_cfg=EnvConfig(action_mode=mode), floorplan=fp, lidar_cfg=QUIET)


# --- twist_to_wheels round-trip ------------------------------------------

def test_twist_to_wheels_inverts_decode_action():
    env = _env()
    env.reset(seed=0)
    for v, omega in [(0.1, 0.0), (0.0, 1.0), (-0.05, -2.0), (0.22, 0.0)]:
        wheels = twist_to_wheels(v, omega)
        v2, omega2 = env._decode_action(wheels)
        # twist_to_wheels returns float32, so 1e-9 is below representable error.
        assert v2 == pytest.approx(v, abs=1e-6)
        assert omega2 == pytest.approx(omega, abs=1e-6)


def test_twist_to_wheels_output_is_clipped_to_unit_range():
    w = twist_to_wheels(10.0, 100.0)
    assert (np.abs(w) <= 1.0 + 1e-12).all()


# --- random --------------------------------------------------------------

def test_random_agent_continuous_actions_are_in_space():
    env = _env()
    obs, _ = env.reset(seed=0)
    agent = RandomAgent(seed=0)
    agent.reset(env)
    for _ in range(20):
        a = agent.act(obs)
        assert env.action_space.contains(a)


def test_random_agent_discrete_actions_are_valid_indices():
    env = _env("discrete")
    obs, _ = env.reset(seed=0)
    agent = RandomAgent(seed=0)
    agent.reset(env)
    for _ in range(20):
        assert 0 <= int(agent.act(obs)) < env.action_space.n


def test_random_agent_is_reproducible():
    # Separate envs: two agents sharing one action_space object would share
    # its RNG, so the second would continue the first's stream, not repeat it.
    env_a, env_b = _env(), _env()
    obs, _ = env_a.reset(seed=0)
    env_b.reset(seed=0)
    a, b = RandomAgent(seed=7), RandomAgent(seed=7)
    a.reset(env_a); b.reset(env_b)
    assert np.allclose([a.act(obs) for _ in range(5)], [b.act(obs) for _ in range(5)])


# --- reactive ------------------------------------------------------------

def test_reactive_drives_forward_when_goal_is_straight_ahead_and_clear():
    env = _env()
    env.reset(seed=0)
    env.pose = np.array([5.0, 5.0, 0.0])
    env.goal = np.array([8.0, 5.0])
    env.scan = np.full(24, env.lidar.cfg.max_range)   # open space, no repulsion
    agent = ReactiveController()
    agent.reset(env)
    v, omega = env._decode_action(agent.act(env._observation()))
    assert v > 0.05
    assert abs(omega) < 0.3


def test_reactive_turns_toward_a_goal_on_its_left():
    env = _env()
    env.reset(seed=0)
    env.pose = np.array([5.0, 5.0, 0.0])
    env.goal = np.array([5.0, 8.0])
    env.scan = np.full(24, env.lidar.cfg.max_range)   # open space, no repulsion
    agent = ReactiveController()
    agent.reset(env)
    _, omega = env._decode_action(agent.act(env._observation()))
    assert omega > 0.2, "positive omega is a left turn"


def test_reactive_steers_away_from_a_close_wall():
    """With a wall hard on the right, the repulsive term must push left."""
    env = _env()
    env.reset(seed=0)
    env.pose = np.array([5.0, 5.0, 0.0])
    env.goal = np.array([9.0, 5.0])
    env.floorplan.segments = np.array([[0.0, 4.8, 10.0, 4.8]])  # wall just to the right
    env.scan = env.lidar.scan(env.pose, env.floorplan.segments, env.rng)
    agent = ReactiveController()
    agent.reset(env)
    _, omega = env._decode_action(agent.act(env._observation()))
    assert omega > 0.0, "must veer left, away from the wall on the right"


def test_reactive_slows_down_when_boxed_in():
    env = _env()
    env.reset(seed=0)
    env.pose = np.array([5.0, 5.0, 0.0])
    env.goal = np.array([9.0, 5.0])
    agent = ReactiveController()
    agent.reset(env)

    env.scan = np.full(24, env.lidar.cfg.max_range)
    v_open, _ = env._decode_action(agent.act(env._observation()))
    env.scan = np.full(24, 0.2)
    v_blocked, _ = env._decode_action(agent.act(env._observation()))
    assert v_blocked < v_open


def test_reactive_gets_trapped_in_the_concave_pocket():
    """The headline baseline failure: potential fields have local minima.

    This test IS the evidence cited in the README, so it must be
    deterministic -- hand map, no sensor noise, fixed start and goal.
    """
    fp = HAND_MAPS["u_trap"]()
    env = RobotNavEnv(env_cfg=EnvConfig(max_steps=400), floorplan=fp, lidar_cfg=QUIET)
    obs, _ = env.reset(seed=0)
    # Place the robot inside the pocket, goal outside and beyond the closed end.
    env.pose = np.array([5.0, 3.0, np.pi / 2])
    env.goal = np.array([5.0, 8.5])
    env.scan = env.lidar.scan(env.pose, fp.segments, env.rng)
    env._prev_dist = float(np.linalg.norm(env.goal - env.pose[:2]))

    agent = ReactiveController()
    agent.reset(env)
    succeeded = False
    for _ in range(400):
        obs, _, term, trunc, info = env.step(agent.act(obs))
        if term or trunc:
            succeeded = info["success"]
            break
    assert not succeeded, "the reactive baseline must fail here -- that is the point"
