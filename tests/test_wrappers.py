import numpy as np
import pytest

from env.robot_env import OBS_DIM, EnvConfig
from env.wrappers import FrameStack, make_env, make_vec_env


def _wrapper_chain(env):
    """Every wrapper from the outside in."""
    chain = []
    while hasattr(env, "env"):
        chain.append(env)
        env = env.env
    return chain


def test_frame_stack_expands_observation_dim():
    obs, _ = FrameStack(make_env(EnvConfig())(), k=4).reset(seed=0)
    assert obs.shape == (4 * OBS_DIM,)


def test_frame_stack_repeats_initial_frame_on_reset():
    obs, _ = FrameStack(make_env(EnvConfig())(), k=3).reset(seed=0)
    frames = obs.reshape(3, OBS_DIM)
    assert np.allclose(frames[0], frames[1])
    assert np.allclose(frames[1], frames[2])


def test_frame_stack_newest_frame_is_last():
    env = FrameStack(make_env(EnvConfig())(), k=2)
    env.reset(seed=0)
    obs = env.step(np.array([1.0, 1.0], dtype=np.float32))[0]
    old, new = obs.reshape(2, OBS_DIM)
    assert not np.allclose(old, new)
    assert new[27] == pytest.approx(1.0, abs=1e-6), "v_prev lives in the newest frame"


def test_frame_stack_k_of_one_is_a_passthrough():
    obs, _ = FrameStack(make_env(EnvConfig())(), k=1).reset(seed=0)
    assert obs.shape == (OBS_DIM,)


def test_make_env_returns_a_thunk_producing_fresh_envs():
    thunk = make_env(EnvConfig())
    assert callable(thunk) and thunk() is not thunk()


def test_make_env_applies_frame_stack():
    obs, _ = make_env(EnvConfig(), frame_stack=4)().reset(seed=0)
    assert obs.shape == (4 * OBS_DIM,)


def test_make_env_reward_normalisation_is_off_by_default():
    """Evaluation must see raw rewards, or reported returns are meaningless."""
    env = make_env(EnvConfig())()
    assert not any("Normalize" in type(w).__name__
                   for w in _wrapper_chain(env)), "eval envs must not normalise"


def test_make_env_can_normalise_rewards_for_training():
    env = make_env(EnvConfig(), normalize_reward=True)()
    names = [type(w).__name__ for w in _wrapper_chain(env)]
    assert any("NormalizeReward" in n for n in names)


def test_normalised_rewards_differ_from_raw():
    raw = make_env(EnvConfig())()
    nrm = make_env(EnvConfig(), normalize_reward=True)()
    raw.reset(seed=0); nrm.reset(seed=0)
    a = np.array([1.0, 0.8], dtype=np.float32)
    r_raw = [raw.step(a)[1] for _ in range(5)]
    r_nrm = [nrm.step(a)[1] for _ in range(5)]
    assert not np.allclose(r_raw, r_nrm)


def test_vec_env_steps_all_sub_envs():
    venv = make_vec_env(4, env_cfg=EnvConfig())
    obs, _ = venv.reset(seed=0)
    assert obs.shape == (4, OBS_DIM)
    obs, rew, term, trunc, _ = venv.step(np.zeros((4, 2), dtype=np.float32))
    assert obs.shape == (4, OBS_DIM)
    assert rew.shape == (4,) and term.shape == (4,) and trunc.shape == (4,)
    venv.close()


def test_vec_env_sub_envs_have_different_maps():
    venv = make_vec_env(3, env_cfg=EnvConfig(), seed=0)
    obs, _ = venv.reset(seed=0)
    assert not (np.allclose(obs[0], obs[1]) and np.allclose(obs[1], obs[2]))
    venv.close()
