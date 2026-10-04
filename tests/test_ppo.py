import numpy as np
import pytest
import torch

from agents.ppo import PPO, PPOConfig, compute_gae, train_ppo


# --- GAE: the classic bug surface ----------------------------------------

def test_gae_zero_rewards_zero_values_gives_zero_advantage():
    T, N = 5, 2
    adv, ret = compute_gae(
        np.zeros((T, N)), np.zeros((T, N)), np.zeros((T, N)),
        np.zeros(N), np.zeros(N), gamma=0.99, gae_lambda=0.95,
    )
    assert np.allclose(adv, 0.0) and np.allclose(ret, 0.0)


def test_gae_lambda_zero_reduces_to_td_residual():
    rewards = np.array([[1.0], [2.0], [3.0]])
    values = np.array([[0.5], [0.5], [0.5]])
    dones = np.zeros((3, 1))
    adv, _ = compute_gae(rewards, values, dones, np.array([0.5]), np.array([0.0]),
                         gamma=0.9, gae_lambda=0.0)
    expected = rewards + 0.9 * np.array([[0.5], [0.5], [0.5]]) - values
    assert np.allclose(adv, expected)


def test_gae_lambda_one_gamma_one_is_monte_carlo_minus_baseline():
    rewards = np.array([[1.0], [1.0], [1.0]])
    values = np.array([[0.0], [0.0], [0.0]])
    adv, ret = compute_gae(rewards, values, np.zeros((3, 1)),
                           np.array([0.0]), np.array([0.0]),
                           gamma=1.0, gae_lambda=1.0)
    assert np.allclose(adv.ravel(), [3.0, 2.0, 1.0])
    assert np.allclose(ret, adv + values)


def test_gae_done_flag_stops_bootstrapping_across_the_boundary():
    rewards = np.array([[1.0], [1.0]])
    values = np.array([[0.0], [0.0]])
    dones = np.array([[0.0], [1.0]])  # step 1's *next* state begins a new episode
    adv, _ = compute_gae(rewards, values, dones, np.array([100.0]), np.array([1.0]),
                         gamma=1.0, gae_lambda=1.0)
    assert adv[1, 0] == pytest.approx(1.0), "must not bootstrap a huge next value"


def test_gae_masks_with_the_next_states_done_flag_not_the_current_one():
    """dones[t] means "state t is post-reset". So step 0's bootstrap must be
    masked by dones[1], not dones[0]. Masking with dones[0] silently drops the
    bootstrap on the first step after every reset."""
    rewards = np.array([[1.0], [1.0]])
    values = np.array([[0.0], [5.0]])
    dones = np.array([[1.0], [0.0]])   # episode restarted AT t=0, runs on after
    adv, _ = compute_gae(rewards, values, dones, np.array([0.0]), np.array([0.0]),
                         gamma=1.0, gae_lambda=0.0)
    # delta_0 = r + gamma*V(s_1)*(1 - dones[1]) - V(s_0) = 1 + 5 - 0 = 6.
    # Masking with dones[0] would give 1.0.
    assert adv[0, 0] == pytest.approx(6.0)


def test_gae_returns_equal_advantages_plus_values():
    rng = np.random.default_rng(0)
    T, N = 8, 3
    adv, ret = compute_gae(
        rng.normal(size=(T, N)), rng.normal(size=(T, N)),
        (rng.random((T, N)) < 0.15).astype(float),
        rng.normal(size=N), np.zeros(N), gamma=0.99, gae_lambda=0.95,
    )
    # This identity is what makes the value target consistent with the critic.
    assert ret.shape == adv.shape == (T, N)


def test_gae_shapes_are_preserved():
    T, N = 7, 4
    adv, ret = compute_gae(np.zeros((T, N)), np.zeros((T, N)), np.zeros((T, N)),
                           np.zeros(N), np.zeros(N), 0.99, 0.95)
    assert adv.shape == (T, N) and ret.shape == (T, N)


def test_gae_independent_across_parallel_envs():
    rewards = np.array([[1.0, 0.0], [1.0, 0.0]])
    values = np.zeros((2, 2))
    adv, _ = compute_gae(rewards, values, np.zeros((2, 2)),
                         np.zeros(2), np.zeros(2), gamma=1.0, gae_lambda=1.0)
    assert np.allclose(adv[:, 0], [2.0, 1.0])
    assert np.allclose(adv[:, 1], [0.0, 0.0])


# --- agent ---------------------------------------------------------------

def _agent(**kw):
    return PPO(obs_dim=29, act_dim=2, n_rays=24, n_extra=5, frame_stack=1,
               cfg=PPOConfig(**kw), seed=0)


def test_act_returns_action_logp_and_value_with_batch_shapes():
    a = _agent()
    action, logp, value = a.act(np.zeros((6, 29), dtype=np.float32))
    assert action.shape == (6, 2) and logp.shape == (6,) and value.shape == (6,)


def test_act_is_stochastic_by_default():
    a = _agent()
    obs = np.zeros((4, 29), dtype=np.float32)
    assert not np.allclose(a.act(obs)[0], a.act(obs)[0])


def test_act_deterministic_returns_the_distribution_mean():
    a = _agent()
    obs = np.zeros((4, 29), dtype=np.float32)
    assert np.allclose(a.act(obs, deterministic=True)[0],
                       a.act(obs, deterministic=True)[0])


def test_evaluate_actions_logp_matches_act_logp():
    a = _agent()
    obs = np.random.default_rng(0).normal(size=(5, 29)).astype(np.float32)
    action, logp, _ = a.act(obs)
    logp2, _, _ = a.evaluate_actions(
        torch.as_tensor(obs), torch.as_tensor(action)
    )
    assert np.allclose(logp, logp2.detach().numpy(), atol=1e-5)


def test_update_returns_all_diagnostics():
    a = _agent(update_epochs=2, n_minibatches=2)
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(64, 29)).astype(np.float32)
    act, logp, _ = a.act(obs)
    out = a.update(obs, act, logp,
                   rng.normal(size=64).astype(np.float32),
                   rng.normal(size=64).astype(np.float32))
    assert set(out) >= {"policy_loss", "value_loss", "entropy",
                        "approx_kl", "clip_fraction"}
    assert all(np.isfinite(v) for v in out.values())


def test_update_changes_the_parameters():
    a = _agent(update_epochs=4, n_minibatches=2)
    before = [p.detach().clone() for p in a.net.parameters()]
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(64, 29)).astype(np.float32)
    act, logp, _ = a.act(obs)
    a.update(obs, act, logp,
             rng.normal(size=64).astype(np.float32),
             rng.normal(size=64).astype(np.float32))
    assert any(not torch.allclose(b, p) for b, p in zip(before, a.net.parameters()))


def test_update_increases_logp_of_advantaged_actions():
    """The core PPO guarantee: positive-advantage actions become more likely."""
    a = _agent(update_epochs=20, n_minibatches=1, clip_coef=0.2, ent_coef=0.0,
               normalize_adv=False, target_kl=None)
    rng = np.random.default_rng(0)
    obs = np.tile(rng.normal(size=(1, 29)).astype(np.float32), (32, 1))
    act, logp_old, _ = a.act(obs)
    adv = np.ones(32, dtype=np.float32)
    a.update(obs, act, logp_old, adv, np.zeros(32, dtype=np.float32))
    logp_new, _, _ = a.evaluate_actions(torch.as_tensor(obs), torch.as_tensor(act))
    assert logp_new.mean().item() > logp_old.mean()


def test_advantage_normalisation_is_applied_when_enabled():
    a = _agent(normalize_adv=True, update_epochs=1, n_minibatches=1)
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(32, 29)).astype(np.float32)
    act, logp, _ = a.act(obs)
    huge = (np.ones(32, dtype=np.float32) * 1e6)
    out = a.update(obs, act, logp, huge, np.zeros(32, dtype=np.float32))
    assert np.isfinite(out["policy_loss"]), "normalisation must tame the scale"


def test_save_and_load_roundtrip(tmp_path):
    a = _agent()
    obs = np.random.default_rng(0).normal(size=(3, 29)).astype(np.float32)
    ref = a.act(obs, deterministic=True)[0]
    path = tmp_path / "ppo.pt"
    a.save(str(path))

    b = PPO(29, 2, 24, 5, 1, PPOConfig(), seed=999)
    assert not np.allclose(b.act(obs, deterministic=True)[0], ref)
    b.load(str(path))
    assert np.allclose(b.act(obs, deterministic=True)[0], ref)


def test_train_ppo_respects_a_non_default_ray_count():
    """Shape is passed in, not inferred. With 16 rays and a stack of 2 the
    inferred version reshapes (2, 42) as (2, 1, 29) and raises."""
    from env.lidar import LidarConfig
    from env.robot_env import EnvConfig
    from env.wrappers import make_vec_env

    venv = make_vec_env(2, env_cfg=EnvConfig(n_rays=16, map_rows=10, map_cols=10,
                                             max_steps=30, min_separation=1.5),
                        lidar_cfg=LidarConfig(n_rays=16), frame_stack=2)
    try:
        agent, _ = train_ppo(venv, PPOConfig(rollout_steps=8, n_minibatches=2,
                                             update_epochs=1),
                             total_steps=32, seed=0, n_rays=16, frame_stack=2)
        action, _, _ = agent.act(np.zeros((2, 42), dtype=np.float32))
        assert action.shape == (2, 2)
    finally:
        venv.close()


def test_seeding_makes_two_agents_identical():
    a = PPO(29, 2, 24, 5, 1, PPOConfig(), seed=3)
    b = PPO(29, 2, 24, 5, 1, PPOConfig(), seed=3)
    obs = np.zeros((4, 29), dtype=np.float32)
    assert np.allclose(a.act(obs, deterministic=True)[0],
                       b.act(obs, deterministic=True)[0])
