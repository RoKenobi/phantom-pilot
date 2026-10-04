import numpy as np
import pytest
import torch

from agents.dqn import (
    DQN, DQNConfig, ReplayBuffer, double_dqn_target, epsilon_at,
)


# --- replay buffer -------------------------------------------------------

def test_buffer_len_grows_then_saturates():
    buf = ReplayBuffer(capacity=3, obs_dim=4)
    for i in range(5):
        buf.add(np.full(4, i, dtype=np.float32), i % 2, float(i),
                np.full(4, i + 1, dtype=np.float32), False)
    assert len(buf) == 3


def test_buffer_overwrites_oldest_entries():
    buf = ReplayBuffer(capacity=2, obs_dim=1)
    for i in range(3):
        buf.add(np.array([i], dtype=np.float32), 0, float(i),
                np.array([i], dtype=np.float32), False)
    obs, *_ = buf.sample(2, np.random.default_rng(0))
    assert 0.0 not in obs.ravel(), "the first entry must have been evicted"


def test_buffer_sample_shapes():
    buf = ReplayBuffer(capacity=100, obs_dim=29)
    for i in range(50):
        buf.add(np.zeros(29, dtype=np.float32), i % 6, 1.0,
                np.ones(29, dtype=np.float32), i % 7 == 0)
    obs, act, rew, nobs, done = buf.sample(16, np.random.default_rng(0))
    assert obs.shape == (16, 29) and nobs.shape == (16, 29)
    assert act.shape == (16,) and rew.shape == (16,) and done.shape == (16,)
    assert act.dtype == np.int64


def test_buffer_preserves_transition_pairing():
    buf = ReplayBuffer(capacity=10, obs_dim=1)
    for i in range(10):
        buf.add(np.array([i], dtype=np.float32), i % 6, float(i * 10),
                np.array([i + 100], dtype=np.float32), bool(i % 2))
    obs, act, rew, nobs, done = buf.sample(10, np.random.default_rng(0))
    for o, r, n in zip(obs.ravel(), rew, nobs.ravel()):
        assert r == pytest.approx(o * 10)
        assert n == pytest.approx(o + 100)


def test_buffer_sampling_is_reproducible():
    buf = ReplayBuffer(capacity=50, obs_dim=2)
    for i in range(50):
        buf.add(np.full(2, i, dtype=np.float32), 0, 0.0,
                np.zeros(2, dtype=np.float32), False)
    a = buf.sample(8, np.random.default_rng(1))[0]
    b = buf.sample(8, np.random.default_rng(1))[0]
    assert np.array_equal(a, b)


# --- epsilon schedule ----------------------------------------------------

def test_epsilon_starts_at_eps_start_and_ends_at_eps_end():
    cfg = DQNConfig(eps_start=1.0, eps_end=0.05, eps_decay_steps=1000)
    assert epsilon_at(0, cfg) == pytest.approx(1.0)
    assert epsilon_at(1000, cfg) == pytest.approx(0.05)
    assert epsilon_at(50_000, cfg) == pytest.approx(0.05)


def test_epsilon_decays_linearly():
    cfg = DQNConfig(eps_start=1.0, eps_end=0.0, eps_decay_steps=100)
    assert epsilon_at(50, cfg) == pytest.approx(0.5)


# --- the Double DQN target ----------------------------------------------

def test_double_dqn_selects_with_online_and_values_with_target():
    """THE distinguishing test. Online and target disagree about the best
    next action; the target value must come from the online net's choice,
    not from the target net's own maximum."""
    rewards = np.array([0.0])
    dones = np.array([0.0])
    next_q_online = np.array([[1.0, 5.0]])   # online prefers action 1
    next_q_target = np.array([[9.0, 2.0]])   # target's own max is action 0
    y = double_dqn_target(rewards, dones, next_q_online, next_q_target, gamma=1.0)
    assert y[0] == pytest.approx(2.0), "must use target_q[argmax_online] = 2.0"
    assert y[0] != pytest.approx(9.0), "9.0 would be vanilla DQN's overestimate"


def test_double_dqn_applies_gamma_and_reward():
    y = double_dqn_target(np.array([1.0]), np.array([0.0]),
                          np.array([[0.0, 1.0]]), np.array([[3.0, 4.0]]), gamma=0.5)
    assert y[0] == pytest.approx(1.0 + 0.5 * 4.0)


def test_double_dqn_done_masks_the_bootstrap():
    y = double_dqn_target(np.array([2.0]), np.array([1.0]),
                          np.array([[0.0, 1.0]]), np.array([[100.0, 100.0]]), gamma=0.99)
    assert y[0] == pytest.approx(2.0), "a terminal transition bootstraps nothing"


def test_double_dqn_batched():
    y = double_dqn_target(
        np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.0]),
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]), gamma=1.0,
    )
    assert y == pytest.approx([5.0, 1.0, 2.0 + 9.0])


# --- agent ---------------------------------------------------------------

def _agent(**kw):
    return DQN(obs_dim=29, n_actions=6, n_rays=24, n_extra=5,
               frame_stack=1, cfg=DQNConfig(**kw), seed=0)


def test_act_returns_a_valid_action_index():
    a = _agent()
    for _ in range(20):
        idx = int(a.act(np.zeros(29, dtype=np.float32), epsilon=0.0))
        assert 0 <= idx < 6


def test_act_with_epsilon_one_is_uniformly_random():
    a = _agent()
    obs = np.zeros(29, dtype=np.float32)
    seen = {int(a.act(obs, epsilon=1.0)) for _ in range(200)}
    assert len(seen) >= 5, "epsilon=1 must explore essentially every action"


def test_act_with_epsilon_zero_is_greedy_and_deterministic():
    a = _agent()
    obs = np.zeros(29, dtype=np.float32)
    choices = {int(a.act(obs, epsilon=0.0)) for _ in range(20)}
    assert len(choices) == 1


def test_target_network_starts_synced_with_online():
    a = _agent()
    for p, q in zip(a.net.parameters(), a.target.parameters()):
        assert torch.allclose(p, q)


def test_target_network_is_frozen():
    a = _agent()
    assert all(not p.requires_grad for p in a.target.parameters())


def test_update_changes_online_but_not_target():
    a = _agent(batch_size=8)
    target_before = [p.detach().clone() for p in a.target.parameters()]
    online_before = [p.detach().clone() for p in a.net.parameters()]
    rng = np.random.default_rng(0)
    batch = (
        rng.normal(size=(8, 29)).astype(np.float32),
        rng.integers(0, 6, size=8),
        rng.normal(size=8).astype(np.float32),
        rng.normal(size=(8, 29)).astype(np.float32),
        np.zeros(8, dtype=np.float32),
    )
    a.update(batch)
    assert any(not torch.allclose(b, p) for b, p in zip(online_before, a.net.parameters()))
    assert all(torch.allclose(b, p) for b, p in zip(target_before, a.target.parameters()))


def test_sync_target_copies_online_weights():
    a = _agent(batch_size=8)
    rng = np.random.default_rng(0)
    a.update((rng.normal(size=(8, 29)).astype(np.float32), rng.integers(0, 6, size=8),
              rng.normal(size=8).astype(np.float32),
              rng.normal(size=(8, 29)).astype(np.float32),
              np.zeros(8, dtype=np.float32)))
    a.sync_target()
    for p, q in zip(a.net.parameters(), a.target.parameters()):
        assert torch.allclose(p, q)


def test_update_reports_loss_and_mean_q():
    a = _agent(batch_size=8)
    rng = np.random.default_rng(0)
    out = a.update((rng.normal(size=(8, 29)).astype(np.float32),
                    rng.integers(0, 6, size=8),
                    rng.normal(size=8).astype(np.float32),
                    rng.normal(size=(8, 29)).astype(np.float32),
                    np.zeros(8, dtype=np.float32)))
    assert set(out) >= {"loss", "mean_q"}
    assert all(np.isfinite(v) for v in out.values())


def test_repeated_updates_drive_q_toward_a_constant_target():
    """Sanity: with a fixed transition and a terminal flag, Q(s,a) must
    converge on the reward."""
    a = _agent(batch_size=16, lr=1e-2)
    obs = np.zeros((16, 29), dtype=np.float32)
    acts = np.zeros(16, dtype=np.int64)
    rew = np.full(16, 3.0, dtype=np.float32)
    batch = (obs, acts, rew, obs.copy(), np.ones(16, dtype=np.float32))
    for _ in range(300):
        a.update(batch)
    q = a.q_values(obs[:1])[0, 0]
    assert q == pytest.approx(3.0, abs=0.3)


def test_save_and_load_roundtrip(tmp_path):
    a = _agent()
    obs = np.random.default_rng(0).normal(size=(3, 29)).astype(np.float32)
    ref = a.q_values(obs)
    path = tmp_path / "dqn.pt"
    a.save(str(path))

    b = DQN(29, 6, 24, 5, 1, DQNConfig(), seed=999)
    assert not np.allclose(b.q_values(obs), ref)
    b.load(str(path))
    assert np.allclose(b.q_values(obs), ref)
