import numpy as np
import pytest
import torch

from agents.dqn import DQN, DQNConfig
from agents.networks import MLPActorCritic, MLPQNetwork
from agents.ppo import PPO, PPOConfig


def test_mlp_actor_critic_shapes():
    net = MLPActorCritic(obs_dim=3, act_dim=1)
    mean, value = net(torch.zeros(4, 3))
    assert mean.shape == (4, 1) and value.shape == (4,)
    assert net.log_std.shape == (1,)


def test_mlp_q_network_shapes():
    assert MLPQNetwork(obs_dim=4, n_actions=2)(torch.zeros(5, 4)).shape == (5, 2)


def test_ppo_accepts_a_custom_network():
    """Required so PPO can run on reference tasks with no LiDAR structure."""
    agent = PPO(obs_dim=3, act_dim=1, cfg=PPOConfig(), seed=0,
                net=MLPActorCritic(3, 1))
    action, logp, value = agent.act(np.zeros((2, 3), dtype=np.float32))
    assert action.shape == (2, 1) and logp.shape == (2,) and value.shape == (2,)


def test_dqn_accepts_a_custom_network():
    agent = DQN(obs_dim=4, n_actions=2, cfg=DQNConfig(), seed=0,
                net=MLPQNetwork(4, 2))
    assert 0 <= int(agent.act(np.zeros(4, dtype=np.float32))) < 2


def test_validate_functions_are_importable_and_declare_thresholds():
    from experiments.validate_algos import validate_dqn, validate_ppo
    assert callable(validate_dqn) and callable(validate_ppo)


@pytest.mark.slow
def test_dqn_solves_cartpole():
    from experiments.validate_algos import validate_dqn
    result = validate_dqn(total_steps=250_000, seed=0)
    assert result["passed"], f"DQN failed its reference task: {result}"


@pytest.mark.slow
def test_ppo_solves_pendulum():
    from experiments.validate_algos import validate_ppo
    result = validate_ppo(total_steps=300_000, seed=0)
    assert result["passed"], f"PPO failed its reference task: {result}"
