"""Double DQN, written from scratch.

Van Hasselt et al. (2016): the online network picks the next action, the
target network scores it. Decoupling the two removes the systematic
overestimation that comes from taking a max over noisy Q-values.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from agents.networks import QNetwork


class ReplayBuffer:
    """Fixed-capacity circular buffer of transitions."""

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self._ptr
        self.obs[i] = obs
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self._ptr = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, self._size, size=batch_size)
        return (self.obs[idx], self.actions[idx], self.rewards[idx],
                self.next_obs[idx], self.dones[idx])


@dataclass
class DQNConfig:
    lr: float = 1e-4
    gamma: float = 0.99
    buffer_size: int = 200_000
    batch_size: int = 128
    learning_starts: int = 5_000
    train_freq: int = 4
    target_update_interval: int = 1_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000
    max_grad_norm: float = 10.0
    hidden: int = 128


def epsilon_at(step: int, cfg: DQNConfig) -> float:
    frac = min(1.0, step / max(cfg.eps_decay_steps, 1))
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def double_dqn_target(rewards, dones, next_q_online, next_q_target, gamma: float):
    """y = r + gamma * (1 - done) * Q_target(s', argmax_a Q_online(s', a))."""
    best = np.argmax(next_q_online, axis=1)
    bootstrapped = next_q_target[np.arange(len(best)), best]
    return rewards + gamma * (1.0 - dones) * bootstrapped


class DQN:
    def __init__(self, obs_dim: int, n_actions: int, n_rays: int = 24,
                 n_extra: int = 5, frame_stack: int = 1,
                 cfg: DQNConfig | None = None, seed: int = 0, net=None) -> None:
        self.cfg = cfg or DQNConfig()
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions
        self.net = net if net is not None else QNetwork(
            n_rays, n_extra, frame_stack, n_actions, self.cfg.hidden
        )
        self.target = copy.deepcopy(self.net)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)

    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(np.atleast_2d(np.asarray(obs, dtype=np.float32)))
        return self.net(t).numpy()

    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.q_values(obs)[0]))

    def update(self, batch) -> dict[str, float]:
        obs, actions, rewards, next_obs, dones = batch
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        next_t = torch.as_tensor(np.asarray(next_obs, dtype=np.float32))
        act_t = torch.as_tensor(np.asarray(actions, dtype=np.int64))

        with torch.no_grad():
            y = double_dqn_target(
                np.asarray(rewards, dtype=np.float64),
                np.asarray(dones, dtype=np.float64),
                self.net(next_t).numpy(),
                self.target(next_t).numpy(),
                self.cfg.gamma,
            )
        y_t = torch.as_tensor(y, dtype=torch.float32)

        q = self.net(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, y_t)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
        self.opt.step()

        return {"loss": float(loss.item()), "mean_q": float(q.mean().item())}

    def sync_target(self) -> None:
        self.target.load_state_dict(self.net.state_dict())

    def save(self, path: str) -> None:
        torch.save({"net": self.net.state_dict(), "opt": self.opt.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.net.load_state_dict(ckpt["net"])
        self.sync_target()
        self.opt.load_state_dict(ckpt["opt"])


def train_dqn(env, cfg: DQNConfig, total_steps: int, seed: int = 0,
              callback=None, net=None, n_rays: int = 24, n_extra: int = 5,
              frame_stack: int = 1) -> tuple[DQN, dict[str, list]]:
    """Single-env off-policy training loop.

    Observation shape is passed in, never inferred from `obs_dim // 29`.
    """
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    agent = DQN(obs_dim, n_actions, n_rays, n_extra, frame_stack, cfg, seed, net=net)
    buf = ReplayBuffer(cfg.buffer_size, obs_dim)

    history = {"step": [], "episode_return": [], "episode_length": [], "success": []}
    obs, _ = env.reset(seed=seed)
    ep_return, ep_len = 0.0, 0

    for step in range(total_steps):
        eps = epsilon_at(step, cfg)
        action = agent.act(obs, eps)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Bootstrap through a timeout: truncation is not a real terminal state.
        buf.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        ep_return += float(reward)
        ep_len += 1

        if terminated or truncated:
            history["step"].append(step)
            history["episode_return"].append(ep_return)
            history["episode_length"].append(ep_len)
            history["success"].append(float(info.get("success", False)))
            obs, _ = env.reset()
            ep_return, ep_len = 0.0, 0

        if step >= cfg.learning_starts and step % cfg.train_freq == 0:
            agent.update(buf.sample(cfg.batch_size, agent.rng))
        if step % cfg.target_update_interval == 0:
            agent.sync_target()
        if callback is not None and step % 10_000 == 0:
            callback(step, history, agent)

    return agent, history
