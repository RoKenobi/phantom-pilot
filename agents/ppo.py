"""Proximal Policy Optimization, written from scratch.

Follows Schulman et al. (2017) with the implementation details that matter in
practice: GAE(lambda), advantage normalisation per minibatch, gradient
clipping, a clipped value loss, and early stopping on approximate KL.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from agents.networks import ActorCritic


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    update_epochs: int = 10
    n_minibatches: int = 32
    rollout_steps: int = 256
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    normalize_adv: bool = True
    hidden: int = 128


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: np.ndarray,
    last_done: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised Advantage Estimation over a (T, N) rollout.

    `dones[t]` marks that state t is the FIRST state of an episode, i.e. the
    step before it terminated. The bootstrap for step t is therefore masked by
    `dones[t + 1]` -- the flag on the state that follows it -- and by
    `last_done` for the final step.
    """
    T = rewards.shape[0]
    advantages = np.zeros_like(rewards, dtype=np.float64)
    last_gae = np.zeros_like(last_value, dtype=np.float64)

    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - last_done
            next_value = last_value
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae

    return advantages, advantages + values


class PPO:
    def __init__(self, obs_dim: int, act_dim: int, n_rays: int = 24,
                 n_extra: int = 5, frame_stack: int = 1,
                 cfg: PPOConfig | None = None, seed: int = 0, net=None) -> None:
        self.cfg = cfg or PPOConfig()
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.net = net if net is not None else ActorCritic(
            n_rays, n_extra, frame_stack, act_dim, self.cfg.hidden
        )
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)

    # -- distribution helpers --------------------------------------------

    def _dist(self, obs: torch.Tensor):
        mean, value = self.net(obs)
        return torch.distributions.Normal(mean, self.net.log_std.exp()), value

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        t = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        dist, value = self._dist(t)
        action = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return (action.numpy(), logp.numpy(), value.numpy())

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist, value = self._dist(obs)
        return (dist.log_prob(actions).sum(-1), dist.entropy().sum(-1), value)

    # -- learning ---------------------------------------------------------

    def update(self, obs, actions, logps, advantages, returns) -> dict[str, float]:
        c = self.cfg
        obs = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        actions = torch.as_tensor(np.asarray(actions, dtype=np.float32))
        logps_old = torch.as_tensor(np.asarray(logps, dtype=np.float32))
        advantages = torch.as_tensor(np.asarray(advantages, dtype=np.float32))
        returns = torch.as_tensor(np.asarray(returns, dtype=np.float32))

        n = obs.shape[0]
        idx = np.arange(n)
        mb_size = max(1, n // c.n_minibatches)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_fraction": 0.0}
        n_updates = 0

        for _ in range(c.update_epochs):
            self.rng.shuffle(idx)
            for start in range(0, n, mb_size):
                mb = idx[start:start + mb_size]
                if len(mb) < 2:
                    continue

                new_logp, entropy, value = self.evaluate_actions(obs[mb], actions[mb])
                log_ratio = new_logp - logps_old[mb]
                ratio = log_ratio.exp()

                mb_adv = advantages[mb]
                if c.normalize_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - c.clip_coef, 1 + c.clip_coef)
                policy_loss = torch.max(pg1, pg2).mean()
                value_loss = 0.5 * ((value - returns[mb]) ** 2).mean()
                ent = entropy.mean()

                loss = policy_loss + c.vf_coef * value_loss - c.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1).abs() > c.clip_coef).float().mean().item()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += ent.item()
                stats["approx_kl"] += approx_kl
                stats["clip_fraction"] += clip_frac
                n_updates += 1

            if c.target_kl is not None and n_updates and \
                    stats["approx_kl"] / n_updates > c.target_kl:
                break

        return {k: v / max(n_updates, 1) for k, v in stats.items()}

    # -- persistence ------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({"net": self.net.state_dict(), "opt": self.opt.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.net.load_state_dict(ckpt["net"])
        self.opt.load_state_dict(ckpt["opt"])


def train_ppo(vec_env, cfg: PPOConfig, total_steps: int, seed: int = 0,
              callback=None, net=None, n_rays: int = 24, n_extra: int = 5,
              frame_stack: int = 1) -> tuple[PPO, dict[str, list]]:
    """Roll out, compute GAE, update. Returns the agent and its history.

    The observation shape is passed in, never inferred. Deriving it from
    `obs_dim // 29` hardcodes 24 rays + 5 extras and mis-reshapes silently
    for any other sensor configuration.
    """
    n_envs = vec_env.num_envs
    obs_dim = int(np.prod(vec_env.single_observation_space.shape))
    act_dim = int(np.prod(vec_env.single_action_space.shape))
    agent = PPO(obs_dim, act_dim, n_rays, n_extra, frame_stack, cfg, seed, net=net)

    history = {"step": [], "episode_return": [], "episode_length": [], "success": []}
    obs, _ = vec_env.reset(seed=seed)
    done = np.zeros(n_envs)
    ep_return = np.zeros(n_envs)
    ep_len = np.zeros(n_envs, dtype=int)
    global_step = 0

    n_iters = max(1, total_steps // (cfg.rollout_steps * n_envs))
    for _ in range(n_iters):
        b_obs = np.zeros((cfg.rollout_steps, n_envs, obs_dim), dtype=np.float32)
        b_act = np.zeros((cfg.rollout_steps, n_envs, act_dim), dtype=np.float32)
        b_logp = np.zeros((cfg.rollout_steps, n_envs), dtype=np.float32)
        b_rew = np.zeros((cfg.rollout_steps, n_envs), dtype=np.float32)
        b_val = np.zeros((cfg.rollout_steps, n_envs), dtype=np.float32)
        b_done = np.zeros((cfg.rollout_steps, n_envs), dtype=np.float32)

        for t in range(cfg.rollout_steps):
            b_obs[t], b_done[t] = obs, done
            action, logp, value = agent.act(obs)
            b_act[t], b_logp[t], b_val[t] = action, logp, value

            obs, reward, terminated, truncated, infos = vec_env.step(
                np.clip(action, -1.0, 1.0)
            )
            done = np.logical_or(terminated, truncated).astype(np.float32)
            b_rew[t] = reward
            ep_return += reward
            ep_len += 1
            global_step += n_envs

            for i in np.flatnonzero(done):
                success = False
                if "final_info" in infos and infos["final_info"][i] is not None:
                    success = bool(infos["final_info"][i].get("success", False))
                history["step"].append(global_step)
                history["episode_return"].append(float(ep_return[i]))
                history["episode_length"].append(int(ep_len[i]))
                history["success"].append(float(success))
                ep_return[i], ep_len[i] = 0.0, 0

        _, _, last_value = agent.act(obs)
        adv, ret = compute_gae(b_rew, b_val, b_done, last_value, done,
                               cfg.gamma, cfg.gae_lambda)

        agent.update(
            b_obs.reshape(-1, obs_dim), b_act.reshape(-1, act_dim),
            b_logp.reshape(-1), adv.reshape(-1).astype(np.float32),
            ret.reshape(-1).astype(np.float32),
        )
        if callback is not None:
            callback(global_step, history)

    return agent, history
