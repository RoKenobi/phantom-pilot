"""Fixed evaluation protocol shared by every agent and baseline.

"Average steps to goal" is not comparable across maps of different
difficulty, so the headline metric here is SPL -- Success weighted by
inverse Path Length (Anderson et al., 2018) -- normalised by the A* oracle's
shortest feasible path for that exact episode.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from baselines.astar_oracle import optimal_path_length


@dataclass
class EpisodeResult:
    success: bool
    collision: bool
    timeout: bool
    steps: int
    ep_return: float
    path_length: float
    optimal_length: float | None
    min_clearance: float
    smoothness: float


def run_episode(env, policy, seed: int) -> EpisodeResult:
    obs, _ = env.reset(seed=seed)
    # `env` may be wrapped (FrameStack). Gymnasium refuses to forward
    # underscore-prefixed attributes through a wrapper, and the map/pose live on
    # the base env either way -- so read internals from `unwrapped`, while
    # reset/step stay on the wrapper so stacked observations reach the policy.
    inner = env.unwrapped
    policy.reset(inner)

    optimal = optimal_path_length(inner.floorplan, inner.pose[:2], inner.goal)
    total, omegas = 0.0, []
    info = {}

    while True:
        obs, reward, terminated, truncated, info = env.step(policy.act(obs))
        total += float(reward)
        omegas.append(inner._prev_omega)
        if terminated or truncated:
            break

    smoothness = float(np.abs(np.diff(omegas)).mean()) if len(omegas) > 1 else 0.0
    return EpisodeResult(
        success=bool(info["success"]),
        collision=bool(info["collision"]),
        timeout=bool(info["timeout"]),
        steps=int(info["steps"]),
        ep_return=total,
        path_length=float(info["path_length"]),
        optimal_length=optimal,
        min_clearance=float(info["min_clearance"]),
        smoothness=smoothness,
    )


def compute_spl(results: list[EpisodeResult]) -> float:
    """SPL = (1/N) * sum_i  S_i * l_i / max(p_i, l_i).

    Episodes whose goal the planner cannot reach are excluded rather than
    scored zero -- an impossible episode says nothing about the policy.
    """
    usable = [r for r in results if r.optimal_length is not None and r.optimal_length > 0]
    if not usable:
        return 0.0
    terms = [
        (1.0 if r.success else 0.0) * r.optimal_length / max(r.path_length, r.optimal_length)
        for r in usable
    ]
    return float(np.mean(terms))


def aggregate(results: list[EpisodeResult]) -> dict[str, float]:
    if not results:
        return {"n_episodes": 0}
    return {
        "success_rate": float(np.mean([r.success for r in results])),
        "collision_rate": float(np.mean([r.collision for r in results])),
        "timeout_rate": float(np.mean([r.timeout for r in results])),
        "mean_return": float(np.mean([r.ep_return for r in results])),
        "mean_steps": float(np.mean([r.steps for r in results])),
        "spl": compute_spl(results),
        "mean_min_clearance": float(np.mean([r.min_clearance for r in results])),
        "mean_smoothness": float(np.mean([r.smoothness for r in results])),
        "n_episodes": len(results),
    }


def evaluate_policy(policy, env_factory, episodes: int = 100,
                    seed: int = 1000, return_episodes: bool = False):
    env = env_factory()
    results = [run_episode(env, policy, seed=seed + i) for i in range(episodes)]
    out = aggregate(results)
    if return_episodes:
        out["episodes"] = [asdict(r) for r in results]
    return out


def evaluate_many(policies: dict, env_factory, episodes: int = 100,
                  seed: int = 1000) -> dict[str, dict]:
    """Every policy sees the same episode seeds, so the comparison is paired."""
    return {
        name: evaluate_policy(p, env_factory, episodes=episodes, seed=seed)
        for name, p in policies.items()
    }
