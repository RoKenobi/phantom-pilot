"""Every experiment in the report, as a sweep over (arm, seed).

Each arm changes exactly one variable against `ppo_potential`, so any
difference in the results is attributable. Five seeds per arm, because a
single-seed RL result is noise.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from baselines.astar_oracle import AStarAgent
from baselines.random_agent import RandomAgent
from baselines.reactive import ReactiveController
from env.lidar import LidarConfig
from env.maps import HAND_MAPS
from env.rewards import REWARD_VARIANTS
from env.robot_env import EnvConfig, RobotNavEnv
from env.wrappers import make_env
from experiments.evaluate import evaluate_policy
from experiments.train import load_config, run_training

SEEDS = [0, 1, 2]   # reduced from 5; see README "Compute budget"
EVAL_EPISODES = 20
EVAL_SEED_BASE = 900_000          # disjoint from any training seed

_BASE_PPO = "experiments/configs/ppo_potential.yaml"
_BASE_DQN = "experiments/configs/dqn_potential.yaml"

ABLATIONS: dict[str, dict] = {
    "ppo_potential":    {"base": _BASE_PPO, "overrides": {}},
    "ppo_naive":        {"base": _BASE_PPO, "overrides": {"reward": {"shaping": "naive"}}},
    "ppo_sparse":       {"base": _BASE_PPO, "overrides": {"reward": "sparse"}},
    "ppo_no_clearance": {"base": _BASE_PPO, "overrides": {"reward": "no_clearance"}},
    "ppo_framestack1":  {"base": _BASE_PPO, "overrides": {"frame_stack": 1}},
    "dqn_potential":    {"base": _BASE_DQN, "overrides": {}},
    # Bonus item: staged difficulty. Differs from ppo_potential in exactly one
    # flag, so any difference is attributable to the curriculum itself.
    "ppo_curriculum":   {"base": _BASE_PPO, "overrides": {"curriculum": True}},
}


def arm_config(name: str) -> dict:
    spec = ABLATIONS[name]
    cfg = copy.deepcopy(load_config(spec["base"]))
    for key, value in spec["overrides"].items():
        if key == "reward" and isinstance(value, str):
            # Named variant from env.rewards, expanded to a plain dict.
            cfg["reward"] = {k: v for k, v in vars(REWARD_VARIANTS[value]).items()}
        elif isinstance(value, dict):
            cfg[key] = {**cfg.get(key, {}), **value}
        else:
            cfg[key] = value
    return cfg


def run_arm(name: str, seed: int, out_root: str,
            total_steps_override: int | None = None) -> dict:
    cfg = arm_config(name)
    if total_steps_override is not None:
        cfg["total_steps"] = total_steps_override
    out_dir = os.path.join(out_root, name, f"seed{seed}")
    summary = run_training(cfg, seed, out_dir)
    summary["arm"] = name
    return summary


def _run_one(args) -> dict:
    # One lane = one process = one torch thread. Without this,each  worker's BLAS
    # spawns threads equal to the core count and the lanes thrash each other.
    torch.set_num_threads(1)
    return run_arm(*args)


def run_all(out_root: str = "models", seeds=None, n_workers: int = 6,
            total_steps_override: int | None = None) -> list[dict]:
    seeds = seeds or SEEDS
    jobs = [(name, s, out_root, total_steps_override)
            for name in ABLATIONS for s in seeds]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(_run_one, jobs))


# --- evaluation suites ---------------------------------------------------

# The hand-designed maps are 20x20 cells (10 m square); training maps are
# 12x12 (6 m). Crossing 10 m at 0.22 m/s takes ~450 steps, so the training
# episode cap of 250 would make those suites unsolvable for reasons that have
# nothing to do with the policy. Budget scales with the map instead.
_HAND_MAP_MAX_STEPS = 600


def _training_env() -> dict:
    """The env parameters the agents were actually trained on."""
    return dict(load_config(_BASE_PPO)["env"])


def _suite(base_env: dict | None = None, floorplan=None, n_pedestrians: int = 0,
           frame_stack: int = 4, action_mode: str | None = None,
           max_steps: int | None = None):
    """Build an evaluation env that MATCHES TRAINING except where a suite
    deliberately varies.

    Evaluating on a different map size, episode cap or obstacle density than
    the agent trained on is an accidental distribution shift, not a
    generalisation test -- the numbers would say nothing about the policy.
    Only the floor plan and the crowd vary here.
    """
    env_kwargs = dict(base_env if base_env is not None else _training_env())
    if action_mode is not None:
        env_kwargs["action_mode"] = action_mode
    if max_steps is not None:
        env_kwargs["max_steps"] = max_steps
    env_kwargs["n_pedestrians"] = n_pedestrians
    n_rays = int(env_kwargs.get("n_rays", 24))

    def factory():
        return make_env(
            env_cfg=EnvConfig(**env_kwargs),
            lidar_cfg=LidarConfig(n_rays=n_rays),
            floorplan=floorplan,
            frame_stack=frame_stack,
        )()
    return factory


def suite_factory(name: str, base_env: dict | None = None,
                  frame_stack: int = 4, action_mode: str | None = None):
    """One named evaluation suite, built against a given training config."""
    if name == "procedural_unseen":
        return _suite(base_env, frame_stack=frame_stack, action_mode=action_mode)
    if name == "dynamic":
        return _suite(base_env, n_pedestrians=8, frame_stack=frame_stack,
                      action_mode=action_mode)
    return _suite(base_env, floorplan=HAND_MAPS[name](), frame_stack=frame_stack,
                  action_mode=action_mode, max_steps=_HAND_MAP_MAX_STEPS)


# `corridor` dropped for compute budget: `u_trap` already covers the concave-
# obstacle case and `apartment` covers unseen layouts at a larger scale.
EVAL_SUITE_NAMES = ["procedural_unseen", "apartment", "u_trap", "dynamic"]
EVAL_SUITES = {n: suite_factory(n) for n in EVAL_SUITE_NAMES}


class _TrainedPolicy:
    """Adapts a trained agent to the baseline `Policy` interface."""

    def __init__(self, agent, discrete: bool) -> None:
        self.agent, self.discrete = agent, discrete

    def reset(self, env) -> None:
        pass

    def act(self, obs):
        if self.discrete:
            return self.agent.act(obs, epsilon=0.0)
        return np.clip(self.agent.act(obs[None, :], deterministic=True)[0][0], -1.0, 1.0)


def _load_agent(arm: str, seed: int, out_root: str, obs_dim: int):
    from agents.dqn import DQN, DQNConfig
    from agents.ppo import PPO, PPOConfig

    cfg = arm_config(arm)
    path = os.path.join(out_root, arm, f"seed{seed}", "model.pt")
    fs = int(cfg.get("frame_stack", 1))
    if cfg["algo"] == "ppo":
        agent = PPO(obs_dim, 2, 24, 5, fs, PPOConfig(**cfg.get("ppo", {})), seed)
        agent.load(path)
        return _TrainedPolicy(agent, discrete=False)
    agent = DQN(obs_dim, 6, 24, 5, fs, DQNConfig(**cfg.get("dqn", {})), seed)
    agent.load(path)
    return _TrainedPolicy(agent, discrete=True)


def evaluate_arms(out_root: str = "models", seeds=None,
                  episodes: int = EVAL_EPISODES) -> dict:
    """Every arm x every seed x every suite, plus the three baselines.

    All policies see the same episode seeds within a suite, so every
    comparison in the report is paired.
    """
    seeds = seeds or SEEDS
    results: dict = {}

    skipped = []
    for arm, spec in ABLATIONS.items():
        cfg = arm_config(arm)
        fs = int(cfg.get("frame_stack", 1))
        mode = cfg["env"].get("action_mode", "continuous")

        have = [s for s in seeds
                if os.path.exists(os.path.join(out_root, arm, f"seed{s}", "model.pt"))]
        if not have:
            skipped.append(arm)
            print(f"  skip {arm}: no checkpoints under {out_root}/{arm}", flush=True)
            continue
        if len(have) < len(seeds):
            print(f"  {arm}: only {len(have)}/{len(seeds)} seeds have checkpoints",
                  flush=True)

        results[arm] = {}
        for suite_name in EVAL_SUITE_NAMES:
            factory = suite_factory(suite_name, base_env=cfg["env"],
                                    frame_stack=fs, action_mode=mode)
            per_seed = []
            for s in have:
                policy = _load_agent(arm, s, out_root, 29 * fs)
                per_seed.append(evaluate_policy(policy, factory, episodes,
                                                EVAL_SEED_BASE))
            results[arm][suite_name] = _mean_std(per_seed)
            print(f"  {arm}/{suite_name}: success="
                  f"{results[arm][suite_name]['success_rate']:.3f} "
                  f"spl={results[arm][suite_name]['spl']:.3f}", flush=True)

    # Only the random policy varies with its own seed; the reactive controller
    # and the A* oracle are deterministic given an episode seed, so repeating
    # them per training seed would just recompute identical numbers.
    baselines = (
        ("random", lambda s: RandomAgent(seed=s), list(seeds)),
        ("reactive", lambda s: ReactiveController(), [0]),
        ("astar_oracle", lambda s: AStarAgent(), [0]),
    )
    for bl_name, make, bl_seeds in baselines:
        results[bl_name] = {}
        for suite_name in EVAL_SUITE_NAMES:
            factory = suite_factory(suite_name, frame_stack=1)
            per_seed = [evaluate_policy(make(s), factory, episodes, EVAL_SEED_BASE)
                        for s in bl_seeds]
            results[bl_name][suite_name] = _mean_std(per_seed)
            print(f"  {bl_name}/{suite_name}: success="
                  f"{results[bl_name][suite_name]['success_rate']:.3f} "
                  f"spl={results[bl_name][suite_name]['spl']:.3f}", flush=True)

    if skipped:
        results["_skipped_arms"] = skipped
    return results


def _mean_std(per_seed: list[dict]) -> dict:
    keys = [k for k in per_seed[0] if k != "episodes"]
    out = {}
    for k in keys:
        vals = [float(d[k]) for d in per_seed]
        out[k] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    out["n_seeds"] = len(per_seed)
    return out


def write_metrics(results: dict, path: str = "results/metrics.json") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["train", "evaluate", "both"], default="both")
    p.add_argument("--out-root", default="models")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--total-steps", type=int, default=None)
    args = p.parse_args()

    if args.stage in ("train", "both"):
        for s in run_all(args.out_root, n_workers=args.workers,
                         total_steps_override=args.total_steps):
            print(f"{s['arm']} seed{s['seed']}: "
                  f"success={s['final_success_rate']:.2f} "
                  f"return={s['final_mean_return']:.2f}")
    if args.stage in ("evaluate", "both"):
        write_metrics(evaluate_arms(args.out_root))
        print("wrote results/metrics.json")


if __name__ == "__main__":
    main()
