"""Config-driven training entry point.

Every experiment arm in the report is one YAML file plus a seed, so results
are reproducible from the command line and the ablation runner is just a
loop over configs.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import yaml

from agents.dqn import DQNConfig, train_dqn
from agents.ppo import PPOConfig, train_ppo
from env.lidar import LidarConfig
from env.rewards import RewardConfig
from env.robot_env import EnvConfig
from env.wrappers import make_env, make_vec_env


@dataclass
class CurriculumStage:
    name: str
    obstacle_density: float
    min_separation: float
    max_steps: int
    steps: int


# Fixed step budgets, not success-gated advancement: a success gate makes the
# total training length seed-dependent, which makes multi-seed curves
# impossible to average cleanly.
DEFAULT_CURRICULUM: list[CurriculumStage] = [
    CurriculumStage("empty",    0.00, 2.0, 300, 150_000),
    CurriculumStage("sparse",   0.08, 3.0, 400, 250_000),
    CurriculumStage("cluttered", 0.15, 4.0, 500, 350_000),
    CurriculumStage("hard",     0.22, 5.0, 500, 250_000),
]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_reward_cfg(d: dict | None) -> RewardConfig:
    return RewardConfig(**(d or {}))


def build_env_cfg(d: dict | None) -> EnvConfig:
    return EnvConfig(**(d or {}))


def build_lidar_cfg(d: dict | None) -> LidarConfig:
    return LidarConfig(**(d or {}))


def _merge_history(dst: dict, src: dict, step_offset: int) -> None:
    dst["step"].extend([s + step_offset for s in src["step"]])
    for k in ("episode_return", "episode_length", "success"):
        dst[k].extend(src[k])


def run_training(cfg: dict, seed: int, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    algo = cfg["algo"]
    reward_cfg = build_reward_cfg(cfg.get("reward"))
    lidar_cfg = build_lidar_cfg(cfg.get("lidar"))
    frame_stack = int(cfg.get("frame_stack", 1))
    total_steps = int(cfg["total_steps"])

    if cfg.get("curriculum"):
        budget = sum(s.steps for s in DEFAULT_CURRICULUM)
        stages = [
            CurriculumStage(s.name, s.obstacle_density, s.min_separation,
                            s.max_steps, max(1, int(total_steps * s.steps / budget)))
            for s in DEFAULT_CURRICULUM
        ]
    else:
        stages = [CurriculumStage("single", cfg["env"].get("obstacle_density", 0.15),
                                  cfg["env"].get("min_separation", 3.0),
                                  cfg["env"].get("max_steps", 500), total_steps)]

    history = {"step": [], "episode_return": [], "episode_length": [], "success": []}
    agent = None
    offset = 0
    stage_names = []

    for stage in stages:
        env_kwargs = dict(cfg["env"])
        env_kwargs.update(obstacle_density=stage.obstacle_density,
                          min_separation=stage.min_separation,
                          max_steps=stage.max_steps)
        env_cfg = build_env_cfg(env_kwargs)

        if algo == "ppo":
            ppo_cfg = PPOConfig(**cfg.get("ppo", {}))
            venv = make_vec_env(int(cfg.get("n_envs", 8)),
                                async_envs=bool(cfg.get("async_envs", True)),
                                env_cfg=env_cfg,
                                reward_cfg=reward_cfg, lidar_cfg=lidar_cfg,
                                frame_stack=frame_stack, seed=seed,
                                normalize_reward=True)
            new_agent, hist = train_ppo(venv, ppo_cfg, stage.steps, seed=seed,
                                        n_rays=env_cfg.n_rays, frame_stack=frame_stack)
            venv.close()
            if agent is not None:
                new_agent.net.load_state_dict(agent.net.state_dict())
            agent = new_agent
        elif algo == "dqn":
            dqn_cfg = DQNConfig(**cfg.get("dqn", {}))
            env = make_env(env_cfg=env_cfg, reward_cfg=reward_cfg,
                           lidar_cfg=lidar_cfg, frame_stack=frame_stack,
                           normalize_reward=True)()
            new_agent, hist = train_dqn(env, dqn_cfg, stage.steps, seed=seed,
                                        n_rays=env_cfg.n_rays, frame_stack=frame_stack)
            env.close()
            if agent is not None:
                new_agent.net.load_state_dict(agent.net.state_dict())
                new_agent.sync_target()
            agent = new_agent
        else:
            raise ValueError(f"unknown algo: {algo!r}")

        _merge_history(history, hist, offset)
        offset += stage.steps
        stage_names.append(stage.name)

    agent.save(os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f)

    tail = history["episode_return"][-50:] or [0.0]
    tail_success = history["success"][-50:] or [0.0]
    summary = {
        "algo": algo, "seed": seed, "total_steps": offset, "stages": stage_names,
        "final_mean_return": float(np.mean(tail)),
        "final_success_rate": float(np.mean(tail_success)),
        "n_episodes": len(history["episode_return"]),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--total-steps", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.total_steps is not None:
        cfg["total_steps"] = args.total_steps
    name = os.path.splitext(os.path.basename(args.config))[0]
    out = args.out or os.path.join("models", f"{name}_seed{args.seed}")

    summary = run_training(cfg, args.seed, out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
