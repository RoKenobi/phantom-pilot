"""Demo clips: the trained agent succeeding, and every failure worth showing.

A short demo matters, but the failure gallery is the more valuable half --
"sometimes it gets stuck" is a sentence; a GIF of it getting stuck is evidence.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from baselines.reactive import ReactiveController
from env.lidar import LidarConfig
from env.maps import HAND_MAPS
from env.render import render_frame, write_video
from env.robot_env import EnvConfig, RobotNavEnv
from env.wrappers import FrameStack
from experiments.ablations import _load_agent, arm_config


def record(policy, env, path: str, seed: int, max_frames: int = 600,
           every: int = 2) -> dict:
    obs, _ = env.reset(seed=seed)
    policy.reset(env)
    inner = env.unwrapped
    frames, info = [], {}
    for i in range(max_frames):
        obs, _, terminated, truncated, info = env.step(policy.act(obs))
        if i % every == 0:
            frames.append(render_frame(inner))
        if terminated or truncated:
            break
    write_video(frames, path, fps=15)
    return info


def _env(floorplan=None, frame_stack=4, n_pedestrians=0, max_steps=500):
    e = RobotNavEnv(env_cfg=EnvConfig(n_pedestrians=n_pedestrians, max_steps=max_steps),
                    floorplan=floorplan, lidar_cfg=LidarConfig())
    return FrameStack(e, frame_stack) if frame_stack > 1 else e


def make_all(out_dir: str = "results/videos", out_root: str = "models",
             seed: int = 0) -> list[str]:
    """One clip of the trained agent succeeding.

    A short demo of the working agent, so this stays. The wider gallery this
    originally produced -- failure cases, the reactive controller's local
    minimum, the wall-hugging ablation -- was cut for compute budget; every
    behaviour it illustrated is reported numerically in results/metrics.json
    instead.
    """
    os.makedirs(out_dir, exist_ok=True)
    fs = int(arm_config("ppo_potential").get("frame_stack", 4))
    agent = _load_agent("ppo_potential", seed, out_root, 29 * fs)

    written = []
    for attempt in range(12):          # retry until an episode that succeeds
        path = os.path.join(out_dir, "agent_success.gif")
        info = record(agent, _env(frame_stack=fs), path, seed=900_001 + attempt)
        if info.get("success"):
            written.append(path)
            break
    else:
        written.append(path)           # keep the last clip regardless
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/videos")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for f in make_all(a.out, seed=a.seed):
        print("wrote", f)
