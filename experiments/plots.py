"""Every figure in the report.

All curves are means over five seeds with a +-1 std band. Each seed's
episode series is resampled onto a shared step grid first, because episodes
end at irregular and seed-dependent step counts -- averaging by episode
index would compare different amounts of experience.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_histories(arm: str, seeds, out_root: str = "models") -> list[dict]:
    out = []
    for s in seeds:
        p = os.path.join(out_root, arm, f"seed{s}", "history.json")
        if os.path.exists(p):
            with open(p) as f:
                out.append(json.load(f))
    return out


def _running_mean(y: np.ndarray, window: int = 50) -> np.ndarray:
    if len(y) < 2:
        return y
    w = max(1, min(window, len(y) // 5))
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")


def smooth_to_grid(histories: list[dict], key: str, n_points: int = 200):
    """Resample every seed onto a common step grid, then average."""
    last = min(max(h["step"]) for h in histories)
    grid = np.linspace(0, last, n_points + 1)[1:]
    curves = []
    for h in histories:
        steps = np.asarray(h["step"], dtype=float)
        vals = _running_mean(np.asarray(h[key], dtype=float))
        curves.append(np.interp(grid, steps, vals))
    stacked = np.stack(curves)
    return grid, stacked.mean(axis=0), stacked.std(axis=0)


def plot_learning_curves(arms, out_root, seeds, path, key="success",
                         ylabel=None, title=None) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=140)
    for arm in arms:
        hs = load_histories(arm, seeds, out_root)
        if not hs:
            continue
        x, mean, std = smooth_to_grid(hs, key)
        line, = ax.plot(x, mean, label=f"{arm} (n={len(hs)})", linewidth=1.8)
        ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=line.get_color())
    ax.set_xlabel("environment steps")
    ax.set_ylabel(ylabel or key.replace("_", " "))
    ax.set_title(title or f"{key} during training (mean +- 1 std over seeds)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ablation_bars(metrics: dict, arms, metric: str, path: str,
                       suite: str = "procedural_unseen", title: str = "") -> None:
    vals = [metrics[a][suite][metric] for a in arms]          # KeyError if missing
    errs = [metrics[a][suite].get(f"{metric}_std", 0.0) for a in arms]
    fig, ax = plt.subplots(figsize=(1.6 * len(arms) + 2, 4.2), dpi=140)
    ax.bar(range(len(arms)), vals, yerr=errs, capsize=4,
           color="#4c78a8", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} on {suite}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_baseline_comparison(metrics: dict, path: str,
                             suite: str = "procedural_unseen") -> None:
    order = ["random", "reactive", "ppo_potential", "dqn_potential", "astar_oracle"]
    names = [n for n in order if n in metrics and suite in metrics[n]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    for ax, metric in zip(axes, ("success_rate", "spl")):
        vals = [metrics[n][suite][metric] for n in names]
        errs = [metrics[n][suite].get(f"{metric}_std", 0.0) for n in names]
        ax.bar(range(len(names)), vals, yerr=errs, capsize=4,
               color="#4c78a8", edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Learned policy vs the classical navigation stack ({suite})")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_all(metrics_path: str = "results/metrics.json",
             out_dir: str = "results/plots", out_root: str = "models",
             seeds=(0, 1, 2, 3, 4)) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    with open(metrics_path) as f:
        metrics = json.load(f)
    written = []

    def p(name):
        written.append(os.path.join(out_dir, name))
        return written[-1]

    plot_learning_curves(["ppo_potential", "dqn_potential"], out_root, seeds,
                         p("curve_algorithms.png"), key="success",
                         ylabel="success rate",
                         title="PPO (continuous) vs Double DQN (discrete)")
    plot_learning_curves(["ppo_potential", "ppo_naive", "ppo_sparse"],
                         out_root, seeds, p("curve_reward_shaping.png"),
                         key="success", ylabel="success rate",
                         title="Reward shaping: potential vs naive vs sparse")
    plot_learning_curves(["ppo_potential", "ppo_framestack1"], out_root, seeds,
                         p("curve_framestack.png"), key="success",
                         ylabel="success rate",
                         title="Partial observability: 4 stacked scans vs 1")
    plot_learning_curves(["ppo_potential"], out_root, seeds,
                         p("curve_return.png"), key="episode_return",
                         ylabel="episode return", title="PPO episode return")
    plot_learning_curves(["ppo_potential", "ppo_curriculum"], out_root, seeds,
                         p("curve_curriculum.png"), key="success",
                         ylabel="success rate",
                         title="Curriculum (staged difficulty) vs a flat task")

    plot_ablation_bars(metrics, ["ppo_potential", "ppo_naive", "ppo_sparse",
                                 "ppo_no_clearance"], "spl",
                       p("bars_reward_spl.png"), title="SPL by reward variant")
    plot_ablation_bars(metrics, ["ppo_potential", "ppo_no_clearance"],
                       "mean_min_clearance", p("bars_clearance.png"),
                       title="Minimum clearance with and without the clearance term")
    plot_ablation_bars(metrics, ["ppo_potential", "dqn_potential"],
                       "mean_smoothness", p("bars_smoothness.png"),
                       title="Trajectory smoothness: continuous vs discretised control")
    plot_baseline_comparison(metrics, p("baseline_comparison.png"))

    for suite in ("apartment", "u_trap", "corridor", "dynamic"):
        if any(suite in v for v in metrics.values()):
            plot_baseline_comparison(metrics, p(f"generalisation_{suite}.png"),
                                     suite=suite)
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/metrics.json")
    ap.add_argument("--out", default="results/plots")
    a = ap.parse_args()
    for f in make_all(a.metrics, a.out):
        print("wrote", f)
