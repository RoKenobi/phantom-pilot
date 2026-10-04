import json

import numpy as np
import pytest

from experiments.plots import (
    plot_ablation_bars, plot_baseline_comparison, smooth_to_grid,
)


def _history(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "step": list(np.cumsum(rng.integers(50, 150, size=n)).astype(int)),
        "episode_return": list(rng.normal(size=n)),
        "episode_length": list(rng.integers(10, 500, size=n)),
        "success": list((rng.random(n) < 0.4).astype(float)),
    }


def test_smooth_to_grid_returns_three_aligned_arrays():
    x, mean, std = smooth_to_grid([_history(seed=i) for i in range(3)],
                                  key="success", n_points=50)
    assert x.shape == mean.shape == std.shape == (50,)


def test_smooth_to_grid_handles_seeds_of_differing_length():
    hs = [_history(n=100, seed=0), _history(n=300, seed=1)]
    x, mean, std = smooth_to_grid(hs, key="success", n_points=40)
    assert np.isfinite(mean).all() and np.isfinite(std).all()


def test_smooth_to_grid_std_is_zero_for_identical_seeds():
    h = _history(seed=0)
    _, _, std = smooth_to_grid([h, h, h], key="episode_return", n_points=30)
    assert np.allclose(std, 0.0, atol=1e-9)


def test_smooth_to_grid_x_is_increasing():
    x, _, _ = smooth_to_grid([_history()], key="success", n_points=25)
    assert (np.diff(x) > 0).all()


def test_plot_ablation_bars_writes_a_file(tmp_path):
    metrics = {
        "ppo_potential": {"procedural_unseen": {"spl": 0.6, "spl_std": 0.05}},
        "ppo_naive": {"procedural_unseen": {"spl": 0.4, "spl_std": 0.08}},
    }
    out = tmp_path / "bars.png"
    plot_ablation_bars(metrics, ["ppo_potential", "ppo_naive"], "spl",
                       str(out), title="SPL by reward variant")
    assert out.exists() and out.stat().st_size > 0


def test_plot_baseline_comparison_writes_a_file(tmp_path):
    metrics = {
        name: {"procedural_unseen": {"success_rate": v, "success_rate_std": 0.02,
                                     "spl": v * 0.8, "spl_std": 0.02}}
        for name, v in (("random", 0.02), ("reactive", 0.45),
                        ("ppo_potential", 0.82), ("astar_oracle", 0.99))
    }
    out = tmp_path / "baselines.png"
    plot_baseline_comparison(metrics, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_ablation_bars_raises_on_a_missing_arm(tmp_path):
    with pytest.raises(KeyError):
        plot_ablation_bars({}, ["nope"], "spl", str(tmp_path / "x.png"), title="t")
