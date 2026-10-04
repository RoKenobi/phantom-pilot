import json

import numpy as np
import pytest
import yaml

from experiments.train import (
    DEFAULT_CURRICULUM, CurriculumStage, build_env_cfg, build_lidar_cfg,
    build_reward_cfg, load_config, run_training,
)


def _min_cfg(algo="ppo"):
    return {
        "algo": algo,
        "env": {"map_rows": 10, "map_cols": 10, "max_steps": 60,
                "min_separation": 1.5, "obstacle_density": 0.1,
                "action_mode": "continuous" if algo == "ppo" else "discrete"},
        "reward": {"shaping": "potential"},
        "lidar": {"noise_std": 0.0, "dropout": 0.0},
        "frame_stack": 1,
        "n_envs": 2,
        "total_steps": 512,
        "curriculum": False,
        "ppo": {"rollout_steps": 32, "n_minibatches": 2, "update_epochs": 2},
        "dqn": {"learning_starts": 32, "batch_size": 16, "buffer_size": 1000,
                "target_update_interval": 50, "eps_decay_steps": 200},
    }


def test_load_config_reads_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"algo": "ppo", "total_steps": 10}))
    assert load_config(str(p))["algo"] == "ppo"


def test_build_reward_cfg_applies_overrides():
    cfg = build_reward_cfg({"shaping": "naive", "goal": 20.0})
    assert cfg.shaping == "naive" and cfg.goal == 20.0


def test_build_reward_cfg_rejects_unknown_keys():
    with pytest.raises(TypeError):
        build_reward_cfg({"not_a_field": 1})


def test_build_env_cfg_applies_overrides():
    assert build_env_cfg({"max_steps": 77}).max_steps == 77


def test_build_lidar_cfg_applies_overrides():
    assert build_lidar_cfg({"n_rays": 36}).n_rays == 36


def test_default_curriculum_is_monotonically_harder():
    d = [s.obstacle_density for s in DEFAULT_CURRICULUM]
    sep = [s.min_separation for s in DEFAULT_CURRICULUM]
    assert d == sorted(d) and sep == sorted(sep)
    assert len(DEFAULT_CURRICULUM) >= 3


def test_curriculum_stage_step_budgets_are_positive():
    assert all(s.steps > 0 for s in DEFAULT_CURRICULUM)


@pytest.mark.slow
def test_run_training_ppo_produces_a_model_and_history(tmp_path):
    out = run_training(_min_cfg("ppo"), seed=0, out_dir=str(tmp_path))
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "history.json").exists()
    hist = json.loads((tmp_path / "history.json").read_text())
    assert set(hist) >= {"step", "episode_return", "episode_length", "success"}
    assert out["total_steps"] >= 512


@pytest.mark.slow
def test_run_training_dqn_produces_a_model_and_history(tmp_path):
    run_training(_min_cfg("dqn"), seed=0, out_dir=str(tmp_path))
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "history.json").exists()


@pytest.mark.slow
def test_run_training_is_reproducible_for_a_fixed_seed(tmp_path):
    a = run_training(_min_cfg("ppo"), seed=5, out_dir=str(tmp_path / "a"))
    b = run_training(_min_cfg("ppo"), seed=5, out_dir=str(tmp_path / "b"))
    assert a["final_mean_return"] == pytest.approx(b["final_mean_return"], rel=1e-6)


@pytest.mark.slow
def test_curriculum_run_visits_every_stage(tmp_path):
    cfg = _min_cfg("ppo")
    cfg["curriculum"] = True
    cfg["total_steps"] = 1024
    out = run_training(cfg, seed=0, out_dir=str(tmp_path))
    assert len(out["stages"]) == len(DEFAULT_CURRICULUM)
