import json

import numpy as np
import pytest

from env.maps import HAND_MAPS
from experiments.ablations import (
    ABLATIONS, EVAL_SUITES, SEEDS, arm_config, write_metrics,
)


def test_seeds_are_distinct_and_enough_for_a_std():
    assert len(SEEDS) == len(set(SEEDS))
    assert len(SEEDS) >= 3, "fewer than 3 seeds cannot support an error bar"


def test_every_reward_variant_has_an_arm():
    names = set(ABLATIONS)
    assert {"ppo_potential", "ppo_naive", "ppo_sparse", "ppo_no_clearance"} <= names


def test_algorithm_comparison_arms_exist():
    assert {"ppo_potential", "dqn_potential"} <= set(ABLATIONS)


def test_observation_ablation_arm_exists():
    assert "ppo_framestack1" in ABLATIONS


def test_arm_config_is_complete_and_runnable():
    for name in ABLATIONS:
        cfg = arm_config(name)
        assert cfg["algo"] in ("ppo", "dqn")
        assert "env" in cfg and "reward" in cfg and "total_steps" in cfg
        assert cfg["total_steps"] > 0


def test_arms_differ_only_in_the_variable_under_test():
    base = arm_config("ppo_potential")
    naive = arm_config("ppo_naive")
    assert naive["reward"]["shaping"] == "naive"
    assert base["reward"]["shaping"] == "potential"
    assert naive["env"] == base["env"], "the env must be held constant"
    assert naive["total_steps"] == base["total_steps"]


def test_framestack_arm_differs_only_in_frame_stack():
    base, fs1 = arm_config("ppo_potential"), arm_config("ppo_framestack1")
    assert base["frame_stack"] == 4 and fs1["frame_stack"] == 1
    assert fs1["reward"] == base["reward"]


def test_dqn_arm_uses_the_discrete_action_space():
    assert arm_config("dqn_potential")["env"]["action_mode"] == "discrete"


def test_eval_suite_matches_the_training_configuration():
    """Evaluating on a different map size or episode cap than training is a
    distribution shift, not a generalisation test."""
    from experiments.ablations import _training_env, suite_factory

    train = _training_env()
    env = suite_factory("procedural_unseen")().unwrapped
    assert env.cfg.map_rows == train["map_rows"]
    assert env.cfg.map_cols == train["map_cols"]
    assert env.cfg.max_steps == train["max_steps"]
    assert env.cfg.obstacle_density == train["obstacle_density"]


def test_hand_map_suites_get_a_budget_scaled_to_their_size():
    from experiments.ablations import _training_env, suite_factory

    env = suite_factory("apartment")().unwrapped
    assert env.cfg.max_steps > _training_env()["max_steps"]


def test_eval_suites_cover_held_out_and_dynamic_settings():
    """Checks the categories that must be covered, not a fixed list -- the exact
    set of hand maps is a compute-budget decision."""
    assert "procedural_unseen" in EVAL_SUITES, "need unseen procedural maps"
    assert "dynamic" in EVAL_SUITES, "need a moving-obstacle suite"
    hand = set(EVAL_SUITES) & set(HAND_MAPS)
    assert hand, "need at least one hand-designed held-out map"
    for name, factory in EVAL_SUITES.items():
        env = factory()
        obs, _ = env.reset(seed=0)
        assert obs.shape[0] % 29 == 0


def test_dynamic_suite_actually_has_pedestrians():
    env = EVAL_SUITES["dynamic"]()
    env.reset(seed=0)
    inner = env.unwrapped
    assert inner.cfg.n_pedestrians > 0


def test_write_metrics_produces_valid_json(tmp_path):
    out = tmp_path / "metrics.json"
    write_metrics({"ppo_potential": {"procedural_unseen": {"spl": 0.5}}}, str(out))
    assert json.loads(out.read_text())["ppo_potential"]["procedural_unseen"]["spl"] == 0.5


@pytest.mark.slow
def test_run_arm_writes_a_checkpoint(tmp_path):
    from experiments.ablations import run_arm
    cfg_out = run_arm("ppo_potential", seed=0, out_root=str(tmp_path),
                      total_steps_override=512)
    assert (tmp_path / "ppo_potential" / "seed0" / "model.pt").exists()
    assert cfg_out["seed"] == 0
