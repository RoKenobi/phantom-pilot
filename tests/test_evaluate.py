import numpy as np
import pytest

from baselines.astar_oracle import AStarAgent
from baselines.random_agent import RandomAgent
from env.lidar import LidarConfig
from env.maps import HAND_MAPS
from env.robot_env import EnvConfig, RobotNavEnv
from experiments.evaluate import (
    EpisodeResult, aggregate, compute_spl, evaluate_many, evaluate_policy, run_episode,
)


def _result(**kw):
    base = dict(success=True, collision=False, timeout=False, steps=100,
                ep_return=5.0, path_length=10.0, optimal_length=10.0,
                min_clearance=0.4, smoothness=0.1)
    base.update(kw)
    return EpisodeResult(**base)


def _factory(fp=None, **cfg_kw):
    def make():
        return RobotNavEnv(env_cfg=EnvConfig(**cfg_kw), floorplan=fp,
                           lidar_cfg=LidarConfig(noise_std=0.0, dropout=0.0))
    return make


# --- SPL -----------------------------------------------------------------

def test_spl_is_one_for_perfectly_optimal_successes():
    assert compute_spl([_result(), _result()]) == pytest.approx(1.0)


def test_spl_is_zero_when_everything_fails():
    assert compute_spl([_result(success=False), _result(success=False)]) == 0.0


def test_spl_halves_when_the_path_is_twice_optimal():
    assert compute_spl([_result(path_length=20.0, optimal_length=10.0)]) == pytest.approx(0.5)


def test_spl_is_capped_at_one_when_the_agent_beats_the_grid_path():
    """A continuously-moving robot can cut corners the 8-connected grid
    planner cannot, so p < l is possible. max(p, l) caps the ratio at 1."""
    assert compute_spl([_result(path_length=5.0, optimal_length=10.0)]) == pytest.approx(1.0)


def test_spl_averages_over_all_episodes_including_failures():
    results = [_result(), _result(success=False)]
    assert compute_spl(results) == pytest.approx(0.5)


def test_spl_skips_episodes_with_no_reachable_optimal_path():
    results = [_result(), _result(optimal_length=None)]
    assert compute_spl(results) == pytest.approx(1.0), "unreachable episodes are excluded"


def test_spl_of_an_empty_list_is_zero():
    assert compute_spl([]) == 0.0


# --- aggregation ---------------------------------------------------------

def test_aggregate_reports_every_required_metric():
    out = aggregate([_result(), _result(success=False, collision=True)])
    assert set(out) >= {"success_rate", "collision_rate", "timeout_rate",
                        "mean_return", "mean_steps", "spl",
                        "mean_min_clearance", "mean_smoothness", "n_episodes"}


def test_aggregate_rates_are_fractions():
    out = aggregate([_result(), _result(success=False, collision=True),
                     _result(success=False, timeout=True)])
    assert out["success_rate"] == pytest.approx(1 / 3)
    assert out["collision_rate"] == pytest.approx(1 / 3)
    assert out["timeout_rate"] == pytest.approx(1 / 3)
    assert out["n_episodes"] == 3


def test_aggregate_mean_steps_counts_all_episodes():
    out = aggregate([_result(steps=10), _result(steps=30, success=False)])
    assert out["mean_steps"] == pytest.approx(20.0)


# --- episode runner ------------------------------------------------------

def test_run_episode_returns_a_populated_result():
    env = _factory()()
    r = run_episode(env, RandomAgent(seed=0), seed=0)
    assert isinstance(r, EpisodeResult)
    assert r.steps > 0
    assert r.success or r.collision or r.timeout, "exactly one outcome must be set"


def test_run_episode_outcomes_are_mutually_exclusive():
    env = _factory()()
    for s in range(5):
        r = run_episode(env, RandomAgent(seed=s), seed=s)
        assert sum([r.success, r.collision, r.timeout]) == 1


def test_run_episode_records_the_optimal_length():
    env = _factory(fp=HAND_MAPS["apartment"]())()
    r = run_episode(env, RandomAgent(seed=0), seed=0)
    assert r.optimal_length is None or r.optimal_length > 0.0


def test_run_episode_is_reproducible_for_a_given_seed():
    env = _factory()()
    a = run_episode(env, RandomAgent(seed=0), seed=7)
    b = run_episode(env, RandomAgent(seed=0), seed=7)
    assert a.steps == b.steps and a.ep_return == pytest.approx(b.ep_return)


def test_run_episode_works_on_a_wrapped_env():
    """Task 16 evaluates through FrameStack; a wrapper must not break the
    harness. Gymnasium refuses to forward private attributes through one."""
    from env.wrappers import FrameStack

    env = FrameStack(_factory()(), k=4)
    r = run_episode(env, RandomAgent(seed=0), seed=0)
    assert isinstance(r, EpisodeResult) and r.steps > 0
    assert r.success or r.collision or r.timeout


def test_run_episode_smoothness_is_zero_for_a_constant_action():
    class Constant:
        def reset(self, env): pass
        def act(self, obs): return np.array([1.0, 1.0], dtype=np.float32)

    env = _factory()()
    assert run_episode(env, Constant(), seed=0).smoothness == pytest.approx(0.0, abs=1e-9)


# --- harness -------------------------------------------------------------

def test_evaluate_policy_runs_the_requested_number_of_episodes():
    out = evaluate_policy(RandomAgent(seed=0), _factory(), episodes=6, seed=100)
    assert out["n_episodes"] == 6


def test_oracle_outperforms_random_on_every_headline_metric():
    fp = HAND_MAPS["apartment"]()
    factory = _factory(fp=fp, max_steps=1200)
    rnd = evaluate_policy(RandomAgent(seed=0), factory, episodes=8, seed=200)
    orc = evaluate_policy(AStarAgent(), factory, episodes=8, seed=200)
    assert orc["success_rate"] > rnd["success_rate"]
    assert orc["spl"] > rnd["spl"]


def test_evaluate_many_uses_identical_seeds_for_every_policy():
    """Paired comparison: each policy must face the same episodes."""
    factory = _factory(fp=HAND_MAPS["apartment"](), max_steps=1200)
    out = evaluate_many({"a": RandomAgent(seed=0), "b": RandomAgent(seed=0)},
                        factory, episodes=4, seed=300)
    assert set(out) == {"a", "b"}
    assert out["a"]["n_episodes"] == out["b"]["n_episodes"] == 4
