import numpy as np
import pytest

from env.rewards import REWARD_VARIANTS, RewardConfig, compute_reward


def _call(cfg, **kw):
    base = dict(prev_dist=2.0, dist=2.0, min_scan=1.0, omega=0.0,
                prev_omega=0.0, reached=False, collided=False)
    base.update(kw)
    return compute_reward(cfg, **base)


def test_goal_reward_awarded_on_success():
    r = _call(RewardConfig(), reached=True)
    assert r.goal == pytest.approx(10.0)


def test_collision_penalty_applied():
    r = _call(RewardConfig(), collided=True)
    assert r.collision == pytest.approx(-10.0)


def test_step_penalty_always_applied():
    assert _call(RewardConfig()).step == pytest.approx(-0.01)


def test_potential_shaping_rewards_progress():
    cfg = RewardConfig(shaping="potential", gamma=0.99)
    r = _call(cfg, prev_dist=2.0, dist=1.0)
    # F = gamma*Phi(s') - Phi(s) = 0.99*(-1.0) - (-2.0) = 1.01
    assert r.shaping == pytest.approx(1.01)


def test_potential_shaping_penalises_regress():
    cfg = RewardConfig(shaping="potential", gamma=0.99)
    r = _call(cfg, prev_dist=1.0, dist=2.0)
    assert r.shaping == pytest.approx(0.99 * -2.0 - -1.0)
    assert r.shaping < 0


def test_potential_shaping_telescopes_to_zero_on_a_closed_loop():
    """The defining property: shaping over any cycle sums to ~0 (gamma=1),
    which is why it cannot be farmed."""
    cfg = RewardConfig(shaping="potential", gamma=1.0)
    dists = [3.0, 2.0, 1.5, 2.5, 3.0]
    total = sum(
        _call(cfg, prev_dist=a, dist=b).shaping for a, b in zip(dists, dists[1:])
    )
    assert total == pytest.approx(0.0, abs=1e-9)


def test_naive_shaping_does_not_telescope():
    """The spec's suggested '+0.1 when closer' is farmable: an oscillation
    that returns to the start nets a positive reward."""
    cfg = RewardConfig(shaping="naive")
    loop = _call(cfg, prev_dist=2.0, dist=1.0).shaping + \
           _call(cfg, prev_dist=1.0, dist=2.0).shaping
    assert loop == pytest.approx(0.0)
    forward_back_forward = loop + _call(cfg, prev_dist=2.0, dist=1.0).shaping
    assert forward_back_forward > 0.0


def test_naive_shaping_gives_fixed_bonus_regardless_of_step_size():
    cfg = RewardConfig(shaping="naive")
    small = _call(cfg, prev_dist=2.0, dist=1.99).shaping
    large = _call(cfg, prev_dist=2.0, dist=0.10).shaping
    assert small == pytest.approx(large) == pytest.approx(0.1)


def test_no_shaping_variant_is_silent():
    assert _call(RewardConfig(shaping="none"), prev_dist=2.0, dist=1.0).shaping == 0.0


def test_clearance_penalty_only_inside_threshold():
    cfg = RewardConfig(clearance_threshold=0.3, clearance_penalty=-0.05)
    assert _call(cfg, min_scan=0.5).clearance == 0.0
    assert _call(cfg, min_scan=0.2).clearance == pytest.approx(-0.05)


def test_jerk_penalty_scales_with_angular_change():
    cfg = RewardConfig(jerk_penalty=-0.01)
    assert _call(cfg, omega=0.0, prev_omega=0.0).jerk == pytest.approx(0.0)
    assert _call(cfg, omega=2.0, prev_omega=0.0).jerk == pytest.approx(-0.02)


def test_total_is_the_sum_of_terms():
    r = _call(RewardConfig(), prev_dist=2.0, dist=1.0, min_scan=0.2,
              omega=1.0, reached=True)
    assert r.total == pytest.approx(
        r.goal + r.collision + r.shaping + r.clearance + r.step + r.jerk
    )


@pytest.mark.parametrize("name", ["potential", "naive", "sparse", "no_clearance"])
def test_named_variants_exist_and_differ(name):
    assert name in REWARD_VARIANTS
    assert isinstance(REWARD_VARIANTS[name], RewardConfig)


def test_sparse_variant_has_no_dense_terms():
    cfg = REWARD_VARIANTS["sparse"]
    r = _call(cfg, prev_dist=2.0, dist=1.0, min_scan=0.1, omega=2.0)
    assert r.shaping == 0.0 and r.clearance == 0.0 and r.jerk == 0.0


def test_no_clearance_variant_keeps_shaping_but_drops_clearance():
    cfg = REWARD_VARIANTS["no_clearance"]
    r = _call(cfg, prev_dist=2.0, dist=1.0, min_scan=0.1)
    assert r.clearance == 0.0
    assert r.shaping != 0.0
