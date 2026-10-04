"""Reward terms, and the four variants compared in the ablation study.

The headline result of this project lives here: potential-based shaping
(Ng, Harada & Russell 1999) is policy-invariant, while the naive
intuitive "+0.1 when closer" scheme is not -- it can be farmed
by oscillating toward and away from the goal.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    shaping: str = "potential"        # "potential" | "naive" | "none"
    goal: float = 10.0
    collision: float = -10.0
    step: float = -0.01
    shaping_scale: float = 1.0
    gamma: float = 0.99
    naive_bonus: float = 0.1
    clearance_penalty: float = -0.05
    clearance_threshold: float = 0.3
    jerk_penalty: float = -0.01


@dataclass
class RewardTerms:
    goal: float = 0.0
    collision: float = 0.0
    shaping: float = 0.0
    clearance: float = 0.0
    step: float = 0.0
    jerk: float = 0.0

    @property
    def total(self) -> float:
        return self.goal + self.collision + self.shaping + self.clearance + self.step + self.jerk

    def as_dict(self) -> dict[str, float]:
        return {
            "goal": self.goal, "collision": self.collision, "shaping": self.shaping,
            "clearance": self.clearance, "step": self.step, "jerk": self.jerk,
            "total": self.total,
        }


def compute_reward(
    cfg: RewardConfig,
    *,
    prev_dist: float,
    dist: float,
    min_scan: float,
    omega: float,
    prev_omega: float,
    reached: bool,
    collided: bool,
) -> RewardTerms:
    """One step of reward, decomposed so each term can be logged separately."""
    t = RewardTerms()

    if reached:
        t.goal = cfg.goal
    if collided:
        t.collision = cfg.collision

    if cfg.shaping == "potential":
        # Phi(s) = -d(s);  F = gamma*Phi(s') - Phi(s)
        t.shaping = cfg.shaping_scale * (cfg.gamma * (-dist) - (-prev_dist))
    elif cfg.shaping == "naive":
        if dist < prev_dist:
            t.shaping = cfg.naive_bonus
        elif dist > prev_dist:
            t.shaping = -cfg.naive_bonus
    elif cfg.shaping != "none":
        raise ValueError(f"unknown shaping mode: {cfg.shaping!r}")

    if cfg.clearance_penalty != 0.0 and min_scan < cfg.clearance_threshold:
        t.clearance = cfg.clearance_penalty

    t.step = cfg.step
    t.jerk = cfg.jerk_penalty * abs(omega - prev_omega)
    return t


REWARD_VARIANTS: dict[str, RewardConfig] = {
    # The proposed design: policy-invariant shaping plus safety and smoothness.
    "potential": RewardConfig(shaping="potential"),
    # The intuitive scheme, reproduced faithfully so its failure is measurable.
    "naive": RewardConfig(shaping="naive"),
    # Bonus item: goal signal only.
    "sparse": RewardConfig(
        shaping="none", clearance_penalty=0.0, jerk_penalty=0.0, step=0.0
    ),
    # Isolates what the clearance term buys: expect wall-scraping without it.
    "no_clearance": RewardConfig(shaping="potential", clearance_penalty=0.0),
}
