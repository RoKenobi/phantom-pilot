# Mobile Robot Navigation with Simulated LiDAR

**A differential-drive robot learning to navigate 2D floor plans from raw range sensing.**

Design document written before implementation. Several of its predictions were later
contradicted by the measured results; see `README.md`.

---

## 0. Locked decisions

| Decision | Choice |
|---|---|
| **Algorithms** | PPO and Double DQN **implemented from scratch in PyTorch**. No Stable-Baselines3 — the point is to make the fundamentals visible in the code |
| **Version control** | None. Plain files in the project directory |
| **Results presentation** | `README.md` as the written report, carrying figures produced by command-line scripts. No notebook |
| **Audience framing** | Balanced across **robotics, ML and RL** — the sim-to-real reasoning, the engineering quality, and the experimental rigour all carry equal weight |
| **Compute** | CPU-only (no CUDA on this machine), 16 cores. Networks are small by design; throughput comes from vectorised envs across cores |

---

## 1. The concept

A TurtleBot3-style differential-drive robot navigates a continuous 2D floor plan to reach a
target while avoiding walls and obstacles. Crucially, it **does not see the world** — it
perceives only:

- `N` simulated LiDAR range readings
- its relative bearing and distance to the goal

No physics engine. No ROS. No Gazebo. Just unicycle kinematics and vectorised ray–segment
intersection — roughly 250 lines of NumPy for the whole simulator.

### Why this instead of a GridWorld

A simple custom GridWorld would be enough to pose this problem. This goes further, for
three reasons:

1. **It changes the problem class.** A GridWorld hands the agent full ground-truth state — an
   MDP. LiDAR gives a local, noisy, ambiguous slice of the world — a **POMDP**. That single
   change unlocks an entire category of analysis (perceptual aliasing, frame stacking,
   recurrent policies) that a GridWorld cannot access.
2. **It is closer to real robotics.** Every design choice maps onto a real decision a robotics
   engineer makes, which makes the "explain your trade-offs" criterion easy to answer well.
3. **It demos better.** A robot tracing a smooth arc through a corridor with LiDAR rays fanning
   out in front of it is a far stronger 20-second video than a square moving on a grid.

### Staying spec-compliant

The floor plan is **generated from an occupancy grid**, then occupied cells are converted to
wall segments. So the map genuinely *is* a gridworld — the robot simply moves continuously
through it with realistic sensing. This also hands us a free discrete grid for the A\* oracle
baseline to plan on.

---

## 2. The framing narrative

Everything in this project hangs off one idea, stated up front in the README:

> The classical robot navigation stack has two layers: a **global planner** (A\*/Dijkstra) that
> computes an optimal path over a known map, and a **local reactive controller** (DWA, potential
> fields) that avoids obstacles using live sensor data. My baselines are exactly those two
> layers. The RL agent is a learned, end-to-end replacement for both, operating from raw sensor
> readings with no map.

This turns a generic "compare against a baseline" exercise into a real
research question, and gives every result a reason to exist.

---

## 3. Environment design

### 3.1 World representation

| Property | Choice |
|---|---|
| Space | Continuous 2D, ~10 m × 10 m |
| Geometry | Walls as line segments, derived from an occupancy grid |
| Robot | Circular, radius ≈ 0.105 m (TurtleBot3 Burger) |
| Timestep | `dt = 0.1 s` |
| Episode cap | 500 steps |

Maps are **procedurally generated** (random rooms, corridors, obstacle scatter) with a fixed
seed per map, plus a small set of hand-designed maps held out for evaluation.

### 3.2 Kinematics — differential drive

The action is a pair of **wheel velocities** `(v_L, v_R)`, converted internally to body twist:

```
v = (v_R + v_L) / 2
ω = (v_R − v_L) / L          L = wheelbase = 0.16 m
```

then integrated with the unicycle model:

```
x'  = x + v·cos(θ)·dt
y'  = y + v·sin(θ)·dt
θ'  = θ + ω·dt
```

Exposing wheels rather than `(v, ω)` costs three extra lines and makes the robot genuinely
differential-drive rather than a unicycle wearing a TurtleBot costume.

Velocity limits use the real TurtleBot3 Burger maxima: `v ≤ 0.22 m/s`, `|ω| ≤ 2.84 rad/s`.

### 3.3 Simulated LiDAR

Modelled on the real **LDS-01** sensor shipped with the TurtleBot3:

| Parameter | Value |
|---|---|
| Rays | 24 (configurable to 36) |
| Field of view | 270° |
| Max range | 3.5 m |
| Min range | 0.12 m |
| Noise | Gaussian, σ ≈ 1 % of range |
| Dropout | ~2 % of rays return max range |

Implemented as a single **vectorised ray–segment intersection** over all rays × all wall
segments, taking the nearest hit per ray. This must not be a Python loop — throughput is the
main performance constraint in the whole project.

Noise and dropout are deliberate: a sim-to-real argument is not credible if the sensor is
perfect.

### 3.4 Observation space

```
obs = [ r_1 … r_24 | d_goal | sin(ψ) | cos(ψ) | v_prev | ω_prev ]        dim = 29
```

- `r_i` — LiDAR ranges, normalised to `[0, 1]` by max range
- `d_goal` — Euclidean distance to goal, normalised by map diagonal
- `ψ` — heading error (angle between robot heading and goal bearing)
- `v_prev, ω_prev` — previous commanded twist

Two deliberate details:

- **`sin(ψ)`/`cos(ψ)` instead of raw `ψ`.** The raw angle has a discontinuity at ±π that forces
  the network to learn a wraparound it should never have to see.
- **Previous action in the observation.** Required to define a smoothness penalty, and gives the
  policy a sense of its own motion.

### 3.5 Action space — two modes, one flag

| Mode | Space | Algorithm |
|---|---|---|
| **Discrete** | 6 actions: forward, fwd-left, fwd-right, spin-left, spin-right, stop | **DQN** |
| **Continuous** | `Box(v ∈ [0, 0.22], ω ∈ [−2.84, 2.84])` | **PPO** |

The same environment supports both. This is where the algorithm comparison comes from for free:
the question is no longer the arbitrary "DQN or PPO?" but the concrete **"what does discretising
the control space cost you in trajectory smoothness?"** — answerable with two plotted
trajectories side by side.

### 3.6 Termination

| Condition | Outcome |
|---|---|
| Robot centre within 0.25 m of goal | success, terminate |
| `min(scan) < robot_radius` (collision) | failure, terminate |
| 500 steps elapsed | timeout, truncate |

Gymnasium API throughout (`reset`/`step`, `terminated` vs `truncated` handled correctly).

---

## 4. Reward design — the centrepiece

### 4.1 The problem with the suggested reward

The intuitive choice is `move closer +0.1`. This is **naive distance shaping, and it is not
policy-invariant.** A robot rounding an obstacle must often move *away* from the goal to make
progress; naive shaping actively penalises the correct behaviour, producing wall-hugging and
dithering at concave corners.

The principled form is **potential-based reward shaping** (Ng, Harada & Russell, 1999), which is
*provably* optimal-policy-preserving:

```
F(s, s') = γ·Φ(s') − Φ(s)        where  Φ(s) = −d_goal(s)
```

Implementing both and demonstrating the naive variant's failure mode is the highest-signal
result in the project — it critiques the prompt with theory *and* evidence.

### 4.2 Proposed reward function

| Term | Value | Rationale |
|---|---|---|
| Reach goal | `+10` | task completion |
| Collision | `−10`, terminate | safety |
| Potential shaping | `γΦ(s') − Φ(s)`, `Φ = −d_goal` | dense guidance, policy-invariant |
| Clearance penalty | `−0.05` per step while `min(scan) < 0.3 m` | discourages wall-scraping |
| Step penalty | `−0.01` | urgency |
| Angular jerk | `−0.01·|ω − ω_prev|` | smooth, hardware-realistic motion |

### 4.3 Reward ablations

1. **Naive `+0.1` closer** vs **potential-based shaping** — expect dithering and distorted paths
   in the naive case.
2. **With vs without clearance penalty** — the visual payoff. Without it the agent learns to
   scrape along walls (locally optimal, alarming on real hardware); with it, it drives down the
   centreline of corridors. Two trajectory overlays make the argument instantly.
3. **Sparse** (goal reward only) vs **shaped** — isolates the sparse-reward regime
   and quantifies exactly what shaping buys.

---

## 5. Algorithms

Both **written from scratch in PyTorch** — no Stable-Baselines3. The point is for the GAE
recursion, the clipped surrogate objective, and the target-network update to be readable in
the source rather than hidden behind `model.learn()`.

Both sit on shared infrastructure (rollout/replay buffers, logging, eval harness):

- **PPO** on the continuous action space — the headline agent. Clipped surrogate objective, GAE(λ),
  entropy bonus, Gaussian policy with state-independent log-std, vectorised envs.
- **DQN** on the discrete action space — the comparison. Double DQN, target network, ε-greedy
  decay, uniform replay.

The comparison is framed as a genuine trade-off: on-policy stability and native continuous
control versus off-policy sample efficiency and simpler tuning.

### 5.0 Validating the implementations first

Before either algorithm touches the robot environment, both are verified against a **known-good
reference task** (`CartPole-v1` for DQN, `Pendulum-v1` for PPO) and must reach the standard
solve threshold. This is non-negotiable: without it, a failure to learn on the navigation task is
ambiguous between "the task is hard", "the reward is wrong" and "my PPO has a sign error". The
reference runs are cheap, and they turn every later debugging session from guesswork into
bisection.

### 5.1 Network architecture

The LiDAR scan is a spatially structured angular signal, not a flat vector — so:

```
scan (24,) ──► Conv1D (circular padding) ──► flatten ──┐
                                                       ├──► MLP ──► π, V  /  Q
goal + odom features (5,) ─────────────────────────────┘
```

**Circular padding** respects the angular wraparound of the scan. The whole network is small
enough to train comfortably on CPU.

---

## 6. Baselines

Three, forming a floor-to-ceiling bracket:

| Baseline | What it is | Expected failure |
|---|---|---|
| **Random** | Uniform samples from the action space | Essentially never reaches the goal |
| **Reactive controller** | Go-to-goal attraction + LiDAR-based repulsion (potential fields / VFF) — a stand-in for DWA | **Local minima**: trapped in concave obstacles and U-shaped rooms. The textbook potential-field failure, fully reproducible |
| **A\* oracle** | Plans on the underlying occupancy grid with full map knowledge | None — this is the ceiling, and supplies the optimal path length `ℓ` |

The oracle is what makes the evaluation rigorous: it converts "average steps taken" (which is
meaningless across maps of differing difficulty) into a **normalised optimality ratio**.

---

## 7. Evaluation

### 7.1 Metrics

Core metrics:

- Success rate
- Average episode reward
- Average steps to target
- Training reward/success curves
- Comparison against baselines

Added, and standard in the navigation literature:

- **SPL** — Success weighted by Path Length (Anderson et al., 2018), the standard embodied-
  navigation metric:
  ```
  SPL = (1/N) · Σᵢ  Sᵢ · ℓᵢ / max(pᵢ, ℓᵢ)
  ```
  where `Sᵢ` is success, `ℓᵢ` the A\* optimal length, `pᵢ` the path actually taken.
- **Collision rate** and **timeout rate** (decomposing failure, rather than lumping it)
- **Minimum clearance** — closest approach to any obstacle, a safety proxy
- **Path smoothness** — mean `|Δω|`, a sim-to-real proxy

### 7.2 Protocol

- **5 seeds per configuration**, reporting mean ± std bands on every curve. Single-seed RL
  results are noise; saying so explicitly is itself evidence of understanding.
- Evaluation on a **held-out set of maps never seen during training**.
- Fixed evaluation episode set across all agents and baselines, so comparisons are paired.

---

## 8. Partial observability — the differentiator

LiDAR cannot distinguish two identical-looking corridors. This is **perceptual aliasing**, and it
makes the task formally a POMDP. Three treatments, in increasing order of power:

| Treatment | Expectation |
|---|---|
| **Single frame** | Oscillates in symmetric corridors; cannot infer its own velocity |
| **Frame stacking (k=4)** | Recovers velocity and short-horizon history; fixes most oscillation |
| **GRU policy** | True memory over the episode |

At minimum the first two get run, with the single-frame failure captured as a GIF. Very few
navigation projects raise this at all, and here it falls naturally out of the environment
choice rather than being bolted on.

---

## 9. Extensions beyond the core task

| Bonus | How it lands |
|---|---|
| **Sparse reward** | Goal-only reward variant, shaping disabled (already an ablation arm) |
| **Curriculum learning** | Empty room → scattered obstacles → dense clutter → narrow corridors → distant goals, gated on rolling success rate |
| **Dynamic obstacles** | Moving "pedestrians" with simple constant-velocity or waypoint motion — the central real-world robotics problem |
| **Exploration strategy** | ε-decay schedules (DQN) vs entropy coefficient (PPO), compared directly |
| **Generalization to unseen maps** | **Effectively free.** LiDAR observations are egocentric and map-agnostic, so a policy trained on procedurally generated floor plans transfers to held-out maps — including a hand-designed apartment layout |

The generalization row is the quiet win: it isn't an add-on, it's a consequence of the
observation design.

---

## 10. Repository structure

```
RL-1/
├── README.md                  # the write-up — the main deliverable
├── idea.md                    # this document
├── requirements.txt           # pinned, reproducible
├── env/
│   ├── robot_env.py           # Gymnasium env: kinematics, termination, reward
│   ├── lidar.py               # vectorised ray casting
│   ├── maps.py                # occupancy grid → wall segments; procedural generation
│   └── render.py              # matplotlib rendering → frames → GIF/MP4
├── agents/
│   ├── ppo.py                 # PPO (continuous)
│   ├── dqn.py                 # Double DQN (discrete)
│   └── networks.py            # Conv1D + MLP policy/value/Q heads
├── baselines/
│   ├── random_agent.py
│   ├── reactive.py            # potential-field / VFF controller
│   └── astar_oracle.py        # grid planner, supplies optimal path lengths
├── experiments/
│   ├── validate_algos.py      # §5.0 — PPO/DQN vs CartPole-v1 / Pendulum-v1 sanity gate
│   ├── train.py               # config-driven training entry point
│   ├── evaluate.py            # fixed eval protocol, all metrics
│   ├── ablations.py           # reward / observation / algorithm sweeps
│   └── configs/               # one YAML per experiment arm
├── results/
│   ├── plots/                 # training curves, ablation bars, trajectory overlays
│   ├── videos/                # success + failure-case demos
│   └── metrics.json           # raw numbers behind every figure
└── models/                    # saved checkpoints
```

---

## 11. Design goals → where they are met

| Requirement | Where |
|---|---|
| Player/agent, target, obstacles | §3 — robot, goal, wall segments |
| Optional hazards/enemies | §9 — dynamic pedestrians |
| Observation/state space defined | §3.4 |
| Action space defined | §3.5 |
| Reward function defined | §4.2 |
| Trained with PPO / DQN | §5 — both |
| Baseline (random or heuristic) | §6 — three, exceeding the requirement |
| Comparison against baseline | §7 |
| Success rate, avg reward, avg steps | §7.1 |
| Training curves | §7.1 |
| README explaining all of the above | §10 |
| Saved model / reproducible instructions | §10 — `models/`, pinned `requirements.txt` |
| Demo video | §10 — `results/videos/` |

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **From-scratch PPO/DQN may contain subtle bugs** (wrong GAE sign, missing advantage normalisation, target-network update off by one) | §5.0 — both algorithms must solve `CartPole-v1` / `Pendulum-v1` before being pointed at the robot task, so algorithm bugs are ruled out before task difficulty is in play |
| Continuous control is harder to learn than a GridWorld | Curriculum; dense potential shaping; start on small maps; the discrete/DQN arm is the fallback if PPO struggles |
| Ray casting too slow → training crawls | Fully vectorised NumPy intersection; parallel envs across the 16 available cores; profile early |
| Reward hacking (e.g. spinning in place to farm shaping) | Potential-based shaping is policy-invariant by construction; step penalty; monitor for degenerate behaviours explicitly |
| Scope sprawl across many ablations | Every ablation must produce one specific figure that answers one specific question — no exploratory runs kept |
| CPU-only training (no CUDA available) | Small networks by design; 16 cores for vectorised envs; this problem does not need a GPU |
| `torch 2.3.0` + `numpy 2.5.1` ABI mismatch | Clean virtual environment with pinned, verified versions — reproducibility is a listed deliverable anyway |

---

## 13. Key references to cite in the README

- Ng, Harada & Russell (1999) — *Policy Invariance Under Reward Transformations* (potential-based shaping)
- Anderson et al. (2018) — *On Evaluation of Embodied Navigation Agents* (SPL)
- Schulman et al. (2017) — *Proximal Policy Optimization*
- van Hasselt et al. (2016) — *Deep RL with Double Q-learning*
- Fox, Burgard & Thrun (1997) — *The Dynamic Window Approach* (reactive baseline context)
