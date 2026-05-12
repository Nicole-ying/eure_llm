# Structured Observation Report

## 1. Behavior Trend Summary

At each evaluation milestone (200k to 1M timesteps), the policy applies near-maximum throttle (action_force ≈ 0.95–0.99) and keeps the car’s heading roughly 0.38 rad (≈22°) to the right. The velocity magnitude remains extremely low (~0.03 m/s), the distance to goal stays nearly constant (≈0.97–1.08), and the goal is never reached (reached_goal = 0.0). Episode length increases modestly from 71 to 99 steps, indicating the agent survives longer but makes no net progress toward the goal.

| Final Timestep (1M) | Value |
|---------------------|-------|
| mean_length         | 99.2  |
| action_force        | 0.954 |
| dist_to_goal        | 0.972 |
| heading             | 0.385 |
| reached_goal        | 0.0   |
| velocity_mag        | 0.030 |

**Trend**: The trajectory is **flat** – no improvement in goal-reaching, no meaningful reduction in distance, and barely any movement. The agent has locked into a low-velocity, constant-throttle strategy.

## 2. Critical Metrics

Three most important task-level metrics and their relevance:

- **dist_to_goal** – Direct measure of task progress. Its near-constant value (~0.97) shows the agent fails to reduce distance to the goal. **Moving in the wrong direction**: slight increase from 0.973 at 800k to 0.972 at 1M is essentially flat, but the overall stagnation is critical.
- **reached_goal** – The ultimate binary indicator. At 0.0 across all evaluations, the agent never completes the task. **No movement in the right direction**.
- **velocity_mag** – Should increase to build momentum. It remains extremely low (≈0.03), inconsistent with the high action_force (~0.95). **Cross‑metric pattern**: the ratio `velocity_mag / action_force` is ~0.03, meaning nearly all applied force yields negligible movement – likely because the car is on a slope or the engine is too weak to overcome inertia. This indicates a **local optimum** where the agent pushes forward but cannot accelerate.

## 3. Reward Component Health

| Component | Mean | Std | % of Total | Status |
|-----------|------|-----|------------|--------|
| _outcome  | 0.422 | 0.906 | 98.1% | Active, dominant, noisy (CV > 1) |
| r_goal    | 0.007 | 0.005 | 1.7% | Inactive (mean ≈ 0) |
| r_progress| -0.001 | 0.002 | 0.2% | Inactive (mean ≈ 0) |

- **Active components**: `_outcome` is the only significantly non-zero component. It is **dominant** (|mean| > 2× the sum of others).
- **Inactive**: `r_goal` and `r_progress` have mean ≈ 0, providing negligible shaping.
- **Suspicious values**: `_outcome` has a high mean (0.422) but is **weakly aligned with task progress** – the distance to goal does not decrease, and the goal is never reached. This suggests `_outcome` rewards something other than goal completion (e.g., survival time, staying near a reference point, or a constant per-step bonus). The agent may be exploiting this reward by simply not crashing or terminating early, rather than moving toward the flag.

## 4. Behavioral Diagnosis

The agent’s current strategy is **full-throttle forward driving** with negligible movement, never reaching the goal. It appears stuck in a **local optimum** where it receives a stream of positive `_outcome` rewards (likely a per‑step survival bonus) without needing to build momentum or approach the flag. This behavior is **high‑effort (max throttle) / no‑gain** – the agent consumes maximum action effort but achieves zero progress. It does **not** match the intended task of oscillating back and forth to build momentum; instead, it drives straight forward (heading ≈ 22°) but lacks the power to climb the hill. The low velocity despite high force hints at a **reward‑hacking interpretation**: the agent may have discovered that applying a constant force prevents episode termination (or earns a steady reward) without moving, effectively exploiting a missing velocity or distance constraint in the reward.

## 5. TDRQ Diagnosis

Overall TDRQ = 32.5/100. The low score is driven primarily by **component imbalance** (score 1.85) and **component inactivity** (score 33.33). Exploration health is perfect (100), so the entropy decline is not alarming. The reward **should be iterated** to add meaningful shaping (e.g., distance‑based progress, momentum bonuses) and to reduce the dominance of the `_outcome` component, which is not tied to task completion.

## 6. Constraint Violations Summary

Two principles are violated, ranked by urgency:

1. **Temporal consistency (medium severity)** – The `heading` metric drifted from early mean 0.12 to late mean 0.38 (relative drift 2.13). This indicates the policy’s steering behavior shifted over training, pointing to intra‑policy inconsistency. The agent changed from nearly neutral heading to a rightward bias, but without improving distance – a sign of unstable strategy selection.

2. **Termination exploitation (medium severity)** – Mean episode length = 93.2, max length = 616, ratio = 0.151. The average length is far below the maximum, suggesting the agent frequently ends episodes early (possibly by crashing or hitting a time limit) rather than continuing to exploit the reward structure. The relatively short average implies the agent may be avoiding longer episodes that require sustained effort.

## 7. Episode Consistency Summary

Early (200k) vs. late (1M) behavior shows a clear drift in `heading` (0.12 → 0.38) and a slight increase in `mean_length` (71 → 99). The policy style is **drifting** – the agent becomes more deterministic (entropy 1.17 → 0.91) and commits to a stronger rightward bias. This drift appears **unhealthy** because it does not correlate with progress (distance unchanged). It reflects a narrowing of the policy toward a suboptimal attractor rather than adaptive exploration.

## 8. Key Numbers for Budget Calculation

Extracted from Evaluation Metrics at final timestep (1,000,000):

- mean_length: 99.2
- action_force: 0.953889
- dist_to_goal: 0.972156
- heading: 0.385081
- reached_goal: 0.0
- velocity_mag: 0.030439