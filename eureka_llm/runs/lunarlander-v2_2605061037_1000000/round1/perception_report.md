# Perception Report: Lunar Lander Training Evaluation

## 1. Behavior Trend Summary

The agent consistently **falls immediately** at every evaluation milestone. Across all 5 checkpoints, the completion rate remains at **0.000**, and the fall rate stays at or near **1.000**. The agent survives for an average of 66–72 timesteps before falling, suggesting it cannot stabilize or land. The trajectory is **flat and non-improving** — there is no sign of learning progress.

- **Final completion_rate:** 0.000
- **Final fall_rate:** 1.000
- **Final mean_length:** 67.2 timesteps

## 2. Critical Metrics

| Metric | Mean | Trend |
|--------|------|-------|
| **distance_to_pad** | 12.27 | **Wrong direction** — agent stays far from landing pad (high distance) |
| **speed** | 7.40 | **Wrong direction** — agent maintains high speed, indicating no deceleration |
| **angle_error** | 0.12 | Moderate — agent is somewhat upright but not controlling orientation |

**Flagged:** Both `distance_to_pad` and `speed` are high and stable, indicating the agent is not attempting to approach or slow down for landing.

## 3. Reward Component Health

- **Active components:** `_outcome` (mean = -0.43), `r_shaped` (mean = -0.12), `r_alive` (mean = 0.02)
- **Dominant component:** `_outcome` accounts for **74.1%** of total reward magnitude — it is the primary driver
- **Inactive/negligible:** `r_landing` (mean = 0.007, very small) — landing behavior is essentially absent
- **Suspicious values:** None are exactly 0, but `r_landing` is effectively zero in practice

## 4. Behavioral Diagnosis

The agent's strategy is to **fall rapidly** — it moves at high speed away from the pad, does not decelerate, and crashes. This is **not reward hacking**; rather, the agent is stuck in a **local optimum** where it collects small positive `r_alive` rewards for a brief period before the large negative `_outcome` penalty for falling dominates. The agent is making **no genuine progress** toward landing.

## 5. Key Numbers for Budget Calculation

- **distance_to_pad:** 12.27
- **speed:** 7.40
- **angle_error:** 0.12
- **fall_rate:** 1.000
- **mean_length:** 67.2
- **completion_rate:** 0.000