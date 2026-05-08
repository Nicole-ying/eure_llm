# Perception Report: Lunar Lander Training Evaluation

## 1. Behavior Trend Summary

The agent is **consistently falling** in every episode across all evaluation milestones. The completion rate is exactly 0.000 throughout training, while the fall rate is a constant 1.000. The mean episode length stabilizes around 63–68 timesteps, indicating the agent falls shortly after takeoff or descent begins. The trajectory is **flat and non-improving** — no progress toward landing is observed.

**Final numbers:** completion_rate = 0.000, fall_rate = 1.000, mean_length = 63.3

## 2. Critical Metrics

| Metric | Trend | Assessment |
|--------|-------|------------|
| **altitude_mean** | Stable ~7.17 (increased slightly from 6.74) | Neutral — agent maintains moderate altitude before falling |
| **distance_to_pad_mean** | Increased from 11.78 to 12.68 | **Moving in wrong direction** — agent drifts away from landing pad |
| **both_legs_down_mean** | Constant 0.0 | **Moving in wrong direction** — never achieves landing configuration |
| **angle_from_vertical_mean** | Decreased from 1.79 to 1.19 | Slight improvement in orientation control, but insufficient |
| **speed_mean** | Stable ~7.9 | Moderate speed maintained, no deceleration toward landing |

## 3. Reward Component Health

- **Active components:** `_outcome` (mean = -0.98), `r_distance` (mean = -0.49)
- **Dominant component:** `_outcome` at 66.4% of total reward — heavily penalizing the agent for falling
- **Inactive/negligible:** `r_alive` (mean = 0.01, near zero) — provides minimal survival incentive
- **Suspicious values:** `both_legs_down_mean` is exactly 0.0 across all timesteps — the agent never achieves a landing posture

## 4. Behavioral Diagnosis

The agent's strategy is to **fall rapidly after a short flight** — it maintains moderate altitude and speed but never attempts to align with the landing pad or deploy landing legs. The agent is **stuck in a local optimum** where it receives a small positive `r_alive` reward for staying alive briefly, but the dominant negative `_outcome` penalty for falling overwhelms any learning signal. The agent is not reward hacking; it is simply failing to learn a viable landing policy.

## 5. Key Numbers for Budget Calculation

| Metric | Value |
|--------|-------|
| altitude_mean | 7.17 |
| altitude_std | 2.76 |
| angle_from_vertical_mean | 1.19 |
| angle_from_vertical_std | 1.13 |
| distance_to_pad_mean | 12.68 |
| distance_to_pad_std | 1.51 |
| speed_mean | 7.90 |
| speed_std | 4.75 |
| both_legs_down_mean | 0.0 |
| awake_mean | 1.0 |
| fall_rate | 1.000 |
| mean_length | 63.3 |
| completion_rate | 0.000 |