## Perception Report: Lunar Lander Training Evaluation

### 1. Behavior Trend Summary

- **200k timesteps:** Agent falls immediately in every episode (fall_rate=1.0). No completions. Mean episode length 626.7 indicates it is falling from a high altitude (mean altitude 12.77) without controlled descent.
- **400k timesteps:** Significant improvement — 90% completion rate, only 10% falls. Mean episode length increases to 929.8, suggesting the agent is learning to descend slowly and land.
- **600k timesteps:** Perfect performance — 100% completion, 0% falls. Mean length drops sharply to 330.0, indicating rapid, efficient landings.
- **800k timesteps:** Regression — completion drops to 70%, falls increase to 30%. Mean length rises to 424.2, suggesting less efficient landings.
- **1M timesteps:** Partial recovery — 80% completion, 20% falls. Mean length increases to 797.1, indicating slower, more cautious landings.

**Final state:** 80% completion, 20% fall rate, mean episode length 797.1. The trajectory is **unstable** — the agent achieved perfect performance at 600k but regressed and has not fully recovered.

### 2. Critical Metrics

| Metric | Trend | Flag |
|--------|-------|------|
| **both_legs_down_mean** | Increasing from 0.0 → 0.58 | ✅ Good — agent is learning to land on both legs |
| **altitude_mean** | Decreasing from 12.77 → 1.75 | ✅ Good — agent is staying closer to ground |
| **distance_to_pad_mean** | Decreasing from 14.28 → 10.37, but increased at 1M | ⚠️ Slight regression — drifting away from pad |

**Wrong direction:** distance_to_pad_mean increased from 9.97 (800k) to 10.37 (1M), and speed_mean dropped from 1.24 to 0.79, suggesting the agent is hovering or drifting rather than moving decisively toward the pad.

### 3. Reward Component Health

**Active components (mean significantly non-zero):**
- `r_progress` (mean=0.564) — strongest positive driver
- `_outcome` (mean=-0.583) — strongest negative driver (penalizing failures)
- `r_alive` (mean=0.500) — stable positive survival bonus
- `r_speed` (mean=-0.150) — moderate negative penalty
- `r_angle` (mean=-0.077) — small negative penalty
- `r_distance` (mean=-0.036) — small negative penalty

**Dominant components:** None exceed 2× the others. `_outcome` (30.5%) and `r_progress` (29.5%) are roughly equal in magnitude.

**Inactive/negligible:** None — all components have non-zero means.

**Suspicious values:** `r_alive` has mean=0.5 with std=0.0 — this is a constant per-step bonus, not suspicious.

### 4. Behavioral Diagnosis

The agent has learned to descend and land on the pad (80% completion), but its strategy is **unstable and inconsistent**. The high `r_progress` combined with negative `_outcome` suggests the agent is making progress toward the pad but frequently fails to execute a stable landing, resulting in falls. The increasing `both_legs_down` and decreasing altitude indicate genuine learning, but the regressing completion rate and rising distance-to-pad suggest the agent is **stuck in a local optimum** where it hovers near the pad but fails to commit to a final landing descent.

### 5. Key Numbers for Budget Calculation

**Environment Metrics:**
- altitude_mean: 1.746701
- altitude_std: 2.438536
- angle_from_vertical_mean: 0.037857
- angle_from_vertical_std: 0.077351
- angular_velocity_mean: 0.061761
- angular_velocity_std: 0.145256
- both_legs_down_mean: 0.580605
- both_legs_down_std: 0.493460
- distance_to_pad_mean: 10.367926
- distance_to_pad_std: 1.231144
- speed_mean: 0.790665
- speed_std: 1.273830
- total_thrust_mean: 0.651988
- total_thrust_std: 0.476340

**Behavior Metrics:**
- fall_rate: 0.200
- mean_length: 797.1
- completion_rate: 0.800