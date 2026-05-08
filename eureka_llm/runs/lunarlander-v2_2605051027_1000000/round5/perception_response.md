## Perception Agent Report

### 1. Behavior Trend Summary

- **200k timesteps:** The agent never completes the task (0% completion). All episodes are truncated (100% truncation) after a mean length of 388 steps. No falls occur.
- **400k timesteps:** Half of the episodes result in successful completion (50% completion), the other half are truncated. Mean episode length increases sharply to 903, indicating the agent can sustain longer trajectories.
- **600k timesteps:** High completion rate achieved (90%), with only 10% truncated. Mean length peaks at 976 steps. No falls at any point.
- **800k timesteps:** Regression: completion rate drops to 60%, truncation rate rises to 40%. Mean length decreases to 770 steps.
- **1,000k timesteps:** Recovery to 90% completion, low truncation (10%), mean length 946 steps. **Final performance is strong with no falls.**

**Overall trajectory:** Improving but non‑monotonic – the agent learns to land reliably (90% completion) after an initial failure period and a mid‑training regression. Final key numbers: **completion_rate = 0.9, fall_rate = 0.0, mean_length = 945.8**.

### 2. Critical Metrics

1. **pad_distance_mean** – Decreases from 0.88 (200k) to a low of 0.37 (800k), then rises to **0.64 at 1M**. This increase at the final evaluation is a **regression**: the agent is landing further from the pad center than it was earlier, despite high completion.
2. **fuel_used_mean** – Steadily increases from 0.62 to **0.96 at 1M**. The agent is using nearly all available fuel on average, indicating a fuel‑inefficient strategy.
3. **vertical_speed_magnitude_mean** – Remains low (0.06–0.13) throughout, final value **0.065**. Controlled descent speed – moving in the right direction initially but stable since 400k.

**Flagged red metric:** **pad_distance_mean** increased sharply from 0.37 (800k) to 0.64 (1M), suggesting a degradation in landing precision.

### 3. Reward Component Health

- **Active components** (mean significantly non‑zero):  
  `r_contact` (0.187), `r_efficiency` (-0.074), `r_landing_shaping` (0.724), `r_progress` (0.556), `r_stability` (-0.043), `r_termination` (0.050). All six components have non‑zero means and non‑zero standard deviations.

- **Dominant components** (|mean| > 2× any other):  
  None. The largest magnitude is `r_landing_shaping` (0.724), next is `r_progress` (0.556) – ratio ≈1.3, below the 2× threshold.

- **Inactive/negligible components**: None. Every component contributes at least ~2.7% of total reward.

- **Suspicious values**: No mean is exactly 0 or suspiciously constant. All standard deviations are non‑zero.

### 4. Behavioral Diagnosis

The agent has learned to consistently land (90% completion) by descending gently (low vertical speed, low angle deviation) while consuming nearly all available fuel. Its strategy relies heavily on landing‑shaping and progress rewards, but final‑approach precision is inconsistent (pad distance rises at the last evaluation). The mid‑training regression (800k) may reflect exploratory drift or a temporary local optimum. Overall, the agent is making genuine progress toward the goal, though fuel efficiency and landing accuracy need improvement.

### 5. Key Numbers for Budget Calculation

**Environment Metrics (final values at 1,000k timesteps):**
- angle_deviation_mean = 0.032189
- angle_deviation_std = 0.043035
- angvel_change_mean = 0.0
- angvel_change_std = 0.0
- fuel_used_mean = 0.961831
- fuel_used_std = 0.191603
- leg_contact_sum_mean = 0.002538
- leg_contact_sum_std = 0.059903
- pad_distance_mean = 0.639689
- pad_distance_std = 0.200113
- vertical_speed_magnitude_mean = 0.06521
- vertical_speed_magnitude_std = 0.100775

**Behavior Metrics (final at 1,000k timesteps):**
- fall_rate = 0.000
- mean_length = 945.8
- completion_rate = 0.900