# Perception Report: Lunar Lander Training Evaluation

## 1. Behavior Trend Summary

At the final evaluation (1,000,000 timesteps), the agent achieves a **completion rate of 0.20**, a **fall rate of 0.00**, and a **truncation rate of 0.80**. The mean episode length is **816.4** (range 52–1991).  

The agent is **not falling** but is **frequently truncated** — meaning episodes are terminated by a time limit or other boundary before reaching the landing pad. The completion rate is low and highly variable across checkpoints: it jumped to 0.60 at 600k, then dropped to 0.10–0.20 afterward. This suggests the agent does **not consistently succeed** and may have regressed after a brief improvement. The mean length increases slightly at the end, indicating longer survival but not necessarily successful landing.

Overall, the trajectory is **not clearly improving**; it is unstable and has regressed from the 600k peak.

## 2. Critical Metrics

| Metric | Trend | Assessment |
|--------|-------|------------|
| **pad_distance_mean** | Decreasing from 0.64 to 0.37 | **Positive**: agent moves closer to pad over time. |
| **fuel_used_mean** | Increasing from 0.73 to 0.93 | **Negative**: fuel consumption rises, suggesting inefficient control or more thruster activity. |
| **angvel_change_mean** | Constant 0.0 throughout | **Suspicious**: zero angular velocity changes — may indicate a lack of rotational control or reward hacking (e.g., no torque commands). |
| **vertical_speed_magnitude_mean** | Low and variable (~0.08–0.13) | Ambiguous: agent may be moving slowly vertically, but not dangerously fast. |
| **leg_contact_sum_mean** | Very low (0.0–0.25) | **Negative**: agent rarely touches the ground with legs, consistent with not completing landings. |

**Wrong direction indicators**: Fuel usage is rising while completion rate is not improving; angvel_change = 0 is a red flag.

## 3. Reward Component Health

- **Active components** (mean significantly non-zero): `r_progress` (0.768), `r_contact` (0.176), `r_efficiency` (-0.068), `r_stability` (-0.054). All have non-zero means.
- **Dominant component**: `r_progress` accounts for **68.9%** of total magnitude, well above 2× others.
- **Inactive/negligible**: `r_termination` has a mean of 0.048 but a huge std (2.19), making it effectively noisy and unreliable.
- **Suspicious values**: None are exactly zero, but `angvel_change_mean` being 0 in environment metrics (not a reward) is suspicious — it may indicate the agent never changes angular velocity, possibly due to a quirk in action space or reward shaping.

## 4. Behavioral Diagnosis

The agent’s current strategy appears to be **hovering or slowly descending toward the pad without making firm contact**, leading to frequent timeouts (truncations) rather than successful landings or falls. It uses significant fuel but rarely touches the ground with its legs. The zero angular velocity change suggests it may have learned to **avoid any rotation commands** — a potential reward-hacking behavior (e.g., staying upright by doing nothing) at the cost of efficient descent and landing.

This is likely a **local optimum**: the agent avoids falling and gets some progress reward by moving horizontally, but it fails to complete the final landing phase (leg contact). The strategy is not genuinely robust.

## 5. Key Numbers for Budget Calculation

| Metric | Last recorded value (1,000k timesteps) |
|--------|----------------------------------------|
| angle_deviation_mean | 0.047181 |
| angvel_change_mean | 0.0 |
| fuel_used_mean | 0.926017 |
| leg_contact_sum_mean | 0.13976 |
| pad_distance_mean | 0.368301 |
| vertical_speed_magnitude_mean | 0.084975 |
| fall_rate | 0.000 |
| mean_length | 816.4 |
| completion_rate | 0.200 |

*(All values from 1,000k timestep unless otherwise noted.)*