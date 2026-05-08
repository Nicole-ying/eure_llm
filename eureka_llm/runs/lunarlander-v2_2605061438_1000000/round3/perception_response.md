## Perception Report: Lunar Lander Training Evaluation

### 1. Behavior Trend Summary

The agent initially mastered the landing task, achieving a 100% completion rate with zero falls through 800,000 timesteps. However, at 1,000,000 timesteps, performance catastrophically collapsed: the completion rate dropped to 0% and the fall rate rose to 100%. The mean episode length dropped from 1,000 to 514.2, indicating the agent now falls early in episodes rather than completing landings.

**Trajectory:** Regressing sharply after sustained high performance.

**Key numbers:** final completion_rate=0.000, fall_rate=1.000, mean_length=514.2

### 2. Critical Metrics

- **altitude_mean:** Dropped from ~42 (200k) to ~2.7 (1M). The agent now stays very close to the ground, consistent with falling early.
- **distance_to_pad_mean:** Decreased from ~43.6 to ~6.97. The agent is closer to the pad on average, but this is because it falls near the pad rather than landing successfully.
- **speed_mean:** Increased from ~0.74 (800k) to 1.35 (1M). Speed is rising in the wrong direction, indicating loss of controlled descent.

**Flagged:** fall_rate increased from 0.0 to 1.0; completion_rate dropped from 1.0 to 0.0; speed_mean rising; angle_from_vertical_mean and angular_velocity_mean both increasing — all moving in the wrong direction.

### 3. Reward Component Health

- **Active components:** All five components have non-zero means.
- **Dominant component:** `r_landing` (mean=2.238, 67.5% of total reward) is the major driver, with high variability (std=3.18).
- **Inactive/negligible:** None are negligible, though `r_speed` (0.7%) and `r_distance` (1.3%) contribute very little.
- **Suspicious values:** `r_alive` has mean=0.5 with std=0.0 — exactly constant, which is expected for a survival bonus but worth noting.

### 4. Behavioral Diagnosis

The agent learned to land successfully and maintained that behavior for 800k timesteps, then suddenly collapsed into a strategy of falling immediately. This is consistent with **reward hacking or catastrophic forgetting** — the agent likely discovered that falling quickly yields a higher cumulative reward than completing the landing, possibly because the landing reward structure became misaligned with survival incentives during late training.

### 5. Key Numbers for Budget Calculation

**Environment Metrics:**
- altitude_mean: 2.732649
- altitude_std: 2.788028
- angle_from_vertical_mean: 0.109555
- angle_from_vertical_std: 0.103651
- angular_velocity_mean: 0.127386
- angular_velocity_std: 0.227862
- both_legs_down_mean: 0.171529
- both_legs_down_std: 0.37697
- distance_to_pad_mean: 6.968868
- distance_to_pad_std: 3.115251
- speed_mean: 1.352628
- speed_std: 1.071811
- total_thrust_mean: 0.657915
- total_thrust_std: 0.474408

**Behavior Metrics:**
- fall_rate: 1.000
- mean_length: 514.2
- completion_rate: 0.000