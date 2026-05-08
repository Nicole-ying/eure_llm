## Perception Agent Report

### 1. Behavior Trend Summary

- **At early milestones (200k–600k):** The agent never completes the task, always gets truncated (truncation_rate = 1.0). Episode length increases from ~70 to ~336 steps, suggesting it is learning to survive longer but not achieve the objective.
- **At intermediate milestone (800k):** Completion rate jumps to 0.5; fall rate remains 0; truncation rate drops to 0.5. Mean episode length grows to ~653 steps.
- **At final milestone (1M):** Completion rate stays at 0.5, truncation rate at 0.5, mean length ~601 steps. The agent is consistently completing half of the episodes without falling, and the other half are truncated (likely timed out or otherwise terminated before success).
- **Overall trajectory:** Improving from complete failure to 50% success, with no falls. The agent is making genuine progress toward landing.

**Final key figures:**  
- completion_rate = 0.500  
- fall_rate = 0.000  
- mean_length = 601.1

### 2. Critical Metrics

| Metric | Trend | Direction |
|--------|-------|-----------|
| **pad_distance_mean** | Steady decrease: 0.98 → 0.37 | ✅ Moving in the right direction (agent gets closer to pad) |
| **vertical_speed_magnitude_mean** | Sharp drop: 0.91 → 0.15 | ✅ Moving in the right direction (softer landings) |
| **leg_contact_sum_mean** | Increase: 0.027 → 0.602 | ✅ Moving in the right direction (more leg contact indicates proper landing posture) |

No metric is moving in a wrong direction. Fuel usage increases (0.16 → 0.82), which is expected as the agent learns to use thrust to control descent.

### 3. Reward Component Health

- **Active components:** All five components have mean values significantly non-zero.
  - r_contact (mean = 0.0136)
  - r_efficiency (mean = -0.0436)
  - r_progress (mean = -0.3734)
  - r_stability (mean = -0.0427)
  - r_termination (mean = -1.0000)
- **Dominant component:** r_termination accounts for 67.9% of total reward magnitude, far exceeding any other component (>2× the next largest, r_progress at 25.3%). The constant value of -1.0 per step suggests a stepwise survival penalty.
- **Inactive/negligible:** None. All components contribute appreciably.
- **Suspicious values:** r_termination has mean exactly −1.0 with zero standard deviation. This suggests it is a fixed per‑step penalty (e.g., time penalty), not a terminal reward. While not exactly zero, the “perfect” consistency is unusual and may reflect a design choice rather than learning dynamics.

### 4. Behavioral Diagnosis

The agent’s strategy is to descend toward the landing pad, reduce vertical speed, and make leg contact to achieve a soft landing. It succeeds in about half of the episodes; in the other half it likely runs out of time or drifts off course. The agent is making genuine progress—it is not reward hacking or stuck in a local optimum, as evidenced by decreasing pad distance, decreasing vertical speed, and increasing leg contact over training.

### 5. Key Numbers for Budget Calculation

| Metric | Value (final) | Source Table |
|--------|---------------|--------------|
| angle_deviation_mean | 0.090946 | Environment Metrics |
| angle_deviation_std | 0.105358 | Environment Metrics |
| angvel_change_mean | 0.0 | Environment Metrics |
| angvel_change_std | 0.0 | Environment Metrics |
| fuel_used_mean | 0.820496 | Environment Metrics |
| fuel_used_std | 0.383774 | Environment Metrics |
| leg_contact_sum_mean | 0.602229 | Environment Metrics |
| leg_contact_sum_std | 0.833495 | Environment Metrics |
| pad_distance_mean | 0.37286 | Environment Metrics |
| pad_distance_std | 0.33947 | Environment Metrics |
| vertical_speed_magnitude_mean | 0.145199 | Environment Metrics |
| vertical_speed_magnitude_std | 0.219299 | Environment Metrics |
| fall_rate | 0.000 | Behavior Metrics |
| mean_length | 601.1 | Behavior Metrics |
| completion_rate | 0.500 | Behavior Metrics |