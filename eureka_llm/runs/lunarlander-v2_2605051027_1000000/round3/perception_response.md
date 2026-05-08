### 1. Behavior Trend Summary

- **200k–400k timesteps:** The agent neither completes nor falls; all episodes are truncated (timeout). Mean episode length increases from 261 to 596, indicating the agent is surviving longer but not making progress toward landing.
- **600k timesteps:** First completions appear (60% completion rate). Remaining 40% are truncations. Mean length peaks at 856 – the agent is both completing and timing out.
- **800k–1M timesteps:** Completion rate drops to 30% and stabilises. Fall rate remains zero. Mean length declines to 712 and then 777, suggesting episodes are ending earlier.
- **Overall trajectory:** The agent initially learned to land (60% completion at 600k) but then regressed and plateaued at 30% completion. It never falls, so it either lands or times out. The trend is flat to slightly regressing.

**Key numbers at 1M timesteps:**  
completion_rate = 0.300, fall_rate = 0.000, mean_length = 712.6

### 2. Critical Metrics

- **Pad distance (mean):** 0.757 → 0.396 (improvement). The agent consistently gets closer to the pad.
- **Vertical speed magnitude (mean):** 0.218 → 0.094 (improvement). The agent descends slower and more controlled.
- **Leg contact sum (mean):** 0.044 → 0.255 (improvement but still low). More contact with legs, but not sustained—std is high (0.631), meaning sporadic touchdown.

**Flagged metric:**  
**Angular velocity change (mean and std) = 0 at all checkpoints.** The agent never generates any angular velocity throughout training. This is a strong indicator that the agent is not using torque/rotation actions. This likely limits landing precision (e.g., cannot tilt to align legs with ground).

### 3. Reward Component Health

- **Active components (mean significantly non-zero):**  
  - `r_contact` (0.20) – positive, encouraging contact.  
  - `r_efficiency` (−0.07) – negative, penalising fuel use or movement.  
  - `r_progress` (0.79) – positive, rewarding movement toward pad.  
  - `r_stability` (−0.06) – negative, penalising instability.  
  - `r_termination` (−1.00) – constant negative per step (episode length penalty).

- **Dominant components (|mean| > 2× others):**  
  - `r_termination` (−1.00) is dominant in magnitude—its absolute mean is roughly 5× that of any other component.  
  - `r_progress` (0.79) is the dominant positive component.

- **Inactive/negligible components:** None; all are active. However, `r_efficiency` and `r_stability` have small magnitudes and high standard deviations relative to their means, but they are not zero.

- **Suspicious values:**  
  - `r_termination` has mean = −1.0 and std = 0.0, meaning every step incurs exactly the same penalty. This is expected by design.  
  - `r_efficiency` and `r_stability` are negative on average, indicating the agent is consistently incurring these penalties.

### 4. Behavioral Diagnosis

The agent’s current strategy is to approach the pad slowly (low vertical speed, decreasing distance) while avoiding falls, but it fails to consistently achieve touchdown—likely because it never uses torque to orient itself for a stable landing. The agent is stuck in a local optimum: it hovers near the pad, uses nearly all fuel, and times out on 70% of episodes, earning a small progress reward to offset the large per-step termination penalty. The zero angular velocity change indicates a missing degree of freedom, preventing it from reliably completing the landing.

### 5. Key Numbers for Budget Calculation

| Metric | Value at 1M timesteps |
|--------|----------------------|
| angle_deviation_mean | 0.06534 |
| angle_deviation_std | 0.090274 |
| angvel_change_mean | 0.0 |
| angvel_change_std | 0.0 |
| fuel_used_mean | 0.94485 |
| fuel_used_std | 0.228273 |
| leg_contact_sum_mean | 0.255403 |
| leg_contact_sum_std | 0.630541 |
| pad_distance_mean | 0.396319 |
| pad_distance_std | 0.280248 |
| vertical_speed_magnitude_mean | 0.09377 |
| vertical_speed_magnitude_std | 0.147411 |
| fall_rate | 0.0 |
| mean_length | 712.6 |
| completion_rate | 0.300 |