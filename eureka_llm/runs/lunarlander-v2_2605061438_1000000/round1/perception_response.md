## Perception Report: Lunar Lander Training Evaluation

### 1. Behavior Trend Summary

The agent is **consistently crashing** in every episode across all evaluation milestones. The completion rate is flat at **0.000**, and the fall rate is constant at **1.000**, meaning the agent always falls before landing. The mean episode length is stable around **58–64 timesteps**, indicating the agent survives briefly but fails to complete the landing sequence. There is **no improvement** over the course of training — the behavior is flat and stuck in a failure pattern.

**Final numbers:** completion_rate = 0.000, fall_rate = 1.000, mean_length = 58.2

### 2. Critical Metrics

| Metric | Trend | Assessment |
|--------|-------|------------|
| **both_legs_down_mean** | Constant at 0.0 | Never lands on legs — critical failure |
| **distance_to_pad_mean** | Stable ~12.4–13.1 | Agent stays far from landing pad |
| **speed_mean** | Stable ~7.9–8.3 | High speed maintained throughout |
| **total_thrust_mean** | Constant at 1.0 | Thrust always at maximum — no throttle modulation |

**Flagged:** `total_thrust_mean` is exactly 1.0 with zero variance — the agent is applying **full thrust at all times**, which is a clear sign of a stuck or degenerate policy.

### 3. Reward Component Health

- **Active components:** `_outcome` (mean = -0.979), `r_distance` (mean = -0.289), `r_alive` (mean = 0.05)
- **Dominant component:** `_outcome` at 74.3% of total reward — the negative terminal outcome penalty dominates the agent's experience
- **Inactive/negligible:** None — all components have non-zero means
- **Suspicious values:** `r_alive` has zero standard deviation (exactly 0.05 every step), which is unusual but may be by design. `total_thrust_mean = 1.0` with zero std is highly suspicious — the agent never varies thrust.

### 4. Behavioral Diagnosis

The agent has learned a **single degenerate strategy**: apply maximum thrust continuously and crash into the ground. It never attempts to modulate throttle, control orientation, or aim for the landing pad. This is a **local optimum** where the agent has discovered that constant full thrust keeps it alive slightly longer (avoiding immediate ground contact) but prevents any controlled descent or landing.

### 5. Key Numbers for Budget Calculation

| Metric | Value |
|--------|-------|
| altitude_mean | 6.733656 |
| altitude_std | 2.712102 |
| angle_from_vertical_mean | 0.984878 |
| angle_from_vertical_std | 0.947506 |
| angular_velocity_mean | 2.443548 |
| angular_velocity_std | 1.448609 |
| both_legs_down_mean | 0.0 |
| both_legs_down_std | 0.0 |
| distance_to_pad_mean | 12.431686 |
| distance_to_pad_std | 1.494275 |
| speed_mean | 8.339719 |
| speed_std | 4.03147 |
| total_thrust_mean | 1.0 |
| total_thrust_std | 0.0 |
| fall_rate | 1.000 |
| mean_length | 58.2 |
| completion_rate | 0.000 |