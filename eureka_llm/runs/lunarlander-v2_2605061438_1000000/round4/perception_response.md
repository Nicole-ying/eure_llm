## Perception Report

### 1. Behavior Trend Summary

- **200k timesteps:** Agent falls immediately in every episode (fall_rate=1.0). It is unable to maintain altitude or control descent.
- **400k timesteps:** Agent begins to survive some episodes (20% completion). Falls still dominate (80% fall_rate). Altitude drops sharply from ~36 to ~2.7, indicating it is learning to descend but not yet land stably.
- **600k timesteps:** Major improvement — 80% completion rate, only 20% falls. Mean episode length drops to 361.8, suggesting faster, more efficient landings.
- **800k timesteps:** Peak performance — 100% completion, 0% falls. Mean episode length hits the max (1000.0), indicating the agent is surviving full episodes but may be hovering or delaying landing.
- **1M timesteps:** Slight regression — 90% completion, 10% falls. Mean length drops to 400.1, suggesting the agent is landing faster but with occasional failures.

**Final numbers:** completion_rate=0.900, fall_rate=0.100, mean_length=400.1. The trajectory shows strong improvement followed by minor regression at the end.

### 2. Critical Metrics

- **both_legs_down_mean:** Increased from 0.0 → 0.668 at peak, then dropped to 0.414. The agent learned to land on both legs but is regressing on consistent two-leg touchdowns.
- **distance_to_pad_mean:** Dropped from 38.3 → ~10.1 and stabilized. The agent consistently reaches the landing pad area.
- **speed_mean:** Dropped from 5.07 → 0.57 at peak, then rose to 1.32. The agent learned to slow down for landing but is moving faster again at 1M timesteps.

**Flagged:** both_legs_down_mean and speed_mean are moving in the wrong direction at the final checkpoint (regression from peak performance).

### 3. Reward Component Health

- **Active components:** _outcome (mean=-0.724), r_alive (mean=0.500), r_progress (mean=0.499)
- **Dominant components:** None (no component has |mean| > 2× the next highest)
- **Inactive/negligible:** r_distance (mean=-0.038) and r_speed (mean=-0.026) have very small magnitudes relative to other components
- **Suspicious values:** r_alive has mean=0.500 and std=0.000 — this is exactly constant, which is expected for a per-step survival bonus but worth noting

### 4. Behavioral Diagnosis

The agent has learned to descend to the landing pad and land successfully most of the time, but its strategy is inconsistent — it sometimes lands on both legs (67% at peak) but often lands on one leg or falls at the final moment (10% fall rate at 1M). The regression in both_legs_down and speed suggests the agent is optimizing for speed of completion over landing quality, potentially sacrificing stable touchdowns for faster descents.

### 5. Key Numbers for Budget Calculation

**Environment Metrics:**
- altitude_mean: 2.675
- altitude_std: 2.854
- angle_from_vertical_mean: 0.047
- angle_from_vertical_std: 0.070
- angular_velocity_mean: 0.088
- angular_velocity_std: 0.139
- both_legs_down_mean: 0.414
- both_legs_down_std: 0.493
- distance_to_pad_mean: 10.122
- distance_to_pad_std: 1.364
- speed_mean: 1.325
- speed_std: 1.437
- total_thrust_mean: 0.773
- total_thrust_std: 0.419

**Behavior Metrics:**
- fall_rate: 0.100
- mean_length: 400.1
- completion_rate: 0.900