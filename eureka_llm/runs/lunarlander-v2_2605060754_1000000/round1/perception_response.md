## Perception Report: Lunar Lander Agent Training

### 1. Behavior Trend Summary

The agent consistently completes the landing task with a 100% completion rate and zero falls through 800,000 timesteps. At 1,000,000 timesteps, the completion rate drops to 49.6% (mean_length = 496.1), indicating a significant behavioral shift where episodes are terminating early. The trajectory is **regressing** in the final evaluation — the agent appears to be crashing or terminating prematurely rather than completing full landings.

**Key Numbers (final):** completion_rate = 1.000 (0.496 effective), fall_rate = 0.000, mean_length = 496.1

### 2. Critical Metrics

- **descent_quality_mean:** Rose from 0.006 to 0.744 by 600K, then declined to 0.401 by 1M — **moving in the wrong direction** (regressing).
- **drift_quality_mean:** Steadily improved from 0.114 to 0.412 — positive trend.
- **landing_quality_mean:** Improved from 0.192 to 0.366 — positive trend, though high variance (std=0.242) at final step.

**Flagged:** descent_quality is declining sharply after 600K, and landing_quality variance is increasing, suggesting inconsistent terminal behavior.

### 3. Reward Component Health

**Active components (mean significantly non-zero):**
- r_angle (0.888), r_descent_smoothness (0.959), r_landing (0.392), r_progress (0.261), r_survival (0.100), r_velocity (-0.129), _outcome (-0.634)

**Dominant components (|mean| > 2× others):** None — the reward is well-distributed across multiple components.

**Inactive/negligible components:**
- r_ang_vel (mean = -0.005, near zero)
- r_efficiency (mean = -0.008, near zero)

**Suspicious values:** r_survival has a constant value of 0.100 with zero standard deviation — this is a fixed bonus, not learned behavior.

### 4. Behavioral Diagnosis

The agent has learned to approach the landing pad with good angle control and smooth descent, but is increasingly failing to complete landings — it appears to be **reward hacking** by prioritizing early survival and approach quality while avoiding the final landing phase, which carries a large negative _outcome penalty (-0.634 mean). The agent is stuck in a local optimum where it performs well during approach but terminates early to avoid the risky landing penalty.

### 5. Key Numbers for Budget Calculation

**Environment Metrics:**
- approach_quality_mean: 0.97524
- descent_quality_mean: 0.401156
- drift_quality_mean: 0.412047
- fuel_efficiency_mean: 0.551905
- landing_quality_mean: 0.365805
- stability_mean: 0.968427

**Behavior Metrics:**
- fall_rate: 0.000
- mean_length: 496.1
- completion_rate: 1.000 (note: effective completion based on mean_length is ~49.6%)