## Perception Report: Agent Behavior Analysis

### 1. Behavior Trend Summary

- **At each evaluation milestone**, the agent consistently fails to complete the task (completion_rate = 0.0), never falls (fall_rate = 0.0), and always truncates episodes (truncation_rate = 1.0). Mean episode length remains flat at approximately 71–72 steps across all milestones.
- **Trajectory is flat**: no improvement in completion or avoidance of truncation. The agent has not learned to land on the pad.
- **Final key numbers**:
  - completion_rate = 0.0
  - fall_rate = 0.0
  - mean_length = 72.3

### 2. Critical Metrics

| Metric | Trend | Assessment |
|--------|-------|------------|
| **angle_deviation_mean** | Decreasing: 0.145 → 0.029 | Positive: agent learns to maintain near‑zero tilt. |
| **pad_distance_mean** | Stable ~1.03–1.04 | **Flat / no progress** – agent does not move toward the pad. |
| **leg_contact_sum_mean** | Decreasing: 0.060 → 0.017 | **Wrong direction** – agent makes less contact over time, indicating it avoids landing. |
| **vertical_speed_magnitude_mean** | Stable ~0.89–0.90 | Agent descends slowly but never reaches the pad (no contact increase). |
| **fuel_used_mean** | Slightly increasing: 0.029 → 0.047 | Very low, suggests minimal thruster use. |
| **angvel_change_mean** | Constant 0.0 | Agent holds a steady orientation (no angular velocity changes). |

**Red flags**: pad_distance not decreasing, leg_contact near zero, no landing attempts.

### 3. Reward Component Health

| Component | Mean | Std | % Total | Status |
|-----------|------|-----|---------|--------|
| r_contact | 0.0086 | 0.0189 | 0.2% | **Inactive** (mean ≈ 0) |
| r_efficiency | -0.0215 | 0.0304 | 0.4% | **Active** (non‑zero) |
| r_progress | -0.4783 | 0.0509 | 8.6% | **Active** |
| r_stability | -0.0277 | 0.0300 | 0.5% | **Active** |
| r_termination | -5.0000 | 0.0000 | 90.3% | **Dominant** (|mean| >> others) |

- **Dominant component**: `r_termination` (mean = −5.0, exactly constant, no variance) – 90.3% of total reward. This is a strong negative penalty applied upon episode termination.
- **Inactive**: `r_contact` (mean ≈ 0, negligible contribution).
- **Suspicious**: `r_termination` has zero standard deviation – the same −5.0 penalty is applied every episode, consistent with all episodes being truncated.
- The agent is overwhelmingly optimizing to avoid even larger penalties from not landing? No, the penalty is fixed −5 per termination; it is the single largest signal.

### 4. Behavioral Diagnosis

The agent has converged to a stable, cautious hovering strategy: it maintains near‑zero tilt, constant angular velocity, and very low fuel usage, while drifting slowly with a vertical speed of ~0.9. It avoids all leg contact and never approaches the landing pad, resulting in every episode being cut short by the time limit (truncation). This is a **local optimum** driven by the dominant termination penalty – the agent likely receives large negative rewards for attempting to land (deviating from safe hover) and has learned that staying aloft reduces those penalties, even though it never achieves completion. The policy has become nearly deterministic (entropy dropped to 0.176), indicating premature convergence to this suboptimal behavior.

### 5. Key Numbers for Budget Calculation

- **angle_deviation_mean**: 0.029381
- **angvel_change_mean**: 0.0
- **fuel_used_mean**: 0.047026
- **leg_contact_sum_mean**: 0.016598
- **pad_distance_mean**: 1.039403
- **vertical_speed_magnitude_mean**: 0.894964
- **fall_rate**: 0.0
- **mean_length**: 72.3
- **completion_rate**: 0.0