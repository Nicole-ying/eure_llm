### 1. Behavior Trend Summary

- **Early (200k timesteps):** The agent never completes the task (completion_rate = 0.0), never falls (fall_rate = 0.0), and every episode ends by truncation (truncation_rate = 1.0). Mean episode length is 78.5 steps.
- **Later (400k–1M timesteps):** The pattern stabilizes: completion_rate remains 0.0, fall_rate 0.0, truncation_rate 1.0. Mean episode length settles at **72.3** steps.
- **Trend:** Flat, no improvement. The agent does not learn to land; it consistently triggers truncation before reaching the landing pad.
- **Final numbers:** completion_rate = 0.0, fall_rate = 0.0, mean_length = 72.3.

### 2. Critical Metrics

| Metric | Trend | Assessment |
|--------|-------|------------|
| **pad_distance_mean** | ~1.04, constant from 400k onward | **Wrong direction** – should decrease toward zero for a successful landing; instead it remains at a fixed offset. |
| **vertical_speed_magnitude_mean** | ~0.89, constant from 400k onward | Steady, non-zero vertical speed. Agent is not decelerating to land. |
| **angle_deviation_mean** | ~0.03, stable and small | Good – the agent maintains near‑upright orientation, but this does not translate to task completion. |

None of these metrics show progress toward the goal. Pad distance is the most critical failing metric.

### 3. Reward Component Health

- **Dominant component:** `r_termination` (mean = -15.0, 93.9% of total reward magnitude). It is constant and exactly -15.0, suggesting a fixed penalty applied at every truncation.
- **Active components (significantly non‑zero):** `r_progress` (mean = -0.9265), `r_stability` (mean = -0.0295), `r_efficiency` (mean = -0.0107). All are negative, indicating the agent is incurring penalties in every step.
- **Inactive/negligible components:** `r_contact` (mean = 0.008, std = 0.019) – very small positive signal, likely spurious.
- **Suspicious mean:** `r_termination` has zero variance (std = 0.0), always exactly -15.0 – it is a constant termination penalty, not learned.

The reward structure is dominated by the termination penalty, which overwhelms all other signals. The agent appears to optimize for avoiding even larger negative values by ending episodes quickly at a stable, unchanging state.

### 4. Behavioral Diagnosis

The agent has converged to a **stuck local optimum**: it maintains a constant posture (low angle deviation, no angular velocity changes) at a fixed distance from the landing pad (~1.04 m) while slowly descending (vertical speed ~0.89 m/s). It never activates leg contact and never reduces pad distance, so it cannot complete the landing. Instead, it reliably triggers truncation (likely after time limit or distance condition) and collects the large negative termination reward. This strategy avoids falling (fall_rate=0) but makes no genuine progress toward the objective.

### 5. Key Numbers for Budget Calculation

| Metric | Final Value (1M timesteps) |
|--------|----------------------------|
| angle_deviation_mean | 0.031136 |
| angle_deviation_std | 0.01959 |
| angvel_change_mean | 0.0 |
| angvel_change_std | 0.0 |
| fuel_used_mean | 0.048409 |
| fuel_used_std | 0.21463 |
| leg_contact_sum_mean | 0.019364 |
| leg_contact_sum_std | 0.1378 |
| pad_distance_mean | 1.039385 |
| pad_distance_std | 0.374536 |
| vertical_speed_magnitude_mean | 0.894868 |
| vertical_speed_magnitude_std | 0.543527 |
| fall_rate | 0.0 |
| mean_length | 72.3 |
| completion_rate | 0.0 |