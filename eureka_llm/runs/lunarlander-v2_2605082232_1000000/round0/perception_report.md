## Perception Report: Training Outcome Analysis

### 1. Behavior Trend Summary

- **What the agent is doing:** At all evaluation milestones, the agent maintains `alive=1.0` (never dies), remains at roughly constant distance from the pad (`distance_to_pad ≈ 1.0`), with a slight negative tilt (`angle ≈ -0.1 rad`), and a steadily negative vertical speed (descending). Leg contact is negligible (`~0.017`), meaning it never establishes stable contact. The agent is essentially **hovering in a slowly descending state but never reaching the pad** – it survives but does not complete the landing task.
- **Trajectory direction:** The mean episode length increases from ~66 to ~76 steps, indicating the agent learns to stay alive longer. However, the key task metric `distance_to_pad` remains flat (no reduction) and `vertical_speed` becomes less negative (i.e., descending slower), which actually moves the agent *away* from a landing. Overall progress is **flat or regressing** – the agent is not solving the objective.
- **Key numbers at 1M timesteps:** `mean_length=76.2`, `alive=1.0`, `angle=-0.104`, `distance_to_pad=1.027`, `leg_contact=0.018`, `vertical_speed=-0.820`.

### 2. Critical Metrics

| Metric | Trend | Flag |
|--------|-------|------|
| **distance_to_pad** | Flat at ~1.0 (0.96 → 1.03) – no movement toward target | ❌ Wrong direction (should decrease) |
| **vertical_speed** | Increasing from -0.97 to -0.82 (less negative) – descending slower | ❌ Wrong direction (should become less negative only if landing, but here it indicates "floating") |
| **mean_length** | Increasing from 66 to 76 – agent survives longer | ✅ Positive, but alone insufficient |
| **leg_contact** | Consistently very low (~0.014-0.018) – no landings | ⚠️ Stagnant |

**Conclusion:** No improvement on the actual task metrics; the agent is exploiting the alive reward without reducing distance.

### 3. Reward Component Health

- **Active components (mean significantly non-zero):**
  - `r_alive` (mean +0.05, 29.8% of total) – positive, constant.
  - `r_distance` (mean -0.095, 56.7% of total) – large negative penalty.
  - `r_leg_contact` (mean +0.016, 9.6%) – positive but small.
- **Dominant components:** None dominates (>80% of total). However `r_distance` is the largest by far (~2× the next) but not technically dominant.
- **Inactive/negligible:** `_outcome` (mean -0.0065, 3.9%, stable) – effectively spurious.
- **Suspicious values:** None; all means are non-zero but reasonable. `r_alive` std = 0 (constant) is expected.

### 4. Behavioral Diagnosis (1–2 sentences)

The agent’s strategy is to **survive indefinitely** by hovering (staying upright, slowly descending) while collecting the per-step `r_alive` bonus. It is **stuck in a local optimum** – avoiding the large negative `r_distance` penalty that would come from moving toward the pad, and never attempting the risky landing that would trigger `r_leg_contact` and `_outcome` rewards.

### 5. TDRQ Diagnosis

- **TDRQ = 55.72 (mixed).** The low score is driven primarily by **component imbalance** (43.26) – `r_distance` dominates the reward signal negatively, while the positive `r_alive` is constant and `r_leg_contact` is weak. Exploration health (50) is average, but the agent has collapsed into a narrow hovering behavior.
- **Recommendation:** This reward design should be **iterated** – the distance penalty is too strong relative to the alive bonus, and the landing reward is too sparse. The agent needs either a shaping that encourages approach or a termination condition that forces landing.

### 6. Key Numbers for Budget Calculation

| Metric | Value at Final Timestep (1M) |
|--------|------------------------------|
| mean_length | 76.2 |
| alive | 1.0 |
| angle | -0.10398 |
| distance_to_pad | 1.027338 |
| leg_contact | 0.018373 |
| vertical_speed | -0.82027 |