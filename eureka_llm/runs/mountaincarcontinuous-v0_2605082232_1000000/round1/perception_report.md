# Post-Training Behavior Report

## 1. Behavior Trend Summary

- **At 200k timesteps:** The agent remains stationary or nearly so (velocity 0.00017) for the full episode length (mean_length=999). It ends at distance 0.73 from the goal, with zero steps and zero success.
- **At 400k:** Slight reduction in episode length (580), velocity rises to 0.00084, but distance to goal increases (0.76). No success.
- **At 600k–1M:** Episodes shorten dramatically to ~81–109 steps. Velocity increases further (~0.011). Distance to goal increases to ~0.89–0.99. Action magnitude stays high (~0.96–0.98), indicating continuous movement. Success remains zero throughout.

The agent is **not improving** in terms of task completion. Instead, it learns to end episodes earlier while moving, but moves *away* from the goal (increasing distance). It never achieves success. The trajectory is regressing with respect to goal proximity.

**Key numbers at final timestep (1M):**  
`mean_length = 81.3`, `distance_to_goal = 0.943`, `steps = 0.0`, `success = 0.0`, `velocity = 0.0115`, `action_magnitude = 0.962`.

## 2. Critical Metrics

| Metric | Trend | Direction |
|--------|-------|-----------|
| **distance_to_goal** | Increasing (0.73 → 0.94) | **Wrong direction** — agent ends further from goal over time |
| **success** | Constant at 0.0 | Never achieved |
| **velocity** | Slight increase (0.0002 → 0.0115) | Ambiguous — movement increases but does not help task |

The increase in `distance_to_goal` is a clear red flag: the agent is moving away from the target.

## 3. Reward Component Health

- **Active components (mean significantly non-zero):** `_outcome` (1.0), `r_progress` (-0.214), `r_success` (0.916). All three contribute meaningfully to the total reward.
- **Dominant components (|mean| > 2× others):** None. `_outcome` (1.0) and `r_success` (0.916) are similar in magnitude; neither is more than double the other.
- **Inactive/negligible:** None.
- **Suspicious values:** `_outcome` has mean exactly 1.0 with zero standard deviation. This suggests it is a constant per-step base reward (e.g., survival bonus) that does not vary across steps or episodes. This is typical for a shaping-free survival term and is not necessarily problematic.

## 4. Behavioral Diagnosis

The agent moves quickly (high action magnitude) but consistently ends episodes farther from the goal, with no successes. It appears to exploit the combination of a constant survival reward (`_outcome`) and a positive `r_success` bonus (which does not require actually reaching the goal, perhaps triggered on proximity or other criteria) while receiving a negative progress penalty. The strategy may involve drifting or circling near some intermediate area that delays the progress penalty while still collecting survival and success signals. This is a **local optimum** where the agent avoids the true goal in order to maximize a misspecified reward signal.

## 5. TDRQ Diagnosis

The overall TDRQ score is 78.87/100 (healthy). The sub-score `component_balance` (53.05) indicates moderate imbalance, driven by the conflicting signs of `r_progress` (negative) and `r_success` (positive) alongside the constant `_outcome`. `component_activity` and `exploration_health` are both 100, so no inactivity or exploration collapse. The main limitation is the poor alignment between reward components. **This reward should be iterated** — the components are not directing the agent toward the goal, likely because `r_progress` punishes progress while `r_success` rewards something unrelated to actual goal arrival.

## 6. Key Numbers for Budget Calculation

| Metric | Value (at 1M timesteps) |
|--------|-------------------------|
| mean_length | 81.3 |
| action_magnitude | 0.962 |
| distance_to_goal | 0.943 |
| steps | 0.0 |
| success | 0.0 |
| velocity | 0.0115 |