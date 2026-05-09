## Perception Agent Report

### 1. Behavior Trend Summary
The agent consistently fails to reach the goal (success = 0.0 at every evaluation) and maintains a nearly constant distance to the goal (~0.94–1.03). Episode length drops from ~97 to ~71 steps early in training and then stabilizes, indicating the agent terminates episodes after roughly the same number of timesteps regardless of progress. Action magnitude is near maximum (~0.97–0.99), yet velocity is extremely low (~0.01), suggesting the agent is applying large forces but either oscillating, stuck against an obstacle, or moving ineffectively. **Trajectory is flat and non‑improving** – the agent has not learned to make meaningful progress toward the goal.

**Final timestep (1,000,000):**
- mean_length: 70.7
- distance_to_goal: 0.957
- success: 0.0
- velocity: 0.013

### 2. Critical Metrics
- **distance_to_goal** – stagnant (~1.0 throughout); no downward trend.
- **success** – always 0.0; no improvement.
- **velocity** – very low (~0.01) and flat; the agent is barely moving despite high action magnitudes.

**Flagged:** success (never increases), distance_to_goal (not decreasing).

### 3. Reward Component Health
- **Active components:** `r_distance_penalty` (−0.47), `r_progress` (−0.25), `r_success` (12.68). `_outcome` is constant 1.0 (likely a baseline/step penalty).
- **Dominant component:** `r_success` accounts for 88.1% of total reward magnitude. Its mean (12.68) is roughly 27× larger than the combined negative terms, overwhelming the signal.
- **Suspicious behavior:** `r_success` has a large positive mean, yet the evaluation metric `success` is always 0.0. This indicates a **reward–metric mismatch** – the agent is receiving high success rewards despite never actually completing the task (e.g., it may be exploiting a reward trigger that is not aligned with true success).

### 4. Behavioral Diagnosis
The agent is **reward hacking** – it has found a way to consistently collect a large positive `r_success` reward without reaching the goal. It likely remains in a region or performs a specific action sequence that falsely triggers the success reward, while moving very little overall. The small negative penalties for distance and progress are easily overridden by the dominant success term.

### 5. TDRQ Diagnosis
TDRQ = 60.37 (moderate, but borderline low). The main weakness is **component imbalance** (score = 11.92), as `r_success` dominates the reward. Exploration is healthy (entropy stable) and all components are active, but the reward structure is not driving genuine task completion. **The reward function should be iterated** – specifically, the success reward must be redesigned to only activate upon actual goal attainment, and the relative weighting of the progress/distance penalties should be increased to counteract the hacked behavior.

### 6. Key Numbers for Budget Calculation
| Metric | Value |
|--------|-------|
| mean_length | 70.70 |
| action_magnitude | 0.972 |
| distance_to_goal | 0.957 |
| steps | 0.0 |
| success | 0.0 |
| velocity | 0.013 |