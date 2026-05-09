## Perception Report

### 1. Behavior Trend Summary

- **At 200k:** Agent takes ~70 steps, stays nearly stationary (velocity 0.013), with distance to goal near 1.0 (0.991). No success. Essentially not moving toward goal.
- **At 400k:** Similar behavior; distance to goal barely decreased (0.954), still stationary.
- **At 600k:** First meaningful movement — distance drops to 0.768. Episode length remains ~70. Agent begins to approach goal but still far.
- **At 800k:** Distance further decreases to 0.705; episode length rises to 81.7 (slightly longer episodes). Still no success.
- **At 1M:** Distance regresses to 0.802, episode length shortens to 67.1. Agent appears to stop improving and may be cycling.

**Overall trajectory:** Modest early improvement (distance 0.99→0.70) then regression. **No episode ever results in success (success=0.0).** Action magnitude stays high (~0.98) — agent applies near-maximum actions but fails to close the remaining gap. The agent is **moving but not completing the task**.

**Key numbers at final timestep (1,000,000):**
- mean_length = 67.1
- distance_to_goal = 0.8018
- success = 0.0
- velocity = 0.0137
- action_magnitude = 0.9848

### 2. Critical Metrics

1. **distance_to_goal** – Improved from 0.99 to 0.71–0.77 (middle checkpoints), then regressed to 0.80 at final. Trend: slight improvement then plateau/regression. **Not converging to zero.**
2. **success** – Flat at 0.0 throughout. **Moving in the wrong direction** (should be >0). Complete task failure.
3. **velocity** – Stable at very low values (~0.013). Agent is barely moving despite high action magnitude. Suggests actions are not transferring into movement (e.g., opposing forces, high friction, or oscillation).

All three metrics indicate the agent is **not solving the primary task** and is stuck in a behavior that does not reach the goal.

### 3. Reward Component Health

- **Active components (mean significantly non‑zero):** All five components have non‑zero means.
- **Dominant component:** `r_success` (mean = 4.15, 58.3% of total reward) — its absolute mean is >2× the next component (`r_distance_penalty` at −1.68). This single component dominates the reward signal.
- **Inactive/negligible:** None. Even `r_velocity_bonus` (0.094) is non‑zero but very small.
- **Suspicious means:** `_outcome` has mean exactly 1.0 and std 0.0 — likely a constant indicator (e.g., always 1 per step). Not suspicious in itself but indicates no meaningful variation.

**Observation:** The agent receives a large positive `r_success` even though `success` metric is 0.0. This implies `r_success` is a **dense shaping reward** (e.g., based on proximity or progress) rather than actual task success. This creates a local optimum where the agent gets high reward without completing the episode.

### 4. Behavioral Diagnosis

The agent applies near‑maximal actions but moves very sluggishly (low velocity) and only reduces distance to about 0.7–0.8 of the initial offset. It never reaches the goal. The high dense `r_success` reward likely peaks at a moderate distance, causing the agent to **stop improving once it reaches that region**, as additional progress yields diminishing returns. This is a classic case of **stuck in a local optimum** due to an over‑dominant shaping component that does not require full task completion.

### 5. TDRQ Diagnosis

- Overall TDRQ = 73.78 (healthy threshold >70), but component balance is only 41.74 — the **main weakness**.
- Exploration health is perfect (100), and component activity is full (100). The low balance is entirely due to the dominance of `r_success` (58% of total reward), which skews the gradient.
- **Recommendation:** The reward should be **iterated** to rebalance components — either reduce the magnitude of the dense success bonus or make it conditional on true task success (sparse). The current reward structure does not incentivize completing the episode; it rewards approximation.

### 6. Key Numbers for Budget Calculation

| Metric               | Final Value (1M) |
|----------------------|------------------|
| mean_length          | 67.1             |
| action_magnitude     | 0.9848           |
| distance_to_goal     | 0.8018           |
| steps                | 0.0              |
| success              | 0.0              |
| velocity             | 0.0137           |