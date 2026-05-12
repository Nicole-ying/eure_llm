## Report: Perception Agent Analysis

### 1. Behavior Trend Summary

At every evaluation milestone, the policy consistently fails to reach the goal (reached_goal=0). The agent’s episodes always hit the maximum episode length (mean_length=999.0) and the distance to the goal remains near 1.0–1.16 (the valley bottom is at ~0.0, goal at ~0.5). The heading fluctuates slightly but stays close to zero, and velocity magnitude is extremely low (0.01–0.02). The action force shows a large spike at 600k (0.989) but then declines, indicating the agent tried a high-force strategy mid-training but ultimately settled on moderate force (0.525 at final evaluation). The policy is **flat/regressing**: no improvement in goal attainment, distance, or velocity.

**Final timestep (1,000,000) key numbers**:
- mean_length = 999.0
- action_force = 0.525
- dist_to_goal = 1.165
- heading = 0.083
- reached_goal = 0.0
- velocity_mag = 0.019

### 2. Critical Metrics

| Metric | Why important | Direction |
|--------|---------------|-----------|
| `dist_to_goal` | Direct measure of progress toward the flag; should decrease toward 0 as the car climbs. | **Stuck** (always ~1.0–1.16, no downward trend) |
| `reached_goal` | The ultimate task success signal. | **Always 0** (never achieved) |
| `velocity_mag` | Necessary for building momentum to overcome the valley; must increase significantly to reach the goal. | **Extremely low** (0.01–0.02), no growth |

**Cross-metric pattern**: Despite a large spike in action_force at 600k (0.989), velocity_mag actually dropped to its lowest (0.002). This indicates that the applied force is being cancelled by equal opposite force within a step or rapidly oscillating direction — the car is not building net momentum. The ratio `action_force / velocity_mag` is huge (~500 at 600k), showing high effort with negligible motion outcome.

### 3. Reward Component Health

- **Active components**: Only `_outcome` (mean=-1.0, std=0.0) – continuously active with constant value.
- **Dominant component**: `_outcome` accounts for **99.7%** of total reward magnitude; dominates completely.
- **Inactive/negligible**: `r_goal` (mean=0.0, std=0.0, inactive); `r_progress` (mean=0.0032, std=0.0041, negligible relative to `_outcome`).
- **Suspicious mean**: `_outcome` has mean = -1.0 with **zero variance** – this component provides no learning signal; the agent can “harvest” it without any informative gradient.
- **High mean but weak alignment**: `r_progress` is slightly positive but its magnitude is dwarfed by the constant step penalty, so it cannot overcome the dominant negative signal. The progress reward does not correlate with meaningful task progress (dist_to_goal remains unchanged).

### 4. Behavioral Diagnosis

The agent’s current strategy is to apply low or oscillating forces that result in minimal net displacement, keeping the car near the bottom of the valley. It is **stuck in a local optimum** where the constant step penalty dominates and the progress reward is too weak to incentivize effective swinging. The policy is **not reward hacking** in the traditional sense, but rather the reward structure itself provides no differential signal for building momentum. Efficiency is extremely poor: high action effort (up to 0.99) yields almost no velocity gain (0.002–0.019) and zero progress toward the goal. This behavior **does not match the intended task** of building momentum by driving back and forth; the agent never increases oscillation amplitude and stays stationary relative to the goal.

### 5. TDRQ Diagnosis

The low TDRQ score (21.81/100) is primarily caused by **component imbalance** (one component accounts for >99% of reward), **component inactivity** (`r_goal` has zero contribution), and **exploration collapse** (entropy decreasing steadily to -2.13, indicating premature convergence to deterministic actions). The reward function should be **iterated**: either increase the relative magnitude of `r_progress`, remove the constant `_outcome` penalty, or redesign `r_goal` to provide a meaningful positive signal when the goal is reachable.

### 6. Constraint Violations Summary

| Principle | Severity | Observation |
|-----------|----------|-------------|
| reward_goal_alignment | **high** | `_outcome` has zero variance and dominates (99.7%); provides no informative learning signal. |
| state_coverage | **medium** | Evaluation episode lengths are perfectly concentrated at the maximum (999.0), indicating the policy never terminates early (no goal reach or failure). However, training episodes show a mean of 728.6 and max 1948 – suggesting exploitation of early termination (possibly due to out-of-bounds resets or goal? but `r_goal` is 0). |
| temporal_consistency | **medium** | `heading` drifted from -0.006 early to 0.092 late (relative drift 15.75), indicating a behavioral shift without corresponding task improvement. |
| termination_exploitation | **medium** | Ratio of mean episode length (728.6) to observed max (1948) is 0.374; agent is terminating well before the maximum allowed steps, possibly by falling off the edge or exploiting a poorly defined terminal condition. |

### 7. Episode Consistency Summary

Early (200k) vs late (1M) evaluation metrics show a small drift in `heading` (from -0.009 to 0.083) and `action_force` (from 0.07 to 0.52), but `dist_to_goal` and `velocity_mag` remain essentially unchanged. The policy style is **drifting** in action space but **not** in task-relevant state space. This drift appears to be **unstable policy behavior** rather than healthy adaptation, because the agent never moves closer to the goal and the entropy is collapsing.

### 8. Key Numbers for Budget Calculation

| Metric | Value (final timestep) |
|--------|------------------------|
| mean_length | 999.0 |
| action_force | 0.524659 |
| dist_to_goal | 1.165413 |
| heading | 0.083283 |
| reached_goal | 0.0 |
| velocity_mag | 0.018682 |