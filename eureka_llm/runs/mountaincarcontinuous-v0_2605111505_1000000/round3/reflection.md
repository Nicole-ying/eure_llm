## Round {N} Reflection

### Expected
The analyst predicted that adding a signed velocity reward (0.3*velocity) and a proximity penalty (-0.01*dist) would break the stationary oscillation, drive the agent rightward, and gradually climb the hill to reach the goal.

### Actual
The agent never reached the goal (reached_goal=0), velocity remained low (~0.03 m/s), distance to goal barely decreased (1.08→0.97), and the `_outcome` component dominated 98% of reward. The agent applied maximum force continuously rightward without oscillating, stuck at the base of the steep slope.

### What We Learned
`Added signed velocity and proximity penalty but _outcome still dominated (98%) → agent drove right monotonically without building momentum | Why: constant _outcome reward dwarfed all new signals, and signed velocity discouraged leftward exploration needed for oscillation | Fix: remove or condition _outcome, increase velocity coefficient to ≥1.0, and add explicit oscillation bonus or negative distance gradient`

### Abstract Principle
When a reward function contains a dominant constant-offset component, any new task-relevant rewards must be significantly larger than that offset to influence policy; otherwise, the agent will exploit the constant term and ignore sparse shaping signals.

### For Next Round
Eliminate the `_outcome` component entirely or replace it with a small state-dependent penalty (e.g., -0.1 only when velocity < threshold). Increase the signed velocity coefficient to at least 1.0, and add a large positive reward for reducing distance (e.g., +1.0 * (prev_dist - curr_dist)). Consider a curriculum that first encourages oscillation by rewarding absolute velocity magnitude regardless of direction.