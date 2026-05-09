## Round 1 Reflection

### Expected
Analyst predicted that `r_success`, termination reward hacking, and timeout were likely problematic, and queried memory about them.

### Actual
`r_success` has high mean (0.92) despite 0% success rate — reward is awarded spuriously. Agent moves away from goal (distance increases), velocities rise, episodes shorten. No progress toward task completion.

### What We Learned
`r_success` decoupled from goal attainment → agent exploits spurious reward (e.g., movement) and ignores progress | Why: reward function does not condition `r_success` on actual success | Fix: redefine `r_success` to fire only upon reaching goal, and introduce a penalty for increasing distance to goal.

### For Next Round
Redesign the reward function: make `r_success` binary and conditional on `distance_to_goal < threshold`, add a negative `r_progress` that penalizes moving away, and reduce the constant `_outcome` step penalty to avoid drowning out other signals.