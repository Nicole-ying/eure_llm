## Round 5 Reflection

### Expected
Removing `_outcome` (set to 0) and increasing `r_velocity` to 5.0×abs(velocity) would drive higher velocity (~0.05–0.07), directional oscillation, decreasing dist_to_goal, and eventual goal attainment within 5–10 evaluations.

### Actual
The agent never reached the goal (reached_goal=0). Velocity remained at ~0.01 m/s, dist_to_goal stagnated at ~0.86, and action force was high (~0.87) with negligible net movement. Reward components: `_outcome` and `r_goal` were zero, but `r_progress` dominated 100% of reward (mean 0.075). `r_velocity` was absent or inactive—the proposed velocity reward did not appear in the active components. The agent remained stuck, applying force without momentum.

### What We Learned
`Removed _outcome & increased r_velocity → agent ignored velocity signal (r_velocity absent in active components) and stayed stationary with high force | Why: The proposed reward change was not reflected in the actual reward function – only r_progress remained active, drowning out any velocity shaping; the agent is effectively optimizing a weak progress signal that does not incentivize momentum building | Fix: Verify that the reward function modification is correctly implemented (ensure r_velocity is included and has non‑zero weight) and redesign r_progress to explicitly reward back‑and‑forth oscillation (e.g., signed velocity or momentum bonus)`

### Abstract Principle
When a reward function is modified, the actual training environment must be audited to confirm the changes took effect; otherwise, the agent continues optimizing the previous, unaltered objective, leading to unchanged failure modes.

### For Next Round
First confirm that the reward function in the training script matches the intended changes (especially that `r_velocity` is present and not overwritten). Then either replace `r_progress` with a dense directional momentum bonus (e.g., `sign(velocity)*velocity` plus a large reward for velocity > threshold) or reintroduce a small `_outcome` penalty conditioned on progress (not termination). Also investigate why high action force yields negligible velocity—possible physics issue (e.g., friction or slope gradient) that may require a different action scaling or a goal‑directed curriculum.