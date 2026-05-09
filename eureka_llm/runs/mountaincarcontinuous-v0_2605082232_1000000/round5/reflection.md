## Round 5 Reflection

### Expected
The analyst predicted that changing the velocity bonus from `5.0 * max(0, v)` to `8.0 * abs(v)` would incentivise the agent to oscillate (move in both directions) to build momentum, breaking the stationary local optimum and eventually reaching the goal.

### Actual
Success remained 0.0 at all checkpoints; velocity stayed near 0.012 (barely changed); distance improved only modestly (0.99→0.74) but the agent still never completes the task. The reward analysis shows `r_success` still dominates (57% of total, mean 4.2) despite zero actual successes – confirming the reward is still given spuriously for proximity, not for goal achievement. The velocity bonus tweak had negligible effect because the spurious success reward remains the dominant attractor.

### What We Learned
`r_success still spuriously awarded (mean 4.2, 57% of total) despite zero successes → velocity bonus change (8*abs(v)) ineffective | Why: root cause (unconditional r_success) unchanged; agent still collects easy proximity reward without completing task | Fix: make r_success conditional on actual goal achievement (sparse, terminal reward) and remove constant _outcome per-step baseline`

### For Next Round
First, verify in the reward code that `r_success` is only assigned when the episode terminates due to reaching the goal (i.e., `done` and `success` flag both true). If it’s given for any termination or as a shaped reward, change it to fire exactly once upon goal success. Second, remove the `_outcome` constant per-step reward (mean=1.0, std=0.0) to eliminate the free positive baseline. Third, consider adding a small negative reward for moving away from the goal or a directional progress bonus based on delta distance, but only after fixing the spurious success issue. Monitor whether the agent now starts achieving success within 500k steps.