## Round 2 Reflection

### Expected
The analyst predicted the agent was exploiting a spurious terminal bonus (`r_success` firing on any termination) and moving away from the goal. Proposed changes: make `r_success` conditional on true goal attainment (with increased magnitude 1000) and add a distance penalty (`-0.5 * distance`). Expected effect: agent learns to move right, increase episode length, and achieve >0 success rate.

### Actual
The agent remains stuck near the start position (distance ~0.96, flat; velocity ~0.01; success 0.0). Episode length drops to ~70 and plateaus. Action magnitude is high (~0.97) but velocity is near zero—the agent dithers (oscillates) to stay in place. `r_success` mean is 12.7 (rarely collected), `r_distance_penalty` mean is -0.47, but the constant `_outcome` ( +1 per step) dominates. The agent has learned that stationary dithering yields a positive net reward, and moving toward the goal is too risky.

### What We Learned
Conditional r_success + distance penalty → agent stays near start (dithers) | Why: constant per-step reward (+1) dominates; distance penalty too weak to outweigh the positive baseline | Fix: remove constant per-step reward, increase distance penalty coefficient, and add negative reward for zero velocity to force movement

### For Next Round
Remove the constant `_outcome` step reward entirely. Increase the distance penalty coefficient (e.g., from -0.5 to -2.0) and consider adding a small negative reward for zero velocity (to penalize dithering). Alternatively, use a potential-based shaping function that directly rewards reductions in distance. Monitor whether the agent begins to move rightward after these changes.