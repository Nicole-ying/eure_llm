## Round 3 Reflection

### Expected
The analyst predicted that increasing the distance penalty coefficient to -2.0 and reducing the success reward to 100 would eliminate reward hacking—the agent would move rightward, reduce distance to goal, and eventually achieve success. The constant per-step reward was left unchanged.

### Actual
The agent does move slightly toward the goal (distance drops from ~1.0 to ~0.79 over 1M steps) but never succeeds (success=0). Episode length remains stable at ~76 steps, and velocity is low (~0.012). The three dominant reward components (r_distance_penalty mean -1.71, r_success mean 1.38, _outcome +1.0) are balanced in magnitude, so no single term dominates—but the constant _outcome still provides a positive baseline that makes stationary dithering viable. The agent has found a local optimum: moving slowly enough to reduce distance penalty while collecting the per-step bonus, but too slowly to reach the goal within the episode horizon.

### What We Learned
Increased distance penalty + reduced success reward → slight approach but no success (agent lingers, never triggers r_success) | Why: constant per-step reward (+1) still provides positive baseline; agent optimizes to slowly reduce distance penalty without needing to finish | Fix: remove constant step reward or replace with potential-based shaping that only rewards net distance reduction, and consider extending episode horizon or adding a velocity bonus/cost.

### For Next Round
Remove the constant `_outcome` per-step reward entirely. Instead, use a potential-based shaping function (e.g., `gamma * next_potential - potential`) to reward any reduction in distance to goal, and add a small negative reward for low velocity to prevent dithering. Alternatively, increase episode length if possible to give the agent more time to reach the goal. Monitor whether the agent now reaches the goal within a reasonable number of steps.