## Round 4 Reflection

### Expected
The analyst predicted that adding a velocity bonus (5.0 × velocity) and tripling the success reward (100→300) would teach the agent to build momentum and consistently reach the goal, overcoming the previous local optimum of slow dithering.

### Actual
Success remained 0.0 at every checkpoint; distance to goal barely improved (0.99→0.80); velocity stayed near 0.014 despite near-max action magnitude (0.98). The reward analysis revealed r_success now dominates at 58% of total reward (mean 4.15 per step) even though the agent never reaches the goal—meaning the reward is being awarded spuriously (likely for proximity or partial progress). The velocity bonus contributed only 1.3% of total reward, too small to shift behavior. The agent simply found a way to collect even more of the easy r_success while still avoiding the real goal.

### What We Learned
`r_success=300 + r_velocity_bonus=5*max(0,v) → agent still never succeeds (r_success dominates 58% but awarded spuriously) | Why: success reward decoupled from actual goal achievement; velocity bonus too weak to overcome existing attractor | Fix: ensure r_success is only given when distance < threshold (use termination condition) and remove constant per-step reward to eliminate free positive baseline`

### For Next Round
First, verify in the reward code whether `r_success` is truly conditioned on `done == True` from reaching the goal or if it's given for any termination. If it's given spuriously, change it to fire only on `success` flag. Second, remove the `_outcome` per-step constant reward entirely to eliminate the free positive baseline. Third, either increase the velocity bonus coefficient (e.g., to 20× velocity) or replace it with a directional progress reward that directly tracks reduction in distance to goal. Monitor whether the agent now learns to reach the goal within 500k steps.