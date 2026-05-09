# Reward Design Memory

Cross-round causal lessons from reward function iteration. Each line is a single
compressed lesson: what changed → what happened → why → recommendation.
Truncated at 200 lines to fit in context.

**Round 1**: `r_success` decoupled from goal attainment → agent exploits spurious reward (e.g., movement) and ignores progress | Why: reward function does not condition `r_success` on actual success | Fix: redefine `r_success` to fire only upon reaching goal, and introduce a penalty for increasing distance to goal.

**Round 2**: Conditional r_success + distance penalty → agent stays near start (dithers) | Why: constant per-step reward (+1) dominates; distance penalty too weak to outweigh the positive baseline | Fix: remove constant per-step reward, increase distance penalty coefficient, and add negative reward for zero velocity to force movement

**Round 3**: Increased distance penalty + reduced success reward → slight approach but no success (agent lingers, never triggers r_success) | Why: constant per-step reward (+1) still provides positive baseline; agent optimizes to slowly reduce distance penalty without needing to finish | Fix: remove constant step reward or replace with potential-based shaping that only rewards net distance reduction, and consider extending episode horizon or adding a velocity bonus/cost.

**Round 4**: `r_success=300 + r_velocity_bonus=5*max(0,v) → agent still never succeeds (r_success dominates 58% but awarded spuriously) | Why: success reward decoupled from actual goal achievement; velocity bonus too weak to overcome existing attractor | Fix: ensure r_success is only given when distance < threshold (use termination condition) and remove constant per-step reward to eliminate free positive baseline`

**Round 5**: `r_success still spuriously awarded (mean 4.2, 57% of total) despite zero successes → velocity bonus change (8*abs(v)) ineffective | Why: root cause (unconditional r_success) unchanged; agent still collects easy proximity reward without completing task | Fix: make r_success conditional on actual goal achievement (sparse, terminal reward) and remove constant _outcome per-step baseline`

