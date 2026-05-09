## Round 3 Reflection

### Expected
Making `r_alive` distance-dependent (0.05 * max(0, 1 - dist)) would incentivize the agent to move toward the pad, reducing hovering and leading to landing.

### Actual
The agent continued to fall straight down with increasing vertical speed (-6.48 → -6.96 m/s), terminated early (~70 steps), and distance to pad barely changed (1.02 → 0.96). `r_alive` was negligible (mean 0.0099) and `_outcome` dominated reward (79.8%, magnitude ~50× larger).

### What We Learned
`r_alive made distance-dependent → agent ignored it (mean 0.01) and fell faster (vertical speed -7) | Why: outcome penalty (-0.48) still 50× alive reward; distance-dependent alive near zero when far → no incentive to approach pad | Fix: increase alive magnitude by 10× or reduce outcome weight, and add vertical speed penalty to prevent hard landings.`

### For Next Round
Reduce the magnitude of the outcome penalty (e.g., scale to -0.1) and increase the alive reward baseline to at least 0.5 per step. Add a small penalty for high vertical speed (e.g., -0.01 * v_z²) to discourage rapid descent without control. Consider making the distance penalty more aggressive (e.g., -0.2 per unit distance) to create a stronger gradient toward the pad.