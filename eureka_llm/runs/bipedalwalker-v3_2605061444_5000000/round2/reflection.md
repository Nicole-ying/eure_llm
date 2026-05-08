## Round 4 Reflection

### Expected
The analyst predicted that fixing the `_outcome` structural bug (moving it before the sum) and adding `r_stability` would maintain 100% completion while developing a more upright, stable gait with smoother locomotion.

### Actual
The agent achieved 100% completion and 0% fall rate, but with a critical difference: the agent initially fell in every episode for ~1.5M timesteps before suddenly transitioning to perfect performance. The `_outcome` component now shows positive mean (0.176) instead of negative (-0.222), confirming the structural fix worked. However, the stability bonus (0.431) is much higher than expected given the 0.5 coefficient, and the agent's early collapse suggests the changes destabilized training before convergence.

### What We Learned
`_outcome fix + r_stability added → agent collapsed for 1.5M steps then recovered to perfect performance | Why: moving _outcome before sum changed reward timing, creating initial instability as agent learned new credit assignment; stability bonus added constraint that delayed convergence | Fix: when fixing reward timing bugs, expect temporary performance regression; consider phased rollout (fix timing first, add stability later) or reduce stability coefficient to 0.2`

### For Next Round
The analyst should expect that fixing reward timing bugs can cause temporary training collapse as the agent re-learns credit assignment. Consider testing the timing fix alone before adding new reward components, or reduce the stability coefficient to 0.2 to minimize disruption during the re-convergence period.