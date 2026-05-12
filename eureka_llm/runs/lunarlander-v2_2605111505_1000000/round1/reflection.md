## Round 1 Reflection

### Expected
The agent would learn to descend gently (vertical speed ~ -0.5 to -1.0), maintain horizontal centering, and begin landing successfully (positive survival fraction) within 1-2M steps, with cumulative rewards of +2 to +5 for successful landings.

### Actual
The agent learned to descend to low altitude (~2.2), reduce vertical speed (~0.7), and make ground contact (legs_ground ~1.0), but never moved horizontally toward the pad (dist_to_pad ~10.5, worsening). Survival fraction remained 0.0. The agent exploited off-pad leg contact as a local optimum, collecting small r_legs but always crashing off-pad (mean crash penalty -10.0). The r_landing_bonus (+20) was never achieved.

### What We Learned
Fixed vertical speed penalty but failed to incentivize horizontal movement → agent hovers low and touches ground far from pad | Why: proximity reward too weak (0.0165), leg contact reward rewards off-pad crashes, and no penalty for being far from pad | Fix: add strong negative reward for distance to pad (e.g., -0.1 * dist_to_pad per step) and make leg contact bonus conditional on being within pad radius (e.g., dist_to_pad < 1.5).

### Abstract Principle
Shaping rewards must directly guide the agent toward the explicit goal metric; rewarding proxy behaviors (e.g., ground contact) without spatial constraints creates local optima that never achieve the actual objective.

### For Next Round
Make the proximity reward dominant and dense: replace the small `r_proximity` with a per-step penalty proportional to `dist_to_pad` (e.g., -0.1 * dist). Condition the `r_legs` bonus on being near the pad (`dist_to_pad < 1.5`). Reduce or remove the constant crash penalty in favor of a shaped term that smoothly penalizes high speed and off-pad ground contact. Consider adding a small bonus for reducing distance to pad each step (dense directional signal).