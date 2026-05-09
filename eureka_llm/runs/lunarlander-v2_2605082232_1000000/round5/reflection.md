## Round 7 Reflection

### Expected
The analyst predicted that adding a moderate outcome penalty (−0.3 for crash, +1.0 for success), softening vertical speed penalty (only penalize >1 m/s), and switching to a continuous leg-contact bonus would let the agent descend slowly, approach the pad, and achieve stable landings with episode length increasing and leg contact above 0.3.

### Actual
The agent improved dramatically to a perfect 1000-step hover at 600k (vertical speed −0.47, distance 0.20, leg contact 0.24) but then regressed to 395.5 steps at 1M (vertical speed −1.19, distance 0.31, leg contact 0.46). The outcome component became 45.8% of total reward, r_vertical_speed was negligible (0.8%), r_leg_contact only 2.9%, and TDRQ flagged low exploration health (50), indicating policy collapse.

### What We Learned
`outcome penalty added + softened vertical speed + continuous leg contact → agent learned to hover near pad (1000 steps peak) then collapsed to 395 steps (vertical speed −1.19, leg contact 0.46) | Why: soft vertical speed penalty (only >1 m/s) and weak leg contact reward (2.9%) never forced sustained flight; policy collapsed due to low entropy (exploration health=50) and exploited a narrow hover-then-fall strategy | Fix: strengthen vertical speed penalty (e.g., quadratic penalty on all negative speed) and add an entropy bonus to maintain exploration, or use a distance-tapered alive reward to encourage staying near pad without regressing`

### For Next Round
The agent is stuck in a local optimum where it hovers briefly then falls (leg contact increases because it crashes). Do not rely solely on a threshold-based vertical speed penalty — instead use a continuous quadratic penalty that penalizes any descent (e.g., `-0.1 * (vertical_speed)^2`) to discourage falling. Also add a small exploration bonus (e.g., `0.01 * policy_entropy`) to prevent policy collapse. Monitor the TDRQ exploration health metric; if it stays below 60, increase the bonus.