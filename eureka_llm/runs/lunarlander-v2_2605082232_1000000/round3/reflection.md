## Round 5 Reflection

### Expected
The analyst predicted that making `r_alive` distance-dependent and `r_leg_contact` conditional on proximity would eliminate early crashes and produce slow, controlled descent with a soft landing at termination.

### Actual
The agent improved sharply by 600k (mean length 838, vertical speed -0.54) but then regressed to 486 steps and vertical speed -0.97 at 1M. Distance improved (0.33) but leg contact increased (0.079). The termination penalty (`_outcome`) dominates 72.6% of total reward magnitude, overwhelming shaping components.

### What We Learned
`distance-dependent alive + conditional leg contact → initial improvement then regression (length 838→486, vertical speed -0.54→-0.97) | Why: termination penalty (72.6% of reward) still dominates; policy overfits to short-term survival then collapses | Fix: reduce _outcome magnitude (e.g., -0.5 instead of -1.0) and proportionally scale alive/distance to maintain gradient`

### For Next Round
Reduce the termination penalty (`_outcome`) from -1.0 to -0.5 to lower its dominance, and scale other components (r_alive, r_distance, r_vertical_speed) upward proportionally so the shaping signals have comparable influence. Monitor TDRQ component imbalance score — target below 20. Also consider adding an early-stop condition if mean length drops below 400 after 300k steps to detect regression faster.