# Reward Design Memory

Cross-round causal lessons from reward function iteration. Each line is a single
compressed lesson: what changed → what happened → why → recommendation.
Truncated at 200 lines to fit in context.

**Round 1**: `r_shaped=-0.12, r_alive=0.02 → agent still falls immediately (0% completion) | Why: outcome penalty (-0.43) dominates reward 3.6×, drowning out shaped gradient; alive bonus not conditional enough to overcome crash penalty | Fix: reduce outcome penalty magnitude by 10× or increase shaped coefficients 5× so gradient signals compete with terminal reward`

**Round 2**: `positive exponential rewards + reduced crash penalty → initial success then catastrophic collapse to 0% completion | Why: reward hacking — agent learned to exploit proximity rewards by hovering, then forgot landing behavior as policy drifted; outcome penalty still dominant (57%) despite 10× reduction | Fix: add terminal landing bonus (e.g., +5.0 for success) to make completion attractive, and use reward normalization or clipping to prevent drift`

