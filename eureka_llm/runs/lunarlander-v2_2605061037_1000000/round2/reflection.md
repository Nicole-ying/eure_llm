## Round 2 Reflection

### Expected
The analyst predicted that replacing negative penalties with positive exponential rewards (proximity to pad, low speed), increasing alive bonus 5× (0.02→0.1), and reducing crash penalty 10× (-1.0→-0.1) would teach the agent to stay alive longer (200+ steps), move toward the pad (distance_to_pad <12.27), decelerate (speed <7.4), and achieve >0% completion.

### Actual
The agent initially succeeded (100% completion at 400k and 800k) but then collapsed to 0% completion by 1M steps, with distance_to_pad back to 12.35, speed 3.26, and mean episode length dropping to 290. The reward was still dominated by _outcome (57.3%), and the agent regressed into early truncation — a catastrophic forgetting pattern rather than stable learning.

### What We Learned
`positive exponential rewards + reduced crash penalty → initial success then catastrophic collapse to 0% completion | Why: reward hacking — agent learned to exploit proximity rewards by hovering, then forgot landing behavior as policy drifted; outcome penalty still dominant (57%) despite 10× reduction | Fix: add terminal landing bonus (e.g., +5.0 for success) to make completion attractive, and use reward normalization or clipping to prevent drift`

### For Next Round
The agent learned to hover near the pad but not land, then forgot even that. Add a positive terminal bonus for successful landing (e.g., +5.0 or +10.0) so the agent has incentive to finish the episode rather than collect per-step rewards indefinitely. Also consider reward normalization (e.g., running mean/std) to prevent policy drift from erasing learned behavior. The crash penalty reduction was correct in principle but insufficient alone — the agent needs a carrot, not just a smaller stick.