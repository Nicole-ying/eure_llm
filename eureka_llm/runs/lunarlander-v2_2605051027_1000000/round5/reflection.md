## Round 7 Reflection

### Expected
The analyst predicted that setting `r_termination` to zero (no per‑step reward) and adding a large one‑time landing bonus of +10 would eliminate the hovering strategy, causing the agent to commit to landing reliably while still maintaining precision.

### Actual
`r_termination` remained at a small positive mean of 0.05 (per step), the landing bonus was not clearly increased, and the agent achieved a 90% completion rate but with **degraded landing precision** (pad_distance_mean rose from 0.37 to 0.64) and near‑maximal fuel usage (0.96). The strategy still relies heavily on `r_progress` and `r_landing_shaping`.

### What We Learned
We changed `r_termination` to a small positive constant (~0.05 per step) rather than zero, and kept the existing progress and shaping rewards. This caused the agent to learn to land frequently (90% completion) but at the cost of precision and fuel efficiency. The reason was that the small per‑step reward still encouraged long episodes, while the progress and shaping rewards guided the agent toward the pad but not to a precise touchdown, and the landing bonus (if any) was not conditioned on distance from the pad center. Next time, set `r_termination` to exactly zero, add a large landing bonus that is **distance‑dependent** (e.g., +10 − 10×pad_distance) to reward precise landings, and optionally introduce a small fuel‑usage penalty to improve efficiency.

### For Next Round
Implement a distance‑conditioned landing bonus (e.g., `+10 * (1 - pad_distance)` upon successful landing) and remove any per‑step termination reward. Monitor pad_distance closely. If fuel efficiency remains poor, add a small negative reward proportional to fuel used (e.g., `-0.1 * fuel_used`).