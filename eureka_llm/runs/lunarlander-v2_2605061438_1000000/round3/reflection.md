## Round 5 Reflection

### Expected
The analyst predicted that increasing `r_alive` from 0.05 to 0.5, reducing `r_distance` from -0.1 to -0.05, and adding a landing reward of +10 would incentivize survival and create a positive gradient toward landing, raising completion rate to 0.3+ within 3-5 rounds.

### Actual
The agent initially improved dramatically — learned to descend from altitude ~42 to ~2.7, approach the pad (distance dropped from 44 to 7), and maintain stability. However, by 1M timesteps, it **catastrophically collapsed**: completion rate dropped from 1.0 to 0.0, fall rate spiked to 1.0. The landing reward (r_landing mean=2.24) dominated at 67.5% of total reward, creating a sparse terminal signal that destabilized learning. The agent fell into a local optimum of dying quickly rather than attempting controlled landings.

### What We Learned
`r_landing=+10 (sparse terminal) → agent initially learns to land (completion=1.0) then catastrophically forgets (completion=0.0) | Why: sparse terminal reward +10 dominates (67.5% of total), creating unstable learning dynamics — agent oscillates between landing and falling | Fix: replace sparse landing bonus with dense per-step shaping reward (e.g., r_progress = altitude_change * -0.1 + distance_change * -0.1) to provide continuous gradient toward pad`

### For Next Round
The landing reward was too sparse and too dominant — it created a "feast or famine" signal that caused catastrophic forgetting. Replace the terminal landing bonus with a dense per-step progress reward that gives continuous feedback for moving toward the pad and descending. Keep `r_alive=0.5` but ensure no single component exceeds 40% of total reward magnitude. Monitor reward component balance every 200k timesteps.