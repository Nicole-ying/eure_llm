## Round 1 Reflection

### Expected
The analyst predicted that replacing the sparse proximity penalty with a shaped reward (penalizing x-distance, y-distance, angle, and velocity) and making the alive bonus orientation-dependent would create a smooth gradient guiding the agent to center, upright, slow landings.

### Actual
The agent remained stuck at 0% completion rate across 1M steps, with distance_to_pad unchanged at ~12.27, speed high at 7.4, and angle_error only slightly improved to 0.12. The shaped reward (r_shaped = -0.12) was dominated by the outcome penalty (-0.43, 74% of total magnitude), and the alive bonus remained functionally constant at 0.02 regardless of orientation.

### What We Learned
`r_shaped=-0.12, r_alive=0.02 → agent still falls immediately (0% completion) | Why: outcome penalty (-0.43) dominates reward 3.6×, drowning out shaped gradient; alive bonus not conditional enough to overcome crash penalty | Fix: reduce outcome penalty magnitude by 10× or increase shaped coefficients 5× so gradient signals compete with terminal reward`

### For Next Round
The shaped reward coefficients are too weak relative to the terminal outcome penalty. Either scale down the crash penalty (e.g., from -100 to -10) or scale up the per-step shaping (r_shaped coefficients 5-10× larger) so the agent can learn from the gradient before dying. Also verify the alive bonus actually varies with angle — the mean of 0.02 suggests it's stuck at max.