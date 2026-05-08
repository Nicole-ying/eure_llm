## Round 3 Reflection

### Expected
The analyst predicted that increasing `r_alive` from 0.01 to 0.05 and reducing `r_distance` scaling from 0.5 to 0.3 would increase mean episode length from ~63 to ~100+, enable exploration, and raise completion rate above 0%.

### Actual
The agent continued falling immediately with no improvement: completion rate remained 0.000, fall rate 1.000, mean episode length actually decreased slightly to ~58. The `_outcome` penalty (mean -0.979) dominated 74.3% of total reward, completely overwhelming the small adjustments to `r_alive` and `r_distance`.

### What We Learned
`_outcome penalty (-100) → agent learns to fail fast (episode length ~58) | Why: outcome penalty magnitude is 2000× larger than per-step rewards (r_alive=0.05, r_distance=-0.289) | Fix: scale down _outcome penalty from -100 to -1 or implement reward normalization across components`

### For Next Round
The core issue is not the balance between alive and distance rewards, but the catastrophic dominance of the terminal outcome penalty. Reduce `_outcome` from -100 to -1 (matching per-step reward magnitudes), or implement adaptive reward normalization. Without this fix, no amount of tuning per-step components will overcome the signal that dying quickly is optimal.