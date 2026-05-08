# Reward Design Memory

Cross-round causal lessons from reward function iteration. Each line is a single
compressed lesson: what changed → what happened → why → recommendation.
Truncated at 200 lines to fit in context.

**Round 1**: `_outcome penalty (-100) → agent learns to fail fast (episode length ~58) | Why: outcome penalty magnitude is 2000× larger than per-step rewards (r_alive=0.05, r_distance=-0.289) | Fix: scale down _outcome penalty from -100 to -1 or implement reward normalization across components`

**Round 2**: `_outcome penalty (-100) → agent learns to fall slowly (episode length ~78) but never lands | Why: outcome penalty still 20× larger than total per-step reward magnitude, and per-step rewards lack landing incentive | Fix: reduce _outcome from -100 to -1 AND add explicit landing reward (e.g., r_landing=+10 for both_legs_down) to create positive gradient toward pad`

**Round 3**: `r_landing=+10 (sparse terminal) → agent initially learns to land (completion=1.0) then catastrophically forgets (completion=0.0) | Why: sparse terminal reward +10 dominates (67.5% of total), creating unstable learning dynamics — agent oscillates between landing and falling | Fix: replace sparse landing bonus with dense per-step shaping reward (e.g., r_progress = altitude_change * -0.1 + distance_change * -0.1) to provide continuous gradient toward pad`

**Round 4**: `r_progress (dense shaping) → agent learns to land reliably (90% completion at 1M) without catastrophic collapse | Why: continuous gradient replaces sparse terminal signal, eliminating oscillation | Fix: add explicit stability penalty (e.g., r_stability = -|angle|*0.5 - |angular_velocity|*0.3) to prevent speed-over-stability tradeoff`

**Round 5**: `r_speed (0.3×) + r_angle (0.5×|angle|+0.3×|angvel|) → agent slows down (speed 0.79) and improves posture (both_legs 58%) but regresses to 80% completion with 20% falls | Why: penalties discourage fast/angled descent but don't incentivize final touchdown commitment — agent hovers safely instead of landing | Fix: add a terminal landing bonus (e.g., +5 for both_legs_down at episode end) and reduce r_speed to 0.2 to avoid over-penalizing approach speed`

