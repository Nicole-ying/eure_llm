## Round 7 Reflection

### Expected
The analyst predicted that increasing the speed penalty (from -0.05 to -0.3 coefficient) and adding an angle/angular velocity penalty would slow descent and improve landing posture, increasing both_legs_down from ~41% to ~70% and reducing falls from 10% to <5% while maintaining 90%+ completion.

### Actual
The agent achieved 100% completion at 600k (best ever) but then regressed to 80% completion with 20% falls at 1M. Both_legs_down reached 58% (improved from 41% but below the 70% target). Speed dropped from 1.32 to 0.79 (good), but distance_to_pad increased from 9.97 to 10.37 (bad). The agent learned to descend slowly but is now **hovering indecisively** near the pad rather than committing to landing — the penalties are strong enough to slow the agent but not strong enough to guide it through the final touchdown phase.

### What We Learned
`r_speed (0.3×) + r_angle (0.5×|angle|+0.3×|angvel|) → agent slows down (speed 0.79) and improves posture (both_legs 58%) but regresses to 80% completion with 20% falls | Why: penalties discourage fast/angled descent but don't incentivize final touchdown commitment — agent hovers safely instead of landing | Fix: add a terminal landing bonus (e.g., +5 for both_legs_down at episode end) and reduce r_speed to 0.2 to avoid over-penalizing approach speed`

### For Next Round
The speed and angle penalties successfully slowed the agent and improved landing posture, but created a new problem: the agent hovers near the pad to avoid penalties, leading to indecision and increased distance-to-pad. The core issue is that penalties discourage bad behavior but don't reward the final landing commitment. Add a terminal bonus for successful both_legs_down landings (+5 to +10) to create a clear incentive to complete the landing. Also reduce r_speed coefficient from 0.3 to 0.2 to give the agent more freedom to approach the pad at moderate speed.