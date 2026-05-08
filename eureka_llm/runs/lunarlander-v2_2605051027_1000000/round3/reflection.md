## Round 5 Reflection

### Expected
The analyst predicted that making `r_progress` a positive reward for reducing pad_distance and vertical_speed, plus adding a one‑time landing bonus, would push the completion rate beyond the 50% plateau observed in Round 4.

### Actual
The completion rate initially rose to 60% at 600k steps, then regressed to 30% by 1M steps. The agent learned to hover near the pad (pad_distance 0.40, vertical_speed 0.09) and increased leg contact, but rarely executed a successful landing. The per‑step `r_termination` remained a constant −1.0, dominating 47% of total reward, effectively penalizing every timestep and discouraging the final approach.

### What We Learned
We made r_progress a positive reward and added a landing bonus. This initially boosted completion rate to 60%, but the constant per‑step termination penalty (−1) created a conflicting incentive: the agent learned to prolong hovering near the pad to collect progress reward while avoiding the additional time penalty that completing the landing would require. Next time, either remove the per‑step termination penalty entirely (set it to zero) or convert it into a small positive reward for surviving, so that the agent is not punished for taking the time needed to land precisely.

### For Next Round
Set `r_termination` to zero (or a small positive constant like +0.1 per step) and keep the positive progress reward and landing bonus from Round 5. Verify that the landing bonus is large enough (e.g., +5 or +10) to outweigh any lingering time penalty. Monitor whether the agent becomes willing to finish episodes instead of hovering indefinitely.