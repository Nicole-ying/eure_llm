## Round 3 Reflection

### Expected
The analyst predicted that reducing the magnitude of the constant termination penalty (r_termination) or introducing a positive incentive for approaching the pad would break the agent out of its hovering behavior and lead to gradual improvement in completion rate.

### Actual
The agent continued to hover at a fixed distance from the pad, maintaining near‑zero angle deviation and constant vertical speed. Completion rate remained 0%, fall rate 0%, and mean episode length stayed at 72.3 steps. r_termination remained dominant (−15.0 per step, 93.9% of total reward), drowning out all other reward components. No progress toward landing was observed.

### What We Learned
We changed the reward structure by reducing r_termination or adding a progress bonus. This caused no behavioral change because the termination penalty still overwhelmed all other signals — the agent simply minimized episode length by hovering and truncating early. The reason was that the penalty was still 20× larger than any other component, making state distinctions invisible. Next time, we must first neutralize the dominant reward component by either removing the constant termination penalty entirely or rescaling it to be comparable in magnitude to the other rewards.

### For Next Round
Eliminate the constant −15 termination penalty. Replace it with a small negative penalty only for time steps spent far from the pad, and introduce a strong positive reward gradient for decreasing pad_distance and reducing vertical speed. Ensure all reward components have similar orders of magnitude (e.g., scale to [−1, 1]).