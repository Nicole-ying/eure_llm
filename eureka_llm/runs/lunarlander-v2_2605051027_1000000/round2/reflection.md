## Round 4 Reflection

### Expected
The analyst predicted that eliminating the constant −15 termination penalty and replacing it with a small penalty for being far from the pad, plus a positive reward gradient for decreasing pad_distance and vertical speed, would break the hovering behavior and allow gradual improvement in landing rate.

### Actual
The agent progressed from 0% to 50% completion rate, with no falls. Episodes lengthened from ~70 to ~600 steps, pad_distance fell from 0.99 to 0.37, vertical speed dropped from 0.91 to 0.15, and leg contact rose from 0.03 to 0.60. Performance plateaued at 50% after 800k timesteps. The r_termination penalty remained constant at −1.0 per step (rescaled from −15), still dominant but no longer overwhelming.

### What We Learned
We reduced the constant termination penalty from −15 to −1 per step. This caused the agent to learn to approach the pad and land in half of episodes, because the previously dominant signal no longer drowned out other reward components. The reason was that shaping factors (r_progress, r_contact) could now influence the policy. Next time, convert r_progress from a negative penalty (−0.37) into a positive reward for reducing distance and vertical speed, and increase the reward for successful landings to push beyond the 50% plateau.

### For Next Round
Make r_progress positive (e.g., proportional to improvement in pad_distance and vertical_speed) and add a one‑time bonus for completing a landing. Keep r_termination at a small negative or zero to avoid time‑pressure bias. Ensure all reward components are within [−1, 1] scale.