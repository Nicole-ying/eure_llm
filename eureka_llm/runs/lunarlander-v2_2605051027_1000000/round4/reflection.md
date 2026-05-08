## Round 6 Reflection

### Expected
The analyst predicted that setting `r_termination` to zero (or a small positive constant) would eliminate the per‑step penalty that discouraged landing, allowing the agent to complete episodes more frequently.

### Actual
`r_termination` was set to a small positive value (mean 0.048, std 2.19). The completion rate spiked to 60% at 600k steps but then regressed to 20% by 1M steps. The agent still predominantly maximizes `r_progress` (68.9% of total reward) and frequently truncates episodes. The landing bonus (implied to be part of `r_termination` or separate) was not large enough to overcome the habit of hovering near the pad.

### What We Learned
We changed r_termination from a constant −1 penalty to a small positive reward (~0.05 per step). This initially boosted completion rate to 60% but caused the agent to regress to a hovering strategy because the progress reward still dominated and the landing bonus (part of termination reward) was too low to make finishing reliably more attractive than collecting progress over long episodes. Next time, increase the landing bonus to at least +10 and consider annealing the progress reward as the agent gets close to the pad, so that finishing becomes the highest‑value action.

### For Next Round
Set `r_termination` to zero (no per‑step reward) and add a large one‑time landing bonus of +10 (or +15) upon successful completion. Optionally, add a small negative reward for truncation (e.g., −5) to discourage time‑outs. Monitor whether the agent finally commits to landing rather than hovering.