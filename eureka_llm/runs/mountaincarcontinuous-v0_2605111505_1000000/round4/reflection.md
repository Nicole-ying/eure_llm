## Round {N} Reflection

### Expected
The analyst predicted that increasing the absolute velocity reward (2.0 * abs(velocity)) and removing the signed direction reward would cause the agent to oscillate left and right, build momentum, and eventually reach the goal. Total reward was expected to rise to ~0.04 per step and velocity >0.1.

### Actual
The agent never reached the goal (reached_goal=0). Velocity remained at ~0.025 m/s, distance to goal stagnated near 0.96, and the `_outcome` component dominated 96.8% of reward. The agent applied near‑max force monotonically without any oscillation. Episode length stayed at 999 except for a brief early‑termination phase at 600k.

### What We Learned
`Left _outcome unchanged → agent minimized net reward by staying nearly stationary (velocity ~0.025) and avoiding movement | Why: _outcome (−0.99) dominated 96.8% of reward, drowning out all shaping signals | Fix: remove or reduce _outcome by ≥10×, or make it contingent on active progress`

### Abstract Principle
When a reward function contains a large constant‑offset penalty, any auxiliary rewards must be of comparable magnitude to influence policy; otherwise the agent will adopt a cost‑minimizing strategy (e.g., staying still or terminating early) rather than pursuing the task.

### For Next Round
Eliminate the `_outcome` component entirely, or set its magnitude to ≤0.1. Re‑introduce a strong signed velocity reward (≥1.0) and a large distance‑decrease bonus (e.g., 5.0 * Δdist). Also consider a small penalty for applying force without velocity change to discourage inefficient pushing.