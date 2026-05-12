## Round {N} Reflection

### Expected
The analyst predicted that removing the constant positional baseline, increasing the velocity coefficient (0.2→0.5), adding a small action penalty, and fixing r_goal logging would force the agent to build momentum, oscillate, and eventually reach the goal.

### Actual
The agent never reached the goal (reached_goal=0), velocity remained negligible (~0.018 m/s), distance to goal stagnated (~1.16), and a constant -1 step penalty (`_outcome`) accounted for 99.7% of total reward. The tiny r_progress (mean 0.0032) was completely overshadowed by the penalty, so the agent settled into a low-force, low-velocity crawl that minimized negative reward but made no task progress.

### What We Learned
`Removed constant positive baseline but left constant -1 step penalty → agent ignored small velocity bonus and stalled | Why: constant penalty dominated reward; no positive signal strong enough to overcome it | Fix: also remove or condition the step penalty, or increase velocity reward to at least 1.0 to counteract the -1 per step`

### Abstract Principle
If a reward function contains any constant offset (positive or negative) with zero variance, the agent will converge to a policy that harvests or minimizes that offset, ignoring any small variable components. All constant offsets must be removed or made state-dependent to allow informative gradients to drive learning.

### For Next Round
Eliminate the `_outcome` step penalty entirely, or replace it with a small negative reward that only applies when velocity is below a threshold (e.g., -0.1 if |v| < 0.01). Increase the velocity coefficient to at least 1.0. Also add a small positive reward for decreasing distance to goal (e.g., +0.5 * (prev_dist – curr_dist)) to provide a directional signal.