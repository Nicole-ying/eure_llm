## Round Reflection

### Expected
The analyst predicted that adding a velocity bonus to r_progress and fixing the terminal bonus condition (only on goal) would break the stationary deadlock, causing the agent to move and eventually learn momentum-based goal reaching.

### Actual
The agent applied maximum force continuously but achieved negligible velocity (0.009), never reached the goal (reached_goal=0), and remained stuck on the incline. The r_goal component remained constant at 1.0 (std=0) across all episodes, indicating the terminal bonus fix was ineffective or not implemented, and the constant reward still dominated.

### What We Learned
`Velocity bonus too small (0.2*|v|) and constant r_goal=1.0 unchanged → agent learned max-force stall, no momentum | Why: constant reward masked movement incentive; terminal condition fix not applied | Fix: remove constant r_goal entirely, increase velocity coefficient (e.g., 0.5), ensure goal-only terminal bonus`

### Abstract Principle
Reward components that provide constant, state-independent positive feedback dominate learning and suppress any small gradient-driven exploration, leading to locally optimal but globally useless policies. Always remove or condition such constants on actual task progress.

### For Next Round
Remove the constant r_goal component (set to 0 unless goal achieved). Increase the velocity bonus to at least 0.5 * |velocity| and add a small penalty for zero velocity to force movement. Verify terminal bonus is only awarded on actual goal achievement (position >= 0.5 and velocity >= 0). Monitor that dist_to_goal decreases over time.