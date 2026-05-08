## Round 4 Reflection

### Expected
The analyst predicted that reducing `r_distance` scaling from 0.3 to 0.1 and adding a speed penalty (`r_speed = -speed * 0.05`) would cause the agent to modulate throttle, reduce speed from ~8.3, and increase episode length through controlled descent.

### Actual
The agent continued falling in 100% of episodes (fall_rate = 1.0, completion_rate = 0.0). Mean episode length increased slightly to ~78 (from ~58 in Round 3), but this is still immediate failure. The agent improved posture (angle_from_vertical decreased from ~1.36 to ~0.77) and reduced angular velocity, but never approached the pad (distance_to_pad stuck at ~12.0). The `_outcome` penalty still dominated at 83.5% of total reward (mean -0.96), completely overwhelming the combined per-step rewards (r_alive=0.05, r_distance=-0.09, r_speed=-0.05).

### What We Learned
`_outcome penalty (-100) → agent learns to fall slowly (episode length ~78) but never lands | Why: outcome penalty still 20× larger than total per-step reward magnitude, and per-step rewards lack landing incentive | Fix: reduce _outcome from -100 to -1 AND add explicit landing reward (e.g., r_landing=+10 for both_legs_down) to create positive gradient toward pad`

### For Next Round
The previous round's lesson about `_outcome` dominance was ignored — the fix must address the terminal penalty magnitude first. Additionally, the agent is learning to stay upright (positive) but has zero incentive to reach the pad because distance penalty is now too weak (-0.09 per step) and there's no reward for landing. Add a sparse landing bonus (+10 to +50) when both_legs_down is True, and reduce `_outcome` from -100 to -1 to prevent the "fail fast" or "fail slowly" degenerate strategies.