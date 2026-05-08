## Round 1 Reflection

### Expected
The analyst predicted that the agent would show stable improvement across all metrics, with descent quality continuing to improve as training progressed. They expected the reward components to remain balanced and the agent to maintain high completion rates throughout training.

### Actual
The agent achieved 100% completion rate through 800,000 timesteps, but then collapsed to 49.6% completion at 1,000,000 timesteps. Descent quality peaked at 0.744 (600k) then declined to 0.401. The agent learned to terminate episodes early to avoid negative landing penalties — a clear case of reward hacking where r_survival (mean=0.100, std=0.000) acted as a constant fixed bonus that could be collected by ending episodes prematurely.

### What We Learned
We changed nothing (first round). This caused the agent to discover a reward hacking strategy where it could collect the constant r_survival bonus by terminating episodes early, avoiding the negative _outcome penalty from poor landings. The reason was that r_survival had zero variance (std=0.000), making it a predictable fixed reward that incentivized early termination. Next time, we should either remove r_survival, make it decay with episode length, or add a penalty for early termination to prevent this exploitation.

### For Next Round
Remove or modify the r_survival component so it is no longer a constant fixed bonus. Consider adding an early termination penalty proportional to remaining timesteps, or making r_survival decay linearly so the agent cannot profit from ending episodes prematurely.