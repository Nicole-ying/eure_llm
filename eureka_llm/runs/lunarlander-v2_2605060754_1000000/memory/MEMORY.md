# Reward Design Memory

This file stores cross-round causal lessons learned during reward function iteration.
Each entry is a concise lesson: "what changed → what happened → why".
This file is truncated at 200 lines to fit in context.

**Round 1**: We changed nothing (first round). This caused the agent to discover a reward hacking strategy where it could collect the constant r_survival bonus by terminating episodes early, avoiding the negative _outcome penalty from poor landings. The reason was that r_survival had zero variance (std=0.000), making it a predictable fixed reward that incentivized early termination. Next time, we should either remove r_survival, make it decay with episode length, or add a penalty for early termination to prevent this exploitation.

