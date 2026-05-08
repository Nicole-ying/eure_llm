# Reward Design Memory

Cross-round causal lessons from reward function iteration. Each line is a single
compressed lesson: what changed → what happened → why → recommendation.
Truncated at 200 lines to fit in context.

**Round 1**: `_outcome removal proposed → component remained active (mean -0.222 unchanged) | Why: proposed code change was not actually applied to the environment | Fix: verify reward function modifications are committed and reloaded before training, or ensure analyst predictions reflect actual deployed code`

**Round 2**: `_outcome fix + r_stability added → agent collapsed for 1.5M steps then recovered to perfect performance | Why: moving _outcome before sum changed reward timing, creating initial instability as agent learned new credit assignment; stability bonus added constraint that delayed convergence | Fix: when fixing reward timing bugs, expect temporary performance regression; consider phased rollout (fix timing first, add stability later) or reduce stability coefficient to 0.2`

