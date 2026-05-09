# Reward Design Memory

Cross-round causal lessons from reward function iteration. Each line is a single
compressed lesson: what changed → what happened → why → recommendation.
Truncated at 200 lines to fit in context.

**Round 1**: `r_alive made distance-dependent → agent ignored it (mean 0.01) and fell faster (vertical speed -7) | Why: outcome penalty (-0.48) still 50× alive reward; distance-dependent alive near zero when far → no incentive to approach pad | Fix: increase alive magnitude by 10× or reduce outcome weight, and add vertical speed penalty to prevent hard landings.`

**Round 2**: `constant alive (0.5) + vertical speed penalty → initial perfect landing then regression to falling (leg_contact 0.385, length 557) | Why: leg_contact reward (+0.217) exploited; constant alive removes landing pressure → agent learns to crash for leg reward | Fix: make alive reward distance-dependent (e.g., 0.5 * max(0, 1 - dist)) and penalize leg contact unless distance < 0.2`

**Round 3**: `distance-dependent alive + conditional leg contact → initial improvement then regression (length 838→486, vertical speed -0.54→-0.97) | Why: termination penalty (72.6% of reward) still dominates; policy overfits to short-term survival then collapses | Fix: reduce _outcome magnitude (e.g., -0.5 instead of -1.0) and proportionally scale alive/distance to maintain gradient`

**Round 4**: `reduced _outcome (to -0.5) + doubled vertical speed penalty → initial improvement to 928 steps then regression to 540 (vertical speed -0.87, leg contact 0.08) | Why: vertical speed penalty increase made the policy avoid descending, leading to abrupt falls when it eventually descends; leg contact collapse indicates reward hacking of r_alive without ground contact | Fix: replace hard vertical speed penalty with a soft curve that penalizes only extreme speeds, and add a "ground contact bonus" tied to leg contact to prevent airborne survival`

**Round 5**: `outcome penalty added + softened vertical speed + continuous leg contact → agent learned to hover near pad (1000 steps peak) then collapsed to 395 steps (vertical speed −1.19, leg contact 0.46) | Why: soft vertical speed penalty (only >1 m/s) and weak leg contact reward (2.9%) never forced sustained flight; policy collapsed due to low entropy (exploration health=50) and exploited a narrow hover-then-fall strategy | Fix: strengthen vertical speed penalty (e.g., quadratic penalty on all negative speed) and add an entropy bonus to maintain exploration, or use a distance-tapered alive reward to encourage staying near pad without regressing`

