## Round 5 Reflection

### Expected
The analyst predicted that switching to an exponential forward reward and adding a velocity-variance exploration bonus would break the local optimum — forward velocity would increase, causing occasional falls (shorter episodes) and greater behavioral diversity.

### Actual
Episode length remained exactly 1000.0 at all evaluations (200K–1M). Reward components show `r_forward` dominates (mean 23.64) while `r_explore` is negligible (mean 0.14). The agent did not fall and did not vary its behavior — it likely found a higher-speed but still repetitive gait that satisfies both terms without true exploration.

### What We Learned
Exponential forward reward increased speed but did not induce falls or variation — velocity-variance exploration bonus (weight 0.1) was too weak relative to dominant forward reward, so policy maintained a repetitive high-speed gait. | Why: The exploration bonus (std of velocity buffer) can be trivially satisfied by any periodic output without true state-space coverage; its small weight is overpowered by the strong exponential gradient. | Fix: Increase exploration weight to ≥0.5 and add action noise (Gaussian σ=0.1) or entropy regularization to force stochastic exploration; monitor velocity variance directly.

### Abstract Principle
When adding an exploration bonus to escape a local optimum, its weight must be comparable to the dominant reward components — otherwise the policy will ignore it and settle into a new attractor that optimizes only the main reward.

### For Next Round
Increase `r_explore` weight to 0.5 and add Gaussian action noise (σ=0.1) during training. Log forward velocity mean and variance across episodes. If variance remains near zero, the policy is still trapped; consider replacing the velocity-std bonus with a state-count novelty bonus.