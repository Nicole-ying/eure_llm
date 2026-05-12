## Round 4 Reflection

### Expected
The analyst predicted that changing `r_delta` from one-sided (increase-only) to absolute difference would cause the agent to vary speed, explore different gaits, and increase forward velocity, while still completing full episodes.

### Actual
The agent remained in a fixed, repetitive gait with constant episode length (1000.0). `r_delta` remained active (mean 2.70, std 0.47) but did not induce any episode-to-episode variation — the policy had found a periodic oscillation that satisfied the absolute difference reward without exploring new speeds.

### What We Learned
Changing r_delta to absolute difference did not induce speed variation — the policy maintained a fixed periodic gait because the reward can be satisfied by constant periodic oscillation without exploring new speeds. | Why: The root cause is the absence of any mechanism (e.g., entropy, noise, termination) that forces behavioral diversity; reward shaping alone cannot break a local optimum if the policy can trivially satisfy the new term within its existing attractor. | Fix: Introduce explicit exploration bonuses (action noise, entropy regularization, or state visitation count) and monitor forward velocity variance directly (not just episode length) to detect subtle speed changes.

### Abstract Principle
Reward shaping that targets a single numerical derivative (even absolute) can be gamed by periodic policies without inducing broad state exploration; to break local optima, either introduce stochasticity in actions or provide novelty bonuses that incentivize visiting new regions of the state space.

### For Next Round
Add action noise (e.g., Gaussian noise with decaying variance) or entropy regularization to the policy before making any further reward changes. Also log forward velocity variance across episodes and compare it to the variance of the delta reward to confirm whether the agent is actually varying speed or just oscillating. If variance remains near zero, the policy is still trapped.