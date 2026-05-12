## Round {N} Reflection

### Expected
The analyst predicted that reducing the `r_smooth` weight from 0.01 to 0.001 would increase total reward per step by ~0.081, and that forward velocity might rise by ~20% due to relaxed smoothness constraint, leading to a net improvement in running speed.

### Actual
No change occurred: the policy remained stuck at a fixed, repetitive gait with constant episode length (1000.0) and flat forward reward (~1.25). `r_smooth` was already negligible (mean -0.009, std 0.001) and contributed only 0.7% of total reward. The agent never explored faster or more efficient gaits; it was in a local optimum.

### What We Learned
Reducing `r_smooth` weight had zero effect because the agent already ignored smoothness (near-zero reward) — the real root cause was a deterministic policy lacking state coverage and exploration. | Why: analyst misidentified the bottleneck as reward weighting when the actual issue is behavioral collapse (no variance in episode length or forward speed). | Fix: Introduce an exploration bonus (e.g., entropy regularization, speed gradient reward, or random action injection) to break the local optimum.

### Abstract Principle
When a policy shows zero variance in outcomes (deterministic, fixed gait), reward component tuning alone cannot induce improvement — the agent must be forced out of its local optimum through explicit exploration mechanisms or diversity-promoting objectives.

### For Next Round
Do not adjust reward weights until you have confirmed the policy is actually exploring. First, measure forward velocity variation and action entropy. If both are near-zero, add an exploration bonus (e.g., state visitation count, action noise schedule) before modifying any reward coefficients.