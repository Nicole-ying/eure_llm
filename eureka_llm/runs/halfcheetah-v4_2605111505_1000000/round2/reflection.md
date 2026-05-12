## Round 3 Reflection

### Expected
The analyst predicted that adding a velocity gradient bonus (`r_delta`) and increasing the forward reward scaling (1.5×) would break the local optimum, causing the agent to accelerate, produce higher forward velocities, and generate variable episode lengths (including early falls).

### Actual
The agent remained in a fixed, repetitive gait with **constant episode length = 1000.0** (no early termination). The new `r_delta` component did become active (mean 0.769), and `r_forward` increased (mean 2.748), but the policy did not explore faster speeds or change its behavior — episode length span ratio stayed at 0.0, indicating zero variance.

### What We Learned
Adding velocity gradient and increasing forward reward did not break the local optimum — the agent maintained constant episode length and stable gait, because the environment has no early termination (fixed horizon) so the policy can safely satisfy the new reward without exploring riskier states. | Why: The root cause is not reward weighting but the absence of any termination condition that would force adaptation; the policy simply optimizes within its existing attractor. | Fix: First verify environment termination conditions; if none exist, add explicit exploration mechanisms (e.g., entropy regularization, action noise schedule, or state visitation bonus) rather than reward reweighting alone.

### Abstract Principle
In environments with fixed horizons (no early termination), reward shaping alone cannot induce behavioral diversity — the policy can indefinitely remain in a local optimum; exploration must be explicitly incentivized through stochasticity or novelty bonuses.

### For Next Round
Check whether the environment has any early termination condition (e.g., falling, exceeding speed threshold). If not, add action noise or entropy bonus to force gait variation before modifying reward coefficients. Also measure forward velocity variance directly (not just episode length) to confirm whether speed is truly stagnant.