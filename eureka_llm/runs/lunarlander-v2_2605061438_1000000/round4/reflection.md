## Round 6 Reflection

### Expected
The analyst predicted that replacing the sparse terminal landing reward (+10) with dense per-step shaping rewards (progress toward pad, descending, legs down) would eliminate catastrophic oscillation and produce stable learning with sustained high completion rates. The main risk identified was that shaping coefficients might be too large, causing uncontrolled descent.

### Actual
The agent showed dramatic improvement — from 0% completion at 200k to 100% at 800k, then slight regression to 90% at 1M. The catastrophic collapse seen in Round 5 was eliminated. However, the regression from 800k→1M (speed increasing from 0.57 to 1.32, legs-down dropping from 0.67 to 0.41) indicates the agent is converging to a local optimum that prioritizes speed over stability — landing faster but less reliably. The shaping reward worked, but the agent is now optimizing for speed of descent rather than stable landing posture.

### What We Learned
`r_progress (dense shaping) → agent learns to land reliably (90% completion at 1M) without catastrophic collapse | Why: continuous gradient replaces sparse terminal signal, eliminating oscillation | Fix: add explicit stability penalty (e.g., r_stability = -|angle|*0.5 - |angular_velocity|*0.3) to prevent speed-over-stability tradeoff`

### For Next Round
The dense shaping reward successfully solved the catastrophic forgetting problem — the agent now learns progressively and maintains high completion rates. However, the 800k→1M regression shows the agent is optimizing for speed over stability (higher speed, fewer legs-down landings). Add a per-step stability penalty that penalizes large angles and angular velocities to encourage controlled, stable landings. Also consider increasing the r_near_landing bonus from 2.0 to 5.0 to incentivize proper leg-down posture at touchdown.