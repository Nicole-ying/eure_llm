## Round 6 Reflection

### Expected
The analyst predicted that increasing the exploration bonus weight to 1.0 and adding Gaussian action noise (σ=0.1) would break the repetitive gait, causing increased velocity variance, occasional falls (shorter episodes), and eventual discovery of higher-speed or more efficient gaits.

### Actual
The agent remained completely flat: episode length always 1000.0, no falls, no behavioral variation. However, forward reward jumped from 23.64 to 61.19 — the policy found a faster but still rigid gait. The exploration bonus (if present) was still negligible; the policy exploited the stronger forward reward gradient while ignoring the exploration term.

### What We Learned
Increasing exploration weight and adding noise did not induce variance or falls — the policy simply converged to a faster repetitive gait. | Why: The velocity-variance bonus can be trivially satisfied by any periodic output with constant amplitude; action noise was absorbed by the deterministic policy (no entropy regularization) and did not force state-space exploration. | Fix: Replace std-velocity bonus with a state-count novelty bonus (e.g., RND) that rewards visiting novel (pose, velocity) states, and add entropy regularization (or use SAC-style stochastic policies) to force genuine stochasticity.

### Abstract Principle
Exploration bonuses that reward variance of a single scalar (e.g., velocity std) are easily gamed by periodic attractors; to escape narrow local optima, use count-based or prediction-error novelty that explicitly rewards visiting unseen state regions.

### For Next Round
Abandon the velocity-variance exploration bonus. Implement a Random Network Distillation (RND) bonus that compares current state features to a fixed random network's output. Keep action noise (σ=0.1) but also add entropy regularization (α=0.01) to the policy loss. Reduce forward reward weight slightly (e.g., multiply by 0.9) to balance novelty vs. speed. Monitor state visitation histograms (e.g., velocity bins).