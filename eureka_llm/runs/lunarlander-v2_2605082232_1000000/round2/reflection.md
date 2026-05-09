```markdown
## Round 4 Reflection

### Expected
Increasing r_alive to a constant 0.5 and adding a vertical speed penalty (−0.05 * |vz|) would make the agent survive longer, control descent, and eventually learn to land.

### Actual
The agent achieved near-perfect behavior at 400k–600k (mean_length=1000, distance≈0.5, vertical_speed≈−0.4, leg_contact=0.0) but then regressed by 1M (mean_length=557, vertical_speed=−0.84, leg_contact=0.385). The agent now falls faster and frequently makes ground contact, terminating early.

### What We Learned
`constant alive (0.5) + vertical speed penalty → initial perfect landing then regression to falling (leg_contact 0.385, length 557) | Why: leg_contact reward (+0.217) exploited; constant alive removes landing pressure → agent learns to crash for leg reward | Fix: make alive reward distance-dependent (e.g., 0.5 * max(0, 1 - dist)) and penalize leg contact unless distance < 0.2`

### For Next Round
Replace the constant alive reward with a distance-dependent one that gives high reward only when close to the pad, to maintain pressure to land. Modify the leg_contact reward to be conditional on being near the pad (e.g., reward only if distance < 0.2 and vertical speed near zero), preventing the agent from falling deliberately. Consider adding an entropy bonus to discourage policy collapse during training.
```