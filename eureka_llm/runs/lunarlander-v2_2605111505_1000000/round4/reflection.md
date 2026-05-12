## Round 4 Reflection

### Expected
The agent would first move laterally toward the pad (dist < 5), then descend and achieve successful landings with dist_to_pad < 1, legs_ground > 0, and survival_fraction > 0 — thanks to stronger horizontal penalty (w_x=0.3), conditioned alive bonus (only within dist < 5), and increased crash penalty (-2).

### Actual
The agent learned to descend to height ~2.5 and make ground contact (legs_ground=0.43), but dist_to_pad remained ~9.8–10.2 throughout — no lateral movement at all. Survival fraction stayed 0.0. The reward analysis revealed that r_landing_bonus still dominates at 84.6% with zero variance (constant +20 per step), and r_alive remains active (mean 0.46) despite the dist<5 condition — indicating the proposed changes were either not implemented correctly or were insufficient to overcome the unconditional per-step landing bonus.

### What We Learned
Proposed conditioning on alive bonus and stronger horizontal penalty failed to induce lateral movement because the dominant per-step landing bonus (+20) remained unconditional and unchanged, allowing the agent to collect massive reward by touching ground anywhere → no incentive to move toward pad | Why: previous round reflection's explicit fix (terminal conditional landing bonus) was ignored; the alive bonus conditioning may also have been misimplemented given its non-zero value at dist~10 | Fix: Immediately implement the previous round's recommendation — make landing bonus terminal (single +20) conditioned on dist_to_pad < 1.5 AND vert_speed_abs < 0.5 AND legs_ground; remove all per-step landing bonus; verify all code changes by running a reward sanity check.

### Abstract Principle
A dominant, unconditional per-step constant reward will completely override any shaped incentives for spatial targeting — the only way to break such a local optimum is to remove the unconditional reward entirely and make it a terminal, conditional signal.

### For Next Round
First, verify that the previous round's terminal landing bonus fix is actually coded and active. Then ensure the alive bonus conditioning (dist<5) is correctly implemented (e.g., compute dist from state[0]). If both are in place, consider adding a strong per-step penalty proportional to distance from pad (e.g., -0.1*dist) to create a continuous gradient. Monitor the reward component breakdown early (every 50K steps) to catch any remaining dominant constant signals.