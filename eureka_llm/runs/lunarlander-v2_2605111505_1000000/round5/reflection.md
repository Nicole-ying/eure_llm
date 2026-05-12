## Round 5 Reflection

### Expected
The agent would begin moving laterally toward the pad because the per-step constant +20 landing bonus was made terminal, the alive bonus gate was removed, and the horizontal penalty weight was increased to 0.5. Net reward at hover would be ~ -4.5, driving the agent to reduce distance.

### Actual
The agent continued to hover at height ~2.84, distance ~9.73, with zero lateral movement and survival fraction 0.0. The reward analysis shows `r_landing_bonus` still has mean 20.0 and std 0.0, indicating it remains a per-step constant reward. The proposed code changes were not actually implemented or were ineffective.

### What We Learned
Proposed terminal-landing-bonus fix had no effect because the constant per-step +20 reward was still present unchanged → agent still collects massive unconditional reward by staying alive near ground | Why: the code change was either not applied, not saved, or overridden; the reward component trace shows zero variance, proving the old unconditional per-step logic is still running | Fix: Immediately verify the actual reward function code (not just proposal text) to confirm the terminal landing bonus is the only source of +20 and that per-step addition is removed; run a reward sanity check printing reward breakdown for one episode before training.

### Abstract Principle
A proposed reward fix that is not verified against the actual running code cannot change behavior; cross-round memory must include a verification step that checks the reward component statistics match the intended design before concluding the change is active.

### For Next Round
First, examine the actual reward computation code in the environment to ensure the `r_landing_bonus` is only added on successful termination and not every step. Print a reward trace for one episode. Then, if still broken, remove the unconditional constant entirely and replace with a terminal bonus conditioned on exact landing criteria. Consider removing `r_crash_penalty` constant as well (std=0) to eliminate all constant offsets.