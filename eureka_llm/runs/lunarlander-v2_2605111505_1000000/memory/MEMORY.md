# Reward Design Memory

Cross-round causal lessons from reward function iteration. Each line is a single
compressed lesson: what changed → what happened → why → recommendation.
Truncated at 200 lines to fit in context.

**Round 1**: Fixed vertical speed penalty but failed to incentivize horizontal movement → agent hovers low and touches ground far from pad | Why: proximity reward too weak (0.0165), leg contact reward rewards off-pad crashes, and no penalty for being far from pad | Fix: add strong negative reward for distance to pad (e.g., -0.1 * dist_to_pad per step) and make leg contact bonus conditional on being within pad radius (e.g., dist_to_pad < 1.5).

**Round 2**: Constant per-step reward offsets (r_landing_bonus +20, r_crash_penalty -10) create a huge survival incentive that dwarfs shaped signals → agent hovers indefinitely far from pad | Why: large constant net +10/step rewards staying alive, not landing | Fix: remove constant per-step r_landing_bonus and r_crash_penalty; make them terminal sparse rewards only on actual landing/crash.

**Round 3**: Height-dependent alive bonus and reduced crash penalty successfully drove descent and ground contact, but the unchanged landing bonus (still unconditional on pad proximity) allowed reward hacking → agent touches ground anywhere far from pad | Why: landing bonus not conditioned on pad proximity; horizontal penalty still too weak relative to +20 per step | Fix: condition landing bonus on dist_to_pad < 1.5 AND low vert_speed; also make landing bonus terminal (single +20) not per-step.

**Round 4**: Proposed conditioning on alive bonus and stronger horizontal penalty failed to induce lateral movement because the dominant per-step landing bonus (+20) remained unconditional and unchanged, allowing the agent to collect massive reward by touching ground anywhere → no incentive to move toward pad | Why: previous round reflection's explicit fix (terminal conditional landing bonus) was ignored; the alive bonus conditioning may also have been misimplemented given its non-zero value at dist~10 | Fix: Immediately implement the previous round's recommendation — make landing bonus terminal (single +20) conditioned on dist_to_pad < 1.5 AND vert_speed_abs < 0.5 AND legs_ground; remove all per-step landing bonus; verify all code changes by running a reward sanity check.

**Round 5**: Proposed terminal-landing-bonus fix had no effect because the constant per-step +20 reward was still present unchanged → agent still collects massive unconditional reward by staying alive near ground | Why: the code change was either not applied, not saved, or overridden; the reward component trace shows zero variance, proving the old unconditional per-step logic is still running | Fix: Immediately verify the actual reward function code (not just proposal text) to confirm the terminal landing bonus is the only source of +20 and that per-step addition is removed; run a reward sanity check printing reward breakdown for one episode before training.

