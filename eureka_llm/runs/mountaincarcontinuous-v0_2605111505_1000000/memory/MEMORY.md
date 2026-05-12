# Reward Design Memory

Cross-round causal lessons from reward function iteration. Each line is a single
compressed lesson: what changed → what happened → why → recommendation.
Truncated at 200 lines to fit in context.

**Round 1**: `Velocity bonus too small (0.2*|v|) and constant r_goal=1.0 unchanged → agent learned max-force stall, no momentum | Why: constant reward masked movement incentive; terminal condition fix not applied | Fix: remove constant r_goal entirely, increase velocity coefficient (e.g., 0.5), ensure goal-only terminal bonus`

**Round 2**: `Removed constant positive baseline but left constant -1 step penalty → agent ignored small velocity bonus and stalled | Why: constant penalty dominated reward; no positive signal strong enough to overcome it | Fix: also remove or condition the step penalty, or increase velocity reward to at least 1.0 to counteract the -1 per step`

**Round 3**: `Added signed velocity and proximity penalty but _outcome still dominated (98%) → agent drove right monotonically without building momentum | Why: constant _outcome reward dwarfed all new signals, and signed velocity discouraged leftward exploration needed for oscillation | Fix: remove or condition _outcome, increase velocity coefficient to ≥1.0, and add explicit oscillation bonus or negative distance gradient`

**Round 4**: `Left _outcome unchanged → agent minimized net reward by staying nearly stationary (velocity ~0.025) and avoiding movement | Why: _outcome (−0.99) dominated 96.8% of reward, drowning out all shaping signals | Fix: remove or reduce _outcome by ≥10×, or make it contingent on active progress`

**Round 5**: `Removed _outcome & increased r_velocity → agent ignored velocity signal (r_velocity absent in active components) and stayed stationary with high force | Why: The proposed reward change was not reflected in the actual reward function – only r_progress remained active, drowning out any velocity shaping; the agent is effectively optimizing a weak progress signal that does not incentivize momentum building | Fix: Verify that the reward function modification is correctly implemented (ensure r_velocity is included and has non‑zero weight) and redesign r_progress to explicitly reward back‑and‑forth oscillation (e.g., signed velocity or momentum bonus)`

