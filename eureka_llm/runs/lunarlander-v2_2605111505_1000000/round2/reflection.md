## Round 2 Reflection

### Expected
The agent would learn to reduce horizontal distance from ~10.5 to <1.5, attempt ground contact only near the pad, and achieve positive survival fraction with lower vertical speed.

### Actual
The agent instead learned to hover at height ~11.5 with near-zero vertical speed (0.28) and angle (0.058 rad), staying far from the pad (dist_to_pad = 10.2). Legs never touch ground (0.0), survival fraction remains 0.0. The policy maximizes constant per-step rewards (+20 r_landing_bonus, -10 r_crash_penalty, +0.12 r_alive) that sum to ~+10 per step, providing >10,000 cumulative reward per episode – completely dominating any shaped signals.

### What We Learned
Constant per-step reward offsets (r_landing_bonus +20, r_crash_penalty -10) create a huge survival incentive that dwarfs shaped signals → agent hovers indefinitely far from pad | Why: large constant net +10/step rewards staying alive, not landing | Fix: remove constant per-step r_landing_bonus and r_crash_penalty; make them terminal sparse rewards only on actual landing/crash.

### Abstract Principle
When sparse terminal rewards are accidentally applied as dense constant offsets, they become the dominant learning signal and must be eliminated before any shaped reward can guide behavior toward the true goal.

### For Next Round
Rewrite the reward function to remove all constant per-step components. Keep only: a per-step survival penalty (e.g., –0.01), a per-step penalty proportional to distance from pad (e.g., –0.1 * dist_to_pad), a small bonus for reducing distance (dense directional), and terminal rewards: +20 for landing on pad (legs_ground and dist_to_pad < 1.5), –10 for crash (high vertical speed or off-pad ground contact). Condition leg contact bonus on being near the pad as proposed. Monitor episode length – if it stays near max, the survival penalty is still too weak.