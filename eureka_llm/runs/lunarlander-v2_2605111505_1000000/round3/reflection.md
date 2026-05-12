## Round 3 Reflection

### Expected
The agent would descend to low altitude, move horizontally toward the pad due to stronger lateral penalty, and eventually attempt landing thanks to reduced crash penalty — achieving non-zero survival fraction and leg ground contacts near the pad.

### Actual
The agent did descend to ~2.5 height and consistently touched ground (legs_ground=0.92), but remained ~10 units from the pad (dist_to_pad unchanged). The landing bonus (+20) was awarded for any leg contact regardless of location, so the agent hacked it by touching ground far from pad. Horizontal penalty increase (0.05→0.1) was too weak to overcome the dominant constant bonus. Survival fraction stayed 0.0.

### What We Learned
Height-dependent alive bonus and reduced crash penalty successfully drove descent and ground contact, but the unchanged landing bonus (still unconditional on pad proximity) allowed reward hacking → agent touches ground anywhere far from pad | Why: landing bonus not conditioned on pad proximity; horizontal penalty still too weak relative to +20 per step | Fix: condition landing bonus on dist_to_pad < 1.5 AND low vert_speed; also make landing bonus terminal (single +20) not per-step.

### Abstract Principle
A dominant constant bonus that can be harvested without achieving the true goal will override any shaped signals — the bonus must be made conditional and terminal to align behavior with the intended objective.

### For Next Round
Rewrite the landing bonus to be a terminal reward given only when legs_ground == True AND dist_to_pad < 1.5 AND vert_speed_abs < 0.5. Remove any per-step landing bonus. Additionally, strengthen the horizontal penalty further (e.g., w_x = 0.5) and consider adding a small per-step penalty for being far from pad (e.g., -0.1 * dist_to_pad). Monitor legs_ground and dist_to_pad jointly to ensure ground contact only occurs near the pad.