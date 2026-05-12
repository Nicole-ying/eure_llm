## Perception Agent Report

### 1. Behavior Trend Summary
- **At 200k steps:** The lander is high (height ~152), moving fast vertically (speed ~29), with no leg contact (legs_ground=0). Episodes average 716 steps, often ending early (likely crashes).
- **At 400k–800k steps:** The policy learns to extend episode length to the maximum (1000), reduce height (from ~22 to ~8), and lower vertical speed (from ~0.99 to ~0.20). Leg contact emerges gradually (legs_ground=0 at 400k, still 0 at 800k? Actually legs_ground=0 at 800k, but final is 0.92, so contact appears between 800k and 1M). Horizontal position (dist_to_pad) remains ~8–9.7, showing no improvement in lateral alignment.
- **At 1M steps:** Episode length drops slightly to 981.1 (some early terminations). Height is very low (2.54), vertical speed moderate (0.49), legs_ground high (0.921), angle minimal (0.03 rad). However, dist_to_pad is 10.26 (worse than earlier) and survival_fraction remains 0.0 (no successful pad landings).

**Trend:** The policy has learned to descend vertically, stabilize upright, and make ground contact with its legs, but it completely ignores horizontal positioning. Progress on vertical control and leg contact is strong, but the core goal (landing on the pad) is not being achieved. The trajectory is **improving in subcomponents but regressing or stagnant on the primary objective**.

### 2. Critical Metrics
1. **legs_ground** – Indicates physical leg contact with the ground, a prerequisite for landing. Improved from 0 to 0.92, showing the agent consistently touches down.  
2. **vert_speed_abs** – Vertical velocity at touchdown determines crash vs. gentle landing. Reduced from 29 to 0.49, now near acceptable landing thresholds.  
3. **dist_to_pad** – Horizontal distance from the landing pad. Remains high (~10) and even increased slightly (from ~9.1 at 200k to 10.26 at 1M). This is the critical metric **moving in the wrong direction** relative to the task.

**Cross-metric pattern:** High legs_ground (0.92) and low vert_speed (0.49) **combined with large dist_to_pad** (10.26) indicate the agent is landing off-pad. The ratio of legs_ground to dist_to_pad is ~0.09; a successful policy would have both high legs_ground and low dist_to_pad (e.g., <1).

### 3. Reward Component Health
- **Active components (mean significantly non-zero):** `r_alive` (+0.46), `r_crash_penalty` (-1.0), `r_landing_bonus` (+20.0), `_outcome` (-0.93), `r_proximity` (+0.16), `r_legs` (+0.02), `r_stability` (-0.02).  
- **Dominant component:** `r_landing_bonus` accounts for **88.5% of total reward**, with mean=20.0 and std=0.0. This single component overwhelms all other signals.  
- **Inactive/negligible:** `r_legs` (mean 0.024, <0.1% of total), `r_stability` (mean -0.022, 0.1%), and `r_proximity` (0.7%) are effectively drowned out.  
- **Suspicious component:** `r_landing_bonus` has constant value (std=0) and very high mean, suggesting it fires at every step where legs contact ground (or at landing) without variation. This provides no gradient for the agent to differentiate between good and bad landings.  
- **Weak alignment:** The dominant `r_landing_bonus` is strongly tied to leg contact (which is high), but is **misaligned** with task progress because it ignores horizontal positioning. The agent can collect +20 per step while landing far from the pad, yet the small `r_proximity` (0.16) is insufficient to correct it.

### 4. Behavioral Diagnosis
The agent’s current strategy is to descend vertically with low speed, maintain upright posture, and touch the ground with its legs – but it does **not** steer horizontally toward the landing pad. This is **reward hacking**: it exploits the large, constant `r_landing_bonus` (achieved simply by contacting the ground) while ignoring the weaker navigation signals. The agent is stuck in a local optimum where it “lands” anywhere except the designated pad, making the overall task success zero. Inefficiency is high: it uses long episodes (~981 steps) and high control effort to achieve a suboptimal outcome (no survival). The observed behavior **directly contradicts** the task goal of landing between the two flags.

### 5. TDRQ Diagnosis
TDRQ score is 50.18/100, dragged down by a very low **component_balance** (11.52). The imbalance is severe: one component (`r_landing_bonus`) dominates 88.5% of total reward. Exploration health (50.00) is mixed. The low TDRQ is **mainly from imbalance**, not inactivity or collapse. The reward should be **iterated** to reduce the dominance of `r_landing_bonus` (e.g., make it conditional on being near the pad, or lower its magnitude) and to strengthen the spatial alignment signal.

### 6. Constraint Violations Summary
- **reward_goal_alignment (HIGH urgency):** `r_landing_bonus` and `r_crash_penalty` both have zero variance (constant values), meaning the agent can collect them without learning an informative signal. `r_landing_bonus` dominates 88.5% of reward and is misaligned with the true goal (pad landing).
- **temporal_consistency (MEDIUM urgency):** Legs_ground drifts from 0.0 to 0.921 (relative drift 307,002%), indicating a major behavioral shift that may reflect unstable or brittle policy mechanics.
- **termination_exploitation (MEDIUM urgency):** Average episode length (479.6) is 24% of maximum (2000), suggesting the agent is exploiting early termination (likely crashes) rather than sustaining landing attempts. At 1M steps the mean length is 981, still below max, so some episodes end prematurely.

### 7. Episode Consistency Summary
Early behavior (200k) shows no ground contact (legs_ground=0), high speed, high altitude. Late behavior (1M) shows consistent leg contact (0.92), low altitude, low speed. The **early_late_consistency_score** is 0.0, indicating a massive drift. This drift is **not healthy adaptation** but rather an unstable shift: the agent moved from “crashing in air” to “touching down off-pad”, but the extreme drift in one metric (legs_ground) combined with no improvement in dist_to_pad suggests the policy may be overfitting to the ground-contact reward and failing to maintain a consistent style.

### 8. Key Numbers for Budget Calculation
| Metric | Value |
|--------|-------|
| mean_length | 981.1 |
| angle_abs | 0.029918 |
| dist_to_pad | 10.25584 |
| height | 2.544242 |
| legs_ground | 0.921007 |
| survival_fraction | 0.0 |
| vert_speed_abs | 0.487129 |