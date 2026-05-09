## Perception Report

### 1. Behavior Trend Summary

- **200k steps:** Agent barely survives (mean episode length 67.9). Falls rapidly (vertical speed −6.97, high horizontal speed 2.35), with significant tilt (angle −0.3988) and angular velocity. No sustained flight.
- **400k steps:** Significant improvement. Mean length jumps to 498.6. Agent stabilises near horizontal (angle 0.0155), low angular velocity, moderate vertical speed (−0.89) and horizontal speed (1.43). Begins to hover.
- **600k steps:** Peak performance. Agent survives full evaluation horizon (mean length 1000). Almost zero tilt, tiny angular velocity, small distance to pad (0.1968), low horizontal speed (0.422), low descent rate (vertical speed −0.470). Leg contact (0.244) suggests occasional ground touch, but controlled.
- **800k steps:** Regression begins. Mean length drops to 859.6. Metrics similar but leg contact doubles (0.480), indicating more ground contact (possibly landing/crashing earlier). Distance to pad remains low (0.192).
- **1M steps:** Further decline. Mean length falls to 395.5. Distance to pad rises to 0.307, vertical speed becomes more negative (−1.19), leg contact stays high (0.462). Agent has lost the ability to maintain stable hover; episodes end prematurely.

**Overall trajectory:** Improving up to 600k, then regressing. Final behaviour is short, unstable flight with frequent ground contact and increasing descent.

**Key final numbers (1M steps):**
- mean_length = 395.5
- angle = −0.0157
- angular_velocity = −0.00069
- distance_to_pad = 0.307
- horizontal_speed = 0.391
- leg_contact = 0.462
- vertical_speed = −1.190

### 2. Critical Metrics

| Metric | Trend | Flag |
|--------|-------|------|
| **distance_to_pad** | Decreased from 0.86 to 0.20 (good), then increased to 0.307 (bad) | ⚠️ Moving in wrong direction after 600k |
| **vertical_speed** | Improved from −6.97 to −0.47 (good), then worsened to −1.19 (bad) | ⚠️ Moving in wrong direction after 600k |
| **leg_contact** | Increased from 0.04 to 0.48 (bad – more ground contact) | ⚠️ Consistently increasing, indicates loss of air time |

Also notable: horizontal_speed declined then slightly rose; angle and angular_velocity remain stable near zero.

### 3. Reward Component Health

| Component | Mean | Std | Active? | Dominant? | Inactive? |
|-----------|------|-----|---------|-----------|-----------|
| _outcome | −0.2113 | 0.3277 | ✅ Yes (45.8% of total) | No (not >2× others) | – |
| r_alive | +0.1508 | 0.1041 | ✅ Yes (32.7%) | No | – |
| r_distance | −0.0822 | 0.0267 | ✅ Yes (17.8%) | No | – |
| r_leg_contact | +0.0133 | 0.0263 | Borderline (2.9%) | No | No |
| r_vertical_speed | −0.0038 | 0.0032 | ❌ No (0.8%, mean ≈ std) | No | ✅ Inactive (stable) |

- **Active components:** _outcome, r_alive, r_distance (all with non‑zero means and reasonable std). r_leg_contact is small but not negligible; r_vertical_speed is negligible.
- **No dominant component** – contributions are spread.
- **Suspicious means:** None exactly zero. The r_vertical_speed mean is very small, but not suspicious given its scale.

### 4. Behavioral Diagnosis

The agent initially learned to hover stably near the pad (low distance, low vertical speed), but after 600k steps it began to regress: episodes shortened, distance increased, vertical speed became more negative, and leg contact rose. This suggests the agent is stuck in a **local optimum of short‑duration hovering** that occasionally makes ground contact; it is not reward‑hacking in a degenerate way, but its policy has lost the ability to maintain sustained flight, possibly due to exploitation of a narrow solution that fails under exploration drift or insufficient entropy.

### 5. TDRQ Diagnosis

Overall TDRQ of 62.39 is **mixed** (borderline healthy). Component balance (54.19) is moderate – no single component dominates, but _outcome and r_alive together account for ~78% of reward magnitude, creating imbalance. Exploration health (50) is low, indicating the policy has collapsed its search space (likely accounting for the regression after 600k). The main weakness is **insufficient exploration** rather than component inactivity. The reward design should be **iterated** to encourage consistent sustained flight (e.g., by strengthening vertical speed or distance penalties) and to maintain higher policy entropy.

### 6. Key Numbers for Budget Calculation

From the final evaluation milestone (1M timesteps):

| Metric | Value |
|--------|-------|
| mean_length | 395.5 |
| angle | −0.015651 |
| angular_velocity | −0.000691 |
| distance_to_pad | 0.306692 |
| horizontal_speed | 0.391099 |
| leg_contact | 0.462200 |
| vertical_speed | −1.190069 |