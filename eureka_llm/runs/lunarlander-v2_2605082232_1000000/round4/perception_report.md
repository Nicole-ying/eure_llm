### 1. Behavior Trend Summary

- **200k**: Agent starts poorly – mean episode length only 84.7 steps, with high vertical speed (−5.66) and high horizontal speed (1.33). It falls quickly and moves erratically.
- **400k**: Significant improvement – mean length jumps to 710.6, vertical speed drops to −0.49, horizontal speed reduces to 0.69, distance to pad decreases to 0.54. Agent learns to slow descent and move less.
- **600k**: Slight regression – mean length drops to 511.0, vertical speed worsens to −0.77, horizontal speed increases slightly (0.87). Leg contact becomes very low (0.009). Agent may be losing balance control.
- **800k**: Best survival – mean length peaks at 928.1, vertical speed −0.51, horizontal speed lowest (0.38), distance to pad smallest (0.22). Leg contact spikes to 0.41, indicating active stepping.
- **1M**: Sharp drop – mean length falls to 540.3, leg contact collapses to 0.08, vertical speed increases to −0.87, distance to pad rises to 0.30. Behavior regresses from the 800k peak, possibly due to policy instability or overfitting.

**Trend**: Not monotonic – improves then degrades. The agent learns to survive longer by reducing speed and staying near the pad, but consistency weakens. Final performance (540 steps) is moderate, not a robust solution.

**Key numbers at 1M**: mean_length = 540.3, distance_to_pad = 0.30, horizontal_speed = 0.54, vertical_speed = −0.87, leg_contact = 0.08.

---

### 2. Critical Metrics

| Metric | Trend | Direction |
|--------|-------|-----------|
| **distance_to_pad** | Decreasing from 0.80 to 0.30 over training | ✅ Good (agent gets closer to pad) |
| **horizontal_speed** | Decreasing from 1.33 to 0.54 | ⚠️ Mixed – less movement may reduce exploration, but might stabilize |
| **vertical_speed** | Improves from −5.66 to −0.87 (less negative) | ✅ Good (agent falls slower) |

**Flag**: **leg_contact** – spikes at 800k (0.41) but collapses to 0.08 at 1M. This is a wrong-direction regression; the agent stops using contact effectively.

---

### 3. Reward Component Health

- **Active components** (mean significantly non‑zero): All five components – `_outcome` (−0.34), `r_alive` (0.14), `r_distance` (−0.084), `r_leg_contact` (0.039), `r_vertical_speed` (−0.072).
- **Dominant components** (|mean| > 2× others): None dominates strictly, but `_outcome` has the largest absolute mean (0.34) – it contributes 50.3% of total reward magnitude.
- **Inactive / negligible**: None. All components have non‑zero means and non‑zero stds.
- **Suspicious values**: No component mean is exactly zero. `r_leg_contact` has high noise (std 0.145 vs mean 0.039, CV > 3.7) – noisy but not inactive.

---

### 4. Behavioral Diagnosis

The agent's strategy appears to be: **slow down, hover near the pad, and try to avoid falling** – but it does not consistently maintain foot contact or actively step to stay upright. The negative `_outcome` (−0.34 per step) indicates the agent frequently fails (likely crashes), and the leg‑contact collapse at the end suggests it stops trying to use its legs properly. It is stuck in a local optimum where it reduces risk by moving slowly and staying low, but fails to develop a robust balancing gait.

---

### 5. TDRQ Diagnosis

The overall TDRQ of 67.34 / 100 is in the “mixed” range. The main weakness is **component balance** (49.65), because `_outcome` carries half the reward weight and is highly variable – this signals that the agent is driven mostly by survival/failure events, not shaped feedback. Exploration health (50.00) is also moderate, likely due to insufficient entropy or diversity. **This reward function should be iterated** – rebalancing by reducing the dominance of the terminal outcome and adding more shaping (e.g., smoother foot‑contact incentives) would likely improve learning stability.

---

### 6. Key Numbers for Budget Calculation

| Metric | Value |
|--------|-------|
| mean_length | 540.3 |
| angle | −0.003937 |
| angular_velocity | 0.000394 |
| distance_to_pad | 0.301905 |
| horizontal_speed | 0.543594 |
| leg_contact | 0.077179 |
| vertical_speed | −0.870713 |

(All from the 1,000,000 timestep row of the Evaluation Metrics table)