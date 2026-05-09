# Agent Behavior Report

## 1. Behavior Trend Summary

- **At 200k timesteps:** The agent completes very short episodes (mean length 70). It is highly unstable: large negative angle (-0.296 rad) and angular velocity (-0.571), with high horizontal speed (1.62) and strong downward vertical speed (-6.75). It is essentially tumbling or falling immediately. Leg contact is minimal (0.019), meaning it rarely touches the ground in a controlled stance.
- **At 400k:** Episodes become much longer (mean 435.7). The agent stabilizes orientation (angle –0.007, angular velocity –0.02), reduces horizontal speed to 0.94, and vertical speed drops to –0.99 (still descending but slowly). Distance to pad is ~0.45. The agent appears to be learning to balance and float or glide toward the pad while controlling its descent.
- **At 600k:** Peak mean episode length (838.2). Orientation is nearly perfect (angle –0.002, angular velocity –0.003). Horizontal speed further reduces to 0.60, vertical speed to –0.54. Distance to pad is 0.41. Leg contact remains low (0.028). The agent is surviving for long durations, moving slowly toward the pad, but still not making solid ground contact.
- **At 800k:** Episode length drops to 637.0. Angle becomes slightly positive (0.011) and angular velocity positive (0.001). Horizontal speed decreases to 0.53, vertical speed becomes more negative (–0.72). Distance to pad increases to 0.50. The agent is regressing: it moves away from the pad and descends faster.
- **At 1M timesteps (final):** Episode length drops further to 486.1. Orientation remains upright (angle 0.012, angular velocity 0.0004). Horizontal speed is lowest at 0.47, distance to pad is 0.33 (closer than before), but vertical speed is –0.97 (faster descent). Leg contact rises to 0.079 (still very low). The agent is surviving moderately long episodes but descending more quickly, suggesting it is falling out of controlled flight or losing altitude.

**Overall trajectory:** Performance improved sharply up to 600k, then declined. The agent is not regressing to random behavior but is settling into a shorter-lived, faster-descending strategy. It is **not making sustained progress** toward the task (likely reaching a pad while maintaining contact). Final key numbers: mean_length = 486.1, distance_to_pad = 0.335, vertical_speed = –0.973, leg_contact = 0.079.

## 2. Critical Metrics

- **mean_length** – peaked at 600k (838.2), then declined to 486.1. **Trend: regressing** (worsening).
- **distance_to_pad** – fluctuated (0.95 → 0.45 → 0.41 → 0.50 → 0.33). Final value is the lowest (best), but the trend is non-monotonic; last drop may be due to faster descent rather than intentional approach.
- **vertical_speed** – improved from –6.75 to –0.54 at 600k, then worsened to –0.97 at 1M. **Trend: going in the wrong direction** (more negative = faster descent, likely indicating loss of control or falling).

Other metrics: horizontal_speed decreased steadily (good for stability but may indicate stagnation); leg_contact remains very low (<8%) throughout, never indicating sustained ground contact – a key task requirement is likely missed.

## 3. Reward Component Health

- **Active components (mean significantly non-zero):** All five components have non-zero means.
  - `_outcome` (mean –0.798, 72.6% of total) – major active component, negative.
  - `r_alive` (mean +0.142) – positive, encourages survival.
  - `r_distance` (mean –0.084) – negative, penalizes something (maybe distance from pad?).
  - `r_leg_contact` (mean +0.038) – positive but noisy.
  - `r_vertical_speed` (mean –0.037) – negative, penalizes downward speed.
- **Dominant components:** None exceed 80% of total. `_outcome` is the largest (72.6%), but not dominant by the strict >80% threshold.
- **Inactive/negligible:** None. All components are active.
- **Suspicious values:** No means are exactly zero. The `r_leg_contact` has high noise (std/mean ratio >3), suggesting inconsistent foot contact behavior. The `_outcome` component has large std relative to mean, indicating highly variable terminal rewards (likely frequent failures).

## 4. Behavioral Diagnosis

The agent's current strategy is to maintain an upright orientation while slowly drifting toward the pad, but it is descending at an increasing rate and rarely establishing ground contact. It appears to be **stuck in a local optimum** where it survives for moderately long episodes by floating or falling gently, but never successfully completing the task (likely landing on the pad with proper foot contact). The large negative `_outcome` reward suggests frequent terminations due to failure (e.g., falling below a height threshold or exceeding time limit). The agent is not reward hacking in the usual sense, but it is avoiding stronger penalties (distance, vertical speed) at the cost of missing the primary goal.

## 5. TDRQ Diagnosis

The overall TDRQ of 57.32 is in the mixed range (40–70). The low score is primarily driven by **component imbalance** (27.37/100) – the `_outcome` component dominates in magnitude (72.6%) while `r_vertical_speed` and `r_leg_contact` contribute little. Exploration health is moderate (50.00), likely reflecting the lack of entropy data. The reward is **unbalanced** but not collapsed. This reward should be **iterated** to re-scale components so that the negative outcome penalty does not overwhelm other signals, and to make foot contact and vertical speed more influential. Diversification (population-based search) could also help escape the local optimum.

## 6. Key Numbers for Budget Calculation

| Metric | Final Value (at 1M timesteps) |
|--------|-------------------------------|
| mean_length | 486.1 |
| angle | 0.01179 rad (≈0.68°) |
| angular_velocity | 0.000425 rad/s |
| distance_to_pad | 0.3348 (units depend on env) |
| horizontal_speed | 0.4729 units/s |
| leg_contact | 0.0786 (fraction of steps in contact) |
| vertical_speed | -0.9727 units/s |

These numbers reflect the agent’s state at the last evaluation point and should be used for reward budget recalibration.