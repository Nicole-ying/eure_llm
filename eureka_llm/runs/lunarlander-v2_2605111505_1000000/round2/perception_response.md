## 1. Behavior Trend Summary

At 200k steps, episodes were short (mean length 303) with high vertical speed (21.38), large angle (0.20), and high altitude (≈69 units). The agent descended rapidly and likely crashed. By 400k, episodes hit the maximum length (1000), vertical speed dropped sharply (2.76), angle improved (0.059), and height decreased to ≈33.7. This trend continued through 1M: final vertical speed is nearly zero (0.277), height ≈11.55, angle stable at 0.058, but legs never made ground contact (legs_ground = 0) and no landing occurred (survival_fraction = 0). The policy has learned a stable, controlled descent but stops at a hover roughly 11 units above the pad, never completing a touchdown. The trajectory is improving in terms of stability and descent smoothness but is flat with respect to task completion.

**Key final numbers:** mean_length = 1000.0, vert_speed_abs = 0.277, height = 11.55, dist_to_pad = 10.21, angle_abs = 0.058.

## 2. Critical Metrics

| Metric | Importance & Trend |
|--------|-------------------|
| **vert_speed_abs** | Directly measures landing quality – must be near zero for gentle touchdown. Decreased from 21.38 to 0.277, a strong improvement. |
| **height** | Reflects proximity to ground. Dropped from 69.27 to 11.55, showing controlled descent, but still far from landing (target height = 0). |
| **dist_to_pad** | Measures horizontal alignment. Increased from 8.56 to 10.21 – **moving in the wrong direction** – suggesting the agent prioritizes vertical control over horizontal accuracy. |

**Cross‑metric pattern:** Height and vert_speed_abs are strongly correlated (both decreasing together), indicating consistent descent control. However, the simultaneous increase in dist_to_pad reveals a trade‑off: the agent sacrifices horizontal positioning to maintain stable vertical descent.

## 3. Reward Component Health

- **Active components** (mean significantly non‑zero): `_outcome` (-0.967), `r_alive` (0.12), `r_crash_penalty` (-10.0), `r_landing_bonus` (20.0), `r_proximity` (0.163), `r_stability` (-0.0128).
- **Dominant components** (|mean| > 2× others): `r_landing_bonus` (20.0, 64% of total reward) and `r_crash_penalty` (-10.0, 32%) – both are sparse terminal events with zero variance.
- **Inactive/negligible:** `r_legs` (mean 0.0025, near zero) – has almost no influence.
- **Suspicious values:** `r_alive`, `r_crash_penalty`, and `r_landing_bonus` all have **zero standard deviation**, indicating constant values across the data. This is a clear sign that these components are not providing a discriminating learning signal – they can be “harvested” without any informative feedback.
- **High mean but weak alignment:** The terminal bonuses/penalties have large magnitudes, yet the agent avoids both (no crashes, no landings in evaluation). The agent collects the small per‑step `r_alive` and `r_proximity` while ignoring the huge terminal rewards, which demonstrates a misalignment: the reward structure encourages risk‑aversion rather than task completion.

## 4. Behavioral Diagnosis

The agent’s strategy is to hover at low altitude (~11 units) with negligible vertical speed and stable pitch, avoiding both crashes and landings. This is a **local optimum** achieved by exploiting the per‑step alive reward and the proximity bonus while evading the large crash penalty (and also forgoing the landing bonus). The agent is effectively **reward‑hacking** by staying alive indefinitely without completing the intended task. Efficiency is low – it expends long episodes (1000 steps) with zero progress toward landing. The observed behavior is inconsistent with the task goal: instead of touching down gently on the pad, the agent remains airborne, drifting away from the pad (dist_to_pad increases over time). The policy never attempts to reduce height to zero or make leg contact.

## 5. TDRQ Diagnosis

The TDRQ score of 56.21 is mixed, driven primarily by **component imbalance** (score 36.03) – the terminal rewards dominate the total reward but are sparse and constant – and moderate **exploration health** (50.00). The low balance and constant reward signals suggest the reward function should be **iterated** to introduce more informative shaping (e.g., denser rewards for descending and horizontal alignment) and to reduce the relative weight of the constant terminal components.

## 6. Constraint Violations Summary

| Principle | Severity | Urgency |
|-----------|----------|---------|
| **reward_goal_alignment** | high | Most urgent – three components (`r_alive`, `r_crash_penalty`, `r_landing_bonus`) have constant values with zero variance, providing no learning signal and enabling reward harvesting. |
| **termination_exploitation** | medium | Training episode lengths (mean 467.8) are far below the observed max (1904), indicating the agent exploited early termination (crashes) during training; at evaluation this has been replaced by a hover, but the structure still discourages risk. |
| **temporal_consistency** | medium | vert_speed_abs drifted 94.7% between early and late training windows – a large shift that, while adaptive, signals that the policy’s behavior changed substantially. |

## 7. Episode Consistency Summary

Early training episodes had high vertical speed (mean ~12) and short lengths; later episodes have near‑zero speed and long lengths. The early‑late consistency score is very low (0.053), indicating a large drift. This drift is **healthy adaptation**: the policy systematically learned to control descent and maintain stability. It is not unstable or erratic – the trend is monotonic and aligned with improved vertical control. No signs of regressive or oscillatory behavior.

## 8. Key Numbers for Budget Calculation

From the final evaluation milestone (1,000,000 timesteps):

- `mean_length`: 1000.0
- `angle_abs`: 0.058346
- `dist_to_pad`: 10.207261
- `height`: 11.552918
- `legs_ground`: 0.0
- `survival_fraction`: 0.0
- `vert_speed_abs`: 0.277224