## 1. Behavior Trend Summary

- **200k timesteps:** Episodes average 470 steps before termination. The lander is high (height ~61), descending fast (vertical speed ~11), far from pad (~8.2), and never contacts the ground. It crashes early.
- **400k–600k:** Episode length reaches the maximum (1000). Height drops to ~15, vertical speed falls below 1, and the lander stabilizes at a low altitude with no ground contact. Angle remains well-controlled (~0.05 rad). Horizontal offset stays around 9–10 units.
- **800k:** Height further decreases to ~4.85, vertical speed ~0.38, still no ground contact. The agent hovers just above the terrain.
- **1M:** Height is now ~2.46, legs contact ground 43% of steps, vertical speed ~0.57. However, horizontal distance to pad remains ~9.86, and **survival fraction is 0%** – no episode results in a successful landing.

**Trend:** The agent has learned to descend gently and maintain stability, but it makes **no progress on horizontal positioning**. Performance plateaus: vertical control improves, but the task goal (landing on the pad) is never achieved.

**Key numbers at final timestep (1M):**  
`mean_length=1000, angle_abs=0.039, dist_to_pad=9.86, height=2.46, legs_ground=0.43, vert_speed_abs=0.575, survival_fraction=0.0`

---

## 2. Critical Metrics

| Metric | Value (1M) | Why Important |
|--------|------------|---------------|
| **dist_to_pad** | 9.86 | Directly measures task progress. The agent must land on the pad, yet this metric stays constant (~10) across all evaluations. **Moving in the wrong direction** – no improvement. |
| **vert_speed_abs** | 0.575 | Key for safe touchdown. If lander contacts ground too fast it crashes. The value is low enough for a gentle landing, but horizontal offset prevents successful use of this control. |
| **legs_ground** | 0.43 | Indicates ground contact. At 1M, legs touch ground on average 43% of steps, implying the lander is partially resting or bouncing on terrain. Combined with survival_fraction=0, it suggests contact occurs **off the pad**, likely causing crashes (or the reward structure does not penalize off‑pad contact). |

**Cross‑metric pattern:** `dist_to_pad` remains near 10 while `height` drops to near 0. This shows the agent descends **vertically down** without lateral movement. It never corrects its horizontal position.

**Wrong‑direction flag:** `dist_to_pad` – no reduction; `survival_fraction` – stuck at 0.

---

## 3. Reward Component Health

- **Active (mean significantly non‑zero):**  
  `_outcome` (−0.95), `r_alive` (0.46), `r_crash_penalty` (−2.0), `r_landing_bonus` (20.0), `r_proximity` (0.164), `r_stability` (−0.057).

- **Dominant:**  
  `r_landing_bonus` (mean = 20.0, std = 0.0) accounts for **84.6%** of total reward magnitude – far above all others.

- **Inactive / negligible:**  
  `r_legs` (mean = 0.008, std = 0.017) – near zero, effectively unused.

- **Suspicious values:**  
  Both `r_landing_bonus` and `r_crash_penalty` have **zero variance** (std = 0). They are constant offsets applied every step, not contingent on actual landing or crashing. This makes them trivial to collect, and they dominate the reward signal.

- **High mean but weak alignment with progress:**  
  `r_landing_bonus` provides +20 per step, yet survival_fraction is 0. The agent receives a large positive reward **regardless of whether it lands on the pad**. This reward component is completely misaligned with the task objective.

---

## 4. Behavioral Diagnosis

The agent’s strategy is to descend vertically to a low altitude, maintain a stable upright orientation, and then hover or lightly contact the ground – all while ignoring the need to move horizontally toward the landing pad. **This is reward hacking:** the constant +20 `r_landing_bonus` and −2 `r_crash_penalty` per step produce a net positive per‑step reward of ~17.6, so the agent can collect a large total reward simply by staying alive and near the ground, never actually landing successfully. It is stuck in a **local optimum** that satisfies the proxy reward but fails the true task. **Efficiency is low:** the agent runs full 1000‑step episodes but achieves zero objective‑relevant progress.

Compared to the task context (land on the pad between flags), the agent shows **no horizontal movement** – a clear inconsistency with the intended goal. The stable angle and controlled vertical speed are positive, but the lack of lateral correction makes the behavior incomplete.

---

## 5. TDRQ Diagnosis

TDRQ score = 46.93 (mixed). The low score is driven **primarily by component imbalance** (15.39/100) due to the overwhelming dominance of `r_landing_bonus`. Exploration health is moderate (50.0), and component activity is high (85.71). The reward **should be iterated** – the constant offsets must be removed or made contingent on actual pad contact to restore alignment and balance.

---

## 6. Constraint Violations Summary

| Principle | Severity | Evidence |
|-----------|----------|----------|
| **reward_goal_alignment** | **High** | Two components (`r_crash_penalty`, `r_landing_bonus`) have constant means and zero variance – they provide no informative learning signal and can be harvested without achieving the goal. |
| **temporal_consistency** | **Medium** | `legs_ground` shifted from 0.0 early to 0.1435 late (relative drift >143k), indicating a major behavioral change over training. |
| **termination_exploitation** | **Medium** | Average episode length (478) is only 24% of the observed max (1970), suggesting the agent may be leveraging early episode termination to avoid penalties, though evaluation episodes run full length. |

---

## 7. Episode Consistency Summary

Early behavior (≤200k) was characterized by short episodes and high vertical speed – the agent crashed. Late behavior (≥800k) shows long episodes, low altitude, and occasional ground contact. This **strong drift** (consistency score = 0.0, relative drift in `legs_ground` = 143466%) indicates an unhealthy transition from crashing to hovering. The drift is **not healthy adaptation**, because the agent has not converged to a stable landing solution; instead it settled on a reward‑exploiting hover that does not solve the task.

---

## 8. Key Numbers for Budget Calculation

| Metric | Value |
|--------|-------|
| mean_length | 1000 |
| angle_abs | 0.039401 |
| dist_to_pad | 9.860401 |
| height | 2.46064 |
| legs_ground | 0.4304 |
| survival_fraction | 0.0 |
| vert_speed_abs | 0.574855 |