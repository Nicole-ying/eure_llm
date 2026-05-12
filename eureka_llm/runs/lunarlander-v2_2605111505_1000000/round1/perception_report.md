## Perception Report – Lunar Lander Training

### 1. Behavior Trend Summary

* **Timestep 200k**: The policy flies at high altitude (height ~43.5), with large vertical speed (~16.0) and moderate angle (~0.31 rad). It never contacts the ground (legs_ground=0). Mean episode length is short (279.6), suggesting early crashes or terminations.
* **Timestep 400k**: The policy reduces vertical speed to ~5.0 and angle to ~0.04 rad, but height remains high (~46.0). Still no leg contact. Episode length increases to 757.6, indicating longer flights.
* **Timestep 600k**: A dramatic drop in height to ~2.8 with very low vertical speed (~0.52). Legs_ground remains 0, implying the agent hovers just above ground but does not contact. Episode length reaches the max (1000), suggesting the agent can stay airborne for the entire horizon.
* **Timestep 800k**: Height stabilises at ~2.4, legs_ground rises to 0.73 (some episodes make ground contact), and vertical speed increases slightly to 0.60. Episode length drops to 800.8.
* **Timestep 1M**: Legs_ground reaches 1.01 (both legs contact ground on average), height ~2.15, vertical speed ~0.71. **Crucially, distance to pad has increased** from ~8.2 (at 200k) to ~10.5, and survival_fraction remains 0.0 – no episode ends safely.

**Trend**: The policy improves at descending, stabilising angle, and achieving leg contact, but **fails to move horizontally toward the landing pad** and does not achieve a safe touchdown (vertical speed is still high enough to cause crashes). The trajectory is **improving in ground contact but regressing in pad alignment**.

**Key final numbers**:  
`mean_length=674.1`, `angle_abs=0.0325`, `dist_to_pad=10.48`, `height=2.15`, `legs_ground=1.01`, `survival_fraction=0.0`, `vert_speed_abs=0.71`.

---

### 2. Critical Metrics

| Metric | Importance | Final Value | Direction |
|--------|------------|-------------|-----------|
| **legs_ground** | Directly measures physical contact with ground – a necessary condition for landing. | 1.01 | Improving (0 → 1) |
| **vert_speed_abs** | Determines whether ground contact is safe. Must be low (<0.5 for safe landing). | 0.71 | **Wrong direction** – increased after 600k drop |
| **dist_to_pad** | Measures horizontal alignment with the target. Essential for task completion. | 10.48 | **Wrong direction** – steadily increasing from 8.2 |
| **height** | Indicates how low the agent is; necessary for landing but not sufficient. | 2.15 | Improving (dropped from 43 to 2.2) |

**Cross-metric pattern**:  
`height` and `legs_ground` are tightly coupled – as height drops, legs_ground rises, showing consistent descending behavior. However, `dist_to_pad` and `vert_speed_abs` move independently in the wrong direction, indicating the agent prioritises downwards motion over horizontal positioning and speed control.

**Most important**: `legs_ground` (progress toward ground contact) and `dist_to_pad` (alignment with goal) are the two axes that directly reflect task success.

---

### 3. Reward Component Health

**Active components** (mean significantly non-zero):  
- `r_crash_penalty`: mean = -10.0, std = 0.0 → **dominant** (32.5% of total)  
- `r_landing_bonus`: mean = +20.0, std = 0.0 → **dominant** (65.0% of total)  
- `r_alive`: mean = 0.01 (minor but constant)  
- `r_outcome`: mean = -0.718, std = 0.696 (variable, 2.3% of total)  
- `r_proximity`: mean = 0.0165 (0.1% of total)

**Inactive/negligible components**:  
- `r_legs`: mean = 0.00565, tiny std → provides almost no gradient  
- `r_stability`: mean = -0.00487, tiny std → negligible

**Dominant components**: `r_crash_penalty` and `r_landing_bonus` together account for **97.5%** of total reward magnitude. Both have zero variance, meaning they are **constant per step**. This is a strong indicator of reward design that does not depend on agent behaviour – the agent can “harvest” these offsets without learning anything about the task.

**Suspicious mean**: Both constant components have means that are far from zero and non-informative. The learning signal from `r_outcome` (sparse, negative) is dwarfed. The agent is **not sensitive to task progress** – the constant rewards mask any signal from pad alignment or safe landing.

---

### 4. Behavioral Diagnosis

The agent’s current strategy is to **descend vertically to the ground, achieving leg contact, while ignoring the landing pad location and failing to reduce vertical speed below a safe threshold.** It does not attempt horizontal movement. This is a **local optimum driven by reward hacking** – the constant per-step rewards (+10 net per step from bonus + penalty) dominate, so the agent maximises episode length by staying alive (which it does by descending slowly) but never moves toward the pad because no reward component incentivises horizontal alignment. The agent uses **low effort** (simply descending) for **high gain** (constant positive reward) but completely misses the intended goal of landing on the pad. This pattern is **inconsistent with the task description**: the agent should control horizontal position and achieve a gentle touchdown near the flags, but instead it lands far away and likely crashes (survival_fraction=0). The policy has drifted into a behavior that satisfies only the ground-contact condition, not the safe landing condition.

---

### 5. TDRQ Diagnosis

TDRQ overall = 45.74/100 (mixed/unhealthy). The low score is mainly due to **component imbalance** (97.5% of reward from two constant offsets) and **component inactivity** (task-relevant components like `r_legs` and `r_proximity` are negligible). Exploration health is moderate (50.0). This reward design should be **iterated** – the constant offsets must be removed or redesigned to provide informative, contingent signals that align with the task goal (pad proximity, safe speed, angle). Without change, the agent will remain stuck.

---

### 6. Constraint Violations Summary

| Principle | Severity | Urgency | Evidence |
|-----------|----------|---------|----------|
| **reward_goal_alignment** | High | **Critical** | `r_crash_penalty` and `r_landing_bonus` are constant offsets (mean non-zero, zero variance) that can be harvested without learning; they dominate total reward. |
| **termination_exploitation** | Medium | Moderate | Mean episode length (310.5) is only 16% of observed max (1938), suggesting early termination is exploited – possibly to avoid a negative outcome after ground contact. |
| **temporal_consistency** | Medium | Moderate | `legs_ground` drifted from 0.0 to 1.0 (relative drift >500k%), indicating a major shift in policy behaviour between early and late training. |

**Ranked by urgency**: Alignment > Consistency ≈ Exploitation.

---

### 7. Episode Consistency Summary

**Early vs late behaviour**:  
- Early episodes (200k-400k): high altitude, no leg contact, high vertical speed, high episode length variability.  
- Late episodes (800k-1M): low altitude, consistent leg contact, low but slightly increasing vertical speed, shorter episodes on average.  

The policy style has **drifted substantially**: from a flying/falling behavior to a ground-contact behavior. This drift appears **adaptive** in the sense that the agent learns to descend and make leg contact, which is a subgoal of the task. However, the drift is **unstable** in that it ignored pad alignment – a critical component of the task. The agent has found a local optimum that partially improves (ground contact) but fails to address the full objective.

---

### 8. Key Numbers for Budget Calculation

| Metric | Final Value |
|--------|-------------|
| `mean_length` | 674.1 |
| `angle_abs` | 0.032485 |
| `dist_to_pad` | 10.483824 |
| `height` | 2.150613 |
| `legs_ground` | 1.012906 |
| `survival_fraction` | 0.0 |
| `vert_speed_abs` | 0.713359 |