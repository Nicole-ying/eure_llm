## Perception Report: Lunar Lander Training Analysis

### 1. Behavior Trend Summary

At each evaluation milestone, the agent exhibits a clear trajectory of descending from high altitude to a stable low hover, but fails to move horizontally toward the landing pad.

- **200k timesteps:** The agent is at a high altitude (height ≈ 87), with high vertical speed (≈25), far from the pad (dist ≈9.4). Mean episode length is 462.7, suggesting frequent crashes or early termination.
- **400k–800k timesteps:** Height drops progressively (22.7 → 3.4), vertical speed reduces to near zero (≈1.1 → 0.46), and episode length increases to the maximum 1000 (timeout). Dist to pad remains nearly constant around 10. Angle stays low (~0.05–0.06 rad). The agent has learned to descend safely and hover, but does not approach the pad.
- **1M timesteps:** Mean episode length slightly decreases to 963.7 (some episodes terminate early, possibly due to crash or instability). Height is very low (2.84), vertical speed stable at 0.51, angle minimal (0.04 rad). Legs ground contact appears in 25.3% of steps, indicating occasional touchdown or bouncing. Survival fraction is 0.0 – no successful landings.

**Final key numbers:**  
- mean_length: 963.7  
- dist_to_pad: 9.73 (essentially unchanged from early training)  
- height: 2.84  
- vert_speed_abs: 0.51  
- legs_ground: 0.253  
- survival_fraction: 0.0  

**Trend:** Improving in vertical control and survival duration, but **flat** in horizontal positioning – no progress toward the pad.

### 2. Critical Metrics

| Metric | Final Value | Why Important |
|--------|-------------|---------------|
| **dist_to_pad** | 9.73 | Directly measures task success. The agent must reduce this to near zero to land on the pad. It remains high and stationary across all evaluations – a clear failure to address the core objective. |
| **height** | 2.84 | Indicates the agent is very close to the ground. Combined with low vertical speed, shows the agent can achieve a safe low hover. However, height does not drop to zero (landing) and is not paired with horizontal movement. |
| **legs_ground** | 0.253 | Measures contact with ground. At final evaluation, 25% of steps have legs on ground, suggesting partial touchdowns or bouncing. But survival_fraction=0 means these contacts do not result in stable landing. This metric is moving in the right direction (from 0 to 0.25) but lacks consistency with landing success. |

**Cross-metric pattern:** The ratio of **dist_to_pad** to **height** at final evaluation is ~3.4 (9.73 / 2.84). The agent has descended effectively but made no horizontal progress – a strong decoupling. If the agent were genuinely trying to land, both metrics would decrease together. Instead, height dropped dramatically while dist_to_pad stayed constant, indicating a behavior that favors vertical stabilization while ignoring lateral movement.

### 3. Reward Component Health

- **Active components (mean significantly non-zero, std > threshold):**  
  - `r_alive` (mean 0.46, low std) – provides consistent positive reward for staying alive.  
  - `r_proximity` (mean 0.165, low std) – small positive incentive to be near pad.  
  - `r_legs` (mean 0.023, noisier) – weak signal for leg contact.  
  - `r_stability` (mean -0.09, variable) – penalizes instability.  
  - `_outcome` (mean -0.92, moderate std) – terminal outcome reward (negative mean indicates episodes are failing).  
  - `r_crash_penalty` (mean -2.0, std=0) – constant negative penalty per step? Suspiciously constant.  
  - `r_landing_bonus` (mean 20.0, std=0) – constant large positive reward per step? Suspiciously constant.

- **Dominant component:** `r_landing_bonus` (84.5% of total reward magnitude) – its mean is 20, over 10× the next largest component. This component is constant (std=0) and completely overshadows all other reward signals.

- **Inactive/negligible components:** None truly zero, but `r_legs` and `r_stability` have very low contribution (<1%).

- **Suspicious components:** Both `r_landing_bonus` and `r_crash_penalty` have **zero variance** across steps. This is highly abnormal for learned rewards; they appear to be constant offsets applied every step regardless of behavior. `r_landing_bonus` in particular, with a huge positive value, acts as a constant reward that the agent can collect simply by staying alive, with no need to land. This strongly misaligns with the task goal.

- **High mean but weak alignment:** `r_landing_bonus` (mean 20) is earned every step, but the agent never actually lands (survival_fraction=0). The constant reward has no correlation with task progress – it merely incentivizes episode continuation, which the agent has learned to do by hovering safely.

### 4. Behavioral Diagnosis

The agent has learned a stable low-altitude hover far from the landing pad, collecting a large constant reward each step while avoiding crashes. It makes no attempt to move horizontally or land – the dominant constant `r_landing_bonus` effectively makes the pad irrelevant. This is a classic **local optimum** driven by reward structure imbalance: the agent achieves high reward with low effort (hovering safely) instead of the intended high-effort task of pinpoint landing. The strategy is **inefficient** in terms of task progress (high effort in vertical control, zero gain in pad approach). The observed behavior is **inconsistent** with the task goal – the agent should be aiming to reduce `dist_to_pad` and achieve leg contact on the pad, but instead it hovers indefinitely near ground, occasionally touching down anywhere, not on the pad.

### 5. TDRQ Diagnosis

The TDRQ score of 51.97 (mixed) is primarily dragged down by **component balance** (15.49/100) due to the extreme dominance of `r_landing_bonus`. Exploration health is moderate (50/100) – the policy seems stable but stuck. This reward function should be **iterated** to remove or redistribute the constant offset, so that landing becomes the only high-value signal and other components (proximity, stability) can actually influence behavior.

### 6. Constraint Violations Summary

| Principle | Urgency | Evidence |
|-----------|---------|----------|
| **reward_goal_alignment** | High | Two components (`r_landing_bonus`, `r_crash_penalty`) have constant values with zero variance; they provide no informative learning signal and overwhelm the reward structure. The constant +20 per step is directly misaligned with the task. |
| **termination_exploitation** | Medium | Average episode length (478.8 across all episodes) is far below the max observed length (1997), with a utilization ratio of 0.24. The agent may be exploiting early termination (possibly by crashing or timing out earlier than necessary) rather than pursuing long episodes – though recent evaluations show longer episodes, the overall training average is still low. |
| **temporal_consistency** | Medium | The `legs_ground` metric shifted from 0.0 (early) to 0.25 (late), a relative drift >80,000%. This indicates a large behavioral change late in training, which could be either healthy adaptation (learning to touch ground) or unstable policy. Given the constant reward dominance, this drift likely reflects the agent discovering ground contact without landing success. |

### 7. Episode Consistency Summary

Early in training (200k steps), the agent was descending with high vertical speed, no ground contact, and short episodes. Late in training (1M steps), it is consistently hovering at low altitude with occasional leg contact and much longer episodes. This is a **drift** from crash-prone behavior to stable hovering. While the drift shows adaptation (learning to survive), it is **unhealthy** in that the policy has converged to a suboptimal fixed hover far from the pad. The drift is not leading to landing; it is plateauing in a local optimum. The policy style is **consistent** in its disregard for horizontal movement, but inconsistent with the intended task progression.

### 8. Key Numbers for Budget Calculation

| Metric | Value (at 1M timesteps) |
|--------|--------------------------|
| mean_length | 963.7 |
| angle_abs | 0.040352 |
| dist_to_pad | 9.728356 |
| height | 2.839274 |
| legs_ground | 0.253087 |
| survival_fraction | 0.0 |
| vert_speed_abs | 0.50885 |