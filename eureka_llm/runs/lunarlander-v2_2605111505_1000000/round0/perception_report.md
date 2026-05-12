# Perception Report: Lunar Lander Training Analysis

## 1. Behavior Trend Summary

At each evaluation milestone, the policy consistently keeps the lander aloft for about 67–72 steps before termination. The agent maintains a small absolute angle (0.01–0.08 rad), suggesting strong attitude control, but does not descend toward the pad. **dist_to_pad** remains near 9.1–10.0, **height** around 7.0–7.2, and **vert_speed_abs** high (6.65–7.07), indicating the lander never significantly reduces altitude or aligns horizontally with the pad. The **survival_fraction** is 0.0 at all checkpoints, meaning **no episode results in a successful landing**. The trajectory is flat and non-improving across all task-level metrics; the policy is stuck in a local optimum where it avoids immediate crash but fails to land.

**Final timestep (1,000,000) key metrics:**
- mean_length: 71.9
- angle_abs: 0.018118
- dist_to_pad: 9.115005
- height: 7.162327
- legs_ground: 0.019471
- survival_fraction: 0.0
- vert_speed_abs: 6.657329

## 2. Critical Metrics

| Metric | Value (final) | Why Important |
|--------|---------------|---------------|
| **survival_fraction** | 0.0 | Direct measure of task success; persistent zero indicates complete failure to land. |
| **dist_to_pad** | 9.115 | Horizontal error relative to target; remains high and unchanging, showing no approach. |
| **vert_speed_abs** | 6.657 | High vertical speed at termination implies harsh impact or crash; should be near 0 for safe landing. |

**Cross-metric pattern:** The combination of **height ≈ 7.2** and **vert_speed_abs ≈ 6.7** suggests the lander starts at high altitude (typical initial height ~10–12 m) and falls nearly freely without powered descent, maintaining high speed until episode ends. The **angle_abs** is very low (0.018), indicating the agent successfully stabilizes pitch but neglects vertical control.

**Flagged wrong direction:** None of the metrics are moving in a positive direction; they are essentially flat with minor fluctuations. Survival fraction remains zero, a critical failure.

## 3. Reward Component Health

- **Active components (mean significantly non-zero):**
  - `_outcome` (mean –4.47, 96.1% of total magnitude) – negative, noisy, dominant.
  - `r_stability` (mean –0.16, 3.5% of total) – negative, stable but small.

- **Dominant component:** `_outcome` – its mean magnitude is >26× the next largest (r_stability) and accounts for 96.1% of total reward. It is **noisy** (std ~2× mean CV>1.5).

- **Inactive/negligible components:**
  - `r_alive` (mean 0.01, std 0.0) – constant tiny positive; essentially a fixed bias.
  - `r_legs` (mean 0.005, std 0.002) – near-zero; leg contact almost never occurs.

- **Suspicious mean values:** `_outcome` is persistently negative and dominant. Its value reflects terminal penalties (e.g., crash or timeout), yet the agent never achieves positive outcomes. The component’s high mean magnitude, combined with its noise, likely drives the policy to avoid the worst terminal states (by delaying crash) rather than achieving the goal. **High mean but weak alignment with progress:** `_outcome` does not reward descent or proximity; it only penalizes failure. The agent exploits this by staying alive for ~70 steps without making progress.

## 4. Behavioral Diagnosis

The agent’s strategy is to **stabilize its attitude (low angle) and avoid immediate catastrophic crash, but it never actively descends toward the landing pad**. It likely hovers or drifts horizontally while maintaining altitude, then eventually crashes due to high vertical speed after the episode length limit (or intentional early termination). This is **inefficient** — high effort in thruster control for stability yields zero successful landings. The behavior is inconsistent with the task goal: the agent should approach the pad with decreasing altitude and low vertical speed, but instead it maintains nearly constant height and large horizontal offset. The low `legs_ground` and zero `survival_fraction` confirm that no gentle touchdown occurs.

## 5. TDRQ Diagnosis

TDRQ of 29.24 / 100 is unhealthy, primarily due to **extreme component imbalance** (balance score 3.86) and partial **component inactivity** (activity score 50.0). The `_outcome` reward dominates, drowning out other signals, while `r_legs` and `r_alive` are nearly constant. Exploration health is moderate (50.0), suggesting the policy may have collapsed to a narrow behavior. The reward function should be **diversified** — either through search (e.g., population-based training) or by redesigning components to actively shape descent and landing, not just penalize failure.

## 6. Constraint Violations Summary

| Principle | Severity | Urgency | Evidence |
|-----------|----------|---------|----------|
| **termination_exploitation** | medium | **High** | Mean episode length (71.5) is only 7.1% of the maximum (1000). The agent deliberately ends episodes early (likely via crash or timeout) instead of attempting a full landing. |
| **state_coverage** | medium | Medium | Episode lengths are concentrated in a narrow window (48–72); the policy is locked into a repetitive local optimum with no variation. |
| **temporal_consistency** | medium | Low | Angle_abs drifted from early mean 0.055 to late mean 0.016 (relative drift 70%). While angle improved, this may reflect a narrowing of behavior rather than healthy adaptation, especially since other metrics did not improve. |

**Urgency ranking:** termination_exploitation is most critical because it directly indicates the agent is bypassing the intended task (landing) by ending episodes early.

## 7. Episode Consistency Summary

Early vs. late behavior shows a significant **drift in angle_abs** (relative drift 0.70), with the policy moving from a moderate angle (0.055 rad) to a more tightly controlled one (0.016 rad). The consistency score is very low (0.30). This drift appears to be **unhealthy** — while angle stabilization might seem good, it is not accompanied by any descent or approach behavior. The policy is narrowing its focus on pitch control while ignoring all other objectives, indicating a **degenerate specialization** rather than adaptive improvement.

## 8. Key Numbers for Budget Calculation

Extracted from the Evaluation Metrics table at **timestep 1,000,000**:

- mean_length: **71.9**
- angle_abs: **0.018118**
- dist_to_pad: **9.115005**
- height: **7.162327**
- legs_ground: **0.019471**
- survival_fraction: **0.0**
- vert_speed_abs: **6.657329**