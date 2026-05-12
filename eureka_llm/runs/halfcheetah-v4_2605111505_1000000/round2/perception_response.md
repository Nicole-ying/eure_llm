# Perception Report: Planar Cheetah Forward Running Policy

## 1. Behavior Trend Summary

- **What the policy is doing:** At every evaluation milestone (200k–1M timesteps), the policy consistently achieves the maximum episode length of 1000 steps without termination. This indicates the cheetah never falls, never exceeds a time limit, and completes every episode. The reward attribution shows that forward progress (r_forward) dominates (77.8% of total reward), with a moderate contribution from a delta component (21.8%) and a tiny negative smoothness penalty. The agent is therefore running forward in a stable, non-falling gait.

- **Improvement / Regression / Flat:** The trajectory is **flat** – mean_length is constant at 1000.0 from the first evaluation onwards, with zero variance. No improvement in episode length (it is already maximal), and no regression. However, no other task-level metrics (e.g., forward speed, displacement) are available to assess whether speed or gait quality improved.

- **Key numbers at final timestep (1M):** mean_length = 1000.0, error = nan.

## 2. Critical Metrics

Only one task-level metric is available in the evaluation table: **mean_length**. The following three are inferred from the provided data as the most important:

| Metric | Value (final) | Why Important | Direction |
|--------|---------------|---------------|-----------|
| **mean_length** | 1000.0 | Direct measure of episode survival – maximum value indicates the agent avoids falling or timeout. Perfect score, but no further differentiation. | Stable (maxed) |
| **r_forward (mean)** | 2.748 | Dominant reward component; directly drives forward movement. Its magnitude and low std (1.29) suggest consistent forward progress. | Stable |
| **length_utilization_ratio** | 1.0 | Ratio of actual episode length to maximum possible length. Value of 1.0 confirms every episode runs to full length – no early terminations. | Stable (maxed) |

- **Flagged metric:** None is moving in the wrong direction, but the **lack of variation** (mean_length_span_ratio = 0.0) is a concern – the policy may be stuck in a single behavior pattern.

- **Cross-metric pattern:** The combination of `length_utilization_ratio = 1.0` and `mean_length_span_ratio = 0.0` indicates that **every episode is identical in length** – no exploration of different termination times or failure modes. This high consistency may mask underlying inefficiency or reward hacking.

## 3. Reward Component Health

| Component | Mean | % Total | Active? | Dominant? | Negligible? |
|-----------|------|---------|---------|-----------|-------------|
| r_forward | 2.7481 | 77.8% | Yes (large positive) | **Yes** (>2× next) | No |
| r_delta | 0.7691 | 21.8% | Yes (moderate positive) | No | No |
| r_smooth | -0.0171 | 0.5% | Yes (small negative) | No | **Barely active** – <1% of total, but std is non-zero |

- **Inactive / negligible:** None (all have non-zero mean and std).
- **Suspicious near-zero across evaluations:** Not applicable – all means are stable across evaluations (no per-evaluation breakdown provided, but overall means are non-zero).
- **High mean but weak alignment with task progress:** r_forward is high and directly aligned with the task (move forward). However, without forward speed or displacement metrics, we cannot confirm that the reward correlates with actual distance traveled. The perfect episode length and lack of variance raise a suspicion that the cheetah might be taking small forward steps that avoid falling but produce minimal net displacement – a potential **reward-hacking** scenario where r_forward is earned without meaningful speed.

## 4. Behavioral Diagnosis

The agent’s current strategy is to **maintain a stable, never-falling gait that consistently runs to the maximum episode length**, earning forward reward at a moderate rate while incurring a small smoothness penalty. This behavior is **genuinely stable** but likely represents a **local optimum** – a repetitive, conservative gait that avoids failure but may not maximize forward speed. The agent is **low gain / low effort**: it achieves full survival with minimal risk, but the forward reward magnitude (2.75 per step) may correspond to a slow trot or bound rather than an efficient high-speed gallop. The strategy is consistent with the task goal of running forward with upright posture and all feet contacting the ground in a regular gait, but the **lack of variation in episode length** and the **complete absence of failure episodes** suggest the policy is not exploring faster or more efficient gaits. This is inconsistent with the intended “as fast as possible” objective – a truly optimal policy would likely show some variation in speed and occasionally risk falling to achieve higher performance.

## 5. TDRQ Diagnosis

The overall TDRQ score of **55.01 / 100** is **mixed**, primarily due to **component imbalance** (score 22.24 out of 100) – r_forward dominates 77.8% of total reward, dwarfing other terms. Exploration health (50.00) is also suboptimal, indicating the policy may have converged prematurely. Component activity is perfect (100.00). The low imbalance score reflects a heavy reliance on a single reward signal, making the policy brittle.

**Recommendation:** This reward should be **diversified** (e.g., via search/population-based training) to encourage exploration of alternative gaits and reduce dominance of the forward component. A more balanced reward with additional shaping terms (e.g., speed reference, energy efficiency, gait regularity) would help escape the local optimum.

## 6. Constraint Violations Summary

| Principle | Severity | Evidence | Urgency |
|-----------|----------|----------|---------|
| **State coverage** | Medium | mean_length_min = max = 1000.0; length_utilization_ratio = 1.0, span_ratio = 0.0. Episode lengths are **entirely concentrated** at the maximum – no variation across 1000 episodes. | High – Lack of diverse states/termination times suggests policy is trapped in a narrow attractor, limiting discovery of faster gaits. |
| **Exploration health** | Medium (implied) | TDRQ exploration_health = 50.0; no entropy trends available. Policy appears to have low stochasticity. | Moderate – Without exploration, the policy cannot improve beyond its current local optimum. |

No other principles (e.g., action smoothness, constraint satisfaction) are directly violated, but the **extreme uniformity** is the primary concern.

## 7. Episode Consistency Summary

The policy exhibits **perfect consistency** across early and late episodes: `early_late_consistency_score = 1.0` and `relative_drift = 0.0`. Every episode runs exactly 1000 steps, and reward components have low std relative to mean. This consistency indicates a **stable, well-converged policy** with no drift. However, the complete absence of drift is more likely a sign of **pathological stability** (stuck at a fixed point) than healthy adaptation. In a healthy training process, some variation in episode length or reward would be expected as the agent refines its gait. The current behavior suggests the policy has **stopped exploring** and is repeating the same deterministic sequence.

## 8. Key Numbers for Budget Calculation

| Metric | Value |
|--------|-------|
| mean_length | 1000.0 |
| error | nan |
| *No other task-level metrics provided.* | |