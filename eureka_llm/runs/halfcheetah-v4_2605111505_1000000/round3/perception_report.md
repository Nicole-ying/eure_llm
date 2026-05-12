# Training Report: Planar Cheetah Locomotion

## 1. Behavior Trend Summary

At all evaluation milestones (200k through 1M timesteps), the policy achieves a **mean episode length of 1000.0** with NaN error, indicating every episode runs to the maximum possible duration without early termination. The policy has converged to a stable, repeatable gait that avoids falling or violating episode termination conditions.

**Trend:** The trajectory is **flat from the first checkpoint onward** — no improvement or regression occurs after 200k timesteps. The policy reached a ceiling (max episode length) early and maintained it.

**Key final metrics:**
- mean_length: **1000.0**

## 2. Critical Metrics

### Task-Level Metrics

| Metric | Final Value | Importance |
|--------|-------------|------------|
| **mean_length** | 1000.0 | Primary measure of task success — indicates the agent avoids termination (falling, timeouts) for the full episode duration. Maximum possible value achieved. |
| *(No other task-level metrics available)* | | |

**Cross-metric pattern:** The *length_utilization_ratio* (1.0) and *mean_length_span_ratio* (0.0) together indicate **zero variance across all episodes** — every single trial runs exactly to the maximum length, producing a perfectly homogeneous trajectory set. This extreme consistency suggests the policy has locked onto a single behavioral mode.

**Metrics moving in wrong direction:** None detected; all metrics are at optimal or near-optimal values relative to the termination criterion.

## 3. Reward Component Health

| Component | Mean | Std | Status | Notes |
|-----------|------|-----|--------|-------|
| **r_forward** | 2.8131 | 0.9231 | **Active, Major** | Largest mean component (50.9% of total). Drives forward locomotion speed. |
| **r_delta** | 2.7000 | 0.4717 | **Active** | Second-largest (48.8%). Likely rewards joint displacement or action distances. |
| **r_smooth** | -0.0172 | 0.0029 | **Active, Minor** | Small negative penalty (0.3% of total). Likely penalizes jerky joint movements. |

**Key observations:**
- **No component is dominant** (>80% of total). The two main components (r_forward and r_delta) are balanced in magnitude.
- **All 3 components are active** (non-zero mean, non-zero std), and none are inactive/negligible.
- **No suspicious means:** r_forward and r_delta are consistently positive and stable across evaluations.
- **Alignment concern:** High r_forward combined with max episode length could indicate the agent runs fast enough to stay upright, but the **complete absence of episode length variation** suggests the forward speed reward may be saturating a local optimum rather than genuinely maximizing velocity. The policy may have found a speed that keeps it alive but not necessarily the fastest possible gait.

## 4. Behavioral Diagnosis

**The agent's current strategy is to execute a highly repeatable, conservative gait that maintains stability at a moderate forward speed, avoiding falls and maximizing episode duration.** This is **genuine progress** toward staying upright but likely represents a **local optimum** — the agent has found a gait that reliably avoids termination but may not be optimizing for maximum forward velocity. The policy is **not reward hacking** in the usual sense (no obvious exploitation of signal artifacts), but it has converged to a **low-variance, low-risk** behavior that achieves maximum episode length at the cost of exploration.

**Efficiency assessment:** The agent appears to be operating at **moderate effort for high task success** — it moves forward consistently (r_forward ~2.8) without excessive joint strain (r_delta ~2.7), but the complete homogeneity of trajectories suggests it is **not efficiently exploring faster gaits**.

**Comparison with Task Context:** The task goal calls for *"running forward as fast as possible"* with *"all four feet contacting the ground in a regular gait cycle."* While the agent maintains a regular gait (no early falls), the **max episode length does not guarantee maximum speed** — the policy could be trotting at a comfortable pace rather than sprinting. The absence of velocity metrics makes it impossible to confirm whether the agent is running at peak speed. The **100% episode survival rate** combined with **no variation** is inconsistent with an agent that is actively trying to push speed limits (which would produce occasional falls at high velocities).

## 5. TDRQ Diagnosis

Total TDRQ is **67.11/100** (mixed range). The score breakdown reveals:

- **Component balance (49.13):** The main weakness — two components (r_forward and r_delta) are nearly equal in magnitude without clear dominance, and the negative r_smooth is negligible. This balanced profile can cause gradient competition between forward speed and joint displacement rewards.
- **Component activity (100.00):** Perfect — all components are active, which is healthy.
- **Exploration health (50.00):** Below healthy threshold — the policy has stopped exploring after achieving max episode length, as evidenced by zero variance in trajectories.

**Diagnosis:** The low TDRQ is primarily driven by **exploration collapse** (the policy converged to a narrow behavioral mode) and **moderate component imbalance** (competing objectives without clear prioritization). **This reward should be diversified** — either by adding explicit velocity metrics or via population-based search to encourage the agent to explore faster gaits beyond the current stable local optimum.

## 6. Constraint Violations Summary

| Principle | Severity | Evidence | Urgency |
|-----------|----------|----------|---------|
| **state_coverage** | **Medium** | mean_length_min=1000.0, mean_length_max=1000.0 — **100% of episodes are identical in length**; the policy has zero behavioral diversity. | **High** — the agent is stuck in a single, narrow local optimum. |

**No other constraints violated.** The high state coverage violation is the sole concern, and it is urgent because it prevents further improvement toward the task goal (maximum forward speed).

## 7. Episode Consistency Summary

**Early vs. Late behavior:** The policy shows **perfect consistency** (early_late_consistency_score = 1.0) with **zero drift** (early_late_relative_drift = 0.0). All 1000 episodes are statistically identical in length.

**Interpretation:** This is **not healthy adaptation** — the lack of drift indicates the policy stopped learning/exploring early in training and has remained frozen in a single behavioral mode. There is no evidence of progressive improvement or meaningful adaptation across the training trajectory. The policy style is **consistently stuck**, not adaptively stable.

## 8. Key Numbers for Budget Calculation

| Metric | Final Value |
|--------|-------------|
| **mean_length** | 1000.0 |
| *(No other task-level metrics available)* | |