## Perception Agent Report: Car Valley Momentum Task

### 1. Behavior Trend Summary

Across all evaluation milestones, the agent consistently fails to reach the goal (reached_goal = 0). At the final timestep (1,000,000), mean episode length is 999.0, action_force is high (0.963), but velocity_mag is extremely low (0.025) and distance to goal remains above 0.96. The only deviation occurs at 600,000 steps, where mean_length drops sharply to 181.7 (likely early terminations), but it recovers to 999 by later evaluations. No consistent improvement in task progress is observed; the trajectory is flat/regressing. The agent is applying large forces but achieving negligible motion and zero goal successes.

**Key numbers (final timestep):**
- mean_length: 999.0
- action_force: 0.963
- dist_to_goal: 0.961
- heading: 0.021
- reached_goal: 0.0
- velocity_mag: 0.025

### 2. Critical Metrics

- **dist_to_goal:** Primary measure of task progress. It remains high (~0.86–1.11) across all evaluations, showing no downward trend. This indicates the agent is not moving closer to the goal.
- **reached_goal:** Binary success metric; persistently zero. The agent never completes the task.
- **velocity_mag:** Measures actual movement. Extremely low (max 0.035) even though action_force is high (0.96+), revealing a severe efficiency failure.

**Metric moving in wrong direction:** None are improving; all are stagnant or slightly worse (e.g., dist_to_goal at 1M = 0.961 vs 0.867 at 800K).

**Cross-metric pattern:** High **action_force** (0.96) combined with very low **velocity_mag** (0.025) indicates that the agent is expending high effort but producing negligible motion. This suggests the car is either stuck, applying force in opposition to its momentum, or oscillating in place without net displacement.

### 3. Reward Component Health

- **Active components:** `_outcome` (mean -0.99, 96.8% of total) and `r_progress` (mean 0.033, 3.2%).
- **Dominant component:** `_outcome` accounts for >95% of the total reward magnitude, making it fully dominant.
- **Inactive/negligible:** `r_goal` (mean 1e-5) is effectively zero — the agent never reaches the goal, so this component contributes nothing.
- **Suspicious mean value:** `_outcome` is persistently near -1.0 across evaluations, suggesting a strong step penalty. Despite high magnitude, it is **not aligned with task progress** — dist_to_goal does not improve, and the agent continues to incur this penalty indefinitely. The agent appears to be trading off (or ignoring) progress optimization in favor of simply surviving as long as possible (mean_length frequently hitting the 999 limit).
- **High mean, weak alignment:** `_outcome` dominates but does not correlate with goal reach or decreasing distance; the agent is neither minimizing steps nor improving position.

### 4. Behavioral Diagnosis

The agent applies near-maximum force (action_force ~0.96) but achieves almost no net velocity (velocity_mag < 0.04) and never moves closer to the goal. Its strategy appears to be **applying constant force in a way that cancels itself out** — possibly oscillating rapidly or stuck in a local minimum where it rocks back and forth without building momentum. This is **not reward hacking** (no clear loophole) but rather a **local optimum** where high-force output yields minimal movement, avoiding early termination but also failing to make progress. The behavior is highly inefficient: **very high effort for negligible gain**. It does **not match the intended task** of building momentum by driving back and forth; the near-zero heading and low velocity suggest the car is not executing coordinated oscillations, and the extremely low heading drift contradicts the expected side-to-side motion needed to ascend the hill.

### 5. TDRQ Diagnosis

Low TDRQ (26.78/100) stems primarily from **component imbalance** (`_outcome` dominates at 96.8%), **exploration collapse** (entropy decreasing from 0.91 to -0.11, indicating premature determinism), and **inactivity** of the goal component. The reward function should be **diversified** — either by reducing the dominance of the outcome penalty or by introducing new components that directly reward momentum building and goal approach, while preserving exploratory potential. A population-based search may help escape the current local optimum.

### 6. Constraint Violations Summary

| Principle | Severity | Evidence | Summary |
|-----------|----------|----------|---------|
| Termination exploitation | Medium | mean_length=689.6 vs max=1691 (ratio 0.408); episodes often end far below the max, possibly from hitting boundaries or stability limits. | The agent appears to be exploiting early termination dynamics — average episode length is well below the maximum, suggesting premature endings that may avoid further penalty. |
| Temporal consistency | Medium | heading drifts from early mean 0.012 to late mean 0.081 (relative drift 5.6); early_late_consistency_score=0.0. | Behavior metric shifts strongly between training windows, indicating unstable or non-consistent policy dynamics. |

**Urgency:** Termination exploitation is more urgent because it directly impacts how the agent interacts with the environment (early exits bypass intended learning). Temporal inconsistency is secondary but indicates instability.

### 7. Episode Consistency Summary

**Early vs. late behavior:** The heading metric shows a large relative drift (5.6) and a consistency score of 0.0, meaning the policy’s directional behavior changed substantially over training. Early episodes had near-zero heading (pointing straight), while later episodes had slightly positive heading (pointing right). This drift **does not reflect healthy adaptation**, as it does not coincide with improved task performance (dist_to_goal unchanged). Instead, it likely reflects the agent narrowing its deterministic strategy to a slightly different, but equally ineffective, region of action space — a sign of **unstable policy collapse** rather than purposeful refinement.

### 8. Key Numbers for Budget Calculation

**From evaluation metrics table (final timestep 1,000,000):**
- mean_length: 999.0
- action_force: 0.963149
- dist_to_goal: 0.960802
- heading: 0.021421
- reached_goal: 0.0
- velocity_mag: 0.02542

**Other milestones:**
- At 200,000: mean_length=999.0, dist_to_goal=1.1088, heading=-0.0104, velocity=0.0003
- At 400,000: mean_length=999.0, dist_to_goal=0.8882, heading=0.0350, velocity=0.0181
- At 600,000: mean_length=181.7, dist_to_goal=0.9144, heading=0.1877, velocity=0.0314
- At 800,000: mean_length=999.0, dist_to_goal=0.8673, heading=0.0345, velocity=0.0349