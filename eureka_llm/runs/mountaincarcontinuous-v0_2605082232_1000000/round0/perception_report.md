```markdown
## 1. Behavior Trend Summary

At each evaluation milestone:
- **200k**: Episodes are very long (mean length 999), agent barely moves (velocity 0.00017), distance to goal ~0.73, no successes.
- **400k**: Length drops to 580.1, velocity rises slightly (0.00084), distance to goal ~0.76, still no success.
- **600k**: Length drops sharply to 109.5, velocity 0.0086, distance to goal ~0.89 – agent is moving a little but getting *further* from the goal.
- **800k**: Length 88.0, velocity 0.0105, distance to goal ~1.00 – near maximal distance, no success.
- **1M**: Length 81.3, velocity 0.0115, distance to goal ~0.94 – slightly closer than before but still far, no success.

**Trend**: The agent is reducing episode length dramatically but is **not reaching the goal**. It is moving very slowly and ends up far from the target. Performance is essentially flat in terms of success, and the decreasing length may indicate it is learning to terminate early (possibly to avoid negative reward or due to timeout). The trajectory is **not improving** on the primary task.

## 2. Critical Metrics

| Metric | Final Value (1M) | Trend |
|--------|------------------|-------|
| **success** | 0.0 | **Flat** – never achieved. Moving in wrong direction (stuck at 0). |
| **distance_to_goal** | 0.943 | **No improvement** – stays high (0.73–1.0). |
| **velocity** | 0.0115 | **Slightly increasing** but remains negligible. |

**Flagged**: *success* is the most important metric and is zero throughout – this is the most critical problem. *distance_to_goal* is not decreasing; the agent is not making progress toward the goal.

## 3. Reward Component Health

| Component | Mean | Std | % of Total | Status |
|-----------|------|-----|------------|--------|
| `_outcome` | 1.0000 | 0.0000 | 46.9% | Active – constant, likely a baseline reward. |
| `r_progress` | -0.2141 | 0.0376 | 10.0% | Active – small negative, stable. |
| `r_success` | 0.9159 | 0.3800 | 43.0% | Active – positive, variable. |

- **Active components**: All three (means significantly non‑zero).  
- **Dominant**: None – no component exceeds 2× the others (`_outcome` and `r_success` are close).  
- **Inactive/negligible**: None.  
- **Suspicious values**: `_outcome` has zero variance (exactly 1.0 every step) – it may be a constant offset term (e.g., a survival bonus) rather than a learned signal.

## 4. Behavioral Diagnosis

The agent is **not moving effectively** towards the goal (high distance, very low velocity) and never succeeds. It receives a large positive `r_success` reward during training (mean 0.92) despite the evaluation success being zero, suggesting either a mismatch between the train-time success trigger and the eval metric, or the agent is exploiting a shaping reward that doesn’t translate to actual goal achievement. The decreasing episode length may indicate it is learning to terminate episodes early to avoid the negative `r_progress` penalty, while still collecting the constant `_outcome` reward. This is a **local optimum** where the agent “survives” but does not solve the task.

## 5. TDRQ Diagnosis

TDRQ overall score is **78.87 / 100** (healthy). The main sub‑score drag is **component_balance** (53.05) – this is due to the constant `_outcome` and the large `r_success` dominating over the weak `r_progress`. However, exploration and component activity are both perfect (100). The low balance indicates the reward structure is not guiding the agent toward the true objective. **The reward design should be kept, but the `_outcome` constant and the `r_success` definition need to be better aligned with the evaluation criterion** (e.g., only reward when the agent actually reaches the goal in eval). Until then, the agent will continue to exploit the current shaping.

## 6. Key Numbers for Budget Calculation

From the final evaluation (timestep 1,000,000):

| Metric | Value |
|--------|-------|
| mean_length | 81.3 |
| action_magnitude | 0.962 |
| distance_to_goal | 0.943 |
| steps | 0.0 |
| success | 0.0 |
| velocity | 0.0115 |
```