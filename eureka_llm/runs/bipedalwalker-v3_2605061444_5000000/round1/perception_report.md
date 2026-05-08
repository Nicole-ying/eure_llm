# Perception Report

## 1. Behavior Trend Summary

- **500k–1M timesteps:** The agent falls immediately in every episode (fall_rate = 1.0, completion_rate = 0.0). Episode lengths are short (70–267 steps), indicating the agent cannot maintain balance.
- **1.5M timesteps:** A sharp behavioral shift occurs — completion_rate jumps to 0.6, fall_rate drops to 0.4, and mean_length increases dramatically to ~1075 steps. The agent has learned to balance and complete the task in most episodes.
- **2M–3.5M timesteps:** Continued improvement. Completion_rate rises to 0.9, then to 1.0 by 3.5M. Fall_rate declines to 0.0. Mean_length stabilizes around 1100–1180 steps.
- **3.5M–5M timesteps:** Fully converged behavior. The agent completes the task in every episode (1.0 completion_rate), never falls (0.0 fall_rate), and maintains a consistent episode length of ~1100–1160 steps.

**Overall trajectory:** Strongly improving from failure to mastery, then flat at optimal performance.

**Key final numbers:** completion_rate = 1.000, fall_rate = 0.000, mean_length = 1147.8

## 2. Critical Metrics

No environment-specific metrics were collected. The only available behavioral metrics are:

- **completion_rate:** Trend is strongly positive — rose from 0.0 to 1.0 and stabilized.
- **fall_rate:** Trend is strongly negative — dropped from 1.0 to 0.0 and stabilized.
- **mean_length:** Trend is positive — increased from ~70 to ~1150 and stabilized.

No metrics are moving in the wrong direction.

## 3. Reward Component Health

- **Active components:** Both `_outcome` (mean = -0.222) and `r_forward` (mean = 3.552) have significantly non-zero means.
- **Dominant component:** `r_forward` is dominant — its mean (3.552) is ~16× larger in magnitude than `_outcome` (-0.222), and it accounts for 94.1% of total reward.
- **Inactive/negligible components:** None. Both components are active.
- **Suspicious values:** None. Both means are non-zero and have non-zero standard deviations.

## 4. Behavioral Diagnosis

The agent has learned to consistently balance and move forward to complete the task, never falling. It is heavily optimizing for forward progress (r_forward dominates at 94% of total reward), while the outcome penalty is small and negative — likely a small penalty for falling that the agent has learned to avoid entirely. This is genuine progress toward the task objective, not reward hacking.

## 5. Key Numbers for Budget Calculation

- **Environment-specific metrics:** None collected
- **fall_rate:** 0.000
- **mean_length:** 1147.8
- **completion_rate:** 1.000