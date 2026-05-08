# Perception Report: Lunar Lander Training Evaluation

## 1. Behavior Trend Summary

At 200K timesteps, the agent was failing catastrophically — falling immediately in every episode (fall_rate=1.000, completion_rate=0.000) with mean episode length of 578.6 steps. By 400K timesteps, the agent achieved perfect completion (completion_rate=1.000, fall_rate=0.000) and maintained 100% completion through all subsequent evaluations. However, mean episode length dropped sharply from 1000.0 at 400K to ~498-516 at 600K-800K, then further to 459.5 at 1M timesteps. The trajectory shows rapid initial improvement followed by a plateau in completion rate but a concerning downward trend in episode length, suggesting the agent is learning to finish faster but potentially at the cost of landing quality.

**Final values:** completion_rate=1.000, fall_rate=0.000, mean_length=459.5

## 2. Critical Metrics

| Metric | Trend | Assessment |
|--------|-------|------------|
| **descent_quality_mean** | Rose from 0.007→0.620 (400K), then declined to 0.379 (1M) | **⚠️ Moving in wrong direction** — dropping after peak |
| **landing_quality_mean** | Rose from 0.174→0.335 (400K), then declined to 0.282 (1M) | **⚠️ Moving in wrong direction** — degrading after peak |
| **approach_quality_mean** | High throughout (0.928→0.971), slight decline from peak 0.988 | Stable but slightly declining |

**Flagged:** Descent quality and landing quality are both degrading after peaking at 400K, while the agent completes the task faster. This suggests a speed-quality tradeoff.

## 3. Reward Component Health

**Active components (mean significantly non-zero):**
- `r_angle` (mean=0.894, 39.4% of total) — dominant, stable
- `_outcome` (mean=-0.555, 24.5% of total) — large negative penalty, variable
- `r_landing` (mean=0.395, 17.4% of total) — noisy but active
- `r_progress` (mean=0.256, 11.3% of total) — variable but active
- `r_survival` (mean=0.100, 4.4% of total) — stable, constant positive

**Dominant components:** None exceed 2× the next largest. `r_angle` (0.894) is ~1.6× `_outcome` (0.555) and ~2.3× `r_landing` (0.395).

**Inactive/negligible components:**
- `r_ang_vel` (mean=-0.004, 0.2%) — negligible
- `r_efficiency` (mean=-0.007, 0.3%) — negligible
- `r_termination` (mean=-0.010, 0.4%) — negligible

**Suspicious values:** `r_survival` has mean=0.100 with std=0.000 — exactly constant, which is expected for a fixed per-step survival bonus but worth noting.

## 4. Behavioral Diagnosis

The agent has learned to reliably complete the landing task without crashing, but its strategy is shifting toward faster, lower-quality landings — it prioritizes maintaining a good angle (r_angle dominates) while rushing the descent, resulting in degraded descent and landing quality over time. This is not reward hacking but a local optimum where the agent optimizes for speed and angle stability at the expense of smooth, high-quality touchdowns.

## 5. Key Numbers for Budget Calculation

**Environment Metrics:**
- approach_quality_mean: 0.970551
- descent_quality_mean: 0.378704
- drift_quality_mean: 0.291178
- fuel_efficiency_mean: 0.545811
- landing_quality_mean: 0.282035
- stability_mean: 0.956747

**Behavior Metrics:**
- fall_rate: 0.000
- mean_length: 459.5
- completion_rate: 1.000