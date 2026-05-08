### 1. Behavior Trend Summary
- At 200k timesteps, the agent rarely completes the task (10% completion) and is mostly truncated (90%), surviving an average of 884 steps.
- By 400k, performance peaks: 100% completion, no falls, no truncations, full 1000-step episodes.
- At 600k, performance drops sharply: 50% completion, 50% truncation, shorter episodes (mean 782 steps).
- At 800k, the agent recovers to 100% completion again.
- At 1M timesteps, performance collapses: 0% completion, 100% truncation, very short episodes (mean 290.5 steps).
- **Trend:** Highly unstable — the agent alternates between perfect performance and complete failure, ending in a severe regression.
- **Final numbers:** completion_rate = 0.000, fall_rate = 0.000, mean_length = 290.5.

### 2. Critical Metrics
- **distance_to_pad** (mean=12.35): The agent is, on average, far from the landing pad, consistent with early truncation.
- **vertical_speed** (mean=-2.41): The agent is descending at a moderate speed, not hovering or climbing.
- **stability** (mean=0.22): Low stability, indicating the agent is wobbling or not maintaining a steady orientation.
- **Flag:** The final collapse to 0% completion and very short episodes is a clear wrong-direction trend.

### 3. Reward Component Health
- **Active components:** All four components have non-zero means.
- **Dominant component:** `_outcome` (57.3% of total reward) is the largest, but not >2× the next largest (`r_shaped` at 27.8%).
- **Inactive/negligible:** None — all components contribute meaningfully.
- **Suspicious values:** `both_legs_contact` has a mean of 0.0 with zero std, indicating the agent never achieves two-legged contact with the pad.

### 4. Behavioral Diagnosis
The agent's strategy is inconsistent and fragile — it sometimes completes the task perfectly but often fails early, getting truncated after short, unstable descents. It never lands with both legs on the pad, suggesting it is reward-hacking by surviving for a while without achieving a proper landing, then collapsing into early truncation.

### 5. Key Numbers for Budget Calculation
- **distance_to_pad:** 12.35
- **speed:** 3.26
- **vertical_speed:** -2.41
- **stability:** 0.22
- **angle_error:** 0.19
- **both_legs_contact:** 0.0
- **fall_rate:** 0.000
- **mean_length:** 290.5
- **completion_rate:** 0.000