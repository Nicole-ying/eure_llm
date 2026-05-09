# Perception Agent Report

## 1. Behavior Trend Summary

At each evaluation milestone, the agent moves continuously (action magnitude ~0.96–0.98) but with very low velocity (~0.012–0.014). The distance to goal decreases slowly from 0.999 to 0.786 over 1M timesteps, indicating a gradual drift toward the goal. However, the success metric remains exactly 0.0 at all milestones, meaning the agent never completes the task. The steps metric (likely number of steps to goal or required steps) is also always 0.0, suggesting the agent never triggers whatever event counts as “success” or “step completion.” Episode lengths stay in a narrow range (67–76 steps, mean 79), far below the max of 999, implying early termination—possibly due to a time limit, falling out of bounds, or reaching a maximum episode length.

The trajectory is **slowly improving** in terms of distance reduction but **flat** in success rate. The agent is not achieving the primary objective.

**Key numbers at final timestep (1M):**
- mean_length: 76.1
- distance_to_goal: 0.786
- success: 0.0
- steps: 0.0
- velocity: 0.012

---

## 2. Critical Metrics

- **distance_to_goal** – decreasing from 0.999 to 0.786 (improving, but still far from 0.0)
- **success** – constant 0.0 (no improvement, critical failure)
- **mean_length** – stable around 70, no trend (flat)

**Flagged**: success is moving in the wrong direction (stuck at zero); steps also stuck at zero.

---

## 3. Reward Component Health

| Component | Mean | Std | % of Total | Activity |
|-----------|------|-----|------------|----------|
| _outcome | 1.0000 | 0.0000 | 23.3% | active |
| r_distance_penalty | -1.7096 | 0.2047 | 39.8% | active |
| r_progress | -0.2024 | 0.0512 | 4.7% | active |
| r_success | 1.3826 | 0.2382 | 32.2% | active |

- **Active components**: all four have mean significantly non-zero.
- **Dominant component**: none exceeds 80% of total reward magnitude. However, r_distance_penalty (39.8%) and r_success (32.2%) together dominate (~72%).
- **Inactive/negligible**: none.
- **Suspicious values**: _outcome has mean exactly 1.0 with zero std — likely a constant offset (e.g., alive bonus). r_success has a positive mean (1.38) despite a 0% success rate, suggesting it is awarded for intermediate progress (e.g., moving toward goal), not terminal success.

---

## 4. Behavioral Diagnosis

The agent’s strategy is to move slowly toward the goal (distance decreases) but never get close enough to trigger a success event. It appears to be caught in a local optimum where reducing distance gives a small positive reward via r_success (or r_progress) while staying far from the goal minimizes the r_distance_penalty. The agent does not exploit larger actions to cover ground quickly, perhaps because high action magnitude is already present but applied ineffectively (velocity low). No reward hacking is evident; rather, the agent is following the shortest path to reward but failing to reach the terminal condition.

---

## 5. TDRQ Diagnosis

TDRQ overall score is **82.09/100** (healthy), but the component_balance subscore is only **60.19** — indicating moderate imbalance between the distance penalty and success reward, which together dominate the reward signal. Exploration health is perfect (100) and component activity is full (100), so the lower balance is the main drag. This reward structure should be **kept** but would benefit from re-weighting to reduce dominance of r_distance_penalty and r_success, or from adding a shaping that encourages faster progress toward the actual goal.

---

## 6. Key Numbers for Budget Calculation

| Metric | Value |
|--------|-------|
| mean_length | 76.1 |
| action_magnitude | 0.963 |
| distance_to_goal | 0.786 |
| steps | 0.0 |
| success | 0.0 |
| velocity | 0.012 |