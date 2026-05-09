## 1. Behavior Trend Summary

At each evaluation milestone (200k–1M timesteps), the agent consistently produces short episodes: mean episode length hovers around 66–71 steps out of a maximum 1000. The agent is not completing the task; it is falling quickly. Key indicators: vertical speed is large and negative (around -7 m/s), horizontal speed positive (~2 m/s), distance to pad near 1.0 (falling from or near the pad), and leg contact extremely low (~0.02). The agent appears to take a few steps or push forward, then loses balance and falls to the ground, terminating the episode. There is no improvement over time—mean length is flat and vertical speed becomes slightly more negative (worse). At the final timestep (1M): mean_length=67.5, angle=-0.061 rad, distance_to_pad=0.964, vertical_speed=-6.96 m/s, horizontal_speed=1.89 m/s.

## 2. Critical Metrics

| Metric | Trend | Final Value | Direction |
|--------|-------|-------------|-----------|
| mean_length | Flat (~66–71) | 67.5 | No improvement |
| vertical_speed | Slightly more negative (worsening) | -6.96 | ⚠️ Wrong direction |
| distance_to_pad | Stable (~0.96–1.02) | 0.964 | No improvement |

The most important env-specific metrics are **mean_length** (episode survival), **vertical_speed** (how fast the agent falls), and **distance_to_pad** (proximity to landing/target). Vertical speed is trending in the wrong direction (becoming more negative), indicating the agent is falling faster at later timesteps. Leg contact remains negligibly low throughout.

## 3. Reward Component Health

- **Active components** (mean significantly non‑zero): `_outcome` (-0.48), `r_distance` (-0.095), `r_leg_contact` (+0.016).  
- **Dominant component**: `_outcome` (|mean| = 0.48) is >5× the next largest (`r_distance`), and constitutes 79.8% of total reward magnitude.  
- **Inactive/negligible**: `r_alive` (mean = 0.0099) is very small and contributes only 1.6% of total reward; it provides little shaping signal.  
- No component has a suspicious mean of exactly 0.

## 4. Behavioral Diagnosis

The agent’s current strategy is to briefly move forward (positive horizontal speed) but then immediately fall, leading to early termination. It is stuck in a local optimum where the large negative outcome penalty for falling dominates the reward, flooding the signal from other components. The agent has not learned to maintain balance or stand upright—it repeatedly executes a short, unbalanced push followed by a fall.

## 5. TDRQ Diagnosis

The low TDRQ score (45.32) is primarily driven by **poor component balance** (20.16/100): the `_outcome` penalty is far larger than all other rewards combined, overwhelming learning. Exploration health is moderate (50). This reward function needs to be **iterated** to reduce the dominance of the terminal penalty and introduce intermediate shaping for stable posture, so the agent can discover a viable balancing strategy instead of being punished immediately for falling.

## 6. Key Numbers for Budget Calculation

| Metric | Value |
|--------|-------|
| mean_length | 67.5 |
| angle | -0.060722 rad |
| angular_velocity | -0.032361 rad/s |
| distance_to_pad | 0.964078 |
| horizontal_speed | 1.89257 m/s |
| leg_contact | 0.019259 |
| vertical_speed | -6.964485 m/s |