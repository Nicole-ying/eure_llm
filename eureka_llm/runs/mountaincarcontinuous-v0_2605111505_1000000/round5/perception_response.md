# Perception Agent Report

## 1. Behavior Trend Summary

At **200k timesteps**, the policy produces extremely long episodes (mean length 999) with very low velocity (~0.004), near‑zero action force (~0.38), and heading near zero. The car is essentially stationary.  
At **400k timesteps**, a shift occurs: episodes become much shorter (mean length 167), action force rises (~0.88), velocity increases (~0.031), and heading becomes positive (~0.18). This suggests the agent briefly learns to apply force and move rightward, but episodes terminate early (likely due to boundary or stability issues).  
From **600k to 1M timesteps**, the policy regresses: episodes return to the maximum truncation length (999), velocity drops back to ~0.01, heading hovers near zero, and action force remains high (~0.87). The agent has lost the short‑episode strategy and is once again stuck, expending force without generating meaningful motion.

**Overall trajectory:** No improvement toward the goal—**reached_goal remains 0.0 at every evaluation**. The only transient positive change (400k) was not sustained, and final performance is similar to the initial naive policy.

**Key numbers at final timestep (1M):**  
- mean_length = 999.0  
- action_force = 0.870526  
- dist_to_goal = 0.862773  
- heading = 0.017217  
- reached_goal = 0.0  
- velocity_mag = 0.011207  

---

## 2. Critical Metrics

| Metric | Why important | Direction | Notes |
|--------|---------------|-----------|-------|
| **reached_goal** | Direct measure of task completion – always 0, never reached. | **Flat (failing)** | No progress toward flag. |
| **dist_to_goal** | Inverse proxy for goal proximity – remains ~0.86–0.94 throughout training. | **Nearly flat** | Only ~15% reduction from initial 1.01; far from the 0.0 target. |
| **velocity_mag** | Indicates whether the car is gaining the momentum needed to climb the hill – stays below 0.04. | **Stagnant** | In a task that requires building speed, this is critically low, especially given high action forces. |

**Cross‑metric pattern:**  
- **High action_force (~0.87) × low velocity_mag (~0.01)** → The applied force is not translating into acceleration. This is consistent with the car either pushing against a steep slope without enough momentum, or braking/oscillating in place. The disconnect between effort and motion is a clear inefficiency.

- **heading drift** (early vs. late windows, relative drift 0.96): early evaluation had positive heading (rightward), later heading near zero. The policy is not maintaining a consistent directional strategy even within a single evaluation run.

---

## 3. Reward Component Health

- **Active components:** Only `r_progress` is active (mean 0.0754, std 0.0607). It is also **dominant** (100% of total reward).  
- **Inactive / negligible:** `_outcome` and `r_goal` both have mean = 0.0 and std = 0.0. They never provide any signal.  
- **Suspicious pattern:** `r_goal` remains exactly zero despite `reached_goal` always being 0. This is expected for a terminal reward if the agent never reaches the goal, but it also means the reward function has no way to shape behavior toward the flag.  
- **Weak alignment:** `r_progress` has moderate positive mean, yet `dist_to_goal` barely changes. The progress reward may be based on small distance improvements or on some internal proxy that is not strongly correlated with actual goal approach. The high variance (std 0.0607 vs. mean 0.0754) suggests the signal is noisy and may be driving inconsistent behavior.

**Warning:** The reward function is effectively single‑objective (`r_progress` only), and that objective is not guiding the agent to the goal. The multi‑objective design (which likely intended `r_goal` to provide a terminal bonus) is completely unused.

---

## 4. Behavioral Diagnosis

The agent’s current strategy is to apply moderate to high force while remaining nearly stationary, producing long episodes that never reach the flag. This is **not reward hacking** (no obvious exploitation of reward gaps) but rather a **stuck local optimum** where the car cannot build the momentum required to climb the steep hill. Efficiency is very poor: high action_force (≈0.87) yields negligible velocity (≈0.01), indicating the agent is exerting effort without gain.  
Compared to the intended task description (“build momentum by driving back and forth”), the agent shows no evidence of a back‑and‑forth swinging motion. The low heading variability and near‑zero velocity are inconsistent with a strategy that requires alternating directions to build speed. The agent appears to be attempting a direct uphill push, which is insufficient with a weak engine.

---

## 5. TDRQ Diagnosis

The TDRQ score of **21.67 / 100** is predominantly due to **component imbalance** (score 0 – a single component dominates) and **component inactivity** (score 33.33 – two of three components are dead). Exploration health is moderate (50.00) because entropy is still decreasing, but not yet collapsed to extreme values.  
**Recommendation:** This reward function should be **iterated** – specifically, the terminal goal reward (`r_goal`) should be made nonzero and the progress reward should be redesigned to encourage the oscillatory momentum‑building behavior, or diversification/search should be used to discover a feasible strategy under the current reward.

---

## 6. Constraint Violations Summary

| Principle | Severity | Evidence | Urgency |
|-----------|----------|----------|---------|
| **termination_exploitation** | medium | mean episode length (760.4) is only 40% of the maximum observed length (1881). The policy is ending episodes early, likely by falling off the environment or exceeding a stability threshold. | **High** – because it prevents the agent from even attempting the full task and may mask learning failures. |
| **temporal_consistency** | medium | The heading metric drifts strongly (relative drift 0.96) between early and late training, indicating the policy’s behavior is not stable across time steps within an evaluation. | **Medium** – indicates lack of a consistent, repeatable strategy. |

Both violations are interrelated: the early termination may cause the policy to never experience long enough trajectories to learn proper momentum building, and the inconsistent heading suggests the agent has not settled on a usable movement pattern.

---

## 7. Episode Consistency Summary

Early training (200k–400k) shows a **dramatic style shift**: from long, stationary episodes to short, higher‑velocity ones. Later stages (600k onward) revert to long episodes with low velocity, but the heading and force patterns differ from the initial stationary behavior.  
This is **unstable and unhealthy adaptation** – the policy is not converging to a consistent, effective gait. The early short episodes may have been an exploratory dead‑end (e.g., falling off the map) that the agent then unlearns. The drift in heading (from positive to near‑zero) further indicates the agent is not maintaining a directional preference. True healthy progress would show a stable, purposeful oscillation with increasing amplitude.

---

## 8. Key Numbers for Budget Calculation

Extracted from the **Evaluation Metrics** table at final timestep (1,000,000):

- **mean_length** = 999.0  
- **action_force** = 0.870526  
- **dist_to_goal** = 0.862773  
- **heading** = 0.017217  
- **reached_goal** = 0.0  
- **velocity_mag** = 0.011207