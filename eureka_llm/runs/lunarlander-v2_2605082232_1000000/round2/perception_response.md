## Behavior Trend Summary

- **200k timesteps**: Agent terminates early (mean length 480 steps). It is launched upward with high vertical speed (4.84) and far from the pad (distance 4.65). Falls quickly, no leg contact.
- **400k / 600k**: Agent survives the full 1000-step episode. It hovers stably near the ground (distance ~0.4–0.6), with slight downward velocity (~‑0.4) and low horizontal speed (~0.5). No leg contact — it avoids touching the ground.
- **800k**: Episode length drops slightly to 983 steps. Distance to pad shrinks further (0.25), leg contact appears (0.004) — occasional ground touch.
- **1M timesteps**: Episode length degrades to 557.6 steps. Leg contact rises sharply (0.385), vertical speed becomes more negative (‑0.84), distance increases slightly (0.31). Agent now frequently contacts the ground, causing early termination.

**Trend**: Initially improved (learning to hover) but then regressed as the agent began crashing into the ground more often. Final behavior: descends too fast, makes ground contact, and ends early.

**Key final numbers** (1M timesteps):  
`mean_length = 557.6`, `distance_to_pad = 0.309`, `horizontal_speed = 0.505`, `vertical_speed = -0.843`, `leg_contact = 0.385`

## Critical Metrics

| Metric | Trend | Direction Flag |
|--------|-------|----------------|
| **distance_to_pad** | Decreased from 4.65 → 0.31 (good, then plateau) | OK |
| **leg_contact** | Increased sharply at end (0 → 0.385) | ❌ Wrong direction (more ground contact) |
| **vertical_speed** | Changed from positive to negative, then more negative (‑0.84) | ❌ Wrong direction (descending faster, causing crashes) |
| **mean_length** | Rose to 1000 then dropped to 557.6 | ❌ Regression in episode survival |

## Reward Component Health

- **Active components** (mean significantly non‑zero):  
  - `_outcome` (‑0.80) – large negative, variable  
  - `r_alive` (0.50) – constant every step  
  - `r_leg_contact` (0.22) – positive, variable  
  - `r_distance` (‑0.08) – small negative, variable  
  - `r_vertical_speed` (‑0.02) – small negative, variable  

- **Dominant components** (|mean| > 2× others): None. The largest magnitude is `_outcome` (49.5% of magnitude sum) but `r_alive` (31%) is also significant; no component exceeds 80% dominance.

- **Inactive/negligible**: None – all components have non‑zero mean and non‑zero std.

- **Suspicious mean**: `r_alive` mean = 0.5 with std = 0.0 – this is a constant step reward, which is plausible (survival bonus) but worth noting.

## Behavioral Diagnosis

The agent’s current strategy is to descend toward the pad (distance shrinks) while maintaining a low horizontal speed, but it fails to arrest its downward velocity, resulting in increasing ground contact and premature episode termination. This is **genuine partial progress** that has stalled into a **local optimum** – it hovers near the ground but cannot land softly, so it eventually crashes.

## TDRQ Diagnosis

Overall TDRQ is 67.72 (moderate). The score is held back by **component balance (50.5)** – the reward is not overly dominated but `_outcome` and `r_alive` together account for 80% of the magnitude, and by **exploration health (50)**, suggesting the policy may be converging to a narrow behavior. The reward is **functional but should be kept** – the issue is not reward design per se but likely insufficient exploration or premature convergence. Diversification (e.g., population‑based training or entropy regularization) may help break out of the crashing pattern.

## Key Numbers for Budget Calculation

From the **final evaluation timestep (1,000,000)**:

| Metric | Value |
|--------|-------|
| mean_length | 557.6 |
| angle | -0.00107 |
| angular_velocity | 3e-06 |
| distance_to_pad | 0.308932 |
| horizontal_speed | 0.504821 |
| leg_contact | 0.384684 |
| vertical_speed | -0.843337 |