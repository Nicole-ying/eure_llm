```python
import math
import numpy as np


def compute_reward(self, state, m_power, s_power, terminated):
    """
    State-based reward with strong horizontal shaping, height‑dependent alive bonus,
    reduced crash penalty, and scaled components to stay within 5× magnitude.

    Dynamics: Gravity constantly pulls the lander down. The agent must fire
    the main engine to slow descent while staying upright and centered.
    The primary task is to reach the pad with both legs touching, low speed,
    and near‑zero angle. The reward is designed to be learnable from dense
    feedback without penalising gentle descent (vy dead zone).
    """
    # --- weights (adjusted per analyst proposal: w_x increased to 0.5) ---
    w_x = 0.5       # horizontal distance penalty weight (increased from 0.3 to create stronger centering gradient; at dist=9.86 → penalty ~ -4.93)
    w_angle = 0.004  # angle penalty weight (# typical |angle| ≤ 0.2 → max penalty ~0.0008)
    w_vy = 0.005     # vertical velocity penalty weight after dead‑zone (# typical high |vy| ~6 → penalty ~0.0275)
    w_vx = 0.005     # horizontal velocity penalty weight (# typical |vx| ≤ 1.5 → max penalty ~0.0075)

    # 1. Stability shaping (negative penalty for deviation from ideal state)
    # Apply dead zone to vertical velocity: only penalize |vy| > 0.5
    vy_penalty = max(0, abs(state[3]) - 0.5)  # no penalty for gentle descent
    r_stability = -(
        w_x * abs(state[0]) +
        w_angle * abs(state[4]) +
        w_vy * vy_penalty +
        w_vx * abs(state[2])
    )

    # 2. Pad proximity reward (encourages moving toward the pad horizontally)
    dist_x = abs(state[0])
    r_proximity = 0.2 * max(0, 1.0 - dist_x)   # increased multiplier for scaling; max +0.2 when centered

    # 3. Leg contact bonus – only when near the pad (dist_x < 1.5)
    leg_contacts = (1.0 if state[6] > 0.5 else 0.0) + (1.0 if state[7] > 0.5 else 0.0)
    if dist_x < 1.5:
        r_legs = 0.1 * leg_contacts   # increased multiplier for scaling; max +0.2 when both legs touch
    else:
        r_legs = 0.0                  # no bonus if far from pad

    # 4. Height‑dependent alive bonus (strongly incentivises descent below 15 units)
    # Removed distance gate (was: *(1.0 if abs(state[0]) < 5.0 else 0.0)) per analyst proposal
    r_alive = 0.5 * max(0, 1.0 - state[1] / 15.0)  # alive bonus, up to 0.5 when height=0

    # --- Build per-step components (no terminal outcome yet) ---
    components = {
        "r_stability": r_stability,
        "r_proximity": r_proximity,
        "r_legs": r_legs,
        "r_alive": r_alive,
    }

    # Total before terminal outcome (sum of all dense components)
    total = r_stability + r_proximity + r_legs + r_alive

    # --- Terminal outcome handling ---
    if terminated:
        # Determine success (same criteria as before)
        on_ground = (state[6] > 0.5 and state[7] > 0.5)
        angle_ok = abs(state[4]) < 0.15
        vert_vel_ok = abs(state[3]) < 0.2
        horiz_vel_ok = abs(state[2]) < 0.2
        centered = abs(state[0]) < 0.3
        success = on_ground and angle_ok and vert_vel_ok and horiz_vel_ok and centered

        if success:
            # Dense success bonus (part of total reward – not _outcome)
            components["r_landing_bonus"] = 20.0  # one‑time bonus on successful landing
            total += 20.0
            # _outcome is for evaluation only, NOT added to total
            components["_outcome"] = 1.0
        else:
            # Crash penalty (increased from -1.0 to -2.0 per analyst proposal)
            components["r_crash_penalty"] = -2.0
            total += -2.0
            components["_outcome"] = -1.0

    return total, components


def metrics_fn(env, action) -> dict:
    """
    Task performance metrics (reward‑agnostic).
    Uses only environment attributes and episode length.
    All accesses are safe (getattr with fallback).
    """
    # Safely access environment attributes
    lander = getattr(env, 'lander', None)
    legs = getattr(env, 'legs', None)
    helipad_y = getattr(env, 'helipad_y', 0.0)

    # Default values if attributes missing
    x = 0.0
    y = 0.0
    vx = 0.0
    vy = 0.0
    angle = 0.0
    leg_contact_left = 0.0
    leg_contact_right = 0.0

    if lander is not None:
        pos = lander.position
        vel = lander.linearVelocity
        x = pos.x
        y = pos.y
        vx = vel.x
        vy = vel.y
        angle = lander.angle

    if legs is not None and len(legs) >= 2:
        leg_contact_left = 1.0 if legs[0].ground_contact else 0.0
        leg_contact_right = 1.0 if legs[1].ground_contact else 0.0

    # Episode progress (steps survived / max steps 1000)
    step_count = getattr(env, '_elapsed_steps', 0)

    # Compute metrics
    metrics = {
        # Horizontal distance to pad centre (raw world coordinates)
        "dist_to_pad": abs(x),
        # Height above ground (raw world coord)
        "height": y - helipad_y,
        # Absolute angle (radians)
        "angle_abs": abs(angle),
        # Vertical speed magnitude (world coord)
        "vert_speed_abs": abs(vy),
        # Number of legs in ground contact
        "legs_ground": int(leg_contact_left + leg_contact_right),
        # Fraction of episode completed (survival)
        "survival_fraction": min(step_count / 1000.0, 1.0),
    }

    return metrics
```