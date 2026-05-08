"""LLM-generated reward function (round 2).
Source: round2
"""

import math
import numpy as np

"""
LLM-generated reward function with proposed changes.
Applies analyst modifications:
  - Reduce termination failure penalty from -15 to -1.
  - Increase success reward from 10 to 50.
  - Expand close-range progress bonus radius from 0.5 to 1.0, coefficient from 1.0 to 2.0.
"""

import math
import numpy as np

def compute_reward(self, state, m_power, s_power, terminated):
    """
    Dense reward shaping for lunar landing.
    Encourages: reaching landing pad, upright orientation, low velocity, both legs contact.
    Returns:
        total_reward: float
        components: dict
    """
    components = {}

    # Unpack state for convenience
    x_norm, y_norm, vx_norm, vy_norm, angle, angvel_norm, leg_left, leg_right = state

    # --- 1. Progress toward landing pad ---
    # Target: x=0 (horizontal center), y=0 (pad altitude)
    # Use squared distance with a shaping bias to avoid early settling
    pad_dist = math.sqrt(x_norm**2 + y_norm**2)
    # Penalty proportional to distance. Typical max pad_dist ~1.5, so max penalty ~ -1.5
    r_progress = -1.0 * pad_dist
    # Close-range bonus: expanded zone to radius 1.0 (was 0.5), coefficient doubled to 2.0
    # At pad_dist=0, bonus = 2.0; at pad_dist=1.0, bonus = 0.0
    if pad_dist < 1.0:
        r_progress += 2.0 * (1.0 - pad_dist)  # max extra +2.0 when at pad
    components['r_progress'] = r_progress

    # --- 2. Stability (angle and angular velocity) ---
    # Angle target: 0 (upright). Use smallest signed angular difference.
    ang_diff = math.atan2(math.sin(angle), math.cos(angle))  # wrap to [-pi, pi]
    # Penalize deviation from zero. Max absolute deviation ~pi, so reward range ~[-1, 0]
    # Coefficient 0.3 means max penalty ~ -0.94 (pi*0.3)
    r_stability_angle = -0.3 * abs(ang_diff)
    # Penalize angular velocity – keep it small (< ~0.5 rad/frame scaled)
    # Coefficient 0.2, max angular velocity norm ~2.0, so max penalty ~ -0.4
    r_stability_angvel = -0.2 * abs(angvel_norm)
    # Combined stability
    r_stability = r_stability_angle + r_stability_angvel
    components['r_stability'] = r_stability

    # --- 3. Leg contact (landing success shaping) ---
    # Reward two legs touching, penalty for only one (tipping)
    if leg_left and leg_right:
        r_contact = 1.0  # both down – immediate small reward
    elif leg_left or leg_right:
        r_contact = -0.5  # unbalanced – penalty
    else:
        r_contact = 0.0
    components['r_contact'] = r_contact

    # --- 4. Fuel efficiency (penalize excessive thrust) ---
    # m_power in [0,1] for main, s_power in [0,1] for side
    # Penalty -0.1 per unit power, max -0.2 per step (both engines full)
    r_efficiency = -0.1 * (m_power + s_power)
    components['r_efficiency'] = r_efficiency

    # --- 5. Termination handling ---
    r_termination = 0.0
    if terminated:
        # Determine success or failure
        # Success: lander not awake AND both legs in contact AND near pad AND low velocity
        success = (not self.lander.awake and leg_left and leg_right and
                   abs(ang_diff) < 0.3 and abs(vy_norm) < 0.1 and pad_dist < 0.2)
        if success:
            r_termination = 50.0  # large positive reward for successful landing (increased from 10)
        else:
            r_termination = -1.0  # reduced from -15 to avoid dominating per-step rewards
        components['r_termination'] = r_termination

    # --- Sum total reward ---
    total = sum(components.values())
    return total, components


def metrics_fn(env, action) -> dict:
    """
    Task-level metrics reward-independent.
    Measures: distance to pad, angular stability, fuel usage, landing success rate (in info),
    velocity at termination, contact balance.
    """
    # Access cross-step state if needed
    pre = getattr(env, '_pre_step_state', {})

    # Get current physics state
    lander = env.unwrapped.lander
    legs = env.unwrapped.legs
    pos = lander.position
    vel = lander.linearVelocity
    angle = lander.angle

    # Normalize to same scale as state[0,1]
    # Use constants from env (assuming they exist; if not, use typical values)
    try:
        SCALE = env.unwrapped.SCALE
        VIEWPORT_W = env.unwrapped.VIEWPORT_W
        VIEWPORT_H = env.unwrapped.VIEWPORT_H
        FPS = env.unwrapped.FPS
        helipad_y = env.unwrapped.helipad_y
        LEG_DOWN = env.unwrapped.LEG_DOWN
    except AttributeError:
        # Fallback to typical LunarLander constants
        SCALE = 30.0
        VIEWPORT_W = 600.0
        VIEWPORT_H = 400.0
        FPS = 50.0
        helipad_y = 0.0
        LEG_DOWN = 1.0

    x_norm = (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2)
    y_norm = (pos.y - (helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2)
    vx_norm = vel.x * (VIEWPORT_W / SCALE / 2) / FPS
    vy_norm = vel.y * (VIEWPORT_H / SCALE / 2) / FPS
    angvel_norm = 20.0 * lander.angularVelocity / FPS

    # 1. Progress metric: Euclidean distance to pad (normalized)
    pad_dist = math.sqrt(x_norm**2 + y_norm**2)

    # 2. Stability metric: absolute angle deviation from 0
    angle_dev = abs(math.atan2(math.sin(angle), math.cos(angle)))

    # 3. Fuel usage: total power applied this step (0-2)
    # Use action to infer (since power not stored)
    m_power = 1.0 if action == 2 else 0.0
    s_power = 1.0 if action in [1, 3] else 0.0
    fuel = m_power + s_power

    # 4. Landing readiness: combined leg contact (0,1,2)
    leg_contact = int(legs[0].ground_contact) + int(legs[1].ground_contact)

    # 5. Vertical speed relative to pad – measure of crash risk
    vert_speed_mag = abs(vy_norm)

    # 6. Cross-step: angular velocity change (if previous state available)
    prev_angvel = pre.get('_prev_angvel', None)
    angvel_change = abs(angvel_norm - prev_angvel) if prev_angvel is not None else 0.0

    return {
        "pad_distance": pad_dist,
        "angle_deviation": angle_dev,
        "fuel_used": fuel,
        "leg_contact_sum": leg_contact,
        "vertical_speed_magnitude": vert_speed_mag,
        "angvel_change": angvel_change
    }


