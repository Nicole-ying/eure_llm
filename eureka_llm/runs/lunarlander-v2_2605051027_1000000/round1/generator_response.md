```python
"""LLM-generated reward function.
Source: round0
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
    # Reward for decreasing distance (max per-step improvement ~0.1, reward ~0.2)
    # Penalty increased from -0.5 to -1.0 per unit distance, max pad_dist ~1.5, so max penalty -1.5 (was -0.75)
    r_progress = -1.0 * pad_dist  # stronger negative penalty proportional to distance
    # Expanded close-range reward: bonus for pad_dist < 0.5 (was 0.1), with increased coefficient 1.0 (was 0.5)
    # At pad_dist=0, bonus = 0.5, total r_progress = 0.5; at pad_dist=0.5, bonus=0, r_progress = -0.5
    if pad_dist < 0.5:
        r_progress += 1.0 * (0.5 - pad_dist)  # max extra +0.5 when very close
    components['r_progress'] = r_progress

    # --- 2. Stability (angle and angular velocity) ---
    # Angle target: 0 (upright). Use smallest signed angular difference.
    ang_diff = math.atan2(math.sin(angle), math.cos(angle))  # wrap to [-pi, pi]
    # Penalize deviation from zero. Max absolute deviation ~pi, so reward range ~[-1, 0]
    r_stability_angle = -0.3 * abs(ang_diff)
    # Penalize angular velocity – keep it small (< ~0.5 rad/frame scaled)
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
    # Penalty reduced from -0.3 to -0.1 per unit of total power (max -0.2 per step instead of -0.6)
    r_efficiency = -0.1 * (m_power + s_power)  # now allows more aggressive use of thrusters
    components['r_efficiency'] = r_efficiency

    # --- 5. Termination handling ---
    r_termination = 0.0
    if terminated:
        # Determine success or failure
        # Success: lander not awake AND both legs in contact AND near pad AND low velocity
        success = (not self.lander.awake and leg_left and leg_right and
                   abs(ang_diff) < 0.3 and abs(vy_norm) < 0.1 and pad_dist < 0.2)
        if success:
            r_termination = 10.0  # large positive for successful landing
        else:
            r_termination = -15.0  # increased from -5 to -15 to strongly discourage termination without landing
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
    prev_action = pre.get('_prev_action', action)  # example

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
    # Use pre_state if available
    prev_angvel = pre.get('_prev_angvel', None)
    angvel_change = abs(angvel_norm - prev_angvel) if prev_angvel is not None else 0.0
    # Store current angvel for next step
    # (in real implementation you'd save this in env._prev_angvel)

    return {
        "pad_distance": pad_dist,
        "angle_deviation": angle_dev,
        "fuel_used": fuel,
        "leg_contact_sum": leg_contact,
        "vertical_speed_magnitude": vert_speed_mag,
        "angvel_change": angvel_change
    }
```