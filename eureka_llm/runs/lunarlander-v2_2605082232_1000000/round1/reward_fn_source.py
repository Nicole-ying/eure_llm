"""LLM-generated reward function (round 1).
Source: round1
"""

import math
import numpy as np

import math
import numpy as np

def compute_reward(self, state, m_power, s_power, terminated):
    """
    Primary signal: approaching the landing pad (minimizing distance).
    Shaping: survival bonus that decays with distance, bonus for both legs touching ground.
    """
    # state[0] = normalized x (target 0), state[1] = normalized y above pad (target 0)
    dist_to_pad = math.sqrt(state[0]**2 + state[1]**2)

    # Reward for being close to the pad: scale so maximum ≈ -0.2
    r_distance = -0.1 * dist_to_pad   # coefficient justified: max distance ~2.1 => -0.21

    # Survival bonus shaped by proximity: far away gives zero, near gives full +0.05
    # (0.05 * max(0, 1 - distance)) ensures hover at distance > 1 yields no alive reward
    r_alive = 0.05 * max(0.0, 1.0 - dist_to_pad)   # 0.05 justified: max alive reward when at pad

    # Bonus for both legs in contact (indicates landing attempt)
    r_leg_contact = 1.0 if (state[6] >= 0.5 and state[7] >= 0.5) else 0.0

    # Compute total BEFORE adding _outcome
    components = {
        "r_distance": r_distance,
        "r_alive": r_alive,
        "r_leg_contact": r_leg_contact,
    }
    total = sum(components.values())

    # Terminal outcome for evaluation only (not added to total)
    if terminated:
        # Success: both legs grounded, near pad center, near-zero velocities, upright
        success = (state[6] >= 0.5 and state[7] >= 0.5 and
                   abs(state[0]) < 0.2 and abs(state[1]) < 0.1 and
                   abs(state[2]) < 0.5 and abs(state[3]) < 0.5 and
                   abs(state[4]) < 0.2)
        components["_outcome"] = 1.0 if success else -1.0  # not added to total
    # else: do NOT add _outcome key at all

    return total, components


def metrics_fn(env, action) -> dict:
    """
    Task-level metrics (reward-agnostic) for monitoring agent performance.
    Accesses env state directly.
    """
    lander = env.lander
    helipad_y = env.helipad_y
    x_scale = env.VIEWPORT_W / env.SCALE / 2
    y_scale = env.VIEWPORT_H / env.SCALE / 2

    # Normalized position (same as observation dims 0,1)
    norm_x = (lander.position.x - x_scale) / x_scale
    norm_y = (lander.position.y - (helipad_y + env.LEG_DOWN / env.SCALE)) / y_scale

    # Euclidean distance to landing pad (in normalized coordinates)
    distance_to_pad = math.sqrt(norm_x**2 + norm_y**2)

    # Vertical speed from physics (positive = downward)
    vertical_speed = lander.linearVelocity.y

    # Angle (radians)
    angle = lander.angle

    # Angular velocity
    angular_velocity = lander.angularVelocity

    # Leg contact flags
    left_contact = 1.0 if env.legs[0].ground_contact else 0.0
    right_contact = 1.0 if env.legs[1].ground_contact else 0.0

    # Horizontal speed (absolute value)
    horizontal_speed = abs(lander.linearVelocity.x)

    return {
        "distance_to_pad": distance_to_pad,
        "vertical_speed": vertical_speed,
        "angle": angle,
        "angular_velocity": angular_velocity,
        "leg_contact": min(left_contact + right_contact, 1.0),  # 1 if at least one leg contacts
        "horizontal_speed": horizontal_speed,
    }


