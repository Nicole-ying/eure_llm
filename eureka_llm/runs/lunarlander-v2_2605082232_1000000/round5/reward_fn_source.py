"""LLM-generated reward function (round 5).
Source: round5
"""

import math
import numpy as np

"""LLM-generated reward function (round 4).
Source: round4
"""

import math
import numpy as np

"""LLM-generated reward function (round 3, modified per analyst proposal).
Source: round3
Proposed changes:
- _outcome: failure penalty reduced from -1.0 to -0.5 (now further adjusted to -0.3)
- r_vertical_speed: coefficient increased from -0.05 to -0.1 (now softened to threshold-based)
- r_leg_contact: changed from binary to continuous near-pad bonus
"""

def compute_reward(self, state, m_power, s_power, terminated):
    """
    Primary signal: approaching the landing pad (minimizing distance).
    Shaping: distance-dependent survival bonus to encourage approach,
    plus thresholded penalty for fast vertical speed, and continuous leg-contact bonus near pad.
    """
    # state[0] = normalized x (target 0), state[1] = normalized y above pad (target 0)
    dist_to_pad = math.sqrt(state[0]**2 + state[1]**2)

    # Distance penalty: -0.1 * dist gives max ≈ -0.21 (at dist~2.1)
    r_distance = -0.1 * dist_to_pad  # coefficient justified: max distance ~2.1 => -0.21

    # Distance-dependent survival bonus: encourages approach while still rewarding staying alive
    # Coefficient 0.5 is scaled such that at dist=0 -> 0.5, dist=1 -> 0.0, for smooth gradient.
    # At mean distance 0.31, value ≈0.345, still sufficient to keep agent alive but now also pushes toward pad.
    r_alive = 0.5 * max(0, 1 - dist_to_pad)  # decays linearly from 0.5 to 0 at distance 1

    # Continuous leg-contact bonus near pad: incentivises ground contact while close to the target.
    # 0.1 * (contact_sum) * (1 - dist) yields max ~0.2 when both legs touch and dist=0.
    # Provides smooth gradient: contact strength multiplied by proximity.
    r_leg_contact = 0.1 * (state[6] + state[7]) * max(0, 1 - dist_to_pad)  # state[6], state[7] are leg-contact flags

    # Softened vertical-speed penalty: only penalizes downward speed > 1.0 m/s.
    # Threshold 1.0 m/s avoids choking slow controlled descent; coefficient 0.05 keeps penalty moderate.
    # Typical max penalty ≈ -0.05 * (max_speed - 1) ≤ -0.45 for speed up to 10 m/s.
    r_vertical_speed = -0.05 * max(0, abs(state[3]) - 1.0)  # state[3] is vertical speed (positive = downward)

    # Compute total BEFORE adding _outcome (evaluation-only)
    components = {
        "r_distance": r_distance,
        "r_alive": r_alive,
        "r_leg_contact": r_leg_contact,
        "r_vertical_speed": r_vertical_speed,
    }
    total = sum(components.values())

    # Terminal outcome for evaluation only (not added to total)
    if terminated:
        # Success: both legs grounded, near pad center, near-zero velocities, upright
        success = (state[6] >= 0.5 and state[7] >= 0.5 and
                   abs(state[0]) < 0.2 and abs(state[1]) < 0.1 and
                   abs(state[2]) < 0.5 and abs(state[3]) < 0.5 and
                   abs(state[4]) < 0.2)
        # Outcome magnitude reduced to -0.3 to avoid dominating total reward
        components["_outcome"] = 1.0 if success else -0.3  # not added to total
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


