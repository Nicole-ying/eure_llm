```python
"""LLM-generated reward function.
Source: round1 (modified per Analyst Proposal)
"""

import math
import numpy as np


def compute_reward(self, state, m_power, s_power, terminated):
    """
    Primary signal: approaching the landing pad (minimizing distance).
    Shaping: survival bonus contingent on closeness, bonus for both legs touching ground.
    No velocity/angle penalties per round 0 sparsity rule.
    """
    # state[0] = normalized x (target 0), state[1] = normalized y above pad (target 0)
    dist_to_pad = math.sqrt(state[0]**2 + state[1]**2)

    # Reward for being close to the pad: scale so maximum ≈ -0.2
    r_distance = -0.1 * dist_to_pad   # coefficient justified: max distance ~2.1 => -0.21

    # Alive reward now depends on proximity: far away gives zero, near gives full bonus.
    # Original constant 0.05 is kept as max; multiplier 1.0 - dist_to_pad decreases linearly.
    r_alive = 0.05 * max(0.0, 1.0 - dist_to_pad)   # 0.05 justified as original small constant

    # Bonus for both legs in contact (indicates landing attempt)
    r_leg_contact = 1.0 if (state[6] >= 0.5 and state[7] >= 0.5) else 0.0

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
        components["_outcome"] = 1.0 if success else -1.0
    # No outcome key on non‑terminal steps to avoid affecting gradient

    return total, components


def metrics_fn(env, action) -> dict:
    """
    Task-level metrics (reward-agnostic) for monitoring agent performance.
    """
    # Access lander body and helipad info directly from the environment
    lander = env.lander
    helipad_y = env.helipad_y

    # Normalized position (same as observation dims 0,1)
    norm_x = (lander.position.x - env.VIEWPORT_W / env.SCALE / 2) / (env.VIEWPORT_W / env.SCALE / 2)
    norm_y = (lander.position.y - (helipad_y + env.LEG_DOWN / env.SCALE)) / (env.VIEWPORT_H / env.SCALE / 2)

    # Euclidean distance to landing pad (in normalized coordinates)
    distance_to_pad = math.sqrt(norm_x**2 + norm_y**2)

    # Vertical speed (observation dim 3, positive = downward)
    vertical_speed = lander.linearVelocity.y  # raw PyBox2D velocity
    # Normalize to same scale as observation: vel.y * (VIEWPORT_H/SCALE/2) / FPS
    scaled_vertical_speed = vertical_speed * (env.VIEWPORT_H / env.SCALE / 2) / env.FPS

    # Angle (radians)
    angle = lander.angle

    # Leg contact flags
    left_contact = 1.0 if env.legs[0].ground_contact else 0.0
    right_contact = 1.0 if env.legs[1].ground_contact else 0.0

    # Survival: lander is awake and not out of bounds (not terminated steps)
    # We use the lander's awake flag and whether it has been terminated
    alive = 1.0 if lander.awake and abs(norm_x) < 1.0 else 0.0

    return {
        "distance_to_pad": distance_to_pad,
        "vertical_speed": scaled_vertical_speed,
        "angle": angle,
        "leg_contact": min(left_contact + right_contact, 1.0),  # 1 if at least one leg contact
        "alive": alive,
    }
```