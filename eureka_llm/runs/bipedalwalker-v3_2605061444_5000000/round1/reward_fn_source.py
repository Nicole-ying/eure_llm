"""LLM-generated reward function (round 1).
Source: round1
"""

import math
import numpy as np

"""LLM-generated reward function.
Source: round0
"""

import math
import numpy as np

def compute_reward(self, action):
    """
    Reward for forward locomotion.
    Primary signal: forward velocity (positive x direction)
    """
    # Forward velocity from observation dim 2
    # This is already normalized: 0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS
    # Positive = moving right/forward
    forward_vel = self.hull.linearVelocity.x
    
    # Scale reward so typical values are ~[-0.5, 1.0] per step
    # Random policy mean was 0.0098 with std 0.052, so typical range is small
    # Scale up to make it meaningful
    r_forward = forward_vel * 2.0  # Scale factor: amplifies small velocity signals to meaningful range
    
    components = {"r_forward": r_forward}
    total = sum(components.values())
    
    # Terminal outcome for evaluation only
    if self.terminated:
        # Success: reached end of terrain
        if self.hull.position.x > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            components["_outcome"] = 1.0
        # Failure: fell or went backward
        else:
            components["_outcome"] = -1.0
    
    return total, components


def metrics_fn(env, action) -> dict:
    """
    Task-level metrics independent of reward.
    """
    pos = env.hull.position
    vel = env.hull.linearVelocity
    
    # Forward progress (cumulative x position)
    forward_progress = pos.x
    
    # Stability: how upright is the hull (angle close to 0)
    stability = 1.0 - abs(env.hull.angle) / math.pi
    
    # Speed: absolute forward speed
    speed = abs(vel.x)
    
    # Ground contact: how many feet are on ground
    left_contact = 1.0 if env.legs[1].ground_contact else 0.0
    right_contact = 1.0 if env.legs[3].ground_contact else 0.0
    feet_contact = left_contact + right_contact
    
    # Energy efficiency: total motor torque used
    total_torque = sum(
        abs(env.joints[i].motorSpeed * env.joints[i].maxMotorTorque)
        for i in range(4)
    )
    
    return {
        "forward_progress": forward_progress,
        "stability": stability,
        "speed": speed,
        "feet_contact": feet_contact,
        "total_torque": total_torque,
    }


