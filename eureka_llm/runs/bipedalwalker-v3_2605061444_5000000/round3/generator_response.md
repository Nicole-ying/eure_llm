```python
"""LLM-generated reward function (round 3).
Source: round3
"""

import math
import numpy as np

def compute_reward(self, action) -> tuple[float, dict]:
    """
    Reward for forward locomotion.
    Primary signal: forward velocity (positive x direction)
    Secondary signal: stability (upright posture)
    Terminal outcome: task completion or failure
    """
    # Forward velocity from observation dim 2
    # This is already normalized: 0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS
    # Positive = moving right/forward
    forward_vel = self.hull.linearVelocity.x
    
    # Scale reward so typical values are ~[-0.5, 1.0] per step
    # Random policy mean was 0.0098 with std 0.052, so typical range is small
    # Scale up to make it meaningful
    r_forward = forward_vel * 2.0  # Scale factor: amplifies small velocity signals to meaningful range
    
    # Stability bonus: reward upright posture (angle close to 0)
    # hull.angle is in radians, normalize to [0,1] where 1 = perfectly upright
    uprightness = 1.0 - abs(self.hull.angle) / math.pi
    # Reduced coefficient to allow more aggressive forward movement while maintaining basic stability
    # The agent already maintains perfect stability (fall_rate=0.0), so the high coefficient is no longer needed
    r_stability = 0.2 * uprightness  # Reduced coefficient: 0.2 allows more aggressive movement while still providing meaningful stability signal
    
    components = {"r_forward": r_forward, "r_stability": r_stability}
    
    # Terminal outcome - computed BEFORE total sum so it affects training
    if self.terminated:
        # Success: reached end of terrain
        if self.hull.position.x > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            components["_outcome"] = 1.0
        # Failure: fell or went backward
        else:
            components["_outcome"] = -1.0
    
    total = sum(components.values())
    return total, components


def metrics_fn(env, action) -> dict:
    """
    Task-level metrics independent of reward.
    Measures actual task performance: progress, stability, efficiency, and safety.
    """
    pos = env.hull.position
    vel = env.hull.linearVelocity
    
    # Forward progress (cumulative x position) - measures how far the agent has traveled
    forward_progress = pos.x
    
    # Stability: how upright is the hull (angle close to 0)
    # 1.0 = perfectly upright, 0.0 = horizontal/fallen
    stability = 1.0 - abs(env.hull.angle) / math.pi
    
    # Speed: absolute forward speed in m/s
    speed = abs(vel.x)
    
    # Ground contact: how many feet are on ground (0-2)
    left_contact = 1.0 if env.legs[1].ground_contact else 0.0
    right_contact = 1.0 if env.legs[3].ground_contact else 0.0
    feet_contact = left_contact + right_contact
    
    # Energy efficiency: total motor torque used (normalized by max possible)
    total_torque = sum(
        abs(env.joints[i].motorSpeed * env.joints[i].maxMotorTorque)
        for i in range(4)
    )
    
    # Vertical oscillation: measures how much the hull bounces up and down
    # Lower values indicate smoother locomotion
    vertical_velocity = abs(vel.y)
    
    return {
        "forward_progress": forward_progress,
        "stability": stability,
        "speed": speed,
        "feet_contact": feet_contact,
        "total_torque": total_torque,
        "vertical_oscillation": vertical_velocity
    }
```