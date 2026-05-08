```python
"""LLM-generated reward function (round 2).
Source: round2
"""

import math
import numpy as np

def compute_reward(self, state, m_power, s_power, terminated):
    components = {}
    
    # Extract state components with descriptive names
    pos_x = state[0]      # horizontal position, normalized [-1, 1]
    pos_y = state[1]      # vertical position relative to helipad
    vel_x = state[2]      # horizontal velocity
    vel_y = state[3]      # vertical velocity
    angle = state[4]      # angular position
    ang_vel = state[5]    # angular velocity
    leg_left = state[6]   # left leg ground contact (0 or 1)
    leg_right = state[7]  # right leg ground contact (0 or 1)
    
    # Shaped reward: positive exponential rewards for proximity and low speed
    # 0.3: max reward ~0.3 when centered (pos_x=0), decays to ~0.18 at edge (pos_x=1)
    # 0.3: max reward ~0.3 when at pad height (pos_y=0), decays to ~0.18 far above
    # 0.2: max reward ~0.2 when stationary, decays to ~0.09 at speed 2
    # -0.05: max penalty ~0.15 when inverted (angle=pi), ~0.025 at 30 degrees
    r_shaped = 0.3 * math.exp(-0.5 * abs(pos_x)) + 0.3 * math.exp(-0.5 * abs(pos_y)) + 0.2 * math.exp(-0.2 * (vel_x**2 + vel_y**2)) - 0.05 * abs(angle)
    
    # Bonus for having both legs on ground (success signal)
    # This is the actual task goal
    both_legs = leg_left * leg_right  # 1.0 only when both legs contact
    r_landing = both_legs * 1.0  # small bonus per step when landed
    
    # Alive bonus proportional to how upright the agent is
    # 0.1: max bonus when upright (angle=0), 0 when inverted (angle=pi)
    # Increased from 0.02 to provide stronger survival incentive
    r_alive = 0.1 * (1.0 - abs(angle) / math.pi)  # alive bonus proportional to uprightness
    
    components["r_shaped"] = r_shaped
    components["r_landing"] = r_landing
    components["r_alive"] = r_alive
    
    total = sum(components.values())
    
    # Terminal outcome signal (for evaluation only)
    if terminated:
        # Success: both legs on ground, roughly upright, low velocity
        is_upright = abs(angle) < 0.5  # roughly vertical
        is_gentle = abs(vel_y) < 0.5   # low vertical speed
        is_centered = abs(pos_x) < 0.3  # on the pad
        
        if both_legs and is_upright and is_gentle and is_centered:
            components["_outcome"] = 10.0  # successful landing bonus
        else:
            components["_outcome"] = -0.1  # crash/failure penalty (reduced 10x to avoid dominating)
        
        # Subtract outcome from total so it doesn't affect gradient
        total = total - components["_outcome"]
    
    return total, components


def metrics_fn(self, env, action) -> dict:
    """
    Task-level metrics independent of the reward function.
    Measures actual landing performance and stability.
    """
    # Access environment physics safely
    unwrapped = env.unwrapped
    
    # Get pre-step state if available for cross-step metrics
    pre_state = getattr(env, '_pre_step_state', {})
    
    # Access lander physics state directly
    if hasattr(unwrapped, 'lander') and unwrapped.lander is not None:
        lander = unwrapped.lander
        pos = lander.position
        vel = lander.linearVelocity
        angle = lander.angle
        
        # Distance to helipad (center of screen, bottom)
        helipad_x = 0.0  # center
        helipad_y = getattr(unwrapped, 'helipad_y', 0.0)
        
        distance_to_pad = math.sqrt(
            (pos.x - helipad_x)**2 + 
            (pos.y - helipad_y)**2
        )
        
        # Speed magnitude
        speed = math.sqrt(vel.x**2 + vel.y**2)
        
        # Vertical speed (positive = moving up, negative = descending)
        vertical_speed = vel.y
        
        # Angular deviation from upright (radians)
        angle_error = abs(angle)
        
        # Ground contact status - use observation-based check
        # Check if both legs are on ground via observation state
        obs = getattr(unwrapped, 'last_observation', None)
        if obs is not None:
            leg_left = obs[6]
            leg_right = obs[7]
            both_legs_contact = 1.0 if (leg_left > 0.5 and leg_right > 0.5) else 0.0
        else:
            both_legs_contact = 0.0
        
        # Stability metric: combination of angular velocity and angle
        # Lower is more stable
        ang_vel = lander.angularVelocity
        stability = abs(angle) + abs(ang_vel) * 0.1
        
        return {
            "distance_to_pad": distance_to_pad,
            "speed": speed,
            "vertical_speed": vertical_speed,
            "angle_error": angle_error,
            "both_legs_contact": both_legs_contact,
            "stability": stability,
        }
    else:
        return {
            "distance_to_pad": 0.0,
            "speed": 0.0,
            "vertical_speed": 0.0,
            "angle_error": 0.0,
            "both_legs_contact": 0.0,
            "stability": 0.0,
        }
```