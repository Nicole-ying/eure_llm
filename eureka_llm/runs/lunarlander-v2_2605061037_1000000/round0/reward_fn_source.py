"""LLM-generated reward function.
Source: round0
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
    
    # Primary reward: proximity to landing pad (vertical closeness)
    # pos_y=0 means at pad level, pos_y>0 means above pad
    # Scale so max reward is ~0.1 per step when at pad
    r_proximity = -abs(pos_y) * 0.2  # penalty for being far from pad
    
    # Bonus for having both legs on ground (success signal)
    # This is the actual task goal
    both_legs = leg_left * leg_right  # 1.0 only when both legs contact
    r_landing = both_legs * 1.0  # small bonus per step when landed
    
    # Alive bonus - small reward each step to counter gravity penalty
    # The agent must act to stay alive, so give a tiny positive reward
    r_alive = 0.01  # small positive reward for surviving each step
    
    components["r_proximity"] = r_proximity
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
            components["_outcome"] = -1.0  # crash/failure penalty
        
        # Subtract outcome from total so it doesn't affect gradient
        total = total - components["_outcome"]
    
    return total, components


def metrics_fn(self, env, action) -> dict:
    """
    Task-level metrics independent of the reward function.
    """
    # Access state from env after step
    state = env.lander_state if hasattr(env, 'lander_state') else None
    
    # Get the observation from the environment
    # In LunarLander, we can read the lander's physics state directly
    if hasattr(env, 'lander') and env.lander is not None:
        pos = env.lander.position
        vel = env.lander.linearVelocity
        angle = env.lander.angle
        ang_vel = env.lander.angularVelocity
        
        # Distance to helipad (center of screen, bottom)
        helipad_x = 0  # center
        helipad_y = env.helipad_y if hasattr(env, 'helipad_y') else 0
        
        distance_to_pad = math.sqrt(
            (pos.x - helipad_x)**2 + 
            (pos.y - helipad_y)**2
        )
        
        # Speed magnitude
        speed = math.sqrt(vel.x**2 + vel.y**2)
        
        # Angular deviation from upright
        angle_error = abs(angle)
        
        # Ground contact status
        left_contact = 1.0 if (hasattr(env, 'legs') and 
                               env.legs[0].ground_contact) else 0.0
        right_contact = 1.0 if (hasattr(env, 'legs') and 
                                env.legs[1].ground_contact) else 0.0
        
        return {
            "distance_to_pad": distance_to_pad,
            "speed": speed,
            "angle_error": angle_error,
            "both_legs_contact": left_contact * right_contact,
            "is_alive": 1.0 if env.lander.awake else 0.0,
        }
    else:
        return {
            "distance_to_pad": 0.0,
            "speed": 0.0,
            "angle_error": 0.0,
            "both_legs_contact": 0.0,
            "is_alive": 1.0,
        }

