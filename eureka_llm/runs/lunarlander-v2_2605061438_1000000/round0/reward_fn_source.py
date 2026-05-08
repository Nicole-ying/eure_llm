"""LLM-generated reward function.
Source: round0
"""

import math
import numpy as np

def compute_reward(self, state, m_power, s_power, terminated):
    """
    Reward function for Lunar Lander.
    
    State dimensions:
    0: Normalized x-position (center=0, edges=±1)
    1: Normalized y-position relative to helipad (helipad=0, higher=positive)
    2: Normalized x-velocity
    3: Normalized y-velocity
    4: Angle (radians)
    5: Angular velocity
    6: Left leg ground contact (0/1)
    7: Right leg ground contact (0/1)
    """
    components = {}
    
    # Primary signal: distance to landing pad
    # We want the lander near (0, 0) in normalized coordinates
    # Scale: state[0] and state[1] are in [-1.5, 1.5], so max distance ~2.1
    # Reward of -1 per unit distance gives reasonable scale
    dist = math.sqrt(state[0]**2 + state[1]**2)
    r_distance = -dist * 0.5  # Scaled so typical values are ~[-0.5, 0]
    components["r_distance"] = r_distance
    
    # Small alive bonus to encourage survival during exploration
    # The agent needs to learn to stay alive before it can land
    # 0.01 per step = 10 per episode max, negligible compared to termination penalty
    r_alive = 0.01
    components["r_alive"] = r_alive
    
    total = sum(components.values())
    
    # Terminal outcome signal — for evaluation only
    if terminated:
        # Success: both legs on ground, near helipad, low velocity
        # state[6] and state[7] are leg contact flags
        on_ground = (state[6] > 0.5 and state[7] > 0.5)
        near_target = dist < 0.3  # Close to helipad
        low_speed = abs(state[2]) < 0.5 and abs(state[3]) < 0.5  # Slow velocity
        upright = abs(state[4]) < 0.5  # Nearly vertical
        
        if on_ground and near_target and low_speed and upright:
            components["_outcome"] = 1.0  # Successful landing
        else:
            components["_outcome"] = -1.0  # Crash/failure
        
        # No modification to total — _outcome is eval-only
    
    return total, components


def metrics_fn(env, action) -> dict:
    """
    Task-level metrics independent of reward function.
    """
    # Access state from environment
    state = env.lander  # We need to read from the environment
    
    # Read the observation from the last step
    # The state is stored in the environment
    pos = env.lander.position
    vel = env.lander.linearVelocity
    
    # Distance to helipad center (in world coordinates)
    helipad_x = 0  # Helipad is at center
    helipad_y = env.helipad_y + 0.1  # Top of helipad
    
    dx = pos.x - helipad_x
    dy = pos.y - helipad_y
    distance_to_pad = math.sqrt(dx*dx + dy*dy)
    
    # Speed magnitude
    speed = math.sqrt(vel.x**2 + vel.y**2)
    
    # Angle from vertical (absolute)
    angle_from_vertical = abs(env.lander.angle)
    
    # Ground contact status
    left_contact = 1.0 if env.legs[0].ground_contact else 0.0
    right_contact = 1.0 if env.legs[1].ground_contact else 0.0
    both_legs_down = left_contact * right_contact  # 1 if both, 0 otherwise
    
    # Is the lander awake? (not sleeping = still in motion)
    awake = 1.0 if env.lander.awake else 0.0
    
    return {
        "distance_to_pad": distance_to_pad,
        "speed": speed,
        "angle_from_vertical": angle_from_vertical,
        "both_legs_down": both_legs_down,
        "awake": awake,
        "altitude": pos.y - env.helipad_y,  # Height above helipad
    }

