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
    # Reward of -0.3 per unit distance gives reasonable scale (~[-0.63, 0])
    # Reduced from 0.5 to 0.3 to allow more exploration before landing
    dist = math.sqrt(state[0]**2 + state[1]**2)
    r_distance = -dist * 0.3  # Scaled so typical values are ~[-0.3, 0]
    components["r_distance"] = r_distance
    
    # Small alive bonus to encourage survival during exploration
    # The agent needs to learn to stay alive before it can land
    # 0.05 per step = 3.15 per episode max (~63 steps), meaningful incentive
    # Increased from 0.01 to 0.05 to give stronger survival motivation
    r_alive = 0.05
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
    Measures actual landing performance and stability.
    """
    # Access the underlying environment
    unwrapped = env.unwrapped
    
    # Get the lander body and state
    lander = unwrapped.lander
    state = unwrapped.state if hasattr(unwrapped, 'state') else None
    
    # Get pre-step state if available for cross-step metrics
    pre_state = getattr(unwrapped, '_pre_step_state', {})
    
    # Distance to helipad center (in normalized coordinates)
    # State[0] and state[1] give normalized position relative to helipad
    if state is not None:
        dx = state[0]
        dy = state[1]
        distance_to_pad = math.sqrt(dx*dx + dy*dy)
    else:
        # Fallback: use physics body position
        helipad_x = 0.0
        helipad_y = getattr(unwrapped, 'helipad_y', 0.0) + 0.1
        dx = lander.position.x - helipad_x
        dy = lander.position.y - helipad_y
        distance_to_pad = math.sqrt(dx*dx + dy*dy)
    
    # Speed magnitude from state (normalized velocities)
    if state is not None:
        speed = math.sqrt(state[2]**2 + state[3]**2)
    else:
        vel = lander.linearVelocity
        speed = math.sqrt(vel.x**2 + vel.y**2)
    
    # Angle from vertical (absolute radians)
    angle_from_vertical = abs(lander.angle)
    
    # Ground contact status using leg sensors
    # Use state[6] and state[7] if available, otherwise check physics contacts
    if state is not None:
        left_contact = 1.0 if state[6] > 0.5 else 0.0
        right_contact = 1.0 if state[7] > 0.5 else 0.0
    else:
        left_contact = 1.0 if getattr(unwrapped.legs[0], 'ground_contact', False) else 0.0
        right_contact = 1.0 if getattr(unwrapped.legs[1], 'ground_contact', False) else 0.0
    
    both_legs_down = left_contact * right_contact  # 1 if both, 0 otherwise
    
    # Altitude above helipad (normalized)
    if state is not None:
        altitude = state[1]  # Positive = above helipad
    else:
        helipad_y = getattr(unwrapped, 'helipad_y', 0.0)
        altitude = lander.position.y - helipad_y
    
    # Angular velocity magnitude (stability metric)
    angular_velocity = abs(lander.angularVelocity)
    
    # Engine power usage (efficiency metric)
    # Use the last action to estimate fuel usage
    main_engine = 0.0
    side_engine = 0.0
    if hasattr(unwrapped, 'continuous') and unwrapped.continuous:
        if len(action) > 0:
            main_engine = max(0.0, action[0])  # Main engine throttle
        if len(action) > 1:
            side_engine = abs(action[1])  # Side engine throttle
    else:
        if action == 2:
            main_engine = 1.0
        elif action == 1:
            side_engine = 1.0
        elif action == 3:
            side_engine = 1.0
    
    total_thrust = main_engine + side_engine
    
    return {
        "distance_to_pad": distance_to_pad,
        "speed": speed,
        "angle_from_vertical": angle_from_vertical,
        "both_legs_down": both_legs_down,
        "altitude": altitude,
        "angular_velocity": angular_velocity,
        "total_thrust": total_thrust,
    }


