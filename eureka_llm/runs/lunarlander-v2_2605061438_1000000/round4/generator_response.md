```python
"""LLM-generated reward function (round 3).
Source: round3
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
    # Reduced from 0.1 to 0.05 to prevent overwhelming the alive bonus
    # At current distance (~12), penalty is -0.6 per step instead of -1.2
    # This gives the agent room to explore without being heavily penalized
    dist = math.sqrt(state[0]**2 + state[1]**2)
    r_distance = -dist * 0.05  # Gentler gradient that still incentivizes approach
    components["r_distance"] = r_distance
    
    # Speed penalty to encourage throttle modulation
    # The agent currently applies full thrust (1.0) constantly with zero variance
    # because there is no penalty for high speed. Adding this creates a direct
    # incentive to reduce throttle and slow down, enabling controlled descent.
    # At current speed ~8.3, penalty is -0.415, comparable to other components
    speed = math.sqrt(state[2]**2 + state[3]**2)
    r_speed = -speed * 0.05  # Penalize high velocity to encourage throttle modulation
    components["r_speed"] = r_speed
    
    # Alive bonus to encourage survival during exploration
    # Increased 10x from 0.05 to 0.5 to create a strong incentive for the agent
    # to stay alive longer, which is the first prerequisite for learning to land.
    # At 0.5 per step, the agent gets +39 per episode (78 steps) instead of +3.9,
    # making survival the dominant per-step signal.
    r_alive = 0.5  # Strong incentive to stay alive
    components["r_alive"] = r_alive
    
    # Dense per-step shaping reward for continuous gradient toward landing
    # Replaces sparse terminal bonus (+10 only on landing) with dense per-step
    # shaping that rewards continuous progress toward the pad (decreasing distance,
    # descending altitude, getting legs down). This provides a smooth gradient
    # every step instead of a feast-or-famine signal, preventing the oscillation
    # between landing and falling.
    if not hasattr(self, '_prev_dist'):
        self._prev_dist = dist
        self._prev_altitude = state[1]
    
    # Reward decreasing distance to pad (moving toward target)
    # Coefficient 2.0: at max delta ~2.1 per step, gives ~4.2 reward, comparable
    # to other components (r_alive=0.5, r_distance=-0.105 at max dist)
    dist_delta = self._prev_dist - dist
    r_progress_dist = dist_delta * 2.0
    
    # Reward decreasing altitude (descending toward ground)
    # Coefficient 1.0: at max delta ~1.5 per step, gives ~1.5 reward, keeping
    # progress rewards within same order of magnitude as other components
    altitude = state[1]
    altitude_delta = self._prev_altitude - altitude
    r_progress_alt = altitude_delta * 1.0
    
    # Small bonus for being close to ground with both legs down
    # 2.0 is modest enough to not dominate, but provides a clear signal when
    # the agent is in the final landing configuration
    near_ground = altitude < 0.1
    legs_down = (state[6] > 0.5 and state[7] > 0.5)
    r_near_landing = 2.0 if (near_ground and legs_down) else 0.0
    
    r_progress = r_progress_dist + r_progress_alt + r_near_landing
    
    self._prev_dist = dist
    self._prev_altitude = altitude
    
    components["r_progress"] = r_progress
    
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
```