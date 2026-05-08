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
    Reward function for lunar lander: land softly on helipad.
    """
    components = {}
    
    # Initialize cross-step tracking
    if not hasattr(self, '_step_count'):
        self._step_count = 0
        self._prev_vel_y = 0.0
        self._prev_angle = 0.0
    
    # Extract state components
    x_pos = state[0]           # [-1.5, 1.5] — horizontal position
    y_pos = state[1]           # [-1.5, 1.5] — vertical position (relative to helipad)
    vel_x = state[2]           # [-5.0, 5.0] — horizontal velocity
    vel_y = state[3]           # [-5.0, 5.0] — vertical velocity
    angle = state[4]           # [-π, π] — lander angle
    ang_vel = state[5]         # [-5.0, 5.0] — angular velocity
    left_contact = state[6]    # [0, 1] — left leg contact
    right_contact = state[7]   # [0, 1] — right leg contact
    
    # ========== 1. Progress toward helipad ==========
    # Reward being close to helipad (x=0, y=0 in normalized coords)
    # Max distance from helipad is sqrt(1.5^2 + 1.5^2) ≈ 2.12
    # Use exponential decay: reward 1.0 when at helipad, ~0.37 at distance 1
    helipad_distance = math.sqrt(x_pos**2 + y_pos**2)
    r_progress = math.exp(-3.0 * helipad_distance)  # 3.0 chosen so reward decays to ~0.05 at max distance
    
    # ========== 2. Velocity penalty (soft landing) ==========
    # Penalize high velocities, especially vertical
    # Target: vel_y ≈ 0 for landing, vel_x ≈ 0 for stability
    # Scale: typical velocities are [-2, 2], use quadratic penalty
    # Coefficient 0.3 gives ~1.2 max penalty at typical velocities, up from 0.1
    vel_penalty = 0.3 * (vel_x**2 + vel_y**2)  # 0.3 keeps this in ~[0, 2.4] range, strongly discourages fast descent
    
    # ========== 3. Angle stability ==========
    # Reward being upright (angle = 0)
    # Use angular difference (smallest signed angle)
    angle_diff = math.atan2(math.sin(angle), math.cos(angle))  # Normalize to [-π, π]
    r_angle = math.exp(-3.0 * angle_diff**2)  # 3.0 gives broader peak: 0.5 at ~0.48 rad, less dominant than before
    
    # ========== 4. Angular velocity penalty ==========
    # Penalize spinning
    ang_vel_penalty = 0.05 * ang_vel**2  # 0.05 keeps in ~[0, 1.25] range
    
    # ========== 5. Ground contact bonus ==========
    # Reward having both legs on ground (successful landing)
    # Only when velocity is low (to avoid rewarding crashes)
    both_legs = left_contact * right_contact
    is_stable_landing = both_legs and abs(vel_y) < 0.5 and abs(angle_diff) < 0.3
    r_landing = 2.0 if is_stable_landing else 0.0
    
    # ========== 6. Fuel efficiency penalty (small) ==========
    # Slight penalty for using engines to encourage efficient landing
    # m_power is [0, 1], s_power is [0, 1]
    r_efficiency = -0.01 * (m_power + s_power)  # Max penalty: -0.02 per step
    
    # ========== 7. Survival bonus ==========
    # Small positive reward each step to counterbalance penalties
    # This prevents the agent from wanting to terminate early
    r_survival = 0.1  # Consistent small positive reward
    
    # ========== 8. Descent smoothness ==========
    # Reward smooth deceleration by penalizing abrupt changes in vertical velocity
    # vel_y_change typically in [0, 10] range; exp(-2.0 * change) gives ~0.14 at change=1.0
    if self._step_count > 0:
        vel_y_change = abs(vel_y - self._prev_vel_y)
        r_descent_smoothness = math.exp(-2.0 * vel_y_change)  # 2.0 chosen so reward decays to ~0.14 at 1 m/s² change
    else:
        r_descent_smoothness = 0.0
    
    # ========== 9. Termination handling ==========
    if terminated:
        # Determine outcome based on observation state
        # Success: both legs on ground, low velocity, upright
        is_success = (left_contact > 0.5 and right_contact > 0.5 and 
                     abs(vel_y) < 0.5 and abs(angle_diff) < 0.3 and
                     helipad_distance < 0.5)
        
        if is_success:
            r_termination = 10.0  # Large bonus for successful landing
            components["_outcome"] = 1.0
        else:
            r_termination = -5.0  # Moderate penalty for crash
            components["_outcome"] = -1.0
    else:
        r_termination = 0.0
    
    # Combine all components
    components["r_progress"] = r_progress
    components["r_velocity"] = -vel_penalty
    components["r_angle"] = r_angle
    components["r_ang_vel"] = -ang_vel_penalty
    components["r_landing"] = r_landing
    components["r_efficiency"] = r_efficiency
    components["r_survival"] = r_survival
    components["r_descent_smoothness"] = r_descent_smoothness
    components["r_termination"] = r_termination
    
    # Compute total (excluding _outcome)
    total = sum(v for k, v in components.items() if k != "_outcome")
    
    # Track for next step
    self._prev_vel_y = vel_y
    self._prev_angle = angle
    self._step_count += 1
    
    return total, components

def metrics_fn(env, action) -> dict:
    """
    Task-level metrics independent of reward function.
    Measures landing quality, stability, and efficiency.
    """
    # Access unwrapped environment
    env = env.unwrapped
    
    # Get pre-step state for cross-step comparison
    pre = getattr(env, '_pre_step_state', {})
    
    # Current state from observation
    state = None  # We'll compute from physics bodies
    if hasattr(env, 'lander') and env.lander is not None:
        pos = env.lander.position
        vel = env.lander.linearVelocity
        angle = env.lander.angle
        ang_vel = env.lander.angularVelocity
        
        # Normalize positions
        x_norm = (pos.x - 0.8) / 0.8  # Approximate normalization
        y_norm = (pos.y - env.helipad_y - 0.1) / 0.6
        
        # Check leg contacts
        left_contact = 1.0 if env.legs[0].ground_contact else 0.0
        right_contact = 1.0 if env.legs[1].ground_contact else 0.0
        
        # ===== Metric 1: Landing Quality Score =====
        # Composite score: 0 (bad) to 1 (perfect landing)
        # Factors: both legs down, low velocity, upright, centered
        both_legs = left_contact * right_contact
        vel_score = math.exp(-5.0 * (vel.x**2 + vel.y**2))
        angle_score = math.exp(-10.0 * angle**2)
        pos_score = math.exp(-3.0 * (x_norm**2 + y_norm**2))
        
        landing_quality = (0.3 * both_legs + 0.3 * vel_score + 
                         0.2 * angle_score + 0.2 * pos_score)
        
        # ===== Metric 2: Stability Score =====
        # Measures how stable the lander is (inverse of oscillations)
        # Based on angular velocity and angle deviation
        stability = math.exp(-2.0 * (ang_vel**2 + angle**2))
        
        # ===== Metric 3: Fuel Efficiency =====
        # Measures how efficiently engines are used
        # Count engine usage this step
        m_power = 0.0
        s_power = 0.0
        if env.continuous:
            # For continuous, we'd need action values
            pass
        else:
            if action == 2:
                m_power = 1.0
            elif action in [1, 3]:
                s_power = 1.0
        
        fuel_used = m_power + s_power
        # Efficiency: 1 - normalized fuel usage (max 2 per step)
        fuel_efficiency = 1.0 - (fuel_used / 2.0)
        
        # ===== Metric 4: Vertical Descent Rate =====
        # For landing, should be negative (descending) but small magnitude
        # Too fast = crash risk, too slow = inefficient
        descent_rate = vel.y  # Negative means descending
        # Ideal: -0.2 to -0.5 (moderate descent)
        if descent_rate < 0:
            descent_quality = math.exp(-10.0 * (descent_rate + 0.3)**2)
        else:
            descent_quality = math.exp(-10.0 * descent_rate**2)
        
        # ===== Metric 5: Horizontal Drift =====
        # Should be minimal for landing
        horizontal_drift = abs(vel.x)
        drift_quality = math.exp(-5.0 * horizontal_drift)
        
        # ===== Metric 6: Approach Angle =====
        # Angle during descent (should be near 0 for upright landing)
        approach_quality = math.exp(-5.0 * angle**2)
        
        return {
            "landing_quality": landing_quality,
            "stability": stability,
            "fuel_efficiency": fuel_efficiency,
            "descent_quality": descent_quality,
            "drift_quality": drift_quality,
            "approach_quality": approach_quality,
        }
    
    return {
        "landing_quality": 0.0,
        "stability": 0.0,
        "fuel_efficiency": 0.0,
        "descent_quality": 0.0,
        "drift_quality": 0.0,
        "approach_quality": 0.0,
    }



