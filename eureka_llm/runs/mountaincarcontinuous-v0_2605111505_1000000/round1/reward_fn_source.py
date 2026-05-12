"""LLM-generated reward function (round 1).
Source: round1
"""

import math
import numpy as np

"""LLM-generated reward function.
Source: round1 - Fixed termination exploitation and added velocity incentive.
"""

import math
import numpy as np

def compute_reward(self, position, velocity, action, terminated):
    """
    Reward function for MountainCar.
    
    Physics: The car must build momentum by oscillating left and right.
    A weak engine cannot push directly up the hill.
    
    Signals:
    - r_progress: normalized position (higher = closer to goal) plus a small
      velocity bonus to encourage movement and break the stationary deadlock.
      Velocity bonus coefficient 0.2 is small (max ~0.014) so it does not
      override the position gradient.
    - r_goal: terminal bonus only when the car actually reaches the goal
      (position >= 0.5 and velocity >= 0). This prevents exploitation of
      timeout terminations.
    - _outcome: evaluation-only flag, not part of the reward total.
    """
    # Normalize position to [0,1] range (min=-1.2, max=0.6 → span=1.8)
    r_position = (position - (-1.2)) / 1.8   # ~0.39 at start, ~1.0 at goal
    # Velocity bonus: 0.2 * |velocity|, max ~0.014 at max speed (~0.07)
    r_velocity = 0.2 * abs(velocity)         # encourage momentum building
    r_progress = r_position + r_velocity      # combined dense signal
    
    components = {"r_progress": r_progress}
    
    if terminated:
        # Determine success using observation state only (no engine internals)
        success = (position >= 0.5) and (velocity >= 0.0)
        if success:
            r_goal = 1.0
            components["r_goal"] = 1.0
            components["_outcome"] = 1.0          # success flag for evaluation
            total = r_progress + r_goal
        else:
            # Failure or timeout – no extra bonus, but signal failure for evaluation
            components["_outcome"] = -1.0          # failure flag for evaluation
            total = r_progress
    else:
        total = r_progress
    
    return float(total), components


def metrics_fn(env, action) -> dict:
    """
    Task‑level metrics independent of the reward.
    All attributes are accessed with getattr for safety.
    """
    # Current observation
    state = getattr(env, 'state', [0.0, 0.0])
    position = state[0] if len(state) > 0 else 0.0
    velocity = state[1] if len(state) > 1 else 0.0
    
    # Goal constants (from the environment)
    goal_position = 0.5
    goal_velocity = 0.0
    
    # Distance remaining to the goal (clipped at 0)
    distance_to_goal = max(0.0, goal_position - position)
    
    # Speed magnitude
    abs_velocity = abs(velocity)
    
    # Action magnitude (force applied)
    action_force = abs(action[0])
    
    # Did the car reach the goal this step?
    reached_goal = 1.0 if (position >= goal_position and velocity >= goal_velocity) else 0.0
    
    # Normalised heading: +1 if moving right, -1 if left, 0 if stopped
    heading = 0.0
    if abs_velocity > 1e-6:
        heading = velocity / abs_velocity
    
    return {
        "dist_to_goal": distance_to_goal,
        "velocity_mag": abs_velocity,
        "action_force": action_force,
        "reached_goal": reached_goal,
        "heading": heading,
    }



