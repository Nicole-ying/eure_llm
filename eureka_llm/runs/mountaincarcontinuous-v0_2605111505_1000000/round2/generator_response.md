```python
"""LLM-generated reward function (round 2).
Source: round2 - Fixed constant r_goal exploit, added velocity incentive and action penalty.
"""

import math
import numpy as np

def compute_reward(self, position, velocity, action, terminated):
    """
    Reward function for MountainCar.
    
    Physics: The car must build momentum by oscillating left and right.
    A weak engine cannot push directly up the hill.
    
    Signals:
    - r_progress: velocity bonus (0.5 * |velocity|) plus action penalty
      (-0.01 * |action|). Removes the constant positional baseline that
      previously masked the need to move.
    - r_goal: always present; 1.0 only when the car reaches the goal
      (position >= 0.5 and velocity >= 0), else 0.0.
    - _outcome: evaluation-only flag, not part of the reward total.
    """
    # Velocity incentive: 0.5 * |velocity|, max ~0.035 at max speed (~0.07)
    # Coefficient 0.5 is chosen to be substantially larger than the action penalty
    # so the agent prefers moving over standing still.
    r_velocity = 0.5 * abs(velocity)
    # Action penalty: -0.01 * |action|, max -0.01 at full throttle (action=1).
    # Discourages wasteful full‑throttle stalls that produce no momentum.
    action_penalty = -0.01 * abs(action[0])
    r_progress = r_velocity + action_penalty   # dense reward (no positional baseline)
    
    components = {"r_progress": r_progress}
    
    if terminated:
        # Determine success using observation state only (no engine internals)
        success = (position >= 0.5) and (velocity >= 0.0)
        components["r_goal"] = 1.0 if success else 0.0
        components["_outcome"] = 1.0 if success else -1.0
        total = r_progress + components["r_goal"]   # reward total before _outcome
    else:
        components["r_goal"] = 0.0   # always present; zero when goal not reached
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
```