"""LLM-generated reward function (round 3).
Source: round3
"""

import math
import numpy as np

"""LLM-generated reward function (round 3).
Source: round3 - Added direction bias and proximity shaping to fix reward_goal_alignment.
"""

import math
import numpy as np

def compute_reward(self, position, velocity, action, terminated):
    """
    Reward function for MountainCar.
    
    Physics: The car must build momentum by oscillating left and right.
    A weak engine cannot push directly up the hill.
    
    Signals:
    - r_progress: dense reward combining absolute velocity bonus (0.5 * |v|)
      and action penalty (-0.01 * |action|). Encourages movement and penalises
      wasteful throttle.
    - r_direction: signed velocity reward (0.3 * velocity). Gives positive
      reward for moving right (toward goal) and slight penalty for moving left,
      breaking symmetry. Coefficient 0.3 is smaller than the absolute velocity
      coefficient (0.5) to keep leftward movement possible for building momentum.
    - r_proximity: distance penalty (-0.01 * max(0, 0.5 - position)). Provides
      a continuous negative gradient toward the goal. Coefficient -0.01 ensures
      the penalty is smaller than velocity rewards (max -0.017 at farthest point)
      so it does not dominate.
    - r_goal: sparse goal bonus (1.0 only when position >= 0.5 and velocity >= 0).
    - _outcome: evaluation-only flag, not part of the reward total.
    """
    # --- Dense components ---
    # Absolute velocity bonus: 0.5 * |v|, max ~0.035 at max speed (~0.07).
    # Chosen to be substantially larger than action penalty so movement is preferred.
    r_velocity = 0.5 * abs(velocity)
    
    # Action penalty: -0.01 * |action|, max -0.01 at full throttle.
    # Discourages wasteful full‑throttle stalls that produce no momentum.
    action_penalty = -0.01 * abs(action[0])
    
    # Signed velocity reward: 0.3 * velocity, max ~0.021 rightward, ~-0.021 leftward.
    # Introduces directional bias to align reward with goal direction.
    r_direction = 0.3 * velocity
    
    # Proximity penalty: -0.01 * max(0, 0.5 - position).
    # Distance ranges from 0 (at goal) to ~1.7 (at left edge), so penalty
    # ranges from 0 to -0.017. Provides shaping signal even when velocity is zero.
    r_proximity = -0.01 * max(0.0, 0.5 - position)
    
    r_progress = r_velocity + action_penalty + r_direction + r_proximity
    
    components = {"r_progress": r_progress}
    
    # --- Sparse goal component ---
    if terminated:
        # Determine success using observation state only (no engine internals)
        success = (position >= 0.5) and (velocity >= 0.0)
        components["r_goal"] = 1.0 if success else 0.0
        components["_outcome"] = 1.0 if success else -1.0
        total = r_progress + components["r_goal"]   # reward total before _outcome
    else:
        components["r_goal"] = 0.0
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


