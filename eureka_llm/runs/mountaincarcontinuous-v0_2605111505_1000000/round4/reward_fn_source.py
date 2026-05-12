"""LLM-generated reward function (round 4).
Source: round4
"""

import math
import numpy as np

"""LLM-generated reward function (round 4).
Source: round3 → round4: applied analyst proposal to fix reward_goal_alignment.
"""

import math
import numpy as np

def compute_reward(self, position, velocity, action, terminated):
    """
    Reward function for MountainCar (round 4).
    
    Physics: The car must build momentum by oscillating left and right.
    A weak engine cannot push directly up the hill.
    
    Changes from round 3 (analyst proposal):
    - Increased r_velocity coefficient from 0.5 → 2.0 to strongly incentivise
      movement in both directions (needed for building momentum).
    - Removed signed direction reward (r_direction = 0.0) because it penalised
      leftward movement and prevented oscillation.
    
    Signals:
    - r_progress: dense reward combining absolute velocity bonus (2.0 * |v|)
      and action penalty (-0.01 * |action|). The new velocity coefficient
      (~0.06 at current speeds) is clearly larger than the action penalty
      (~0.01), making movement beneficial.
    - r_proximity: distance penalty (-0.01 * max(0, 0.5 - position)).
      Provides a weak shaping gradient toward the goal.
      Coefficient -0.01 keeps the penalty small (< -0.017) so it never dominates.
    - r_goal: sparse goal bonus (1.0 only when position >= 0.5 and velocity >= 0).
    - _outcome: evaluation-only flag, not part of reward total.
    """
    # --- Dense components ---
    # Absolute velocity bonus: 2.0 * |v|, max ~0.14 at max speed (~0.07).
    # Increased from 0.5 to provide stronger movement incentive.
    r_velocity = 2.0 * abs(velocity)
    
    # Action penalty: -0.01 * |action|, max -0.01 at full throttle.
    # Discourages wasteful full‑throttle stalls that produce no momentum.
    action_penalty = -0.01 * abs(action[0])
    
    # Signed velocity reward: removed (set to 0.0) to allow free leftward
    # movement needed for building oscillation momentum.
    r_direction = 0.0
    
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


