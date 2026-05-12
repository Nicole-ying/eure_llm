```python
"""Reward function for MountainCar (round 5).
Addresses reward_goal_alignment by removing the dominating _outcome penalty
and increasing the velocity coefficient to drive meaningful movement.
"""

import math
import numpy as np

def compute_reward(self, position, velocity, action, terminated):
    """
    Reward function for MountainCar (round 5).

    Physics: The car must build momentum by oscillating left and right.
    A weak engine cannot push directly up the hill.

    Changes from round 4 (analyst proposal):
    - _outcome set to 0.0 (removed from training signal) because it was
      dominating the reward (96.8% magnitude) and driving the agent to
      maximise episode length rather than make progress.
    - r_velocity coefficient increased from 2.0 → 5.0 to provide a strong
      incentive for movement now that the shaping signal is no longer drowned.
      At max speed (~0.07) this gives 0.35 per step (2.5× the previous 0.14).

    Components:
    - r_progress: dense reward combining velocity bonus (5.0 * |v|),
      action penalty (-0.01 * |action|), and proximity shaping.
    - r_goal: sparse goal bonus (1.0 when position >= 0.5 and velocity >= 0).
    - _outcome: evaluation-only flag, value 0.0 (no gradient impact).
    """
    # --- Dense components ---
    # Velocity bonus: 5.0 * abs(velocity). Coefficient 5.0 chosen to give
    # ~0.35/step at max speed (~0.07), making movement clearly more rewarding
    # than the action penalty (~0.01/step) and providing a strong gradient.
    r_velocity = 5.0 * abs(velocity)

    # Action penalty: -0.01 * |action|, max -0.01 at full throttle.
    # Discourages wasteful full‑throttle stalls that produce no momentum.
    action_penalty = -0.01 * abs(action[0])

    # Signed velocity reward: removed (set to 0.0) to allow free leftward
    # movement needed for building oscillation momentum.
    r_direction = 0.0

    # Proximity penalty: -0.01 * max(0, 0.5 - position).
    # Distance ranges from 0 (at goal) to ~1.7 (at left edge), so penalty
    # ranges from 0 to -0.017. Provides weak shaping even when velocity is zero.
    r_proximity = -0.01 * max(0.0, 0.5 - position)

    r_progress = r_velocity + action_penalty + r_direction + r_proximity

    components = {"r_progress": r_progress}

    # --- Sparse goal component ---
    if terminated:
        # Determine success using observation state only (no engine internals)
        success = (position >= 0.5) and (velocity >= 0.0)
        components["r_goal"] = 1.0 if success else 0.0
        # _outcome set to 0.0 – removed from training signal as per proposal.
        # This does not affect the reward total (it is only stored for evaluation).
        components["_outcome"] = 0.0
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
```