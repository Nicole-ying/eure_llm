"""LLM-generated reward function (round 2).
Source: round2
"""

import math
import numpy as np

"""
LLM-generated reward function (round 2).
Changes:
- r_success only fires on true goal attainment (increased to 1000.0)
- Added r_distance_penalty to directly penalize moving away from goal
"""

import math
import numpy as np

def compute_reward(self, position, velocity, action, terminated):
    """
    Reward design for a MountainCar-like environment.
    Goal: reach position >= goal_position with velocity >= goal_velocity.

    Dense shaping: progress reward (rightward motion) + distance penalty (proximity to goal).
    Sparse terminal bonus: large positive reward only when goal is reached.
    """
    components = {}

    # -- Progress reward: encourage moving toward the right (high position) --
    # position is in [-1.2, 0.6]; goal is at 0.6.
    # Scale so that moving from left edge to goal gives ~1.8 reward cumulatively.
    r_progress = position * 0.5  # range [-0.6, 0.3] per step
    components["r_progress"] = r_progress

    # -- Distance penalty: penalize being far from the goal --
    # goal_position is 0.6; penalty is proportional to how far left we are.
    # Coefficient 0.5 makes the penalty range [-0.9, 0] per step, within 5x of r_progress.
    r_distance_penalty = -0.5 * (self.goal_position - position)  # range [-0.9, 0]
    components["r_distance_penalty"] = r_distance_penalty

    # -- Terminal success reward (sparse) --
    # Award only when the car actually reaches the goal (position and velocity thresholds).
    # Large magnitude (1000) to dominate dense shaping when achieved.
    r_success = 0.0
    if terminated and (position >= self.goal_position and velocity >= self.goal_velocity):
        r_success = 1000.0  # strong signal for completing the task
    components["r_success"] = r_success

    total = sum(components.values())

    # -- Outcome signal for evaluation (excluded from reward) --
    if terminated:
        # Environment step() only terminates on goal success, so outcome is always success.
        components["_outcome"] = 1.0
    # else no _outcome

    return total, components


def metrics_fn(env, action) -> dict:
    """
    Task-level metrics independent of the reward function.
    Measures distance to goal, velocity, success, steps taken, and action magnitude.
    """
    # Read current state from env (after step)
    pos = env.state[0]
    vel = env.state[1]

    # Goal parameters (set in environment)
    goal_pos = env.goal_position
    goal_vel = env.goal_velocity

    # Distance to goal (positive means still left)
    dist = goal_pos - pos

    # Success flag (exactly same condition as termination)
    success = 1.0 if (pos >= goal_pos and vel >= goal_vel) else 0.0

    # Step counter (initialized as 0 in reset, incremented each step)
    step_count = getattr(env, '_step_count', 0)

    # Action magnitude
    action_mag = float(np.abs(action[0]))

    return {
        "distance_to_goal": dist,
        "velocity": vel,
        "success": success,
        "steps": step_count,
        "action_magnitude": action_mag
    }


