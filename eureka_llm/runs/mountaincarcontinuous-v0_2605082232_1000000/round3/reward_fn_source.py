"""LLM-generated reward function (round 3).
Source: round3
"""

import math
import numpy as np

import math
import numpy as np

"""
LLM-generated reward function (round 2, modified).
Changes applied:
- r_distance_penalty coefficient increased from -0.5 to -2.0 (stronger penalty for being far from goal)
- r_success reduced from 1000.0 to 100.0 (lower reward-to-cost ratio for false triggers)
"""

def compute_reward(self, position, velocity, action, terminated):
    """
    Reward design for a MountainCar-like environment.
    Goal: reach position >= goal_position with velocity >= goal_velocity.

    Dense shaping: progress reward (rightward motion) + distance penalty (proximity to goal).
    Sparse terminal bonus: positive reward only when goal is reached.
    """
    components = {}

    # -- Progress reward: encourage moving toward the right (high position) --
    # position is in [-1.2, 0.6]; goal is at 0.6.
    # Scale so that moving from left edge to goal gives ~1.8 reward cumulatively.
    r_progress = position * 0.5  # range [-0.6, 0.3] per step
    components["r_progress"] = r_progress

    # -- Distance penalty: penalize being far from the goal --
    # goal_position is 0.6; penalty is proportional to how far left we are.
    # Coefficient 2.0 makes the penalty roughly 4× stronger than before,
    # strongly incentivizing the agent to approach the goal (range [-3.6, 0] per step).
    r_distance_penalty = -2.0 * (self.goal_position - position)  # range [-3.6, 0]
    components["r_distance_penalty"] = r_distance_penalty

    # -- Terminal success reward (sparse) --
    # Award only when the car actually reaches the goal (position and velocity thresholds).
    # Reduced to 100.0 to keep the sparse signal meaningful but not overwhelming the dense terms.
    r_success = 0.0
    if terminated and (position >= self.goal_position and velocity >= self.goal_velocity):
        r_success = 100.0  # strong but balanced signal for completing the task
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


