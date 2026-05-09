"""LLM-generated reward function (round 5).
Source: round5
"""

import math
import numpy as np

"""LLM-generated reward function (round 5).
Source: round5
Changes:
- Updated r_velocity_bonus: from 5.0 * max(0.0, velocity) to 8.0 * abs(velocity) to reward motion in both directions.
"""

import math
import numpy as np

def compute_reward(self, position, velocity, action, terminated):
    """
    Reward design for a MountainCar-like environment.
    Goal: reach position >= goal_position with velocity >= goal_velocity.

    Dense shaping: progress reward, distance penalty, and velocity bonus.
    Sparse terminal bonus: positive reward only when goal is reached.
    """
    components = {}

    # -- Progress reward: encourage moving toward the right (high position) --
    # position in [-1.2, 0.6]; goal at 0.6.
    # Scale so that moving from left edge to goal gives ~1.8 reward cumulatively.
    r_progress = position * 0.5  # range [-0.6, 0.3] per step
    components["r_progress"] = r_progress

    # -- Distance penalty: penalize being far from the goal --
    # goal_position is 0.6; coefficient 2.0 strongly incentivises approaching goal
    # (range [-3.6, 0] per step at extreme positions)
    r_distance_penalty = -2.0 * (self.goal_position - position)  # range [-3.6, 0]
    components["r_distance_penalty"] = r_distance_penalty

    # -- Velocity bonus: reward building momentum in either direction --
    # The agent must oscillate (move left and right) to build kinetic energy.
    # abs(velocity) rewards both forward and backward motion (essential for
    # accumulating speed). Coefficient 8.0 gives up to ~0.56 per step at max speed
    # (0.07), making motion attractive relative to the distance penalty.
    r_velocity_bonus = 8.0 * abs(velocity)  # range [0, ~0.56] per step
    components["r_velocity_bonus"] = r_velocity_bonus

    # -- Terminal success reward (sparse) --
    # Award only when the car actually reaches the goal (position and velocity thresholds).
    # Increased from 100.0 to 300.0 to make goal attainment competitively attractive compared
    # to the dense shaping rewards (which total ~1.5–2.0 per step).
    r_success = 0.0
    if terminated and (position >= self.goal_position and velocity >= self.goal_velocity):
        r_success = 300.0  # strong sparse signal for completing the task
    components["r_success"] = r_success

    total = sum(components.values())

    # -- Outcome signal for evaluation (excluded from reward) --
    if terminated:
        # Environment step() terminates only on goal success, so outcome is always success.
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


