"""LLM-generated reward function.
Source: round0
"""

import math
import numpy as np

def compute_reward(self, position, velocity, action, terminated):
    """
    Reward design for a MountainCar-like environment.
    Goal: reach position >= goal_position with velocity >= goal_velocity.
    
    Primary signal: dense shaping that encourages high position (rightward motion).
    Terminal bonus: large positive reward when goal is reached.
    No velocity penalties or efficiency terms per round 0 sparsity guidelines.
    """
    components = {}
    
    # -- Progress reward: encourage moving toward the right (high position) --
    # position is in [-1.2, 0.6]; goal is at 0.6.
    # Scale so that moving from the left edge to the goal gives ~1.8 reward over the episode.
    r_progress = position * 0.5  # range [-0.6, 0.3] per step, average ~0 over a random rollout
    components["r_progress"] = r_progress
    
    # -- Terminal success reward --
    r_success = 0.0
    if terminated:
        r_success = 100.0  # large bonus to dominate the sparse progress signal
    components["r_success"] = r_success
    
    total = sum(components.values())
    
    # -- Outcome signal for evaluation (excluded from reward) --
    if terminated:
        # Success condition already matches termination
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

