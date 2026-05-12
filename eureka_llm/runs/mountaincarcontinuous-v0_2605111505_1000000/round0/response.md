```python

def compute_reward(self, position, velocity, action, terminated):
    """
    Reward function for MountainCar.
    
    Physics: The car must build momentum by oscillating left and right.
    A weak engine cannot push directly up the hill.
    
    Primary signal: Normalized position (higher = closer to goal).
    This gives a dense, learnable signal that encourages the agent
    to move toward the flag on the right hill.
    
    Terminal bonus: +1 when the car reaches the goal position (≥0.5) with
    non‑negative velocity. This provides a strong final incentive.
    """
    # Normalize position to [0,1] range (min=-1.2, max=0.6 → span=1.8)
    r_progress = (position - (-1.2)) / 1.8   # ~0.39 at start, ~1.0 at goal
    
    components = {"r_progress": r_progress}
    
    if terminated:
        # Termination only occurs on success: position ≥ 0.5 & velocity ≥ 0
        r_goal = 1.0
        components["r_goal"] = r_goal
        components["_outcome"] = 1.0          # for evaluation only
        total = r_progress + r_goal
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

```