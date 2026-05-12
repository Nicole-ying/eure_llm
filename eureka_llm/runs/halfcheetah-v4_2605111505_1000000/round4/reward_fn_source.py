"""LLM-generated reward function (round 4).
Source: round4
"""

import math
import numpy as np

"""LLM-generated reward function (round 3, modified per analyst proposal).
Addresses exploration_balance and state_coverage by introducing an exponential
forward-velocity reward and an explicit velocity-variance exploration bonus.
Source: round3 (modified)
"""

import math
import numpy as np
import collections

def compute_reward(self, obs, action, terminated, truncated, info):
    """
    Primary dynamics: forward propulsion via leg actuation.
    The dominant physical process is the conversion of joint torques into horizontal movement
    while maintaining balance on a flat plane. Since there is no gravity-induced failure,
    the agent can stand still indefinitely, so the reward must actively drive forward motion.
    The simplest learnable signal is the body's forward velocity, which directly reflects
    task progress. A smoothness penalty prevents erratic movements that waste energy without
    contributing to speed.
    """
    import numpy as np

    # Initialize cross-step state if needed
    if not hasattr(self, '_step_count'):
        self._prev_action = np.zeros(6)   # first step: assume zero previous action
        self._prev_forward_vel = 0.0      # first step: assume zero previous velocity
        self._step_count = 0
        self._vel_buffer = collections.deque(maxlen=200)  # stores recent velocities for exploration bonus
    self._step_count += 1

    # Extract forward velocity from observation.
    # Based on typical HalfCheetah-v2 observation order (index 8 = x velocity).
    # If actual environment differs, this line would be adjusted.
    forward_vel = obs[8]  # shape: scalar

    # Reward component 1: forward velocity (primary task signal)
    # Changed from linear to exponential to create increasing gradient with speed,
    # breaking the local optimum and strongly incentivizing exploration of higher velocities.
    # weight 1.5: increased pressure to achieve higher absolute speed (per analyst proposal)
    # Exponential base (math.exp(forward_vel) - 1.0) ensures reward stays >= 0, with gradient proportional to current velocity.
    r_forward = 1.5 * (math.exp(forward_vel) - 1.0)

    # Reward component 2: action smoothness penalty (encourage smooth, efficient actuation)
    # Weight 0.001: reduced from 0.01 to reduce penalty magnitude, encouraging faster forward speed.
    # The penalty is small enough not to dominate r_forward, but still discourages erratic jerks.
    action_delta = action - self._prev_action
    r_smooth = -0.001 * np.sum(action_delta ** 2)  # weight 0.001: avoids suppressing needed motion
    self._prev_action = action.copy()  # store for next step

    # Reward component 3: velocity gradient bonus (r_delta), per analyst proposal
    # Changed from one-sided (increase-only) to two-sided (absolute difference) to encourage
    # speed variation and exploration of different gaits (addresses exploration_balance state_coverage).
    r_delta = abs(forward_vel - self._prev_forward_vel)
    self._prev_forward_vel = forward_vel  # store for next step

    # Reward component 4 (new): Exploration bonus based on velocity standard deviation over last 200 steps.
    # Weight 0.1: provides a modest bonus (~0.01–0.05 per step) proportional to recent velocity variation,
    # encouraging the agent to try different speeds directly addressing state_coverage.
    self._vel_buffer.append(forward_vel)
    if len(self._vel_buffer) == 200:
        std_vel = np.std(list(self._vel_buffer))
        r_explore = 0.1 * std_vel
    else:
        r_explore = 0.0

    components = {
        "r_forward": r_forward,
        "r_smooth": r_smooth,
        "r_delta": r_delta,
        "r_explore": r_explore,
    }
    total = sum(components.values())

    # Terminal outcome (for evaluation only) – not added to total reward
    if terminated:
        # No early termination possible (all episodes truncated), so outcome is neutral.
        components["_outcome"] = 0.0

    return total, components


def metrics_fn(env, action) -> dict:
    """
    Task-level metrics independent of reward.
    Collects key performance indicators for the HalfCheetah running task.
    """
    import numpy as np

    # Safely access observation from env (assumes env has obs after step)
    obs = getattr(env, 'obs', None)
    if obs is None:
        # Fallback: try env.base_env.obs (if wrapped)
        base = getattr(env, 'base_env', None)
        if base is not None:
            obs = getattr(base, 'obs', None)
        if obs is None:
            return {"error": "observation unavailable"}

    # Assume HalfCheetah observation indices (common for 17-dim version):
    # 0: torso height, 8: x velocity, 10: torso angular velocity, 12-16: joint velocities (if needed)
    forward_vel = obs[8] if len(obs) > 8 else 0.0
    torso_ang_vel = obs[10] if len(obs) > 10 else 0.0

    # Action magnitude (mean absolute torque)
    action_mag = float(np.mean(np.abs(action)))

    # Action smoothness (mean absolute change from previous action)
    # Use stored previous action from env if available
    prev_action = getattr(env, '_prev_action', np.zeros_like(action))
    action_delta = action - prev_action
    action_smoothness = float(np.mean(np.abs(action_delta)))

    # Approximate forward power (velocity * action magnitude) – not a reward, but useful metric
    forward_power = forward_vel * action_mag if abs(forward_vel) > 0.01 else 0.0

    # Additional metric: acceleration (derived from stored previous velocity)
    prev_vel = getattr(env, '_prev_forward_vel', None)
    if prev_vel is not None:
        acceleration = float(forward_vel - prev_vel)
    else:
        acceleration = 0.0

    return {
        "forward_velocity": float(forward_vel),
        "torso_angular_velocity": float(torso_ang_vel),
        "action_magnitude": action_mag,
        "action_smoothness": action_smoothness,
        "forward_power": forward_power,
        "acceleration": acceleration,
    }


