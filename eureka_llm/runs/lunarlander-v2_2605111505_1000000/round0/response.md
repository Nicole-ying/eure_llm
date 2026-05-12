```python
def compute_reward(self, state, m_power, s_power, terminated):
    """
    State-based reward with a dense stability shaping, small leg contact bonus,
    and a large terminal outcome signal for successful soft landing or crash.

    Dynamics: Gravity constantly pulls the lander down. The agent must fire
    the main engine to slow descent while staying upright and centered.
    The primary task is to reach the pad with both legs touching, low speed,
    and near‑zero angle. The reward is designed to be learnable from dense
    feedback without penalising engine usage (which is needed to counteract gravity).
    """
    # --- state component weights (scaled to keep per-step reward in [-0.8, 0.01]) ---
    w_x = 0.20       # x position (range ~ [-1,1])
    w_angle = 0.08   # angle (range ~ [-π,π]) -> scaled to same magnitude as x
    w_vy = 0.10      # vertical velocity (range ~ [-2,0.5])
    w_vx = 0.10      # horizontal velocity (range ~ [-1.5,1.5])

    # 1. Stability shaping (negative penalty for deviation from ideal state)
    r_stability = -(
        w_x * abs(state[0]) +
        w_angle * abs(state[4]) +
        w_vy * abs(state[3]) +
        w_vx * abs(state[2])
    )

    # 2. Leg contact bonus (encourages attempting to land)
    leg_contacts = (1.0 if state[6] > 0.5 else 0.0) + (1.0 if state[7] > 0.5 else 0.0)
    r_legs = 0.10 * leg_contacts  # +0.10 per leg that touches ground

    # 3. Alive bonus (small positive to encourage staying in the episode)
    r_alive = 0.01

    # --- terminal outcome ---
    # Determine if termination is a successful soft landing
    success = False
    if terminated:
        # Soft landing criteria (from state, no access to internal physics)
        on_ground = (state[6] > 0.5 and state[7] > 0.5)
        angle_ok = abs(state[4]) < 0.15
        vert_vel_ok = abs(state[3]) < 0.2
        horiz_vel_ok = abs(state[2]) < 0.2
        centered = abs(state[0]) < 0.3
        success = on_ground and angle_ok and vert_vel_ok and horiz_vel_ok and centered

    # Build components dict (includes _outcome for evaluation, not added to total)
    components = {
        "r_stability": r_stability,
        "r_legs": r_legs,
        "r_alive": r_alive,
    }
    total = r_stability + r_legs + r_alive

    if terminated:
        if success:
            components["_outcome"] = 10.0   # positive bonus for successful landing
        else:
            components["_outcome"] = -10.0  # crash penalty
        # _outcome is NOT added to total

    return total, components


def metrics_fn(env, action) -> dict:
    """
    Task performance metrics (reward‑agnostic).
    Uses only state information and episode length.
    """
    # Safely access environment attributes (all access from step source code)
    # We can reconstruct a simplified state from lander body position/velocity
    # but here we use the observation state that was computed in step.
    # The state is stored in env._pre_step_state? Not reliable.
    # Instead, compute metrics directly from env.lander (Box2D body).
    lander = getattr(env, 'lander', None)
    legs = getattr(env, 'legs', None)
    helipad_y = getattr(env, 'helipad_y', 0.0)

    # Default values if attributes missing (should not happen)
    x = 0.0
    y = 0.0
    vx = 0.0
    vy = 0.0
    angle = 0.0
    leg_contact_left = 0.0
    leg_contact_right = 0.0

    if lander is not None:
        pos = lander.position
        vel = lander.linearVelocity
        # Normalise to same scale as observation (approximate)
        # Obs dim 0: (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2)
        # We skip exact normalisation; use raw coordinates from Box2D world.
        x = pos.x
        y = pos.y
        vx = vel.x
        vy = vel.y
        angle = lander.angle

    if legs is not None and len(legs) >= 2:
        leg_contact_left = 1.0 if legs[0].ground_contact else 0.0
        leg_contact_right = 1.0 if legs[1].ground_contact else 0.0

    # Episode progress (steps survived / max steps 1000)
    # Standard Gymnasium attribute for elapsed steps
    step_count = getattr(env, '_elapsed_steps', 0)

    # Compute metrics
    metrics = {
        # Horizontal distance to pad centre (raw world coordinates)
        "dist_to_pad": abs(x),
        # Height above ground (raw world coord, can be negative if crashed through)
        "height": y - helipad_y,
        # Absolute angle (radians)
        "angle_abs": abs(angle),
        # Vertical speed magnitude (world coord)
        "vert_speed_abs": abs(vy),
        # Number of legs in ground contact
        "legs_ground": int(leg_contact_left + leg_contact_right),
        # Fraction of episode completed (survival)
        "survival_fraction": min(step_count / 1000.0, 1.0),
    }

    return metrics
```