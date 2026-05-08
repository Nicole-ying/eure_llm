## Part 1 — Thorough Environment Analysis

### A. Termination Analysis

| Condition | Line | Success or Failure? | Evidence |
|-----------|------|---------------------|----------|
| `self.game_over` becomes True | Line in `if self.game_over or ...` | Failure | `game_over` is set elsewhere (likely when lander crashes into ground/walls or exceeds angle threshold) |
| `abs(state[0]) >= 1.0` | Same line | Failure | Landed outside viewport horizontally – out of acceptable bounds |
| `not self.lander.awake` | Line after | **Success** (likely) | In Box2D, a body becomes "not awake" when sleeping, which happens after it comes to rest on a surface. Combined with legs contact and low velocity, this indicates safe landing. The lander stopping on the pad is the goal. |

### B. Self Variables Available

| Variable | Type/Shape | Physical meaning | Relevant to task? |
|----------|------------|-----------------|-------------------|
| `self.lander` | `Body` | Box2D rigid body of the lander | Yes – position, velocity, angle, angular velocity, awake state |
| `self.legs` | list of 2 `Fixture` | Leg fixtures for ground contact detection | Yes – `ground_contact` in state[6], state[7] |
| `self.helipad_y` | float | Y‑coordinate of the landing pad (ground level) | Yes – vertical landing target |
| `self.world` | `World` | Box2D physics world | Yes – time stepping, gravity |
| `self.game_over` | bool | True if lander has crashed | Yes – failure trigger |
| `self.enable_wind` | bool | Whether wind/turbulence are active | Indirect – affects dynamics |
| `self.wind_idx`, `self.torque_idx` | int | Counters for wind/torque functions | Marginal – not directly used |
| `self.wind_power`, `self.turbulence_power` | float | Magnitude of wind/torque forces | Indirect |
| `self._pre_step_state` | dict | Snapshot of all `self._*` attributes before reward computation | Yes – for cross-step metrics |
| `self.render_mode` | str or None | Rendering flag | No |
| `self.np_random` | RandomState | Random number generator | No |

### C. Action Space Analysis

`Discrete(4)` actions:  
- **0**: No engine  
- **1**: Fire left orientation engine (clockwise torque)  
- **2**: Fire main engine (upward thrust)  
- **3**: Fire right orientation engine (counter‑clockwise torque)  

The main engine applies force opposite the lander’s rotation (along the “tip” vector), producing upward thrust relative to the lander frame. Side engines apply torque and also a small lateral force. All engines have random dispersion to simulate noise.

### D. Observation Cross‑Reference

| Obs dim | State expression | Physical variable | Units / Scaling |
|---------|------------------|------------------|----------------|
| 0 | `(pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2)` | Normalized horizontal position (center=0, left=-1, right=+1) | dimensionless |
| 1 | `(pos.y - (helipad_y + LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2)` | Normalized vertical position relative to landing pad (pad=0, above=positive) | dimensionless |
| 2 | `vel.x * (VIEWPORT_W/SCALE/2) / FPS` | Horizontal velocity | dimensionless |
| 3 | `vel.y * (VIEWPORT_H/SCALE/2) / FPS` | Vertical velocity | dimensionless |
| 4 | `self.lander.angle` | Angle (radians), 0 = upright | rad |
| 5 | `20.0 * self.lander.angularVelocity / FPS` | Angular velocity | dimensionless (scaled) |
| 6 | `1.0 if self.legs[0].ground_contact else 0.0` | Left leg ground contact | binary |
| 7 | `1.0 if self.legs[1].ground_contact else 0.0` | Right leg ground contact | binary |

---

## Part 2 — compute_reward

```python
def compute_reward(self, state, m_power, s_power, terminated):
    """
    Dense reward shaping for lunar landing.
    Encourages: reaching landing pad, upright orientation, low velocity, both legs contact.
    Returns:
        total_reward: float
        components: dict
    """
    components = {}

    # Unpack state for convenience
    x_norm, y_norm, vx_norm, vy_norm, angle, angvel_norm, leg_left, leg_right = state

    # --- 1. Progress toward landing pad ---
    # Target: x=0 (horizontal center), y=0 (pad altitude)
    # Use squared distance with a shaping bias to avoid early settling
    pad_dist = math.sqrt(x_norm**2 + y_norm**2)
    # Reward for decreasing distance (max per-step improvement ~0.1, reward ~0.2)
    r_progress = -0.5 * pad_dist  # negative penalty proportional to distance; max ~ -0.75 (distance ~1.5)
    # Add a small positive reward for being close (to guide final approach)
    if pad_dist < 0.1:
        r_progress += 0.5 * (0.1 - pad_dist)  # up to +0.05 extra
    components['r_progress'] = r_progress

    # --- 2. Stability (angle and angular velocity) ---
    # Angle target: 0 (upright). Use smallest signed angular difference.
    ang_diff = math.atan2(math.sin(angle), math.cos(angle))  # wrap to [-pi, pi]
    # Penalize deviation from zero. Max absolute deviation ~pi, so reward range ~[-1, 0]
    r_stability_angle = -0.3 * abs(ang_diff)
    # Penalize angular velocity – keep it small (< ~0.5 rad/frame scaled)
    r_stability_angvel = -0.2 * abs(angvel_norm)
    # Combined stability
    r_stability = r_stability_angle + r_stability_angvel
    components['r_stability'] = r_stability

    # --- 3. Leg contact (landing success shaping) ---
    # Reward two legs touching, penalty for only one (tipping)
    if leg_left and leg_right:
        r_contact = 1.0  # both down – immediate small reward
    elif leg_left or leg_right:
        r_contact = -0.5  # unbalanced – penalty
    else:
        r_contact = 0.0
    components['r_contact'] = r_contact

    # --- 4. Fuel efficiency (penalize excessive thrust) ---
    # m_power in [0,1] for main, s_power in [0,1] for side
    # Penalize using fuel – encourage efficient flight
    r_efficiency = -0.3 * (m_power + s_power)  # max -0.6 per step
    components['r_efficiency'] = r_efficiency

    # --- 5. Termination handling ---
    r_termination = 0.0
    if terminated:
        # Determine success or failure
        # Success: lander not awake AND both legs in contact AND near pad AND low velocity
        success = (not self.lander.awake and leg_left and leg_right and
                   abs(ang_diff) < 0.3 and abs(vy_norm) < 0.1 and pad_dist < 0.2)
        if success:
            r_termination = 10.0  # large positive for successful landing
        else:
            r_termination = -5.0  # moderate penalty for crash/out-of-bounds
        components['r_termination'] = r_termination

    # --- Sum total reward ---
    total = sum(components.values())
    return total, components
```

---

## Part 3 — metrics_fn

```python
def metrics_fn(env, action) -> dict:
    """
    Task-level metrics reward-independent.
    Measures: distance to pad, angular stability, fuel usage, landing success rate (in info), 
    velocity at termination, contact balance.
    """
    # Access cross-step state if needed
    pre = getattr(env, '_pre_step_state', {})
    prev_action = pre.get('_prev_action', action)  # example

    # Get current physics state
    lander = env.unwrapped.lander
    legs = env.unwrapped.legs
    pos = lander.position
    vel = lander.linearVelocity
    angle = lander.angle
    # Normalize to same scale as state[0,1]
    # Use constants from env (assuming they exist; if not, use typical values)
    try:
        SCALE = env.unwrapped.SCALE
        VIEWPORT_W = env.unwrapped.VIEWPORT_W
        VIEWPORT_H = env.unwrapped.VIEWPORT_H
        FPS = env.unwrapped.FPS
        helipad_y = env.unwrapped.helipad_y
        LEG_DOWN = env.unwrapped.LEG_DOWN
    except AttributeError:
        # Fallback to typical LunarLander constants
        SCALE = 30.0
        VIEWPORT_W = 600.0
        VIEWPORT_H = 400.0
        FPS = 50.0
        helipad_y = 0.0
        LEG_DOWN = 1.0

    x_norm = (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2)
    y_norm = (pos.y - (helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2)
    vx_norm = vel.x * (VIEWPORT_W / SCALE / 2) / FPS
    vy_norm = vel.y * (VIEWPORT_H / SCALE / 2) / FPS
    angvel_norm = 20.0 * lander.angularVelocity / FPS

    # 1. Progress metric: Euclidean distance to pad (normalized)
    pad_dist = math.sqrt(x_norm**2 + y_norm**2)
    # 2. Stability metric: absolute angle deviation from 0
    angle_dev = abs(math.atan2(math.sin(angle), math.cos(angle)))
    # 3. Fuel usage: total power applied this step (0-2)
    # Use action to infer (since power not stored)
    m_power = 1.0 if action == 2 else 0.0
    s_power = 1.0 if action in [1, 3] else 0.0
    fuel = m_power + s_power
    # 4. Landing readiness: combined leg contact (0,1,2)
    leg_contact = int(legs[0].ground_contact) + int(legs[1].ground_contact)
    # 5. Vertical speed relative to pad – measure of crash risk
    vert_speed_mag = abs(vy_norm)
    # 6. Cross-step: angular velocity change (if previous state available)
    # Use pre_state if available
    prev_angvel = pre.get('_prev_angvel', None)
    angvel_change = abs(angvel_norm - prev_angvel) if prev_angvel is not None else 0.0
    # Store current angvel for next step
    # (in real implementation you'd save this in env._prev_angvel)

    return {
        "pad_distance": pad_dist,
        "angle_deviation": angle_dev,
        "fuel_used": fuel,
        "leg_contact_sum": leg_contact,
        "vertical_speed_magnitude": vert_speed_mag,
        "angvel_change": angvel_change
    }
```

---

## Part 4 — Risk Self-Check

### 1. Component Balance

| Component | Approx. typical per-step value | Max possible (absolute) | Dominance? |
|-----------|--------------------------------|------------------------|------------|
| `r_progress` | -0.5 to +0.1 | ~0.75 | Moderate |
| `r_stability` | -0.3 to 0 | ~0.5 | Moderate |
| `r_contact` | -0.5 to +1 | 1 | Largest positive |
| `r_efficiency` | -0.3 to 0 | ~0.6 | Moderate |
| `r_termination` | 0 (non-termination) | 10 (success) or -5 (failure) | High but only on terminal step – not dominating per-step |

All per-step components are within same order (~1). Termination component is an order of magnitude larger, but it appears only once per episode, so per-step average ~0.1, acceptable.

### 2. Reward Hacking

- **r_progress**: Agent could simply stay still at pad to get low distance, but then y_norm would be negative? Actually pad distance at good landing is near zero – fine. However, if agent learns to exploit by staying in a region with small distance but high velocity, that's penalized by other components.
- **r_contact**: Agent could try to sacrifice stability and just tip over to get both legs touching ground? But that would cause high angle penalty and likely crash. Also angular and velocity penalties outweigh.
- **r_efficiency**: Agent could learn to never use engines (fuel=0) to avoid penalty, but then it would crash due to gravity. So it's necessary to use engines – the shaping encourages minimal usage, not zero.
- **r_termination**: If success detection is too loose, agent might "cheat" by sleeping on the ground away from pad. We've added pad_dist and angle/velocity constraints, so only valid landings get +10. Failure penalty -5 is moderate – agent will not purposely crash.

### 3. Boundedness

All components are naturally bounded:
- `r_progress`: distance ≤ ~1.5 → reward ~[-0.75, +0.05]
- `r_stability`: angle ≤ π → max -0.94; angvel ≤ ~5 → max -1.0
- `r_contact`: [-0.5, 1.0]
- `r_efficiency`: [-0.6, 0]
- `r_termination`: [-5, 10]

No component grows unbounded.

### 4. Dense Feedback

Reward is computed every step (including shaping). Termination adds extra signal. Dense shaping guides learning.

### 5. Penalty Calibration

Failure penalty is -5, success reward is +10. Typical per-step reward is ~-0.5 to +0.5. Over 100-step episode, total shaping ~ -50 to +50. A -5 penalty on failure is ~10% of typical total, not dominating. It is < 100× per-step reward (since per-step ~0.5, 100× = 50). Acceptable. It does not cause pathological risk aversion because the penalty is moderate and the agent can compensate by safe flying.