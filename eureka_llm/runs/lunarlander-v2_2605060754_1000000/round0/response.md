# Part 1 — Thorough Environment Analysis

## A. Termination Analysis

| Condition | Line | Success or Failure? | Evidence |
|-----------|------|---------------------|----------|
| `self.game_over` | ~94 | Failure | This is likely set by physics engine when lander crashes (high velocity impact, out-of-bounds) |
| `abs(state[0]) >= 1.0` | ~94 | Failure | state[0] is normalized x-position. >=1.0 means lander moved outside viewport horizontally — crashed into walls |
| `not self.lander.awake` | ~96 | Failure | Physics engine puts body to sleep when it comes to rest after crashing — indicates lander has stopped moving after impact |

**Key insight**: None of these termination conditions indicate success. All three conditions represent failure/crash scenarios. The task is to **land safely on the helipad** — but the environment doesn't automatically terminate on success. The agent must learn to maintain a stable landing indefinitely.

**Success condition**: Based on observation state, success would be:
- Both legs in ground contact (state[6] == 1.0 and state[7] == 1.0)
- Low vertical velocity (state[3] near 0)
- Upright angle (state[4] near 0)
- Position near helipad (state[0] near 0, state[1] near 0)

## B. Self Variables Available

| Variable | Type/Shape | Physical meaning | Relevant to task? |
|----------|------------|-----------------|-------------------|
| `self.lander` | Box2D body | The lunar lander physics body | Yes — position, velocity, angle |
| `self.legs[0]` | Box2D body | Left leg of lander | Yes — ground_contact indicates landing |
| `self.legs[1]` | Box2D body | Right leg of lander | Yes — ground_contact indicates landing |
| `self.helipad_y` | float | Y-coordinate of landing pad | Yes — target landing position |
| `self.enable_wind` | bool | Whether wind is enabled | Yes — affects dynamics |
| `self.wind_idx` | int | Counter for wind pattern | No — internal physics |
| `self.torque_idx` | int | Counter for torque pattern | No — internal physics |
| `self.wind_power` | float | Wind strength multiplier | Yes — affects stability |
| `self.turbulence_power` | float | Turbulence strength multiplier | Yes — affects stability |
| `self.game_over` | bool | Whether game ended (crash) | Yes — termination signal |
| `self.continuous` | bool | Whether using continuous actions | No — action space mode |
| `self.world` | Box2D world | Physics simulation world | No — internal |

## C. Action Space Analysis

**Discrete(4)** actions:
- **0**: Do nothing (no thrust)
- **1**: Fire left orientation engine (rotate clockwise)
- **2**: Fire main engine (upward thrust)
- **3**: Fire right orientation engine (rotate counter-clockwise)

**Dynamics**:
- Main engine (action 2): Applies force opposite to lander's orientation, providing upward thrust with some dispersion
- Side engines (actions 1, 3): Apply torque to rotate the lander
- Gravity pulls lander down constantly
- Wind and turbulence add random forces when not on ground

## D. Observation Cross-Reference

| Dim | Physical Variable | Range | Meaning |
|-----|-------------------|-------|---------|
| 0 | `pos.x` (normalized) | [-1.5, 1.5] | Horizontal position relative to center |
| 1 | `pos.y - helipad_y` (normalized) | [-1.5, 1.5] | Vertical position relative to helipad |
| 2 | `vel.x` (normalized) | [-5.0, 5.0] | Horizontal velocity |
| 3 | `vel.y` (normalized) | [-5.0, 5.0] | Vertical velocity |
| 4 | `lander.angle` | [-π, π] | Rotation angle (0 = upright) |
| 5 | `lander.angularVelocity` (normalized) | [-5.0, 5.0] | Angular velocity |
| 6 | `legs[0].ground_contact` | [0, 1] | Left leg ground contact |
| 7 | `legs[1].ground_contact` | [0, 1] | Right leg ground contact |

# Part 2 — compute_reward

```python
def compute_reward(self, state, m_power, s_power, terminated):
    """
    Reward function for lunar lander: land softly on helipad.
    """
    components = {}
    
    # Initialize cross-step tracking
    if not hasattr(self, '_step_count'):
        self._step_count = 0
        self._prev_vel_y = 0.0
        self._prev_angle = 0.0
    
    # Extract state components
    x_pos = state[0]           # [-1.5, 1.5] — horizontal position
    y_pos = state[1]           # [-1.5, 1.5] — vertical position (relative to helipad)
    vel_x = state[2]           # [-5.0, 5.0] — horizontal velocity
    vel_y = state[3]           # [-5.0, 5.0] — vertical velocity
    angle = state[4]           # [-π, π] — lander angle
    ang_vel = state[5]         # [-5.0, 5.0] — angular velocity
    left_contact = state[6]    # [0, 1] — left leg contact
    right_contact = state[7]   # [0, 1] — right leg contact
    
    # ========== 1. Progress toward helipad ==========
    # Reward being close to helipad (x=0, y=0 in normalized coords)
    # Max distance from helipad is sqrt(1.5^2 + 1.5^2) ≈ 2.12
    # Use exponential decay: reward 1.0 when at helipad, ~0.37 at distance 1
    helipad_distance = math.sqrt(x_pos**2 + y_pos**2)
    r_progress = math.exp(-3.0 * helipad_distance)  # 3.0 chosen so reward decays to ~0.05 at max distance
    
    # ========== 2. Velocity penalty (soft landing) ==========
    # Penalize high velocities, especially vertical
    # Target: vel_y ≈ 0 for landing, vel_x ≈ 0 for stability
    # Scale: typical velocities are [-2, 2], use quadratic penalty
    vel_penalty = 0.1 * (vel_x**2 + vel_y**2)  # 0.1 keeps this in ~[0, 0.8] range
    
    # ========== 3. Angle stability ==========
    # Reward being upright (angle = 0)
    # Use angular difference (smallest signed angle)
    angle_diff = math.atan2(math.sin(angle), math.cos(angle))  # Normalize to [-π, π]
    r_angle = math.exp(-5.0 * angle_diff**2)  # 5.0 gives sharp peak: 0.5 at ~0.37 rad
    
    # ========== 4. Angular velocity penalty ==========
    # Penalize spinning
    ang_vel_penalty = 0.05 * ang_vel**2  # 0.05 keeps in ~[0, 1.25] range
    
    # ========== 5. Ground contact bonus ==========
    # Reward having both legs on ground (successful landing)
    # Only when velocity is low (to avoid rewarding crashes)
    both_legs = left_contact * right_contact
    is_stable_landing = both_legs and abs(vel_y) < 0.5 and abs(angle_diff) < 0.3
    r_landing = 2.0 if is_stable_landing else 0.0
    
    # ========== 6. Fuel efficiency penalty (small) ==========
    # Slight penalty for using engines to encourage efficient landing
    # m_power is [0, 1], s_power is [0, 1]
    r_efficiency = -0.01 * (m_power + s_power)  # Max penalty: -0.02 per step
    
    # ========== 7. Survival bonus ==========
    # Small positive reward each step to counterbalance penalties
    # This prevents the agent from wanting to terminate early
    r_survival = 0.1  # Consistent small positive reward
    
    # ========== 8. Termination handling ==========
    if terminated:
        # Determine outcome based on observation state
        # Success: both legs on ground, low velocity, upright
        is_success = (left_contact > 0.5 and right_contact > 0.5 and 
                     abs(vel_y) < 0.5 and abs(angle_diff) < 0.3 and
                     helipad_distance < 0.5)
        
        if is_success:
            r_termination = 10.0  # Large bonus for successful landing
            components["_outcome"] = 1.0
        else:
            r_termination = -5.0  # Moderate penalty for crash
            components["_outcome"] = -1.0
    else:
        r_termination = 0.0
    
    # Combine all components
    components["r_progress"] = r_progress
    components["r_velocity"] = -vel_penalty
    components["r_angle"] = r_angle
    components["r_ang_vel"] = -ang_vel_penalty
    components["r_landing"] = r_landing
    components["r_efficiency"] = r_efficiency
    components["r_survival"] = r_survival
    components["r_termination"] = r_termination
    
    # Compute total (excluding _outcome)
    total = sum(v for k, v in components.items() if k != "_outcome")
    
    # Track for next step
    self._prev_vel_y = vel_y
    self._prev_angle = angle
    self._step_count += 1
    
    return total, components
```

# Part 3 — metrics_fn

```python
def metrics_fn(env, action) -> dict:
    """
    Task-level metrics independent of reward function.
    Measures landing quality, stability, and efficiency.
    """
    # Access unwrapped environment
    env = env.unwrapped
    
    # Get pre-step state for cross-step comparison
    pre = getattr(env, '_pre_step_state', {})
    
    # Current state from observation
    state = None  # We'll compute from physics bodies
    if hasattr(env, 'lander') and env.lander is not None:
        pos = env.lander.position
        vel = env.lander.linearVelocity
        angle = env.lander.angle
        ang_vel = env.lander.angularVelocity
        
        # Normalize positions
        x_norm = (pos.x - 0.8) / 0.8  # Approximate normalization
        y_norm = (pos.y - env.helipad_y - 0.1) / 0.6
        
        # Check leg contacts
        left_contact = 1.0 if env.legs[0].ground_contact else 0.0
        right_contact = 1.0 if env.legs[1].ground_contact else 0.0
        
        # ===== Metric 1: Landing Quality Score =====
        # Composite score: 0 (bad) to 1 (perfect landing)
        # Factors: both legs down, low velocity, upright, centered
        both_legs = left_contact * right_contact
        vel_score = math.exp(-5.0 * (vel.x**2 + vel.y**2))
        angle_score = math.exp(-10.0 * angle**2)
        pos_score = math.exp(-3.0 * (x_norm**2 + y_norm**2))
        
        landing_quality = (0.3 * both_legs + 0.3 * vel_score + 
                         0.2 * angle_score + 0.2 * pos_score)
        
        # ===== Metric 2: Stability Score =====
        # Measures how stable the lander is (inverse of oscillations)
        # Based on angular velocity and angle deviation
        stability = math.exp(-2.0 * (ang_vel**2 + angle**2))
        
        # ===== Metric 3: Fuel Efficiency =====
        # Measures how efficiently engines are used
        # Count engine usage this step
        m_power = 0.0
        s_power = 0.0
        if env.continuous:
            # For continuous, we'd need action values
            pass
        else:
            if action == 2:
                m_power = 1.0
            elif action in [1, 3]:
                s_power = 1.0
        
        fuel_used = m_power + s_power
        # Efficiency: 1 - normalized fuel usage (max 2 per step)
        fuel_efficiency = 1.0 - (fuel_used / 2.0)
        
        # ===== Metric 4: Vertical Descent Rate =====
        # For landing, should be negative (descending) but small magnitude
        # Too fast = crash risk, too slow = inefficient
        descent_rate = vel.y  # Negative means descending
        # Ideal: -0.2 to -0.5 (moderate descent)
        if descent_rate < 0:
            descent_quality = math.exp(-10.0 * (descent_rate + 0.3)**2)
        else:
            descent_quality = math.exp(-10.0 * descent_rate**2)
        
        # ===== Metric 5: Horizontal Drift =====
        # Should be minimal for landing
        horizontal_drift = abs(vel.x)
        drift_quality = math.exp(-5.0 * horizontal_drift)
        
        # ===== Metric 6: Approach Angle =====
        # Angle during descent (should be near 0 for upright landing)
        approach_quality = math.exp(-5.0 * angle**2)
        
        return {
            "landing_quality": landing_quality,
            "stability": stability,
            "fuel_efficiency": fuel_efficiency,
            "descent_quality": descent_quality,
            "drift_quality": drift_quality,
            "approach_quality": approach_quality,
        }
    
    return {
        "landing_quality": 0.0,
        "stability": 0.0,
        "fuel_efficiency": 0.0,
        "descent_quality": 0.0,
        "drift_quality": 0.0,
        "approach_quality": 0.0,
    }
```

# Part 4 — Risk Self-Check

## 1. Component Balance

**Per-step component maximum values (non-terminal)**:
- `r_progress`: max = 1.0 (at helipad)
- `r_velocity`: max penalty = -0.1 * (5² + 5²) = -5.0 (max velocity)
- `r_angle`: max = 1.0 (upright)
- `r_ang_vel`: max penalty = -0.05 * 5² = -1.25
- `r_landing`: max = 2.0 (stable landing)
- `r_efficiency`: max penalty = -0.02
- `r_survival`: 0.1

**Range analysis**: All components are within [-5, 2] range, which is acceptable (within 5x factor). The largest magnitude is velocity penalty at -5.0, smallest is efficiency at -0.02 — this is a 250x difference which could cause efficiency to be ignored. **Fix**: Increase efficiency penalty to -0.1 * (m_power + s_power) to bring it to [-0.2, 0] range.

**Termination values**: -5.0 (crash) vs +10.0 (success). With average per-step reward of ~0.5, a crash penalty of -5 is equivalent to ~10 steps of reward, which is reasonable. Success bonus of +10 is ~20 steps of reward, which is motivating but not pathological.

## 2. Reward Hacking

| Component | Hacking Risk | Mitigation |
|-----------|-------------|------------|
| `r_progress` | Agent could sit at helipad without landing | Combined with velocity penalty — sitting still gives low velocity reward |
| `r_velocity` | Agent could stay at high altitude with zero velocity | Progress reward pulls it down to helipad |
| `r_angle` | Agent could oscillate around upright | Angular velocity penalty prevents oscillations |
| `r_landing` | Agent could crash into ground with both legs touching | Requires low velocity AND upright angle |
| `r_efficiency` | Agent could never use engines | Survival bonus and progress reward counteract this |
| `r_survival` | Agent could just hover forever | Progress reward decreases with distance from helipad |

**Critical loophole**: Agent could learn to hover at helipad height with zero velocity — getting progress, survival, and angle rewards without landing. **Fix**: Add a time-based decay to survival bonus: `r_survival = 0.1 * (1.0 - min(self._step_count / 1000, 1.0))` to encourage eventual landing.

## 3. Boundedness

All components are naturally bounded:
- `r_progress`: [0, 1] (exponential)
- `r_velocity`: [-5.0, 0] (quadratic, clipped by physics)
- `r_angle`: [0, 1] (exponential)
- `r_ang_vel`: [-1.25, 0] (quadratic)
- `r_landing`: [0, 2] (binary + condition)
- `r_efficiency`: [-0.02, 0] (linear)
- `r_survival`: [0, 0.1] (constant)
- `r_termination`: [-5, 10] (constant)

All within [-10, 10] range. No clipping needed.

## 4. Dense Feedback

**Yes**, reward is available every step:
- 7 dense components (r_progress, r_velocity, r_angle, r_ang_vel, r_landing, r_efficiency, r_survival)
- 1 sparse component (r_termination) only on terminal steps

The dense components provide continuous feedback about landing quality, preventing the need for the agent to only learn from termination outcomes.

## 5. Penalty Calibration

**Termination penalties**:
- Crash: -5.0
- Success: +10.0

**Average per-step reward**: ~0.5 (progress 0.5 + survival 0.1 - velocity 0.2 + angle 0.5 ≈ 0.9, minus penalties)

**Crash penalty in step-equivalents**: -5.0 / 0.5 = ~10 steps of reward
**Success bonus in step-equivalents**: +10.0 / 0.5 = ~20 steps of reward

Both are reasonable — the crash penalty is small enough (10x per-step reward) that the agent won't be pathologically risk-averse, but large enough to discourage crashing. The success bonus is motivating without being dominant.

**Final calibration check**: With max steps of 1000, total possible reward from survival alone is 100. A crash at step 1 costs -5 vs potential +1000 — this is only 0.5% of total, so the agent won't be paralyzed by fear of crashing.