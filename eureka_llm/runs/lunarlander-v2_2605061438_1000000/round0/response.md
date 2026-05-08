# Phase 1: Understanding the Environment

Let me trace through the code carefully.

## Observation Dimensions Analysis

Looking at how `state` is constructed:

1. **dim 0** (range [-1.5, 1.5]): `(pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2)` — This is **normalized x-position** of the lander relative to the center of the viewport. Range [-1, 1] means center is 0, edges are ±1.

2. **dim 1** (range [-1.5, 1.5]): `(pos.y - (helipad_y + LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2)` — This is **normalized y-position** relative to the helipad. The helipad_y is subtracted, so 0 means the lander is at helipad height.

3. **dim 2** (range [-5.0, 5.0]): `vel.x * (VIEWPORT_W/SCALE/2) / FPS` — **Normalized x-velocity**

4. **dim 3** (range [-5.0, 5.0]): `vel.y * (VIEWPORT_H/SCALE/2) / FPS` — **Normalized y-velocity**

5. **dim 4** (range [-π, π]): `self.lander.angle` — **Angle of the lander** (radians)

6. **dim 5** (range [-5.0, 5.0]): `20.0 * self.lander.angularVelocity / FPS` — **Angular velocity**

7. **dim 6** (binary 0/1): `1.0 if self.legs[0].ground_contact else 0.0` — **Left leg ground contact**

8. **dim 7** (binary 0/1): `1.0 if self.legs[0].ground_contact else 0.0` — **Right leg ground contact** (note: there's a typo in the code — both are `self.legs[0]`, but logically dim 7 should be leg[1])

## Termination Conditions

The episode terminates when:
1. `self.game_over` is True (set elsewhere, likely from crash conditions)
2. `abs(state[0]) >= 1.0` — lander goes off-screen horizontally
3. `not self.lander.awake` — lander has stopped moving (likely crashed/landed)

## What is the Task Goal?

This is a **lunar lander** environment! The agent must:
- Land softly on the landing pad (helipad)
- Maintain proper orientation (upright)
- Use the legs to make contact with the ground
- Avoid crashing or flying off-screen

The termination condition `not self.lander.awake` suggests the lander "sleeps" when it comes to rest — likely after a successful landing or a crash.

## The Simplest Reward Signal

The primary signal should be: **reward for being close to the landing pad with both legs down, moving slowly**. But since this is sparse, I'll use distance to the target as the primary signal.

Given the high failure rate (100% with random policy), the agent needs to learn to survive first. The simplest signal is:
- **Distance to helipad** (negative, so minimizing it is good)
- **Small alive bonus** to encourage survival

# Phase 2: Implementation

```python
def compute_reward(self, state, m_power, s_power, terminated):
    """
    Reward function for Lunar Lander.
    
    State dimensions:
    0: Normalized x-position (center=0, edges=±1)
    1: Normalized y-position relative to helipad (helipad=0, higher=positive)
    2: Normalized x-velocity
    3: Normalized y-velocity
    4: Angle (radians)
    5: Angular velocity
    6: Left leg ground contact (0/1)
    7: Right leg ground contact (0/1)
    """
    components = {}
    
    # Primary signal: distance to landing pad
    # We want the lander near (0, 0) in normalized coordinates
    # Scale: state[0] and state[1] are in [-1.5, 1.5], so max distance ~2.1
    # Reward of -1 per unit distance gives reasonable scale
    dist = math.sqrt(state[0]**2 + state[1]**2)
    r_distance = -dist * 0.5  # Scaled so typical values are ~[-0.5, 0]
    components["r_distance"] = r_distance
    
    # Small alive bonus to encourage survival during exploration
    # The agent needs to learn to stay alive before it can land
    # 0.01 per step = 10 per episode max, negligible compared to termination penalty
    r_alive = 0.01
    components["r_alive"] = r_alive
    
    total = sum(components.values())
    
    # Terminal outcome signal — for evaluation only
    if terminated:
        # Success: both legs on ground, near helipad, low velocity
        # state[6] and state[7] are leg contact flags
        on_ground = (state[6] > 0.5 and state[7] > 0.5)
        near_target = dist < 0.3  # Close to helipad
        low_speed = abs(state[2]) < 0.5 and abs(state[3]) < 0.5  # Slow velocity
        upright = abs(state[4]) < 0.5  # Nearly vertical
        
        if on_ground and near_target and low_speed and upright:
            components["_outcome"] = 1.0  # Successful landing
        else:
            components["_outcome"] = -1.0  # Crash/failure
        
        # No modification to total — _outcome is eval-only
    
    return total, components


def metrics_fn(env, action) -> dict:
    """
    Task-level metrics independent of reward function.
    """
    # Access state from environment
    state = env.lander  # We need to read from the environment
    
    # Read the observation from the last step
    # The state is stored in the environment
    pos = env.lander.position
    vel = env.lander.linearVelocity
    
    # Distance to helipad center (in world coordinates)
    helipad_x = 0  # Helipad is at center
    helipad_y = env.helipad_y + 0.1  # Top of helipad
    
    dx = pos.x - helipad_x
    dy = pos.y - helipad_y
    distance_to_pad = math.sqrt(dx*dx + dy*dy)
    
    # Speed magnitude
    speed = math.sqrt(vel.x**2 + vel.y**2)
    
    # Angle from vertical (absolute)
    angle_from_vertical = abs(env.lander.angle)
    
    # Ground contact status
    left_contact = 1.0 if env.legs[0].ground_contact else 0.0
    right_contact = 1.0 if env.legs[1].ground_contact else 0.0
    both_legs_down = left_contact * right_contact  # 1 if both, 0 otherwise
    
    # Is the lander awake? (not sleeping = still in motion)
    awake = 1.0 if env.lander.awake else 0.0
    
    return {
        "distance_to_pad": distance_to_pad,
        "speed": speed,
        "angle_from_vertical": angle_from_vertical,
        "both_legs_down": both_legs_down,
        "awake": awake,
        "altitude": pos.y - env.helipad_y,  # Height above helipad
    }
```