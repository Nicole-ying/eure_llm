# Phase 1: Understanding the Environment

Let me trace through the code carefully:

## Observation Dimensions Analysis

Looking at how `state` is built:

1. **dim 0** (range ≈ [-1.5, 1.5]): `(pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2)` — This is the **horizontal position** normalized to [-1, 1]. The mean near 0.076 suggests the lander starts roughly centered.

2. **dim 1** (range ≈ [-1.5, 1.5]): `(pos.y - (helipad_y + LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2)` — This is the **vertical position** relative to the helipad. The mean of 0.956 and max of 1.656 suggest the lander starts high above the pad.

3. **dim 2** (range ≈ [-5, 5]): `vel.x * (VIEWPORT_W/SCALE/2) / FPS` — **Horizontal velocity**, scaled.

4. **dim 3** (range ≈ [-5, 5]): `vel.y * (VIEWPORT_H/SCALE/2) / FPS` — **Vertical velocity**, scaled. The negative mean (-0.667) indicates the lander is falling.

5. **dim 4** (range ≈ [-π, π]): `self.lander.angle` — **Angle/rotation** of the lander. ±π range confirms this is an angle.

6. **dim 5** (range ≈ [-5, 5]): `20.0 * self.lander.angularVelocity / FPS` — **Angular velocity**, scaled.

7. **dim 6** (binary 0/1): `1.0 if self.legs[0].ground_contact else 0.0` — **Left leg ground contact**.

8. **dim 7** (binary 0/1): `1.0 if self.legs[1].ground_contact else 0.0` — **Right leg ground contact**.

## Termination Conditions

The episode terminates when:
- `self.game_over` is True (set somewhere else in the code)
- `abs(state[0]) >= 1.0` — lander goes off-screen horizontally
- `not self.lander.awake` — lander crashes/comes to rest (likely from hitting ground too hard)

## Task Goal

This is clearly **Lunar Lander** from Gymnasium. The goal is to:
1. Land softly on the landing pad
2. With both legs touching the ground
3. Without crashing or flying off-screen

The key success condition (not shown in the termination code but standard for Lunar Lander) is:
- Both legs have ground contact (`state[6] == 1.0` and `state[7] == 1.0`)
- The lander is roughly upright (angle near 0)
- Low velocity on touchdown

## Simplest Reward Signal

The **primary signal** should be: **Reward for being on the landing pad with both legs down**. This is the ultimate goal.

But this is extremely sparse — random exploration will almost never achieve it.

The **minimum shaping** needed: **Reward for being close to the landing pad (low position y) with low velocity**, since the pad is at the bottom of the screen.

Looking at the stats: dim 1 (vertical position) has mean 0.956 and max 1.656, meaning the lander starts high. The goal is to get dim 1 close to 0 (at pad level) with both legs contacting.

# Phase 2: Writing the Reward Function

```python
def compute_reward(self, state, m_power, s_power, terminated):
    components = {}
    
    # Extract state components with descriptive names
    pos_x = state[0]      # horizontal position, normalized [-1, 1]
    pos_y = state[1]      # vertical position relative to helipad
    vel_x = state[2]      # horizontal velocity
    vel_y = state[3]      # vertical velocity
    angle = state[4]      # angular position
    ang_vel = state[5]    # angular velocity
    leg_left = state[6]   # left leg ground contact (0 or 1)
    leg_right = state[7]  # right leg ground contact (0 or 1)
    
    # Primary reward: proximity to landing pad (vertical closeness)
    # pos_y=0 means at pad level, pos_y>0 means above pad
    # Scale so max reward is ~0.1 per step when at pad
    r_proximity = -abs(pos_y) * 0.2  # penalty for being far from pad
    
    # Bonus for having both legs on ground (success signal)
    # This is the actual task goal
    both_legs = leg_left * leg_right  # 1.0 only when both legs contact
    r_landing = both_legs * 1.0  # small bonus per step when landed
    
    # Alive bonus - small reward each step to counter gravity penalty
    # The agent must act to stay alive, so give a tiny positive reward
    r_alive = 0.01  # small positive reward for surviving each step
    
    components["r_proximity"] = r_proximity
    components["r_landing"] = r_landing
    components["r_alive"] = r_alive
    
    total = sum(components.values())
    
    # Terminal outcome signal (for evaluation only)
    if terminated:
        # Success: both legs on ground, roughly upright, low velocity
        is_upright = abs(angle) < 0.5  # roughly vertical
        is_gentle = abs(vel_y) < 0.5   # low vertical speed
        is_centered = abs(pos_x) < 0.3  # on the pad
        
        if both_legs and is_upright and is_gentle and is_centered:
            components["_outcome"] = 10.0  # successful landing bonus
        else:
            components["_outcome"] = -1.0  # crash/failure penalty
        
        # Subtract outcome from total so it doesn't affect gradient
        total = total - components["_outcome"]
    
    return total, components


def metrics_fn(self, env, action) -> dict:
    """
    Task-level metrics independent of the reward function.
    """
    # Access state from env after step
    state = env.lander_state if hasattr(env, 'lander_state') else None
    
    # Get the observation from the environment
    # In LunarLander, we can read the lander's physics state directly
    if hasattr(env, 'lander') and env.lander is not None:
        pos = env.lander.position
        vel = env.lander.linearVelocity
        angle = env.lander.angle
        ang_vel = env.lander.angularVelocity
        
        # Distance to helipad (center of screen, bottom)
        helipad_x = 0  # center
        helipad_y = env.helipad_y if hasattr(env, 'helipad_y') else 0
        
        distance_to_pad = math.sqrt(
            (pos.x - helipad_x)**2 + 
            (pos.y - helipad_y)**2
        )
        
        # Speed magnitude
        speed = math.sqrt(vel.x**2 + vel.y**2)
        
        # Angular deviation from upright
        angle_error = abs(angle)
        
        # Ground contact status
        left_contact = 1.0 if (hasattr(env, 'legs') and 
                               env.legs[0].ground_contact) else 0.0
        right_contact = 1.0 if (hasattr(env, 'legs') and 
                                env.legs[1].ground_contact) else 0.0
        
        return {
            "distance_to_pad": distance_to_pad,
            "speed": speed,
            "angle_error": angle_error,
            "both_legs_contact": left_contact * right_contact,
            "is_alive": 1.0 if env.lander.awake else 0.0,
        }
    else:
        return {
            "distance_to_pad": 0.0,
            "speed": 0.0,
            "angle_error": 0.0,
            "both_legs_contact": 0.0,
            "is_alive": 1.0,
        }
```

The reward function is deliberately sparse with only 3 components:
- **r_proximity**: Primary shaping signal - encourages moving toward the landing pad
- **r_landing**: Task completion signal - rewards the actual goal state (both legs on ground)
- **r_alive**: Minimal survival bonus to counteract gravity without being too strong

The termination penalty is small (-1.0 for crash, +10.0 for success) to avoid pathological risk aversion while still providing a meaningful outcome signal for evaluation.