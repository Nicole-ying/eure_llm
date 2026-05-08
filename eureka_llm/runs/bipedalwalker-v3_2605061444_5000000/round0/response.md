# Phase 1: Understanding the Environment

Let me trace through the code carefully.

## Observation Structure Analysis

Looking at the state construction in `step()`:

1. **dim 0**: `self.hull.angle` - hull's angular orientation (radians, ±π range)
2. **dim 1**: `2.0 * self.hull.angularVelocity / FPS` - normalized angular velocity
3. **dim 2**: `0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS` - normalized horizontal velocity
4. **dim 3**: `0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS` - normalized vertical velocity
5. **dim 4**: `self.joints[0].angle` - left hip joint angle
6. **dim 5**: `self.joints[0].speed / SPEED_HIP` - left hip joint speed
7. **dim 6**: `self.joints[1].angle + 1.0` - left knee joint angle (offset by 1.0)
8. **dim 7**: `self.joints[1].speed / SPEED_KNEE` - left knee joint speed
9. **dim 8**: `1.0 if self.legs[1].ground_contact else 0.0` - left foot ground contact
10. **dim 9**: `self.joints[2].angle` - right hip joint angle
11. **dim 10**: `self.joints[2].speed / SPEED_HIP` - right hip joint speed
12. **dim 11**: `self.joints[3].angle + 1.0` - right knee joint angle (offset by 1.0)
13. **dim 12**: `self.joints[3].speed / SPEED_KNEE` - right knee joint speed
14. **dim 13**: `1.0 if self.legs[3].ground_contact else 0.0` - right foot ground contact
15. **dims 14-23**: `l.fraction for l in self.lidar` - 10 lidar distance readings (0=hit, 1=no hit)

## Termination Conditions

The episode terminates when:
1. `self.game_over` is True (likely from falling/unstable state)
2. `pos[0] < 0` - hull moved backward past start
3. `pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP` - hull reached the end of terrain

## Task Goal Inference

This is clearly a bipedal locomotion task - a two-legged robot that needs to walk forward. The key clues:
- Two legs with hip and knee joints (4 motors total)
- Lidar sensors for terrain sensing
- Ground contact sensors on feet
- Termination when moving backward or when the terrain ends (success condition)
- High failure rate (57%) from falling/instability

The **goal** is to move forward as far as possible without falling, ideally reaching the end of the terrain.

## Simplest Possible Reward Signal

**Primary signal**: Forward velocity (dim 2 - horizontal velocity). The agent needs to move right (positive x direction). This gives immediate feedback on every step.

## Minimum Shaping Needed

The primary signal (velocity) is dense enough - every step gives feedback. However, the agent might learn to just fall forward quickly. A small alive bonus would help keep the agent upright, but per the "keep it sparse" rule, let's start with just velocity.

Actually, looking more carefully - the agent needs to survive to make progress. With 57% termination rate under random actions, the agent needs to learn stability. But velocity alone should work since falling terminates the episode and stops reward accumulation.

Let me go with just forward velocity as the primary signal.

```python
def compute_reward(self, action):
    """
    Reward for forward locomotion.
    Primary signal: forward velocity (positive x direction)
    """
    # Forward velocity from observation dim 2
    # This is already normalized: 0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS
    # Positive = moving right/forward
    forward_vel = self.hull.linearVelocity.x
    
    # Scale reward so typical values are ~[-0.5, 1.0] per step
    # Random policy mean was 0.0098 with std 0.052, so typical range is small
    # Scale up to make it meaningful
    r_forward = forward_vel * 2.0
    
    components = {"r_forward": r_forward}
    total = sum(components.values())
    
    # Terminal outcome for evaluation only
    if self.terminated:
        # Success: reached end of terrain
        if self.hull.position.x > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            components["_outcome"] = 1.0
        # Failure: fell or went backward
        else:
            components["_outcome"] = -1.0
    
    return total, components


def metrics_fn(env, action) -> dict:
    """
    Task-level metrics independent of reward.
    """
    pos = env.hull.position
    vel = env.hull.linearVelocity
    
    # Forward progress (cumulative x position)
    forward_progress = pos.x
    
    # Stability: how upright is the hull (angle close to 0)
    stability = 1.0 - abs(env.hull.angle) / math.pi
    
    # Speed: absolute forward speed
    speed = abs(vel.x)
    
    # Ground contact: how many feet are on ground
    left_contact = 1.0 if env.legs[1].ground_contact else 0.0
    right_contact = 1.0 if env.legs[3].ground_contact else 0.0
    feet_contact = left_contact + right_contact
    
    # Energy efficiency: total motor torque used
    total_torque = sum(
        abs(env.joints[i].motorSpeed * env.joints[i].maxMotorTorque)
        for i in range(4)
    )
    
    return {
        "forward_progress": forward_progress,
        "stability": stability,
        "speed": speed,
        "feet_contact": feet_contact,
        "total_torque": total_torque,
    }
```