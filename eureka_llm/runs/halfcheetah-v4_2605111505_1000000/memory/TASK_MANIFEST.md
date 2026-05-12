# Task Manifest

## Environment Description
## Task Goal
Control a planar cheetah (6 joints: 2 legs × 3 joints each) to run forward as fast as possible along the x-axis. The cheetah should maintain an upright, forward-leaning posture with all four feet contacting the ground in a regular gait cycle.

## Termination Analysis
See step() source for termination conditions.

## Observation Space
See step() source for observation structure.

## Action Space
See step() source for action structure.

## Step Source Code
```python
def step(self, action):
    self._pre_step_state = {k: v for k, v in vars(self).items() if k.startswith("_") and k != "_pre_step_state"}
    obs, _official_reward, terminated, truncated, info = self.base_env.step(action)
    reward, components = self.compute_reward(obs, action, terminated, truncated, info)
    info = dict(info or {})
    info["reward_components"] = components
    info["_pre_step_state"] = self._pre_step_state
    return obs, reward, terminated, truncated, info

```
