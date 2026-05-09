def step(self, action):
    self._pre_step_state = {k: v for k, v in vars(self).items() if k.startswith("_") and k != "_pre_step_state"}
    obs, _official_reward, terminated, truncated, info = self.base_env.step(action)
    reward, components = self.compute_reward(obs, action, terminated, truncated, info)
    info = dict(info or {})
    info["reward_components"] = components
    info["_pre_step_state"] = self._pre_step_state
    return obs, reward, terminated, truncated, info
