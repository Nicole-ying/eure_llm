"""Ant-v4 wrapper env with LLM reward hook (Phase-4 migration)."""

from __future__ import annotations
from typing import Optional
import numpy as np
import gymnasium as gym


class AntLLMEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode: Optional[str] = None):
        self.render_mode = render_mode
        self.base_env = gym.make("Ant-v4", render_mode=render_mode)
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        self._last_obs = None

    def step(self, action: np.ndarray):
        self._pre_step_state = {k: v for k, v in vars(self).items() if k.startswith("_") and k != "_pre_step_state"}
        obs, _official_reward, terminated, truncated, info = self.base_env.step(action)
        self._last_obs = obs
        reward, components = self.compute_reward(obs, action, terminated, truncated, info)
        info = dict(info or {})
        info["reward_components"] = components
        info["_pre_step_state"] = self._pre_step_state
        return obs, reward, terminated, truncated, info

    def compute_reward(self, obs, action, terminated, truncated, info):
        return 0.0, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._last_obs = obs
        return obs, info

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
