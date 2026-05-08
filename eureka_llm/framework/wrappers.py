"""
wrappers.py — Wrappers for the eureka_llm framework.

EpisodeInfoWrapper — captures end-of-episode state for generic completion detection
ComponentTrackerWrapper — logs reward components per episode to JSONL (CARD data)
PickleSafeInfoWrapper — strips non-picklable items from info dict (SubprocVecEnv safety)
"""

import json
import pickle
from pathlib import Path

import numpy as np
import gymnasium as gym


class EpisodeInfoWrapper(gym.Wrapper):
    """
    Captures end-of-episode state into the info dict before auto-reset.
    Generic — works with any Gymnasium environment.

    Adds:
        info["_episode_terminated"] = terminated  (on EVERY step)
        info["_episode_end"] = True  (on terminal steps)
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["_episode_terminated"] = terminated
        info["_episode_truncated"] = truncated
        if terminated or truncated:
            info["_episode_end"] = True
        return obs, reward, terminated, truncated, info


class MetricsTrackingWrapper(gym.Wrapper):
    """
    Calls metrics_fn (if available on the env class) on each step and stores
    the results in info["env_metrics"] for downstream tracking.

    metrics_fn is injected onto the env class by inject_and_register() in
    train.py, so it is available via env.unwrapped.metrics_fn.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Check if metrics_fn is available on the base env class
        metrics_fn = getattr(self.env.unwrapped, "metrics_fn", None)
        if metrics_fn is not None:
            try:
                metrics = metrics_fn(self.env.unwrapped, action)
                if isinstance(metrics, dict):
                    info["env_metrics"] = metrics
            except Exception:
                pass  # metrics_fn is LLM-generated — silently skip on error
        return obs, reward, terminated, truncated, info


class ComponentTrackerWrapper(gym.Wrapper):
    """
    Accumulates reward components across each episode and writes per-episode
    summaries to a JSONL log file.

    Reads from info["reward_components"] (set by the env's step() return).
    Also reads from info["env_metrics"] if available (set by MetricsTrackingWrapper).
    """

    def __init__(self, env, log_path: Path):
        super().__init__(env)
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._episode = 0
        self._reset_accumulators()

    def _reset_accumulators(self):
        self._step_count = 0
        self._components: dict[str, list] = {}
        self._env_metrics: dict[str, list] = {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        for name, value in info.get("reward_components", {}).items():
            self._components.setdefault(name, []).append(float(value))

        for name, value in info.get("env_metrics", {}).items():
            self._env_metrics.setdefault(name, []).append(float(value))

        if terminated or truncated:
            self._save_episode()
            self._episode += 1
            self._reset_accumulators()

        return obs, reward, terminated, truncated, info

    def _save_episode(self):
        record = {
            "episode": self._episode,
            "length": self._step_count,
            "component_means": {
                k: round(float(np.mean(v)), 6) for k, v in self._components.items()
            },
            "component_stds": {
                k: round(float(np.std(v)), 6) for k, v in self._components.items()
            },
        }
        if self._env_metrics:
            record["env_metrics_means"] = {
                k: round(float(np.mean(v)), 6) for k, v in self._env_metrics.items()
            }
            record["env_metrics_stds"] = {
                k: round(float(np.std(v)), 6) for k, v in self._env_metrics.items()
            }
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class PickleSafeInfoWrapper(gym.Wrapper):
    """
    Strip non-picklable items from the info dict.

    LLM-generated reward functions may store Box2D (SWIG) objects as instance
    attributes that get captured in the env's step info.  When the env runs in
    SubprocVecEnv, the worker sends (obs, reward, done, info) through a pipe,
    which requires pickle — any non-picklable value in info causes a crash.

    This wrapper sanitises the info dict so the experiment survives bad data:
    non-picklable entries are replaced with their string representation rather
    than crashing the entire training run.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = _sanitize_pickle(info)
        return obs, reward, terminated, truncated, info


def _sanitize_pickle(obj, depth=0):
    """Recursively strip non-picklable values from info structures."""
    if depth > 20:  # prevent infinite recursion
        return str(obj)

    # Check if obj itself is picklable
    try:
        pickle.dumps(obj)
        return obj
    except Exception:
        pass

    if isinstance(obj, dict):
        return {k: _sanitize_pickle(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq_type = type(obj)
        cleaned = [_sanitize_pickle(v, depth + 1) for v in obj]
        return seq_type(cleaned)

    # Fallback: convert to string representation
    return repr(obj)
