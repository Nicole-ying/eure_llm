"""
env_explorer.py — Auto-discovery for unknown environments.

Loads an env from a .py file (or gym registry), runs N random rollouts,
collects observation statistics, termination patterns, and info keys.

Usage:
    python env_explorer.py --env-py envs/BipedalWalker-v3/env.py
    python env_explorer.py --gym-env BipedalWalker-v3    # standard gym env
"""

import argparse
import importlib.util
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import gymnasium as gym


def load_env_from_file(env_py_path: str):
    """Load environment class from a .py file (no gym registration needed)."""
    p = Path(env_py_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"env.py not found: {p}")

    spec = importlib.util.spec_from_file_location("_env_module", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Find the first gym.Env subclass
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and issubclass(obj, gym.Env) and obj is not gym.Env:
            return obj
    raise ValueError(f"No gym.Env subclass found in {p}")


def explore_from_file(env_py_path: str, n_episodes: int = 30,
                      max_steps: int = 0, seed: int = 42) -> dict:
    """Explore environment loaded from a .py file."""
    env_class = load_env_from_file(env_py_path)
    env = env_class()
    result = _explore(env, n_episodes, max_steps, seed)
    result["env_id"] = Path(env_py_path).resolve().parent.name
    return result


def explore_from_gym(env_id: str, n_episodes: int = 30,
                     max_steps: int = 0, seed: int = 42) -> dict:
    """Explore a standard gym environment."""
    env = gym.make(env_id)
    return _explore(env, n_episodes, max_steps, seed)


def _explore(env, n_episodes: int, max_steps: int, seed: int) -> dict:
    """Core exploration logic."""
    obs_space = env.observation_space
    act_space = env.action_space
    is_cont_obs = hasattr(obs_space, "shape") and len(obs_space.shape) == 1
    obs_dim = obs_space.shape[0] if is_cont_obs else None

    # Auto-detect max steps
    if max_steps <= 0:
        max_steps = getattr(env, '_max_episode_steps', 1000)
        spec = getattr(env, 'spec', None)
        if spec and spec.max_episode_steps:
            max_steps = spec.max_episode_steps

    obs_min = obs_max = obs_sum = obs_sum2 = None
    obs_count = 0
    ep_lengths, term_reasons = [], []
    info_key_values = defaultdict(list)
    rng = np.random.default_rng(seed)

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        step, done, reason = 0, False, "max_steps"

        if is_cont_obs:
            a = np.asarray(obs, dtype=float)
            obs_min = a.copy() if obs_min is None else np.minimum(obs_min, a)
            obs_max = a.copy() if obs_max is None else np.maximum(obs_max, a)
            obs_sum = a.copy() if obs_sum is None else obs_sum + a
            obs_sum2 = a**2 if obs_sum2 is None else obs_sum2 + a**2
            obs_count += 1

        while not done and step < max_steps:
            obs, _reward, terminated, truncated, info = env.step(act_space.sample())
            if is_cont_obs:
                a = np.asarray(obs, dtype=float)
                obs_min = np.minimum(obs_min, a)
                obs_max = np.maximum(obs_max, a)
                obs_sum += a
                obs_sum2 += a**2
                obs_count += 1
            for k, v in info.items():
                if any(x in k.lower() for x in ("reward", "fitness", "score")):
                    continue
                if isinstance(v, (int, float, bool, np.floating, np.integer)):
                    info_key_values[k].append(
                        bool(v) if isinstance(v, (bool, np.bool_)) else
                        round(float(v), 6) if isinstance(v, (float, np.floating)) else
                        int(v) if isinstance(v, (int, np.integer)) else v
                    )
            step += 1
            if terminated: reason = "terminated"; done = True
            elif truncated: reason = "truncated"; done = True

        ep_lengths.append(step)
        term_reasons.append(reason)

    env.close()

    # Build obs stats
    obs_stats = []
    if is_cont_obs and obs_count > 0:
        mean_arr = obs_sum / obs_count
        std_arr = np.sqrt(np.maximum(obs_sum2 / obs_count - mean_arr**2, 0))
        low_arr = np.asarray(obs_space.low, dtype=float)
        high_arr = np.asarray(obs_space.high, dtype=float)
        for d in range(obs_dim):
            obs_stats.append({
                "dim": d,
                "space_low": _safe_float(low_arr[d]),
                "space_high": _safe_float(high_arr[d]),
                "mean": _safe_float(mean_arr[d]),
                "std": _safe_float(std_arr[d]),
                "sample_min": _safe_float(obs_min[d]),
                "sample_max": _safe_float(obs_max[d]),
            })

    term_counts = {r: term_reasons.count(r) for r in set(term_reasons)}
    info_summary = {}
    for k, vals in info_key_values.items():
        arr = np.array([v for v in vals if isinstance(v, (int, float))], dtype=float)
        info_summary[k] = {
            "appears_in_n_steps": len(vals),
            "type": type(vals[0]).__name__ if vals else "unknown",
            "min": _safe_float(arr.min()),
            "max": _safe_float(arr.max()),
        }

    ep_arr = np.array(ep_lengths)

    # Run zero-action baseline to detect passive dynamics
    try:
        zero_action_data = run_zero_action_baseline(env, n_episodes=min(10, n_episodes), max_steps=max_steps)
    except Exception:
        zero_action_data = {"error": "zero-action baseline failed"}

    return {
        "zero_action": zero_action_data.get("zero_action", {}),
        "env_id": getattr(env, 'spec', None).id if getattr(env, 'spec', None) else "custom",
        "spaces": {
            "observation": {"shape": list(obs_space.shape) if hasattr(obs_space, "shape") else None,
                           "low": _safe_float(np.min(obs_space.low)) if hasattr(obs_space, "low") else None,
                           "high": _safe_float(np.max(obs_space.high)) if hasattr(obs_space, "high") else None},
            "action": {"type": type(act_space).__name__,
                      "shape": list(act_space.shape) if hasattr(act_space, "shape") else None,
                      "n": int(act_space.n) if hasattr(act_space, "n") else None},
        },
        "obs_dim": obs_dim,
        "episode_length_stats": {
            "mean": round(float(ep_arr.mean()), 2),
            "std": round(float(ep_arr.std()), 2),
            "min": int(ep_arr.min()),
            "max": int(ep_arr.max()),
        },
        "termination_summary": {k: {"count": v, "fraction": round(v / n_episodes, 3)}
                                 for k, v in term_counts.items()},
        "info_keys": info_summary,
        "obs_dim_stats": obs_stats,
        "max_episode_steps": max_steps,
    }


def run_zero_action_baseline(env, n_episodes: int = 10, max_steps: int = 0) -> dict:
    """Run episodes with zero action to detect passive dynamics.

    Critical for understanding whether the environment has gravity,
    requires continuous action to stay alive, etc.

    Returns dict with zero_action key matching the exploration schema.
    """
    if max_steps <= 0:
        max_steps = getattr(env, '_max_episode_steps', 1000)
        spec = getattr(env, 'spec', None)
        if spec and spec.max_episode_steps:
            max_steps = spec.max_episode_steps

    # Determine zero action based on action space
    act_space = env.action_space
    if hasattr(act_space, "shape") and act_space.shape:
        zero_action = np.zeros(act_space.shape, dtype=np.float32)
    elif hasattr(act_space, "n"):
        zero_action = 0
    else:
        zero_action = 0

    ep_lengths = []
    term_reasons = []
    rng = np.random.default_rng(42)

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        step, done, reason = 0, False, "max_steps"

        while not done and step < max_steps:
            obs, _reward, terminated, truncated, _info = env.step(zero_action)
            step += 1
            if terminated:
                reason = "terminated"
                done = True
            elif truncated:
                reason = "truncated"
                done = True

        ep_lengths.append(step)
        term_reasons.append(reason)

    max_st = max_steps
    early_deaths = sum(1 for l in ep_lengths if l < max_st * 0.2)

    return {
        "zero_action": {
            "n_episodes": n_episodes,
            "mean_length": round(float(np.mean(ep_lengths)), 2),
            "min_length": int(min(ep_lengths)),
            "max_length": int(max(ep_lengths)),
            "early_deaths": early_deaths,
            "death_rate": round(early_deaths / n_episodes, 3),
            "termination_reasons": {
                r: term_reasons.count(r) for r in set(term_reasons)
            },
            "gravity_hypothesis": (
                "strong" if early_deaths / n_episodes > 0.5
                else "weak" if early_deaths > 0
                else "none"
            ),
        }
    }


def _safe_float(v):
    if isinstance(v, (np.floating, float)):
        if math.isnan(v): return "nan"
        if math.isinf(v): return "+inf" if v > 0 else "-inf"
        return round(float(v), 6)
    return v


def extract_compute_reward_signature(step_source: str) -> str:
    """Extract compute_reward(..) argument list from step() source."""
    m = re.search(r'self\.compute_reward\(([^)]+)\)', step_source)
    if m:
        return m.group(1).strip()
    return "action"  # fallback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-py", type=str, default=None, help="Path to env.py")
    parser.add_argument("--gym-env", type=str, default=None, help="Gym env ID")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    if args.env_py:
        desc = explore_from_file(args.env_py, args.episodes, seed=args.seed)
    elif args.gym_env:
        desc = explore_from_gym(args.gym_env, args.episodes, seed=args.seed)
    else:
        parser.print_help()
        exit(1)

    print(json.dumps(desc, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(desc, indent=2), encoding="utf-8")
        print(f"\nSaved → {args.out}")
