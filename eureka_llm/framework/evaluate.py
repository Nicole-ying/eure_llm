"""
evaluate.py — Behavior-metric evaluation (independent of reward function).

Uses _episode_terminated from info dict for generic completion detection.
Works with any Gymnasium environment.

Usage:
    python evaluate.py --run-dir path/to/run/ --episodes 100
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def evaluate(run_dir: Path, episodes: int = 100) -> dict:
    """Run behavior evaluation on a trained model."""
    import re, sys
    cfg_path = run_dir / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text("utf-8"))
    env_id = re.sub(r"-round\d+$", "", cfg.get("env_id", ""))
    if not env_id:
        print("ERROR: No env_id in config and no fallback available.")
        sys.exit(1)

    base_env = DummyVecEnv([lambda: gym.make(env_id)])
    vn_path = run_dir / "vecnormalize.pkl"
    if vn_path.exists():
        env = VecNormalize.load(str(vn_path), base_env)
        env.training = False
        env.norm_reward = False
    else:
        env = base_env

    model = PPO.load(run_dir / "model")

    completed = fell = truncated_count = 0
    lengths = []
    obs = env.reset()
    current_length = 0

    while len(lengths) < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, dones, infos = env.step(action)
        current_length += 1

        if dones[0]:
            lengths.append(current_length)
            info = infos[0]
            episode_truncated = info.get("_episode_truncated", False)

            # Read _outcome from reward_components (set by LLM on termination)
            # +1.0 = success, -1.0 = failure/crash, 0.0 = neutral
            reward_comps = info.get("reward_components", {})
            outcome = reward_comps.get("_outcome", None) if isinstance(reward_comps, dict) else None

            if outcome == -1.0:
                fell += 1
            elif outcome == 1.0:
                completed += 1
            elif outcome == 0.0:
                truncated_count += 1
            elif episode_truncated:
                completed += 1
            else:
                truncated_count += 1

            current_length = 0
            obs = env.reset()

    env.close()
    n = len(lengths)
    return {
        "episodes": n,
        "completion_rate": round(completed / n, 4),
        "fall_rate": round(fell / n, 4),
        "truncation_rate": round(truncated_count / n, 4),
        "mean_length": round(float(np.mean(lengths)), 2),
        "std_length": round(float(np.std(lengths)), 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    results = evaluate(Path(args.run_dir), args.episodes)
    print(json.dumps(results, indent=2))

    out_path = Path(args.run_dir) / "behavior_metrics.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved → {out_path}")
