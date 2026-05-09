"""
train_baseline_lunarlander_discrete.py — Train PPO baseline (DISCRETE actions).

Uses LunarLander-v2 with default discrete action space (no continuous=True).
Same hyperparameters as ppo_1M_lunarlander.yaml.
"""
import gymnasium as gym
import numpy as np
import yaml
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "configs" / "ppo_1M_lunarlander.yaml"
OUTPUT_DIR = HERE / "runs" / "lunarlander_baseline_discrete"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

ppo_cfg = cfg["ppo"]
n_envs = cfg.get("n_envs", 16)
total_steps = cfg["total_timesteps"]
seed = cfg.get("seed", 42)

# ── Env factory (discrete LunarLander, official reward) ──
def make_env(rank):
    def _init():
        env = gym.make("LunarLander-v2")  # discrete by default
        env = Monitor(env, filename=str(OUTPUT_DIR / f"monitor_{rank}"))
        return env
    return _init

env = DummyVecEnv([make_env(i) for i in range(n_envs)])

model = PPO(
    policy=ppo_cfg["policy"],
    env=env,
    learning_rate=ppo_cfg["learning_rate"],
    n_steps=ppo_cfg["n_steps"],
    batch_size=ppo_cfg["batch_size"],
    n_epochs=ppo_cfg["n_epochs"],
    gamma=ppo_cfg["gamma"],
    gae_lambda=ppo_cfg["gae_lambda"],
    clip_range=ppo_cfg["clip_range"],
    ent_coef=ppo_cfg["ent_coef"],
    vf_coef=ppo_cfg["vf_coef"],
    max_grad_norm=ppo_cfg["max_grad_norm"],
    seed=seed,
    device=cfg.get("device", "cuda"),
    verbose=1,
)

print(f"\n{'='*60}")
print(f"  Training LunarLander-v2 BASELINE (official reward, DISCRETE)")
print(f"  Steps: {total_steps:,} | n_envs: {n_envs} | device: {cfg.get('device')}")
print(f"{'='*60}\n")

model.learn(total_timesteps=total_steps)

model.save(str(OUTPUT_DIR / "model_final"))
print(f"\nModel saved → {OUTPUT_DIR / 'model_final.zip'}")

# ── Quick 10-episode eval ──
print("\n" + "="*60)
print("  Evaluating on LunarLander-v2 (10 episodes, discrete)...")
print("="*60)

eval_env = gym.make("LunarLander-v2")
rewards = []
for ep in range(10):
    obs, _ = eval_env.reset()
    done = False
    total = 0.0
    length = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        total += r
        length += 1
    rewards.append(total)
    print(f"  Episode {ep+1:>2}: reward = {total:>8.2f}, length = {length}")

print(f"\n  Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
print(f"\n  Baseline saved to: {OUTPUT_DIR}")

env.close()
eval_env.close()
