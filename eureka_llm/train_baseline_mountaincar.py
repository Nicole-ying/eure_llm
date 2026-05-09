"""
train_baseline_mountaincar.py — Train PPO baseline with official reward.

Same hyperparameters as ppo_1M_mountaincar_rlzoo.yaml but uses the
official MountainCarContinuous-v0 reward function (no LLM injection).
"""
import gymnasium as gym
import numpy as np
import yaml
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "configs" / "ppo_1M_mountaincar_rlzoo.yaml"
OUTPUT_DIR = HERE / "runs" / "mountaincar_baseline_official"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

ppo_cfg = cfg["ppo"]
n_envs = cfg.get("n_envs", 4)
total_steps = cfg["total_timesteps"]
seed = cfg.get("seed", 42)

# ── Env factory (official reward, no LLM injection) ──
def make_env(rank):
    def _init():
        env = gym.make("MountainCarContinuous-v0")
        env = Monitor(env, filename=str(OUTPUT_DIR / f"monitor_{rank}"))
        return env
    return _init

env = DummyVecEnv([make_env(i) for i in range(n_envs)])
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

# ── Model ──
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
    use_sde=ppo_cfg.get("use_sde", False),
    sde_sample_freq=ppo_cfg.get("sde_sample_freq", -1),
    policy_kwargs=eval(ppo_cfg.get("policy_kwargs", "{}")),
    seed=seed,
    device=cfg.get("device", "cpu"),
    verbose=1,
)

print(f"\n{'='*60}")
print(f"  Training MountainCarContinuous-v0 BASELINE (official reward)")
print(f"  Steps: {total_steps:,} | n_envs: {n_envs} | device: {cfg.get('device')}")
print(f"  PPO: n_steps={ppo_cfg['n_steps']}, batch_size={ppo_cfg['batch_size']}")
print(f"{'='*60}\n")

model.learn(total_timesteps=total_steps)

model.save(str(OUTPUT_DIR / "model_final"))
env.save(str(OUTPUT_DIR / "vecnormalize.pkl"))
print(f"\nModel saved → {OUTPUT_DIR / 'model_final.zip'}")

# ── Evaluate ──
print("\n" + "="*60)
print("  Evaluating on official environment (10 episodes)...")
print("="*60)

eval_env = VecNormalize.load(
    str(OUTPUT_DIR / "vecnormalize.pkl"),
    DummyVecEnv([lambda: gym.make("MountainCarContinuous-v0")])
)
eval_env.training = False
eval_env.norm_reward = False

rewards = []
for ep in range(10):
    obs = eval_env.reset()
    done = False
    total = 0.0
    length = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _ = eval_env.step(action)
        total += r[0]
        length += 1
    rewards.append(total)
    print(f"  Episode {ep+1:>2}: reward = {total:>8.2f}, length = {length}")

print(f"\n  Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
print(f"  Baseline result saved to: {OUTPUT_DIR}")

env.close()
eval_env.close()
