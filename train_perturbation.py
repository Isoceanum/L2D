import os
import random
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import torch

from config import MAX_EPISODE_STEPS, TIMESTEPS

# --- Register Learn2Drive environment ---
gym.envs.registration.register(
    id="L2D-v0",
    entry_point="learn2drive:Learn2Drive",  # Update this if module is renamed
    max_episode_steps=MAX_EPISODE_STEPS,
    reward_threshold=900,
)

# Root directory where all runs will be saved
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "local_models"))
os.makedirs(models_dir, exist_ok=True)

# Find the next available run folder: 000000, 000001, ...
existing = [d for d in os.listdir(models_dir) if d.isdigit()]
existing = sorted(int(name) for name in existing)
next_id = (max(existing) + 1) if existing else 0

# Format as 6-digit zero-padded string
run_name = f"{next_id:04d}"
output_dir = os.path.join(models_dir, run_name)
os.makedirs(output_dir, exist_ok=True)

# --- Create env (1 for now) ---
env = make_vec_env(
    "L2D-v0",
    n_envs=1
)

env = gym.make("L2D-v0", render_mode="rgb_array")
env.reset()

# --- Set up PPO ---
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    n_steps=1024,
    batch_size=256,
    device="cuda",  # ✅ Force CPU on MacBook
    tensorboard_log=output_dir,
)

print(f"🚀 Using device: {model.device}")

# --- Train ---
start = time.time()
model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
elapsed = time.time() - start

# --- Save model ---
model_path = os.path.join(output_dir, "model")
model.save(model_path)

# --- Evaluate ---
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print("✅ Training complete.")
print(f"⏱️  Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
print(f"📈 Mean reward: {mean_reward:.1f} ± {std_reward:.1f}")
print(f"💾 Saved to: {model_path}")

base_dir = os.path.dirname(__file__)  # Directory containing this script
description_path = os.path.join(base_dir, "description.txt")
config_path = os.path.join(base_dir, "config.py")
current_date = time.strftime("%a %b %d %H:%M:%S %Z %Y")

out_path = os.path.join(output_dir, "out.txt")

# Read contents
with open(description_path, "r") as f:
    description = f.read().strip()

with open(config_path, "r") as f:
    config_text = f.read().strip()

# Format and write
content = f"""
================ DESCRIPTION ================
{description}

================ CONFIG ================

{config_text}

================ TRAINING SUMMARY ================
✅ Training complete.
🕓 Date: {current_date}
⏱️  Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}
📈 Mean reward: {mean_reward:.1f} ± {std_reward:.1f}
💾 Saved to: {model_path}

"""

with open(out_path, "w") as f:
    f.write(content)