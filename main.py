import os
import time
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from config import *

# Register your custom vectorized env
gym.envs.registration.register(
    id="L2D-v0",
    entry_point="learn2drive:Learn2Drive",
    max_episode_steps=1000,
    reward_threshold=900,
)

# --- CLI args ---
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

# --- Ensure output dir exists ---
os.makedirs(args.output_dir, exist_ok=True)

# --- Create vectorized env ---
env = make_vec_env(
	"L2D-v0",
	n_envs=1, # Use 1 env for now to debug behavior
)

# --- Set up PPO model for vector obs ---
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    n_steps=1024,
    batch_size=256,
    device="cpu",  # Will fall back to CPU if not available
    tensorboard_log=args.output_dir,
)

print(f"🚀 Using device: {model.device}")

# --- Train ---
start = time.time()
model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
elapsed = time.time() - start

# --- Save model ---
model_path = os.path.join(args.output_dir, "model")
model.save(model_path)

# --- Evaluate ---
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
print("✅ Training complete.")
print(f"⏱️  Training time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
print(f"📈 Mean reward: {mean_reward:.1f} ± {std_reward:.1f}")
print(f"💾 Saved to: {model_path}")