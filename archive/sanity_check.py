import time
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create the environment
env = make_vec_env("CarRacing-v3", n_envs=1, env_kwargs={"render_mode": "rgb_array"})

# Create the PPO model
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    n_steps=512,
    batch_size=64,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Train the model
start = time.time()
model.learn(total_timesteps=1000)
elapsed = time.time() - start

print("✅ Training complete.")
print(f"⏱️  Training time: {elapsed:.1f} seconds")