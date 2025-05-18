import os
import csv
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from perturbation.wrapper import PerturbationWrapper
from perturbation.WheelLockPerturbation import WheelLockPerturbation

def make_perturbed_env(perturbation_prob=1.0):
    env = gym.make("L2D-v0", render_mode=None)
    env = Monitor(env)
    env = PerturbationWrapper(
        env,
        perturbation_pool=[WheelLockPerturbation(omega_scale=0.80, wheel_idx=2)],
        perturbation_prob=perturbation_prob,
    )
    return env

def evaluate(model_path, output_csv, n_episodes=100):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = PPO.load(model_path)
    env = make_perturbed_env()

    print(f"✅ Loaded model: {model_path}")
    print(f"🎮 Evaluating on {n_episodes} episodes with perturbation...")

    rows = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated

        rows.append({
            "episode": episode,
            "total_reward": total_reward,
            "steps": step_count,
        })
        print(f"  • Episode {episode+1}: reward={total_reward:.2f}, steps={step_count}")

    env.close()

    # Write to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "total_reward", "steps"])
        writer.writeheader()
        writer.writerows(rows)

    rewards = [r["total_reward"] for r in rows]
    print("\n✅ Evaluation complete.")
    print(f"📈 Mean reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"📄 Results saved to: {output_csv}")

