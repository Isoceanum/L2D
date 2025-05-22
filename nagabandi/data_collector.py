import os
import time
import numpy as np
import gymnasium as gym

from stable_baselines3.common.monitor import Monitor
from perturbation.wrapper import PerturbationWrapper
from perturbation.WheelLockPerturbation import WheelLockPerturbation

def make_env(perturbation_prob=1.0):
    env = gym.make("L2D-v0", render_mode=None)
    env = Monitor(env)
    env = PerturbationWrapper(
        env,
        perturbation_pool=[WheelLockPerturbation(omega_scale=0.80, wheel_idx=2)],
        perturbation_prob=perturbation_prob,
    )
    return env



def collect_transitions(num_episodes: int, save_path: str):
    env = make_env()
    all_transitions = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            transition = {
                "obs": obs,
                "action": action,
                "next_obs": next_obs,
            }
            all_transitions.append(transition)

            obs = next_obs
            
            
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"transitions.npz")

    obs_array = np.array([t["obs"] for t in all_transitions])
    action_array = np.array([t["action"] for t in all_transitions])
    next_obs_array = np.array([t["next_obs"] for t in all_transitions])

    np.savez_compressed(
        file_path,
        obs=obs_array,
        actions=action_array,
        next_obs=next_obs_array,
    )

    print(f"✅ Saved {len(all_transitions)} transitions to {file_path}")