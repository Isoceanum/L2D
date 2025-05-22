import torch
import gymnasium as gym
import numpy as np
from nagabandi.dynamics_model import DynamicsModel
from nagabandi.mpc_controller import MPCController, reward_fn
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

def evaluate_mpc(model_path: str, n_episodes: int = 5):
    env = make_env(perturbation_prob=0.0)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = DynamicsModel(obs_dim, act_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    controller = MPCController(model=model)
    
    returns = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = controller.act(obs, reward_fn)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        returns.append(total_reward)
        print(f"Episode {episode + 1}: total_reward = {total_reward:.2f}")
        
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    print(f"\n✅ MPC Evaluation complete over {n_episodes} episodes.")
    print(f"📈 Mean return: {mean_return:.2f} ± {std_return:.2f}")
        
        
