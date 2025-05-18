import os
import random
import time
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor



from stable_baselines3 import PPO

from perturbation.Factory import generate_training_pool_v1
from perturbation.WheelLockPerturbation import WheelLockPerturbation
from perturbation.wrapper import PerturbationWrapper
from config import MAX_EPISODE_STEPS, TIMESTEPS

# --- Register your custom Learn2Drive environment ---
register(
    id="L2D-v0",
    entry_point="l2d.learn2drive:Learn2Drive",
    max_episode_steps=MAX_EPISODE_STEPS,
)


def run_nominal(output_dir: str):
    # Create env with perturbation wrapper before vectorizing
    def make_env():
        rng = random.Random()
        perturbation_pool = generate_training_pool_v1(rng, size=5)
        
        env = gym.make("L2D-v0")
        env = Monitor(env)
        env = PerturbationWrapper(
            env,
            perturbation_pool=perturbation_pool,
            perturbation_prob=0
        )
        return env

    env = DummyVecEnv([make_env])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        tensorboard_log=output_dir,
    )

    # Start timing
    start = time.time()

    model.learn(
        total_timesteps=TIMESTEPS,
        progress_bar=True
    )
    
    elapsed = time.time() - start
    model_path = os.path.join(output_dir, "model.zip")
    model.save(model_path)

    # Evaluate model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)

    # Print summary
    print("✅ Training complete.")
    print(f"⏱️  Duration: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
    print(f"📈 Mean reward: {mean_reward:.1f} ± {std_reward:.1f}")
    print(f"💾 Model saved to: {model_path}")


def run_perturbation(output_dir: str):
    # Create env with perturbation wrapper before vectorizing
    def make_env():
        rng = random.Random()
        perturbation_pool = generate_training_pool_v1(rng, size=5)
        
        env = gym.make("L2D-v0")
        env = Monitor(env)
        env = PerturbationWrapper(
            env,
            perturbation_pool=[WheelLockPerturbation(omega_scale=0.80, wheel_idx=2)],
            perturbation_prob=0.3,  # 30% chance per episode
        )
        return env

    env = DummyVecEnv([make_env])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        tensorboard_log=output_dir,
    )

    # Start timing
    start = time.time()

    model.learn(
        total_timesteps=TIMESTEPS,
        progress_bar=True
    )
    
    elapsed = time.time() - start
    model_path = os.path.join(output_dir, "model.zip")
    model.save(model_path)

    # Evaluate model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)

    # Print summary
    print("✅ Training complete.")
    print(f"⏱️  Duration: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
    print(f"📈 Mean reward: {mean_reward:.1f} ± {std_reward:.1f}")
    print(f"💾 Model saved to: {model_path}")
