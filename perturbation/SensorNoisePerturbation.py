from perturbation.BasePerturbation import Perturbation
import numpy as np


OBS_INDEX_MAP = {
    "front": 0,
    "left": 1,
    "right": 2,
    "speed": 3,
    "ang_vel": 4,
    "steering": 5,
    "gas": 6,
    "brake": 7,
}


class SensorNoisePerturbation(Perturbation):
    def __init__(self, start_step=100, targets=None, noise_std=0.5, duration=None):
        """
        Args:
            start_step (int): when to start applying noise
            targets (list of str): observation names to perturb (e.g., "front", "speed", "gas")
            noise_std (float): standard deviation of Gaussian noise
            duration (int): how long the noise is applied
        """
        super().__init__(name="sensor_noise", start_step=start_step, duration=duration)
        self.targets = targets or ["front"]
        self.noise_std = noise_std

    def apply(self, env):
        base_env = env.unwrapped
        print(f"Applying sensor noise to: {self.targets}")  

        def filter_fn(obs):
            if self.is_expired(base_env.l2d_step_count):
                base_env.l2d_observation_filter = None
                return obs

            noisy_obs = obs.copy()
            for name in self.targets:
                idx = OBS_INDEX_MAP.get(name)
                if idx is not None:
                    noisy_obs[idx] += np.random.normal(0.0, self.noise_std)
            return noisy_obs

        base_env.l2d_observation_filter = filter_fn
        env.unwrapped.l2d_active_perturbation = self

