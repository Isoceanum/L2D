from perturbation.BasePerturbation import Perturbation
import numpy as np

class TorqueJitterPerturbation(Perturbation):
    def __init__(self, start_step=100, duration=None, noise_strength=5.0):
        """
        Args:
            start_step (int): when to start gravel effect
            noise_strength (float): how strong the torque noise is
            duration (int or None): how long it lasts
        """
        super().__init__(name="gravel", start_step=start_step, duration=duration)
        self.noise_strength = noise_strength

    def apply(self, env):
        base_env = env.unwrapped

        def step_hook():
            if self.is_expired(base_env.l2d_step_count):
                base_env.l2d_perturbation_step = None
                return

            # Add noise to rear wheel torque (simulate rough terrain)
            for w in base_env.car.wheels[2:4]:  # rear wheels
                torque_noise = np.random.uniform(-self.noise_strength, self.noise_strength)
                w.omega += torque_noise

        base_env.l2d_perturbation_step = step_hook
        env.unwrapped.l2d_active_perturbation = self
