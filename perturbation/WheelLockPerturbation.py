from perturbation.BasePerturbation import Perturbation


class WheelLockPerturbation(Perturbation):
    def __init__(self, start_step=100, duration=None, omega_scale= 0.4, wheel_idx=2):
        super().__init__(name="wheel_lock", start_step=start_step, duration=duration)
        self.wheel_idx = wheel_idx
        self.omega_scale = omega_scale
        

    def apply(self, env):
        base_env = env.unwrapped

        def step_hook():
            if self.is_expired(base_env.l2d_step_count):
                base_env.l2d_perturbation_step = None
                return
            
            w = base_env.car.wheels[self.wheel_idx]
            w.omega *= self.omega_scale

        base_env.l2d_perturbation_step = step_hook
        env.unwrapped.l2d_active_perturbation = self
