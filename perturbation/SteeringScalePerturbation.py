from perturbation.BasePerturbation import Perturbation


class SteeringScalePerturbation(Perturbation):
    def __init__(self, start_step=100, duration=None, scale_left=1.0, scale_right=1.0):
        """
        Args:
            start_step (int): when to start
            scale_left (float): scaling factor for left wheel steer (0.0–1.0)
            scale_right (float): scaling factor for right wheel steer (0.0–1.0)
            duration (int or None): how long it lasts
        """
        super().__init__(name="steer_scale", start_step=start_step, duration=duration)
        self.scale_left = scale_left
        self.scale_right = scale_right

    def apply(self, env):
        base_env = env.unwrapped
        original_steer_fn = base_env.car.steer

        def scaled_steer(target):
            original_steer_fn(target)
            base_env.car.wheels[0].steer *= self.scale_left
            base_env.car.wheels[1].steer *= self.scale_right

        def step_hook():
            if self.is_expired(base_env.l2d_step_count):
                base_env.car.steer = original_steer_fn
                base_env.l2d_perturbation_step = None
                return

            base_env.car.steer = scaled_steer

        base_env.l2d_perturbation_step = step_hook
        env.unwrapped.l2d_active_perturbation = self
