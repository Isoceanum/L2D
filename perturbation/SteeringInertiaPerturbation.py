from perturbation.BasePerturbation import Perturbation

class SteeringInertiaPerturbation(Perturbation):
    def __init__(self, start_step=1, duration=None, inertia_factor=0.08):
        """
        Args:
            start_step (int): when the perturbation begins
            inertia_factor (float): rate of steering update (lower = more sluggish)
            duration (int or None): how long it lasts
        """
        super().__init__(name="steering_inertia", start_step=start_step, duration=duration)
        self.inertia_factor = inertia_factor

    def apply(self, env):
        base_env = env.unwrapped
        original_steer_fn = base_env.car.steer

        # Keep track of internal smoothed state
        steering_state = {"current": 0.0}

        def inertial_steer(target):
            current = steering_state["current"]
            delta = target - current
            update = self.inertia_factor * delta
            current += update
            steering_state["current"] = current
            original_steer_fn(current)

        def step_hook():
            if self.is_expired(base_env.l2d_step_count):
                base_env.car.steer = original_steer_fn
                base_env.l2d_perturbation_step = None
                return

            base_env.car.steer = inertial_steer
            

        base_env.l2d_perturbation_step = step_hook
        env.unwrapped.l2d_active_perturbation = self
