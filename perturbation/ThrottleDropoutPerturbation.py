from perturbation.BasePerturbation import Perturbation

class ThrottleDropoutPerturbation(Perturbation):
    def __init__(self, start_step=100, duration=100):
        """
        Args:
            start_step (int): when to start the dropout
            duration (int): how many steps to keep gas disabled
        """
        super().__init__(name="throttle_dropout", start_step=start_step, duration=duration)

    def apply(self, env):
        base_env = env.unwrapped
        original_gas_fn = base_env.car.gas

        def patched_gas(gas_value):
            base_env.car.wheels[2].gas = 0.0
            base_env.car.wheels[3].gas = 0.0

        def step_hook():
            if self.is_expired(base_env.l2d_step_count):
                base_env.car.gas = original_gas_fn
                base_env.l2d_perturbation_step = None
                return

            base_env.car.gas = patched_gas

        base_env.l2d_perturbation_step = step_hook
        env.unwrapped.l2d_active_perturbation = self
