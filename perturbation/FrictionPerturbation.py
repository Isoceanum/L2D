from perturbation.BasePerturbation import Perturbation

class FrictionPerturbation(Perturbation):
    def __init__(self, start_step=100, duration=None, road_friction=-100.0):
        """
        Args:
            start_step (int): timestep when rain begins
            road_friction (float): new road friction value
            duration (int or None): how long rain lasts; None = full episode
        """
        super().__init__(name="rain", start_step=start_step, duration=duration)
        self.road_friction = road_friction
        
        
    def apply(self, env):
        base_env = env.unwrapped
        original_frictions = {tile: tile.road_friction for tile in base_env.road}

        def step_hook():
            if self.is_expired(base_env.l2d_step_count):
                # Restore original friction values
                for tile, orig in original_frictions.items():
                    tile.road_friction = orig
                base_env.l2d_perturbation_step = None
                return

            # Apply reduced friction
            for tile in base_env.road:
                tile.road_friction = self.road_friction 

        base_env.l2d_perturbation_step = step_hook
        env.unwrapped.l2d_active_perturbation = self
