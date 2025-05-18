# perturbations/wind.py

import numpy as np
from perturbation.BasePerturbation import Perturbation
import Box2D
import math

class WindPerturbation(Perturbation):
    def __init__(self, start_step=100, duration=None, direction_deg=0, strength=400):
        """
        Args:
            start_step (int): timestep when wind begins
            direction_deg (float): direction wind is blowing *from*, in degrees (e.g., 0 = from left)
            strength (float): force magnitude (Newtons)
            duration (int or None): number of steps wind is active, or None for infinite
        """
        super().__init__(name="wind", start_step=start_step, duration=duration)
        self.strength = strength
        self.direction_deg = direction_deg
        self.force_vec = self._compute_force_vec(direction_deg, strength)

    def _compute_force_vec(self, angle_deg, strength):
        # Convert FROM direction to TO direction
        angle_rad = math.radians(angle_deg)
        x = strength * math.cos(angle_rad)
        y = strength * math.sin(angle_rad)
        return Box2D.b2Vec2(x, y)

    def apply(self, env):  
        base_env = env.unwrapped
        def step_hook():
            if self.is_expired(base_env.l2d_step_count):
                base_env.l2d_perturbation_step = None
                return           
            base_env.car.hull.ApplyForceToCenter(self.force_vec, wake=True)
            
        base_env.l2d_perturbation_step = step_hook
        env.unwrapped.l2d_active_perturbation = self

