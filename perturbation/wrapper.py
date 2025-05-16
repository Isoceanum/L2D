import random
import gymnasium as gym


class PerturbationWrapper(gym.Wrapper):
    def __init__(self, env, perturbation_pool=None, seed=None, perturbation_prob=1.0):
        super().__init__(env)
        
        self.perturbation_pool = perturbation_pool
        self.perturbation = None
        self.current_step = 0
        self.rng = random.Random(seed)
        self.perturbation_prob = perturbation_prob

    def reset(self, **kwargs):
        self.env.unwrapped.l2d_active_perturbation = None
        self.current_step = 0
        self.perturbation = None
        
        if self.rng.random() < self.perturbation_prob:
            self.perturbation = self.rng.choice(self.perturbation_pool)
            self.perturbation.reset()
        return self.env.reset(**kwargs)

    def step(self, action):
        self.current_step += 1
        
        if self.perturbation and not self.perturbation.active:
            if self.perturbation.should_activate(self.current_step):
                self.perturbation.apply(self.env)
                self.perturbation.active = True
                
        return self.env.step(action)