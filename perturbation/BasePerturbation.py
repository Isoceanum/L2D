class Perturbation:
    def __init__(self, name, start_step, duration=None):
        self.name = name
        self.start_step = start_step
        self.duration = duration
    
    def should_activate(self, step: int) -> bool:
        return step == self.start_step  # or range logic
    
    def apply(self, env):
        """Override this in subclasses to apply the actual perturbation."""
        raise NotImplementedError
    
    def is_expired(self, step: int) -> bool:
        if self.duration is None:
            return False
        return step > self.start_step + self.duration

    def reset(self):
        """Reset internal state, if needed. Called on env.reset()."""
        self.active = False
        
    def __str__(self):
        exclude = {"name", "active"}
        param_str = ", ".join(
            f"{k}={v}" for k, v in self.__dict__.items()
            if k not in exclude and not k.startswith("_") and v is not None
        )
        return f"{self.__class__.__name__}({param_str})"



