import random
from perturbation.BasePerturbation import Perturbation
from perturbation.FrictionPerturbation import FrictionPerturbation
from perturbation.SensorNoisePerturbation import SensorNoisePerturbation
from perturbation.SteeringInertiaPerturbation import SteeringInertiaPerturbation
from perturbation.SteeringScalePerturbation import SteeringScalePerturbation
from perturbation.ThrottleDropoutPerturbation import ThrottleDropoutPerturbation
from perturbation.TorqueJitterPerturbation import TorqueJitterPerturbation
from perturbation.WheelLockPerturbation import WheelLockPerturbation
from perturbation.WindPerturbation import WindPerturbation


# Sample a value from a coarse bucket list with added local jitter
def _sample_jittered_bucket(rng: random.Random, buckets: list[float], jitter: float) -> float:
    return rng.choice(buckets) + rng.uniform(-jitter, jitter)

def make_friction_perturbation(rng: random.Random) -> Perturbation:
    return FrictionPerturbation(
        start_step = rng.randint(200, 300),
        road_friction = _sample_jittered_bucket(rng, buckets=[-150.0, -100.0, -75.0, -50.0, -25.0], jitter=5.0),
    )
    
def make_sensor_noise_perturbation(rng: random.Random) -> Perturbation:
    return SensorNoisePerturbation(
        start_step = rng.randint(200, 300),
        targets=rng.choice(["front","left", "right","speed","ang_vel","steering", "gas", "brake"]),
        noise_std = rng.uniform(0.3, 1.0)
    )
    
def make_steering_inertia_perturbation(rng: random.Random) -> Perturbation:
    return SteeringInertiaPerturbation(
        start_step = rng.randint(200, 300),
        inertia_factor = rng.uniform(0.02, 0.15),
    )
    
def make_steering_scale_perturbation(rng: random.Random) -> Perturbation:
    which = rng.choice(["left", "right"])
    return SteeringScalePerturbation(
        start_step = rng.randint(200, 300),
        scale_left = rng.uniform(0.0, 0.5) if which == "left" else 1.0,
        scale_right = rng.uniform(0.0, 0.5) if which == "right" else 1.0
    )
    
    
def make_throttle_dropout_perturbation(rng: random.Random) -> Perturbation:
    return ThrottleDropoutPerturbation(
        start_step = rng.randint(200, 300),
    )
    
def make_torque_jitter_perturbation(rng: random.Random) -> Perturbation:
    return TorqueJitterPerturbation(
        start_step = rng.randint(200, 300),
        noise_strength = rng.uniform(1.0, 6.0)
    )
    
def make_wheel_lock_perturbation(rng: random.Random) -> Perturbation:
    return WheelLockPerturbation(
        start_step = rng.randint(200, 300),
        omega_scale = rng.uniform(0.2, 0.6),
        wheel_idx = rng.choice([0, 1, 2, 3])
    )
    
    
def make_wind_perturbation(rng: random.Random) -> Perturbation:
    return WindPerturbation(
        start_step = rng.randint(200, 300),
        direction_deg = rng.choice([0, 90, 180, 270]),
        strength = rng.uniform(300, 800)
    )
    
    
def generate_training_pool(rng: random.Random, size) -> list[Perturbation]:
    """
    Generates a pool of perturbations for training.
    
    Args:
        rng (random.Random): Random number generator for reproducibility.
        size (int): Number of perturbations to generate.
        
    Returns:
        list[Perturbation]: List of generated perturbations.
    """
    perturbations = []
    for _ in range(size):
        perturbation_type = rng.choice([
            make_friction_perturbation,
            make_sensor_noise_perturbation,
            make_steering_inertia_perturbation,
            make_steering_scale_perturbation,
            make_throttle_dropout_perturbation,
            make_torque_jitter_perturbation,
            make_wheel_lock_perturbation,
            make_wind_perturbation
        ])
        perturbations.append(perturbation_type(rng))
    return perturbations



def generate_evaluation_pool() -> list[Perturbation]:
    return [
        FrictionPerturbation(start_step=200, road_friction=-100),
        WindPerturbation(start_step=200, direction_deg=180, strength=600),
        SteeringScalePerturbation(start_step=200, scale_left=1.0, scale_right=0.3),
        WheelLockPerturbation(start_step=200, omega_scale=0.3, wheel_idx=1),
        SensorNoisePerturbation(start_step=200, targets="front", noise_std=0.8),
        ]
