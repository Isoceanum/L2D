# constants.py

L2D_CATEGORY_WALL = 0x0002
L2D_CATEGORY_CAR  = 0x0004
L2D_RAY_LENGTH = 20.0  # Max raycast length in meters
L2D_OBSERVATION_SIZE = 10

# --- L2D Reward Weights ---
L2D_SPEED_REWARD_WEIGHT = 0.1     # Reward per m/s of forward velocity
L2D_TIME_PENALTY = 0.1            # Constant penalty per step (time cost)
L2D_CENTERING_REWARD_WEIGHT = 0.5   # Centering reward weight
L2D_WALL_PROXIMITY_PENALTY_WEIGHT = 1 # Penalty for being close to walls


L2D_CENTER_REWARD_WEIGHT = 1   # Centering reward weight
L2D_ALIGNMENT_ZONE_RATIO = 0.3 # Zone within which alignment is "perfect"
L2D_SPEED_REWARD_WEIGHT = 0.05 # Reward per m/s of forward velocity