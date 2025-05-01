# constants.py


import math

# --- Faulty car Constants ---
SIZE = 0.02
ENGINE_POWER = 100_000_000 * SIZE * SIZE
WHEEL_MOMENT_OF_INERTIA = 4000 * SIZE * SIZE
FRICTION_LIMIT = (
    100_000_0 * SIZE * SIZE
)  # friction ~= mass ~= size^2 (calculated implicitly using density)
WHEEL_R = 27
WHEEL_W = 14
WHEELPOS = [(-55, +80), (+55, +80), (-55, -82), (+55, -82)]
HULL_POLY1 = [(-60, +130), (+60, +130), (+60, +110), (-60, +110)]
HULL_POLY2 = [(-15, +120), (+15, +120), (+20, +20), (-20, 20)]
HULL_POLY3 = [
    (+25, +20),
    (+50, -10),
    (+50, -40),
    (+20, -90),
    (-20, -90),
    (-50, -40),
    (-50, -10),
    (-25, +20),
]
HULL_POLY4 = [(-50, -120), (+50, -120), (+50, -90), (-50, -90)]
WHEEL_COLOR = (0, 0, 0)
WHEEL_WHITE = (77, 77, 77)
MUD_COLOR = (102, 102, 0)


# --- L2D Constants ---
STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = (max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE)

# --- Car Custom config ---
STEERING_INERTIA = 1
FIXED_THROTTLE = None
THROTTLE_BRAKE_RATIO = None #0.2  # Ratio of throttle to brake

# --- L2D Custome configs ---
L2D_CATEGORY_WALL = 0x0002
L2D_CATEGORY_CAR  = 0x0004

L2D_RAY_NOISE_STD_DEV = 0.3
L2D_RAY_ROUND_DIGITS = 1
L2D_RAY_LENGTH = 20.0  # Max raycast length in meters
L2D_OBSERVATION_SIZE = 10

# --- L2D Reward Weights ---
L2D_SPEED_REWARD_WEIGHT = 0.1     # Reward per m/s of forward velocity
L2D_TIME_PENALTY = 0.1            # Constant penalty per step (time cost)
L2D_CENTERING_REWARD_WEIGHT = 0.5   # Centering reward weight
L2D_WALL_PROXIMITY_PENALTY_WEIGHT = 1 # Penalty for being close to walls
STEERING_JITTER_WEIGHT = 5.0 # Penalty for steering jitter


L2D_CENTER_REWARD_WEIGHT = 1   # Centering reward weight
L2D_ALIGNMENT_ZONE_RATIO = 0.3 # Zone within which alignment is "perfect"
L2D_SPEED_REWARD_WEIGHT = 0.05 # Reward per m/s of forward velocity