
import numpy as np

from config import *
from l2d.constants import *
    
def calculate_signed_offset(env) -> float:
    car_pos = np.array(env.car.hull.position)
    car_heading = env.car.hull.GetWorldVector((0, 1))  # car's forward direction
    car_right = np.array([-car_heading[1], car_heading[0]])  # 90 degrees to the right
    
    min_distance = float("inf")
    signed_offset = 0.0
    
    idx0 = env.l2d_last_segment_idx
    window = 5  # try segments 4 from last closest
    
    min_i = max(1, idx0 - window)
    max_i = min(len(env.track), idx0 + window)    
    
    for i in range(min_i, max_i):
        x1, y1 = env.track[i - 1][2:4]
        x2, y2 = env.track[i][2:4]
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        ab = b - a
        ap = car_pos - a

        ab_len_squared = np.dot(ab, ab)
        if ab_len_squared == 0:
            continue

        t = np.clip(np.dot(ap, ab) / ab_len_squared, 0.0, 1.0)
        closest_point = a + t * ab
        diff = car_pos - closest_point
        dist = np.linalg.norm(diff)

        if dist < min_distance:
            min_distance = dist
            signed_offset = np.dot(diff, car_right)  # positive if to the right 
            env.l2d_last_segment_idx = i  # update the cache
            
    return signed_offset

def l2d_calculate_step_reward (mode, env, action) -> float:
    match mode:
        case "baseline":
            return _reward_baseline(env, action)
        case "GOAT":
            return _reward_function_goat(env, action)
        case "center_track_alignment":
            return _reward_center_track_alignment(env, action)
        case "wall_avoidance":
            return _reward_wall_avoidance(env, action)
        case "wall_avoidance_scaled":
            return _reward_scaled_wall_penalty(env, action)
        case "center_track_projection":
            return _reward_center_track_projection(env, action)
        case "reward_align_speed":
            return _reward_align_speed(env, action)
        case "reward_align_speed_with_smoothness":
            return _reward_align_speed_with_smoothness(env, action)
        case "def_reward":
            return _def_reward(env, action)
        case "scaled_wall_penalty":
            return _reward_function_1(env, action)
        case "2":
            return _reward_function_2(env, action)
        case "3":
            return _reward_function_3(env, action)
        case "4":
            return _reward_function_4(env, action)
        case "5":
            return _reward_function_5(env, action)
        case "6":
            return _reward_function_6(env, action)
        case "8":
            return _reward_function_8(env, action) 
        case "9":
            return _reward_function_9(env, action)
        case "10":
            return _reward_function_10(env, action)
        case "11":
            return _reward_function_11(env, action)
        case "12":
            return _reward_function_12(env, action)
        case "13":
            return _reward_function_13(env, action)
        case _:
            raise ValueError(f"Unknown reward function named: {mode}")

# baseline
def _reward_baseline(env, action) -> float:
    if action is None:
        return 0.0
    step_reward = 0.0

    # 1. Apply time penalty
    env.reward -= L2D_TIME_PENALTY
    
    # Indirectly add reward from friction detector for discovering new tiles
    step_reward = env.reward - env.prev_reward
    env.prev_reward = env.reward
    return step_reward

# GOATED 
def _reward_function_goat(env, action) -> float:
    step_reward = 0.0
    
    # Time penalty    
    step_reward -= L2D_TIME_PENALTY
    
    car_pos = np.array(env.car.hull.position)
    car_heading = env.car.hull.GetWorldVector((0, 1))  # car's forward direction
    car_right = np.array([-car_heading[1], car_heading[0]])  # 90 degrees to the right

    min_distance = float("inf")
    signed_offset = 0.0
    
    idx0 = env.l2d_last_segment_idx
    window = 5  # try segments 4 from last closest
    
    min_i = max(1, idx0 - window)
    max_i = min(len(env.track), idx0 + window)

    for i in range(min_i, max_i):
        x1, y1 = env.track[i - 1][2:4]
        x2, y2 = env.track[i][2:4]
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        ab = b - a
        ap = car_pos - a

        ab_len_squared = np.dot(ab, ab)
        if ab_len_squared == 0:
            continue

        t = np.clip(np.dot(ap, ab) / ab_len_squared, 0.0, 1.0)
        closest_point = a + t * ab
        diff = car_pos - closest_point
        dist = np.linalg.norm(diff)

        if dist < min_distance:
            min_distance = dist
            signed_offset = np.dot(diff, car_right)  # positive if to the right 
            env.l2d_last_segment_idx = i  # update the cache
    
    # signed_offset 5,  0,  -5
    TRACK_EDGE_OFFSET = 5.0  # max distance from center
    scaled_center_value = 1.0 - abs(signed_offset) / TRACK_EDGE_OFFSET
    scaled_center_value = np.clip(scaled_center_value, 0.0, 1.0)
    
    if env.car.outside_track:
        step_reward -= L2D_OFF_TRACK_PENALTY
    
    step_reward += scaled_center_value * env.l2d_tile_reward
    env.l2d_tile_reward = 0.0
    
    #update global reward
    env.reward += step_reward

    return step_reward
    
# Experimental reward functions
def _reward_center_track_alignment(env, action) -> float:
    if action is None:
        return 0.0

    step_reward = 0.0

    # Time penalty
    env.reward -= L2D_TIME_PENALTY
    step_reward = env.reward - env.prev_reward
    env.prev_reward = env.reward

    # Get side distances
    left = env.car.l2d_rays.get("left_90", 0.0)
    right = env.car.l2d_rays.get("right_90", 0.0)

    if left > 0 and right > 0:
        # Centering reward
        balance = 1.0 - abs(left - right) / (left + right + 1e-6)
        center_reward = L2D_CENTERING_REWARD_WEIGHT * balance
        step_reward += center_reward

        # Wall proximity penalty (sharp)
        wall_penalty = L2D_WALL_PROXIMITY_PENALTY_WEIGHT * (
            1.0 / (left + 1e-3) + 1.0 / (right + 1e-3)
        )
        step_reward -= wall_penalty
         

    return step_reward

def _reward_wall_avoidance(env, action) -> float:
    if action is None:
        return 0.0

    step_reward = 0.0

    # 1. Apply time penalty
    env.reward -= L2D_TIME_PENALTY

    # 2. Reward from tile discovery (via contact listener)
    step_reward = env.reward - env.prev_reward
    env.prev_reward = env.reward

    # 3. Harsh wall proximity penalty
    left = env.car.l2d_rays.get("left_90", 0.0)
    right = env.car.l2d_rays.get("right_90", 0.0)

    WALL_THRESHOLD = 0.8  # meters — tune for your track width
    WALL_PENALTY_SCALE = L2D_WALL_PROXIMITY_PENALTY_WEIGHT  # from constants

    wall_penalty = 0.0
    if left < WALL_THRESHOLD:

        wall_penalty += WALL_PENALTY_SCALE * (1.0 - left / WALL_THRESHOLD)

    if right < WALL_THRESHOLD:
        wall_penalty += WALL_PENALTY_SCALE * (1.0 - right / WALL_THRESHOLD)

    if wall_penalty > 0:
        step_reward -= wall_penalty

    return step_reward

def _reward_scaled_wall_penalty(env, action) -> float:
    if action is None:
        return 0.0

    step_reward = 0.0

    # 1. Apply time penalty
    env.reward -= L2D_TIME_PENALTY

    # 2. Reward for discovering new tiles
    step_reward = env.reward - env.prev_reward
    env.prev_reward = env.reward

    # 3. Non-linear ramped wall penalty
    left = env.car.l2d_rays.get("left_90", 0.0)
    right = env.car.l2d_rays.get("right_90", 0.0)

    WALL_CRITICAL_DISTANCE = 4.0
    MAX_PENALTY = 10.0

    wall_penalty = 0.0
    
    if left < WALL_CRITICAL_DISTANCE:
        closeness = 1.0 - (left / WALL_CRITICAL_DISTANCE)
        wall_penalty = MAX_PENALTY * closeness**2

    elif right < WALL_CRITICAL_DISTANCE:
        closeness = 1.0 - (right / WALL_CRITICAL_DISTANCE)
        wall_penalty = MAX_PENALTY * closeness**2

    if wall_penalty > 0:
        step_reward -= wall_penalty      
        

    return step_reward

def _reward_center_track_projection(env, action) -> float:
    if action is None:
        return 0.0

    step_reward = 0.0

    # Time penalty
    env.reward -= L2D_TIME_PENALTY
    step_reward = env.reward - env.prev_reward
    env.prev_reward = env.reward

    car_pos = np.array(env.car.hull.position)
    car_heading = env.car.hull.GetWorldVector((0, 1))  # car's forward direction
    car_right = np.array([-car_heading[1], car_heading[0]])  # 90 degrees to the right

    min_distance = float("inf")
    signed_offset = 0.0

    for i in range(1, len(env.track)):
        x1, y1 = env.track[i - 1][2:4]
        x2, y2 = env.track[i][2:4]
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        ab = b - a
        ap = car_pos - a

        ab_len_squared = np.dot(ab, ab)
        if ab_len_squared == 0:
            continue

        t = np.clip(np.dot(ap, ab) / ab_len_squared, 0.0, 1.0)
        closest_point = a + t * ab
        diff = car_pos - closest_point
        dist = np.linalg.norm(diff)

        if dist < min_distance:
            min_distance = dist
            signed_offset = np.dot(diff, car_right)  # positive if to the right 
            
    max_offset = 40 / 6.0  # full width in meters (same as TRACK_WIDTH)

    desired_zone = max_offset * 0.5
    normalized_offset = abs(signed_offset) / desired_zone
    alignment_score = np.clip(1.0 - normalized_offset, -1.0, 1.0)
    

    step_reward += alignment_score * L2D_CENTER_REWARD_WEIGHT
        
    return step_reward

def _reward_align_speed(env, action) -> float:
    if action is None:
        return 0.0

    step_reward = 0.0
    # Time penalty    
    step_reward -= L2D_TIME_PENALTY
    
    car_pos = np.array(env.car.hull.position)
    car_heading = env.car.hull.GetWorldVector((0, 1))  # car's forward direction
    car_right = np.array([-car_heading[1], car_heading[0]])  # 90 degrees to the right

    min_distance = float("inf")
    signed_offset = 0.0
    
    idx0 = env.l2d_last_segment_idx
    window = 5  # try segments 4 from last closest
    
    min_i = max(1, idx0 - window)
    max_i = min(len(env.track), idx0 + window)

    for i in range(min_i, max_i):
        x1, y1 = env.track[i - 1][2:4]
        x2, y2 = env.track[i][2:4]
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        ab = b - a
        ap = car_pos - a

        ab_len_squared = np.dot(ab, ab)
        if ab_len_squared == 0:
            continue

        t = np.clip(np.dot(ap, ab) / ab_len_squared, 0.0, 1.0)
        closest_point = a + t * ab
        diff = car_pos - closest_point
        dist = np.linalg.norm(diff)

        if dist < min_distance:
            min_distance = dist
            signed_offset = np.dot(diff, car_right)  # positive if to the right 
            env.l2d_last_segment_idx = i  # update the cache
            
    max_offset = 40 / 6.0  # full width in meters (same as TRACK_WIDTH)

    desired_zone = max_offset * L2D_ALIGNMENT_ZONE_RATIO
    normalized_offset = abs(signed_offset) / desired_zone
    alignment_reward = np.clip(1.0 - normalized_offset, -1.0, 1.0) * L2D_CENTER_REWARD_WEIGHT
    
    print(normalized_offset)
    
    # Speed reward (scaled down)
    speed_reward = np.linalg.norm(env.car.hull.linearVelocity)
    
    if speed_reward > 10:
        alignment_reward = 0.0

    step_reward += (
    alignment_reward * L2D_CENTER_REWARD_WEIGHT  +
    speed_reward * L2D_SPEED_REWARD_WEIGHT
    )

    print(f"{step_reward:.5f}")
    
    env.reward += step_reward
        
    return step_reward

def _reward_align_speed_with_smoothness(env, action) -> float:
    if action is None:
        return 0.0

    step_reward = 0.0
    
    # Time penalty
    step_reward -= L2D_TIME_PENALTY
    
    # 2. Reward from tile discovery (via contact listener)
    step_reward += env.reward - env.prev_reward
    env.prev_reward = env.reward
    
    car_pos = np.array(env.car.hull.position)
    car_heading = env.car.hull.GetWorldVector((0, 1))  # car's forward direction
    car_right = np.array([-car_heading[1], car_heading[0]])  # 90 degrees to the right

    min_distance = float("inf")
    signed_offset = 0.0
    
    idx0 = env.l2d_last_segment_idx
    window = 3  # try segments 4 from last closest
    
    min_i = max(1, idx0 - window)
    max_i = min(len(env.track), idx0 + window)

    for i in range(min_i, max_i):
        x1, y1 = env.track[i - 1][2:4]
        x2, y2 = env.track[i][2:4]
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        ab = b - a
        ap = car_pos - a

        ab_len_squared = np.dot(ab, ab)
        if ab_len_squared == 0:
            continue

        t = np.clip(np.dot(ap, ab) / ab_len_squared, 0.0, 1.0)
        closest_point = a + t * ab
        diff = car_pos - closest_point
        dist = np.linalg.norm(diff)

        if dist < min_distance:
            min_distance = dist
            signed_offset = np.dot(diff, car_right)  # positive if to the right 
            env.l2d_last_segment_idx = i  # update the cache
            
    max_offset = 40 / 6.0  # full width in meters (same as TRACK_WIDTH)

    desired_zone = max_offset * L2D_ALIGNMENT_ZONE_RATIO
    normalized_offset = abs(signed_offset) / desired_zone
    alignment_reward = np.clip(1.0 - normalized_offset, -1.0, 1.0) * L2D_CENTER_REWARD_WEIGHT

    # Speed reward (scaled down)
    speed = np.linalg.norm(env.car.hull.linearVelocity)
    speed_reward = L2D_SPEED_REWARD_WEIGHT * speed
    
    # jittery steering penalty
    current_target_steer = env.car.target_steer
    prev_target_steer = env.car.prev_steer
    
    jitter_penalty = abs(current_target_steer - prev_target_steer) * STEERING_JITTER_WEIGHT
        
    step_reward += speed_reward * alignment_reward - jitter_penalty
    

    return step_reward

def _def_reward(env, action) -> float:
    if action is None:
        return 0.0

    step_reward = 0.0

    # Time penalty
    env.reward -= L2D_TIME_PENALTY
    step_reward = env.reward - env.prev_reward
    env.prev_reward = env.reward

    car_pos = np.array(env.car.hull.position)
    car_heading = env.car.hull.GetWorldVector((0, 1))  # car's forward direction
    car_right = np.array([-car_heading[1], car_heading[0]])  # 90 degrees to the right
    
    _min_distance = float("inf")
    _signed_offset = 0.0
    
    _idx0 = env.l2d_last_segment_idx
    _window = 4  # try segments 4 from last closest
    
    min_i = max(1, _idx0 - _window)
    max_i = min(len(env.track), _idx0 + _window)

    for i in range(min_i, max_i):
        x1, y1 = env.track[i - 1][2:4]
        x2, y2 = env.track[i][2:4]
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        ab = b - a
        ap = car_pos - a

        ab_len_squared = np.dot(ab, ab)
        if ab_len_squared == 0:
            continue

        t = np.clip(np.dot(ap, ab) / ab_len_squared, 0.0, 1.0)
        closest_point = a + t * ab
        diff = car_pos - closest_point
        dist = np.linalg.norm(diff)

        if dist < _min_distance:
            _min_distance = dist
            _signed_offset = np.dot(diff, car_right)  # positive if to the right 
            env.l2d_last_segment_idx = i  # update the cache
            
    _max_offset = 40 / 6.0  # full width in meters (same as TRACK_WIDTH)
    
   

    desired_zone = _max_offset * L2D_ALIGNMENT_ZONE_RATIO
    _normalized_offset = abs(_signed_offset) / desired_zone

    
    if speed < 20:
        return step_reward
    
    
    
    _alignment_reward = np.clip(1.0 - _normalized_offset, -1.0, 1.0)
    

    speed = np.linalg.norm(env.car.hull.linearVelocity)
    
    if speed < 20:
        return step_reward

    step_reward += _alignment_reward * L2D_CENTER_REWARD_WEIGHT    
        
    return step_reward

def _reward_function_1(env, action) -> float:
    if action is None:
        return 0.0

    step_reward = 0.0

    # 1. Time penalty
    env.reward -= L2D_TIME_PENALTY
    step_reward = env.reward - env.prev_reward
    env.prev_reward = env.reward

    # 2. Track alignment offset (from _reward_align_speed)
    car_pos = np.array(env.car.hull.position)
    car_heading = env.car.hull.GetWorldVector((0, 1))
    car_right = np.array([-car_heading[1], car_heading[0]])

    min_distance = float("inf")
    signed_offset = 0.0

    idx0 = env.l2d_last_segment_idx
    window = 4

    min_i = max(1, idx0 - window)
    max_i = min(len(env.track), idx0 + window)

    for i in range(min_i, max_i):
        x1, y1 = env.track[i - 1][2:4]
        x2, y2 = env.track[i][2:4]
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        ab = b - a
        ap = car_pos - a

        ab_len_squared = np.dot(ab, ab)
        if ab_len_squared == 0:
            continue

        t = np.clip(np.dot(ap, ab) / ab_len_squared, 0.0, 1.0)
        closest_point = a + t * ab
        diff = car_pos - closest_point
        dist = np.linalg.norm(diff)

        if dist < min_distance:
            min_distance = dist
            signed_offset = np.dot(diff, car_right)
            env.l2d_last_segment_idx = i

    desired_zone = 40 / 6.0  * L2D_ALIGNMENT_ZONE_RATIO
    
    normalized_offset = abs(signed_offset) / desired_zone
    
    #print(f"normalized_offset: {normalized_offset:.5f}")
    _alignment_reward = np.clip(1.0 - normalized_offset, -1.0, 1.0)
    
    

    # 3. Apply penalty if outside center zone
    WALL_CRITICAL_OFFSET = 3.0  # normalized offset threshold
    MAX_PENALTY = 5.0
    
    step_reward = 0.0
        
    target_speed = 25.0
    tolerance = 5.0  # 20–30 range
    scale = 1.0      # peak reward

    speed = np.linalg.norm(env.car.hull.linearVelocity)
    speed_diff = (speed - target_speed) / tolerance
    speed_reward = np.exp(-speed_diff**2) * scale
    step_reward += speed_reward
    
    print(f"speed_reward: {speed_reward:.5f}")
        
    return step_reward

def _reward_function_2(env, action) -> float:
    if action is None:
        return 0.0

    step_reward = 0.0

    # 1. Apply time penalty
    env.reward -= L2D_TIME_PENALTY

    # 2. Reward for discovering new tiles
    step_reward = env.reward - env.prev_reward
    env.prev_reward = env.reward

    #3. Non-linear ramped wall penalty
    left = env.car.l2d_rays.get("left_90", 0.0)
    right = env.car.l2d_rays.get("right_90", 0.0)

    WALL_CRITICAL_DISTANCE = 4.0
    MAX_PENALTY = 10.0

    wall_penalty = 0.0
    
    if left < WALL_CRITICAL_DISTANCE:
        closeness = 1.0 - (left / WALL_CRITICAL_DISTANCE)
        wall_penalty = MAX_PENALTY * closeness**2

    elif right < WALL_CRITICAL_DISTANCE:
        closeness = 1.0 - (right / WALL_CRITICAL_DISTANCE)
        wall_penalty = MAX_PENALTY * closeness**2
        
    #if wall_penalty > 0:
        #step_reward -= wall_penalty

        
    car_pos = np.array(env.car.hull.position)
    car_heading = env.car.hull.GetWorldVector((0, 1))  # car's forward direction
    car_right = np.array([-car_heading[1], car_heading[0]])  # 90 degrees to the right

    min_distance = float("inf")
    signed_offset = 0.0
    
    idx0 = env.l2d_last_segment_idx
    window = 5  # try segments 4 from last closest
    
    min_i = max(1, idx0 - window)
    max_i = min(len(env.track), idx0 + window)

    for i in range(min_i, max_i):
        x1, y1 = env.track[i - 1][2:4]
        x2, y2 = env.track[i][2:4]
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        ab = b - a
        ap = car_pos - a

        ab_len_squared = np.dot(ab, ab)
        if ab_len_squared == 0:
            continue

        t = np.clip(np.dot(ap, ab) / ab_len_squared, 0.0, 1.0)
        closest_point = a + t * ab
        diff = car_pos - closest_point
        dist = np.linalg.norm(diff)

        if dist < min_distance:
            min_distance = dist
            signed_offset = np.dot(diff, car_right)  # positive if to the right 
            env.l2d_last_segment_idx = i  # update the cache
            
    normalized_offset = abs(signed_offset) / (40 / 6.0)
    
    
    #print(f"penalty: {normalized_offset:.5f}", f"wall_penalty: {left:.5f}")
    
    MAX_PENALTY = 10.0
    CENTER_CRITICAL_OFFSET = 0.4  # no penalty within this zone
    MAX_OFFSET = 1.0  # expected max normalized offset before full penalty

    center_penalty = 0.0

    if normalized_offset > CENTER_CRITICAL_OFFSET:
        overshoot = normalized_offset - CENTER_CRITICAL_OFFSET
        scaled = overshoot / (MAX_OFFSET - CENTER_CRITICAL_OFFSET)
        scaled = np.clip(scaled, 0.0, 1.0)
        center_penalty = MAX_PENALTY * scaled**2
        
    print(f"penalty: {center_penalty:.5f}", f"wall_penalty: {wall_penalty:.5f}")
        
    step_reward -= center_penalty  
               
    return step_reward

def _reward_function_3(env, action) -> float:
    if action is None:
        return 0.0

    step_reward = 0.0
    # Time penalty    
    step_reward -= L2D_TIME_PENALTY
    
    # 2. Reward for discovering new tiles
    tile_reward = env.reward - env.prev_reward
    env.prev_reward = env.reward
    
    car_pos = np.array(env.car.hull.position)
    car_heading = env.car.hull.GetWorldVector((0, 1))  # car's forward direction
    car_right = np.array([-car_heading[1], car_heading[0]])  # 90 degrees to the right

    min_distance = float("inf")
    signed_offset = 0.0
    
    idx0 = env.l2d_last_segment_idx
    window = 5  # try segments 4 from last closest
    
    min_i = max(1, idx0 - window)
    max_i = min(len(env.track), idx0 + window)

    for i in range(min_i, max_i):
        x1, y1 = env.track[i - 1][2:4]
        x2, y2 = env.track[i][2:4]
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        ab = b - a
        ap = car_pos - a

        ab_len_squared = np.dot(ab, ab)
        if ab_len_squared == 0:
            continue

        t = np.clip(np.dot(ap, ab) / ab_len_squared, 0.0, 1.0)
        closest_point = a + t * ab
        diff = car_pos - closest_point
        dist = np.linalg.norm(diff)

        if dist < min_distance:
            min_distance = dist
            signed_offset = np.dot(diff, car_right)  # positive if to the right 
            env.l2d_last_segment_idx = i  # update the cache
            
    max_offset = 40 / 6.0  # full width in meters (same as TRACK_WIDTH)

    desired_zone = max_offset * L2D_ALIGNMENT_ZONE_RATIO
    normalized_offset = abs(signed_offset) / desired_zone
    alignment_reward = np.clip(1.0 - normalized_offset, -1.0, 1.0) * L2D_CENTER_REWARD_WEIGHT
    
    #step_reward += alignment_reward * tile_reward
    left = env.car.l2d_rays.get("left_90", 0.0)
    right = env.car.l2d_rays.get("right_90", 0.0)
    
    if left < 2.0 or right < 2.0:
        step_reward -= 1.0

    else :
        step_reward += tile_reward

    
    if tile_reward > 0:
        #print(f"rew: {alignment_reward * tile_reward:.5f}", f"tile: { tile_reward:.5f}")
        #print(f"step_reward: {step_reward:.5f}")
        pass
    
    return step_reward

def _reward_function_4(env, action) -> float:
    step_reward = 0.0
    
    car_pos = np.array(env.car.hull.position)
    car_heading = env.car.hull.GetWorldVector((0, 1))  # car's forward direction
    car_right = np.array([-car_heading[1], car_heading[0]])  # 90 degrees to the right

    min_distance = float("inf")
    signed_offset = 0.0
    
    idx0 = env.l2d_last_segment_idx
    window = 5  # try segments 4 from last closest
    
    min_i = max(1, idx0 - window)
    max_i = min(len(env.track), idx0 + window)

    for i in range(min_i, max_i):
        x1, y1 = env.track[i - 1][2:4]
        x2, y2 = env.track[i][2:4]
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        ab = b - a
        ap = car_pos - a

        ab_len_squared = np.dot(ab, ab)
        if ab_len_squared == 0:
            continue

        t = np.clip(np.dot(ap, ab) / ab_len_squared, 0.0, 1.0)
        closest_point = a + t * ab
        diff = car_pos - closest_point
        dist = np.linalg.norm(diff)

        if dist < min_distance:
            min_distance = dist
            signed_offset = np.dot(diff, car_right)  # positive if to the right 
            env.l2d_last_segment_idx = i  # update the cache
    
    normalized_offset = abs(signed_offset) / (40 / 6.0 * L2D_ALIGNMENT_ZONE_RATIO)
    alignment_reward = 1.0 - normalized_offset
    
    speed = np.linalg.norm(env.car.hull.linearVelocity)
    target = 25.0
    tolerance = 5.0
    speed_reward = np.exp(-((speed - target) / tolerance) ** 2)
    
    step_reward = (alignment_reward * speed_reward)
    # Du kan ikke API haier
    return step_reward

def _reward_function_5(env, action) -> float:
    step_reward = 0.0
    
    car_pos = np.array(env.car.hull.position)
    car_heading = env.car.hull.GetWorldVector((0, 1))  # car's forward direction
    car_right = np.array([-car_heading[1], car_heading[0]])  # 90 degrees to the right

    min_distance = float("inf")
    signed_offset = 0.0
    
    idx0 = env.l2d_last_segment_idx
    window = 5  # try segments 4 from last closest
    
    min_i = max(1, idx0 - window)
    max_i = min(len(env.track), idx0 + window)

    for i in range(min_i, max_i):
        x1, y1 = env.track[i - 1][2:4]
        x2, y2 = env.track[i][2:4]
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        ab = b - a
        ap = car_pos - a

        ab_len_squared = np.dot(ab, ab)
        if ab_len_squared == 0:
            continue

        t = np.clip(np.dot(ap, ab) / ab_len_squared, 0.0, 1.0)
        closest_point = a + t * ab
        diff = car_pos - closest_point
        dist = np.linalg.norm(diff)

        if dist < min_distance:
            min_distance = dist
            signed_offset = np.dot(diff, car_right)  # positive if to the right 
            env.l2d_last_segment_idx = i  # update the cache
            
            
    TRACK_HALF_WIDTH = 40 / 6.0
    alignment_zone = TRACK_HALF_WIDTH * L2D_ALIGNMENT_ZONE_RATIO
    normalized_offset = abs(signed_offset) / alignment_zone
    alignment_reward = np.exp(-normalized_offset**2)
        
    speed = np.linalg.norm(env.car.hull.linearVelocity)
    target_speed = 25.0
    tolerance = 5.0
    speed_reward = np.exp(-((speed - target_speed) / tolerance) ** 2)
    
    
    step_reward = 0.5 * alignment_reward + 0.5 * speed_reward
    return step_reward
        
def _reward_function_6(env, action) -> float:
    step_reward = 0.0
    
    # Time penalty    
    step_reward -= L2D_TIME_PENALTY
    
    car_pos = np.array(env.car.hull.position)
    car_heading = env.car.hull.GetWorldVector((0, 1))  # car's forward direction
    car_right = np.array([-car_heading[1], car_heading[0]])  # 90 degrees to the right

    min_distance = float("inf")
    signed_offset = 0.0
    
    idx0 = env.l2d_last_segment_idx
    window = 5  # try segments 4 from last closest
    
    min_i = max(1, idx0 - window)
    max_i = min(len(env.track), idx0 + window)

    for i in range(min_i, max_i):
        x1, y1 = env.track[i - 1][2:4]
        x2, y2 = env.track[i][2:4]
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        ab = b - a
        ap = car_pos - a

        ab_len_squared = np.dot(ab, ab)
        if ab_len_squared == 0:
            continue

        t = np.clip(np.dot(ap, ab) / ab_len_squared, 0.0, 1.0)
        closest_point = a + t * ab
        diff = car_pos - closest_point
        dist = np.linalg.norm(diff)

        if dist < min_distance:
            min_distance = dist
            signed_offset = np.dot(diff, car_right)  # positive if to the right 
            env.l2d_last_segment_idx = i  # update the cache
    
    TRACK_EDGE_OFFSET = 5.0  # max distance from center
    scaled_center_value = 1.0 - abs(signed_offset) / TRACK_EDGE_OFFSET
    scaled_center_value = np.clip(scaled_center_value, 0.0, 1.0)
    
    step_reward += scaled_center_value * env.l2d_tile_reward
    env.l2d_tile_reward = 0.0
    
    #update global reward
    env.reward += step_reward

    return step_reward
    
def _reward_function_8(env, action) -> float:
    step_reward = 0.0
    
    # Time penalty    
    step_reward -= L2D_TIME_PENALTY
    
    car_pos = np.array(env.car.hull.position)
    car_heading = env.car.hull.GetWorldVector((0, 1))  # car's forward direction
    car_right = np.array([-car_heading[1], car_heading[0]])  # 90 degrees to the right

    min_distance = float("inf")
    signed_offset = 0.0
    
    idx0 = env.l2d_last_segment_idx
    window = 5  # try segments 4 from last closest
    
    min_i = max(1, idx0 - window)
    max_i = min(len(env.track), idx0 + window)

    for i in range(min_i, max_i):
        x1, y1 = env.track[i - 1][2:4]
        x2, y2 = env.track[i][2:4]
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        ab = b - a
        ap = car_pos - a

        ab_len_squared = np.dot(ab, ab)
        if ab_len_squared == 0:
            continue

        t = np.clip(np.dot(ap, ab) / ab_len_squared, 0.0, 1.0)
        closest_point = a + t * ab
        diff = car_pos - closest_point
        dist = np.linalg.norm(diff)

        if dist < min_distance:
            min_distance = dist
            signed_offset = np.dot(diff, car_right)  # positive if to the right 
            env.l2d_last_segment_idx = i  # update the cache
    
    # signed_offset 5,  0,  -5
    TRACK_EDGE_OFFSET = 5.0  # max distance from center
    
    normalized_offset = abs(signed_offset) / TRACK_EDGE_OFFSET
    scaled_center_value =  1.0 - normalized_offset
    scaled_center_value = np.clip(scaled_center_value, 0.0, 1.0)

    if env.car.outside_track:
        step_reward -= L2D_OFF_TRACK_PENALTY
    
    step_reward += scaled_center_value * env.l2d_tile_reward

        
    env.l2d_tile_reward = 0.0
    
    #update global reward
    env.reward += step_reward

    return step_reward

# Quadratic
def _reward_function_9(env, action) -> float:
    step_reward = 0.0
    
    # Time penalty    
    step_reward -= L2D_TIME_PENALTY
    
    signed_offset = calculate_signed_offset(env)
    
    # signed_offset 5,  0,  -5
    TRACK_EDGE_OFFSET = 5.0  # max distance from center
    
    normalized_offset = abs(signed_offset) / TRACK_EDGE_OFFSET
    scaled_center_value = 1.0 - normalized_offset**2
    scaled_center_value = np.clip(scaled_center_value, 0.0, 1.0)

    if env.car.outside_track:
        step_reward -= L2D_OFF_TRACK_PENALTY
    
    step_reward += scaled_center_value * env.l2d_tile_reward

        
    env.l2d_tile_reward = 0.0
    env.reward += step_reward

    return step_reward

# Exponential decay
def _reward_function_10(env, action) -> float:
    step_reward = 0.0
    
    # Time penalty    
    step_reward -= L2D_TIME_PENALTY
    
    signed_offset = calculate_signed_offset(env)
    
    # signed_offset 5,  0,  -5
    TRACK_EDGE_OFFSET = 5.0  # max distance from center
    
    normalized_offset = abs(signed_offset) / TRACK_EDGE_OFFSET
    scaled_center_value = np.exp(-3 * normalized_offset)
    scaled_center_value = np.clip(scaled_center_value, 0.0, 1.0)

    if env.car.outside_track:
        step_reward -= L2D_OFF_TRACK_PENALTY
    
    step_reward += scaled_center_value * env.l2d_tile_reward

        
    env.l2d_tile_reward = 0.0
    env.reward += step_reward

    return step_reward

# Cosine-shaped curve
def _reward_function_11(env, action) -> float:
    step_reward = 0.0
    
    # Time penalty    
    step_reward -= L2D_TIME_PENALTY
    
    signed_offset = calculate_signed_offset(env)
    
    # signed_offset 5,  0,  -5
    TRACK_EDGE_OFFSET = 5.0  # max distance from center
    
    normalized_offset = abs(signed_offset) / TRACK_EDGE_OFFSET
    scaled_center_value = 0.5 * (1 + np.cos(np.pi * normalized_offset))
    scaled_center_value = np.clip(scaled_center_value, 0.0, 1.0)

    if env.car.outside_track:
        step_reward -= L2D_OFF_TRACK_PENALTY
    
    step_reward += scaled_center_value * env.l2d_tile_reward

        
    env.l2d_tile_reward = 0.0
    env.reward += step_reward

    return step_reward

# Inverse-square
def _reward_function_12(env, action) -> float:
    step_reward = 0.0
    
    # Time penalty    
    step_reward -= L2D_TIME_PENALTY
    
    signed_offset = calculate_signed_offset(env)
    
    # signed_offset 5,  0,  -5
    TRACK_EDGE_OFFSET = 5.0  # max distance from center
    
    normalized_offset = abs(signed_offset) / TRACK_EDGE_OFFSET
    scaled_center_value = 1 / (1 + 5 * normalized_offset**2)
    scaled_center_value = np.clip(scaled_center_value, 0.0, 1.0)

    if env.car.outside_track:
        step_reward -= L2D_OFF_TRACK_PENALTY
    
    step_reward += scaled_center_value * env.l2d_tile_reward

        
    env.l2d_tile_reward = 0.0
    env.reward += step_reward

    return step_reward

# Qustom
def _reward_function_13(env, action) -> float:
    step_reward = 0.0
    
    # Time penalty    
    step_reward -= L2D_TIME_PENALTY
    
    signed_offset = calculate_signed_offset(env)
    
    # signed_offset 5,  0,  -5
    TRACK_EDGE_OFFSET = 5.0  # max distance from center
    
    normalized_offset = abs(signed_offset) / TRACK_EDGE_OFFSET
    scaled_center_value = (1.0 - normalized_offset) ** 3
    scaled_center_value = np.clip(scaled_center_value, 0.0, 1.0)

    if env.car.outside_track:
        step_reward -= L2D_OFF_TRACK_PENALTY
    
    step_reward += scaled_center_value * env.l2d_tile_reward

        
    env.l2d_tile_reward = 0.0
    env.reward += step_reward

    return step_reward