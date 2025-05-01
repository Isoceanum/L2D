
import numpy as np

from constants import * 

def l2d_calculate_step_reward (mode, env, action) -> float:
    match mode:
        case "baseline":
            return _reward_baseline(env, action)
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
        case _:
            raise ValueError(f"Unknown reward function named: {mode}")

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
        
    print("alignment_score: ", L2D_CENTER_REWARD_WEIGHT * alignment_score)
    return step_reward

def _reward_align_speed(env, action) -> float:
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
    alignment_score = np.clip(1.0 - normalized_offset, -1.0, 1.0) * L2D_CENTER_REWARD_WEIGHT

    # Speed reward (scaled down)
    speed = np.linalg.norm(env.car.hull.linearVelocity)
    speed_reward = L2D_SPEED_REWARD_WEIGHT * speed
    
    if alignment_score > 0:
        step_reward += speed_reward * alignment_score 
    else:
        step_reward += alignment_score 
    
    #env.reward += step_reward
    
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
    