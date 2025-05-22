import numpy as np
import torch


class MPCController:
    def __init__(self, model, plan_horizon=5, num_candidates=100):
        self.model = model  # the trained dynamics model (f_θ)
        self.plan_horizon = plan_horizon  # how many steps to plan into the future
        self.num_candidates = num_candidates  # how many action sequences to sample
        
        self.device = torch.device("cpu")  # later you can switch to GPU if needed
        self.action_low = torch.tensor([-1.0, 0.0, 0.0])
        self.action_high = torch.tensor([1.0, 1.0, 1.0])

    def act(self, obs: np.ndarray, reward_fn):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, obs_dim]

        best_return = -np.inf
        best_action = None

        for _ in range(self.num_candidates):
            # Sample a random action sequence
            action_seq = torch.rand(self.plan_horizon, len(self.action_low)).to(self.device)
            action_seq = self.action_low + (self.action_high - self.action_low) * action_seq

            total_reward = 0.0
            sim_obs = obs.clone()

            for t in range(self.plan_horizon):
                act = action_seq[t].unsqueeze(0)  # [1, act_dim]
                delta = self.model(sim_obs, act)  # predict Δs
                sim_obs = sim_obs + delta

                # Evaluate predicted reward
                reward = reward_fn(sim_obs.squeeze(0).detach().cpu().numpy(),act.squeeze(0).detach().cpu().numpy())
                total_reward += reward

            # Only update best after full sequence
            if total_reward > best_return:
                best_return = total_reward
                best_action = action_seq[0]

        return best_action.cpu().numpy()


def reward_fnOld(obs: np.ndarray, act: np.ndarray) -> float:
    time_penalty = -0.1  # constant penalty each step (same as PPO)

    # Centering approximation: if left ray is much shorter, you're too far right (and vice versa)
    left = obs[1]
    right = obs[2]
    center_offset = np.abs(left - right)
    center_penalty = center_offset / 20.0  # normalize

    # Optional: forward speed bonus
    speed_bonus = obs[3]

    # Penalty for steering (encourages stability)
    steer_penalty = np.abs(obs[5])  # or act[0] if preferred

    return speed_bonus - 0.5 * center_penalty - 0.1 * steer_penalty + time_penalty

def reward_fn(obs: np.ndarray, act: np.ndarray) -> float:
    time_penalty = -0.1

    left = obs[1]
    right = obs[2]

    # Approximate centering
    center_offset = np.abs(left - right)
    center_penalty = center_offset / 10.0  # tuneable

    # Use speed as a proxy for tile progression
    forward_speed = obs[3]  # higher speed → likely to move forward

    # Scale "reward" by how centered the agent is
    centered_speed_bonus = forward_speed * (1.0 - center_penalty)

    steer_penalty = np.abs(act[0])

    return centered_speed_bonus - 0.1 * steer_penalty + time_penalty

    
        

