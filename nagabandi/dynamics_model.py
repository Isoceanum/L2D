import torch
import torch.nn as nn

class DynamicsModel(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=[512, 512]):
        super().__init__()

        input_dim = obs_dim + action_dim
        output_dim = obs_dim
        
        layers = [] # list to store layers in order
        last_dim = input_dim # start with input size (obs + act)
        for h in hidden_sizes:  # for each hidden layer size
            layers.append(nn.Linear(last_dim, h))  # fully connected layer
            layers.append(nn.ReLU()) # non-linearity
            last_dim = h # update for next layer's input size
            
        layers.append(nn.Linear(last_dim, output_dim))  # final layer: outputs Δs
        self.net = nn.Sequential(*layers)  # combine all layers into a single module


    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)  # combine obs and action into one tensor
        return self.net(x)  # run through MLP to predict Δs