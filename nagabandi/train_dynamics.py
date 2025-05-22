import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from nagabandi.dynamics_model import DynamicsModel  # your learned model

def load_dataset(npz_path):
    data = np.load(npz_path)
    obs = data["obs"]  # shape: [N, obs_dim]
    actions = data["actions"]  # shape: [N, action_dim]
    next_obs = data["next_obs"]  # shape: [N, obs_dim]
    delta = next_obs - obs  # Δs = s′ - s

    return obs, actions, delta


def prepare_tensors(obs, actions, delta):
    obs_tensor = torch.tensor(obs, dtype=torch.float32)  # [N, obs_dim]
    actions_tensor = torch.tensor(actions, dtype=torch.float32)  # [N, act_dim]
    delta_tensor = torch.tensor(delta, dtype=torch.float32)  # [N, obs_dim]

    dataset = torch.utils.data.TensorDataset(obs_tensor, actions_tensor, delta_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    return dataloader


def train_model(model, dataloader, num_epochs=20, lr=1e-3):
    model.train()  # set to training mode
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam optimizer
    loss_fn = torch.nn.MSELoss()  # mean squared error loss

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for obs, act, delta in dataloader:
            pred = model(obs, act)  # predict Δs
            loss = loss_fn(pred, delta)  # compare to true Δs

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")


def train_dynamics_model(directory: str, num_epochs: int = 20):
    npz_path = os.path.join(directory, "transitions.npz")
    obs, actions, delta = load_dataset(npz_path)
    dataloader = prepare_tensors(obs, actions, delta)

    obs_dim = obs.shape[1]
    act_dim = actions.shape[1]
    model = DynamicsModel(obs_dim, act_dim)

    train_model(model, dataloader, num_epochs=num_epochs)

    model_path = os.path.join(directory, "dynamics_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Saved trained model to {model_path}")


