"""
Dataset and model definitions for the 2D particle-in-potential task.

Typical usage:

    from models import (
        ParticleTrajectoryDataset, TrajectoryPredictor, masked_mse_loss,
    )

    dataset = ParticleTrajectoryDataset("particle_2d_dataset.npz")
    model = TrajectoryPredictor(
        traj_len=2001, state_dim=4, pot_latent_dim=256, hidden_dim=512,
    )
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ParticleTrajectoryDataset(Dataset):
    """Wraps an .npz file produced by `data_generation.generate_dataset`.

    Computes normalization statistics over the full dataset at init time.
    The statistics are exposed as attributes (`pot_mean`, `pot_std`,
    `init_mean`, `init_std`, `traj_mean`, `traj_std`) so downstream code
    can un-normalize predictions.
    """

    def __init__(
        self,
        npz_path,
        normalize_potential=True,
        normalize_init=True,
        normalize_target=True,
    ):
        data = np.load(npz_path)

        self.potentials = data["potentials"].astype(np.float32)
        self.init_states = data["init_states"].astype(np.float32)
        self.trajectories = data["trajectories"].astype(np.float32)
        self.traj_mask = data["traj_mask"].astype(np.float32)

        self.normalize_potential = normalize_potential
        self.normalize_init = normalize_init
        self.normalize_target = normalize_target

        self.pot_mean = self.potentials.mean()
        self.pot_std = self.potentials.std() + 1e-8

        self.init_mean = self.init_states.mean(axis=0, keepdims=True)
        self.init_std = self.init_states.std(axis=0, keepdims=True) + 1e-8

        self.traj_mean = self.trajectories.mean(axis=(0, 1), keepdims=True)
        self.traj_std = self.trajectories.std(axis=(0, 1), keepdims=True) + 1e-8

    def __len__(self):
        return len(self.potentials)

    def __getitem__(self, idx):
        pot = self.potentials[idx]
        init_state = self.init_states[idx]
        traj = self.trajectories[idx]
        mask = self.traj_mask[idx]

        if self.normalize_potential:
            pot = (pot - self.pot_mean) / self.pot_std

        if self.normalize_init:
            init_state = (init_state - self.init_mean[0]) / self.init_std[0]

        if self.normalize_target:
            traj = (traj - self.traj_mean[0]) / self.traj_std[0]

        return {
            "potential": torch.from_numpy(pot).unsqueeze(0),
            "init_state": torch.from_numpy(init_state),
            "trajectory": torch.from_numpy(traj),
            "traj_mask": torch.from_numpy(mask),
        }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PotentialEncoder(nn.Module):
    """CNN encoder: (B, 1, 64, 64) -> (B, out_dim)."""

    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class TrajectoryPredictor(nn.Module):
    """Predicts a full trajectory from a potential grid + initial state.

    Input:
        potential:  (B, 1, nx, ny)
        init_state: (B, 4)       -- (x, y, px, py)
    Output:
        (B, traj_len, state_dim)
    """

    def __init__(self, traj_len=301, state_dim=4, pot_latent_dim=256, hidden_dim=512):
        super().__init__()
        self.traj_len = traj_len
        self.state_dim = state_dim

        self.potential_encoder = PotentialEncoder(out_dim=pot_latent_dim)

        self.head = nn.Sequential(
            nn.Linear(pot_latent_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, traj_len * state_dim),
        )

    def forward(self, potential, init_state):
        z_pot = self.potential_encoder(potential)
        z = torch.cat([z_pot, init_state], dim=-1)
        out = self.head(z)
        return out.view(-1, self.traj_len, self.state_dim)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def masked_mse_loss(pred, target, mask):
    """MSE averaged only over valid timesteps (mask == 1)."""
    mask = mask.unsqueeze(-1)
    sq = (pred - target) ** 2
    sq = sq * mask
    denom = mask.sum() * pred.shape[-1] + 1e-8
    return sq.sum() / denom
