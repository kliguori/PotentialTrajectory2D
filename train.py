"""
Training utilities for the 2D particle-in-potential task.

Typical usage from a notebook:

    from train import TrainConfig, run_training

    model, history = run_training(
        npz_path="particle_2d_dataset.npz",
        checkpoint_path="best_model.pt",
        traj_len=2001,
        train_cfg=TrainConfig(num_epochs=30),
    )
"""

from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from models import (
    ParticleTrajectoryDataset,
    TrajectoryPredictor,
    masked_mse_loss,
)


@dataclass
class TrainConfig:
    batch_size: int = 32
    num_epochs: int = 30
    lr: float = 1e-3
    seed: int = 42
    train_frac: float = 0.8
    val_frac: float = 0.1
    num_workers: int = 0


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        potential = batch["potential"].to(device)
        init_state = batch["init_state"].to(device)
        target = batch["trajectory"].to(device)
        mask = batch["traj_mask"].to(device)

        optimizer.zero_grad()
        pred = model(potential, init_state)
        loss = masked_mse_loss(pred, target, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * potential.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        potential = batch["potential"].to(device)
        init_state = batch["init_state"].to(device)
        target = batch["trajectory"].to(device)
        mask = batch["traj_mask"].to(device)

        pred = model(potential, init_state)
        loss = masked_mse_loss(pred, target, mask)

        total_loss += loss.item() * potential.size(0)

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Split helper -- reused from the investigation notebook so that the test
# set is reconstructed identically.
# ---------------------------------------------------------------------------

def build_splits(dataset, train_frac=0.8, val_frac=0.1, seed=42):
    n_total = len(dataset)
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )
    return train_set, val_set, test_set


# ---------------------------------------------------------------------------
# Top-level training entry point
# ---------------------------------------------------------------------------

def run_training(
    npz_path,
    checkpoint_path="best_model.pt",
    traj_len=2001,
    state_dim=4,
    pot_latent_dim=256,
    hidden_dim=512,
    train_cfg=None,
    device=None,
    verbose=True,
):
    """Train a TrajectoryPredictor and save the best checkpoint.

    Returns
    -------
    model : TrajectoryPredictor
        The in-memory model with its best-val weights loaded.
    history : dict
        Dict with "train_loss", "val_loss" lists and "best_val_loss" scalar.
    """
    if train_cfg is None:
        train_cfg = TrainConfig()

    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    dataset = ParticleTrajectoryDataset(npz_path)
    train_set, val_set, _test_set = build_splits(
        dataset,
        train_frac=train_cfg.train_frac,
        val_frac=train_cfg.val_frac,
        seed=train_cfg.seed,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("Using device:", device)

    model = TrajectoryPredictor(
        traj_len=traj_len,
        state_dim=state_dim,
        pot_latent_dim=pot_latent_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    history = {"train_loss": [], "val_loss": [], "best_val_loss": float("inf")}

    for epoch in range(1, train_cfg.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = eval_one_epoch(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
            )

        if val_loss < history["best_val_loss"]:
            history["best_val_loss"] = val_loss
            torch.save(model.state_dict(), checkpoint_path)

    # Reload best weights before returning.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    if verbose:
        print("Best val loss:", history["best_val_loss"])
        print("Training config:", asdict(train_cfg))

    return model, history
