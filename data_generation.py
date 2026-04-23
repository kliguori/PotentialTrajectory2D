"""
Data generation for 2D particle-in-potential trajectories.

Typical usage:

    from data_generation import (
        GridConfig, TrajConfig, DataConfig, generate_dataset,
    )

    generate_dataset(
        path="particle_2d_dataset.npz",
        n_examples=2000,
        seed=42,
        grid_cfg=GridConfig(xlim=4.0, ylim=4.0, nx=64, ny=64),
        traj_cfg=TrajConfig(dt=0.01, steps=2000, mass=0.2),
        data_cfg=DataConfig(),
    )
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class GridConfig:
    xlim: float = 4.0
    ylim: float = 4.0
    nx: int = 64
    ny: int = 64


@dataclass
class TrajConfig:
    dt: float = 0.01
    steps: int = 300
    mass: float = 1.0


@dataclass
class DataConfig:
    offset_range: Tuple[float, float] = (-5.0, 5.0)

    central_prob: float = 0.25
    arbitrary_prob: float = 0.75

    fourier_prob: float = 0.5
    gaussian_bump_prob: float = 0.5

    max_fourier_mode: int = 4
    fourier_alpha: float = 3.0

    min_bumps: int = 4
    max_bumps: int = 9

    max_init_speed: float = 2.5


# ---------------------------------------------------------------------------
# Grid and interpolation utilities
# ---------------------------------------------------------------------------

def make_grid(cfg: GridConfig):
    xs = np.linspace(-cfg.xlim, cfg.xlim, cfg.nx)
    ys = np.linspace(-cfg.ylim, cfg.ylim, cfg.ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return xs, ys, X, Y


def bilinear_interp(field, xs, ys, x, y):
    if x < xs[0] or x > xs[-1] or y < ys[0] or y > ys[-1]:
        return None

    i = np.searchsorted(xs, x) - 1
    j = np.searchsorted(ys, y) - 1
    i = np.clip(i, 0, len(xs) - 2)
    j = np.clip(j, 0, len(ys) - 2)

    x1, x2 = xs[i], xs[i + 1]
    y1, y2 = ys[j], ys[j + 1]

    tx = 0.0 if x2 == x1 else (x - x1) / (x2 - x1)
    ty = 0.0 if y2 == y1 else (y - y1) / (y2 - y1)

    f11 = field[i, j]
    f12 = field[i, j + 1]
    f21 = field[i + 1, j]
    f22 = field[i + 1, j + 1]

    return (
        (1 - tx) * (1 - ty) * f11
        + (1 - tx) * ty * f12
        + tx * (1 - ty) * f21
        + tx * ty * f22
    )


# ---------------------------------------------------------------------------
# Potential samplers
# ---------------------------------------------------------------------------

def central_potential(X, Y, rng):
    r2 = X**2 + Y**2
    choice = rng.integers(3)

    if choice == 0:
        k = rng.uniform(0.3, 2.0)
        V = 0.5 * k * r2
        meta = {"type": "central_harmonic", "k": float(k)}

    elif choice == 1:
        a = rng.uniform(0.02, 0.15)
        b = rng.uniform(0.0, 0.8)
        V = b * r2 + a * r2**2
        meta = {"type": "central_quartic", "a": float(a), "b": float(b)}

    else:
        A = rng.uniform(1.0, 4.0)
        sigma = rng.uniform(0.6, 1.8)
        V = -A * np.exp(-r2 / (2 * sigma**2))
        meta = {"type": "central_gaussian", "A": float(A), "sigma": float(sigma)}

    return V, meta


def arbitrary_smooth_fourier_potential(X, Y, rng, max_mode=4, alpha=3.0):
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()

    Xs = 2 * np.pi * (X - x_min) / (x_max - x_min)
    Ys = 2 * np.pi * (Y - y_min) / (y_max - y_min)

    V = np.zeros_like(X)

    for mx in range(max_mode + 1):
        for my in range(max_mode + 1):
            if mx == 0 and my == 0:
                continue

            scale = 1.0 / (1.0 + mx**2 + my**2) ** (alpha / 2)

            acc = rng.normal(scale=scale)
            acs = rng.normal(scale=scale)
            asc = rng.normal(scale=scale)
            ass = rng.normal(scale=scale)

            V += (
                acc * np.cos(mx * Xs) * np.cos(my * Ys)
                + acs * np.cos(mx * Xs) * np.sin(my * Ys)
                + asc * np.sin(mx * Xs) * np.cos(my * Ys)
                + ass * np.sin(mx * Xs) * np.sin(my * Ys)
            )

    conf = rng.uniform(0.0, 0.15)
    V += conf * (X**2 + Y**2)

    V = V / (np.std(V) + 1e-8)

    meta = {
        "type": "arbitrary_smooth",
        "generator": "fourier",
        "max_mode": int(max_mode),
        "alpha": float(alpha),
        "conf": float(conf),
    }
    return V, meta


def arbitrary_smooth_gaussian_potential(X, Y, rng, nbumps=6):
    V = np.zeros_like(X)

    for _ in range(nbumps):
        A = rng.uniform(-2.5, 2.5)
        x0 = rng.uniform(X.min() * 0.8, X.max() * 0.8)
        y0 = rng.uniform(Y.min() * 0.8, Y.max() * 0.8)
        sx = rng.uniform(0.4, 1.8)
        sy = rng.uniform(0.4, 1.8)
        theta = rng.uniform(0, 2 * np.pi)

        Xc = X - x0
        Yc = Y - y0

        Xr = np.cos(theta) * Xc + np.sin(theta) * Yc
        Yr = -np.sin(theta) * Xc + np.cos(theta) * Yc

        V += A * np.exp(-(Xr**2 / (2 * sx**2) + Yr**2 / (2 * sy**2)))

    conf = rng.uniform(0.0, 0.15)
    V += conf * (X**2 + Y**2)

    V = V / (np.std(V) + 1e-8)

    meta = {
        "type": "arbitrary_smooth",
        "generator": "gaussian_bumps",
        "nbumps": int(nbumps),
        "conf": float(conf),
    }
    return V, meta


def sample_potential(X, Y, data_cfg: DataConfig, rng):
    family = rng.choice(
        ["central", "arbitrary_smooth"],
        p=[data_cfg.central_prob, data_cfg.arbitrary_prob],
    )

    if family == "central":
        V0, meta = central_potential(X, Y, rng)
    else:
        generator = rng.choice(
            ["fourier", "gaussian_bumps"],
            p=[data_cfg.fourier_prob, data_cfg.gaussian_bump_prob],
        )

        if generator == "fourier":
            V0, meta = arbitrary_smooth_fourier_potential(
                X, Y, rng,
                max_mode=data_cfg.max_fourier_mode,
                alpha=data_cfg.fourier_alpha,
            )
        else:
            nbumps = int(rng.integers(data_cfg.min_bumps, data_cfg.max_bumps + 1))
            V0, meta = arbitrary_smooth_gaussian_potential(X, Y, rng, nbumps=nbumps)

    c = rng.uniform(*data_cfg.offset_range)
    V = V0 + c
    meta["offset"] = float(c)

    return V, meta


# ---------------------------------------------------------------------------
# Physics: force, integration, conserved quantities
# ---------------------------------------------------------------------------

def compute_force_grid(V, xs, ys):
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    dVdx = np.gradient(V, dx, axis=0)
    dVdy = np.gradient(V, dy, axis=1)

    Fx = -dVdx
    Fy = -dVdy
    return Fx, Fy


def sample_initial_condition(V, Fx, Fy, xs, ys, traj_cfg: TrajConfig, data_cfg: DataConfig, rng):
    x_margin = 0.7 * (xs[-1] - xs[0]) / 2
    y_margin = 0.7 * (ys[-1] - ys[0]) / 2

    for _ in range(300):
        x0 = rng.uniform(-x_margin, x_margin)
        y0 = rng.uniform(-y_margin, y_margin)

        px0, py0 = rng.uniform(
            -data_cfg.max_init_speed,
            data_cfg.max_init_speed,
            size=2,
        ) * traj_cfg.mass

        V0 = bilinear_interp(V, xs, ys, x0, y0)
        Fx0 = bilinear_interp(Fx, xs, ys, x0, y0)
        Fy0 = bilinear_interp(Fy, xs, ys, x0, y0)

        if V0 is None or Fx0 is None or Fy0 is None:
            continue
        if np.isfinite(V0) and np.isfinite(Fx0) and np.isfinite(Fy0):
            return np.array([x0, y0, px0, py0], dtype=np.float32)

    raise RuntimeError("Failed to sample a valid initial condition.")


def velocity_verlet_rollout(V, Fx, Fy, xs, ys, init_state, traj_cfg: TrajConfig):
    m = traj_cfg.mass
    dt = traj_cfg.dt
    steps = traj_cfg.steps

    traj = np.zeros((steps + 1, 4), dtype=np.float32)
    traj[0] = init_state

    x, y, px, py = map(float, init_state)

    for t in range(steps):
        fx = bilinear_interp(Fx, xs, ys, x, y)
        fy = bilinear_interp(Fy, xs, ys, x, y)

        if fx is None or fy is None:
            return traj[:t + 1]

        px_half = px + 0.5 * dt * fx
        py_half = py + 0.5 * dt * fy

        x_new = x + dt * px_half / m
        y_new = y + dt * py_half / m

        fx_new = bilinear_interp(Fx, xs, ys, x_new, y_new)
        fy_new = bilinear_interp(Fy, xs, ys, x_new, y_new)

        if fx_new is None or fy_new is None:
            return traj[:t + 1]

        px_new = px_half + 0.5 * dt * fx_new
        py_new = py_half + 0.5 * dt * fy_new

        x, y, px, py = x_new, y_new, px_new, py_new
        traj[t + 1] = np.array([x, y, px, py], dtype=np.float32)

    return traj


def compute_energy_along_traj(traj, V, xs, ys, mass=1.0):
    Es = []
    for x, y, px, py in traj:
        v = bilinear_interp(V, xs, ys, float(x), float(y))
        if v is None:
            Es.append(np.nan)
        else:
            Es.append((px**2 + py**2) / (2 * mass) + v)
    return np.array(Es, dtype=np.float32)


def compute_Lz_along_traj(traj):
    x = traj[:, 0]
    y = traj[:, 1]
    px = traj[:, 2]
    py = traj[:, 3]
    return x * py - y * px


# ---------------------------------------------------------------------------
# Single example + full dataset
# ---------------------------------------------------------------------------

def generate_example(grid_cfg: GridConfig, traj_cfg: TrajConfig, data_cfg: DataConfig, rng):
    xs, ys, X, Y = make_grid(grid_cfg)

    V, meta = sample_potential(X, Y, data_cfg, rng)
    Fx, Fy = compute_force_grid(V, xs, ys)

    init_state = sample_initial_condition(V, Fx, Fy, xs, ys, traj_cfg, data_cfg, rng)
    traj = velocity_verlet_rollout(V, Fx, Fy, xs, ys, init_state, traj_cfg)

    energies = compute_energy_along_traj(traj, V, xs, ys, mass=traj_cfg.mass)
    Lz = compute_Lz_along_traj(traj)
    is_central = 1 if meta["type"].startswith("central") else 0

    return {
        "potential": V.astype(np.float32),
        "init_state": init_state.astype(np.float32),
        "trajectory": traj.astype(np.float32),
        "energy": energies.astype(np.float32),
        "angular_momentum": Lz.astype(np.float32),
        "is_central": np.int32(is_central),
        "meta": meta,
    }


def generate_dataset(
    path,
    n_examples,
    seed=0,
    grid_cfg=None,
    traj_cfg=None,
    data_cfg=None,
    verbose=True,
):
    """Generate `n_examples` trajectories and save them to an .npz file.

    Returns the path that was written to (convenient for chaining).
    """
    if grid_cfg is None:
        grid_cfg = GridConfig()
    if traj_cfg is None:
        traj_cfg = TrajConfig()
    if data_cfg is None:
        data_cfg = DataConfig()

    rng = np.random.default_rng(seed)
    xs, ys, _, _ = make_grid(grid_cfg)

    potentials = np.zeros((n_examples, grid_cfg.nx, grid_cfg.ny), dtype=np.float32)
    init_states = np.zeros((n_examples, 4), dtype=np.float32)
    trajectories = np.zeros((n_examples, traj_cfg.steps + 1, 4), dtype=np.float32)
    traj_mask = np.zeros((n_examples, traj_cfg.steps + 1), dtype=np.float32)

    energies = np.zeros((n_examples, traj_cfg.steps + 1), dtype=np.float32)
    Lzs = np.zeros((n_examples, traj_cfg.steps + 1), dtype=np.float32)

    offsets = np.zeros(n_examples, dtype=np.float32)
    is_central = np.zeros(n_examples, dtype=np.int32)
    family_id = np.zeros(n_examples, dtype=np.int32)
    generator_id = np.zeros(n_examples, dtype=np.int32)

    for i in range(n_examples):
        ex = generate_example(grid_cfg, traj_cfg, data_cfg, rng)
        T = len(ex["trajectory"])

        potentials[i] = ex["potential"]
        init_states[i] = ex["init_state"]
        trajectories[i, :T] = ex["trajectory"]
        traj_mask[i, :T] = 1.0
        energies[i, :T] = ex["energy"]
        Lzs[i, :T] = ex["angular_momentum"]

        offsets[i] = ex["meta"]["offset"]
        is_central[i] = ex["is_central"]

        if ex["meta"]["type"].startswith("central"):
            family_id[i] = 0
            generator_id[i] = 0
        else:
            family_id[i] = 1
            generator_id[i] = 1 if ex["meta"]["generator"] == "fourier" else 2

        if verbose and ((i + 1) % 100 == 0 or (i + 1) == n_examples):
            print(f"Generated {i + 1}/{n_examples}")

    np.savez_compressed(
        path,
        potentials=potentials,
        init_states=init_states,
        trajectories=trajectories,
        traj_mask=traj_mask,
        energies=energies,
        angular_momenta=Lzs,
        offsets=offsets,
        is_central=is_central,
        family_id=family_id,
        generator_id=generator_id,
        xs=xs.astype(np.float32),
        ys=ys.astype(np.float32),
        dt=np.float32(traj_cfg.dt),
        mass=np.float32(traj_cfg.mass),
    )

    if verbose:
        print(f"Saved dataset to {path}")

    return path
