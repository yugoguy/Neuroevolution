"""Toy 2D binary-classification datasets (circle, XOR, spiral).

Mirrors the tasks in hardmaru's backprop-neat-js. Each generator returns
(X [N, 2], y [N]) with y in {0, 1}. Coordinates are centered near the origin on
roughly a [-6, 6] scale, matching the reference demo.
"""

from __future__ import annotations

import numpy as np

DATASETS = ("circle", "xor", "spiral")


def make_dataset(name: str, n: int, rng: np.random.Generator,
                 noise: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    if name == "circle":
        return _circle(n, rng, noise)
    if name == "xor":
        return _xor(n, rng, noise)
    if name == "spiral":
        return _spiral(n, rng, noise)
    raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")


def _circle(n: int, rng: np.random.Generator, noise: float) -> tuple[np.ndarray, np.ndarray]:
    """Inner disk (class 1) vs. outer ring (class 0)."""
    half = n // 2
    r_in = rng.uniform(0.0, 2.5, half)
    r_out = rng.uniform(3.5, 6.0, n - half)
    a_in = rng.uniform(0, 2 * np.pi, half)
    a_out = rng.uniform(0, 2 * np.pi, n - half)
    X = np.concatenate([
        np.stack([r_in * np.cos(a_in), r_in * np.sin(a_in)], 1),
        np.stack([r_out * np.cos(a_out), r_out * np.sin(a_out)], 1),
    ])
    y = np.concatenate([np.ones(half), np.zeros(n - half)])
    return _finish(X, y, rng, noise)


def _xor(n: int, rng: np.random.Generator, noise: float) -> tuple[np.ndarray, np.ndarray]:
    """Checkerboard quadrants: class = 1 iff x and y share sign."""
    X = rng.uniform(-5.0, 5.0, size=(n, 2))
    y = (np.sign(X[:, 0]) == np.sign(X[:, 1])).astype(float)
    return _finish(X, y, rng, noise)


def _spiral(n: int, rng: np.random.Generator, noise: float) -> tuple[np.ndarray, np.ndarray]:
    """Two intertwined spiral arms."""
    half = n // 2
    t = np.sqrt(rng.uniform(0, 1, half)) * 3.0 * np.pi
    r = t * 0.6
    arm0 = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    arm1 = np.stack([r * np.cos(t + np.pi), r * np.sin(t + np.pi)], 1)
    X = np.concatenate([arm0, arm1[: n - half]])
    y = np.concatenate([np.zeros(half), np.ones(n - half)])
    return _finish(X, y, rng, noise)


def _finish(X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
            noise: float) -> tuple[np.ndarray, np.ndarray]:
    X = X + rng.normal(0.0, noise, X.shape)
    perm = rng.permutation(len(y))
    return X[perm].astype(np.float32), y[perm].astype(np.float32)
