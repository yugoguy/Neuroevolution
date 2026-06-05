"""Backprop inner loop and fitness.

For each genome, the dense weight matrix `W` is trained by gradient descent
(Adam) on the binary cross-entropy of its sigmoid output, then scored. This
replaces NEAT's genetic weight search; the GA evolves only topology.

Parallelism (two axes, both via the shared dense tensor shape):
  - population P: `vmap` the whole train-then-score routine over genomes, so all
    `W [P, n_max, n_max]` train together on the GPU;
  - data N: inside each genome the N points are one batched matmul.

Gradients reach only real connections because the forward pass uses
`W * conn_mask` (see converter), so dead entries get zero gradient and Adam's
moments for them stay zero. No phantom edges, no re-masking.

Fitness = -BCE - penalty_conn * sqrt(#enabled connections), following otoro's
backprop-NEAT (the sqrt is concave: 50 vs 51 connections is ~equivalent, 4 vs 5
is not). An optional node penalty is available.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from converter import Static, forward_single

_EPS = 1e-7


def _bce_per_genome(W, static: Static, obs, y, act_fns, num_passes):
    """Per-genome BCE over the data batch. W: [P, n, n] -> losses [P]."""
    f = partial(forward_single, act_fns=act_fns, num_passes=num_passes)
    out = jax.vmap(f, in_axes=(0, 0, None))(W, static, obs)   # [P, N, 1]
    p = jnp.clip(out[..., 0], _EPS, 1.0 - _EPS)               # [P, N]
    yy = y[None, :]
    return -jnp.mean(yy * jnp.log(p) + (1.0 - yy) * jnp.log(1.0 - p), axis=1)  # [P]


def _accuracy_per_genome(W, static: Static, obs, y, act_fns, num_passes):
    f = partial(forward_single, act_fns=act_fns, num_passes=num_passes)
    out = jax.vmap(f, in_axes=(0, 0, None))(W, static, obs)   # [P, N, 1]
    pred = (out[..., 0] > 0.5).astype(jnp.float32)            # [P, N]
    return jnp.mean((pred == y[None, :]).astype(jnp.float32), axis=1)  # [P]


def train_and_score(
    W,
    static: Static,
    X_train, y_train, X_test, y_test,
    act_fns: tuple[Callable, ...],
    *,
    num_passes: int,
    steps: int,
    lr: float,
    penalty_conn: float,
    penalty_node: float = 0.0,
    b1: float = 0.9, b2: float = 0.999,
):
    """Train every genome's W (masked Adam) then score it.

    Returns (fitness [P], W_trained [P,n,n], train_acc [P], test_acc [P],
             bce [P], conn_counts [P]).
    """
    Xtr = jnp.asarray(X_train); ytr = jnp.asarray(y_train)
    Xte = jnp.asarray(X_test);  yte = jnp.asarray(y_test)

    def total_loss(W):
        losses = _bce_per_genome(W, static, Xtr, ytr, act_fns, num_passes)
        return losses.sum(), losses                # sum -> per-genome grads in W

    grad_fn = jax.value_and_grad(total_loss, has_aux=True)

    @jax.jit
    def run(W):
        m = jnp.zeros_like(W); v = jnp.zeros_like(W)

        def adam_step(carry, t):
            W, m, v = carry
            (_, _losses), g = grad_fn(W)
            m = b1 * m + (1.0 - b1) * g
            v = b2 * v + (1.0 - b2) * (g * g)
            mh = m / (1.0 - b1 ** (t + 1))
            vh = v / (1.0 - b2 ** (t + 1))
            W = W - lr * mh / (jnp.sqrt(vh) + _EPS)
            return (W, m, v), None

        (W, _, _), _ = jax.lax.scan(adam_step, (W, m, v), jnp.arange(steps))
        return W

    W = run(W)

    bce = _bce_per_genome(W, static, Xtr, ytr, act_fns, num_passes)
    train_acc = _accuracy_per_genome(W, static, Xtr, ytr, act_fns, num_passes)
    test_acc = _accuracy_per_genome(W, static, Xte, yte, act_fns, num_passes)

    conn_counts = static.conn_mask.sum(axis=(1, 2))            # [P]
    node_counts = static.node_mask.sum(axis=1)                 # [P]
    fitness = -bce - penalty_conn * jnp.sqrt(conn_counts) - penalty_node * jnp.sqrt(node_counts)

    import numpy as np
    return (np.asarray(fitness), np.asarray(W), np.asarray(train_acc),
            np.asarray(test_acc), np.asarray(bce), np.asarray(conn_counts))
