"""Genome -> dense tensors and the differentiable JAX forward pass.

`express` compiles one irregular genome into fixed-size dense tensors of width
`n_max`, separating the *trainable* weight matrix `W` from the *static* structure
(`Static`: masks, activation ids, slot indices). A whole population shares one
tensor shape, so backprop and scoring batch on the GPU.

Why the split: backprop differentiates the loss w.r.t. `W` only. The forward pass
uses `W_eff = W * conn_mask`, so the gradient on any non-edge is exactly zero —
disabled/absent connections never acquire a weight, the evolved topology is
preserved, and the connection-count penalty stays meaningful. No post-step
re-masking is needed because the mask lives inside the computation graph.

Forward evaluation: a NEAT genome is a feed-forward DAG. We evaluate by repeated
synchronous propagation: clamp inputs, apply `x <- activation(W_eff x)`
`num_passes` times, re-clamping inputs each step. After P passes every node at
depth <= P holds its correct value, so `num_passes` need only exceed the longest
path; `n_max` is an always-safe upper bound.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from genome import Genome, io_ids, reserved_node_count, INPUT


# --- Differentiable activation implementations -----------------------------
# Restricted to differentiable functions (relu/abs are differentiable a.e.).
# `config` chooses *which* are available by name; this provides the impls.

DEFAULT_ACTIVATIONS: dict[str, Callable] = {
    "tanh": jnp.tanh,
    "relu": lambda x: jnp.maximum(x, 0.0),
    "sigmoid": jax.nn.sigmoid,
    "sin": jnp.sin,
    "gauss": lambda x: jnp.exp(-(x ** 2)),
    "abs": jnp.abs,
    "square": lambda x: x ** 2,
}


def activation_ids(names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(names)}


def activation_fns(names: list[str]) -> tuple[Callable, ...]:
    return tuple(DEFAULT_ACTIVATIONS[name] for name in names)


# --- Compiled model ---------------------------------------------------------

class Static(NamedTuple):
    conn_mask: jnp.ndarray   # [n_max, n_max]  1.0 where an enabled edge exists
    act_ids: jnp.ndarray     # [n_max]         activation id per slot
    node_mask: jnp.ndarray   # [n_max]         1.0 for existing nodes
    in_slots: jnp.ndarray    # [num_inputs]
    bias_slot: jnp.ndarray   # scalar
    out_slots: jnp.ndarray   # [num_outputs]


def _slot_map(genome: Genome, num_inputs: int, num_outputs: int, n_max: int) -> dict[int, int]:
    """Reserved ids keep their own slot; hidden nodes appended in id order."""
    reserved = reserved_node_count(num_inputs, num_outputs)
    slot = {nid: nid for nid in range(reserved)}
    hidden_ids = sorted(nid for nid in genome.node_genes if nid >= reserved)
    if reserved + len(hidden_ids) > n_max:
        raise ValueError(f"genome needs {reserved + len(hidden_ids)} nodes > n_max={n_max}")
    for offset, nid in enumerate(hidden_ids):
        slot[nid] = reserved + offset
    return slot


def express(
    genome: Genome,
    num_inputs: int,
    num_outputs: int,
    n_max: int,
    act_to_id: dict[str, int],
) -> tuple[np.ndarray, Static]:
    """Compile a genome into (W_init [n_max, n_max], Static)."""
    inputs, bias, outputs = io_ids(num_inputs, num_outputs)
    slot = _slot_map(genome, num_inputs, num_outputs, n_max)

    W = np.zeros((n_max, n_max), dtype=np.float32)
    conn_mask = np.zeros((n_max, n_max), dtype=np.float32)
    act_ids = np.zeros(n_max, dtype=np.int32)
    node_mask = np.zeros(n_max, dtype=np.float32)

    for nid, node in genome.node_genes.items():
        s = slot[nid]
        node_mask[s] = 1.0
        # Inputs/bias are clamped every pass, so their activation is never
        # applied; id 0 is a safe placeholder.
        act_ids[s] = 0 if node.type == INPUT else act_to_id[node.activation]

    for c in genome.conn_genes.values():
        if c.enabled:
            j, i = slot[c.out_node], slot[c.in_node]
            W[j, i] = c.weight
            conn_mask[j, i] = 1.0

    static = Static(
        conn_mask=conn_mask,
        act_ids=act_ids,
        node_mask=node_mask,
        in_slots=np.array([slot[i] for i in inputs], dtype=np.int32),
        bias_slot=np.int32(slot[bias]),
        out_slots=np.array([slot[o] for o in outputs], dtype=np.int32),
    )
    return W, static


def stack_models(models: list[tuple[np.ndarray, Static]]) -> tuple[jnp.ndarray, Static]:
    """Stack per-genome (W, Static) into a leading population axis."""
    Ws = jnp.stack([jnp.asarray(W) for W, _ in models])
    statics = [s for _, s in models]
    stacked = Static(*(jnp.stack([jnp.asarray(getattr(s, f)) for s in statics])
                       for f in Static._fields))
    return Ws, stacked


def write_back(genome: Genome, W: np.ndarray, num_inputs: int, num_outputs: int,
               n_max: int) -> None:
    """Copy trained weights from `W` back into the genome's enabled connections.

    Lamarckian: the genome carries its trained weights forward as the warm-start
    for its offspring's backprop.
    """
    slot = _slot_map(genome, num_inputs, num_outputs, n_max)
    W = np.asarray(W)
    for c in genome.conn_genes.values():
        if c.enabled:
            c.weight = float(W[slot[c.out_node], slot[c.in_node]])


# --- Forward pass -----------------------------------------------------------

def forward_single(
    W: jnp.ndarray,
    static: Static,
    obs: jnp.ndarray,
    act_fns: tuple[Callable, ...],
    num_passes: int,
) -> jnp.ndarray:
    """Evaluate one network on a batch of observations.

    obs: [N, num_inputs] -> returns [N, num_outputs] (raw output-node values).
    """
    n = W.shape[0]
    N = obs.shape[0]
    W_eff = W * static.conn_mask

    x = jnp.zeros((N, n))
    x = x.at[:, static.in_slots].set(obs)
    x = x.at[:, static.bias_slot].set(1.0)

    idx = jnp.broadcast_to(static.act_ids[None, None, :], (1, N, n))

    def step(x, _):
        pre = x @ W_eff.T                                       # [N, n]
        acts = jnp.stack([f(pre) for f in act_fns], axis=0)     # [A, N, n]
        post = jnp.take_along_axis(acts, idx, axis=0)[0]        # [N, n]
        x = post * static.node_mask
        x = x.at[:, static.in_slots].set(obs)
        x = x.at[:, static.bias_slot].set(1.0)
        return x, None

    x, _ = jax.lax.scan(step, x, None, length=num_passes)
    return x[:, static.out_slots]


def build_forward(act_fns: tuple[Callable, ...], num_passes: int) -> Callable:
    """Return a jitted forward over a stacked population and a shared obs batch.

    Maps (W [P, n, n], Static [P, ...], obs [N, num_inputs]) -> [P, N, num_outputs].
    """
    f = partial(forward_single, act_fns=act_fns, num_passes=num_passes)
    return jax.jit(jax.vmap(f, in_axes=(0, 0, None)))
