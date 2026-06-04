"""Genome -> model converter and the JAX feed-forward pass.

`express` compiles one irregular genome into fixed-size dense tensors (a `Model`)
so a whole population shares one tensor shape and can be batched on the GPU.
`stack_models` stacks per-genome models into a leading population axis.
`build_forward` returns a jitted, vmapped forward function.

Dense slot layout mirrors the canonical id layout: reserved nodes keep slots
0..reserved-1 (sensor inputs, then bias, then outputs), and hidden nodes are
appended in id order. Unused slots are masked out.

Forward evaluation
------------------
A NEAT genome is a feed-forward DAG of varying depth. To keep one fixed,
batchable computation for every genome we evaluate it by repeated synchronous
propagation: clamp the inputs, apply `x <- activation(W x)` `num_passes` times,
re-clamping inputs each step. After P passes, every node at depth <= P holds its
correct value, so `num_passes` only needs to be at least the longest path. This
is a deliberate simplicity/speed trade-off (see DESIGN NOTE in the report): a
fixed pass count avoids per-genome control flow at the cost of some redundant
matmuls. `num_passes` is a hyperparameter; `n_max` is a always-safe upper bound.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from genome import Genome, io_ids, reserved_node_count, INPUT, OUTPUT


# --- Activation implementations --------------------------------------------
# `config` chooses *which* activations are available (by name); this module
# provides the implementations. The chosen ordered name list defines the
# integer ids stored per node.

DEFAULT_ACTIVATIONS: dict[str, Callable] = {
    "identity": lambda x: x,
    "tanh": jnp.tanh,
    "relu": lambda x: jnp.maximum(x, 0.0),
    "sigmoid": jax.nn.sigmoid,
    "sin": jnp.sin,
    "gauss": lambda x: jnp.exp(-(x ** 2)),
    "abs": jnp.abs,
}


def activation_ids(names: list[str]) -> dict[str, int]:
    """Map activation name -> integer id (its index in the ordered list)."""
    return {name: i for i, name in enumerate(names)}


def activation_fns(names: list[str]) -> tuple[Callable, ...]:
    """Ordered tuple of activation callables matching `activation_ids`."""
    return tuple(DEFAULT_ACTIVATIONS[name] for name in names)


# --- Compiled model ---------------------------------------------------------

class Model(NamedTuple):
    W: jnp.ndarray          # [n_max, n_max]  W[j, i] = weight of edge i -> j
    act_ids: jnp.ndarray    # [n_max]         activation id per slot
    mask: jnp.ndarray       # [n_max]         1.0 for existing nodes, else 0.0
    in_slots: jnp.ndarray   # [num_inputs]    slots fed the observation vector
    bias_slot: jnp.ndarray  # scalar          slot fed the constant 1.0
    out_slots: jnp.ndarray  # [num_outputs]   slots read as outputs


def express(
    genome: Genome,
    num_inputs: int,
    num_outputs: int,
    n_max: int,
    act_to_id: dict[str, int],
) -> Model:
    """Compile a genome into fixed-size dense tensors of width `n_max`."""
    inputs, bias, outputs = io_ids(num_inputs, num_outputs)
    reserved = reserved_node_count(num_inputs, num_outputs)

    # Slot assignment: reserved ids keep their own value as slot; hidden nodes
    # are appended in ascending id order.
    slot: dict[int, int] = {nid: nid for nid in range(reserved)}
    hidden_ids = sorted(nid for nid in genome.node_genes if nid >= reserved)
    if reserved + len(hidden_ids) > n_max:
        raise ValueError(f"genome needs {reserved + len(hidden_ids)} nodes > n_max={n_max}")
    for offset, nid in enumerate(hidden_ids):
        slot[nid] = reserved + offset

    W = np.zeros((n_max, n_max), dtype=np.float32)
    act_ids = np.zeros(n_max, dtype=np.int32)
    mask = np.zeros(n_max, dtype=np.float32)

    for nid, node in genome.node_genes.items():
        s = slot[nid]
        mask[s] = 1.0
        # Input/bias values are clamped every pass, so their activation is never
        # applied; give them id 0 instead of requiring it to be in the set.
        act_ids[s] = 0 if node.type == INPUT else act_to_id[node.activation]

    for c in genome.conn_genes.values():
        if c.enabled:
            W[slot[c.out_node], slot[c.in_node]] = c.weight

    return Model(
        W=W,
        act_ids=act_ids,
        mask=mask,
        in_slots=np.array([slot[i] for i in inputs], dtype=np.int32),
        bias_slot=np.int32(slot[bias]),
        out_slots=np.array([slot[o] for o in outputs], dtype=np.int32),
    )


def stack_models(models: list[Model]) -> Model:
    """Stack per-genome models into a leading population axis (as JAX arrays)."""
    return Model(*(jnp.stack([jnp.asarray(getattr(m, f)) for m in models])
                   for f in Model._fields))


# --- Forward pass -----------------------------------------------------------

def _forward_single(
    model: Model,
    obs: jnp.ndarray,
    act_fns: tuple[Callable, ...],
    num_passes: int,
) -> jnp.ndarray:
    """Evaluate one network on one observation; returns raw output values."""
    x = jnp.zeros(model.W.shape[0])
    x = x.at[model.in_slots].set(obs)
    x = x.at[model.bias_slot].set(1.0)

    def step(x, _):
        pre = model.W @ x
        acts = jnp.stack([f(pre) for f in act_fns], axis=0)        # [num_act, n]
        post = jnp.take_along_axis(acts, model.act_ids[None, :], axis=0)[0]
        x = post * model.mask
        x = x.at[model.in_slots].set(obs)
        x = x.at[model.bias_slot].set(1.0)
        return x, None

    x, _ = jax.lax.scan(step, x, None, length=num_passes)
    return x[model.out_slots]


def build_forward(act_fns: tuple[Callable, ...], num_passes: int) -> Callable:
    """Return a jitted forward pass over a batched Model and obs batch.

    The returned function maps (stacked Model [P, ...], obs [P, num_inputs]) to
    raw outputs [P, num_outputs]. Action thresholding is the task module's job.
    """
    f = partial(_forward_single, act_fns=act_fns, num_passes=num_passes)
    return jax.jit(jax.vmap(f, in_axes=(0, 0)))
