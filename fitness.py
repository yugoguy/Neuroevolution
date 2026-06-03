"""Fitness evaluation.

Rolls the whole population through SlimeVolley in parallel on the GPU and returns
each genome's episode return (net score against the built-in opponent).

One env per genome (batch B = population size). The rollout is a `lax.scan` over
the fixed horizon; in training mode the task never terminates early, so the
return is simply the summed per-step reward — no done-masking required.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from converter import Model


def evaluate(
    models: Model,
    task,
    forward_fn,
    key: jax.Array,
    action_threshold: float = 0.0,
) -> jax.Array:
    """Return fitness [P]: episode return for each genome in `models`.

    `models` is a stacked (batched) Model of width P. `forward_fn` is the jitted
    forward built by `converter.build_forward`. `task` is a training-mode
    SlimeVolley (test=False) whose horizon defines the rollout length.
    """
    pop = models.W.shape[0]
    keys = jax.random.split(key, pop)
    state = task.reset(keys)

    def step(carry, _):
        state, total = carry
        out = forward_fn(models, state.obs)                 # [P, 3] raw outputs
        action = (out > action_threshold).astype(jnp.float32)
        state, reward, _done = task.step(state, action)
        return (state, total + reward), None

    (_, total), _ = jax.lax.scan(
        step, (state, jnp.zeros(pop)), None, length=task.max_steps
    )
    return total
