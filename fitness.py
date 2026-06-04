"""Fitness evaluation.

Rolls genomes through SlimeVolley in parallel on the GPU. Each genome is
evaluated over `episodes` rollouts (different env seeds); the genome's fitness
is the mean episode return, optionally plus a survival term.

Parallelism: the rollouts are the population crossed with episodes -> P*episodes
independent games. These are stacked into one batch axis and run together. When
that is too large for GPU memory, `batch_size` splits them into chunks run in
sequence (default 0 = one batch of everything). So both genomes and episodes are
parallelized; `batch_size` is the ratio of how many of the P*episodes rollouts
run at once.

Accumulation is masked after an episode signals done, so this is correct in both
training mode (full horizon, score margin; survival term constant) and test mode
(real match, ends early; survival term meaningful).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import tree_util

from converter import Model


def _rollout(models, task, forward_fn, keys, action_threshold, survival_weight):
    """One batched rollout over keys.shape[0] games; returns per-game fitness."""
    b = keys.shape[0]
    state = task.reset(keys)
    init = (state, jnp.zeros(b), jnp.zeros(b), jnp.zeros(b))  # state, return, done, alive

    def step(carry, _):
        state, ret, done_acc, alive = carry
        out = forward_fn(models, state.obs)
        action = (out > action_threshold).astype(jnp.float32)
        state, reward, done = task.step(state, action)
        not_done = 1.0 - done_acc
        ret = ret + reward * not_done
        alive = alive + not_done
        done_acc = jnp.maximum(done_acc, done.astype(jnp.float32))
        return (state, ret, done_acc, alive), None

    (_, ret, _, alive), _ = jax.lax.scan(step, init, None, length=task.max_steps)
    return ret + survival_weight * (alive / task.max_steps)


def evaluate(
    models: Model,
    task,
    forward_fn,
    key: jax.Array,
    *,
    episodes: int = 1,
    action_threshold: float = 0.0,
    survival_weight: float = 0.0,
    batch_size: int = 0,
) -> jax.Array:
    """Return fitness [P] = per-genome mean return over `episodes` rollouts."""
    pop = models.W.shape[0]
    total = pop * episodes

    # One rollout per (genome, episode). Replicate each genome `episodes` times;
    # rollouts are ordered genome-major, so a [pop, episodes] reshape recovers them.
    rep = jnp.repeat(jnp.arange(pop), episodes)
    big = tree_util.tree_map(lambda a: a[rep], models)
    keys = jax.random.split(key, total)

    bs = total if not batch_size else batch_size
    chunks = []
    for s in range(0, total, bs):
        sub = tree_util.tree_map(lambda a: a[s:s + bs], big)
        chunks.append(_rollout(sub, task, forward_fn, keys[s:s + bs],
                               action_threshold, survival_weight))

    ret = jnp.concatenate(chunks)              # [total]
    return ret.reshape(pop, episodes).mean(axis=1)
