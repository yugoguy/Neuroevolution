"""Fitness evaluation.

Rolls the whole population through SlimeVolley in parallel on the GPU and returns
each genome's fitness. One env per genome (batch B = population size); the rollout
is a `lax.scan` over the horizon.

Fitness = mean over `episodes` of the episode return, optionally plus a survival
term. Accumulation is masked after an episode signals done, so this is correct in
both modes:
  - training mode (test=False): the game never ends early, so the return is the
    net score margin over the full horizon and the survival term is a constant
    (set survival_weight=0).
  - test mode (test=True): a real 5-life match that can end early, where surviving
    longer is meaningful and `survival_weight` differentiates agents.
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
    *,
    episodes: int = 1,
    action_threshold: float = 0.0,
    survival_weight: float = 0.0,
) -> jax.Array:
    """Return fitness [P], averaged over `episodes` rollouts."""
    pop = models.W.shape[0]
    max_steps = task.max_steps

    def one_episode(ep_key):
        keys = jax.random.split(ep_key, pop)
        state = task.reset(keys)
        init = (state, jnp.zeros(pop), jnp.zeros(pop), jnp.zeros(pop))  # state, return, done_acc, alive

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

        (_, ret, _, alive), _ = jax.lax.scan(step, init, None, length=max_steps)
        return ret + survival_weight * (alive / max_steps)

    ep_keys = jax.random.split(key, episodes)
    fits = jnp.stack([one_episode(k) for k in ep_keys], axis=0)   # [episodes, P]
    return fits.mean(axis=0)
