"""Environment fetcher.

Thin wrapper over EvoJAX's SlimeVolley task. Keeps the rest of the library
independent of EvoJAX specifics: callers ask for a task and its dimensions.

SlimeVolley facts (verified against the EvoJAX source):
  - observation: 12-dim; action: 3-dim, interpreted as binary via `a > 0`
    (forward, backward, jump). The agent plays the right side against the
    built-in baseline opponent.
  - training mode (test=False): plays the full `max_steps` with no early
    termination; per-step reward is +1/-1 per point, so the episode return is
    the net score margin.
  - test mode (test=True): a real match that ends when a side loses 5 lives.
  - reset(keys) expects a batched key array [B, 2]; step(state, action) takes
    action [B, 3] and returns (state, reward [B], done [B]); state.obs is [B, 12].
"""

from __future__ import annotations

from evojax.task.slimevolley import SlimeVolley


def make_task(max_steps: int = 3000, test: bool = False) -> SlimeVolley:
    """Create a SlimeVolley task (training rollouts use test=False)."""
    return SlimeVolley(max_steps=max_steps, test=test)


def task_dims(task: SlimeVolley) -> tuple[int, int]:
    """Return (num_inputs, num_outputs) for the task."""
    return task.obs_shape[0], task.act_shape[0]
