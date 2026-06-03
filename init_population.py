"""Initial population.

Produces `pop_size` minimal genomes (inputs+bias fully connected to outputs, no
hidden nodes). Topology is identical across individuals; only connection weights
are randomized, so the run starts simple and complexifies through mutation.

Uses the canonical builder in `genome.py`, so the numbering matches whatever the
innovation registry was initialized with.
"""

from __future__ import annotations

import numpy as np

from genome import Genome, initial_genome, reserved_conn_count


def init_population(
    pop_size: int,
    num_inputs: int,
    num_outputs: int,
    rng: np.random.Generator,
    weight_init_std: float,
    output_activation: str = "tanh",
) -> list[Genome]:
    n_conn = reserved_conn_count(num_inputs, num_outputs)
    pop: list[Genome] = []
    for _ in range(pop_size):
        weights = rng.normal(0.0, weight_init_std, size=n_conn)
        pop.append(initial_genome(num_inputs, num_outputs, output_activation, weights))
    return pop
