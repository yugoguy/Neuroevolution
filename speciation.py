"""Speciation.

Clusters genomes by compatibility distance using NEAT's greedy first-match
scheme: each genome joins the first species whose representative is within the
threshold, otherwise it founds a new species and becomes a representative that
later genomes may join. The first generation (no prior representatives) is
handled by the same single pass.

Excess and disjoint counts are merged into one "unmatched" term, since the
original NEAT used equal coefficients for them in practice.

This module only assigns species and picks representatives. Fitness sharing and
offspring allocation belong to the reproduction module.
"""

from __future__ import annotations

import numpy as np

from genome import Genome


def compatibility_distance(
    g1: Genome,
    g2: Genome,
    c_unmatched: float,
    c_weight: float,
    normalize_threshold: int = 20,
) -> float:
    """δ = c_unmatched · (excess+disjoint)/N + c_weight · mean |Δweight|.

    N is the larger gene count, but set to 1 for small genomes (<= threshold) to
    avoid over-penalizing them, following the original NEAT.
    """
    k1, k2 = set(g1.conn_genes), set(g2.conn_genes)
    matching = k1 & k2
    unmatched = len(k1 ^ k2)

    if matching:
        wdiff = np.mean([abs(g1.conn_genes[i].weight - g2.conn_genes[i].weight)
                         for i in matching])
    else:
        wdiff = 0.0

    n = max(len(k1), len(k2))
    n = 1 if n <= normalize_threshold else n
    return c_unmatched * unmatched / n + c_weight * float(wdiff)


def speciate(
    genomes: list[Genome],
    representatives: dict[int, Genome],
    rng: np.random.Generator,
    threshold: float,
    c_unmatched: float,
    c_weight: float,
    normalize_threshold: int = 20,
) -> tuple[list[int], dict[int, Genome]]:
    """Assign each genome to a species.

    `representatives` maps species id -> representative genome from the previous
    generation (empty on the first generation). Returns:
      - assignment: species id per genome (aligned with `genomes`);
      - new_representatives: one freshly chosen representative per surviving
        species, for use next generation.
    """
    reps = dict(representatives)
    next_species_id = (max(reps) + 1) if reps else 0

    assignment: list[int] = []
    members: dict[int, list[Genome]] = {}

    for g in genomes:
        placed = None
        for sid, rep in reps.items():
            d = compatibility_distance(g, rep, c_unmatched, c_weight, normalize_threshold)
            if d < threshold:
                placed = sid
                break
        if placed is None:
            placed = next_species_id
            next_species_id += 1
            reps[placed] = g          # founder becomes representative immediately
        assignment.append(placed)
        members.setdefault(placed, []).append(g)

    # New representative per surviving species: a random current member.
    new_reps: dict[int, Genome] = {}
    for sid, mem in members.items():
        new_reps[sid] = mem[int(rng.integers(len(mem)))]

    return assignment, new_reps
