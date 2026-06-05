"""Speciation.

Clusters genomes by compatibility distance using NEAT's greedy first-match
scheme: each genome joins the first species whose representative is within the
threshold, otherwise founds a new species. Excess and disjoint counts are merged
into one "unmatched" term (the original used equal coefficients in practice).

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
    """delta = c_unmatched * (excess+disjoint)/N + c_weight * mean|dweight|.

    N is the larger gene count, set to 1 for small genomes (<= threshold).
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
    rep_selection: str = "random",
) -> tuple[list[int], dict[int, Genome]]:
    """Assign each genome to a species; return (assignment, new representatives)."""
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
            reps[placed] = g
        assignment.append(placed)
        members.setdefault(placed, []).append(g)

    new_reps: dict[int, Genome] = {}
    for sid, mem in members.items():
        if rep_selection == "first":
            new_reps[sid] = mem[0]
        else:
            new_reps[sid] = mem[int(rng.integers(len(mem)))]

    return assignment, new_reps
