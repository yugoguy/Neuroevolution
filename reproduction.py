"""Selection and reproduction.

Builds the next generation from the current one, given per-genome fitness and a
species assignment. Implements NEAT's explicit fitness sharing and per-species
offspring allocation, a survival cull, and elitism (champion carried over).

Decoupling: the actual variation operators arrive as callables, so this module
depends on neither the mutation/crossover internals nor the config:
  - crossover_fn(parent1, parent2, fitness1, fitness2) -> child
  - mutate_fn(genome) -> mutated copy
The orchestrator binds rng / registry / rates into these before calling.

Negative fitness: SlimeVolley returns can be negative, which breaks proportional
allocation, so fitnesses are shifted to be non-negative first (DESIGN NOTE: a
flat shift by the population minimum; reported in config commentary).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from genome import Genome


def _allocate(scores: dict[int, float], pop_size: int) -> dict[int, int]:
    """Largest-remainder allocation of `pop_size` slots proportional to scores."""
    total = sum(scores.values())
    if total <= 0:
        # Degenerate: split evenly.
        sids = list(scores)
        base = pop_size // len(sids)
        counts = {sid: base for sid in sids}
        for sid in sids[: pop_size - base * len(sids)]:
            counts[sid] += 1
        return counts

    exact = {sid: pop_size * s / total for sid, s in scores.items()}
    counts = {sid: int(math.floor(v)) for sid, v in exact.items()}
    remainder = pop_size - sum(counts.values())
    # Hand out the leftover slots to the largest fractional parts.
    frac = sorted(scores, key=lambda sid: exact[sid] - counts[sid], reverse=True)
    for sid in frac[:remainder]:
        counts[sid] += 1
    return counts


def reproduce(
    genomes: list[Genome],
    fitnesses: np.ndarray,
    assignment: list[int],
    rng: np.random.Generator,
    *,
    pop_size: int,
    survival_threshold: float,
    elitism_min_species_size: int,
    mutate_only_prob: float,
    parent_selection: str,
    interspecies_mating_prob: float,
    crossover_fn: Callable[[Genome, Genome, float, float], Genome],
    mutate_fn: Callable[[Genome], Genome],
    allowed_species: set | None = None,
) -> list[Genome]:
    fitnesses = np.asarray(fitnesses, dtype=float)

    # Group genome indices by species.
    species: dict[int, list[int]] = {}
    for idx, sid in enumerate(assignment):
        species.setdefault(sid, []).append(idx)

    # Species barred by the stagnation rules get no offspring (and no elite copy).
    allowed = set(species) if allowed_species is None else (set(allowed_species) & set(species))
    if not allowed:
        allowed = set(species)

    # Shift fitness non-negative, then explicit fitness sharing: each genome's
    # contribution is divided by its species size. The shifted values double as
    # selection weights for parent_selection == "fitness_weighted".
    shifted = fitnesses - fitnesses.min() + 1e-6
    species_score = {
        sid: float(np.sum(shifted[idxs]) / len(idxs))
        for sid, idxs in species.items() if sid in allowed
    }
    alloc = _allocate(species_score, pop_size)

    def pick(survivors: list[int]) -> int:
        if parent_selection == "fitness_weighted" and len(survivors) > 1:
            w = shifted[survivors]
            return int(rng.choice(survivors, p=w / w.sum()))
        return survivors[int(rng.integers(len(survivors)))]

    def pick_two(survivors: list[int]) -> tuple[int, int]:
        if parent_selection == "fitness_weighted":
            w = shifted[survivors]
            i, j = rng.choice(survivors, size=2, replace=False, p=w / w.sum())
            return int(i), int(j)
        i, j = rng.choice(len(survivors), size=2, replace=False)
        return survivors[i], survivors[j]

    next_pop: list[Genome] = []
    for sid, idxs in species.items():
        n = alloc.get(sid, 0)
        if n == 0:
            continue

        # Rank members of this species by raw fitness, best first.
        ranked = sorted(idxs, key=lambda i: fitnesses[i], reverse=True)
        n_survivors = max(1, math.ceil(survival_threshold * len(ranked)))
        survivors = ranked[:n_survivors]

        produced = 0
        # Elitism: carry the champion over unchanged for large-enough species.
        if len(ranked) >= elitism_min_species_size:
            next_pop.append(genomes[ranked[0]].copy())
            produced = 1

        while produced < n:
            if len(survivors) == 1 or rng.random() < mutate_only_prob:
                child = mutate_fn(genomes[pick(survivors)])
            else:
                a, b = pick_two(survivors)
                # Rarely, draw the second parent from anywhere in the population.
                if rng.random() < interspecies_mating_prob:
                    b = int(rng.integers(len(genomes)))
                child = crossover_fn(genomes[a], genomes[b], fitnesses[a], fitnesses[b])
                child = mutate_fn(child)
            next_pop.append(child)
            produced += 1

    # Guard against rounding drift.
    if len(next_pop) > pop_size:
        next_pop = next_pop[:pop_size]
    while len(next_pop) < pop_size:
        best = int(np.argmax(fitnesses))
        next_pop.append(mutate_fn(genomes[best]))

    return next_pop
