"""Stagnation tracking (original NEAT, Section 3.3 footnote + Section 4.1).

Two rules from the 2002 paper decide which species may reproduce each generation:

  - Per-species: "If the maximum fitness of a species did not improve in 15
    generations, the networks in the stagnant species were not allowed to
    reproduce." (`max_stagnation`)
  - Population stall: "In rare cases when the fitness of the entire population
    does not improve for more than 20 generations, only the top two species are
    allowed to reproduce." (`population_stall`) This overrides the per-species
    rule when it triggers.

The tracker holds per-species and whole-population best-fitness history across
generations and returns the set of species permitted to reproduce. Reproduction
itself stays mechanism-only; this module owns the policy. Either rule is disabled
by setting its threshold to 0.

Guard (not in the paper, to avoid an empty population): if the per-species rule
would bar every species, the single best species is spared. The orchestrator
also tracks the all-time best genome separately, so barring never loses it from
the reported result.
"""

from __future__ import annotations

import numpy as np


class StagnationTracker:
    def __init__(self, max_stagnation: int, population_stall: int):
        self.max_stagnation = max_stagnation        # 0 disables the per-species rule
        self.population_stall = population_stall     # 0 disables the population-stall rule
        self._species_best: dict[int, float] = {}
        self._species_last_improved: dict[int, int] = {}
        self._pop_best = -np.inf
        self._pop_last_improved = 0

    def update(self, gen: int, assignment: list[int], fitnesses) -> set[int]:
        """Record this generation and return the species allowed to reproduce."""
        f = np.asarray(fitnesses, dtype=float)

        # Max fitness per species this generation.
        species_max: dict[int, float] = {}
        for idx, sid in enumerate(assignment):
            if sid not in species_max or f[idx] > species_max[sid]:
                species_max[sid] = float(f[idx])

        # Update per-species improvement records; new species start "fresh".
        for sid, m in species_max.items():
            if sid not in self._species_best or m > self._species_best[sid]:
                self._species_best[sid] = m
                self._species_last_improved[sid] = gen
        # Drop records for species that no longer exist (ids are never reused).
        live = set(species_max)
        self._species_best = {s: v for s, v in self._species_best.items() if s in live}
        self._species_last_improved = {
            s: v for s, v in self._species_last_improved.items() if s in live
        }

        # Whole-population improvement.
        pop_max = max(species_max.values())
        if pop_max > self._pop_best:
            self._pop_best = pop_max
            self._pop_last_improved = gen

        # Population-stall rule overrides everything when it triggers.
        if self.population_stall and (gen - self._pop_last_improved) >= self.population_stall:
            top2 = sorted(species_max, key=lambda s: species_max[s], reverse=True)[:2]
            return set(top2)

        # Per-species stagnation rule.
        if self.max_stagnation:
            allowed = {
                s for s in live
                if (gen - self._species_last_improved[s]) < self.max_stagnation
            }
            if not allowed:                      # guard: never empty the population
                allowed = {max(species_max, key=lambda s: species_max[s])}
            return allowed

        return live
