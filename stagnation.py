"""Stagnation tracking (original NEAT, Section 3.3 footnote + Section 4.1).

  - Per-species: a species whose max fitness has not improved in
    `max_stagnation` generations may not reproduce.
  - Population stall: if the whole population's fitness stalls for
    `population_stall` generations, only the top two species reproduce (this
    overrides the per-species rule).

Either rule is disabled by setting its threshold to 0. Guard (not in the paper):
if the per-species rule would bar every species, the single best is spared.
"""

from __future__ import annotations

import numpy as np


class StagnationTracker:
    def __init__(self, max_stagnation: int, population_stall: int):
        self.max_stagnation = max_stagnation
        self.population_stall = population_stall
        self._species_best: dict[int, float] = {}
        self._species_last_improved: dict[int, int] = {}
        self._pop_best = -np.inf
        self._pop_last_improved = 0

    def update(self, gen: int, assignment: list[int], fitnesses) -> set[int]:
        f = np.asarray(fitnesses, dtype=float)

        species_max: dict[int, float] = {}
        for idx, sid in enumerate(assignment):
            if sid not in species_max or f[idx] > species_max[sid]:
                species_max[sid] = float(f[idx])

        for sid, m in species_max.items():
            if sid not in self._species_best or m > self._species_best[sid]:
                self._species_best[sid] = m
                self._species_last_improved[sid] = gen
        live = set(species_max)
        self._species_best = {s: v for s, v in self._species_best.items() if s in live}
        self._species_last_improved = {
            s: v for s, v in self._species_last_improved.items() if s in live
        }

        pop_max = max(species_max.values())
        if pop_max > self._pop_best:
            self._pop_best = pop_max
            self._pop_last_improved = gen

        if self.population_stall and (gen - self._pop_last_improved) >= self.population_stall:
            top2 = sorted(species_max, key=lambda s: species_max[s], reverse=True)[:2]
            return set(top2)

        if self.max_stagnation:
            allowed = {
                s for s in live
                if (gen - self._species_last_improved[s]) < self.max_stagnation
            }
            if not allowed:
                allowed = {max(species_max, key=lambda s: species_max[s])}
            return allowed

        return live
