"""Evolution recorder.

A cross-cutting logger the orchestrator feeds once per generation. It owns no
evolution logic; it derives statistics from the population state and keeps them
plus a serializable snapshot of each generation's best genome. Records are plain
JSON-friendly dicts so they can be dumped, reloaded, and plotted offline.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from dataclasses import asdict

import numpy as np

from genome import Genome, NodeGene, ConnGene, HIDDEN
from graph_utils import max_depth


def _dist(values) -> dict:
    """Summary of a distribution: min / mean / max."""
    a = np.asarray(values, dtype=float)
    return {"min": float(a.min()), "mean": float(a.mean()), "max": float(a.max())}


def _activation_counts(genome: Genome) -> Counter:
    """Activation usage over the genome's hidden nodes (outputs are fixed)."""
    return Counter(
        g.activation for g in genome.node_genes.values() if g.type == HIDDEN
    )


def load_genome(snapshot: dict) -> Genome:
    """Reconstruct a Genome from a recorded snapshot (keys may be str after JSON)."""
    nodes = {int(k): NodeGene(**v) for k, v in snapshot["node_genes"].items()}
    conns = {int(k): ConnGene(**v) for k, v in snapshot["conn_genes"].items()}
    return Genome(node_genes=nodes, conn_genes=conns)


class Recorder:
    def __init__(self):
        self.records: list[dict] = []
        self.best_snapshots: dict[int, dict] = {}   # gen -> asdict(best genome)
        self.species_best_snapshots: dict[int, dict] = {}  # gen -> {species_id: asdict(species best)}
        self._prev_species: set = set()
        self._best_fitness = -np.inf
        self._stagnation = 0

    def record(self, gen, pop, fitnesses, assignment, gen_time=None) -> dict:
        f = np.asarray(fitnesses, dtype=float)

        nodes = np.array([len(g.node_genes) for g in pop])
        hidden = np.array([sum(1 for x in g.node_genes.values() if x.type == HIDDEN)
                           for g in pop])
        conns = np.array([sum(c.enabled for c in g.conn_genes.values()) for g in pop])
        depths = np.array([max_depth(g) for g in pop])

        # Activation usage across the whole population.
        act_pop: Counter = Counter()
        for g in pop:
            act_pop.update(_activation_counts(g))

        # Species dynamics.
        sizes = Counter(assignment)
        species_fit = {}
        species_best = {}
        for sid in sizes:
            idxs = [i for i, s in enumerate(assignment) if s == sid]
            species_fit[sid] = {
                "size": len(idxs),
                "mean": float(f[idxs].mean()),
                "max": float(f[idxs].max()),
            }
            best_idx = max(idxs, key=lambda i: f[i])
            species_best[sid] = asdict(pop[best_idx])
        self.species_best_snapshots[gen] = species_best
        cur_species = set(sizes)
        new_species = len(cur_species - self._prev_species)
        extinct = len(self._prev_species - cur_species)
        self._prev_species = cur_species

        # Best genome + stagnation.
        bi = int(np.argmax(f))
        best = pop[bi]
        if f[bi] > self._best_fitness:
            self._best_fitness = float(f[bi])
            self._stagnation = 0
        else:
            self._stagnation += 1

        # Structural search frontier.
        all_innovs = set().union(*[set(g.conn_genes) for g in pop]) if pop else set()
        max_innov = max(all_innovs) if all_innovs else 0

        rec = {
            "gen": gen,
            "fitness": {
                "min": float(f.min()), "mean": float(f.mean()),
                "median": float(np.median(f)), "max": float(f.max()),
                "std": float(f.std()),
            },
            "complexity": {
                "nodes": _dist(nodes), "hidden": _dist(hidden),
                "enabled_conns": _dist(conns), "depth": _dist(depths),
            },
            "activations_population": dict(act_pop),
            "activations_best": dict(_activation_counts(best)),
            "species": {
                "count": len(sizes), "sizes": dict(sizes), "fitness": species_fit,
                "new": new_species, "extinct": extinct, "largest": max(sizes.values()),
            },
            "innovation": {"max": int(max_innov), "distinct": len(all_innovs)},
            "stagnation": self._stagnation,
            "best": {
                "index": bi, "fitness": float(f[bi]), "nodes": int(nodes[bi]),
                "hidden": int(hidden[bi]), "enabled_conns": int(conns[bi]),
                "depth": int(depths[bi]),
            },
        }
        if gen_time is not None:
            rec["gen_time_s"] = float(gen_time)

        self.records.append(rec)
        self.best_snapshots[gen] = asdict(best)
        return rec

    # --- Persistence --------------------------------------------------------

    def dump_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"records": self.records,
                       "best_snapshots": self.best_snapshots,
                       "species_best_snapshots": self.species_best_snapshots}, fh)

    def dump_pickle(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
