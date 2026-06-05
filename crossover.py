"""Crossover.

Recombines two parents aligned by innovation number:
  - matching genes: inherited from a random parent;
  - disjoint/excess genes: from the fitter parent (from both on a fitness tie).
A gene disabled in either parent stays disabled unless a re-enable roll succeeds.
Inherited weights act as the backprop warm-start for the child.
"""

from __future__ import annotations

import numpy as np
from dataclasses import replace

from genome import Genome, ConnGene


def crossover(
    parent1: Genome,
    parent2: Genome,
    fitness1: float,
    fitness2: float,
    rng: np.random.Generator,
    reenable_prob: float,
    inherit_from_fitter_prob: float = 0.5,
) -> Genome:
    fitter, other = (parent1, parent2) if fitness1 >= fitness2 else (parent2, parent1)
    tie = fitness1 == fitness2

    innovations = set(fitter.conn_genes)
    if tie:
        innovations |= set(other.conn_genes)

    child_conns: dict[int, ConnGene] = {}
    for innov in innovations:
        g_fit = fitter.conn_genes.get(innov)
        g_oth = other.conn_genes.get(innov)

        if g_fit is not None and g_oth is not None:           # matching
            chosen = replace(g_fit if rng.random() < inherit_from_fitter_prob else g_oth)
            disabled_in_either = (not g_fit.enabled) or (not g_oth.enabled)
        elif g_fit is not None:                                # disjoint/excess of fitter
            chosen = replace(g_fit)
            disabled_in_either = not g_fit.enabled
        else:                                                  # tie: only in other parent
            chosen = replace(g_oth)
            disabled_in_either = not g_oth.enabled

        if disabled_in_either:
            chosen.enabled = rng.random() < reenable_prob
        child_conns[innov] = chosen

    node_ids = set()
    for c in child_conns.values():
        node_ids.add(c.in_node)
        node_ids.add(c.out_node)
    node_ids |= set(fitter.node_genes)   # guarantees inputs/bias/outputs

    child_nodes = {}
    for nid in node_ids:
        src = fitter.node_genes.get(nid) or other.node_genes.get(nid)
        child_nodes[nid] = replace(src)

    return Genome(node_genes=child_nodes, conn_genes=child_conns)
