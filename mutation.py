"""Mutation operators.

Structural operators (add-connection, add-node) are exactly NEAT's. Weight
mutation is retained but minor: backprop owns the weights, so weight mutation
only jitters the warm-start an offspring inherits. All rates/scales are arguments.

Feed-forward invariant: a connection is added only if it keeps the enabled graph
acyclic (reachability check).
"""

from __future__ import annotations

import numpy as np

from genome import Genome, NodeGene, ConnGene, HIDDEN, OUTPUT
from innovation import InnovationRegistry


def _reaches(genome: Genome, src: int, dst: int) -> bool:
    """True if `dst` is reachable from `src` along enabled connections."""
    adj: dict[int, list[int]] = {}
    for c in genome.conn_genes.values():
        if c.enabled:
            adj.setdefault(c.in_node, []).append(c.out_node)
    stack = [src]
    seen = {src}
    while stack:
        node = stack.pop()
        if node == dst:
            return True
        for nxt in adj.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def mutate_weights(
    genome: Genome,
    rng: np.random.Generator,
    perturb_std: float,
    replace_prob: float,
    weight_init_std: float,
) -> None:
    """Perturb each connection weight, occasionally replacing it outright."""
    for c in genome.conn_genes.values():
        if rng.random() < replace_prob:
            c.weight = float(rng.normal(0.0, weight_init_std))
        else:
            c.weight += float(rng.normal(0.0, perturb_std))


def mutate_add_connection(
    genome: Genome,
    rng: np.random.Generator,
    registry: InnovationRegistry,
    weight_init_std: float,
    max_tries: int,
) -> None:
    """Add one feed-forward connection between two currently unlinked nodes."""
    sources = [n for n, g in genome.node_genes.items() if g.type != OUTPUT]
    targets = [n for n, g in genome.node_genes.items() if g.type in (HIDDEN, OUTPUT)]
    if not targets or not sources:
        return
    existing = {(c.in_node, c.out_node) for c in genome.conn_genes.values()}

    for _ in range(max_tries):
        a = int(rng.choice(sources))
        b = int(rng.choice(targets))
        if a == b or (a, b) in existing:
            continue
        if _reaches(genome, b, a):           # adding a->b would close a cycle
            continue
        innov = registry.connection(a, b)
        genome.conn_genes[innov] = ConnGene(
            innov, a, b, float(rng.normal(0.0, weight_init_std)), True
        )
        return


def mutate_add_node(
    genome: Genome,
    rng: np.random.Generator,
    registry: InnovationRegistry,
    n_max: int,
    new_activation: str,
    activation_names: list[str],
) -> None:
    """Split a random enabled connection, inserting a hidden node.

    in->new gets weight 1, new->out inherits the old weight, so the function is
    unchanged at the moment of mutation. Skipped if the genome already holds
    `n_max` nodes. New node activation is `new_activation`, or a random pick from
    `activation_names` when that is "random".
    """
    if len(genome.node_genes) >= n_max:
        return
    enabled = [c for c in genome.conn_genes.values() if c.enabled]
    if not enabled:
        return

    act = str(rng.choice(activation_names)) if new_activation == "random" else new_activation
    c = enabled[int(rng.integers(len(enabled)))]
    node_id, innov_in, innov_out = registry.node_split(c.innovation)

    c.enabled = False
    genome.node_genes[node_id] = NodeGene(node_id, HIDDEN, act)
    genome.conn_genes[innov_in] = ConnGene(innov_in, c.in_node, node_id, 1.0, True)
    genome.conn_genes[innov_out] = ConnGene(innov_out, node_id, c.out_node, c.weight, True)


def mutate_activation(
    genome: Genome,
    rng: np.random.Generator,
    activation_names: list[str],
) -> None:
    """Reassign the activation of a random hidden node (outputs stay sigmoid)."""
    candidates = [n for n, g in genome.node_genes.items() if g.type == HIDDEN]
    if not candidates or not activation_names:
        return
    nid = int(rng.choice(candidates))
    genome.node_genes[nid].activation = str(rng.choice(activation_names))


def mutate(
    genome: Genome,
    rng: np.random.Generator,
    registry: InnovationRegistry,
    *,
    n_max: int,
    weight_init_std: float,
    p_weight: float,
    perturb_std: float,
    replace_prob: float,
    p_add_connection: float,
    p_add_node: float,
    p_activation: float,
    activation_names: list[str],
    add_conn_max_tries: int,
    new_node_activation: str,
) -> Genome:
    """Return a mutated copy of `genome`, applying each operator by probability."""
    g = genome.copy()
    if rng.random() < p_weight:
        mutate_weights(g, rng, perturb_std, replace_prob, weight_init_std)
    if rng.random() < p_add_connection:
        mutate_add_connection(g, rng, registry, weight_init_std, add_conn_max_tries)
    if rng.random() < p_add_node:
        mutate_add_node(g, rng, registry, n_max, new_node_activation, activation_names)
    if rng.random() < p_activation:
        mutate_activation(g, rng, activation_names)
    return g
