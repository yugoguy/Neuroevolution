"""Genome graph utilities.

Pure structural queries over a genome's enabled connection graph, shared by the
recorder (depth statistics) and viz (layered layout). No evolutionary logic.
"""

from __future__ import annotations

from genome import Genome


def node_depths(genome: Genome) -> dict[int, int]:
    """Longest-path depth of each node from the source nodes, over enabled edges.

    Inputs/bias sit at depth 0; depth is the length of the longest enabled path
    reaching a node. Relies on the feed-forward (acyclic) invariant, so a
    Kahn-style topological sweep terminates.
    """
    adj: dict[int, list[int]] = {}
    indeg: dict[int, int] = {n: 0 for n in genome.node_genes}
    for c in genome.conn_genes.values():
        if c.enabled:
            adj.setdefault(c.in_node, []).append(c.out_node)
            indeg[c.out_node] = indeg.get(c.out_node, 0) + 1

    depth = {n: 0 for n in genome.node_genes}
    queue = [n for n in genome.node_genes if indeg.get(n, 0) == 0]
    while queue:
        n = queue.pop()
        for m in adj.get(n, ()):
            depth[m] = max(depth[m], depth[n] + 1)
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    return depth


def max_depth(genome: Genome) -> int:
    d = node_depths(genome)
    return max(d.values()) if d else 0
