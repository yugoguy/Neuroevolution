"""Global innovation registry.

Hands out unique innovation numbers for new connections and unique ids for new
nodes. Within a single generation, the *same* structural mutation must receive
the *same* number, so that genomes which independently discover the same edge
stay aligned for crossover. Caches are therefore reset once per generation.

Starting counters are derived from the canonical layout in `genome.py`, so the
registry never has to inspect a genome to stay consistent with how the initial
population was numbered.
"""

from __future__ import annotations

from genome import reserved_node_count, reserved_conn_count


class InnovationRegistry:
    def __init__(self, num_inputs: int, num_outputs: int):
        # First free ids/innovations sit just past the reserved minimal genome.
        self._next_node_id = reserved_node_count(num_inputs, num_outputs)
        self._next_innovation = reserved_conn_count(num_inputs, num_outputs)
        self._conn_cache: dict[tuple[int, int], int] = {}
        self._node_cache: dict[int, tuple[int, int, int]] = {}

    def new_generation(self) -> None:
        """Clear per-generation caches. Counters keep growing across the run."""
        self._conn_cache.clear()
        self._node_cache.clear()

    def connection(self, in_node: int, out_node: int) -> int:
        """Innovation number for an (in_node -> out_node) edge.

        Identical edges added anywhere this generation share one number.
        """
        key = (in_node, out_node)
        innov = self._conn_cache.get(key)
        if innov is None:
            innov = self._next_innovation
            self._next_innovation += 1
            self._conn_cache[key] = innov
        return innov

    def node_split(self, split_innovation: int) -> tuple[int, int, int]:
        """Allocate a node + two innovations for splitting a given connection.

        Returns (new_node_id, innov_in, innov_out) where innov_in numbers the
        in_node -> new_node edge and innov_out the new_node -> out_node edge.
        Splitting the same connection again this generation reuses all three.
        """
        cached = self._node_cache.get(split_innovation)
        if cached is None:
            node_id = self._next_node_id
            innov_in = self._next_innovation
            innov_out = self._next_innovation + 1
            self._next_node_id += 1
            self._next_innovation += 2
            cached = (node_id, innov_in, innov_out)
            self._node_cache[split_innovation] = cached
        return cached
