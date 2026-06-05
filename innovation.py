"""Global innovation registry.

Hands out unique innovation numbers for new connections and unique ids for new
nodes. Within one generation the *same* structural mutation must receive the
*same* number, so genomes that independently discover the same edge stay aligned
for crossover. Caches reset once per generation; counters keep growing.
"""

from __future__ import annotations

from genome import reserved_node_count, reserved_conn_count


class InnovationRegistry:
    def __init__(self, num_inputs: int, num_outputs: int):
        self._next_node_id = reserved_node_count(num_inputs, num_outputs)
        self._next_innovation = reserved_conn_count(num_inputs, num_outputs)
        self._conn_cache: dict[tuple[int, int], int] = {}
        self._node_cache: dict[int, tuple[int, int, int]] = {}

    def new_generation(self) -> None:
        self._conn_cache.clear()
        self._node_cache.clear()

    def connection(self, in_node: int, out_node: int) -> int:
        key = (in_node, out_node)
        innov = self._conn_cache.get(key)
        if innov is None:
            innov = self._next_innovation
            self._next_innovation += 1
            self._conn_cache[key] = innov
        return innov

    def node_split(self, split_innovation: int) -> tuple[int, int, int]:
        """Allocate (new_node_id, innov_in, innov_out) for splitting a connection.

        innov_in numbers the in_node -> new_node edge, innov_out the
        new_node -> out_node edge. Re-splitting the same connection this
        generation reuses all three.
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
