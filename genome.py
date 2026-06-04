"""NEAT genome: the shared data contract.

A genome describes one individual as two dicts:
  - node_genes: node id -> NodeGene
  - conn_genes: innovation number -> ConnGene

All other modules (mutation, crossover, speciation, converter) read and write
this structure and nothing else. Evolutionary logic lives in those modules,
not here.

Canonical id / innovation layout
--------------------------------
Reserved nodes occupy a fixed, predictable id range so that the innovation
registry, the population initializer, and the genome->model converter all agree
without talking to each other. For S sensor inputs and O outputs:

    ids 0 .. S-1      sensor inputs   (fed the observation vector)
    id  S             bias node       (an INPUT node, always emits 1.0)
    ids S+1 .. S+O    outputs
    ids S+O+1 ..      hidden          (assigned by the registry as they appear)

The bias node is an ordinary INPUT node; it is identified purely by its id
(== S), so a connection weight out of it acts as the target node's bias. No
separate gene field or node type is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


INPUT = "input"
HIDDEN = "hidden"
OUTPUT = "output"


@dataclass
class NodeGene:
    id: int
    type: str               # INPUT | HIDDEN | OUTPUT
    activation: str         # label, e.g. "tanh"; mapped to a fn at convert time


@dataclass
class ConnGene:
    innovation: int         # global structural id; aligns genomes for crossover
    in_node: int
    out_node: int
    weight: float
    enabled: bool = True


@dataclass
class Genome:
    node_genes: dict[int, NodeGene] = field(default_factory=dict)
    conn_genes: dict[int, ConnGene] = field(default_factory=dict)

    def copy(self) -> "Genome":
        """New dicts holding fresh gene objects.

        Offspring are produced by copying a parent and editing the copy, so the
        parent's genes must never be shared by reference.
        """
        return Genome(
            node_genes={i: replace(n) for i, n in self.node_genes.items()},
            conn_genes={k: replace(c) for k, c in self.conn_genes.items()},
        )


# --- Canonical layout (single source of truth) -----------------------------

def io_ids(num_inputs: int, num_outputs: int) -> tuple[list[int], int, list[int]]:
    """Return (sensor_input_ids, bias_id, output_ids) under the canonical layout."""
    inputs = list(range(num_inputs))
    bias = num_inputs
    outputs = list(range(num_inputs + 1, num_inputs + 1 + num_outputs))
    return inputs, bias, outputs


def reserved_node_count(num_inputs: int, num_outputs: int) -> int:
    """Number of fixed nodes: sensors + bias + outputs."""
    return num_inputs + 1 + num_outputs


def reserved_conn_count(num_inputs: int, num_outputs: int) -> int:
    """Connections in the minimal genome: every input (incl. bias) -> every output."""
    return (num_inputs + 1) * num_outputs


def initial_genome(
    num_inputs: int,
    num_outputs: int,
    output_activation: str,
    weights,
) -> Genome:
    """Build the minimal fully-connected genome (no hidden nodes).

    Every input and the bias node connect to every output. Innovation numbers
    are assigned deterministically as `in_id * num_outputs + output_index`, so
    the layout matches `reserved_conn_count` and the registry's starting counter.

    `weights` is a sequence of length `reserved_conn_count(...)`, indexed by
    innovation number; the caller (init_population) supplies per-individual values.
    """
    inputs, bias, outputs = io_ids(num_inputs, num_outputs)

    nodes: dict[int, NodeGene] = {}
    for i in inputs:
        nodes[i] = NodeGene(i, INPUT, "identity")
    nodes[bias] = NodeGene(bias, INPUT, "identity")
    for o in outputs:
        nodes[o] = NodeGene(o, OUTPUT, output_activation)

    conns: dict[int, ConnGene] = {}
    for in_id in inputs + [bias]:
        for k, out_id in enumerate(outputs):
            innov = in_id * num_outputs + k
            conns[innov] = ConnGene(innov, in_id, out_id, float(weights[innov]), True)

    return Genome(node_genes=nodes, conn_genes=conns)
