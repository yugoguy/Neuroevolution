"""Results and visualization.

  - draw_network: topology diagram of one genome (inputs left, outputs right,
    hidden placed by depth; node color = activation, edge color = weight sign,
    width = |weight|, dashed = disabled).
  - decision_boundary: the genome's predicted class probability over a 2D grid,
    with the dataset overlaid. Reuses the same JAX forward pass as training.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from genome import Genome, io_ids, INPUT, OUTPUT
from graph_utils import node_depths
from converter import express, forward_single, activation_fns


# --- Topology diagram -------------------------------------------------------

def draw_network(genome: Genome, num_inputs: int, num_outputs: int, path: str) -> None:
    import matplotlib.pyplot as plt

    inputs, bias, outputs = io_ids(num_inputs, num_outputs)
    depth = node_depths(genome)
    max_d = max(depth.values()) if depth else 1

    def column(nid):
        g = genome.node_genes[nid]
        if g.type == INPUT:
            return 0
        if g.type == OUTPUT:
            return max_d + 1
        return max(1, min(max_d, depth[nid]))

    cols: dict[int, list[int]] = {}
    for nid in genome.node_genes:
        cols.setdefault(column(nid), []).append(nid)

    pos = {}
    for col, nids in cols.items():
        for k, nid in enumerate(sorted(nids)):
            pos[nid] = (col, (k + 1) / (len(nids) + 1))

    acts = sorted({g.activation for g in genome.node_genes.values()})
    cmap = plt.get_cmap("tab10")
    color = {a: cmap(i % 10) for i, a in enumerate(acts)}

    fig, ax = plt.subplots(figsize=(8, 5))
    for c in genome.conn_genes.values():
        if c.in_node not in pos or c.out_node not in pos:
            continue
        x0, y0 = pos[c.in_node]
        x1, y1 = pos[c.out_node]
        ax.plot([x0, x1], [y0, y1],
                color=("tab:red" if c.weight < 0 else "tab:blue"),
                alpha=0.6 if c.enabled else 0.12,
                linestyle="-" if c.enabled else "--",
                linewidth=0.3 + 2.0 * min(abs(c.weight), 3.0) / 3.0, zorder=1)
    for nid, (x, y) in pos.items():
        g = genome.node_genes[nid]
        ax.scatter([x], [y], s=240, color=color[g.activation],
                   edgecolors="black", zorder=2)
        ax.annotate(g.activation, (x, y), fontsize=6, ha="center", va="center")

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color[a],
                          markeredgecolor="black", label=a) for a in acts]
    ax.legend(handles=handles, fontsize=7, loc="upper center",
              ncol=len(acts), bbox_to_anchor=(0.5, 1.12))
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --- Decision boundary ------------------------------------------------------

def decision_boundary(
    genome: Genome,
    num_inputs: int,
    num_outputs: int,
    n_max: int,
    act_to_id: dict[str, int],
    activation_name_list: list[str],
    num_passes: int,
    X: np.ndarray,
    y: np.ndarray,
    path: str,
    lim: float = 6.0,
    res: int = 200,
) -> None:
    """Plot predicted probability over a grid with the dataset overlaid."""
    import matplotlib.pyplot as plt

    W, static = express(genome, num_inputs, num_outputs, n_max, act_to_id)
    act_fns = activation_fns(activation_name_list)

    gx, gy = np.meshgrid(np.linspace(-lim, lim, res), np.linspace(-lim, lim, res))
    grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    out = forward_single(jnp.asarray(W), static, jnp.asarray(grid), act_fns, num_passes)
    prob = np.asarray(out[:, 0]).reshape(res, res)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(gx, gy, prob, levels=20, cmap="RdBu", alpha=0.7, vmin=0, vmax=1)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=8, c="tab:red", edgecolors="none")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=8, c="tab:blue", edgecolors="none")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
