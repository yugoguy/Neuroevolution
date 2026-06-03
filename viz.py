"""Results and visualization.

Two deliverables for the report:
  - draw_network: a topology diagram of one genome (inputs left, outputs right,
    hidden placed by depth; node color = activation, edge width = |weight|,
    dashed = disabled).
  - render_gif: an episode of the evolved agent vs. the built-in opponent,
    rendered straight from EvoJAX's own SlimeVolley.render and saved as a GIF.

Both reuse the same JAX forward pass as training, so there is no second network
implementation to keep in sync.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from genome import Genome, io_ids, INPUT, OUTPUT
from graph_utils import node_depths
from converter import express, stack_models, activation_fns, build_forward


# --- Topology diagram -------------------------------------------------------

def draw_network(genome: Genome, num_inputs: int, num_outputs: int, path: str) -> None:
    """Render a genome's topology to `path` (PNG)."""
    import matplotlib.pyplot as plt

    inputs, bias, outputs = io_ids(num_inputs, num_outputs)
    depth = node_depths(genome)
    max_depth = max(depth.values()) if depth else 1

    # x by role/depth, y spread within each column.
    def column(nid):
        g = genome.node_genes[nid]
        if g.type == INPUT:
            return 0
        if g.type == OUTPUT:
            return max_depth + 1
        return max(1, min(max_depth, depth[nid]))

    cols: dict[int, list[int]] = {}
    for nid in genome.node_genes:
        cols.setdefault(column(nid), []).append(nid)

    pos = {}
    for col, nids in cols.items():
        for k, nid in enumerate(sorted(nids)):
            y = (k + 1) / (len(nids) + 1)
            pos[nid] = (col, y)

    # Color per activation.
    acts = sorted({g.activation for g in genome.node_genes.values()})
    cmap = plt.get_cmap("tab10")
    color = {a: cmap(i % 10) for i, a in enumerate(acts)}

    fig, ax = plt.subplots(figsize=(8, 5))
    for c in genome.conn_genes.values():
        if c.in_node not in pos or c.out_node not in pos:
            continue
        x0, y0 = pos[c.in_node]
        x1, y1 = pos[c.out_node]
        ax.plot(
            [x0, x1], [y0, y1],
            color=("tab:red" if c.weight < 0 else "tab:blue"),
            alpha=0.6 if c.enabled else 0.12,
            linestyle="-" if c.enabled else "--",
            linewidth=0.3 + 2.0 * min(abs(c.weight), 3.0) / 3.0,
            zorder=1,
        )
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


# --- Gameplay GIF -----------------------------------------------------------

def render_gif(
    genome: Genome,
    num_inputs: int,
    num_outputs: int,
    n_max: int,
    act_to_id: dict[str, int],
    activation_name_list: list[str],
    num_passes: int,
    path: str,
    max_steps: int = 3000,
    seed: int = 0,
    action_threshold: float = 0.0,
) -> None:
    """Play one test-mode episode vs. the baseline and save it as a GIF."""
    from evojax.task.slimevolley import SlimeVolley

    task = SlimeVolley(max_steps=max_steps, test=True)
    fwd = build_forward(activation_fns(activation_name_list), num_passes)
    model = stack_models([express(genome, num_inputs, num_outputs, n_max, act_to_id)])

    keys = jax.random.split(jax.random.PRNGKey(seed), 1)   # batch of 1
    state = task.reset(keys)

    frames = []
    for _ in range(max_steps):
        out = fwd(model, state.obs)                        # [1, 3]
        action = (out > action_threshold).astype(jnp.float32)
        state, _reward, done = task.step(state, action)
        # SlimeVolley.render wants an unbatched state; pull task 0 out of the batch.
        single = jax.tree_util.tree_map(lambda x: x[0], state)
        frames.append(SlimeVolley.render(single))
        if bool(done[0]):
            break

    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=33, loop=0
    )
