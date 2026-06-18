# %% [markdown]
# NEAT on Neural Slime Volleyball — Colab runner.
# Runtime > Change runtime type > GPU.

# %% Install dependencies and confirm a GPU is visible
!pip -q install evojax matplotlib networkx
import jax
print("devices:", jax.devices())          # expect a CudaDevice; if CPU, switch runtime to GPU

# %% Clone the library (SlimeVolley branch) and make it importable
!rm -rf Neuroevolution
!git clone -b SlimeVolley https://github.com/yugoguy/Neuroevolution.git
import sys
sys.path.append("/content/Neuroevolution")

# %% Hyperparameters
#@markdown ### Run
pop_size = 256          #@param {type:"integer"}
num_generations = 1000   #@param {type:"integer"}
n_max = 64              #@param {type:"integer"}
seed = 0                #@param {type:"integer"}
verbose = True          #@param {type:"boolean"}

#@markdown ### Network / forward pass
num_passes = 64                 #@param {type:"integer"}
output_activation = "tanh"      #@param ["tanh","relu","sigmoid","sin"]
new_node_activation = "random"  #@param ["random","tanh","relu","sigmoid","sin"]

#@markdown ### Initialization
weight_init_std = 1.0   #@param {type:"number"}

#@markdown ### Mutation
p_weight = 0.8          #@param {type:"number"}
perturb_std = 0.1       #@param {type:"number"}
replace_prob = 0.001      #@param {type:"number"}
p_add_connection = 0.009  #@param {type:"number"}
p_add_node = 0.009       #@param {type:"number"}
p_activation = 0.005      #@param {type:"number"}
add_conn_max_tries = 20 #@param {type:"integer"}

#@markdown ### Crossover
reenable_prob = 0.025            #@param {type:"number"}
inherit_from_fitter_prob = 0.9  #@param {type:"number"}
interspecies_mating_prob = 0.001 #@param {type:"number"}

#@markdown ### Speciation
compat_threshold = 0.475      #@param {type:"number"}
c_unmatched = 1.0           #@param {type:"number"}
c_weight = 0.4              #@param {type:"number"}
normalize_threshold = 0    #@param {type:"integer"}
rep_selection = "random"    #@param ["random","first"]

#@markdown ### Reproduction
survival_threshold = 0.2        #@param {type:"number"}
elitism_min_species_size = 2    #@param {type:"integer"}
mutate_only_prob = 0.3         #@param {type:"number"}
parent_selection = "uniform"    #@param ["uniform","fitness_weighted"]
max_stagnation = 100             #@param {type:"integer"}
population_stall = 100            #@param {type:"integer"}

#@markdown ### Evaluation
max_steps = 1000            #@param {type:"integer"}
episodes_per_genome = 5     #@param {type:"integer"}
eval_batch_size = 0         #@param {type:"integer"}
eval_test_mode = False      #@param {type:"boolean"}
survival_weight = 0       #@param {type:"number"}
action_threshold = 0.0      #@param {type:"number"}

from config import Config
cfg = Config(
    pop_size=pop_size, num_generations=num_generations, n_max=n_max, seed=seed, verbose=verbose,
    num_passes=num_passes, output_activation=output_activation, new_node_activation=new_node_activation,
    weight_init_std=weight_init_std,
    p_weight=p_weight, perturb_std=perturb_std, replace_prob=replace_prob,
    p_add_connection=p_add_connection, p_add_node=p_add_node, p_activation=p_activation,
    add_conn_max_tries=add_conn_max_tries,
    reenable_prob=reenable_prob, inherit_from_fitter_prob=inherit_from_fitter_prob,
    interspecies_mating_prob=interspecies_mating_prob,
    compat_threshold=compat_threshold, c_unmatched=c_unmatched, c_weight=c_weight,
    normalize_threshold=normalize_threshold, rep_selection=rep_selection,
    survival_threshold=survival_threshold, elitism_min_species_size=elitism_min_species_size,
    mutate_only_prob=mutate_only_prob, parent_selection=parent_selection,
    max_stagnation=max_stagnation, population_stall=population_stall,
    max_steps=max_steps, episodes_per_genome=episodes_per_genome, eval_batch_size=eval_batch_size,
    eval_test_mode=eval_test_mode, survival_weight=survival_weight, action_threshold=action_threshold,
)

# %% Run evolution (first generation is slow due to JIT compilation)
from orchestrator import evolve
best, rec = evolve(cfg)
rec.dump_json("history.json")
print("best fitness:", rec.records[-1]["best"]["fitness"])

# %% Fitness, complexity, species over generations
import matplotlib.pyplot as plt
g = [r["gen"] for r in rec.records]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(g, [r["fitness"]["max"] for r in rec.records], label="max")
ax[0].plot(g, [r["fitness"]["mean"] for r in rec.records], label="mean")
ax[0].set_title("fitness"); ax[0].set_xlabel("generation"); ax[0].legend()
ax[1].plot(g, [r["complexity"]["hidden"]["mean"] for r in rec.records], label="hidden nodes")
ax[1].plot(g, [r["complexity"]["depth"]["mean"] for r in rec.records], label="depth")
ax[1].set_title("complexity (pop mean)"); ax[1].set_xlabel("generation"); ax[1].legend()
ax[2].plot(g, [r["species"]["count"] for r in rec.records])
ax[2].set_title("species count"); ax[2].set_xlabel("generation")
plt.tight_layout(); plt.show()

# %% Activation usage across the population over time
import numpy as np
names = list(cfg.activation_names)
usage = np.array([[r["activations_population"].get(n, 0) for n in names] for r in rec.records])
plt.figure(figsize=(8, 4))
plt.stackplot(g, usage.T, labels=names)
plt.legend(loc="upper left", ncol=4, fontsize=8)
plt.title("activation usage (population)"); plt.xlabel("generation"); plt.show()

# %% Best network topology
from viz import draw_network
from IPython.display import Image
draw_network(best, cfg.num_inputs, cfg.num_outputs, "best_topology.png")
Image("best_topology.png")

# %% Topology at several generations (complexification story)
from recorder import load_genome
for gi in [0, cfg.num_generations // 2, cfg.num_generations - 1]:
    if gi in rec.best_snapshots:
        draw_network(load_genome(rec.best_snapshots[gi]),
                     cfg.num_inputs, cfg.num_outputs, f"topo_gen{gi}.png")

# %% Render a match vs. the built-in opponent as a GIF
from converter import activation_ids
from viz import render_gif
render_gif(best, cfg.num_inputs, cfg.num_outputs, cfg.n_max,
           activation_ids(names), names, cfg.num_passes,
           "play.gif", max_steps=3000, seed=0, fps=30)
Image("play.gif")

# %% Species timeline (lifespans over generations; color = best fitness, size = members)
import matplotlib.pyplot as plt
from matplotlib import cm
records = rec.records

fit_lookup, size_lookup, species_gens = {}, {}, {}
for r in records:
    for sid, st in r["species"]["fitness"].items():
        sid = int(sid)
        fit_lookup[(r["gen"], sid)] = st["max"]
        size_lookup[(r["gen"], sid)] = st["size"]
        species_gens.setdefault(sid, []).append(r["gen"])

sids = sorted(species_gens)
all_fit = list(fit_lookup.values())
vmin, vmax = (min(all_fit), max(all_fit)) if all_fit else (0, 1)

fig, ax = plt.subplots(figsize=(12, 0.35 * len(sids) + 2))
for row, sid in enumerate(sids):
    gens = species_gens[sid]
    ax.plot([min(gens), max(gens)], [row, row], color="0.85", lw=1, zorder=1)
    sc = ax.scatter(gens, [row] * len(gens),
                    c=[fit_lookup[(g, sid)] for g in gens],
                    s=[12 + 3 * size_lookup[(g, sid)] for g in gens],
                    cmap="viridis", vmin=vmin, vmax=vmax, zorder=2)
ax.set_yticks(range(len(sids))); ax.set_yticklabels([f"sp {s}" for s in sids])
ax.set_xlabel("generation"); ax.set_title("species timeline (color = best fitness, size = members)")
fig.colorbar(sc, ax=ax, label="best fitness")
plt.tight_layout(); plt.show()
