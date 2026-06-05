# %% [markdown]
# Backprop NEAT on 2D classification (circle / XOR / spiral) — Colab runner.
# NEAT evolves topology; backprop (Adam) fits the weights of each network.
# Runtime > Change runtime type > GPU is optional (small problems run on CPU too).

# %% Confirm JAX sees a device.
# Colab ships JAX preinstalled and version-matched on a FRESH runtime — do not
# reinstall/upgrade jax (that causes the PJRT "expected 48, got 40" mismatch).
# If you ever hit that error, just Runtime > Disconnect and delete runtime, then
# run from the top.
import jax
print("devices:", jax.devices())   # CudaDevice on GPU runtime, else CPU

# %% Clone the library (BackpropNEAT branch) and make it importable
!rm -rf Neuroevolution
!git clone -b BackpropNEAT https://github.com/yugoguy/Neuroevolution.git
import sys
sys.path.append("/content/Neuroevolution")

# %% Hyperparameters
#@markdown ### Task
dataset = "xor"        #@param ["circle", "xor", "spiral"]
n_train = 250             #@param {type:"integer"}
n_test = 250              #@param {type:"integer"}
noise = 0.1               #@param {type:"number"}

#@markdown ### Run
pop_size = 120            #@param {type:"integer"}
num_generations = 20      #@param {type:"integer"}
n_max = 32                #@param {type:"integer"}
seed = 0                  #@param {type:"integer"}

#@markdown ### Backprop inner loop
backprop_steps = 100      #@param {type:"integer"}
learning_rate = 0.01      #@param {type:"number"}
num_passes = 32           #@param {type:"integer"}

#@markdown ### Complexity penalty
penalty_conn = 0.01       #@param {type:"number"}
penalty_node = 0.0        #@param {type:"number"}

#@markdown ### Initialization
weight_init_std = 0.5     #@param {type:"number"}

#@markdown ### Mutation (structural ops are NEAT's; weight mutation only jitters the backprop warm-start)
p_weight = 0.2            #@param {type:"number"}
perturb_std = 0.1         #@param {type:"number"}
replace_prob = 0.05       #@param {type:"number"}
p_add_connection = 0.3    #@param {type:"number"}
p_add_node = 0.15         #@param {type:"number"}
p_activation = 0.1        #@param {type:"number"}
add_conn_max_tries = 20   #@param {type:"integer"}
new_node_activation = "random"  #@param ["random", "tanh", "relu", "sigmoid", "sin", "gauss", "abs", "square"]

#@markdown ### Crossover
reenable_prob = 0.25             #@param {type:"number"}
inherit_from_fitter_prob = 0.5   #@param {type:"number"}
interspecies_mating_prob = 0.001 #@param {type:"number"}

#@markdown ### Speciation
compat_threshold = 0.5    #@param {type:"number"}
c_unmatched = 1.0         #@param {type:"number"}
c_weight = 0.4            #@param {type:"number"}
normalize_threshold = 0  #@param {type:"integer"}
rep_selection = "random"  #@param ["random", "first"]

#@markdown ### Reproduction
survival_threshold = 0.3        #@param {type:"number"}
elitism_min_species_size = 5    #@param {type:"integer"}
mutate_only_prob = 0.25         #@param {type:"number"}
parent_selection = "uniform"    #@param ["uniform", "fitness_weighted"]
max_stagnation = 15             #@param {type:"integer"}
population_stall = 20           #@param {type:"integer"}

from config import Config
cfg = Config(
    dataset=dataset, n_train=n_train, n_test=n_test, noise=noise,
    pop_size=pop_size, num_generations=num_generations, n_max=n_max, seed=seed,
    backprop_steps=backprop_steps, learning_rate=learning_rate, num_passes=num_passes,
    penalty_conn=penalty_conn, penalty_node=penalty_node,
    weight_init_std=weight_init_std,
    p_weight=p_weight, perturb_std=perturb_std, replace_prob=replace_prob,
    p_add_connection=p_add_connection, p_add_node=p_add_node, p_activation=p_activation,
    add_conn_max_tries=add_conn_max_tries, new_node_activation=new_node_activation,
    reenable_prob=reenable_prob, inherit_from_fitter_prob=inherit_from_fitter_prob,
    interspecies_mating_prob=interspecies_mating_prob,
    compat_threshold=compat_threshold, c_unmatched=c_unmatched, c_weight=c_weight,
    normalize_threshold=normalize_threshold, rep_selection=rep_selection,
    survival_threshold=survival_threshold, elitism_min_species_size=elitism_min_species_size,
    mutate_only_prob=mutate_only_prob, parent_selection=parent_selection,
    max_stagnation=max_stagnation, population_stall=population_stall,
)

# %% Visualize the dataset first (same RNG sequence evolve uses, so this is the
# actual train/test split the networks are scored on)
import numpy as np
import matplotlib.pyplot as plt
from dataset import make_dataset
_drng = np.random.default_rng(cfg.seed + 1)
Xtr, ytr = make_dataset(cfg.dataset, cfg.n_train, _drng, cfg.noise)
Xte, yte = make_dataset(cfg.dataset, cfg.n_test, _drng, cfg.noise)
fig, axd = plt.subplots(1, 2, figsize=(10, 5))
for a, (X, y, name) in zip(axd, [(Xtr, ytr, "train"), (Xte, yte, "test")]):
    a.scatter(X[y == 0, 0], X[y == 0, 1], s=10, c="tab:red", edgecolors="none")
    a.scatter(X[y == 1, 0], X[y == 1, 1], s=10, c="tab:blue", edgecolors="none")
    a.set_title(f"{cfg.dataset} — {name} (n={len(y)}, noise={cfg.noise})")
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()

# %% Run evolution (first generation is slow due to JIT compilation)
from orchestrator import evolve
best, rec = evolve(cfg)
rec.dump_json("history.json")
print("best fitness:", rec.records[-1]["best"]["fitness"])
print("best test acc:", rec.records[-1].get("accuracy", {}).get("test_best"))

# %% Fitness / accuracy / complexity / species over generations
import matplotlib.pyplot as plt
g = [r["gen"] for r in rec.records]
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].plot(g, [r["fitness"]["max"] for r in rec.records], label="max")
ax[0, 0].plot(g, [r["fitness"]["mean"] for r in rec.records], label="mean")
ax[0, 0].set_title("fitness"); ax[0, 0].legend()
ax[0, 1].plot(g, [r.get("accuracy", {}).get("train_best", 0) for r in rec.records], label="train best")
ax[0, 1].plot(g, [r.get("accuracy", {}).get("test_best", 0) for r in rec.records], label="test best")
ax[0, 1].set_title("best-genome accuracy"); ax[0, 1].legend()
ax[1, 0].plot(g, [r["complexity"]["hidden"]["mean"] for r in rec.records], label="hidden")
ax[1, 0].plot(g, [r["complexity"]["enabled_conns"]["mean"] for r in rec.records], label="conns")
ax[1, 0].set_title("complexity (pop mean)"); ax[1, 0].legend()
ax[1, 1].plot(g, [r["species"]["count"] for r in rec.records])
ax[1, 1].set_title("species count")
for a in ax.ravel():
    a.set_xlabel("generation")
plt.tight_layout(); plt.show()

# %% Activation usage across the population over time
import numpy as np
names = list(cfg.activation_names)
usage = np.array([[r["activations_population"].get(n, 0) for n in names] for r in rec.records])
plt.figure(figsize=(8, 4))
plt.stackplot(g, usage.T, labels=names)
plt.legend(loc="upper left", ncol=4, fontsize=8)
plt.title("activation usage (population hidden nodes)"); plt.xlabel("generation"); plt.show()

# %% Best network topology
from viz import draw_network
from IPython.display import Image
draw_network(best, cfg.num_inputs, cfg.num_outputs, "best_topology.png")
Image("best_topology.png")

# %% Decision boundary of the best network (re-generate the same test set to overlay)
from dataset import make_dataset
from converter import activation_ids
from viz import decision_boundary
data_rng = np.random.default_rng(cfg.seed + 1)
_ = make_dataset(cfg.dataset, cfg.n_train, data_rng, cfg.noise)   # advance rng to match training
Xte, yte = make_dataset(cfg.dataset, cfg.n_test, data_rng, cfg.noise)
decision_boundary(best, cfg.num_inputs, cfg.num_outputs, cfg.n_max,
                  activation_ids(names), names, cfg.num_passes,
                  Xte, yte, "best_boundary.png")
Image("best_boundary.png")

# %% Topology at several generations (complexification story)
from recorder import load_genome
for gi in [0, cfg.num_generations // 2, cfg.num_generations - 1]:
    if gi in rec.best_snapshots:
        draw_network(load_genome(rec.best_snapshots[gi]),
                     cfg.num_inputs, cfg.num_outputs, f"topo_gen{gi}.png")
