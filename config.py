"""Configuration.

All hyperparameters and non-trivial design knobs live here, in one frozen
dataclass. The orchestrator unpacks these and passes explicit arguments down to
the low-level modules, which never import this file (keeping them decoupled and
swappable). Defaults target Neural Slime Volleyball.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- Problem (verified from the task; orchestrator reads them off the env) ---
    num_inputs: int = 12
    num_outputs: int = 3

    # --- Population / run ---
    pop_size: int = 256
    num_generations: int = 100
    n_max: int = 64              # per-genome node budget == converter tensor width
    seed: int = 0

    # --- Network / forward pass ---
    # DESIGN NOTE: the converter evaluates each DAG by `num_passes` synchronous
    # propagation steps. Correctness requires num_passes >= longest path in the
    # network; n_max is always safe. Lower it to trade a margin of safety for
    # speed once typical depths are known. 32 comfortably exceeds expected depth
    # for this task while staying well below n_max.
    num_passes: int = 32
    activation_names: tuple[str, ...] = (
        "identity", "tanh", "relu", "sigmoid", "sin", "gauss", "abs",
    )
    output_activation: str = "tanh"

    # --- Initialization ---
    weight_init_std: float = 1.0

    # --- Mutation rates / scales ---
    p_weight: float = 0.9        # probability of running weight perturbation
    perturb_std: float = 0.1     # Gaussian step for a perturbed weight
    replace_prob: float = 0.1    # chance a perturbed weight is replaced outright
    p_add_connection: float = 0.1
    p_add_node: float = 0.05
    p_activation: float = 0.1

    # --- Crossover ---
    reenable_prob: float = 0.25  # chance a disabled inherited gene re-enables

    # --- Speciation ---
    compat_threshold: float = 3.0
    c_unmatched: float = 1.0     # weight on (excess+disjoint)/N
    c_weight: float = 0.4        # weight on mean |Δweight| of matching genes
    normalize_threshold: int = 20

    # --- Reproduction ---
    survival_threshold: float = 0.2     # top fraction kept as potential parents
    elitism_min_species_size: int = 5   # species this big carry their champion
    mutate_only_prob: float = 0.25      # offspring made by mutation alone

    # --- Evaluation ---
    max_steps: int = 3000
    action_threshold: float = 0.0       # action = (output > threshold)
