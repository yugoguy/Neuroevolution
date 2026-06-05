"""Configuration.

Every non-trivial choice is a knob here; low-level modules take explicit
arguments and never import this file. The orchestrator unpacks these.

Fixed structural choices (not knobs, because a knob would be inert or wrong):
  - input/bias activation is "identity": inputs are clamped each pass, so the
    label has no effect on computation.
  - output activation is "sigmoid": the loss is binary cross-entropy on it.
  - the forward-pass scheme (its depth `num_passes` IS a knob).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- Problem ---
    num_inputs: int = 2
    num_outputs: int = 1

    # --- Dataset ---
    dataset: str = "spiral"        # "circle" | "xor" | "spiral"
    n_train: int = 250
    n_test: int = 250
    noise: float = 0.5

    # --- Run ---
    pop_size: int = 120
    num_generations: int = 60
    n_max: int = 32                # per-genome node budget == converter tensor width
    seed: int = 0
    verbose: bool = True
    print_top_species: int = 3     # per gen, show this many top species' best structure (0 = off)

    # --- Network / forward pass ---
    # num_passes: synchronous propagation steps. Correctness needs it >= longest
    # path; n_max is always safe. Lower it only if you know depth is bounded.
    num_passes: int = 32
    activation_names: tuple[str, ...] = ("tanh", "relu", "sigmoid", "sin", "gauss", "abs", "square")
    output_activation: str = "sigmoid"     # fixed; BCE is computed on this
    new_node_activation: str = "random"    # "random" => sample activation_names

    # --- Backprop (inner loop) ---
    backprop_steps: int = 200      # full-batch Adam steps per genome per generation
    learning_rate: float = 0.05

    # --- Complexity penalty ---
    penalty_conn: float = 0.01     # fitness -= penalty_conn * sqrt(#enabled conns)
    penalty_node: float = 0.0      # optional: -= penalty_node * sqrt(#nodes)

    # --- Initialization ---
    weight_init_std: float = 0.5

    # --- Mutation (structural ops are NEAT's; weight mutation only jitters warm-start) ---
    p_weight: float = 0.2
    perturb_std: float = 0.1
    replace_prob: float = 0.05
    p_add_connection: float = 0.3
    p_add_node: float = 0.15
    p_activation: float = 0.1
    add_conn_max_tries: int = 20

    # --- Crossover ---
    reenable_prob: float = 0.25
    inherit_from_fitter_prob: float = 0.5
    interspecies_mating_prob: float = 0.001

    # --- Speciation ---
    compat_threshold: float = 1.5
    c_unmatched: float = 1.0
    c_weight: float = 0.4
    normalize_threshold: int = 20
    rep_selection: str = "random"

    # --- Reproduction ---
    survival_threshold: float = 0.3
    elitism_min_species_size: int = 5
    mutate_only_prob: float = 0.25
    parent_selection: str = "uniform"
    max_stagnation: int = 15       # 0 disables
    population_stall: int = 20      # 0 disables
