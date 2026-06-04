"""Configuration.

Every non-trivial choice is a knob here; the low-level modules take explicit
arguments and never import this file. The orchestrator unpacks these and passes
them down. Comments mark values that were previously hardcoded defaults.

Structural (intentionally NOT knobs, because a knob would be inert or wrong):
  - input/bias node activation is "identity": inputs are clamped each forward
    pass and never transformed, so the label has no effect on computation.
  - weight perturbation is additive Gaussian (its scale `perturb_std` IS a knob).
  - the forward-pass evaluation scheme (its depth `num_passes` IS a knob).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- Problem (orchestrator overrides these from the env) ---
    num_inputs: int = 12
    num_outputs: int = 3

    # --- Run ---
    pop_size: int = 256
    num_generations: int = 100
    n_max: int = 64                 # per-genome node budget == converter tensor width
    seed: int = 0
    verbose: bool = True            # print one progress line per generation

    # --- Network / forward pass ---
    # num_passes: synchronous propagation steps. Correctness needs it >= longest
    # path; n_max is always safe. 32 exceeds expected depth here, well below n_max.
    num_passes: int = 32
    activation_names: tuple[str, ...] = (
        "identity", "tanh", "relu", "sigmoid", "sin", "gauss", "abs",
    )
    output_activation: str = "tanh"
    new_node_activation: str = "random"   # "random" => sample from activation_names; else a name

    # --- Initialization ---
    weight_init_std: float = 1.0

    # --- Mutation rates / scales ---
    p_weight: float = 0.8                  # prob a genome's weights are mutated at all (paper: 0.8)
    perturb_std: float = 0.1
    replace_prob: float = 0.1
    p_add_connection: float = 0.1
    p_add_node: float = 0.05
    p_activation: float = 0.1
    add_conn_max_tries: int = 20          # attempts to find a valid acyclic edge

    # --- Crossover ---
    reenable_prob: float = 0.25
    inherit_from_fitter_prob: float = 0.5  # matching gene: prob of taking the fitter parent's (0.5 = unbiased NEAT)
    interspecies_mating_prob: float = 0.001  # prob a 2nd parent is drawn from any species (paper: 0.001)

    # --- Speciation ---
    compat_threshold: float = 3.0
    c_unmatched: float = 1.0
    c_weight: float = 0.4
    normalize_threshold: int = 20
    rep_selection: str = "random"          # "random" | "first"

    # --- Reproduction ---
    survival_threshold: float = 0.2
    elitism_min_species_size: int = 5
    mutate_only_prob: float = 0.25
    parent_selection: str = "uniform"      # "uniform" | "fitness_weighted"
    max_stagnation: int = 15               # species stuck this many gens can't reproduce (0 = off; paper: 15)
    population_stall: int = 20              # whole-pop stuck this many gens => only top-2 reproduce (0 = off; paper: 20)

    # --- Evaluation ---
    max_steps: int = 1000                  # episode length (lower => faster generations)
    episodes_per_genome: int = 1           # rollouts averaged per genome (>1 => less noisy, N x slower)
    eval_batch_size: int = 0               # rollouts (genome x episode) run at once; 0 = all in one batch
    eval_test_mode: bool = False           # False: full-horizon score margin; True: real 5-life match
    survival_weight: float = 0.0           # + w * (steps_survived / max_steps); only meaningful in test mode
    action_threshold: float = 0.0          # action = (output > threshold)
