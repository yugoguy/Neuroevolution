"""Orchestrator: the NEAT evolution loop.

Assembles every module into one run. This is the only assembly-tier file besides
reproduction: it owns the config, generation boundaries, RNG, and logging, and
passes explicit arguments / pre-bound callables to the decoupled modules.

Per generation:
  express + stack -> evaluate (GPU rollout) -> log/track best
  -> speciate -> advance innovation generation -> reproduce.
"""

from __future__ import annotations

import time

import numpy as np
import jax

from config import Config
from innovation import InnovationRegistry
from init_population import init_population
from mutation import mutate
from crossover import crossover
from speciation import speciate
from reproduction import reproduce
from stagnation import StagnationTracker
from recorder import Recorder
from converter import express, stack_models, activation_ids, activation_fns, build_forward
from fitness import evaluate
from env import make_task, task_dims


def evolve(config: Config, callback=None):
    """Run evolution. Returns (best_genome, recorder).

    Progress: if `callback` is given it is called with each generation's record
    dict; otherwise, when config.verbose is set, a one-line summary is printed.
    """
    rng = np.random.default_rng(config.seed)
    key = jax.random.PRNGKey(config.seed)

    # Environment first, so dimensions come from the task itself.
    task = make_task(max_steps=config.max_steps, test=config.eval_test_mode)
    num_inputs, num_outputs = task_dims(task)

    # Activation table shared by converter and mutation.
    names = list(config.activation_names)
    act_to_id = activation_ids(names)
    act_fns = activation_fns(names)
    forward_fn = build_forward(act_fns, config.num_passes)

    registry = InnovationRegistry(num_inputs, num_outputs)
    pop = init_population(
        config.pop_size, num_inputs, num_outputs, rng,
        weight_init_std=config.weight_init_std,
        output_activation=config.output_activation,
    )

    representatives: dict[int, object] = {}
    best_genome, best_fitness = None, -np.inf
    recorder = Recorder()
    stagnation = StagnationTracker(config.max_stagnation, config.population_stall)

    for gen in range(config.num_generations):
        t0 = time.time()

        # --- Evaluate ---
        models = stack_models(
            [express(g, num_inputs, num_outputs, config.n_max, act_to_id) for g in pop]
        )
        key, sub = jax.random.split(key)
        fitnesses = np.asarray(
            evaluate(
                models, task, forward_fn, sub,
                episodes=config.episodes_per_genome,
                action_threshold=config.action_threshold,
                survival_weight=config.survival_weight,
                batch_size=config.eval_batch_size,
            )
        )

        gen_best = int(np.argmax(fitnesses))
        if fitnesses[gen_best] > best_fitness:
            best_fitness = float(fitnesses[gen_best])
            best_genome = pop[gen_best].copy()

        # --- Speciate ---
        assignment, representatives = speciate(
            pop, representatives, rng,
            threshold=config.compat_threshold,
            c_unmatched=config.c_unmatched,
            c_weight=config.c_weight,
            normalize_threshold=config.normalize_threshold,
            rep_selection=config.rep_selection,
        )

        rec = recorder.record(gen, pop, fitnesses, assignment, gen_time=time.time() - t0)
        if callback is not None:
            callback(rec)
        elif config.verbose:
            print(
                f"gen {rec['gen']:3d} | best {rec['fitness']['max']:6.2f} "
                f"mean {rec['fitness']['mean']:6.2f} | species {rec['species']['count']:2d} "
                f"| hidden {rec['complexity']['hidden']['mean']:4.1f} "
                f"depth {rec['complexity']['depth']['mean']:4.1f} "
                f"| {rec['gen_time_s']:5.1f}s"
            )

        # --- Reproduce (new structural mutations belong to the next generation) ---
        allowed = stagnation.update(gen, assignment, fitnesses)
        registry.new_generation()
        crossover_fn = lambda p1, p2, f1, f2: crossover(
            p1, p2, f1, f2, rng, config.reenable_prob, config.inherit_from_fitter_prob
        )
        mutate_fn = lambda g: mutate(
            g, rng, registry,
            n_max=config.n_max,
            weight_init_std=config.weight_init_std,
            p_weight=config.p_weight,
            perturb_std=config.perturb_std,
            replace_prob=config.replace_prob,
            p_add_connection=config.p_add_connection,
            p_add_node=config.p_add_node,
            p_activation=config.p_activation,
            activation_names=names,
            add_conn_max_tries=config.add_conn_max_tries,
            new_node_activation=config.new_node_activation,
        )
        pop = reproduce(
            pop, fitnesses, assignment, rng,
            pop_size=config.pop_size,
            survival_threshold=config.survival_threshold,
            elitism_min_species_size=config.elitism_min_species_size,
            mutate_only_prob=config.mutate_only_prob,
            parent_selection=config.parent_selection,
            interspecies_mating_prob=config.interspecies_mating_prob,
            crossover_fn=crossover_fn,
            mutate_fn=mutate_fn,
            allowed_species=allowed,
        )

    return best_genome, recorder
