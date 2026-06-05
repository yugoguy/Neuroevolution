"""Orchestrator: the Backprop NEAT evolution loop.

Per generation:
  express + stack -> train weights (backprop) + score -> write trained weights
  back into genomes (Lamarckian) -> log/track best -> speciate -> advance
  innovation generation -> reproduce.

The GA evolves topology only; backprop fits the weights inside each evaluation.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

from config import Config
from innovation import InnovationRegistry
from init_population import init_population
from mutation import mutate
from crossover import crossover
from speciation import speciate
from reproduction import reproduce
from stagnation import StagnationTracker
from recorder import Recorder
from converter import express, stack_models, write_back, activation_ids, activation_fns
from train import train_and_score
from dataset import make_dataset
from graph_utils import max_depth
from genome import HIDDEN


def evolve(config: Config, callback=None):
    """Run evolution. Returns (best_genome, recorder)."""
    rng = np.random.default_rng(config.seed)
    data_rng = np.random.default_rng(config.seed + 1)

    num_inputs, num_outputs = config.num_inputs, config.num_outputs
    X_train, y_train = make_dataset(config.dataset, config.n_train, data_rng, config.noise)
    X_test, y_test = make_dataset(config.dataset, config.n_test, data_rng, config.noise)

    names = list(config.activation_names)
    act_to_id = activation_ids(names)
    act_fns = activation_fns(names)

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

        # --- Express, train (backprop), score ---
        W, static = stack_models(
            [express(g, num_inputs, num_outputs, config.n_max, act_to_id) for g in pop]
        )
        fitnesses, W_trained, train_acc, test_acc, _bce, _conns = train_and_score(
            W, static, X_train, y_train, X_test, y_test, act_fns,
            num_passes=config.num_passes,
            steps=config.backprop_steps,
            lr=config.learning_rate,
            penalty_conn=config.penalty_conn,
            penalty_node=config.penalty_node,
        )

        # --- Lamarckian write-back: trained weights persist into the genomes ---
        for g, Wg in zip(pop, W_trained):
            write_back(g, Wg, num_inputs, num_outputs, config.n_max)

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

        rec = recorder.record(gen, pop, fitnesses, assignment,
                              train_acc=train_acc, test_acc=test_acc,
                              gen_time=time.time() - t0)
        if callback is not None:
            callback(rec)
        elif config.verbose:
            acc = rec.get("accuracy", {})
            sp, cx, b = rec["species"], rec["complexity"], rec["best"]
            if gen == 0:
                print(f"[{config.dataset}]  "
                      "gen |    fit max    mean |  acc best tr/te | mean te | "
                      "species (new/ext,max) | best h/c/d | pop h/c | stag |  time")
            print(
                f"      {rec['gen']:4d} | "
                f"{rec['fitness']['max']:8.3f} {rec['fitness']['mean']:7.3f} | "
                f"     {acc.get('train_best', 0):.2f}/{acc.get('test_best', 0):.2f} | "
                f"  {acc.get('test_mean', 0):.2f} | "
                f"{sp['count']:5d} (+{sp['new']} -{sp['extinct']}, {sp['largest']:>2}) | "
                f"{b['hidden']:3d}/{b['enabled_conns']:3d}/{b['depth']:2d} | "
                f"{cx['hidden']['mean']:4.1f}/{cx['enabled_conns']['mean']:5.1f} | "
                f"{rec['stagnation']:4d} | "
                f"{rec['gen_time_s']:5.1f}s"
            )
            if config.print_top_species:
                members: dict[int, list[int]] = {}
                for i, s in enumerate(assignment):
                    members.setdefault(s, []).append(i)
                top = sorted(members,
                             key=lambda s: max(fitnesses[i] for i in members[s]),
                             reverse=True)[:config.print_top_species]
                for s in top:
                    bi = max(members[s], key=lambda i: fitnesses[i])
                    g = pop[bi]
                    h = sum(1 for n in g.node_genes.values() if n.type == HIDDEN)
                    c = sum(1 for cc in g.conn_genes.values() if cc.enabled)
                    acts = Counter(n.activation for n in g.node_genes.values()
                                   if n.type == HIDDEN)
                    acts_str = " ".join(f"{a}:{k}" for a, k in acts.most_common()) or "-"
                    print(f"           sp {s:>3}: fit {fitnesses[bi]:8.3f} "
                          f"size {len(members[s]):3d} | h {h:2d} c {c:3d} d {max_depth(g):2d} "
                          f"| {acts_str}")

        # --- Reproduce ---
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
