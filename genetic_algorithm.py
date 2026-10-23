"""Genetic algorithm for the first-stage search (§2.7).

Chromosome = (rho, sigma):
  - rho   : permutation of rotation IDs — construction order of the nominal plan
  - sigma : mapping rotation -> proposed resource

Operators and parameters follow the methodology exactly:
  - OX two-point crossover on rho
  - uniform crossover on sigma
  - crossover probability 0.8
  - mutation probability 0.05; when triggered, 50/50 swap-in-rho or resample-one-sigma
  - initial population 150 (uniform permutation rho, uniform sigma over A)
  - max 400 generations
  - tournament selection, size 3
  - elitism: keep the 2 best individuals of the previous generation,
    replacing the 2 worst children
  - seeds 1..5

Selection uses the lexicographic ranking of §2.6/§2.7:
    (I_pi, J3_pi, -J1, J2)
Rankings are produced by `gate_model.rank_from_pi`; the GA on the full
instance uses π only.

Determinism: every random decision goes through a `random.Random(seed)`,
never the global `random` module. The cache is keyed by decoded-plan
signature, so identical plans share one π evaluation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from types import MappingProxyType
from typing import Hashable, Mapping, Sequence

from gate_model import (
    GateId,
    PassengerFlow,
    PlanRanking,
    ProtocolResult,
    ResourceId,
    Rotation,
    RotationId,
    Scenario,
    WaveId,
    decode_chromosome,
    evaluate_robust_protocol,
    rank_from_pi,
)


# ---------------------------------------------------------------------------
# Public configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GAConfig:
    population_size: int = 150
    max_generations: int = 400
    crossover_probability: float = 0.8
    mutation_probability: float = 0.05
    tournament_size: int = 3
    elitism: int = 2
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    rho_mutation_share: float = 0.5

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if self.max_generations < 0:
            raise ValueError("max_generations must be >= 0")
        if not (0.0 <= self.crossover_probability <= 1.0):
            raise ValueError("crossover_probability must be in [0, 1]")
        if not (0.0 <= self.mutation_probability <= 1.0):
            raise ValueError("mutation_probability must be in [0, 1]")
        if self.tournament_size < 1:
            raise ValueError("tournament_size must be >= 1")
        if self.tournament_size > self.population_size:
            raise ValueError("tournament_size must be <= population_size")
        if not (0 <= self.elitism < self.population_size):
            raise ValueError("elitism must be in [0, population_size)")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if not (0.0 <= self.rho_mutation_share <= 1.0):
            raise ValueError("rho_mutation_share must be in [0, 1]")


@dataclass(frozen=True)
class GAIndividual:
    rho: tuple[RotationId, ...]
    sigma: Mapping[RotationId, ResourceId]
    plan: Mapping[RotationId, ResourceId] | None
    ranking: PlanRanking


@dataclass(frozen=True)
class GARunResult:
    seed: int
    best_individual: GAIndividual
    best_per_generation: tuple[PlanRanking, ...]
    final_population: tuple[GAIndividual, ...]
    cache_size: int
    protocol_calls: int


# ---------------------------------------------------------------------------
# Operators (§2.7)
# ---------------------------------------------------------------------------

def order_crossover_2pt(
    rng: random.Random,
    parent1: Sequence[RotationId],
    parent2: Sequence[RotationId],
) -> tuple[RotationId, ...]:
    """OX two-point crossover on a permutation, preserving relative order.

    Two cut points c1 <= c2 are sampled uniformly. The segment [c1, c2) of
    parent1 is copied verbatim into the child; the remaining positions are
    filled with the elements of parent2 in their order of appearance,
    skipping those already present.
    """
    n = len(parent1)
    if n != len(parent2):
        raise ValueError("OX requires parents of equal length")
    if n < 2:
        return tuple(parent1)
    c1, c2 = sorted(rng.sample(range(n + 1), 2))
    child: list[RotationId | None] = [None] * n
    in_segment: set[RotationId] = set()
    for k in range(c1, c2):
        child[k] = parent1[k]
        in_segment.add(parent1[k])
    fill = (x for x in parent2 if x not in in_segment)
    for k in range(n):
        if child[k] is None:
            child[k] = next(fill)
    return tuple(child)  # type: ignore[arg-type]


def uniform_crossover_sigma(
    rng: random.Random,
    parent1: Mapping[RotationId, ResourceId],
    parent2: Mapping[RotationId, ResourceId],
) -> dict[RotationId, ResourceId]:
    """Uniform crossover on sigma: each coordinate from one parent with prob 1/2."""
    if set(parent1) != set(parent2):
        raise ValueError("sigma crossover requires matching key sets")
    return {
        rid: (parent1[rid] if rng.random() < 0.5 else parent2[rid])
        for rid in parent1
    }


def mutate_chromosome(
    rng: random.Random,
    rho: Sequence[RotationId],
    sigma: Mapping[RotationId, ResourceId],
    resources: Sequence[ResourceId],
    *,
    rho_share: float = 0.5,
) -> tuple[tuple[RotationId, ...], dict[RotationId, ResourceId]]:
    """One mutation (§2.7): swap two rho positions or resample one sigma."""
    rho_list = list(rho)
    sigma_dict = dict(sigma)
    resources_list = list(resources)
    if not resources_list:
        raise ValueError("resources must be non-empty")
    do_rho = rng.random() < rho_share and len(rho_list) >= 2
    if do_rho:
        i, j = rng.sample(range(len(rho_list)), 2)
        rho_list[i], rho_list[j] = rho_list[j], rho_list[i]
    else:
        rid = rng.choice(list(sigma_dict.keys()))
        sigma_dict[rid] = rng.choice(resources_list)
    return tuple(rho_list), sigma_dict


# ---------------------------------------------------------------------------
# Evaluation with plan cache
# ---------------------------------------------------------------------------

def _infeasible_ranking(num_scenarios: int, gates: Sequence[GateId]) -> PlanRanking:
    """Ranking for a chromosome that cannot even produce a nominal plan.

    Treated as worse than any chromosome with a plan (§2.7): I_pi strictly
    larger than any reachable count, J3_pi undefined.
    """
    I_pi = num_scenarios + 1
    J1, J2 = 0, len(gates)
    return PlanRanking(
        I_pi=I_pi,
        J3_pi=None,
        J1=J1,
        J2=J2,
        is_pi_admissible=False,
        is_star_admissible=None,
        J3_star=None,
        delta_rob=None,
        delta_sc=None,
        key=(I_pi, float("inf"), -J1, J2),
    )


def evaluate_chromosome(
    rho: Sequence[RotationId],
    sigma: Mapping[RotationId, ResourceId],
    *,
    rotations: Sequence[Rotation],
    passenger_flows: Sequence[PassengerFlow],
    gates: Sequence[GateId],
    compatibility: Mapping[tuple[RotationId, GateId], bool],
    walking_times: Mapping[tuple[ResourceId, ResourceId], float],
    q_scores: Mapping[RotationId, int],
    stress_sets: Mapping[WaveId, frozenset[RotationId]],
    scenarios: Sequence[Scenario],
    t_min: float,
    K: int,
    stand_id: ResourceId = 0,
    finite_stands: Sequence[ResourceId] | None = None,
    cache: dict[frozenset, PlanRanking] | None = None,
) -> GAIndividual:
    """Decode + evaluate under π + rank. Cached by decoded-plan signature."""
    if cache is None:
        cache = {}
    try:
        plan = decode_chromosome(
            rho, sigma, rotations, gates, compatibility, q_scores, t_min,
            stand_id=stand_id, finite_stands=finite_stands,
        )
    except ValueError:
        return GAIndividual(
            rho=tuple(rho),
            sigma=MappingProxyType(dict(sigma)),
            plan=None,
            ranking=_infeasible_ranking(len(scenarios), gates),
        )

    signature = frozenset(plan.items())
    ranking = cache.get(signature)
    if ranking is None:
        evaluation = evaluate_robust_protocol(
            plan, scenarios, rotations, passenger_flows, gates,
            compatibility, walking_times, q_scores, stress_sets, t_min, K,
            stand_id=stand_id, finite_stands=finite_stands,
        )
        ranking = rank_from_pi(evaluation, plan, gates)
        cache[signature] = ranking

    return GAIndividual(
        rho=tuple(rho),
        sigma=MappingProxyType(dict(sigma)),
        plan=MappingProxyType(dict(plan)),
        ranking=ranking,
    )


# ---------------------------------------------------------------------------
# Selection and child generation
# ---------------------------------------------------------------------------

def _tournament_select(
    rng: random.Random,
    population: Sequence[GAIndividual],
    size: int,
) -> GAIndividual:
    contenders = rng.sample(list(population), size)
    return min(contenders, key=lambda ind: ind.ranking.key)


def _generate_child(
    rng: random.Random,
    population: Sequence[GAIndividual],
    resources: Sequence[ResourceId],
    config: GAConfig,
) -> tuple[tuple[RotationId, ...], dict[RotationId, ResourceId]]:
    p1 = _tournament_select(rng, population, config.tournament_size)
    p2 = _tournament_select(rng, population, config.tournament_size)

    if rng.random() < config.crossover_probability:
        rho = order_crossover_2pt(rng, p1.rho, p2.rho)
        sigma = uniform_crossover_sigma(rng, p1.sigma, p2.sigma)
    else:
        rho = tuple(p1.rho)
        sigma = dict(p1.sigma)

    if rng.random() < config.mutation_probability:
        rho, sigma = mutate_chromosome(
            rng, rho, sigma, resources,
            rho_share=config.rho_mutation_share,
        )

    return rho, sigma


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_ga(
    seed: int,
    config: GAConfig,
    *,
    rotations: Sequence[Rotation],
    passenger_flows: Sequence[PassengerFlow],
    gates: Sequence[GateId],
    compatibility: Mapping[tuple[RotationId, GateId], bool],
    walking_times: Mapping[tuple[ResourceId, ResourceId], float],
    q_scores: Mapping[RotationId, int],
    stress_sets: Mapping[WaveId, frozenset[RotationId]],
    scenarios: Sequence[Scenario],
    resources: Sequence[ResourceId],
    t_min: float,
    K: int,
    stand_id: ResourceId = 0,
    finite_stands: Sequence[ResourceId] | None = None,
) -> GARunResult:
    """One GA run with the given seed and config (§2.7)."""
    rng = random.Random(seed)
    cache: dict[frozenset, PlanRanking] = {}
    protocol_calls = [0]

    def _evaluate(rho, sigma):
        before = len(cache)
        individual = evaluate_chromosome(
            rho, sigma,
            rotations=rotations, passenger_flows=passenger_flows,
            gates=gates, compatibility=compatibility,
            walking_times=walking_times, q_scores=q_scores,
            stress_sets=stress_sets, scenarios=scenarios, t_min=t_min, K=K,
            stand_id=stand_id, finite_stands=finite_stands, cache=cache,
        )
        if len(cache) > before:
            protocol_calls[0] += 1
        return individual

    rotation_ids = [r.id for r in rotations]
    resource_list = list(resources)

    # Initial population: uniform rho, uniform sigma over A (§2.7).
    population: list[GAIndividual] = []
    for _ in range(config.population_size):
        rho = tuple(rng.sample(rotation_ids, len(rotation_ids)))
        sigma = {rid: rng.choice(resource_list) for rid in rotation_ids}
        population.append(_evaluate(rho, sigma))
    population.sort(key=lambda ind: ind.ranking.key)

    history: list[PlanRanking] = [population[0].ranking]
    best_ever = population[0]

    for _gen in range(config.max_generations):
        children: list[GAIndividual] = []
        for _ in range(config.population_size):
            rho, sigma = _generate_child(rng, population, resource_list, config)
            children.append(_evaluate(rho, sigma))

        children.sort(key=lambda ind: ind.ranking.key)

        if config.elitism > 0:
            elites = population[:config.elitism]
            survivors = children[: config.population_size - config.elitism]
            population = sorted(
                survivors + elites, key=lambda ind: ind.ranking.key,
            )
        else:
            population = children

        if population[0].ranking.key < best_ever.ranking.key:
            best_ever = population[0]
        history.append(population[0].ranking)

    return GARunResult(
        seed=seed,
        best_individual=best_ever,
        best_per_generation=tuple(history),
        final_population=tuple(population),
        cache_size=len(cache),
        protocol_calls=protocol_calls[0],
    )


def run_ga_multiple_seeds(
    config: GAConfig,
    *,
    rotations: Sequence[Rotation],
    passenger_flows: Sequence[PassengerFlow],
    gates: Sequence[GateId],
    compatibility: Mapping[tuple[RotationId, GateId], bool],
    walking_times: Mapping[tuple[ResourceId, ResourceId], float],
    q_scores: Mapping[RotationId, int],
    stress_sets: Mapping[WaveId, frozenset[RotationId]],
    scenarios: Sequence[Scenario],
    resources: Sequence[ResourceId],
    t_min: float,
    K: int,
    stand_id: ResourceId = 0,
    finite_stands: Sequence[ResourceId] | None = None,
) -> list[GARunResult]:
    """Run the GA once per seed in `config.seeds` (§2.7: seeds 1..5)."""
    return [
        run_ga(
            seed, config,
            rotations=rotations, passenger_flows=passenger_flows,
            gates=gates, compatibility=compatibility,
            walking_times=walking_times, q_scores=q_scores,
            stress_sets=stress_sets, scenarios=scenarios,
            resources=resources, t_min=t_min, K=K,
            stand_id=stand_id, finite_stands=finite_stands,
        )
        for seed in config.seeds
    ]