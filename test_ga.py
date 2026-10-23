"""Tests for the genetic operators and the GA loop (§2.7)."""

import random
import unittest
import warnings as _warnings

from gate_model import (
    PassengerFlow, Rotation,
    build_reference_assignment, build_stress_sets, compute_q_scores,
    generate_scenarios, resources_of,
)
from genetic_algorithm import (
    GAConfig,
    mutate_chromosome,
    order_crossover_2pt,
    run_ga,
    run_ga_multiple_seeds,
    uniform_crossover_sigma,
)

_warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*stand model.*")


class OperatorTests(unittest.TestCase):

    def test_ox_produces_permutation(self):
        p1 = ("A", "B", "C", "D", "E")
        p2 = ("E", "D", "C", "B", "A")
        child = order_crossover_2pt(random.Random(42), p1, p2)
        self.assertEqual(sorted(child), sorted(p1))
        self.assertEqual(len(set(child)), 5)

    def test_ox_deterministic_with_seed(self):
        p1 = ("A", "B", "C", "D", "E")
        p2 = ("E", "D", "C", "B", "A")
        c1 = order_crossover_2pt(random.Random(42), p1, p2)
        c2 = order_crossover_2pt(random.Random(42), p1, p2)
        self.assertEqual(c1, c2)

    def test_ox_small_permutation_returns_parent(self):
        c = order_crossover_2pt(random.Random(0), ("A",), ("A",))
        self.assertEqual(c, ("A",))

    def test_uniform_sigma_picks_from_parents(self):
        p1 = {"A": 1, "B": 2, "C": 3}
        p2 = {"A": 4, "B": 5, "C": 6}
        child = uniform_crossover_sigma(random.Random(42), p1, p2)
        for rid in ("A", "B", "C"):
            self.assertIn(child[rid], (p1[rid], p2[rid]))

    def test_mutation_preserves_validity(self):
        rho = ("A", "B", "C", "D")
        sigma = {"A": 1, "B": 2, "C": 3, "D": 4}
        resources = [1, 2, 3, 4]
        for seed in range(20):
            new_rho, new_sigma = mutate_chromosome(
                random.Random(seed), rho, sigma, resources,
            )
            self.assertEqual(sorted(new_rho), sorted(rho))
            self.assertEqual(set(new_sigma), set(sigma))
            for rid in new_sigma:
                self.assertIn(new_sigma[rid], resources)


class GATests(unittest.TestCase):

    def setUp(self):
        self.rotations = [
            Rotation("A", 100, 140, "M", "W1"),
            Rotation("B", 180, 220, "M", "W1"),
            Rotation("C", 110, 150, "M", "W2"),
            Rotation("D", 260, 300, "M", "W2"),
        ]
        self.flows = [
            PassengerFlow("P1", "A", "B", 10, 25),
            PassengerFlow("P2", "C", "D", 8, 25),
        ]
        self.gates = [1, 2, 3, 4]
        self.zones = {"Z1": [1, 2], "Z2": [3, 4]}
        self.compatibility = {
            (r.id, g): True for r in self.rotations for g in self.gates
        }
        self.compatibility.update({(r.id, 0): True for r in self.rotations})
        self.walking = {
            (i, j): (0 if i == j else 5)
            for i in [0] + self.gates for j in [0] + self.gates
        }
        self.resources = resources_of(self.gates)
        self.q = compute_q_scores(self.rotations, self.flows)
        _, self.stress = build_stress_sets(self.rotations, self.q)
        ref = build_reference_assignment(
            self.rotations, self.gates, self.compatibility, 10,
        )
        self.scenarios = generate_scenarios(
            self.zones, 1, [0, 30, 60], 15, ref, self.rotations, self.flows,
        )

    def _run(self, seed, config, K):
        return run_ga(
            seed, config,
            rotations=self.rotations, passenger_flows=self.flows,
            gates=self.gates, compatibility=self.compatibility,
            walking_times=self.walking, q_scores=self.q,
            stress_sets=self.stress, scenarios=self.scenarios,
            resources=self.resources, t_min=10, K=K,
        )

    def test_ga_deterministic_with_same_seed(self):
        config = GAConfig(population_size=8, max_generations=3, seeds=(1,))
        r1 = self._run(1, config, K=4)
        r2 = self._run(1, config, K=4)
        self.assertEqual(r1.best_individual.ranking.key,
                         r2.best_individual.ranking.key)
        self.assertEqual(r1.cache_size, r2.cache_size)

    def test_ga_elitism_preserves_best(self):
        config = GAConfig(population_size=10, max_generations=5, seeds=(1,))
        result = self._run(1, config, K=4)
        keys = [r.key for r in result.best_per_generation]
        for prev, cur in zip(keys, keys[1:]):
            self.assertLessEqual(cur, prev,
                                 f"best regressed: {prev} -> {cur}")

    def test_ga_finds_feasible_plan_with_adequate_budget(self):
        config = GAConfig(population_size=20, max_generations=10, seeds=(1,))
        result = self._run(1, config, K=4)
        self.assertEqual(result.best_individual.ranking.I_pi, 0)
        self.assertTrue(result.best_individual.ranking.is_pi_admissible)
        self.assertIsNotNone(result.best_individual.plan)

    def test_ga_history_length(self):
        config = GAConfig(population_size=6, max_generations=4, seeds=(1,))
        result = self._run(1, config, K=4)
        self.assertEqual(len(result.best_per_generation), 5)

    def test_ga_cache_and_protocol_calls_are_consistent(self):
        config = GAConfig(population_size=6, max_generations=3, seeds=(1,))
        result = self._run(1, config, K=4)
        self.assertEqual(result.cache_size, result.protocol_calls)
        self.assertLessEqual(
            result.protocol_calls,
            config.population_size * (1 + config.max_generations),
        )

    def test_ga_multiple_seeds_returns_one_per_seed(self):
        config = GAConfig(population_size=6, max_generations=2, seeds=(1, 2))
        results = run_ga_multiple_seeds(
            config,
            rotations=self.rotations, passenger_flows=self.flows,
            gates=self.gates, compatibility=self.compatibility,
            walking_times=self.walking, q_scores=self.q,
            stress_sets=self.stress, scenarios=self.scenarios,
            resources=self.resources, t_min=10, K=4,
        )
        self.assertEqual([r.seed for r in results], [1, 2])

    def test_ga_default_config_uses_seeds_1_to_5(self):
        config = GAConfig()
        self.assertEqual(config.seeds, (1, 2, 3, 4, 5))
        self.assertEqual(config.population_size, 150)
        self.assertEqual(config.max_generations, 400)
        self.assertEqual(config.crossover_probability, 0.8)
        self.assertEqual(config.mutation_probability, 0.05)
        self.assertEqual(config.tournament_size, 3)
        self.assertEqual(config.elitism, 2)

    def test_ga_config_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            GAConfig(population_size=1)
        with self.assertRaises(ValueError):
            GAConfig(elitism=10, population_size=10)
        with self.assertRaises(ValueError):
            GAConfig(crossover_probability=1.5)
        with self.assertRaises(ValueError):
            GAConfig(tournament_size=200, population_size=10)
        with self.assertRaises(ValueError):
            GAConfig(seeds=())


if __name__ == "__main__":
    unittest.main(verbosity=2)