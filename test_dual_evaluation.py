"""Conformance tests for §2.6: J3_*, delta_rob, delta_sc."""

import unittest
import warnings as _warnings

from gate_model import (
    PassengerFlow, Rotation, Scenario,
    build_reference_assignment, build_stress_sets, compute_q_scores,
    generate_scenarios, operational_protocol, resources_of,
)
from recourse_milp import evaluate_dual_recourse

_warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*stand model.*")


class DualEvaluationTests(unittest.TestCase):

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

    def _scenarios(self):
        ref = build_reference_assignment(
            self.rotations, self.gates, self.compatibility, 10,
        )
        return generate_scenarios(
            self.zones, 1, [0, 30, 60], 15, ref,
            self.rotations, self.flows,
        )

    def _evaluate(self, plan, K):
        return evaluate_dual_recourse(
            plan, self._scenarios(), self.rotations, self.flows, self.gates,
            self.compatibility, self.walking, self.q, self.stress,
            self.resources, t_min=10, K=K,
        )

    def test_full_evaluation_with_adequate_budget(self):
        plan = {"A": 1, "B": 2, "C": 3, "D": 4}
        ev = self._evaluate(plan, K=len(self.rotations))

        self.assertTrue(ev.x_is_pi_admissible, ev.inconsistency_warnings)
        self.assertTrue(ev.x_is_star_admissible)
        self.assertEqual(ev.I_pi, 0)
        self.assertIsNotNone(ev.J3_pi)
        self.assertIsNotNone(ev.J3_star)
        self.assertIsNotNone(ev.delta_rob)
        self.assertIsNotNone(ev.delta_sc)
        self.assertGreaterEqual(ev.delta_rob, 0)
        self.assertGreaterEqual(ev.delta_sc, 0)
        self.assertGreaterEqual(ev.delta_rob, ev.delta_sc)
        self.assertEqual(ev.inconsistency_warnings, ())

    def test_pi_admissibility_implies_star_admissibility(self):
        plan = {"A": 1, "B": 2, "C": 3, "D": 4}
        ev = self._evaluate(plan, K=0)
        if ev.x_is_pi_admissible:
            self.assertTrue(ev.x_is_star_admissible)
        self.assertFalse(ev.x_is_pi_admissible)
        self.assertGreater(ev.I_pi, 0)
        self.assertIsNone(ev.J3_pi)
        self.assertIsNone(ev.delta_rob)
        self.assertIsNone(ev.delta_sc)

    def test_every_scenario_has_consistent_dual_results(self):
        plan = {"A": 1, "B": 2, "C": 3, "D": 4}
        ev = self._evaluate(plan, K=4)
        for sid, r in ev.per_scenario.items():
            self.assertEqual(sid, r.scenario_id)
            if r.pi_feasible and r.star_feasible:
                self.assertLessEqual(r.L_star, r.L_pi)
            self.assertFalse(
                r.pi_feasible and not r.star_feasible,
                f"I1 violated at {sid}",
            )

    def test_delta_rob_equals_difference_of_robust_values(self):
        plan = {"A": 1, "B": 2, "C": 3, "D": 4}
        ev = self._evaluate(plan, K=4)
        self.assertEqual(ev.delta_rob, ev.J3_pi - ev.J3_star)
        expected_delta_sc = max(
            r.L_pi - r.L_star for r in ev.per_scenario.values()
        )
        self.assertEqual(ev.delta_sc, expected_delta_sc)

    def test_family_coverage_in_dual_results(self):
        plan = {"A": 1, "B": 2, "C": 3, "D": 4}
        ev = self._evaluate(plan, K=4)
        families = {r.family for r in ev.per_scenario.values()}
        self.assertEqual(families, {
            "closure_only", "delay_only", "alert_only", "closure_delay",
            "closure_alert", "delay_alert", "closure_delay_alert",
        })


if __name__ == "__main__":
    unittest.main(verbosity=2)