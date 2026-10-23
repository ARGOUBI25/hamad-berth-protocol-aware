"""Tests for the joint nominal+recourse MILP (§2.8)."""

import dataclasses
import unittest
import warnings as _warnings
from types import MappingProxyType

from gate_model import (
    PassengerFlow,
    Rotation,
    build_reduced_instance,
    build_reference_assignment,
    build_stress_sets,
    compute_J1_J2,
    compute_q_scores,
    generate_scenarios,
    resources_of,
    validate_nominal_plan,
)
from joint_milp import solve_joint_milp
from recourse_milp import solve_recourse_milp

_warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*stand model.*"
)


class JointMilpTests(unittest.TestCase):

    def setUp(self):
        self.rotations = [
            Rotation("A", 100, 150, "M", "W1"),
            Rotation("B", 130, 180, "M", "W1"),
        ]
        self.flows = [PassengerFlow("P", "A", "B", 30, 10)]
        self.gates = [1, 2]
        self.finite_stands = [10]
        self.zones = {"Z1": [1], "Z2": [2]}
        self.compatibility = {
            (r.id, a): True
            for r in self.rotations
            for a in self.gates + self.finite_stands
        }
        self.walking = {
            (a, b): (0 if a == b else 5)
            for a in [0] + self.gates + self.finite_stands
            for b in [0] + self.gates + self.finite_stands
        }
        self.resources = resources_of(
            self.gates, 0, self.finite_stands,
        )
        self.q = compute_q_scores(self.rotations, self.flows)
        _, self.stress = build_stress_sets(self.rotations, self.q)
        self.t_min = 10

    def _reduced(self):
        return build_reduced_instance(
            rotations=self.rotations,
            passenger_flows=self.flows,
            gates=self.gates,
            zones_dict=self.zones,
            compatibility=self.compatibility,
            walking_times=self.walking,
        )

    def _scenarios(self):
        reduced = self._reduced()
        ref = build_reference_assignment(
            reduced.rotations, reduced.gates, reduced.compatibility,
            self.t_min, finite_stands=self.finite_stands,
        )
        base = generate_scenarios(
            reduced.zones, 1, [0, 30, 60], 0, ref,
            reduced.rotations, reduced.passenger_flows,
        )
        return [
            s for s in base
            if s.family in ("closure_only", "delay_only")
        ]

    def _solve(self, K):
        reduced = self._reduced()
        # Ensure walking_times includes finite_stands (they're not in reduced zones)
        # reduced.walking_times is immutable, so create a mutable copy and replace
        walking_dict = dict(reduced.walking_times)
        for i in self.finite_stands:
            for j in [0] + list(reduced.gates) + self.finite_stands:
                if (i, j) not in walking_dict:
                    walking_dict[(i, j)] = self.walking[(i, j)]
            for j in [0] + list(reduced.gates) + self.finite_stands:
                if (j, i) not in walking_dict:
                    walking_dict[(j, i)] = self.walking[(j, i)]
        reduced = dataclasses.replace(
            reduced, walking_times=MappingProxyType(walking_dict)
        )
        _, stress = build_stress_sets(
            reduced.rotations,
            compute_q_scores(
                reduced.rotations, reduced.passenger_flows,
            ),
        )
        return solve_joint_milp(
            reduced, self._scenarios(), stress, self.resources,
            self.t_min, K, finite_stands=self.finite_stands,
        )

    def test_optimal_on_feasible_instance(self):
        result = self._solve(K=4)
        self.assertEqual(result.status, "OPTIMAL", result.notes)
        self.assertEqual(result.phase_reached, 3, result.notes)
        self.assertIsNotNone(result.J3_star)
        self.assertIsNotNone(result.J1)
        self.assertIsNotNone(result.J2)
        self.assertIsNotNone(result.nominal_plan)
        self.assertEqual(
            len(result.nominal_plan), len(self.rotations),
        )

    def test_infeasible_with_zero_budget_on_dual_closures(self):
        result = self._solve(K=0)
        self.assertEqual(result.status, "INFEASIBLE")
        self.assertIsNone(result.J3_star)
        self.assertIsNone(result.nominal_plan)

    def test_reported_J1_J2_match_nominal_plan(self):
        result = self._solve(K=4)
        J1, J2 = compute_J1_J2(result.nominal_plan, self.gates)
        self.assertEqual(result.J1, J1)
        self.assertEqual(result.J2, J2)

    def test_nominal_plan_is_valid(self):
        result = self._solve(K=4)
        validate_nominal_plan(
            result.nominal_plan, self.rotations, self.gates,
            self.compatibility, self.t_min,
            finite_stands=self.finite_stands,
        )

    def test_joint_matches_recourse_milp_on_found_plan(self):
        reduced = self._reduced()
        # Ensure walking_times includes finite_stands (they're not in reduced zones)
        # reduced.walking_times is immutable, so create a mutable copy and replace
        walking_dict = dict(reduced.walking_times)
        for i in self.finite_stands:
            for j in [0] + list(reduced.gates) + self.finite_stands:
                if (i, j) not in walking_dict:
                    walking_dict[(i, j)] = self.walking[(i, j)]
            for j in [0] + list(reduced.gates) + self.finite_stands:
                if (j, i) not in walking_dict:
                    walking_dict[(j, i)] = self.walking[(j, i)]
        reduced = dataclasses.replace(
            reduced, walking_times=MappingProxyType(walking_dict)
        )
        _, stress = build_stress_sets(
            reduced.rotations,
            compute_q_scores(
                reduced.rotations, reduced.passenger_flows,
            ),
        )
        scenarios = self._scenarios()
        joint = solve_joint_milp(
            reduced, scenarios, stress, self.resources,
            self.t_min, K=4, finite_stands=self.finite_stands,
        )
        self.assertEqual(joint.status, "OPTIMAL")
        losses = []
        for scenario in scenarios:
            recourse = solve_recourse_milp(
                joint.nominal_plan, scenario, self.rotations,
                self.flows, self.gates, self.compatibility, self.walking,
                stress, self.resources, self.t_min, K=4,
                finite_stands=self.finite_stands,
            )
            self.assertTrue(recourse.feasible, recourse.reason)
            losses.append(recourse.broken_passengers or 0)
        self.assertEqual(joint.J3_star, max(losses))

    def test_deterministic(self):
        r1 = self._solve(K=4)
        r2 = self._solve(K=4)
        self.assertEqual(r1.status, r2.status)
        self.assertEqual(r1.J3_star, r2.J3_star)
        self.assertEqual(r1.J1, r2.J1)
        self.assertEqual(r1.J2, r2.J2)
        self.assertEqual(
            dict(r1.nominal_plan), dict(r2.nominal_plan),
        )

    def test_no_inconsistency_in_notes(self):
        result = self._solve(K=4)
        for note in result.notes:
            self.assertNotIn("violated", note.lower(), note)
            self.assertNotIn("bug", note.lower(), note)


if __name__ == "__main__":
    unittest.main(verbosity=2)