"""Unit tests for the MILP building blocks (§2.8). Solver-free."""

import unittest
import warnings as _warnings

from gate_model import (
    PassengerFlow, Rotation, Scenario,
    build_b_constants, build_integer_extension, budget_constraints,
    build_stress_sets, check_constraints, compute_q_scores,
    eta_and_z_constraints, evaluate_integer_assignment, loss_expression,
    operational_protocol, resources_of,
)

_warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*stand model.*"
)


def _tight_instance():
    rotations = [
        Rotation("A", 100, 120, "M", "W1"),
        Rotation("B", 130, 150, "M", "W1"),
    ]
    flows = [PassengerFlow("P", "A", "B", 50, 20)]
    gates = [1, 2]
    compatibility = {(r.id, g): True for r in rotations for g in gates}
    compatibility.update({(r.id, 0): True for r in rotations})
    walking = {
        (i, j): (0 if i == j else 5)
        for i in [0] + gates for j in [0] + gates
    }
    return rotations, flows, gates, compatibility, walking


class MilpBuildingBlocksTests(unittest.TestCase):

    def setUp(self):
        (
            self.rotations, self.flows, self.gates,
            self.compatibility, self.walking,
        ) = _tight_instance()
        self.resources = resources_of(self.gates)
        self.q = compute_q_scores(self.rotations, self.flows)
        _, self.stress = build_stress_sets(self.rotations, self.q)

    def test_b_constants_equal_definition_at_no_shock(self):
        no_shock = Scenario("ns", 1, frozenset(), None, 0, 0, "test")
        b = build_b_constants(
            self.flows, self.resources, self.walking, [no_shock],
            self.rotations, self.stress,
        )
        self.assertEqual(b[("P", 1, 2, "ns")], 0)
        self.assertEqual(b[("P", 1, 1, "ns")], 0)

    def test_b_constants_capture_connection_breaking(self):
        delay = Scenario("d", 1, frozenset(), "W1", 30, 0, "delay_only")
        b = build_b_constants(
            self.flows, self.resources, self.walking, [delay],
            self.rotations, self.stress,
        )
        self.assertEqual(b[("P", 1, 2, "d")], 1)

    def test_eta_constraints_admit_reference_extension(self):
        plan = {"A": 1, "B": 2}
        no_shock = Scenario("ns", 1, frozenset(), None, 0, 0, "test")
        constraints = eta_and_z_constraints(
            self.resources, self.rotations, self.flows, [no_shock],
        )
        extension = build_integer_extension(
            plan, {"ns": plan}, self.flows, self.resources,
        )
        check_constraints(extension, constraints)
        self.assertEqual(extension[("eta", "P", 1, 2, "ns")], 1.0)
        self.assertEqual(extension[("eta", "P", 2, 1, "ns")], 0.0)

    def test_eta_constraints_reject_inconsistent_value(self):
        plan = {"A": 1, "B": 2}
        no_shock = Scenario("ns", 1, frozenset(), None, 0, 0, "test")
        constraints = eta_and_z_constraints(
            self.resources, self.rotations, self.flows, [no_shock],
        )
        extension = build_integer_extension(
            plan, {"ns": plan}, self.flows, self.resources,
        )
        bad = dict(extension)
        bad[("eta", "P", 1, 1, "ns")] = 1.0
        with self.assertRaises(AssertionError):
            check_constraints(bad, constraints)
        bad2 = dict(extension)
        bad2[("eta", "P", 1, 2, "ns")] = 0.0
        with self.assertRaises(AssertionError):
            check_constraints(bad2, constraints)

    def test_z_constraints_represent_no_change_correctly(self):
        plan = {"A": 1, "B": 2}
        no_shock = Scenario("ns", 1, frozenset(), None, 0, 0, "test")
        constraints = eta_and_z_constraints(
            self.resources, self.rotations, self.flows, [no_shock],
        )
        extension = build_integer_extension(
            plan, {"ns": plan}, self.flows, self.resources,
        )
        check_constraints(extension, constraints)
        self.assertEqual(extension[("z", "A", 1, "ns")], 1.0)
        self.assertEqual(extension[("z", "A", 2, "ns")], 0.0)

    def test_z_constraints_detect_a_modification(self):
        plan = {"A": 1, "B": 2}
        recourse = {"A": 2, "B": 2}
        no_shock = Scenario("ns", 1, frozenset(), None, 0, 0, "test")
        constraints = eta_and_z_constraints(
            self.resources, self.rotations, self.flows, [no_shock],
        )
        extension = build_integer_extension(
            plan, {"ns": recourse}, self.flows, self.resources,
        )
        check_constraints(extension, constraints)
        self.assertEqual(extension[("z", "A", 1, "ns")], 0.0)
        self.assertEqual(extension[("z", "A", 2, "ns")], 0.0)
        self.assertEqual(extension[("z", "B", 2, "ns")], 1.0)

    def test_budget_constraint_accepts_no_modifications(self):
        plan = {"A": 1, "B": 2}
        no_shock = Scenario("ns", 1, frozenset(), None, 0, 0, "test")
        extension = build_integer_extension(
            plan, {"ns": plan}, self.flows, self.resources,
        )
        budget = budget_constraints(
            self.rotations, self.resources, [no_shock], K=0,
        )
        check_constraints(extension, budget)

    def test_budget_constraint_rejects_overspend(self):
        plan = {"A": 1, "B": 2}
        recourse = {"A": 2, "B": 1}
        no_shock = Scenario("ns", 1, frozenset(), None, 0, 0, "test")
        extension = build_integer_extension(
            plan, {"ns": recourse}, self.flows, self.resources,
        )
        strict = budget_constraints(
            self.rotations, self.resources, [no_shock], K=1,
        )
        with self.assertRaises(AssertionError):
            check_constraints(extension, strict)
        loose = budget_constraints(
            self.rotations, self.resources, [no_shock], K=2,
        )
        check_constraints(extension, loose)

    def test_L_xi_matches_recourse_protocol_at_breaking_scenario(self):
        plan = {"A": 1, "B": 2}
        delay = Scenario("d", 1, frozenset(), "W1", 30, 0, "delay_only")
        result = operational_protocol(
            plan, delay, self.rotations, self.flows, self.gates,
            self.compatibility, self.walking, self.q, self.stress, 10, 4,
        )
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.broken_passengers, 50)
        b = build_b_constants(
            self.flows, self.resources, self.walking, [delay],
            self.rotations, self.stress,
        )
        L = evaluate_integer_assignment(
            {"d": result.assignment}, b, self.flows, self.resources,
        )
        self.assertEqual(L["d"], result.broken_passengers)

    def test_L_xi_matches_recourse_protocol_at_no_shock(self):
        plan = {"A": 1, "B": 2}
        no_shock = Scenario("ns", 1, frozenset(), None, 0, 0, "test")
        result = operational_protocol(
            plan, no_shock, self.rotations, self.flows, self.gates,
            self.compatibility, self.walking, self.q, self.stress, 10, 4,
        )
        self.assertEqual(result.broken_passengers, 0)
        b = build_b_constants(
            self.flows, self.resources, self.walking, [no_shock],
            self.rotations, self.stress,
        )
        L = evaluate_integer_assignment(
            {"ns": result.assignment}, b, self.flows, self.resources,
        )
        self.assertEqual(L["ns"], 0)

    def test_loss_expression_coefficients_match_direct_evaluation(self):
        plan = {"A": 1, "B": 2}
        delay = Scenario("d", 1, frozenset(), "W1", 30, 0, "delay_only")
        b = build_b_constants(
            self.flows, self.resources, self.walking, [delay],
            self.rotations, self.stress,
        )
        expr = loss_expression(self.flows, self.resources, b, delay)
        coeffs = dict(expr)
        self.assertEqual(coeffs[("eta", "P", 1, 2, "d")], 50.0)
        self.assertNotIn(("eta", "P", 1, 1, "d"), coeffs)
        extension = build_integer_extension(
            plan, {"d": plan}, self.flows, self.resources,
        )
        lhs = sum(c * extension.get(var, 0.0) for var, c in expr)
        self.assertEqual(lhs, 50.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)