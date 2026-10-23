import unittest
import warnings as _warnings

from gate_model import (
    PassengerFlow,
    Rotation,
    Scenario,
    build_reference_assignment,
    build_stress_sets,
    compute_q_scores,
    decode_chromosome,
    generate_scenarios,
    operational_protocol,
    _door_stress_scores,
)

_warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*stand model.*"
)


class GateModelTests(unittest.TestCase):
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

    def test_q_scores_count_each_flow_for_each_rotation(self):
        q = compute_q_scores(self.rotations, self.flows)
        self.assertEqual(q["A"], 10)
        self.assertEqual(q["B"], 10)
        self.assertEqual(q["C"], 8)
        self.assertEqual(q["D"], 8)

    def test_stress_sets_select_only_part_of_each_wave(self):
        q = compute_q_scores(self.rotations, self.flows)
        gamma_f, stress = build_stress_sets(self.rotations, q)
        self.assertEqual(gamma_f, 1)
        self.assertEqual(stress["W1"], frozenset({"A"}))
        self.assertEqual(stress["W2"], frozenset({"C"}))

    def test_scenarios_separate_gamma_levels_and_include_seven_families(self):
        ref = build_reference_assignment(
            self.rotations, self.gates, self.compatibility, 10
        )
        scenarios = generate_scenarios(
            self.zones, 1, [0, 30, 60], 15, ref, self.rotations, self.flows
        )
        self.assertEqual(len(scenarios), 29)
        self.assertTrue(all(s.gamma_g == 1 for s in scenarios))
        self.assertEqual(
            {s.family for s in scenarios},
            {
                "closure_only", "delay_only", "alert_only", "closure_delay",
                "closure_alert", "delay_alert", "closure_delay_alert",
            },
        )

    def test_decoder_returns_complete_nominal_assignment(self):
        q = compute_q_scores(self.rotations, self.flows)
        sigma = {r.id: 1 for r in self.rotations}
        plan = decode_chromosome(
            [r.id for r in self.rotations], sigma, self.rotations, self.gates,
            self.compatibility, q, 10,
        )
        self.assertEqual(set(plan), {r.id for r in self.rotations})

    def test_protocol_counts_broken_passengers_with_correct_time_direction(self):
        plan = {"A": 1, "B": 2, "C": 3, "D": 4}
        q = compute_q_scores(self.rotations, self.flows)
        _, stress = build_stress_sets(self.rotations, q)
        no_shock = Scenario("no-shock-test", 1, frozenset(), None, 0, 0, "test")
        result = operational_protocol(
            plan, no_shock, self.rotations, self.flows, self.gates,
            self.compatibility, self.walking, q, stress, 10, 4,
        )
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.broken_passengers, 0)

    def test_zero_budget_fails_when_a_nominal_gate_is_closed(self):
        plan = {"A": 1, "B": 2, "C": 3, "D": 4}
        q = compute_q_scores(self.rotations, self.flows)
        _, stress = build_stress_sets(self.rotations, q)
        scenario = Scenario("closed", 1, frozenset({1}), None, 0, 0, "closure_only")
        result = operational_protocol(
            plan, scenario, self.rotations, self.flows, self.gates,
            self.compatibility, self.walking, q, stress, 10, 0,
        )
        self.assertFalse(result.feasible)
        self.assertIsNone(result.broken_passengers)

    # ---- Point 1 : régression _door_stress_scores ----------------------

    def test_door_stress_counts_internal_flow_once(self):
        """Un flux interne à une porte doit peser n_l, pas 2·n_l."""
        rotations = [
            Rotation("A", 100, 140, "M", "W1"),
            Rotation("B", 200, 240, "M", "W1"),
            Rotation("C", 300, 340, "M", "W2"),
            Rotation("D", 400, 440, "M", "W2"),
        ]
        flows = [
            PassengerFlow("P1", "A", "B", 100, 25),
            PassengerFlow("P2", "C", "D", 120, 25),
        ]
        gates = [1, 2, 3, 4]
        ref = {"A": 1, "B": 1, "C": 3, "D": 2}

        scores = _door_stress_scores(gates, ref, flows)
        self.assertEqual(scores[1], 100)
        self.assertEqual(scores[2], 120)
        self.assertEqual(scores[3], 120)
        self.assertEqual(scores[4], 0)

    def test_closure_choice_respects_or_semantics(self):
        """Dans Z1={1,2}, la porte 2 doit l'emporter (s_1=100 < s_2=120)."""
        rotations = [
            Rotation("A", 100, 140, "M", "W1"),
            Rotation("B", 200, 240, "M", "W1"),
            Rotation("C", 300, 340, "M", "W2"),
            Rotation("D", 400, 440, "M", "W2"),
        ]
        flows = [
            PassengerFlow("P1", "A", "B", 100, 25),
            PassengerFlow("P2", "C", "D", 120, 25),
        ]
        gates = [1, 2, 3, 4]
        zones = {"Z1": [1, 2], "Z2": [3, 4]}
        ref = {"A": 1, "B": 1, "C": 3, "D": 2}

        scenarios = generate_scenarios(
            zones, 1, [0, 30, 60], 15, ref, rotations, flows,
        )
        closures = {
            s.closed_gates for s in scenarios if s.family == "closure_only"
        }
        self.assertIn(frozenset({2}), closures)
        self.assertIn(frozenset({3}), closures)
        self.assertNotIn(frozenset({1}), closures)

    # ---- Point 5 : départage par identifiant stable --------------------

    def test_decoder_tie_break_uses_stable_id_not_priority(self):
        """§2.7: à charge égale, l'identifiant stable départage."""
        rotations = [
            Rotation("A", 100, 140, "M", "W1"),
            Rotation("B", 200, 240, "M", "W1"),
        ]
        flows = [PassengerFlow("P", "A", "B", 10, 25)]
        gates = [1, 2]
        compatibility = {(r.id, g): True for r in rotations for g in gates}

        sigma = {"A": 0, "B": 3}
        q = compute_q_scores(rotations, flows)

        plan = decode_chromosome(
            ["A", "B"], sigma, rotations, gates, compatibility, q, 10,
            gate_priority={1: 2, 2: 1},
        )
        self.assertEqual(plan["B"], 1)

    # ---- Point 2/A : mode stands finis --------------------------------

    def test_aggregated_stand_mode_emits_warning(self):
        """finite_stands=None doit avertir (modèle agrégé)."""
        rotations = [Rotation("A", 100, 140, "M", "W1")]
        gates = [1]
        compatibility = {("A", 1): True, ("A", 0): True}
        with _warnings.catch_warnings(record=True) as captured:
            _warnings.simplefilter("always")
            build_reference_assignment(rotations, gates, compatibility, t_min=10)
        self.assertTrue(
            any(
                issubclass(w.category, RuntimeWarning)
                and "stand model" in str(w.message)
                for w in captured
            ),
            "expected a RuntimeWarning about the aggregated stand model",
        )

    def test_finite_stands_enforce_temporal_conflicts(self):
        """Deux rotations chevauchantes ne peuvent partager le même stand fini."""
        rotations = [
            Rotation("A", 100, 140, "M", "W1"),
            Rotation("B", 110, 150, "M", "W1"),
        ]
        gates = [1]
        finite_stands = [10]
        compatibility = {
            (r.id, res): True for r in rotations for res in gates + finite_stands
        }
        ref = build_reference_assignment(
            rotations, gates, compatibility, t_min=10,
            finite_stands=finite_stands,
        )
        self.assertEqual(ref["A"], 1)
        self.assertEqual(ref["B"], 10)
        self.assertNotIn(0, ref.values())

    def test_finite_stand_model_raises_when_infeasible(self):
        """Le mode fini échoue explicitement, sans repli silencieux sur 0."""
        rotations = [
            Rotation("A", 100, 140, "M", "W1"),
            Rotation("B", 110, 150, "M", "W1"),
            Rotation("C", 120, 160, "M", "W1"),
        ]
        gates = [1]
        finite_stands = [10]
        compatibility = {
            (r.id, res): True for r in rotations for res in gates + finite_stands
        }
        with self.assertRaises(ValueError):
            build_reference_assignment(
                rotations, gates, compatibility, t_min=10,
                finite_stands=finite_stands,
            )

    def test_operational_protocol_uses_finite_stand_on_gate_closure(self):
        """Après une fermeture de porte, π se replie vers un stand fini."""
        rotations = [
            Rotation("A", 100, 140, "M", "W1"),
            Rotation("B", 200, 240, "M", "W1"),
        ]
        flows = [PassengerFlow("P", "A", "B", 10, 5)]
        gates = [1, 2]
        finite_stands = [10, 20]
        compatibility = {
            (r.id, res): True for r in rotations for res in gates + finite_stands
        }
        walking = {
            (i, j): (0 if i == j else 5)
            for i in gates + finite_stands for j in gates + finite_stands
        }
        q = compute_q_scores(rotations, flows)
        _, stress = build_stress_sets(rotations, q)
        plan = {"A": 1, "B": 2}
        scenario = Scenario("closure", 1, frozenset({1}), None, 0, 0, "closure_only")
        result = operational_protocol(
            plan, scenario, rotations, flows, gates,
            compatibility, walking, q, stress, 10, 4,
            finite_stands=finite_stands,
        )
        self.assertTrue(result.feasible, result.reason)
        self.assertIn(result.assignment["A"], finite_stands + [2])
        self.assertNotIn(0, result.assignment.values())


if __name__ == "__main__":
    unittest.main(verbosity=2)