"""Tests for the deterministic reduced-instance construction (§2.8)."""

import unittest
import warnings as _warnings

from gate_model import (
    PassengerFlow, Rotation,
    build_reduced_instance,
    build_reference_assignment, build_stress_sets,
    compute_q_scores, generate_scenarios,
)

_warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*stand model.*")


class ReducedInstanceTests(unittest.TestCase):

    def setUp(self):
        self.rotations = [
            Rotation("A", 100, 140, "M", "W1"),
            Rotation("B", 180, 220, "M", "W1"),
            Rotation("C", 250, 290, "M", "W2"),
            Rotation("D", 310, 350, "M", "W2"),
            Rotation("E", 370, 410, "M", "W2"),
        ]
        self.zones = {
            "Z1": [1],
            "Z2": [2, 3],
            "Z3": [4, 5, 6],
        }
        self.gates = [1, 2, 3, 4, 5, 6]
        self.flows = [
            PassengerFlow("P1", "C", "D", 10, 25),
            PassengerFlow("P2", "D", "E", 8, 25),
            PassengerFlow("P3", "A", "D", 20, 25),
            PassengerFlow("P4", "C", "B", 15, 25),
        ]
        self.compatibility = {
            (r.id, g): True for r in self.rotations for g in self.gates
        }
        self.compatibility.update({(r.id, 0): True for r in self.rotations})
        self.walking = {
            (i, j): (0 if i == j else 5)
            for i in [0] + self.gates for j in [0] + self.gates
        }

    def _build(self):
        return build_reduced_instance(
            rotations=self.rotations,
            passenger_flows=self.flows,
            gates=self.gates,
            zones_dict=self.zones,
            compatibility=self.compatibility,
            walking_times=self.walking,
        )

    def test_kept_wave_is_largest(self):
        r = self._build()
        self.assertEqual(r.kept_wave, "W2")

    def test_kept_wave_tie_broken_by_stable_id(self):
        rotations = [
            Rotation("A", 100, 140, "M", "W1"),
            Rotation("B", 180, 220, "M", "W2"),
        ]
        r = build_reduced_instance(
            rotations=rotations, passenger_flows=[],
            gates=self.gates, zones_dict=self.zones,
            compatibility=self.compatibility,
            walking_times=self.walking,
        )
        self.assertEqual(r.kept_wave, "W1")

    def test_kept_zones_are_two_largest_by_gate_count(self):
        r = self._build()
        self.assertEqual(r.kept_zones, ("Z3", "Z2"))
        self.assertEqual(set(r.gates), {2, 3, 4, 5, 6})

    def test_kept_zones_tie_broken_by_stable_id(self):
        zones = {"Z1": [1, 2], "Z2": [3, 4], "Z3": [5]}
        r = build_reduced_instance(
            rotations=self.rotations, passenger_flows=self.flows,
            gates=[1, 2, 3, 4, 5], zones_dict=zones,
            compatibility=self.compatibility,
            walking_times=self.walking,
        )
        self.assertEqual(r.kept_zones, ("Z1", "Z2"))

    def test_fewer_than_two_zones_are_all_kept(self):
        zones = {"Z1": [1, 2]}
        r = build_reduced_instance(
            rotations=self.rotations, passenger_flows=self.flows,
            gates=[1, 2], zones_dict=zones,
            compatibility=self.compatibility,
            walking_times=self.walking,
        )
        self.assertEqual(r.kept_zones, ("Z1",))

    def test_kept_rotations_are_wave_members(self):
        r = self._build()
        self.assertEqual([x.id for x in r.rotations], ["C", "D", "E"])

    def test_rotations_without_wave_are_excluded(self):
        rotations = [
            Rotation("A", 100, 140, "M", None),
            Rotation("B", 180, 220, "M", "W1"),
        ]
        r = build_reduced_instance(
            rotations=rotations, passenger_flows=[],
            gates=self.gates, zones_dict=self.zones,
            compatibility=self.compatibility,
            walking_times=self.walking,
        )
        self.assertEqual(r.kept_wave, "W1")
        self.assertNotIn("A", [x.id for x in r.rotations])

    def test_flows_with_one_endpoint_outside_are_dropped_and_reported(self):
        r = self._build()
        self.assertEqual([f.id for f in r.passenger_flows], ["P1", "P2"])
        self.assertEqual(set(r.dropped_flows), {"P3", "P4"})

    def test_q_recomputation_reflects_only_kept_flows(self):
        r = self._build()
        q = compute_q_scores(r.rotations, r.passenger_flows)
        self.assertEqual(q["C"], 10)
        self.assertEqual(q["D"], 18)
        self.assertEqual(q["E"], 8)
        self.assertNotIn("A", q)

    def test_scenarios_regenerated_on_reduced_instance(self):
        r = self._build()
        q = compute_q_scores(r.rotations, r.passenger_flows)
        ref = build_reference_assignment(
            r.rotations, r.gates, r.compatibility, 10,
        )
        gamma_f, stress = build_stress_sets(r.rotations, q)
        self.assertEqual(gamma_f, 1)
        self.assertEqual(set(stress.keys()), {"W2"})
        scenarios = generate_scenarios(
            r.zones, 1, [0, 30, 60], 15, ref, r.rotations, r.passenger_flows,
        )
        self.assertEqual(len(scenarios), 17)
        self.assertTrue(all(s.gamma_g == 1 for s in scenarios))

    def test_reduction_is_deterministic(self):
        r1 = self._build()
        r2 = self._build()
        self.assertEqual(r1.kept_wave, r2.kept_wave)
        self.assertEqual(r1.kept_zones, r2.kept_zones)
        self.assertEqual([x.id for x in r1.rotations],
                         [x.id for x in r2.rotations])
        self.assertEqual(r1.dropped_flows, r2.dropped_flows)

    def test_missing_walking_time_raises(self):
        incomplete = {
            (i, j): 5 for (i, j) in self.walking
            if not (i == 4 and j == 5)
        }
        with self.assertRaises(ValueError) as ctx:
            build_reduced_instance(
                rotations=self.rotations, passenger_flows=self.flows,
                gates=self.gates, zones_dict=self.zones,
                compatibility=self.compatibility,
                walking_times=incomplete,
            )
        self.assertIn("walking_times", str(ctx.exception))

    def test_no_waves_raises(self):
        rotations = [Rotation("A", 100, 140, "M", None)]
        with self.assertRaises(ValueError):
            build_reduced_instance(
                rotations=rotations, passenger_flows=[],
                gates=self.gates, zones_dict=self.zones,
                compatibility=self.compatibility,
                walking_times=self.walking,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)