"""Tests for J_1, J_2, and the §2.6/§2.7 lexicographic ranking."""

import unittest
import warnings as _warnings
from unittest.mock import MagicMock

from gate_model import (
    PlanRanking,
    compute_J1_J2,
    rank_from_dual,
    rank_from_pi,
)

_warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*stand model.*")


def _mock_dual(I_pi, J3_pi, J3_star=None, delta_rob=None, delta_sc=None):
    return MagicMock(
        I_pi=I_pi,
        J3_pi=J3_pi,
        J3_star=J3_star,
        delta_rob=delta_rob,
        delta_sc=delta_sc,
        x_is_pi_admissible=(I_pi == 0),
        x_is_star_admissible=(J3_star is not None),
    )


class J1J2Tests(unittest.TestCase):

    def test_all_on_gates(self):
        J1, J2 = compute_J1_J2(
            {"A": 1, "B": 2, "C": 3, "D": 4}, [1, 2, 3, 4],
        )
        self.assertEqual((J1, J2), (4, 4))

    def test_shared_gate(self):
        J1, J2 = compute_J1_J2({"A": 1, "B": 1, "C": 2}, [1, 2, 3, 4])
        self.assertEqual((J1, J2), (3, 2))

    def test_stand_excluded_from_count(self):
        J1, J2 = compute_J1_J2({"A": 0, "B": 2, "C": 3, "D": 3}, [1, 2, 3, 4])
        self.assertEqual((J1, J2), (3, 2))

    def test_all_on_stand(self):
        J1, J2 = compute_J1_J2({"A": 0, "B": 0}, [1, 2, 3, 4])
        self.assertEqual((J1, J2), (0, 0))

    def test_empty_assignment(self):
        J1, J2 = compute_J1_J2({}, [1, 2, 3, 4])
        self.assertEqual((J1, J2), (0, 0))


class RankingTests(unittest.TestCase):

    def test_admissible_beats_inadmissible(self):
        r_adm = rank_from_dual(_mock_dual(I_pi=0, J3_pi=1000), {"A": 1}, [1])
        r_inadm = rank_from_dual(_mock_dual(I_pi=1, J3_pi=None), {"A": 1}, [1])
        self.assertLess(r_adm, r_inadm)
        self.assertGreater(r_inadm, r_adm)
        self.assertNotEqual(r_adm, r_inadm)

    def test_I_pi_minimized_first_among_inadmissible(self):
        r1 = rank_from_dual(_mock_dual(I_pi=1, J3_pi=None), {"A": 1}, [1])
        r3 = rank_from_dual(_mock_dual(I_pi=3, J3_pi=None), {"A": 1}, [1])
        self.assertLess(r1, r3)

    def test_J3_pi_minimized_among_admissible(self):
        r_low = rank_from_dual(_mock_dual(I_pi=0, J3_pi=10), {"A": 1}, [1])
        r_high = rank_from_dual(_mock_dual(I_pi=0, J3_pi=50), {"A": 1}, [1])
        self.assertLess(r_low, r_high)

    def test_J1_maximized_as_third_criterion(self):
        r_bigJ1 = rank_from_dual(
            _mock_dual(I_pi=0, J3_pi=10), {"A": 1, "B": 2}, [1, 2, 3, 4],
        )
        r_smallJ1 = rank_from_dual(
            _mock_dual(I_pi=0, J3_pi=10), {"A": 1, "B": 0}, [1, 2, 3, 4],
        )
        self.assertLess(r_bigJ1, r_smallJ1)

    def test_J2_minimized_as_fourth_criterion(self):
        r_fewJ2 = rank_from_dual(
            _mock_dual(I_pi=0, J3_pi=10), {"A": 1, "B": 1}, [1, 2, 3, 4],
        )
        r_manyJ2 = rank_from_dual(
            _mock_dual(I_pi=0, J3_pi=10), {"A": 1, "B": 2}, [1, 2, 3, 4],
        )
        self.assertLess(r_fewJ2, r_manyJ2)

    def test_J3_ignored_when_inadmissible(self):
        r_a = rank_from_dual(_mock_dual(I_pi=1, J3_pi=None), {"A": 1, "B": 2}, [1, 2])
        r_b = rank_from_dual(_mock_dual(I_pi=1, J3_pi=None), {"A": 1, "B": 0}, [1, 2])
        self.assertLess(r_a, r_b)

    def test_sorting_mixed_plans(self):
        plans = [
            rank_from_dual(_mock_dual(I_pi=2, J3_pi=None), {"A": 1}, [1]),
            rank_from_dual(_mock_dual(I_pi=0, J3_pi=20), {"A": 1}, [1]),
            rank_from_dual(_mock_dual(I_pi=1, J3_pi=None), {"A": 1}, [1]),
            rank_from_dual(_mock_dual(I_pi=0, J3_pi=5),  {"A": 1}, [1]),
        ]
        ordered = sorted(plans)
        self.assertEqual([p.I_pi for p in ordered], [0, 0, 1, 2])
        self.assertEqual(ordered[0].J3_pi, 5)
        self.assertEqual(ordered[1].J3_pi, 20)

    def test_metadata_does_not_affect_ordering(self):
        r1 = rank_from_dual(
            _mock_dual(I_pi=0, J3_pi=10, J3_star=8, delta_rob=2, delta_sc=2),
            {"A": 1}, [1],
        )
        r2 = rank_from_dual(
            _mock_dual(I_pi=0, J3_pi=10, J3_star=None),
            {"A": 1}, [1],
        )
        self.assertEqual(r1, r2)
        self.assertFalse(r1 < r2)
        self.assertFalse(r2 < r1)
        self.assertEqual(hash(r1), hash(r2))

    def test_rank_from_pi_without_milp(self):
        ev = MagicMock()
        ev.scenario_results = {
            "s1": MagicMock(feasible=True),
            "s2": MagicMock(feasible=True),
        }
        ev.admissible = True
        ev.worst_case_broken_passengers = 5
        r = rank_from_pi(ev, {"A": 1, "B": 2}, [1, 2])
        self.assertEqual((r.I_pi, r.J3_pi, r.J1, r.J2), (0, 5, 2, 2))
        self.assertIsNone(r.J3_star)
        self.assertIsNone(r.delta_rob)
        self.assertIsNone(r.is_star_admissible)

    def test_rank_from_pi_with_failures(self):
        ev = MagicMock()
        ev.scenario_results = {
            "s1": MagicMock(feasible=True),
            "s2": MagicMock(feasible=False),
            "s3": MagicMock(feasible=False),
        }
        ev.admissible = False
        ev.worst_case_broken_passengers = None
        r = rank_from_pi(ev, {"A": 1}, [1])
        self.assertEqual(r.I_pi, 2)
        self.assertIsNone(r.J3_pi)
        self.assertFalse(r.is_pi_admissible)

    def test_total_ordering_operators(self):
        r1 = rank_from_dual(_mock_dual(I_pi=0, J3_pi=5), {"A": 1}, [1])
        r2 = rank_from_dual(_mock_dual(I_pi=0, J3_pi=10), {"A": 1}, [1])
        self.assertTrue(r1 < r2)
        self.assertTrue(r1 <= r2)
        self.assertTrue(r2 > r1)
        self.assertTrue(r2 >= r1)
        self.assertFalse(r1 == r2)


if __name__ == "__main__":
    unittest.main(verbosity=2)