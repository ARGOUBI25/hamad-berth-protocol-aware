"""Three classical baselines for the berth allocation problem.

Each baseline produces a nominal plan x, which is then evaluated
under the real scenarios using the same pipeline as π and MILP:

    - greedy_first_fit : first compatible free berth in arrival order
    - greedy_best_fit  : compatible berth minimizing wasted LOA
    - deterministic_milp : joint MILP solved on a single nominal scenario
                           (no closures, no delays, no alerts)

The outputs are ProtocolResult-compatible so they can be compared
side-by-side with π, MILP, and GA.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from gate_model import (
    GateId, PassengerFlow, ResourceId, Rotation, RotationId,
    Scenario, WaveId, compute_q_scores, resources_of,
)


def _compatible_berths(
    rot: Rotation,
    gates: Sequence[GateId],
    compatibility: Mapping[tuple[RotationId, GateId], bool],
) -> list[GateId]:
    return [g for g in gates if compatibility.get((rot.id, g), False)]


def _has_conflict(
    arr: float, dep: float,
    intervals: Sequence[tuple[float, float]],
    t_min: float,
) -> bool:
    for other_arr, other_dep in intervals:
        if not (dep + t_min <= other_arr or arr >= other_dep + t_min):
            return True
    return False


def greedy_first_fit(
    rotations: Sequence[Rotation],
    gates: Sequence[GateId],
    compatibility: Mapping[tuple[RotationId, GateId], bool],
    t_min: float,
    *,
    stand_id: ResourceId = 0,
) -> dict[RotationId, ResourceId]:
    """Assign each rotation (sorted by arrival) to the first compatible free berth."""
    ordered = sorted(rotations, key=lambda r: (r.arr, str(r.id)))
    intervals: dict[GateId, list[tuple[float, float]]] = {g: [] for g in gates}
    plan: dict[RotationId, ResourceId] = {}

    for rot in ordered:
        chosen: ResourceId | None = None
        for g in gates:
            if not compatibility.get((rot.id, g), False):
                continue
            if not _has_conflict(rot.arr, rot.dep, intervals[g], t_min):
                chosen = g
                break
        if chosen is None:
            chosen = stand_id
        else:
            intervals[chosen].append((rot.arr, rot.dep))
        plan[rot.id] = chosen
    return plan


def greedy_best_fit(
    rotations: Sequence[Rotation],
    gates: Sequence[GateId],
    compatibility: Mapping[tuple[RotationId, GateId], bool],
    t_min: float,
    *,
    stand_id: ResourceId = 0,
    berth_lengths: Mapping[GateId, float] | None = None,
) -> dict[RotationId, ResourceId]:
    """Assign each rotation to the compatible free berth minimizing wasted LOA.

    Wasted LOA = berth_length - vessel_length (or just index if berth_lengths
    is None, which falls back to first-fit tie-breaking).
    """
    ordered = sorted(rotations, key=lambda r: (r.arr, str(r.id)))
    intervals: dict[GateId, list[tuple[float, float]]] = {g: [] for g in gates}
    plan: dict[RotationId, ResourceId] = {}

    for rot in ordered:
        candidates = []
        for g in gates:
            if not compatibility.get((rot.id, g), False):
                continue
            if _has_conflict(rot.arr, rot.dep, intervals[g], t_min):
                continue
            waste = 0.0
            if berth_lengths is not None:
                waste = berth_lengths.get(g, 0.0) - getattr(rot, "length", 0.0)
            candidates.append((waste, str(g), g))
        if candidates:
            candidates.sort()
            chosen = candidates[0][2]
        else:
            chosen = stand_id
        if chosen != stand_id:
            intervals[chosen].append((rot.arr, rot.dep))
        plan[rot.id] = chosen
    return plan


def deterministic_milp(
    reduced,
    resources: Sequence[ResourceId],
    t_min: float,
    K: int,
    stress_sets: Mapping[WaveId, frozenset[RotationId]],
    *,
    time_limit_ms: int = 600_000,
) -> dict[RotationId, ResourceId]:
    """Solve the joint MILP on a single nominal scenario (no closures, no delays).

    Returns the resulting x plan, which is then re-evaluated under the
    real scenarios.
    """
    from joint_milp import solve_joint_milp

    # Single nominal scenario: no closure, no delay, no alert
    nominal = Scenario(
        id="nominal_baseline",
        gamma_g=1,
        closed_gates=frozenset(),
        wave=None,
        delta=0.0,
        beta=0.0,
        family="nominal",
    )
    result = solve_joint_milp(
        reduced,
        [nominal],
        stress_sets,
        resources,
        t_min,
        K,
        time_limit_ms=time_limit_ms,
    )
    if result.nominal_plan is None:
        raise RuntimeError(
            f"Deterministic MILP did not return a plan: {result.status}"
        )
    return dict(result.nominal_plan)