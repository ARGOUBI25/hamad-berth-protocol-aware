"""Exact recourse MILP per (x, xi) and dual evaluation π vs MILP (§2.5, §2.6, §2.8).

Solver: OR-Tools MPSolver, SCIP backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

from ortools.linear_solver import pywraplp

from gate_model import (
    GateId, PassengerFlow, ProtocolResult, ResourceId, Rotation,
    RotationId, Scenario, WaveId,
    _has_conflict, _resolve_stand_mode, _schedule_for_scenario,
    operational_protocol, validate_nominal_plan,
)


# SCIP deterministic parameters (same block as in joint_milp.py).
_SCIP_DETERMINISTIC_PARAMS = (
    "randomization/randomseedshift = 1\n"
    "parallel/maxnthreads = 1\n"
)


def _status_name(status: int) -> str:
    return {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }.get(status, str(status))


def _walk_time(walking_times, origin, destination) -> float:
    try:
        return float(walking_times[(origin, destination)])
    except KeyError as exc:
        raise ValueError(
            f"Missing directed walking time w({origin!r}, {destination!r})"
        ) from exc


def _resource_allowed(rid, resource, compatibility, stand_id, gates_set):
    if resource == stand_id:
        return True
    if resource in gates_set:
        return bool(compatibility.get((rid, resource), False))
    return bool(compatibility.get((rid, resource), True))


def solve_recourse_milp(
    nominal_assignment: Mapping[RotationId, ResourceId],
    scenario: Scenario,
    rotations: Sequence[Rotation],
    passenger_flows: Sequence[PassengerFlow],
    gates: Sequence[GateId],
    compatibility: Mapping[tuple[RotationId, GateId], bool],
    walking_times: Mapping[tuple[ResourceId, ResourceId], float],
    stress_sets: Mapping[WaveId, frozenset[RotationId]],
    resources: Sequence[ResourceId],
    t_min: float,
    K: int,
    *,
    stand_id: ResourceId = 0,
    finite_stands: Sequence[ResourceId] | None = None,
    time_limit_ms: int = 60_000,
    mip_gap: float = 0.0,
) -> ProtocolResult:
    validate_nominal_plan(
        nominal_assignment, rotations, gates, compatibility, t_min,
        stand_id=stand_id, finite_stands=finite_stands,
    )
    schedule = _schedule_for_scenario(rotations, scenario, stress_sets)
    t_min_scenario = t_min + scenario.beta
    gates_set = set(gates)
    _, _, unlimited = _resolve_stand_mode(gates, stand_id, finite_stands)
    closed = set(scenario.closed_gates)
    active_resources = [r for r in resources if r not in closed]
    rotation_ids = [r.id for r in rotations]

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("SCIP backend unavailable in this OR-Tools build")
    # FIX: force deterministic SCIP behaviour (same y across runs)
    try:
        solver.SetSolverSpecificParametersAsString(_SCIP_DETERMINISTIC_PARAMS)
    except Exception:
        pass
    solver.SetTimeLimit(time_limit_ms)
    params = pywraplp.MPSolverParameters()
    params.SetDoubleParam(params.RELATIVE_MIP_GAP, mip_gap)

    y: dict[tuple[RotationId, ResourceId], pywraplp.Variable] = {}
    for rid in rotation_ids:
        for a in resources:
            y[(rid, a)] = solver.BoolVar(f"y[{rid},{a}]")

    for rid in rotation_ids:
        solver.Add(solver.Sum(y[(rid, a)] for a in resources) == 1)
        for a in resources:
            if not _resource_allowed(rid, a, compatibility, stand_id, gates_set):
                solver.Add(y[(rid, a)] == 0)
        for j in closed:
            if (rid, j) in y:
                solver.Add(y[(rid, j)] == 0)

    ids_sorted = sorted(rotation_ids, key=str)
    for a in active_resources:
        if a in unlimited:
            continue
        for idx_i, rid_i in enumerate(ids_sorted):
            arr_i, dep_i = schedule[rid_i]
            for rid_k in ids_sorted[idx_i + 1:]:
                arr_k, dep_k = schedule[rid_k]
                if _has_conflict(arr_i, dep_i, [(arr_k, dep_k)], t_min_scenario):
                    solver.Add(y[(rid_i, a)] + y[(rid_k, a)] <= 1)

    solver.Add(
        solver.Sum(y[(rid, nominal_assignment[rid])] for rid in rotation_ids)
        >= len(rotation_ids) - K
    )

    objective = solver.Objective()
    for flow in passenger_flows:
        for a in active_resources:
            for b in active_resources:
                T_l = schedule[flow.dep_rot][1] - schedule[flow.arr_rot][0]
                w_ab = _walk_time(walking_times, a, b)
                if T_l >= flow.min_conn + w_ab:
                    continue
                eta = solver.BoolVar(f"eta[{flow.id},{a},{b}]")
                y_a = y[(flow.arr_rot, a)]
                y_d = y[(flow.dep_rot, b)]
                solver.Add(eta <= y_a)
                solver.Add(eta <= y_d)
                solver.Add(eta >= y_a + y_d - 1)
                objective.SetCoefficient(eta, float(flow.n))
    objective.SetMinimization()

    status = solver.Solve(params)
    if status == pywraplp.Solver.INFEASIBLE:
        return ProtocolResult(
            False, None, 0, None,
            f"MILP infeasible for scenario {scenario.id}: no recourse within K={K}",
        )
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return ProtocolResult(
            False, None, 0, None,
            f"MILP status {_status_name(status)} for scenario {scenario.id}",
        )

    assignment: dict[RotationId, ResourceId] = {}
    for rid in rotation_ids:
        for a in resources:
            if y[(rid, a)].solution_value() > 0.5:
                assignment[rid] = a
                break
    modifications = sum(
        1 for rid in rotation_ids
        if assignment[rid] != nominal_assignment[rid]
    )
    if modifications > K:
        raise AssertionError(
            f"MILP returned {modifications} modifications > K={K}"
        )
    return ProtocolResult(
        True, assignment, modifications,
        int(round(objective.Value())),
        f"status={_status_name(status)}",
    )


# =========================================================================
# Dual recourse evaluation (§2.6)
# =========================================================================

@dataclass(frozen=True)
class ScenarioDualResult:
    scenario_id: str
    family: str
    pi_feasible: bool
    star_feasible: bool
    L_pi: int | None
    L_star: int | None
    pi_modifications: int | None
    star_modifications: int | None
    pi_reason: str | None
    star_reason: str | None


@dataclass(frozen=True)
class DualRecourseEvaluation:
    x_is_pi_admissible: bool
    x_is_star_admissible: bool
    I_pi: int
    J3_pi: int | None
    J3_star: int | None
    delta_rob: int | None
    delta_sc: int | None
    per_scenario: Mapping[str, ScenarioDualResult]
    inconsistency_warnings: tuple[str, ...]


def evaluate_dual_recourse(
    nominal_assignment: Mapping[RotationId, ResourceId],
    scenarios: Sequence[Scenario],
    rotations: Sequence[Rotation],
    passenger_flows: Sequence[PassengerFlow],
    gates: Sequence[GateId],
    compatibility: Mapping[tuple[RotationId, GateId], bool],
    walking_times: Mapping[tuple[ResourceId, ResourceId], float],
    q_scores: Mapping[RotationId, int],
    stress_sets: Mapping[WaveId, frozenset[RotationId]],
    resources: Sequence[ResourceId],
    t_min: float,
    K: int,
    *,
    stand_id: ResourceId = 0,
    finite_stands: Sequence[ResourceId] | None = None,
    time_limit_ms: int = 60_000,
    mip_gap: float = 0.0,
) -> DualRecourseEvaluation:
    per_scenario: dict[str, ScenarioDualResult] = {}
    inconsistencies: list[str] = []
    pi_failures = 0

    for scenario in scenarios:
        pi_result = operational_protocol(
            nominal_assignment, scenario, rotations, passenger_flows, gates,
            compatibility, walking_times, q_scores, stress_sets, t_min, K,
            stand_id=stand_id, finite_stands=finite_stands,
        )
        milp_result = solve_recourse_milp(
            nominal_assignment, scenario, rotations, passenger_flows, gates,
            compatibility, walking_times, stress_sets, resources, t_min, K,
            stand_id=stand_id, finite_stands=finite_stands,
            time_limit_ms=time_limit_ms, mip_gap=mip_gap,
        )
        if not pi_result.feasible:
            pi_failures += 1
        if pi_result.feasible and not milp_result.feasible:
            inconsistencies.append(
                f"I1 violated on {scenario.id!r}: π feasible but MILP infeasible"
            )
        if pi_result.feasible and milp_result.feasible:
            L_pi = pi_result.broken_passengers or 0
            L_star = milp_result.broken_passengers or 0
            if L_pi < L_star:
                inconsistencies.append(
                    f"I2 violated on {scenario.id!r}: L_pi={L_pi} < L_star={L_star}"
                )
        per_scenario[scenario.id] = ScenarioDualResult(
            scenario_id=scenario.id,
            family=scenario.family,
            pi_feasible=pi_result.feasible,
            star_feasible=milp_result.feasible,
            L_pi=pi_result.broken_passengers if pi_result.feasible else None,
            L_star=milp_result.broken_passengers if milp_result.feasible else None,
            pi_modifications=pi_result.modifications if pi_result.feasible else None,
            star_modifications=(
                milp_result.modifications if milp_result.feasible else None
            ),
            pi_reason=pi_result.reason,
            star_reason=milp_result.reason,
        )

    x_is_pi_admissible = pi_failures == 0
    x_is_star_admissible = all(r.star_feasible for r in per_scenario.values())
    J3_pi = (
        max(r.L_pi for r in per_scenario.values() if r.L_pi is not None)
        if x_is_pi_admissible else None
    )
    J3_star = (
        max(r.L_star for r in per_scenario.values() if r.L_star is not None)
        if x_is_star_admissible else None
    )
    delta_rob = None
    delta_sc = None
    if x_is_pi_admissible and x_is_star_admissible:
        delta_rob = J3_pi - J3_star
        if delta_rob < 0:
            inconsistencies.append(f"I3 violated: delta_rob={delta_rob} < 0")
        deltas = [
            r.L_pi - r.L_star for r in per_scenario.values()
            if r.L_pi is not None and r.L_star is not None
        ]
        delta_sc = max(deltas) if deltas else 0
        if delta_sc < 0:
            inconsistencies.append(f"I4 violated: delta_sc={delta_sc} < 0")

    return DualRecourseEvaluation(
        x_is_pi_admissible=x_is_pi_admissible,
        x_is_star_admissible=x_is_star_admissible,
        I_pi=pi_failures,
        J3_pi=J3_pi,
        J3_star=J3_star,
        delta_rob=delta_rob,
        delta_sc=delta_sc,
        per_scenario=per_scenario,
        inconsistency_warnings=tuple(inconsistencies),
    )