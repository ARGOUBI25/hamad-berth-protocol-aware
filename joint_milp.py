"""Joint first-stage/recourse MILP on the reduced instance (§2.8)."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from ortools.linear_solver import pywraplp

from gate_model import (
    GateId, PassengerFlow, ReducedInstance, ResourceId, Rotation,
    RotationId, Scenario, WaveId,
    _has_conflict, _resolve_stand_mode, _schedule_for_scenario, _walk_time,
)
from recourse_milp import _status_name


# SCIP deterministic parameters (same block in both solvers).
# Forcing a fixed seed and single thread makes the joint MILP return
# the same x* across runs, which is required for reproducibility in
# the sensitivity analysis.
_SCIP_DETERMINISTIC_PARAMS = (
    "randomization/randomseedshift = 1\n"
    "parallel/maxnthreads = 1\n"
)


@dataclass(frozen=True)
class JointMilpResult:
    status: str
    J3_star: int | None
    J1: int | None
    J2: int | None
    best_bound: float | None
    gap: float | None
    nominal_plan: Mapping[RotationId, ResourceId] | None
    recourse_per_scenario: Mapping[str, Mapping[RotationId, ResourceId]] | None
    phase_reached: int
    notes: tuple[str, ...]


def solve_joint_milp(
    reduced: ReducedInstance,
    scenarios: Sequence[Scenario],
    stress_sets: Mapping[WaveId, frozenset[RotationId]],
    resources: Sequence[ResourceId],
    t_min: float,
    K: int,
    *,
    finite_stands: Sequence[ResourceId] | None = None,
    time_limit_ms: int = 600_000,
    mip_gap: float = 0.0,
) -> JointMilpResult:
    if not scenarios:
        raise ValueError("At least one scenario is required")
    if K < 0:
        raise ValueError("K must be non-negative")

    rotations = reduced.rotations
    flows = reduced.passenger_flows
    gates = reduced.gates
    compatibility = reduced.compatibility
    walking_times = reduced.walking_times
    stand_id = reduced.stand_id

    ordered_stands, _, unlimited = _resolve_stand_mode(gates, stand_id, finite_stands)
    gates_set = set(gates)
    resources_set = set(resources)
    for g in gates:
        if g not in resources_set:
            raise ValueError(f"Resource set missing gate {g!r}")
    for s in (ordered_stands if finite_stands is not None else [stand_id]):
        if s not in resources_set:
            raise ValueError(f"Resource set missing stand {s!r}")

    rotation_ids = [r.id for r in rotations]
    M = len(rotation_ids)
    by_id = {r.id: r for r in rotations}

    def allowed(rid, a):
        if a == stand_id:
            return True
        if a in gates_set:
            return bool(compatibility.get((rid, a), False))
        return bool(compatibility.get((rid, a), True))

    compatible = {rid: [a for a in resources if allowed(rid, a)] for rid in rotation_ids}
    for rid, comp in compatible.items():
        if not comp:
            raise ValueError(f"Rotation {rid!r} has no compatible resource")

    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("SCIP backend unavailable")
    # FIX: force deterministic SCIP behaviour (same plan across runs)
    try:
        solver.SetSolverSpecificParametersAsString(_SCIP_DETERMINISTIC_PARAMS)
    except Exception:
        pass
    solver.SetTimeLimit(time_limit_ms)
    params = pywraplp.MPSolverParameters()
    params.SetDoubleParam(params.RELATIVE_MIP_GAP, mip_gap)

    x = {}
    for rid in rotation_ids:
        for a in compatible[rid]:
            x[(rid, a)] = solver.BoolVar(f"x[{rid},{a}]")

    y, z = {}, {}
    for scenario in scenarios:
        xi = scenario.id
        closed = set(scenario.closed_gates)
        for rid in rotation_ids:
            for a in compatible[rid]:
                if a in closed:
                    continue
                y[(rid, a, xi)] = solver.BoolVar(f"y[{rid},{a},{xi}]")
                z[(rid, a, xi)] = solver.BoolVar(f"z[{rid},{a},{xi}]")

    for rid in rotation_ids:
        solver.Add(solver.Sum(x[(rid, a)] for a in compatible[rid]) == 1)

    ids_sorted = sorted(rotation_ids, key=str)
    for a in resources:
        if a in unlimited:
            continue
        for idx_i, rid_i in enumerate(ids_sorted):
            if (rid_i, a) not in x:
                continue
            ri = by_id[rid_i]
            for rid_k in ids_sorted[idx_i + 1:]:
                if (rid_k, a) not in x:
                    continue
                rk = by_id[rid_k]
                if _has_conflict(ri.arr, ri.dep, [(rk.arr, rk.dep)], t_min):
                    solver.Add(x[(rid_i, a)] + x[(rid_k, a)] <= 1)

    for scenario in scenarios:
        xi = scenario.id
        schedule = _schedule_for_scenario(rotations, scenario, stress_sets)
        t_min_scenario = t_min + scenario.beta
        closed = set(scenario.closed_gates)
        active = [a for a in resources if a not in closed]

        for rid in rotation_ids:
            active_compat = [a for a in compatible[rid] if a in active]
            if not active_compat:
                raise ValueError(
                    f"Scenario {xi!r}: rotation {rid!r} has no compatible open resource"
                )
            solver.Add(solver.Sum(y[(rid, a, xi)] for a in active_compat) == 1)

        for a in resources:
            if a in unlimited or a in closed:
                continue
            for idx_i, rid_i in enumerate(ids_sorted):
                if (rid_i, a, xi) not in y:
                    continue
                arr_i, dep_i = schedule[rid_i]
                for rid_k in ids_sorted[idx_i + 1:]:
                    if (rid_k, a, xi) not in y:
                        continue
                    arr_k, dep_k = schedule[rid_k]
                    if _has_conflict(arr_i, dep_i, [(arr_k, dep_k)], t_min_scenario):
                        solver.Add(y[(rid_i, a, xi)] + y[(rid_k, a, xi)] <= 1)

        z_vars = [z[(rid, a, xi)] for rid in rotation_ids for a in compatible[rid] if a not in closed]
        solver.Add(solver.Sum(z_vars) >= M - K)
        for rid in rotation_ids:
            for a in compatible[rid]:
                if a in closed:
                    continue
                zv, xv, yv = z[(rid, a, xi)], x[(rid, a)], y[(rid, a, xi)]
                solver.Add(zv <= xv)
                solver.Add(zv <= yv)
                solver.Add(zv >= xv + yv - 1)

    L_terms: dict[str, list] = {}
    for scenario in scenarios:
        xi = scenario.id
        schedule = _schedule_for_scenario(rotations, scenario, stress_sets)
        closed = set(scenario.closed_gates)
        active = [a for a in resources if a not in closed]
        terms = []
        for flow in flows:
            T_l = schedule[flow.dep_rot][1] - schedule[flow.arr_rot][0]
            for a in active:
                if (flow.arr_rot, a, xi) not in y:
                    continue
                for b in active:
                    if (flow.dep_rot, b, xi) not in y:
                        continue
                    w_ab = _walk_time(walking_times, a, b)
                    if T_l >= flow.min_conn + w_ab:
                        continue
                    eta = solver.BoolVar(f"eta[{flow.id},{a},{b},{xi}]")
                    ya, yd = y[(flow.arr_rot, a, xi)], y[(flow.dep_rot, b, xi)]
                    solver.Add(eta <= ya)
                    solver.Add(eta <= yd)
                    solver.Add(eta >= ya + yd - 1)
                    terms.append((eta, float(flow.n)))
        L_terms[xi] = terms

    z_max = solver.NumVar(0, solver.infinity(), "z_max")
    for scenario in scenarios:
        xi = scenario.id
        if L_terms[xi]:
            solver.Add(z_max >= solver.Sum(c * v for v, c in L_terms[xi]))

    obj = solver.Objective()
    obj.SetCoefficient(z_max, 1.0)
    obj.SetMinimization()
    status1 = solver.Solve(params)
    notes: list[str] = []

    if status1 == pywraplp.Solver.INFEASIBLE:
        return JointMilpResult(
            status="INFEASIBLE", J3_star=None, J1=None, J2=None,
            best_bound=None, gap=None,
            nominal_plan=None, recourse_per_scenario=None,
            phase_reached=1,
            notes=("Phase 1 infeasible: no robust plan on the reduced instance",),
        )
    if status1 not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return JointMilpResult(
            status=_status_name(status1), J3_star=None, J1=None, J2=None,
            best_bound=None, gap=None,
            nominal_plan=None, recourse_per_scenario=None,
            phase_reached=1,
            notes=(f"Phase 1 failed: {_status_name(status1)}",),
        )

    z_star_raw = z_max.solution_value()
    if status1 == pywraplp.Solver.FEASIBLE:
        bound1 = obj.BestBound()
        gap1 = abs(z_star_raw - bound1) / max(1.0, abs(z_star_raw))
        notes.append(
            f"Phase 1 not proven optimal: best={z_star_raw}, bound={bound1}, gap={gap1:.2e}"
        )
        return _extract(
            solver, x, y, gates, scenarios, rotation_ids, resources,
            J3_star=z_star_raw, J1=None, J2=None,
            status="FEASIBLE", best_bound=bound1, gap=gap1,
            phase_reached=1, notes=tuple(notes),
        )

    z_star = int(round(z_star_raw))
    solver.Add(z_max <= float(z_star))
    J1_vars = [x[(rid, a)] for rid in rotation_ids for a in compatible[rid] if a in gates_set]
    obj2 = solver.Objective()
    obj2.Clear()
    if J1_vars:
        for v in J1_vars:
            obj2.SetCoefficient(v, 1.0)
        obj2.SetMaximization()
        status2 = solver.Solve(params)
    else:
        status2 = pywraplp.Solver.OPTIMAL

    if J1_vars and status2 not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        notes.append(f"Phase 2 failed: {_status_name(status2)}")
        return _extract(
            solver, x, y, gates, scenarios, rotation_ids, resources,
            J3_star=z_star, J1=None, J2=None,
            status=_status_name(status2), best_bound=None, gap=None,
            phase_reached=1, notes=tuple(notes),
        )
    if J1_vars and status2 == pywraplp.Solver.FEASIBLE:
        J1_best = int(round(obj2.Value()))
        bound2 = obj2.BestBound()
        gap2 = abs(J1_best - bound2) / max(1.0, abs(J1_best))
        notes.append(
            f"Phase 2 not proven optimal: best J1={J1_best}, bound={bound2}, gap={gap2:.2e}"
        )
        return _extract(
            solver, x, y, gates, scenarios, rotation_ids, resources,
            J3_star=z_star, J1=J1_best, J2=None,
            status="FEASIBLE", best_bound=bound2, gap=gap2,
            phase_reached=2, notes=tuple(notes),
        )

    J1_star = int(round(obj2.Value())) if J1_vars else 0
    if J1_vars:
        solver.Add(solver.Sum(J1_vars) >= float(J1_star))
    gate_used = {}
    for g in gates:
        gate_used[g] = solver.BoolVar(f"gate_used[{g}]")
        for rid in rotation_ids:
            if (rid, g) in x:
                solver.Add(gate_used[g] >= x[(rid, g)])
    obj3 = solver.Objective()
    obj3.Clear()
    for g in gates:
        obj3.SetCoefficient(gate_used[g], 1.0)
    obj3.SetMinimization()
    status3 = solver.Solve(params)

    if status3 not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        notes.append(f"Phase 3 failed: {_status_name(status3)}")
        return _extract(
            solver, x, y, gates, scenarios, rotation_ids, resources,
            J3_star=z_star, J1=J1_star, J2=None,
            status=_status_name(status3), best_bound=None, gap=None,
            phase_reached=2, notes=tuple(notes),
        )

    J2_best = int(round(obj3.Value()))
    final_status = _status_name(status3)
    final_bound = obj3.BestBound() if status3 == pywraplp.Solver.FEASIBLE else float(J2_best)
    final_gap = (
        abs(J2_best - final_bound) / max(1.0, abs(J2_best))
        if status3 == pywraplp.Solver.FEASIBLE else 0.0
    )
    if status3 == pywraplp.Solver.FEASIBLE:
        notes.append(
            f"Phase 3 not proven optimal: best J2={J2_best}, bound={final_bound}, gap={final_gap:.2e}"
        )
    return _extract(
        solver, x, y, gates, scenarios, rotation_ids, resources,
        J3_star=z_star, J1=J1_star, J2=J2_best,
        status=final_status, best_bound=final_bound, gap=final_gap,
        phase_reached=3, notes=tuple(notes),
    )


def _extract(solver, x, y, gates, scenarios, rotation_ids, resources, *,
             J3_star, J1, J2, status, best_bound, gap, phase_reached, notes):
    gates_set = set(gates)
    nominal_plan = {}
    for rid in rotation_ids:
        for a in resources:
            if (rid, a) in x and x[(rid, a)].solution_value() > 0.5:
                nominal_plan[rid] = a
                break
    if len(nominal_plan) != len(rotation_ids):
        notes = tuple(notes) + ("WARNING: solver returned incomplete solution",)

    recourse = {}
    for scenario in scenarios:
        xi = scenario.id
        plan = {}
        for rid in rotation_ids:
            for a in resources:
                if (rid, a, xi) in y and y[(rid, a, xi)].solution_value() > 0.5:
                    plan[rid] = a
                    break
        recourse[xi] = plan

    if J1 is None or J2 is None:
        on_gates = [a for a in nominal_plan.values() if a in gates_set]
        if J1 is None:
            J1 = len(on_gates)
        if J2 is None:
            J2 = len(set(on_gates))

    return JointMilpResult(
        status=status,
        J3_star=int(round(J3_star)) if J3_star is not None else None,
        J1=int(round(J1)) if J1 is not None else None,
        J2=int(round(J2)) if J2 is not None else None,
        best_bound=best_bound, gap=gap,
        nominal_plan=MappingProxyType(nominal_plan) if nominal_plan else None,
        recourse_per_scenario=MappingProxyType(recourse) if recourse else None,
        phase_reached=phase_reached, notes=notes,
    )