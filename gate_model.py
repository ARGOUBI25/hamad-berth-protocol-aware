from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import total_ordering
from math import floor
from types import MappingProxyType
from typing import Any, Hashable, Iterable, Mapping, Sequence

RotationId = Hashable
GateId = Hashable
WaveId = Hashable
ResourceId = Hashable
VarKey = tuple


# =========================================================================
# Data structures
# =========================================================================

@dataclass(frozen=True)
class Rotation:
    id: RotationId
    arr: float
    dep: float
    gabarit: Hashable | None = None
    wave: WaveId | None = None

    def __post_init__(self) -> None:
        if self.dep <= self.arr:
            raise ValueError(f"Rotation {self.id!r}: dep must be greater than arr")

    @property
    def duration(self) -> float:
        return self.dep - self.arr

    @property
    def mid(self) -> float:
        return (self.arr + self.dep) / 2.0


@dataclass(frozen=True)
class PassengerFlow:
    id: Hashable
    arr_rot: RotationId
    dep_rot: RotationId
    n: int
    min_conn: float

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"Passenger flow {self.id!r}: n must be >= 1")
        if self.min_conn < 0:
            raise ValueError(f"Passenger flow {self.id!r}: min_conn must be >= 0")
        if self.arr_rot == self.dep_rot:
            raise ValueError(
                f"Passenger flow {self.id!r}: arrival and departure rotations must differ"
            )


@dataclass(frozen=True)
class Scenario:
    id: str
    gamma_g: int
    closed_gates: frozenset[GateId]
    wave: WaveId | None
    delta: float
    beta: float
    family: str


@dataclass(frozen=True)
class ProtocolResult:
    feasible: bool
    assignment: Mapping[RotationId, ResourceId] | None
    modifications: int
    broken_passengers: int | None
    reason: str | None = None


@dataclass(frozen=True)
class RobustEvaluation:
    admissible: bool
    worst_case_broken_passengers: int | None
    scenario_results: Mapping[str, ProtocolResult]


@dataclass(frozen=True)
class LinearConstraint:
    coeffs: tuple[tuple[VarKey, float], ...]
    sense: str  # "<=", ">=", "=="
    rhs: float
    label: str = ""


# =========================================================================
# Helpers
# =========================================================================

def _stable_key(value: Hashable) -> tuple[int, Any, str]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, float(value), type(value).__name__)
    return (1, str(value), type(value).__name__)


def _gate_order(
    gates: Iterable[GateId],
    gate_priority: Mapping[GateId, int] | None = None,
) -> list[GateId]:
    gates_list = list(gates)
    if gate_priority is not None:
        missing = [g for g in gates_list if g not in gate_priority]
        if missing:
            raise ValueError(f"gate_priority is missing gate IDs: {missing!r}")
        return sorted(gates_list, key=lambda g: (gate_priority[g], _stable_key(g)))
    return sorted(gates_list, key=_stable_key)


def _resolve_stand_mode(
    gates: Sequence[GateId],
    stand_id: ResourceId,
    finite_stands: Sequence[ResourceId] | None,
) -> tuple[list[ResourceId], set[ResourceId], set[ResourceId]]:
    """Resolve the two mutually exclusive stand models (§2.2, §2.5).

    Returns (ordered_stands, conflict_checked_resources, unlimited_resources).
    """
    if finite_stands is None:
        return [], set(gates), {stand_id}
    finite_list = list(finite_stands)
    if len(set(finite_list)) != len(finite_list):
        raise ValueError("finite_stands must not contain duplicate IDs")
    finite_set = set(finite_list)
    if stand_id in finite_set:
        raise ValueError(
            f"stand_id={stand_id!r} must not appear in finite_stands"
        )
    if finite_set & set(gates):
        raise ValueError("finite_stands must not overlap with gates")
    ordered_stands = sorted(finite_list, key=_stable_key)
    return ordered_stands, set(gates) | finite_set, set()


def _warn_if_aggregated(
    finite_stands: Sequence[ResourceId] | None,
    stand_id: ResourceId,
) -> None:
    if finite_stands is None:
        warnings.warn(
            f"Aggregated infinite stand model in use (stand_id={stand_id!r}). "
            f"This is a simplification, not the finite-capacity formulation of §2.2. "
            f"Pass finite_stands=[...] and an independent capacity audit before "
            f"drawing conclusions on stand capacity.",
            RuntimeWarning,
            stacklevel=3,
        )


def _has_conflict(
    arr: float,
    dep: float,
    other_intervals: Sequence[tuple[float, float]],
    t_min: float,
) -> bool:
    for other_arr, other_dep in other_intervals:
        separated_before = dep + t_min <= other_arr
        separated_after = arr >= other_dep + t_min
        if not (separated_before or separated_after):
            return True
    return False


def _walk_time(
    walking_times: Mapping[tuple[ResourceId, ResourceId], float],
    origin: ResourceId,
    destination: ResourceId,
) -> float:
    try:
        value = walking_times[(origin, destination)]
    except KeyError as exc:
        raise ValueError(
            f"Missing directed walking time w({origin!r}, {destination!r})"
        ) from exc
    if value < 0:
        raise ValueError(
            f"Walking time w({origin!r}, {destination!r}) must be non-negative"
        )
    return float(value)


# =========================================================================
# Validation
# =========================================================================

def validate_inputs(
    rotations: Sequence[Rotation],
    passenger_flows: Sequence[PassengerFlow],
    gates: Sequence[GateId],
    stand_id: ResourceId = 0,
) -> None:
    rotation_ids = [r.id for r in rotations]
    if len(set(rotation_ids)) != len(rotation_ids):
        raise ValueError("Rotation IDs must be unique")
    if len(set(gates)) != len(gates):
        raise ValueError("Gate IDs must be unique")
    if stand_id in set(gates):
        raise ValueError("The aggregate stand ID must not also be a fixed gate ID")
    rotation_id_set = set(rotation_ids)
    flow_ids = [p.id for p in passenger_flows]
    if len(set(flow_ids)) != len(flow_ids):
        raise ValueError("Passenger-flow IDs must be unique")
    for flow in passenger_flows:
        if flow.arr_rot not in rotation_id_set or flow.dep_rot not in rotation_id_set:
            raise ValueError(
                f"Passenger flow {flow.id!r} refers to an unknown rotation"
            )


def validate_nominal_plan(
    assignment: Mapping[RotationId, ResourceId],
    rotations: Sequence[Rotation],
    gates: Sequence[GateId],
    compatibility: Mapping[tuple[RotationId, GateId], bool],
    t_min: float,
    *,
    stand_id: ResourceId = 0,
    finite_stands: Sequence[ResourceId] | None = None,
) -> None:
    _, conflict_resources, unlimited = _resolve_stand_mode(
        gates, stand_id, finite_stands
    )
    by_id = {r.id: r for r in rotations}
    if set(assignment) != set(by_id):
        raise ValueError("Every rotation must receive exactly one nominal assignment")
    intervals: dict[ResourceId, list[tuple[RotationId, float, float]]] = {
        r: [] for r in conflict_resources
    }
    for rid, resource in assignment.items():
        if resource in unlimited:
            continue
        if resource not in conflict_resources:
            raise ValueError(f"Rotation {rid!r} assigned to unknown resource {resource!r}")
        if not compatibility.get((rid, resource), False):
            raise ValueError(
                f"Incompatible nominal assignment: rotation {rid!r}, resource {resource!r}"
            )
        rot = by_id[rid]
        for other_id, other_arr, other_dep in intervals[resource]:
            if _has_conflict(rot.arr, rot.dep, [(other_arr, other_dep)], t_min):
                raise ValueError(
                    f"Nominal temporal conflict on resource {resource!r}: "
                    f"{rid!r} and {other_id!r}"
                )
        intervals[resource].append((rid, rot.arr, rot.dep))


# =========================================================================
# Core computations
# =========================================================================

def compute_q_scores(
    rotations: Sequence[Rotation],
    passenger_flows: Sequence[PassengerFlow],
) -> dict[RotationId, int]:
    q = {r.id: 0 for r in rotations}
    for flow in passenger_flows:
        q[flow.arr_rot] += flow.n
        q[flow.dep_rot] += flow.n
    return q


def build_stress_sets(
    rotations: Sequence[Rotation],
    q_scores: Mapping[RotationId, int],
) -> tuple[int, dict[WaveId, frozenset[RotationId]]]:
    by_wave: dict[WaveId, list[Rotation]] = {}
    for rot in rotations:
        if rot.wave is not None:
            by_wave.setdefault(rot.wave, []).append(rot)
    if not by_wave or max(len(items) for items in by_wave.values()) == 0:
        raise ValueError("At least one non-empty arrival wave is required")
    gamma_f = max(1, floor(max(len(items) for items in by_wave.values()) / 2))
    stress_sets: dict[WaveId, frozenset[RotationId]] = {}
    for wave, items in by_wave.items():
        count = min(gamma_f, len(items))
        chosen = sorted(
            items, key=lambda r: (-q_scores.get(r.id, 0), _stable_key(r.id))
        )[:count]
        stress_sets[wave] = frozenset(r.id for r in chosen)
    return gamma_f, stress_sets


def build_reference_assignment(
    rotations: Sequence[Rotation],
    gates: Sequence[GateId],
    compatibility: Mapping[tuple[RotationId, GateId], bool],
    t_min: float,
    *,
    stand_id: ResourceId = 0,
    gate_priority: Mapping[GateId, int] | None = None,
    finite_stands: Sequence[ResourceId] | None = None,
) -> dict[RotationId, ResourceId]:
    _warn_if_aggregated(finite_stands, stand_id)
    ordered_stands, conflict_resources, unlimited = _resolve_stand_mode(
        gates, stand_id, finite_stands
    )
    ordered_gates = _gate_order(gates, gate_priority)
    assigned: dict[RotationId, ResourceId] = {}
    intervals: dict[ResourceId, list[tuple[float, float]]] = {
        r: [] for r in conflict_resources
    }
    for rot in sorted(rotations, key=lambda r: (r.arr, _stable_key(r.id))):
        chosen: ResourceId | None = None
        for gate in ordered_gates:
            if not compatibility.get((rot.id, gate), False):
                continue
            if not _has_conflict(rot.arr, rot.dep, intervals[gate], t_min):
                chosen = gate
                break
        if chosen is None:
            for stand in ordered_stands:
                if not compatibility.get((rot.id, stand), False):
                    continue
                if not _has_conflict(rot.arr, rot.dep, intervals[stand], t_min):
                    chosen = stand
                    break
        if chosen is None:
            if unlimited:
                chosen = stand_id
            else:
                raise ValueError(
                    f"build_reference_assignment: no finite resource for "
                    f"rotation {rot.id!r}"
                )
        assigned[rot.id] = chosen
        if chosen in conflict_resources:
            intervals[chosen].append((rot.arr, rot.dep))
    return assigned


def _door_stress_scores(
    gates: Sequence[GateId],
    reference_assignment: Mapping[RotationId, ResourceId],
    passenger_flows: Sequence[PassengerFlow],
) -> dict[GateId, int]:
    """s_j° = Σ_l n_l · 1{g°_{ρ^a(l)} = j ∨ g°_{ρ^d(l)} = j} (§2.3)."""
    result = {g: 0 for g in gates}
    for flow in passenger_flows:
        arr_gate = reference_assignment.get(flow.arr_rot)
        dep_gate = reference_assignment.get(flow.dep_rot)
        if arr_gate in result:
            result[arr_gate] += flow.n
        if dep_gate in result and dep_gate != arr_gate:
            result[dep_gate] += flow.n
    return result


# =========================================================================
# Scenario generation
# =========================================================================

def generate_scenarios(
    zones_dict: Mapping[Hashable, Sequence[GateId]],
    gamma_g: int,
    delta_values: Sequence[float],
    beta_max: float,
    reference_assignment: Mapping[RotationId, ResourceId],
    rotations: Sequence[Rotation],
    passenger_flows: Sequence[PassengerFlow],
    *,
    stand_id: ResourceId = 0,
) -> list[Scenario]:
    if gamma_g < 1:
        raise ValueError("gamma_g must be >= 1")
    if beta_max < 0 or any(delta < 0 for delta in delta_values):
        raise ValueError("delta values and beta_max must be non-negative")
    deltas = sorted({float(d) for d in delta_values if d > 0})
    if not deltas:
        raise ValueError("At least one positive delay value is required")
    gates = [g for zone_gates in zones_dict.values() for g in zone_gates]
    if len(set(gates)) != len(gates):
        raise ValueError("zones_dict must be a partition")
    if stand_id in set(gates):
        raise ValueError("The stand ID must not appear among fixed gates")
    q_scores = compute_q_scores(rotations, passenger_flows)
    gamma_f, stress_sets = build_stress_sets(rotations, q_scores)
    door_scores = _door_stress_scores(gates, reference_assignment, passenger_flows)
    nonempty_closures: list[frozenset[GateId]] = []
    for zone_gates in zones_dict.values():
        ordered = sorted(
            zone_gates,
            key=lambda gate: (-door_scores.get(gate, 0), _stable_key(gate)),
        )
        n_close = min(gamma_g, len(ordered))
        closure = frozenset(ordered[:n_close])
        if closure and closure not in nonempty_closures:
            nonempty_closures.append(closure)
    waves = sorted(stress_sets, key=_stable_key)
    scenarios: list[Scenario] = []
    seen: set[tuple[frozenset[GateId], WaveId | None, float, float]] = set()

    def add(family, closed, wave, delta, beta):
        key = (closed, wave, float(delta), float(beta))
        if key in seen:
            return
        seen.add(key)
        scenarios.append(Scenario(
            id=f"sc_{len(scenarios) + 1:05d}",
            gamma_g=gamma_g,
            closed_gates=closed,
            wave=wave,
            delta=float(delta),
            beta=float(beta),
            family=family,
        ))

    for closed in nonempty_closures:
        add("closure_only", closed, None, 0, 0)
    for wave in waves:
        for delta in deltas:
            add("delay_only", frozenset(), wave, delta, 0)
    if beta_max > 0:
        add("alert_only", frozenset(), None, 0, beta_max)
    for closed in nonempty_closures:
        for wave in waves:
            for delta in deltas:
                add("closure_delay", closed, wave, delta, 0)
    if beta_max > 0:
        for closed in nonempty_closures:
            add("closure_alert", closed, None, 0, beta_max)
        for wave in waves:
            for delta in deltas:
                add("delay_alert", frozenset(), wave, delta, beta_max)
        for closed in nonempty_closures:
            for wave in waves:
                for delta in deltas:
                    add("closure_delay_alert", closed, wave, delta, beta_max)
    return scenarios


def generate_scenarios_by_gamma(
    zones_dict, gamma_g_values, delta_values, beta_max,
    reference_assignment, rotations, passenger_flows, *, stand_id=0,
):
    return {
        gamma_g: generate_scenarios(
            zones_dict, gamma_g, delta_values, beta_max,
            reference_assignment, rotations, passenger_flows,
            stand_id=stand_id,
        )
        for gamma_g in gamma_g_values
    }


# =========================================================================
# Protocol π
# =========================================================================

def _schedule_for_scenario(
    rotations: Sequence[Rotation],
    scenario: Scenario,
    stress_sets: Mapping[WaveId, frozenset[RotationId]],
) -> dict[RotationId, tuple[float, float]]:
    shifted = (
        stress_sets.get(scenario.wave, frozenset())
        if scenario.wave is not None else frozenset()
    )
    schedule: dict[RotationId, tuple[float, float]] = {}
    for rot in rotations:
        shift = scenario.delta if rot.id in shifted else 0.0
        schedule[rot.id] = (rot.arr + shift, rot.dep + shift)
    return schedule


def decode_chromosome(
    rho: Sequence[RotationId],
    sigma: Mapping[RotationId, ResourceId],
    rotations: Sequence[Rotation],
    gates: Sequence[GateId],
    compatibility: Mapping[tuple[RotationId, GateId], bool],
    q_scores: Mapping[RotationId, int],
    t_min: float,
    *,
    stand_id: ResourceId = 0,
    gate_priority: Mapping[GateId, int] | None = None,
    finite_stands: Sequence[ResourceId] | None = None,
) -> dict[RotationId, ResourceId]:
    _warn_if_aggregated(finite_stands, stand_id)
    ordered_stands, _, unlimited = _resolve_stand_mode(
        gates, stand_id, finite_stands
    )
    rotation_by_id = {r.id: r for r in rotations}
    ids = set(rotation_by_id)
    if len(rho) != len(ids) or set(rho) != ids:
        raise ValueError("rho must contain every rotation ID exactly once")
    if set(sigma) != ids:
        raise ValueError("sigma must map every rotation ID to exactly one resource")
    ordered_gates = _gate_order(gates, gate_priority)
    gates_set = set(ordered_gates)
    finite_set = set(ordered_stands)
    all_constrained = gates_set | finite_set
    intervals: dict[ResourceId, list[tuple[float, float]]] = {
        r: [] for r in all_constrained
    }
    gate_load = {g: 0 for g in ordered_gates}
    assignment: dict[RotationId, ResourceId] = {}

    def feasible_on(rot, resource):
        if resource in unlimited:
            return True
        if resource not in all_constrained:
            return False
        if not compatibility.get((rot.id, resource), False):
            return False
        return not _has_conflict(rot.arr, rot.dep, intervals[resource], t_min)

    for rot_id in rho:
        rot = rotation_by_id[rot_id]
        requested = sigma[rot_id]
        chosen: ResourceId | None = None
        if feasible_on(rot, requested):
            chosen = requested
        if chosen is None:
            cands = [g for g in ordered_gates if feasible_on(rot, g)]
            cands.sort(key=lambda g: (gate_load[g], _stable_key(g)))
            if cands:
                chosen = cands[0]
        if chosen is None:
            for stand in ordered_stands:
                if feasible_on(rot, stand):
                    chosen = stand
                    break
        if chosen is None:
            if unlimited:
                chosen = stand_id
            else:
                raise ValueError(
                    f"decode_chromosome: no finite resource for {rot_id!r}"
                )
        assignment[rot_id] = chosen
        if chosen in all_constrained:
            intervals[chosen].append((rot.arr, rot.dep))
            if chosen in gates_set:
                gate_load[chosen] += q_scores.get(rot_id, 0)
    validate_nominal_plan(
        assignment, rotations, gates, compatibility, t_min,
        stand_id=stand_id, finite_stands=finite_stands,
    )
    return assignment


def operational_protocol(
    nominal_assignment: Mapping[RotationId, ResourceId],
    scenario: Scenario,
    rotations: Sequence[Rotation],
    passenger_flows: Sequence[PassengerFlow],
    gates: Sequence[GateId],
    compatibility: Mapping[tuple[RotationId, GateId], bool],
    walking_times: Mapping[tuple[ResourceId, ResourceId], float],
    q_scores: Mapping[RotationId, int],
    stress_sets: Mapping[WaveId, frozenset[RotationId]],
    t_min: float,
    K: int,
    *,
    stand_id: ResourceId = 0,
    gate_priority: Mapping[GateId, int] | None = None,
    finite_stands: Sequence[ResourceId] | None = None,
) -> ProtocolResult:
    _warn_if_aggregated(finite_stands, stand_id)
    ordered_stands, conflict_resources, unlimited = _resolve_stand_mode(
        gates, stand_id, finite_stands
    )
    if K < 0:
        raise ValueError("K must be non-negative")
    validate_nominal_plan(
        nominal_assignment, rotations, gates, compatibility, t_min,
        stand_id=stand_id, finite_stands=finite_stands,
    )
    gates_set = set(gates)
    ordered_gates = _gate_order(gates, gate_priority)
    active_gates = [g for g in ordered_gates if g not in scenario.closed_gates]
    active_stands = list(ordered_stands)
    active_resources = active_gates + active_stands
    schedule = _schedule_for_scenario(rotations, scenario, stress_sets)
    t_min_scenario = t_min + scenario.beta
    intervals: dict[ResourceId, list[tuple[float, float]]] = {
        r: [] for r in conflict_resources
    }
    assignment: dict[RotationId, ResourceId] = {}
    modifications = 0
    order = sorted(rotations, key=lambda r: (-q_scores.get(r.id, 0), _stable_key(r.id)))

    def feasible_on(rid, resource):
        if resource in unlimited:
            return True
        if resource not in conflict_resources:
            return False
        if resource in gates_set and resource in scenario.closed_gates:
            return False
        if not compatibility.get((rid, resource), False):
            return False
        arr, dep = schedule[rid]
        return not _has_conflict(arr, dep, intervals[resource], t_min_scenario)

    for rot in order:
        rid = rot.id
        nominal = nominal_assignment[rid]
        chosen: ResourceId | None = None
        if feasible_on(rid, nominal):
            chosen = nominal
        else:
            if modifications >= K:
                return ProtocolResult(
                    False, None, modifications, None,
                    f"Budget K exhausted before reallocating rotation {rid!r}",
                )
            cands = [r for r in active_resources if r != nominal]
            cands.sort(key=lambda r: (
                _walk_time(walking_times, nominal, r), _stable_key(r),
            ))
            for res in cands:
                if feasible_on(rid, res):
                    chosen = res
                    break
            if chosen is None and unlimited and nominal != stand_id:
                chosen = stand_id
            if chosen is None:
                return ProtocolResult(
                    False, None, modifications, None,
                    f"No admissible resource for rotation {rid!r}",
                )
            if chosen != nominal:
                modifications += 1
                if modifications > K:
                    return ProtocolResult(
                        False, None, modifications - 1, None, "Budget K exceeded"
                    )
        assignment[rid] = chosen
        if chosen in conflict_resources:
            arr, dep = schedule[rid]
            intervals[chosen].append((arr, dep))
    broken = 0
    for flow in passenger_flows:
        available = schedule[flow.dep_rot][1] - schedule[flow.arr_rot][0]
        walk = _walk_time(walking_times, assignment[flow.arr_rot], assignment[flow.dep_rot])
        if available < flow.min_conn + walk:
            broken += flow.n
    return ProtocolResult(True, assignment, modifications, broken, None)


def evaluate_robust_protocol(
    nominal_assignment, scenarios, rotations, passenger_flows, gates,
    compatibility, walking_times, q_scores, stress_sets, t_min, K, *,
    stand_id=0, gate_priority=None, finite_stands=None,
):
    results: dict[str, ProtocolResult] = {}
    for scenario in scenarios:
        results[scenario.id] = operational_protocol(
            nominal_assignment, scenario, rotations, passenger_flows, gates,
            compatibility, walking_times, q_scores, stress_sets, t_min, K,
            stand_id=stand_id, gate_priority=gate_priority,
            finite_stands=finite_stands,
        )
    if not results or not all(r.feasible for r in results.values()):
        return RobustEvaluation(False, None, results)
    return RobustEvaluation(
        True,
        max(r.broken_passengers or 0 for r in results.values()),
        results,
    )


# =========================================================================
# J1 / J2 / lexicographic ranking (§2.4, §2.6, §2.7)
# =========================================================================

def compute_J1_J2(
    assignment: Mapping[RotationId, ResourceId],
    gates: Sequence[GateId],
) -> tuple[int, int]:
    gates_set = set(gates)
    on_gates = [r for r in assignment.values() if r in gates_set]
    return len(on_gates), len(set(on_gates))


@total_ordering
@dataclass(frozen=True, eq=False)
class PlanRanking:
    I_pi: int
    J3_pi: int | None
    J1: int
    J2: int
    is_pi_admissible: bool
    is_star_admissible: bool | None
    J3_star: int | None
    delta_rob: int | None
    delta_sc: int | None
    key: tuple

    def __lt__(self, other):
        if not isinstance(other, PlanRanking):
            return NotImplemented
        return self.key < other.key

    def __eq__(self, other):
        if not isinstance(other, PlanRanking):
            return NotImplemented
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)


def _ranking_key(I_pi, J3_pi, J1, J2):
    J3 = J3_pi if J3_pi is not None else float("inf")
    return (I_pi, J3, -J1, J2)


def rank_from_dual(evaluation, nominal_assignment, gates) -> PlanRanking:
    J1, J2 = compute_J1_J2(nominal_assignment, gates)
    return PlanRanking(
        I_pi=evaluation.I_pi,
        J3_pi=evaluation.J3_pi,
        J1=J1, J2=J2,
        is_pi_admissible=evaluation.x_is_pi_admissible,
        is_star_admissible=evaluation.x_is_star_admissible,
        J3_star=evaluation.J3_star,
        delta_rob=evaluation.delta_rob,
        delta_sc=evaluation.delta_sc,
        key=_ranking_key(evaluation.I_pi, evaluation.J3_pi, J1, J2),
    )


def rank_from_pi(evaluation: RobustEvaluation, nominal_assignment, gates) -> PlanRanking:
    I_pi = sum(1 for r in evaluation.scenario_results.values() if not r.feasible)
    J3_pi = evaluation.worst_case_broken_passengers if evaluation.admissible else None
    J1, J2 = compute_J1_J2(nominal_assignment, gates)
    return PlanRanking(
        I_pi=I_pi, J3_pi=J3_pi, J1=J1, J2=J2,
        is_pi_admissible=evaluation.admissible,
        is_star_admissible=None, J3_star=None,
        delta_rob=None, delta_sc=None,
        key=_ranking_key(I_pi, J3_pi, J1, J2),
    )


# =========================================================================
# MILP building blocks (§2.8) — solver-agnostic
# =========================================================================

def resources_of(
    gates: Sequence[GateId],
    stand_id: ResourceId = 0,
    finite_stands: Sequence[ResourceId] | None = None,
) -> list[ResourceId]:
    _resolve_stand_mode(gates, stand_id, finite_stands)
    gates_list = sorted(gates, key=_stable_key)
    if finite_stands is None:
        return gates_list + [stand_id]
    return gates_list + sorted(finite_stands, key=_stable_key)


def build_b_constants(
    passenger_flows: Sequence[PassengerFlow],
    resources: Sequence[ResourceId],
    walking_times: Mapping[tuple[ResourceId, ResourceId], float],
    scenarios: Sequence[Scenario],
    rotations: Sequence[Rotation],
    stress_sets: Mapping[WaveId, frozenset[RotationId]],
) -> dict[tuple[Hashable, ResourceId, ResourceId, str], int]:
    result: dict[tuple[Hashable, ResourceId, ResourceId, str], int] = {}
    for scenario in scenarios:
        schedule = _schedule_for_scenario(rotations, scenario, stress_sets)
        for flow in passenger_flows:
            T_l = schedule[flow.dep_rot][1] - schedule[flow.arr_rot][0]
            for a in resources:
                for b in resources:
                    w = _walk_time(walking_times, a, b)
                    result[(flow.id, a, b, scenario.id)] = (
                        1 if T_l < flow.min_conn + w else 0
                    )
    return result


def eta_and_z_constraints(
    resources, rotations, passenger_flows, scenarios,
) -> list[LinearConstraint]:
    constraints: list[LinearConstraint] = []
    for scenario in scenarios:
        xi = scenario.id
        for flow in passenger_flows:
            for a in resources:
                for b in resources:
                    y_a = ("y", flow.arr_rot, a, xi)
                    y_d = ("y", flow.dep_rot, b, xi)
                    eta = ("eta", flow.id, a, b, xi)
                    constraints.append(LinearConstraint(
                        ((eta, 1.0), (y_a, -1.0)), "<=", 0.0,
                        f"eta_ub_arr[{flow.id},{a},{b},{xi}]",
                    ))
                    constraints.append(LinearConstraint(
                        ((eta, 1.0), (y_d, -1.0)), "<=", 0.0,
                        f"eta_ub_dep[{flow.id},{a},{b},{xi}]",
                    ))
                    constraints.append(LinearConstraint(
                        ((eta, 1.0), (y_a, -1.0), (y_d, -1.0)), ">=", -1.0,
                        f"eta_lb[{flow.id},{a},{b},{xi}]",
                    ))
        for rot in rotations:
            for a in resources:
                x = ("x", rot.id, a)
                y = ("y", rot.id, a, xi)
                z = ("z", rot.id, a, xi)
                constraints.append(LinearConstraint(
                    ((z, 1.0), (x, -1.0)), "<=", 0.0,
                    f"z_ub_x[{rot.id},{a},{xi}]",
                ))
                constraints.append(LinearConstraint(
                    ((z, 1.0), (y, -1.0)), "<=", 0.0,
                    f"z_ub_y[{rot.id},{a},{xi}]",
                ))
                constraints.append(LinearConstraint(
                    ((z, 1.0), (x, -1.0), (y, -1.0)), ">=", -1.0,
                    f"z_lb[{rot.id},{a},{xi}]",
                ))
    return constraints


def budget_constraints(rotations, resources, scenarios, K) -> list[LinearConstraint]:
    M = len(rotations)
    out: list[LinearConstraint] = []
    for scenario in scenarios:
        coeffs = tuple(
            (("z", rot.id, a, scenario.id), -1.0)
            for rot in rotations for a in resources
        )
        out.append(LinearConstraint(
            coeffs, "<=", float(K - M),
            f"budget[{scenario.id}]",
        ))
    return out


def loss_expression(
    passenger_flows, resources, b_constants, scenario,
) -> tuple[tuple[VarKey, float], ...]:
    coeffs: dict[VarKey, float] = {}
    for flow in passenger_flows:
        for a in resources:
            for b in resources:
                if b_constants.get((flow.id, a, b, scenario.id), 0):
                    key = ("eta", flow.id, a, b, scenario.id)
                    coeffs[key] = coeffs.get(key, 0.0) + float(flow.n)
    return tuple(coeffs.items())


def evaluate_integer_assignment(
    y_per_scenario, b_constants, passenger_flows, resources,
) -> dict[str, int]:
    out: dict[str, int] = {}
    for xi, y in y_per_scenario.items():
        loss = 0
        for flow in passenger_flows:
            a_star = y.get(flow.arr_rot)
            b_star = y.get(flow.dep_rot)
            if a_star is None or b_star is None:
                raise ValueError(f"y[{xi}] missing assignment for a rotation")
            if b_constants[(flow.id, a_star, b_star, xi)]:
                loss += flow.n
        out[xi] = loss
    return out


def build_integer_extension(
    x_plan, y_plans, passenger_flows, resources,
) -> dict[VarKey, float]:
    a: dict[VarKey, float] = {}
    for rid, x_res in x_plan.items():
        for r in resources:
            a[("x", rid, r)] = 1.0 if r == x_res else 0.0
    for xi, y in y_plans.items():
        for rid, y_res in y.items():
            for r in resources:
                a[("y", rid, r, xi)] = 1.0 if r == y_res else 0.0
                a[("z", rid, r, xi)] = (
                    1.0 if (r == y_res and r == x_plan.get(rid)) else 0.0
                )
        for flow in passenger_flows:
            y_a = y[flow.arr_rot]
            y_d = y[flow.dep_rot]
            for r1 in resources:
                for r2 in resources:
                    a[("eta", flow.id, r1, r2, xi)] = (
                        1.0 if r1 == y_a and r2 == y_d else 0.0
                    )
    return a


def check_constraints(
    assignment: Mapping[VarKey, float],
    constraints: Sequence[LinearConstraint],
    *,
    tol: float = 1e-9,
) -> None:
    for c in constraints:
        lhs = sum(coef * assignment.get(var, 0.0) for var, coef in c.coeffs)
        ok = (
            (c.sense == "<=" and lhs <= c.rhs + tol)
            or (c.sense == ">=" and lhs >= c.rhs - tol)
            or (c.sense == "==" and abs(lhs - c.rhs) <= tol)
        )
        if not ok:
            raise AssertionError(
                f"constraint {c.label!r} violated: lhs={lhs}, "
                f"sense={c.sense}, rhs={c.rhs}"
            )


# =========================================================================
# Reduced instance (§2.8)
# =========================================================================

@dataclass(frozen=True)
class ReducedInstance:
    rotations: tuple[Rotation, ...]
    passenger_flows: tuple[PassengerFlow, ...]
    gates: tuple[GateId, ...]
    zones: Mapping[Hashable, tuple[GateId, ...]]
    compatibility: Mapping[tuple[RotationId, GateId], bool]
    walking_times: Mapping[tuple[ResourceId, ResourceId], float]
    kept_wave: WaveId
    kept_zones: tuple[Hashable, ...]
    dropped_flows: tuple[Hashable, ...]
    stand_id: ResourceId
    n_original_rotations: int
    n_original_flows: int
    n_original_gates: int


def build_reduced_instance(
    *,
    rotations, passenger_flows, gates, zones_dict, compatibility,
    walking_times, stand_id=0,
) -> ReducedInstance:
    validate_inputs(rotations, passenger_flows, gates, stand_id=stand_id)
    gates_set = set(gates)
    zone_gate_sets = {zid: tuple(zg) for zid, zg in zones_dict.items()}
    if not zone_gate_sets:
        raise ValueError("zones_dict must contain at least one zone")
    for zid, zg in zone_gate_sets.items():
        missing = [g for g in zg if g not in gates_set]
        if missing:
            raise ValueError(f"zone {zid!r} references unknown gates: {missing!r}")
    by_wave: dict[WaveId, list[Rotation]] = {}
    for rot in rotations:
        if rot.wave is not None:
            by_wave.setdefault(rot.wave, []).append(rot)
    if not by_wave:
        raise ValueError("no rotation carries a wave ID")
    kept_wave = min(
        by_wave.keys(),
        key=lambda w: (-len(by_wave[w]), _stable_key(w)),
    )
    kept_rotations = tuple(r for r in rotations if r.wave == kept_wave)
    kept_rotation_ids = {r.id for r in kept_rotations}
    ordered_zone_ids = sorted(
        zone_gate_sets.keys(),
        key=lambda z: (-len(zone_gate_sets[z]), _stable_key(z)),
    )
    kept_zones = tuple(ordered_zone_ids[:2])
    kept_gates_unsorted = [g for zid in kept_zones for g in zone_gate_sets[zid]]
    kept_gates = tuple(sorted(set(kept_gates_unsorted), key=_stable_key))
    reduced_zones = {zid: zone_gate_sets[zid] for zid in kept_zones}
    kept_flows: list[PassengerFlow] = []
    dropped: list[Hashable] = []
    for flow in passenger_flows:
        if flow.arr_rot in kept_rotation_ids and flow.dep_rot in kept_rotation_ids:
            kept_flows.append(flow)
        else:
            dropped.append(flow.id)
    all_kept_resources = set(kept_gates) | {stand_id}
    reduced_compat = {
        (rid, gid): value
        for (rid, gid), value in compatibility.items()
        if rid in kept_rotation_ids and gid in all_kept_resources
    }
    required_pairs = [
        (a, b) for a in all_kept_resources for b in all_kept_resources
    ]
    missing_pairs = [p for p in required_pairs if p not in walking_times]
    if missing_pairs:
        raise ValueError(
            f"walking_times missing pairs: {missing_pairs[:5]}"
            f"{' ...' if len(missing_pairs) > 5 else ''}"
        )
    reduced_walk = {
        (a, b): float(walking_times[(a, b)]) for (a, b) in required_pairs
    }
    return ReducedInstance(
        rotations=kept_rotations,
        passenger_flows=tuple(kept_flows),
        gates=kept_gates,
        zones=MappingProxyType(reduced_zones),
        compatibility=MappingProxyType(reduced_compat),
        walking_times=MappingProxyType(reduced_walk),
        kept_wave=kept_wave,
        kept_zones=kept_zones,
        dropped_flows=tuple(dropped),
        stand_id=stand_id,
        n_original_rotations=len(rotations),
        n_original_flows=len(passenger_flows),
        n_original_gates=len(gates),
    )