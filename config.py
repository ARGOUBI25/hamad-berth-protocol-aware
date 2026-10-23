"""Central configuration for the robust gate-assignment pipeline.

All numeric parameters required by §2.7 (GA) and §2.8 (MILP) are gathered
here so that the paper's settings can be checked against the code without
hunting through modules.

The DEMO_* values describe the small synthetic instance used by
`benchmark_run.py`, `example_run.py` and `main.py --mode benchmark`.
They are NOT empirical airport data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from gate_model import PassengerFlow, Rotation
from genetic_algorithm import GAConfig


# =========================================================================
# §2.7 — Genetic algorithm parameters
# =========================================================================

GA_DEFAULT = GAConfig(
    population_size=150,
    max_generations=400,
    crossover_probability=0.8,
    mutation_probability=0.05,
    tournament_size=3,
    elitism=2,
    seeds=(1, 2, 3, 4, 5),
    rho_mutation_share=0.5,
)
"""Default GA configuration, §2.7."""

GA_DEMO = GAConfig(
    population_size=60,
    max_generations=50,
    seeds=(1, 2, 3, 4, 5),
)
"""Reduced-size GA for the synthetic demo, keeps runtime under one minute."""


# =========================================================================
# §2.3 — Scenario-generation parameters
# =========================================================================

GAMMA_G_VALUES: tuple[int, ...] = (3, 6, 10)
"""Closure levels studied in the paper."""

GAMMA_G_DEMO: int = 1
"""Demo uses the smallest closure level (only one gate per zone)."""

DELTA_VALUES: tuple[float, ...] = (0.0, 30.0, 60.0)
BETA_MAX: float = 15.0


# =========================================================================
# §2.5 — Separation and budget
# =========================================================================

T_MIN: float = 10.0
"""Minimum temporal separation between two rotations sharing a resource."""

K_DEMO: int = 4
"""Change budget for the demo pipeline."""


# =========================================================================
# Solver parameters
# =========================================================================

MILP_TIME_LIMIT_MS_RECOURSE: int = 60_000
MILP_TIME_LIMIT_MS_JOINT: int = 600_000
MILP_GAP: float = 0.0


# =========================================================================
# Instance container
# =========================================================================

@dataclass(frozen=True)
class Instance:
    """Full instance: everything the pipeline needs, in one object."""
    rotations: tuple[Rotation, ...]
    passenger_flows: tuple[PassengerFlow, ...]
    gates: tuple[int, ...]
    zones: Mapping[str, tuple[int, ...]]
    finite_stands: tuple[int, ...]
    compatibility: Mapping[tuple, bool]
    walking_times: Mapping[tuple, float]
    t_min: float
    K: int
    gamma_g: int
    delta_values: tuple[float, ...]
    beta_max: float
    stand_id: int = 0


# =========================================================================
# Demo instance (synthetic)
# =========================================================================

_DEMO_ROTATIONS = [
    Rotation("A", 100, 130, "M", "W1"),
    Rotation("B", 150, 180, "M", "W1"),
    Rotation("C", 250, 280, "M", "W2"),
    Rotation("D", 300, 330, "M", "W2"),
    Rotation("E", 350, 380, "M", "W2"),
    Rotation("F", 400, 430, "M", "W2"),
]
_DEMO_FLOWS = [
    PassengerFlow("P1", "C", "D", 100, 46),
    PassengerFlow("P2", "E", "F", 60, 46),
    PassengerFlow("P3", "A", "B", 20, 30),
]
_DEMO_GATES = [1, 2, 3, 4]
_DEMO_ZONES = {"Z1": [1, 2], "Z2": [3, 4], "Z3": []}
_DEMO_FINITE_STANDS = [10, 11]
_DEMO_ALL_RESOURCES = _DEMO_GATES + _DEMO_FINITE_STANDS


def demo_instance() -> Instance:
    """The synthetic instance used by the demo pipeline."""
    compat = {
        (r.id, a): True for r in _DEMO_ROTATIONS for a in _DEMO_ALL_RESOURCES
    }
    # Include stand_id=0 (aggregated stand) in walking_times
    all_resources_with_stand = [0] + _DEMO_ALL_RESOURCES
    walking = {
        (a, b): (0.0 if a == b else 5.0)
        for a in all_resources_with_stand for b in all_resources_with_stand
    }
    return Instance(
        rotations=tuple(_DEMO_ROTATIONS),
        passenger_flows=tuple(_DEMO_FLOWS),
        gates=tuple(_DEMO_GATES),
        zones={z: tuple(g) for z, g in _DEMO_ZONES.items()},
        finite_stands=tuple(_DEMO_FINITE_STANDS),
        compatibility=compat,
        walking_times=walking,
        t_min=T_MIN,
        K=K_DEMO,
        gamma_g=GAMMA_G_DEMO,
        delta_values=DELTA_VALUES,
        beta_max=BETA_MAX,
    )


# =========================================================================
# Instance loading (JSON)
# =========================================================================

def load_instance(path: str | Path) -> Instance:
    """Load an instance from a JSON file.

    JSON schema (all keys except rotations/passenger_flows/gates/zones are
    optional and fall back to config defaults):

      {
        "rotations": [{"id": "A", "arr": 100, "dep": 130,
                        "gabarit": "M", "wave": "W1"}, ...],
        "passenger_flows": [{"id": "P1", "arr_rot": "A", "dep_rot": "B",
                              "n": 50, "min_conn": 20}, ...],
        "gates": [1, 2, 3, 4],
        "zones": {"Z1": [1, 2], "Z2": [3, 4]},
        "finite_stands": [10, 11],
        "stand_id": 0,
        "t_min": 10.0,
        "K": 4,
        "gamma_g": 1,
        "delta_values": [0, 30, 60],
        "beta_max": 15.0,
        "compatibility": {
          "default": true,
          "false_pairs": [["A", 1]]
        },
        "walking_times": {
          "default_same": 0.0,
          "default_diff": 5.0,
          "overrides": [[1, 2, 8.5]]
        }
      }
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    rotations = tuple(
        Rotation(
            id=r["id"], arr=float(r["arr"]), dep=float(r["dep"]),
            gabarit=r.get("gabarit"), wave=r.get("wave"),
        )
        for r in raw["rotations"]
    )
    flows = tuple(
        PassengerFlow(
            id=f["id"], arr_rot=f["arr_rot"], dep_rot=f["dep_rot"],
            n=int(f["n"]), min_conn=float(f["min_conn"]),
        )
        for f in raw["passenger_flows"]
    )
    gates = tuple(raw["gates"])
    zones = {z: tuple(gs) for z, gs in raw["zones"].items()}
    finite_stands = tuple(raw.get("finite_stands", []))
    stand_id = int(raw.get("stand_id", 0))

    resources = list(gates) + list(finite_stands)
    if stand_id not in resources:
        resources.append(stand_id)

    compat_spec = raw.get("compatibility", {"default": True})
    compat_default = bool(compat_spec.get("default", True))
    compat: dict[tuple, bool] = {
        (r.id, a): compat_default for r in rotations for a in resources
    }
    for pair in compat_spec.get("false_pairs", []):
        rid, res = pair[0], pair[1]
        if (rid, res) in compat:
            compat[(rid, res)] = False

    walk_spec = raw.get(
        "walking_times", {"default_same": 0.0, "default_diff": 5.0}
    )
    same = float(walk_spec.get("default_same", 0.0))
    diff = float(walk_spec.get("default_diff", 5.0))
    walking: dict[tuple, float] = {
        (a, b): (same if a == b else diff)
        for a in resources for b in resources
    }
    for triple in walk_spec.get("overrides", []):
        a, b, w = triple
        walking[(a, b)] = float(w)

    return Instance(
        rotations=rotations,
        passenger_flows=flows,
        gates=gates,
        zones=zones,
        finite_stands=finite_stands,
        compatibility=compat,
        walking_times=walking,
        t_min=float(raw.get("t_min", T_MIN)),
        K=int(raw.get("K", K_DEMO)),
        gamma_g=int(raw.get("gamma_g", GAMMA_G_DEMO)),
        delta_values=tuple(raw.get("delta_values", DELTA_VALUES)),
        beta_max=float(raw.get("beta_max", BETA_MAX)),
        stand_id=stand_id,
    )