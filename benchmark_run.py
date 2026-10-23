"""Benchmark pipeline: demo GA + demo MILP reduced instance.

Run with: python benchmark_run.py
or via CLI: python main.py benchmark
"""
import dataclasses
from types import MappingProxyType

from config import MILP_GAP, MILP_TIME_LIMIT_MS_JOINT, GA_DEMO, demo_instance
from gate_model import (
    build_reduced_instance, build_reference_assignment, build_stress_sets,
    compute_q_scores, generate_scenarios, resources_of,
)
from genetic_algorithm import run_ga_multiple_seeds
from joint_milp import solve_joint_milp


def main() -> None:
    print("=" * 70)
    print("BENCHMARK PIPELINE (demo instance)")
    print("=" * 70)

    inst = demo_instance()
    finite_stands = inst.finite_stands or None

    # ---- 1. Reduced instance
    print("\n1. Building reduced instance...")
    reduced = build_reduced_instance(
        rotations=inst.rotations, passenger_flows=inst.passenger_flows,
        gates=inst.gates, zones_dict=inst.zones,
        compatibility=inst.compatibility, walking_times=inst.walking_times,
        stand_id=inst.stand_id,
    )
    print(f"   Kept wave: {reduced.kept_wave}, zones: {reduced.kept_zones}")
    print(f"   Rotations: {[r.id for r in reduced.rotations]}")
    print(f"   Flows: {[f.id for f in reduced.passenger_flows]}")
    print(f"   Dropped flows: {list(reduced.dropped_flows)}")

    # Ensure walking_times includes finite_stands (they're not in reduced zones)
    if finite_stands:
        walking_dict = dict(reduced.walking_times)
        all_resources = [inst.stand_id] + list(reduced.gates) + list(finite_stands)
        for i in finite_stands:
            for j in all_resources:
                if (i, j) not in walking_dict:
                    walking_dict[(i, j)] = inst.walking_times[(i, j)]
            for j in all_resources:
                if (j, i) not in walking_dict:
                    walking_dict[(j, i)] = inst.walking_times[(j, i)]
        reduced = dataclasses.replace(
            reduced, walking_times=MappingProxyType(walking_dict)
        )

    q = compute_q_scores(reduced.rotations, reduced.passenger_flows)
    ref = build_reference_assignment(
        reduced.rotations, reduced.gates, reduced.compatibility,
        inst.t_min, stand_id=inst.stand_id, finite_stands=finite_stands,
    )
    _, stress = build_stress_sets(reduced.rotations, q)
    scenarios = generate_scenarios(
        reduced.zones, inst.gamma_g, inst.delta_values, inst.beta_max, ref,
        reduced.rotations, reduced.passenger_flows, stand_id=inst.stand_id,
    )
    resources = resources_of(reduced.gates, inst.stand_id, finite_stands)
    print(f"   Scenarios: {len(scenarios)}")

    # ---- 2. GA
    print("\n2. Running GA (§2.7)...")
    ga_runs = run_ga_multiple_seeds(
        GA_DEMO,
        rotations=reduced.rotations, passenger_flows=reduced.passenger_flows,
        gates=reduced.gates, compatibility=reduced.compatibility,
        walking_times=reduced.walking_times, q_scores=q, stress_sets=stress,
        scenarios=scenarios, resources=resources,
        t_min=inst.t_min, K=inst.K,
        stand_id=inst.stand_id, finite_stands=finite_stands,
    )
    ga_best = min(ga_runs, key=lambda r: r.best_individual.ranking.key)
    print(f"   Best seed: {ga_best.seed}")
    print(f"   Ranking: I_pi={ga_best.best_individual.ranking.I_pi}, "
          f"J3_pi={ga_best.best_individual.ranking.J3_pi}, "
          f"J1={ga_best.best_individual.ranking.J1}, "
          f"J2={ga_best.best_individual.ranking.J2}")
    print(f"   Plan: {dict(ga_best.best_individual.plan) if ga_best.best_individual.plan else None}")

    # ---- 3. Joint MILP
    print("\n3. Running joint MILP (§2.8)...")
    joint = solve_joint_milp(
        reduced, scenarios, stress, resources, inst.t_min, inst.K,
        finite_stands=finite_stands,
        time_limit_ms=MILP_TIME_LIMIT_MS_JOINT, mip_gap=MILP_GAP,
    )
    print(f"   Status: {joint.status}, phase reached: {joint.phase_reached}")
    print(f"   J3* = {joint.J3_star}, J1 = {joint.J1}, J2 = {joint.J2}")
    print(f"   Best bound: {joint.best_bound}, gap: {joint.gap}")
    print(f"   Plan: {dict(joint.nominal_plan) if joint.nominal_plan else None}")
    for note in joint.notes:
        print(f"     note: {note}")

    print("\n" + "=" * 70)
    print("Benchmark complete ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
