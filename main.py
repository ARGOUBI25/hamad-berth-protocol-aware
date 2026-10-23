"""Command-line entry point for the robust gate-assignment pipeline.

Usage
-----
  python main.py tests                     Run the full unit-test suite.
  python main.py example                   Run the synthetic smoke test.
  python main.py benchmark                 Run the demo pipeline (§2.7 + §2.8).
  python main.py ga [--seed N] [--K K]     Run one GA run on the demo instance.
  python main.py joint [--K K]             Run the joint MILP on the demo
                                             reduced instance only.
  python main.py instance FILE.json        Run the pipeline on an external
                                             instance.
  python main.py hamad FILE.json [--K K]   Run the Hamad Port maritime
                                             calibrated instance.

All numeric defaults come from config.py. Use `--no-warnings` to suppress
the aggregated-stand warning if the demo uses finite stands.
"""
from __future__ import annotations

import argparse
import sys
import unittest
import warnings
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*stand model.*"
)


# ---------------------------------------------------------------------------
# Sub-command: tests
# ---------------------------------------------------------------------------

def cmd_tests(_args: argparse.Namespace) -> int:
    loader = unittest.TestLoader()
    suite = loader.discover(".", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


# ---------------------------------------------------------------------------
# Sub-command: example
# ---------------------------------------------------------------------------

def cmd_example(_args: argparse.Namespace) -> int:
    from example_run import main as ex_main
    ex_main()
    return 0


# ---------------------------------------------------------------------------
# Sub-command: benchmark
# ---------------------------------------------------------------------------

def cmd_benchmark(_args: argparse.Namespace) -> int:
    from benchmark_run import main as bm_main
    bm_main()
    return 0


# ---------------------------------------------------------------------------
# Sub-command: ga (single seed on the demo instance)
# ---------------------------------------------------------------------------

def cmd_ga(args: argparse.Namespace) -> int:
    from config import demo_instance
    from gate_model import (
        build_reference_assignment, build_stress_sets, compute_q_scores,
        generate_scenarios, resources_of,
    )
    from genetic_algorithm import GAConfig, run_ga

    inst = demo_instance()
    if args.K is not None:
        inst = _with_K(inst, args.K)

    q = compute_q_scores(inst.rotations, inst.passenger_flows)
    ref = build_reference_assignment(
        inst.rotations, inst.gates, inst.compatibility, inst.t_min,
        stand_id=inst.stand_id, finite_stands=inst.finite_stands or None,
    )
    _, stress = build_stress_sets(inst.rotations, q)
    scenarios = generate_scenarios(
        inst.zones, inst.gamma_g, inst.delta_values, inst.beta_max, ref,
        inst.rotations, inst.passenger_flows, stand_id=inst.stand_id,
    )
    resources = resources_of(
        inst.gates, inst.stand_id, inst.finite_stands or None,
    )

    config = GAConfig(
        population_size=args.population or 60,
        max_generations=args.generations or 50,
        seeds=(args.seed,),
    )
    result = run_ga(
        args.seed, config,
        rotations=inst.rotations, passenger_flows=inst.passenger_flows,
        gates=inst.gates, compatibility=inst.compatibility,
        walking_times=inst.walking_times, q_scores=q, stress_sets=stress,
        scenarios=scenarios, resources=resources,
        t_min=inst.t_min, K=inst.K,
        stand_id=inst.stand_id, finite_stands=inst.finite_stands or None,
    )

    print("=== GA single-seed run (demo instance) ===")
    print(f"Seed: {args.seed}")
    r = result.best_individual.ranking
    print(f"Ranking: (I_pi={r.I_pi}, J3_pi={r.J3_pi}, "
          f"J1={r.J1}, J2={r.J2})")
    print(f"Plan: {dict(result.best_individual.plan) if result.best_individual.plan else None}")
    print(f"Cache size: {result.cache_size}, protocol_calls: {result.protocol_calls}")
    print(f"Scenarios evaluated: {len(scenarios)}")
    return 0


# ---------------------------------------------------------------------------
# Sub-command: joint (MILP only, on the reduced demo instance)
# ---------------------------------------------------------------------------

def cmd_joint(args: argparse.Namespace) -> int:
    from config import (
        MILP_GAP, MILP_TIME_LIMIT_MS_JOINT, demo_instance,
    )
    from gate_model import (
        build_reduced_instance, build_reference_assignment,
        build_stress_sets, compute_q_scores, generate_scenarios, resources_of,
    )
    from joint_milp import solve_joint_milp

    inst = demo_instance()
    if args.K is not None:
        inst = _with_K(inst, args.K)

    reduced = build_reduced_instance(
        rotations=inst.rotations, passenger_flows=inst.passenger_flows,
        gates=inst.gates, zones_dict=inst.zones,
        compatibility=inst.compatibility, walking_times=inst.walking_times,
        stand_id=inst.stand_id,
    )
    finite_stands = inst.finite_stands or None
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

    result = solve_joint_milp(
        reduced, scenarios, stress, resources, inst.t_min, inst.K,
        finite_stands=finite_stands,
        time_limit_ms=MILP_TIME_LIMIT_MS_JOINT, mip_gap=MILP_GAP,
    )

    print("=== Joint MILP on the demo reduced instance ===")
    print(f"Reduced wave: {reduced.kept_wave}, zones: {reduced.kept_zones}")
    print(f"Status: {result.status}, phase reached: {result.phase_reached}")
    print(f"J3* = {result.J3_star}, J1 = {result.J1}, J2 = {result.J2}")
    print(f"Best bound: {result.best_bound}, gap: {result.gap}")
    print(f"Plan: {dict(result.nominal_plan) if result.nominal_plan else None}")
    for note in result.notes:
        print(f"  note: {note}")
    return 0


# ---------------------------------------------------------------------------
# Sub-command: instance FILE.json
# ---------------------------------------------------------------------------

def cmd_instance(args: argparse.Namespace) -> int:
    from config import (
        MILP_GAP, MILP_TIME_LIMIT_MS_JOINT, GA_DEMO, load_instance,
    )
    from gate_model import (
        build_reduced_instance, build_reference_assignment,
        build_stress_sets, compute_J1_J2, compute_q_scores,
        generate_scenarios, resources_of,
    )
    from genetic_algorithm import run_ga_multiple_seeds
    from joint_milp import solve_joint_milp
    from recourse_milp import evaluate_dual_recourse

    inst = load_instance(args.path)
    if args.K is not None:
        inst = _with_K(inst, args.K)

    finite_stands = inst.finite_stands or None

    reduced = build_reduced_instance(
        rotations=inst.rotations, passenger_flows=inst.passenger_flows,
        gates=inst.gates, zones_dict=inst.zones,
        compatibility=inst.compatibility, walking_times=inst.walking_times,
        stand_id=inst.stand_id,
    )
    print("=== Reduced instance ===")
    print(f"Kept wave: {reduced.kept_wave}, zones: {reduced.kept_zones}")
    print(f"Rotations: {[r.id for r in reduced.rotations]}")
    print(f"Gates: {list(reduced.gates)}, stands: {list(finite_stands or [])}")
    print(f"Dropped flows: {list(reduced.dropped_flows)}")

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

    joint = solve_joint_milp(
        reduced, scenarios, stress, resources, inst.t_min, inst.K,
        finite_stands=finite_stands,
        time_limit_ms=MILP_TIME_LIMIT_MS_JOINT, mip_gap=MILP_GAP,
    )
    print("\n=== Joint MILP ===")
    print(f"Status: {joint.status}, phase: {joint.phase_reached}")
    print(f"J3* = {joint.J3_star}, J1 = {joint.J1}, J2 = {joint.J2}")
    print(f"Best bound: {joint.best_bound}, gap: {joint.gap}")
    for note in joint.notes:
        print(f"  note: {note}")

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
    print("\n=== GA ===")
    print(f"Best seed: {ga_best.seed}")
    r = ga_best.best_individual.ranking
    print(f"Ranking: (I_pi={r.I_pi}, J3_pi={r.J3_pi}, "
          f"J1={r.J1}, J2={r.J2})")

    if joint.nominal_plan is not None:
        print("\n=== Dual recourse on x* (joint MILP plan) ===")
        dual = evaluate_dual_recourse(
            joint.nominal_plan, scenarios,
            reduced.rotations, reduced.passenger_flows, reduced.gates,
            reduced.compatibility, reduced.walking_times,
            q, stress, resources, inst.t_min, inst.K,
            stand_id=inst.stand_id, finite_stands=finite_stands,
        )
        _print_dual(dual)
        _check_cross_invariants(joint, dual)

    if ga_best.best_individual.plan is not None:
        print("\n=== Dual recourse on x_ga (GA plan) ===")
        dual_ga = evaluate_dual_recourse(
            ga_best.best_individual.plan, scenarios,
            reduced.rotations, reduced.passenger_flows, reduced.gates,
            reduced.compatibility, reduced.walking_times,
            q, stress, resources, inst.t_min, inst.K,
            stand_id=inst.stand_id, finite_stands=finite_stands,
        )
        _print_dual(dual_ga)
        _check_cross_invariants(None, dual_ga,
                                ga_ranking=ga_best.best_individual.ranking)

    return 0


# ---------------------------------------------------------------------------
# Sub-command: hamad FILE.json  (maritime calibrated instance)
# ---------------------------------------------------------------------------

def cmd_hamad(args: argparse.Namespace) -> int:
    """Run the Hamad Port calibrated instance through the pipeline.

    Uses the aggregated-stand mode (anchorage id 0) so that no artificial
    capacity constraint is imposed on the anchorage. The 1440-min walking
    penalty to/from the anchorage is baked into the instance itself.
    """
    from hamad_adapter import load_hamad_instance, print_summary
    from gate_model import (
        build_reduced_instance, build_reference_assignment,
        build_stress_sets, compute_q_scores, generate_scenarios,
        resources_of,
    )
    from genetic_algorithm import GAConfig, run_ga_multiple_seeds
    from joint_milp import solve_joint_milp
    from recourse_milp import evaluate_dual_recourse

    inst = load_hamad_instance(args.path)
    if args.K is not None:  # ← d'abord
        inst["K"] = args.K
    print_summary(inst)

    reduced = build_reduced_instance(
        rotations=inst["rotations"],
        passenger_flows=inst["flows"],
        gates=inst["gates"],
        zones_dict=inst["zones"],
        compatibility=inst["compatibility"],
        walking_times=inst["walking_times"],
        stand_id=0,
    )
    print(f"\n=== Reduced instance ===")
    print(f"Kept wave: {reduced.kept_wave}, zones: {reduced.kept_zones}")
    print(f"Rotations: {[r.id for r in reduced.rotations]}")
    print(f"Gates: {list(reduced.gates)}")
    print(f"Dropped flows: {list(reduced.dropped_flows)}")

    q = compute_q_scores(reduced.rotations, reduced.passenger_flows)
    ref = build_reference_assignment(
        reduced.rotations, reduced.gates, reduced.compatibility,
        inst["t_min"], stand_id=0,
    )
    _, stress = build_stress_sets(reduced.rotations, q)
    scenarios = generate_scenarios(
        reduced.zones, inst["gamma_g"], inst["delta_values"],
        inst["beta_max"], ref, reduced.rotations, reduced.passenger_flows,
        stand_id=0,
    )
    resources = resources_of(reduced.gates, 0)

    print(f"\nScenarios: {len(scenarios)}")
    print(f"Reference g°: {dict(ref)}")

    print("\n=== Joint MILP ===")
    joint = solve_joint_milp(
        reduced, scenarios, stress, resources,
        inst["t_min"], inst["K"],
    )
    print(f"Status: {joint.status}, phase: {joint.phase_reached}")
    print(f"J3* = {joint.J3_star}, J1 = {joint.J1}, J2 = {joint.J2}")
    print(f"Best bound: {joint.best_bound}, gap: {joint.gap}")
    print(f"Plan: {dict(joint.nominal_plan) if joint.nominal_plan else None}")
    for note in joint.notes:
        print(f"  note: {note}")

    print("\n=== GA (5 seeds) ===")
    ga_runs = run_ga_multiple_seeds(
        GAConfig(population_size=60, max_generations=50, seeds=(1, 2, 3, 4, 5)),
        rotations=reduced.rotations,
        passenger_flows=reduced.passenger_flows,
        gates=reduced.gates,
        compatibility=reduced.compatibility,
        walking_times=reduced.walking_times,
        q_scores=q, stress_sets=stress, scenarios=scenarios,
        resources=resources, t_min=inst["t_min"], K=inst["K"],
        stand_id=0,
    )
    ga_best = min(ga_runs, key=lambda r: r.best_individual.ranking.key)
    print(f"Best seed: {ga_best.seed}")
    r = ga_best.best_individual.ranking
    print(f"Ranking: (I_pi={r.I_pi}, J3_pi={r.J3_pi}, J1={r.J1}, J2={r.J2})")
    print(f"GA plan: {dict(ga_best.best_individual.plan) if ga_best.best_individual.plan else None}")

    if joint.nominal_plan is not None:
        print("\n=== Dual recourse on x* (joint MILP plan) ===")
        dual = evaluate_dual_recourse(
            joint.nominal_plan, scenarios,
            reduced.rotations, reduced.passenger_flows, reduced.gates,
            reduced.compatibility, reduced.walking_times,
            q, stress, resources, inst["t_min"], inst["K"],
            stand_id=0,
        )
        _print_dual(dual)
        _check_cross_invariants(joint, dual)

    if ga_best.best_individual.plan is not None:
        print("\n=== Dual recourse on x_ga (GA plan) ===")
        dual_ga = evaluate_dual_recourse(
            ga_best.best_individual.plan, scenarios,
            reduced.rotations, reduced.passenger_flows, reduced.gates,
            reduced.compatibility, reduced.walking_times,
            q, stress, resources, inst["t_min"], inst["K"],
            stand_id=0,
        )
        _print_dual(dual_ga)
        _check_cross_invariants(None, dual_ga,
                                ga_ranking=ga_best.best_individual.ranking)

    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _with_K(inst, K):
    from dataclasses import replace
    return replace(inst, K=int(K))


def _print_dual(ev) -> None:
    print(f"π-admissible: {ev.x_is_pi_admissible}")
    print(f"MILP-admissible: {ev.x_is_star_admissible}")
    print(f"I_pi = {ev.I_pi}, J3_pi = {ev.J3_pi}, J3* = {ev.J3_star}")
    print(f"Δ_rob = {ev.delta_rob}, Δ_sc = {ev.delta_sc}")
    if ev.inconsistency_warnings:
        print(f"Inconsistencies: {ev.inconsistency_warnings}")


def _check_cross_invariants(joint, dual, ga_ranking=None) -> None:
    ok = True
    if joint is not None and dual.J3_star is not None and joint.J3_star is not None:
        same = joint.J3_star == dual.J3_star
        ok &= same
        print(f"{'[OK]' if same else '[FAIL]'} "
              f"Joint J3* == dual recourse J3*: "
              f"{joint.J3_star} vs {dual.J3_star}")
    if (ga_ranking is not None
            and dual.J3_pi is not None
            and ga_ranking.J3_pi is not None):
        same = ga_ranking.J3_pi == dual.J3_pi
        ok &= same
        print(f"{'[OK]' if same else '[FAIL]'} "
              f"GA J3_pi == dual recourse J3_pi: "
              f"{ga_ranking.J3_pi} vs {dual.J3_pi}")
    if dual.delta_rob is not None:
        same = dual.delta_rob == dual.J3_pi - dual.J3_star
        ok &= same
        print(f"{'[OK]' if same else '[FAIL]'} Δ_rob = J3_pi − J3*")
    if not dual.inconsistency_warnings:
        print("[OK] No inconsistency warnings")
    else:
        ok = False
        print(f"[FAIL] Inconsistencies: {dual.inconsistency_warnings}")
    if not ok:
        print("[WARN] Cross-invariants failed — see above")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Robust airport gate assignment — CLI entry point.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("tests", help="Run the full unit-test suite.").set_defaults(
        func=cmd_tests,
    )
    sub.add_parser("example", help="Run the synthetic smoke test.").set_defaults(
        func=cmd_example,
    )
    sub.add_parser("benchmark", help="Run the demo pipeline.").set_defaults(
        func=cmd_benchmark,
    )

    ga = sub.add_parser("ga", help="Run one GA run on the demo instance.")
    ga.add_argument("--seed", type=int, default=1)
    ga.add_argument("--K", type=int, default=None)
    ga.add_argument("--population", type=int, default=None)
    ga.add_argument("--generations", type=int, default=None)
    ga.set_defaults(func=cmd_ga)

    joint = sub.add_parser(
        "joint", help="Run the joint MILP on the demo reduced instance.",
    )
    joint.add_argument("--K", type=int, default=None)
    joint.set_defaults(func=cmd_joint)

    inst = sub.add_parser(
        "instance", help="Run the full pipeline on a JSON instance.",
    )
    inst.add_argument("path", type=str, help="Path to instance JSON file.")
    inst.add_argument("--K", type=int, default=None)
    inst.set_defaults(func=cmd_instance)

    hamad = sub.add_parser(
        "hamad", help="Run the Hamad Port maritime calibrated instance.",
    )
    hamad.add_argument("path", type=str, help="Path to hamad_instance.json.")
    hamad.add_argument("--K", type=int, default=None)
    hamad.set_defaults(func=cmd_hamad)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())