"""Run the three baselines on one instance and compare against π/MILP/GA.

Usage:
    python run_baselines.py hamad_instance.json --output baselines_29mai
    python run_baselines.py hamad_instance_20250516.json --output baselines_16mai
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from hamad_adapter import load_hamad_instance
from gate_model import (
    build_reduced_instance, build_reference_assignment,
    build_stress_sets, compute_J1_J2, compute_q_scores,
    generate_scenarios, resources_of,
)
from baselines import greedy_first_fit, greedy_best_fit, deterministic_milp
from recourse_milp import evaluate_dual_recourse


DELTA_SETS = {
    "low": (30.0, 60.0, 90.0),
    "high": (60.0, 90.0, 120.0),
    "wide": (30.0, 60.0, 90.0, 120.0),
}


def evaluate_plan(
    plan,
    scenarios,
    reduced,
    q,
    stress,
    resources,
    t_min,
    K,
    time_limit_ms=600_000,
):
    """Run dual recourse on a plan, return (J3_pi, J3_star, delta_rob, J1, J2)."""
    dual = evaluate_dual_recourse(
        plan, scenarios,
        reduced.rotations, reduced.passenger_flows, reduced.gates,
        reduced.compatibility, reduced.walking_times,
        q, stress, resources, t_min, K,
        stand_id=0, time_limit_ms=time_limit_ms,
    )
    J1, J2 = compute_J1_J2(plan, reduced.gates)
    return {
        "J3_pi": dual.J3_pi,
        "J3_star": dual.J3_star,
        "delta_rob": dual.delta_rob,
        "delta_sc": dual.delta_sc,
        "I_pi": dual.I_pi,
        "J1": J1,
        "J2": J2,
        "admissible_pi": dual.x_is_pi_admissible,
        "admissible_star": dual.x_is_star_admissible,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", type=str)
    parser.add_argument("--output", type=str, default="baselines_output")
    parser.add_argument("--K", type=int, default=4, help="Budget at which to compare (default 4)")
    parser.add_argument("--time-limit-ms", type=int, default=600_000)
    args = parser.parse_args(argv)

    outdir = Path(args.output)
    outdir.mkdir(exist_ok=True)

    inst = load_hamad_instance(args.instance)
    inst["K"] = args.K
    print(f"Loaded {len(inst['rotations'])} vessels, "
          f"{len(inst['flows'])} flows, K={args.K}")

    reduced = build_reduced_instance(
        rotations=inst["rotations"],
        passenger_flows=inst["flows"],
        gates=inst["gates"],
        zones_dict=inst["zones"],
        compatibility=inst["compatibility"],
        walking_times=inst["walking_times"],
        stand_id=0,
    )
    q = compute_q_scores(reduced.rotations, reduced.passenger_flows)
    ref = build_reference_assignment(
        reduced.rotations, reduced.gates, reduced.compatibility,
        inst["t_min"], stand_id=0,
    )
    _, stress = build_stress_sets(reduced.rotations, q)
    scenarios = generate_scenarios(
        reduced.zones, inst["gamma_g"], inst["delta_values"],
        inst["beta_max"], ref,
        reduced.rotations, reduced.passenger_flows, stand_id=0,
    )
    resources = resources_of(reduced.gates, 0)

    print(f"Reduced wave: {reduced.kept_wave}, "
          f"{len(reduced.rotations)} rotations, "
          f"{len(scenarios)} scenarios")

    results = {}

    # --- Greedy first-fit ---
    print("\n>>> Greedy first-fit")
    t0 = time.perf_counter()
    plan_ff = greedy_first_fit(
        reduced.rotations, reduced.gates, reduced.compatibility, inst["t_min"],
    )
    elapsed = time.perf_counter() - t0
    print(f"    plan built in {elapsed:.3f}s")
    metrics_ff = evaluate_plan(
        plan_ff, scenarios, reduced, q, stress, resources,
        inst["t_min"], args.K, args.time_limit_ms,
    )
    metrics_ff["method"] = "greedy_first_fit"
    metrics_ff["plan"] = dict(plan_ff)
    metrics_ff["build_time_s"] = elapsed
    results["greedy_first_fit"] = metrics_ff
    print(f"    J3_pi={metrics_ff['J3_pi']} J3*={metrics_ff['J3_star']} "
          f"Δrob={metrics_ff['delta_rob']}")

    # --- Greedy best-fit ---
    print("\n>>> Greedy best-fit")
    berth_lengths = {b: _get_berth_length(b, inst) for b in reduced.gates}
    t0 = time.perf_counter()
    plan_bf = greedy_best_fit(
        reduced.rotations, reduced.gates, reduced.compatibility, inst["t_min"],
        berth_lengths=berth_lengths,
    )
    elapsed = time.perf_counter() - t0
    print(f"    plan built in {elapsed:.3f}s")
    metrics_bf = evaluate_plan(
        plan_bf, scenarios, reduced, q, stress, resources,
        inst["t_min"], args.K, args.time_limit_ms,
    )
    metrics_bf["method"] = "greedy_best_fit"
    metrics_bf["plan"] = dict(plan_bf)
    metrics_bf["build_time_s"] = elapsed
    results["greedy_best_fit"] = metrics_bf
    print(f"    J3_pi={metrics_bf['J3_pi']} J3*={metrics_bf['J3_star']} "
          f"Δrob={metrics_bf['delta_rob']}")

    # --- Deterministic MILP ---
    print("\n>>> Deterministic MILP (single nominal scenario)")
    t0 = time.perf_counter()
    try:
        plan_det = deterministic_milp(
            reduced, resources, inst["t_min"], args.K, stress,
            time_limit_ms=args.time_limit_ms,
        )
        elapsed = time.perf_counter() - t0
        print(f"    plan built in {elapsed:.1f}s")
        metrics_det = evaluate_plan(
            plan_det, scenarios, reduced, q, stress, resources,
            inst["t_min"], args.K, args.time_limit_ms,
        )
        metrics_det["method"] = "deterministic_milp"
        metrics_det["plan"] = dict(plan_det)
        metrics_det["build_time_s"] = elapsed
        results["deterministic_milp"] = metrics_det
        print(f"    J3_pi={metrics_det['J3_pi']} J3*={metrics_det['J3_star']} "
              f"Δrob={metrics_det['delta_rob']}")
    except Exception as e:
        print(f"    FAILED: {e}")
        results["deterministic_milp"] = {"method": "deterministic_milp",
                                          "error": str(e)}

    # --- Write CSV ---
    csv_path = outdir / "baselines_summary.csv"
    fieldnames = [
        "method", "J3_pi", "J3_star", "delta_rob", "delta_sc",
        "I_pi", "J1", "J2", "admissible_pi", "admissible_star",
        "build_time_s",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m, r in results.items():
            writer.writerow({k: r.get(k) for k in fieldnames})

    # --- Write LaTeX ---
    tex_path = outdir / "baselines_table.tex"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Baseline comparison at $K=" + str(args.K) + r"$. "
        r"$\Delta_{\mathrm{rob}} = J_3^\pi(x) - J_3^*(x)$; "
        r"lower is better.}",
        r"\label{tab:baselines_" + outdir.name + "}",
        r"\begin{tabular}{l|cc|c|cc}",
        r"\hline",
        r"Method & $J_3^\pi$ & $J_3^*$ & $\Delta_{\mathrm{rob}}$ "
        r"& $J_1$ & $J_2$ \\",
        r"\hline",
    ]
    for m, r in results.items():
        method_escaped = m.replace("_", "\\_")
        if "error" in r:
            lines.append(
                f"{method_escaped} & -- & -- & -- & -- & -- \\\\"
            )
            continue
        lines.append(
            f"{method_escaped} & {r['J3_pi']} & {r['J3_star']} "
            f"& {r['delta_rob']} & {r['J1']} & {r['J2']} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    tex_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {tex_path}")
    return 0


def _get_berth_length(berth_id, inst):
    for b in inst.get("berths", []):
        if b["id"] == berth_id:
            return b.get("length_m", 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())