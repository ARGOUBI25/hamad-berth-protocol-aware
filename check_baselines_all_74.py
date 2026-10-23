"""Évalue les 3 baselines (greedy FF, greedy BF, det MILP) sur les 74
combinaisons du loose et les 10 du tight, pour obtenir un grid-wide
rate comparable à celui du joint MILP et du GA."""

import csv
from pathlib import Path
from hamad_adapter import load_hamad_instance
from gate_model import (
    build_reduced_instance, build_reference_assignment, build_stress_sets,
    compute_q_scores, generate_scenarios, resources_of,
)
from baselines import greedy_first_fit, greedy_best_fit, deterministic_milp
from recourse_milp import evaluate_dual_recourse
from sensitivity_hamad import regenerate_flows_at_alpha, NOMINAL_ALPHA

DELTA_MAP = {
    "low":  (30.0, 60.0, 90.0),
    "high": (60.0, 90.0, 120.0),
    "wide": (30.0, 60.0, 90.0, 120.0),
}

def eval_baselines(instance_path, csv_path, label):
    inst = load_hamad_instance(instance_path)
    combos = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            combos.append({
                "alpha": float(r["alpha"]),
                "delta_name": r["delta"] if "delta" in r else r["delta_name"],
                "delta_values": DELTA_MAP[r["delta"] if "delta" in r else r["delta_name"]],
                "beta": float(r["beta"]),
                "gamma_g": int(r["gamma_g"]),
                "K": int(r["K"]),
            })

    # plans det MILP : un par alpha
    plans_det = {}
    plans_ff = None
    plans_bf = None
    berth_lengths = None

    results = []
    for i, combo in enumerate(combos, 1):
        alpha = combo["alpha"]
        flows = inst["flows"] if abs(alpha - NOMINAL_ALPHA) < 1e-6 else regenerate_flows_at_alpha(inst, alpha)

        reduced = build_reduced_instance(
            rotations=inst["rotations"], passenger_flows=flows,
            gates=inst["gates"], zones_dict=inst["zones"],
            compatibility=inst["compatibility"], walking_times=inst["walking_times"], stand_id=0,
        )
        if berth_lengths is None:
            berth_lengths = {b: inst["berth_meta"][b]["length_m"] for b in reduced.gates}

        # plans : construits une seule fois
        if plans_ff is None:
            plans_ff = greedy_first_fit(reduced.rotations, reduced.gates, reduced.compatibility, inst["t_min"])
        if plans_bf is None:
            plans_bf = greedy_best_fit(reduced.rotations, reduced.gates, reduced.compatibility, inst["t_min"], berth_lengths=berth_lengths)
        if alpha not in plans_det:
            q = compute_q_scores(reduced.rotations, reduced.passenger_flows)
            _, stress = build_stress_sets(reduced.rotations, q)
            plans_det[alpha] = deterministic_milp(reduced, resources_of(reduced.gates, 0), inst["t_min"], combo["K"], stress, time_limit_ms=600_000)

        # scénarios pour ce combo
        q = compute_q_scores(reduced.rotations, reduced.passenger_flows)
        ref = build_reference_assignment(reduced.rotations, reduced.gates, reduced.compatibility, inst["t_min"], stand_id=0)
        _, stress = build_stress_sets(reduced.rotations, q)
        scen = generate_scenarios(reduced.zones, combo["gamma_g"], combo["delta_values"], combo["beta"], ref, reduced.rotations, reduced.passenger_flows, stand_id=0)
        res = resources_of(reduced.gates, 0)

        # évaluation des 3 plans sur les 23 scénarios
        for method, plan in [("ff", plans_ff), ("bf", plans_bf), ("det", plans_det[alpha])]:
            dual = evaluate_dual_recourse(plan, scen, reduced.rotations, reduced.passenger_flows, reduced.gates, reduced.compatibility, reduced.walking_times, q, stress, res, inst["t_min"], combo["K"], stand_id=0, time_limit_ms=60_000)
            status = "inf" if dual.I_pi > 0 else ("exact" if dual.delta_rob == 0 else "sub")
            results.append({
                "method": method, "alpha": alpha, "delta": combo["delta_name"],
                "gamma_g": combo["gamma_g"], "beta": combo["beta"], "K": combo["K"],
                "I_pi": dual.I_pi, "J": dual.J3_star, "J_pi": dual.J3_pi,
                "delta_rob": dual.delta_rob, "status": status,
            })
        if i % 10 == 0:
            print(f"[{label}][{i}/{len(combos)}] a={alpha:.2f} d={combo['delta_name']} G={combo['gamma_g']}")

    out = Path(f"sensitivity_output/baselines_all_{label}.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # résumé
    print(f"\n=== {label} ===")
    for method in ["ff", "bf", "det"]:
        m = [r for r in results if r["method"] == method]
        n_exact = sum(1 for r in m if r["status"] == "exact")
        n_sub = sum(1 for r in m if r["status"] == "sub")
        n_inf = sum(1 for r in m if r["status"] == "inf")
        print(f"{method}: {n_exact} exact, {n_sub} sub, {n_inf} inf ({n_exact/len(m)*100:.0f}% exact)")
    print(f"Written: {out}")

eval_baselines("hamad_instance.json", "sensitivity_output/sensitivity_aggregated_29mai.csv", "loose")
eval_baselines("hamad_instance_20250516.json", "sensitivity_output/sensitivity_raw.csv", "tight")