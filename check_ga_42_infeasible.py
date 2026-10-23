"""Lance le GA (5 seeds) sur toutes les combinaisons du CSV.
Rapporte Iπ(GA) et Δrob(GA) pour chacune."""

import csv
from pathlib import Path

from hamad_adapter import load_hamad_instance
from gate_model import (
    build_reduced_instance, build_reference_assignment, build_stress_sets,
    compute_q_scores, generate_scenarios, resources_of,
)
from genetic_algorithm import GAConfig, run_ga_multiple_seeds
from recourse_milp import evaluate_dual_recourse
from sensitivity_hamad import regenerate_flows_at_alpha, NOMINAL_ALPHA

DELTA_MAP = {
    "low":  (30.0, 60.0, 90.0),
    "high": (60.0, 90.0, 120.0),
    "wide": (30.0, 60.0, 90.0, 120.0),
}

# charge TOUTES les combinaisons
path = "sensitivity_output/sensitivity_aggregated_29mai.csv"
if not Path(path).exists():
    path = "sensitivity_output/sensitivity_aggregated.csv"

combos = []
with open(path, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        combos.append({
            "alpha": float(r["alpha"]),
            "delta_name": r["delta"],
            "delta_values": DELTA_MAP[r["delta"]],
            "beta": float(r["beta"]),
            "gamma_g": int(r["gamma_g"]),
            "K": int(r["K"]),
            "milp_rob_raw": r.get("x_star_delta_rob_mean", ""),
        })

print(f"Loaded {len(combos)} combos from {path}\n")

inst = load_hamad_instance("hamad_instance.json")
results = []

for i, combo in enumerate(combos, 1):
    alpha = combo["alpha"]
    delta_name = combo["delta_name"]
    delta_values = combo["delta_values"]
    beta = combo["beta"]
    gamma_g = combo["gamma_g"]
    K = combo["K"]

    if abs(alpha - NOMINAL_ALPHA) < 1e-6:
        flows = inst["flows"]
    else:
        flows = regenerate_flows_at_alpha(inst, alpha)

    reduced = build_reduced_instance(
        rotations=inst["rotations"], passenger_flows=flows,
        gates=inst["gates"], zones_dict=inst["zones"],
        compatibility=inst["compatibility"],
        walking_times=inst["walking_times"], stand_id=0,
    )
    q = compute_q_scores(reduced.rotations, reduced.passenger_flows)
    ref = build_reference_assignment(
        reduced.rotations, reduced.gates, reduced.compatibility,
        inst["t_min"], stand_id=0,
    )
    _, stress = build_stress_sets(reduced.rotations, q)
    scen = generate_scenarios(
        reduced.zones, gamma_g, delta_values, beta, ref,
        reduced.rotations, reduced.passenger_flows, stand_id=0,
    )
    res = resources_of(reduced.gates, 0)

    ga_runs = run_ga_multiple_seeds(
        GAConfig(population_size=60, max_generations=50, seeds=(1, 2, 3, 4, 5)),
        rotations=reduced.rotations,
        passenger_flows=reduced.passenger_flows,
        gates=reduced.gates,
        compatibility=reduced.compatibility,
        walking_times=reduced.walking_times,
        q_scores=q, stress_sets=stress, scenarios=scen,
        resources=res, t_min=inst["t_min"], K=K, stand_id=0,
    )
    ga_best = min(ga_runs, key=lambda x: x.best_individual.ranking.key)

    if ga_best.best_individual.plan is None:
        I_pi_ga, J_ga, J_pi_ga, drob_ga = -1, None, None, None
    else:
        dual = evaluate_dual_recourse(
            ga_best.best_individual.plan, scen,
            reduced.rotations, reduced.passenger_flows, reduced.gates,
            reduced.compatibility, reduced.walking_times,
            q, stress, res, inst["t_min"], K, stand_id=0,
            time_limit_ms=60_000,
        )
        I_pi_ga = dual.I_pi
        J_ga = dual.J3_star
        J_pi_ga = dual.J3_pi
        drob_ga = dual.delta_rob

    # catégorie du MILP pour comparaison
    milp_rob = combo["milp_rob_raw"]
    if milp_rob == "" or milp_rob == "None":
        milp_cat = "inf"
    elif milp_rob in ("0", "0.0"):
        milp_cat = "opt"
    else:
        milp_cat = "sub"

    status = "opt" if drob_ga == 0 else ("inf" if I_pi_ga > 0 else "sub")
    print(f"[{i:2d}/{len(combos)}] a={alpha:.2f} d={delta_name:>4s} G={gamma_g} b={beta:.0f} K={K} | "
          f"MILP={milp_cat:>3s} -> GA: Iπ={I_pi_ga:2d} {status}")

    results.append({
        "alpha": alpha, "delta": delta_name, "gamma_g": gamma_g,
        "beta": beta, "K": K,
        "milp_category": milp_cat,
        "I_pi_ga": I_pi_ga, "J_ga": J_ga, "J_pi_ga": J_pi_ga,
        "delta_rob_ga": drob_ga, "ga_status": status,
    })

out = Path("sensitivity_output/ga_all_74.csv")
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

# résumé
n_inf_milp = sum(1 for r in results if r["milp_category"] == "inf")
n_opt_milp = sum(1 for r in results if r["milp_category"] == "opt")
n_sub_milp = sum(1 for r in results if r["milp_category"] == "sub")

n_ga_adm = sum(1 for r in results if r["I_pi_ga"] == 0)
n_ga_opt = sum(1 for r in results if r["ga_status"] == "opt")
n_ga_inf = sum(1 for r in results if r["ga_status"] == "inf")

print(f"\n=== SUMMARY ===")
print(f"Total combos     : {len(results)}")
print(f"MILP: opt={n_opt_milp}, sub={n_sub_milp}, inf={n_inf_milp}")
print(f"GA  : opt={n_ga_opt}, sub={n_ga_adm - n_ga_opt}, inf={n_ga_inf}")
print(f"GA rescues (MILP inf -> GA admiss): "
      f"{sum(1 for r in results if r['milp_category']=='inf' and r['I_pi_ga']==0)}/{n_inf_milp}")
print(f"Written: {out}")