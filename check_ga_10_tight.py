"""Lance le GA (5 seeds) sur les 10 combinaisons de l'instance tight
(16 mai), lues depuis sensitivity_output/sensitivity_raw.csv, et croise
avec la catégorie MILP déjà enregistrée."""

import ast
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

CSV_PATH = "sensitivity_output/sensitivity_raw.csv"
if not Path(CSV_PATH).exists():
    raise FileNotFoundError(f"{CSV_PATH} introuvable")

# ---- 1. Charger les 10 combinaisons tight ----
combos = []
with open(CSV_PATH, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rob = (r.get("x_star_delta_rob") or "").strip()
        if rob == "" or rob == "None":
            milp_cat = "inf"
        elif rob in ("0", "0.0"):
            milp_cat = "exact"
        else:
            milp_cat = "sub"

        J_star_raw = (r.get("joint_J3_star") or "").strip()
        J_star = int(float(J_star_raw)) if J_star_raw else None

        combos.append({
            "alpha": float(r["alpha"]),
            "delta_name": r["delta_name"],
            "delta_values": tuple(ast.literal_eval(r["delta_values"])),
            "beta": float(r["beta"]),
            "gamma_g": int(r["gamma_g"]),
            "K": int(r["K"]),
            "milp_cat": milp_cat,
            "J_star": J_star,
        })

print(f"Loaded {len(combos)} tight combos from {CSV_PATH}")
print(f"Expected: 10\n")

if len(combos) != 10:
    print(f"WARNING: expected 10 combos, got {len(combos)}. Continuing anyway.\n")

# ---- 2. Lancer le GA sur chaque ----
inst = load_hamad_instance("hamad_instance_20250516.json")
results = []

for i, combo in enumerate(combos, 1):
    alpha = combo["alpha"]
    delta_name = combo["delta_name"]
    delta_values = combo["delta_values"]
    beta = combo["beta"]
    gamma_g = combo["gamma_g"]
    K = combo["K"]
    milp_cat = combo["milp_cat"]
    J_star = combo["J_star"]

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
        ga_cat = "no_plan"
        I_pi_ga, J_ga, J_pi_ga, drob_ga, cert = -1, None, None, None, "-"
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
        if I_pi_ga > 0:
            ga_cat = "inf"
        elif drob_ga == 0:
            ga_cat = "exact"
        else:
            ga_cat = "sub"
        # certification : J_π(GA) == J* (si J* prouvé)
        if J_star is not None and J_pi_ga == J_star:
            cert = "yes"
        elif J_star is None:
            cert = "no J*"
        else:
            cert = "no"

    print(f"[{i:2d}/{len(combos)}] a={alpha:.2f} d={delta_name:>4s} "
          f"G={gamma_g} b={beta:.0f} K={K} | MILP={milp_cat:>5s} "
          f"-> GA: Ipi={I_pi_ga:2d} {ga_cat} | cert={cert}")

    results.append({
        "alpha": alpha, "delta": delta_name, "gamma_g": gamma_g,
        "beta": beta, "K": K,
        "milp_cat": milp_cat, "J_star": J_star,
        "I_pi_ga": I_pi_ga, "J_ga": J_ga, "J_pi_ga": J_pi_ga,
        "delta_rob_ga": drob_ga, "ga_cat": ga_cat, "cert": cert,
    })

# ---- 3. Écrire le CSV ----
out = Path("sensitivity_output/ga_10_tight.csv")
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

# ---- 4. Résumé ----
n_milp_exact = sum(1 for r in results if r["milp_cat"] == "exact")
n_milp_sub = sum(1 for r in results if r["milp_cat"] == "sub")
n_milp_inf = sum(1 for r in results if r["milp_cat"] == "inf")
n_ga_exact = sum(1 for r in results if r["ga_cat"] == "exact")
n_ga_sub = sum(1 for r in results if r["ga_cat"] == "sub")
n_ga_inf = sum(1 for r in results if r["ga_cat"] == "inf")
n_cert = sum(1 for r in results if r["cert"] == "yes")
n_rescue = sum(
    1 for r in results
    if r["milp_cat"] == "inf" and r["ga_cat"] == "exact"
)

print(f"\n=== SUMMARY (tight instance, 10 combos) ===")
print(f"MILP: exact={n_milp_exact}, sub={n_milp_sub}, inf={n_milp_inf}")
print(f"GA  : exact={n_ga_exact}, sub={n_ga_sub}, inf={n_ga_inf}")
print(f"GA certified π-optimal: {n_cert}/{len(results)}")
print(f"GA rescues (MILP inf -> GA exact): {n_rescue}/{n_milp_inf}")
print(f"Written: {out}")