"""Rerun GA on the 8 suboptimal combos from May 29 to recover GA comparison."""
import csv
from pathlib import Path

from hamad_adapter import load_hamad_instance
from gate_model import (
    build_reduced_instance, build_reference_assignment,
    build_stress_sets, compute_q_scores, generate_scenarios, resources_of,
)
from genetic_algorithm import GAConfig, run_ga_multiple_seeds
from recourse_milp import evaluate_dual_recourse

# Load instance
inst = load_hamad_instance("hamad_instance.json")
inst["K"] = 5

# Load suboptimal combos from aggregated CSV
suboptimal = []
with open("sensitivity_output/sensitivity_aggregated.csv") as f:
    for r in csv.DictReader(f):
        if r.get("x_star_delta_rob_mean") not in ("", "0", "0.0", None):
            suboptimal.append({
                "alpha": float(r["alpha"]),
                "delta": r["delta"],
                "gamma_g": int(r["gamma_g"]),
                "K": int(r["K"]),
            })

print(f"Found {len(suboptimal)} suboptimal combos")
for c in suboptimal:
    print(f"  α={c['alpha']} δ={c['delta']} Γ={c['gamma_g']} K={c['K']}")

DELTA_SETS = {
    "low": (30.0, 60.0, 90.0),
    "high": (60.0, 90.0, 120.0),
    "wide": (30.0, 60.0, 90.0, 120.0),
}

# Regenerate flows at each alpha (same as sensitivity_hamad.py)
from sensitivity_hamad import regenerate_flows_at_alpha, NOMINAL_ALPHA

results = []
for combo in suboptimal:
    alpha = combo["alpha"]
    if abs(alpha - NOMINAL_ALPHA) < 1e-6:
        flows = inst["flows"]
    else:
        flows = regenerate_flows_at_alpha(inst, alpha)

    reduced = build_reduced_instance(
        rotations=inst["rotations"],
        passenger_flows=flows,
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
        reduced.zones, combo["gamma_g"],
        DELTA_SETS[combo["delta"]], inst["beta_max"], ref,
        reduced.rotations, reduced.passenger_flows, stand_id=0,
    )
    resources = resources_of(reduced.gates, 0)

    print(f"\n>>> Rerun GA for α={alpha} δ={combo['delta']} Γ={combo['gamma_g']} K={combo['K']}")
    ga_runs = run_ga_multiple_seeds(
        GAConfig(population_size=60, max_generations=50, seeds=(1, 2, 3, 4, 5)),
        rotations=reduced.rotations,
        passenger_flows=reduced.passenger_flows,
        gates=reduced.gates,
        compatibility=reduced.compatibility,
        walking_times=reduced.walking_times,
        q_scores=q, stress_sets=stress, scenarios=scenarios,
        resources=resources, t_min=inst["t_min"], K=combo["K"],
        stand_id=0,
    )
    ga_best = min(ga_runs, key=lambda r: r.best_individual.ranking.key)

    dual_ga = None
    if ga_best.best_individual.plan is not None:
        dual_ga = evaluate_dual_recourse(
            ga_best.best_individual.plan, scenarios,
            reduced.rotations, reduced.passenger_flows, reduced.gates,
            reduced.compatibility, reduced.walking_times,
            q, stress, resources, inst["t_min"], combo["K"],
            stand_id=0, time_limit_ms=600_000,
        )

    row = {
        "alpha": alpha, "delta": combo["delta"],
        "gamma_g": combo["gamma_g"], "K": combo["K"],
        "ga_best_seed": ga_best.seed,
        "ga_best_J3_pi": ga_best.best_individual.ranking.J3_pi,
        "ga_best_delta_rob": dual_ga.delta_rob if dual_ga else None,
        "ga_seed_J3_pi": [r.best_individual.ranking.J3_pi for r in ga_runs],
    }
    print(f"    GA: seed={row['ga_best_seed']} J3_pi={row['ga_best_J3_pi']} "
          f"Δrob_GA={row['ga_best_delta_rob']} | "
          f"all seeds={row['ga_seed_J3_pi']}")
    results.append(row)

# Write results
outpath = Path("sensitivity_output/ga_suboptimal_results.csv")
with outpath.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    for r in results:
        r_copy = dict(r)
        r_copy["ga_seed_J3_pi"] = str(r["ga_seed_J3_pi"])
        writer.writerow(r_copy)

print(f"\nWrote {outpath}")
print("\n=== SUMMARY ===")
print(f"{'α':>5} {'δ':>5} {'Γ':>2} {'K':>2} | {'π Δrob':>8} {'GA Δrob':>8} | {'GA beats π?':>12}")
for r in results:
    ga_beats = "YES" if r["ga_best_delta_rob"] is not None and r["ga_best_delta_rob"] < 490 else "NO"
    print(f"{r['alpha']:>5} {r['delta']:>5} {r['gamma_g']:>2} {r['K']:>2} | "
          f"{'?':>8} {r['ga_best_delta_rob']!s:>8} | {ga_beats:>12}")