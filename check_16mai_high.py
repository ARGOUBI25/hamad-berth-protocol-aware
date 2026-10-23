import time
from hamad_adapter import load_hamad_instance
from gate_model import (
    build_reduced_instance, build_reference_assignment, build_stress_sets,
    compute_q_scores, generate_scenarios, resources_of,
)
from baselines import greedy_first_fit, greedy_best_fit, deterministic_milp
from genetic_algorithm import GAConfig, run_ga_multiple_seeds
from joint_milp import solve_joint_milp
from recourse_milp import evaluate_dual_recourse

inst = load_hamad_instance("hamad_instance_20250516.json")
alpha, K = 0.53, 4
delta_set = (60.0, 90.0, 120.0)  # "high"
gamma_g, beta = 1, 25.0

# regénère les flux à alpha=0.53 (au cas où alpha≠nominal)
from sensitivity_hamad import regenerate_flows_at_alpha, NOMINAL_ALPHA
flows = inst["flows"] if abs(alpha - NOMINAL_ALPHA) < 1e-6 else regenerate_flows_at_alpha(inst, alpha)

reduced = build_reduced_instance(
    rotations=inst["rotations"], passenger_flows=flows,
    gates=inst["gates"], zones_dict=inst["zones"],
    compatibility=inst["compatibility"], walking_times=inst["walking_times"],
    stand_id=0,
)
q = compute_q_scores(reduced.rotations, reduced.passenger_flows)
ref = build_reference_assignment(
    reduced.rotations, reduced.gates, reduced.compatibility, inst["t_min"], stand_id=0,
)
_, stress = build_stress_sets(reduced.rotations, q)
scenarios = generate_scenarios(
    reduced.zones, gamma_g, delta_set, beta, ref,
    reduced.rotations, reduced.passenger_flows, stand_id=0,
)
resources = resources_of(reduced.gates, 0)

def eval_plan(label, plan):
    dual = evaluate_dual_recourse(
        plan, scenarios, reduced.rotations, reduced.passenger_flows,
        reduced.gates, reduced.compatibility, reduced.walking_times,
        q, stress, resources, inst["t_min"], K, stand_id=0,
        time_limit_ms=600_000,
    )
    print(f"  {label:25s} | Iπ={dual.I_pi:2d}/23 | J(x)={dual.J3_star} | "
          f"Jπ(x)={dual.J3_pi} | Δrob={dual.delta_rob}")

print(f"=== Instance 16 mai, α={alpha}, δ=high, Γ={gamma_g}, β={beta}, K={K} ===")
print(f"Scenarios: {len(scenarios)}")

# joint MILP
print("\n>>> Joint robust MILP")
t0 = time.perf_counter()
joint = solve_joint_milp(reduced, scenarios, stress, resources, inst["t_min"], K, time_limit_ms=600_000)
print(f"  Status={joint.status} phase={joint.phase_reached} J*={joint.J3_star} gap={joint.gap}")
if joint.nominal_plan:
    eval_plan("Joint MILP x*", joint.nominal_plan)

# GA
print("\n>>> GA (5 seeds)")
ga_runs = run_ga_multiple_seeds(
    GAConfig(population_size=60, max_generations=50, seeds=(1,2,3,4,5)),
    rotations=reduced.rotations, passenger_flows=reduced.passenger_flows,
    gates=reduced.gates, compatibility=reduced.compatibility,
    walking_times=reduced.walking_times, q_scores=q, stress_sets=stress,
    scenarios=scenarios, resources=resources, t_min=inst["t_min"], K=K, stand_id=0,
)
ga_best = min(ga_runs, key=lambda r: r.best_individual.ranking.key)
if ga_best.best_individual.plan:
    eval_plan(f"GA (best seed {ga_best.seed})", ga_best.best_individual.plan)

# Baselines
print("\n>>> Baselines")
plan_ff = greedy_first_fit(reduced.rotations, reduced.gates, reduced.compatibility, inst["t_min"])
eval_plan("Greedy first-fit", plan_ff)
plan_bf = greedy_best_fit(
    reduced.rotations, reduced.gates, reduced.compatibility, inst["t_min"],
    berth_lengths={b: inst["berth_meta"][b]["length_m"] for b in reduced.gates},
)
eval_plan("Greedy best-fit", plan_bf)
plan_det = deterministic_milp(reduced, resources, inst["t_min"], K, stress, time_limit_ms=600_000)
eval_plan("Deterministic MILP", plan_det)