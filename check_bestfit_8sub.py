# check_bestfit_8sub.py
from hamad_adapter import load_hamad_instance
from gate_model import (
    build_reduced_instance, build_reference_assignment, build_stress_sets,
    compute_q_scores, generate_scenarios, resources_of,
)
from baselines import greedy_best_fit
from recourse_milp import evaluate_dual_recourse
from sensitivity_hamad import regenerate_flows_at_alpha, NOMINAL_ALPHA

suboptimal = [
    (0.40, (30., 60., 90.), 1, 4),
    (0.40, (60., 90., 120.), 2, 4),
    (0.40, (30., 60., 90., 120.), 1, 4),
    (0.53, (30., 60., 90.), 2, 4),
    (0.53, (60., 90., 120.), 1, 4),
    (0.53, (30., 60., 90., 120.), 2, 4),
    (0.60, (30., 60., 90.), 1, 4),
    (0.60, (30., 60., 90., 120.), 1, 4),
]

inst = load_hamad_instance("hamad_instance.json")
print(f"{'α':>5} {'δ-set':>22} {'Γ':>2} K | Iπ/23 | J(x) | Jπ(x) | Δrob | certified?")
for alpha, delta_set, gamma_g, K in suboptimal:
    flows = inst["flows"] if abs(alpha - NOMINAL_ALPHA) < 1e-6 else regenerate_flows_at_alpha(inst, alpha)
    reduced = build_reduced_instance(
        rotations=inst["rotations"], passenger_flows=flows,
        gates=inst["gates"], zones_dict=inst["zones"],
        compatibility=inst["compatibility"], walking_times=inst["walking_times"], stand_id=0,
    )
    q = compute_q_scores(reduced.rotations, reduced.passenger_flows)
    ref = build_reference_assignment(reduced.rotations, reduced.gates, reduced.compatibility, inst["t_min"], stand_id=0)
    _, stress = build_stress_sets(reduced.rotations, q)
    scen = generate_scenarios(reduced.zones, gamma_g, delta_set, 25.0, ref, reduced.rotations, reduced.passenger_flows, stand_id=0)
    res = resources_of(reduced.gates, 0)
    berth_lengths = {b: inst["berth_meta"][b]["length_m"] for b in reduced.gates}
    plan = greedy_best_fit(reduced.rotations, reduced.gates, reduced.compatibility, inst["t_min"], berth_lengths=berth_lengths)
    dual = evaluate_dual_recourse(
        plan, scen, reduced.rotations, reduced.passenger_flows, reduced.gates,
        reduced.compatibility, reduced.walking_times, q, stress, res,
        inst["t_min"], K, stand_id=0, time_limit_ms=600_000,
    )
    ds = f"{delta_set[0]:.0f}-{delta_set[-1]:.0f}"
    print(f"{alpha:>5.2f} {ds:>22} {gamma_g:>2} {K} | {dual.I_pi:>2}/23 | {dual.J3_star} | {dual.J3_pi} | {dual.delta_rob} | {'YES' if dual.delta_rob == 0 else 'NO' if dual.delta_rob else 'infeasible'}")