"""Synthetic smoke test — simple demonstration of the pipeline.

Run with: python example_run.py
or via CLI: python main.py example
"""
from gate_model import (
    PassengerFlow, Rotation, build_reference_assignment, build_stress_sets,
    compute_q_scores, decode_chromosome, evaluate_robust_protocol,
    generate_scenarios, resources_of,
)


def main() -> None:
    print("=" * 70)
    print("SYNTHETIC SMOKE TEST — not empirical airport data")
    print("=" * 70)

    rotations = [
        Rotation("A", 100, 140, "M", "W1"),
        Rotation("B", 180, 220, "M", "W1"),
        Rotation("C", 110, 150, "M", "W2"),
        Rotation("D", 260, 300, "M", "W2"),
    ]
    flows = [
        PassengerFlow("P1", "A", "B", 10, 25),
        PassengerFlow("P2", "C", "D", 8, 25),
    ]
    gates = [1, 2, 3, 4]
    zones = {"Z1": [1, 2], "Z2": [3, 4]}
    compatibility = {(r.id, g): True for r in rotations for g in gates}
    compatibility.update({(r.id, 0): True for r in rotations})
    walking_times = {
        (i, j): (0 if i == j else 5)
        for i in [0] + gates for j in [0] + gates
    }

    print("\n1. Computing q-scores...")
    q = compute_q_scores(rotations, flows)
    print(f"   q-scores: {q}")

    print("\n2. Building reference assignment...")
    ref = build_reference_assignment(rotations, gates, compatibility, t_min=10)
    print(f"   reference: {ref}")

    print("\n3. Building stress sets (γ_f = max(⌊|M_v|/2⌋))...")
    gamma_f, stress_sets = build_stress_sets(rotations, q)
    print(f"   γ_f = {gamma_f}, stress_sets: {stress_sets}")

    print("\n4. Generating scenarios (§2.3)...")
    scenarios = generate_scenarios(
        zones, gamma_g=1, delta_values=[0, 30, 60], beta_max=15,
        reference_assignment=ref, rotations=rotations, passenger_flows=flows,
    )
    print(f"   {len(scenarios)} scenarios generated")
    families = {s.family for s in scenarios}
    print(f"   families: {families}")

    print("\n5. Decoding a nominal plan (§2.7)...")
    rho = tuple(r.id for r in rotations)
    sigma = {rid: ref[rid] for rid in ref}
    nominal = decode_chromosome(
        rho, sigma, rotations, gates, compatibility, q, t_min=10,
    )
    print(f"   nominal plan: {nominal}")

    print("\n6. Evaluating robustness under π (§2.6)...")
    resources = resources_of(gates)
    evaluation = evaluate_robust_protocol(
        nominal, scenarios, rotations, flows, gates, compatibility,
        walking_times, q, stress_sets, t_min=10, K=len(rotations),
    )
    print(f"   robustly admissible (π): {evaluation.admissible}")
    print(f"   worst-case broken passengers: {evaluation.worst_case_broken_passengers}")
    failed = [sid for sid, result in evaluation.scenario_results.items() if not result.feasible]
    print(f"   failed scenarios: {failed}")

    print("\n" + "=" * 70)
    print("Smoke test complete — all basic functions operational ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
