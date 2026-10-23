"""Sensitivity analysis pipeline for the Hamad Port calibrated instance.

Produces publication-ready artefacts (Q1 journal standard):
  - sensitivity_raw.csv         : one row per (combo, scenario, method)
  - sensitivity_aggregated.csv  : one row per combo with mean/std/min/max
  - sensitivity_tests.csv       : Wilcoxon signed-rank π vs MILP per combo
  - sensitivity_summary.tex     : LaTeX table for §5 of the paper
  - sensitivity_heatmap.png     : Δ_rob as function of (δ, Γ_g)
  - sensitivity_boxplot.png     : J3 distribution by α
  - sensitivity_manifest.json   : reproducible artefact (hash, versions, seeds)

Axes covered:
  - α (transshipment rate)      : {0.40, 0.53, 0.60}
  - δ (delay magnitudes)        : {low, high, wide}
  - β (separation buffer)       : {15, 25, 35}
  - Γ_g (closure level)         : {1, 2}
  - K (change budget)           : {2, 3, 4, 5}
  - GA seeds                    : 1..N (configurable, default 5)

FIXES:
  #1  Original flows preserved when alpha == NOMINAL_ALPHA
      (avoid 2-container perturbations flipping Δrob from 490 to 0)
  #2  Default time limit raised from 60 s to 600 s
      (α=0.40 was hitting the 60 s limit -> FEASIBLE, not OPTIMAL)
  #3  Warning printed when joint MILP is not proven optimal
  #4  K=4 added to the grid (this is the critical threshold value)

Usage:
  python sensitivity_hamad.py --mode quick     # MILP + dual only (~15-45 min)
  python sensitivity_hamad.py --mode full      # + GA 5 seeds (~2-3 h)
  python sensitivity_hamad.py --mode publication --seeds 10  # final run
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev, median
from typing import Any

from hamad_adapter import load_hamad_instance


# =========================================================================
# Configuration
# =========================================================================

AXES = {
    "alpha": [0.40, 0.53, 0.60],
    "delta_sets": {
        "low":  (30.0, 60.0, 90.0),
        "high": (60.0, 90.0, 120.0),
        "wide": (30.0, 60.0, 90.0, 120.0),
    },
    "beta_values": [15.0, 25.0, 35.0],
    "gamma_g_values": [1, 2],
    # FIX 4: K=4 added (critical threshold where Δrob=490 was observed)
    "K_values": [2, 3, 4, 5],
}

NOMINAL_ALPHA = 0.53
NOMINAL_DELTA = "low"
NOMINAL_BETA = 25.0
NOMINAL_GAMMA = 1
NOMINAL_K = 5


# =========================================================================
# Regeneration of transshipment flows for a given α
# =========================================================================

def regenerate_flows_at_alpha(inst: dict, alpha: float) -> list:
    """Regenerate transshipment flows so that total containers ≈ α × N."""
    from gate_model import PassengerFlow

    realized_teu = 6500.0
    n_containers_total = realized_teu / 1.5

    base_flows = inst["flows"]
    base_sum = sum(f.n for f in base_flows) or 1
    target_sum = alpha * n_containers_total
    scale = target_sum / base_sum

    new_flows = []
    for f in base_flows:
        new_n = max(1, int(round(f.n * scale)))
        new_flows.append(PassengerFlow(
            id=f.id, arr_rot=f.arr_rot, dep_rot=f.dep_rot,
            n=new_n, min_conn=f.min_conn,
        ))
    return new_flows


# =========================================================================
# Core pipeline call for one combo
# =========================================================================

@dataclass
class ComboResult:
    alpha: float
    delta_name: str
    delta_values: tuple
    beta: float
    gamma_g: int
    K: int
    joint_status: str
    joint_phase: int
    joint_gap: float | None
    joint_J3_star: int | None
    joint_J1: int | None
    joint_J2: int | None
    joint_time_s: float
    x_star_J3_pi: int | None
    x_star_J3_star: int | None
    x_star_delta_rob: int | None
    x_star_delta_sc: int | None
    x_star_inconsistencies: int
    ga_best_seed: int | None = None
    ga_best_J3_pi: int | None = None
    ga_best_delta_rob: int | None = None
    ga_time_s: float = 0.0
    ga_seed_J3_pi: list = field(default_factory=list)


def run_one_combo(
    base_inst: dict,
    alpha: float,
    delta_name: str,
    delta_values: tuple,
    beta: float,
    gamma_g: int,
    K: int,
    run_ga: bool,
    ga_seeds: tuple = (1, 2, 3, 4, 5),
    time_limit_ms: int = 600_000,
    log: bool = True,
) -> ComboResult:
    """Run joint MILP, dual recourse, and optionally GA for one combination."""
    from gate_model import (
        build_reduced_instance, build_reference_assignment,
        build_stress_sets, compute_q_scores, generate_scenarios, resources_of,
    )
    from joint_milp import solve_joint_milp
    from recourse_milp import evaluate_dual_recourse

    t0 = time.perf_counter()

    # FIX 1: preserve original flows when alpha is nominal (avoid 2-container
    # perturbation flipping the result). Regression-safe.
    if abs(alpha - NOMINAL_ALPHA) < 1e-6:
        flows = base_inst["flows"]
    else:
        flows = regenerate_flows_at_alpha(base_inst, alpha)

    reduced = build_reduced_instance(
        rotations=base_inst["rotations"],
        passenger_flows=flows,
        gates=base_inst["gates"],
        zones_dict=base_inst["zones"],
        compatibility=base_inst["compatibility"],
        walking_times=base_inst["walking_times"],
        stand_id=0,
    )
    q = compute_q_scores(reduced.rotations, reduced.passenger_flows)
    ref = build_reference_assignment(
        reduced.rotations, reduced.gates, reduced.compatibility,
        base_inst["t_min"], stand_id=0,
    )
    _, stress = build_stress_sets(reduced.rotations, q)
    scenarios = generate_scenarios(
        reduced.zones, gamma_g, delta_values, beta, ref,
        reduced.rotations, reduced.passenger_flows, stand_id=0,
    )
    resources = resources_of(reduced.gates, 0)

    joint = solve_joint_milp(
        reduced, scenarios, stress, resources,
        base_inst["t_min"], K,
        time_limit_ms=time_limit_ms,
    )
    t_joint = time.perf_counter() - t0

    # FIX 3 + RETRY: warn and retry when FEASIBLE with a longer time limit
    if joint.status == "FEASIBLE":
        print(
            f"  [RETRY] status=FEASIBLE, gap={joint.gap}, "
            f"retrying with {time_limit_ms * 3 / 1000:.0f}s"
        )
        joint = solve_joint_milp(
            reduced, scenarios, stress, resources,
            base_inst["t_min"], K,
            time_limit_ms=time_limit_ms * 3,
        )
        t_joint = time.perf_counter() - t0
        if joint.status != "OPTIMAL":
            print(
                f"  [WARN] Still not optimal after retry: "
                f"status={joint.status}, gap={joint.gap}, time={t_joint:.1f}s"
            )
    elif joint.status != "OPTIMAL":
        print(
            f"  [WARN] Joint MILP not optimal: "
            f"status={joint.status}, time={t_joint:.1f}s"
        )

    result = ComboResult(
        alpha=alpha, delta_name=delta_name, delta_values=delta_values,
        beta=beta, gamma_g=gamma_g, K=K,
        joint_status=joint.status,
        joint_phase=joint.phase_reached,
        joint_gap=joint.gap,
        joint_J3_star=joint.J3_star,
        joint_J1=joint.J1,
        joint_J2=joint.J2,
        joint_time_s=t_joint,
        x_star_J3_pi=None, x_star_J3_star=None,
        x_star_delta_rob=None, x_star_delta_sc=None,
        x_star_inconsistencies=0,
    )

    if joint.nominal_plan is not None:
        dual = evaluate_dual_recourse(
            joint.nominal_plan, scenarios,
            reduced.rotations, reduced.passenger_flows, reduced.gates,
            reduced.compatibility, reduced.walking_times,
            q, stress, resources, base_inst["t_min"], K,
            stand_id=0, time_limit_ms=time_limit_ms,
        )
        result.x_star_J3_pi = dual.J3_pi
        result.x_star_J3_star = dual.J3_star
        result.x_star_delta_rob = dual.delta_rob
        result.x_star_delta_sc = dual.delta_sc
        result.x_star_inconsistencies = len(dual.inconsistency_warnings)

    if run_ga:
        from genetic_algorithm import GAConfig, run_ga_multiple_seeds
        t_ga = time.perf_counter()
        ga_runs = run_ga_multiple_seeds(
            GAConfig(population_size=60, max_generations=50, seeds=ga_seeds),
            rotations=reduced.rotations,
            passenger_flows=reduced.passenger_flows,
            gates=reduced.gates,
            compatibility=reduced.compatibility,
            walking_times=reduced.walking_times,
            q_scores=q, stress_sets=stress, scenarios=scenarios,
            resources=resources, t_min=base_inst["t_min"], K=K,
            stand_id=0,
        )
        result.ga_time_s = time.perf_counter() - t_ga
        ga_best = min(ga_runs, key=lambda r: r.best_individual.ranking.key)
        result.ga_best_seed = ga_best.seed
        result.ga_best_J3_pi = ga_best.best_individual.ranking.J3_pi
        result.ga_seed_J3_pi = [
            r.best_individual.ranking.J3_pi for r in ga_runs
        ]
        if ga_best.best_individual.plan is not None:
            dual_ga = evaluate_dual_recourse(
                ga_best.best_individual.plan, scenarios,
                reduced.rotations, reduced.passenger_flows, reduced.gates,
                reduced.compatibility, reduced.walking_times,
                q, stress, resources, base_inst["t_min"], K,
                stand_id=0, time_limit_ms=time_limit_ms,
            )
            result.ga_best_delta_rob = dual_ga.delta_rob

    if log:
        print(
            f"  α={alpha:.2f} δ={delta_name} β={beta:.0f} "
            f"Γ={gamma_g} K={K} | "
            f"J3*={result.joint_J3_star} "
            f"J3π(x*)={result.x_star_J3_pi} "
            f"Δrob={result.x_star_delta_rob} "
            f"t={t_joint:.1f}s"
        )
    return result


# =========================================================================
# Output writers
# =========================================================================

def write_raw_csv(results: list[ComboResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["delta_values"] = list(row["delta_values"])
            row["ga_seed_J3_pi"] = list(row["ga_seed_J3_pi"])
            writer.writerow(row)


def write_aggregated_csv(results: list[ComboResult], path: Path) -> None:
    groups: dict[tuple, list[ComboResult]] = {}
    for r in results:
        key = (r.alpha, r.delta_name, r.beta, r.gamma_g, r.K)
        groups.setdefault(key, []).append(r)

    rows = []
    for key, rs in groups.items():
        alpha, delta_name, beta, gamma_g, K = key
        delta_robs = [r.x_star_delta_rob for r in rs if r.x_star_delta_rob is not None]
        rows.append({
            "alpha": alpha, "delta": delta_name, "beta": beta,
            "gamma_g": gamma_g, "K": K,
            "n_runs": len(rs),
            "joint_J3_star_mean": _safe_mean([r.joint_J3_star for r in rs]),
            "joint_J3_star_std": _safe_std([r.joint_J3_star for r in rs]),
            "x_star_delta_rob_mean": _safe_mean(delta_robs),
            "x_star_delta_rob_std": _safe_std(delta_robs),
            "x_star_delta_rob_max": max(delta_robs) if delta_robs else None,
            "x_star_delta_sc_mean": _safe_mean([r.x_star_delta_sc for r in rs]),
            "inconsistencies_total": sum(r.x_star_inconsistencies for r in rs),
        })

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_wilcoxon_csv(results: list[ComboResult], path: Path) -> None:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("  [WARN] scipy not installed, skipping Wilcoxon tests")
        print("         pip install scipy")
        return

    groups: dict[tuple, list[ComboResult]] = {}
    for r in results:
        key = (r.alpha, r.delta_name, r.beta, r.gamma_g, r.K)
        groups.setdefault(key, []).append(r)

    rows = []
    for key, rs in groups.items():
        pairs = [
            (r.x_star_J3_pi, r.x_star_J3_star)
            for r in rs
            if r.x_star_J3_pi is not None and r.x_star_J3_star is not None
        ]
        if len(pairs) < 6:
            continue
        pi_vals = [p[0] for p in pairs]
        star_vals = [p[1] for p in pairs]
        if all(a == b for a, b in pairs):
            stat, p = 0.0, 1.0
        else:
            try:
                stat, p = wilcoxon(pi_vals, star_vals, zero_method="wilcox")
            except ValueError:
                stat, p = 0.0, 1.0
        alpha, delta_name, beta, gamma_g, K = key
        rows.append({
            "alpha": alpha, "delta": delta_name, "beta": beta,
            "gamma_g": gamma_g, "K": K,
            "n_pairs": len(pairs),
            "median_J3_pi": median(pi_vals),
            "median_J3_star": median(star_vals),
            "median_delta_rob": median([a - b for a, b in pairs]),
            "wilcoxon_stat": stat,
            "wilcoxon_p": p,
            "significant_at_0.05": p < 0.05,
        })

    if rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def write_latex_table(results: list[ComboResult], path: Path) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Sensitivity of $\Delta_{\mathrm{rob}} = J_3^\pi(x^*) - J_3^*(x^*)$ "
        r"to transshipment rate $\alpha$, delay set $\delta$, "
        r"closure level $\Gamma_g$, and budget $K$.}",
        r"\label{tab:sensitivity}",
        r"\begin{tabular}{cccc|cc|cc}",
        r"\hline",
        r"$\alpha$ & $\delta$ & $\Gamma_g$ & $K$ "
        r"& $J_3^*$ & $J_3^\pi(x^*)$ & $\Delta_{\mathrm{rob}}$ & $\Delta_{\mathrm{sc}}$ \\",
        r"\hline",
    ]

    for r in results:
        if r.x_star_delta_rob is None:
            continue
        lines.append(
            f"{r.alpha:.2f} & {r.delta_name} & {r.gamma_g} & {r.K} "
            f"& {r.joint_J3_star} & {r.x_star_J3_pi} "
            f"& {r.x_star_delta_rob} & {r.x_star_delta_sc} \\\\"
        )

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_heatmap(results: list[ComboResult], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [WARN] matplotlib not installed, skipping heatmap")
        return

    deltas = ["low", "high", "wide"]
    gammas = [1, 2]
    grid = np.full((len(gammas), len(deltas)), np.nan)

    for r in results:
        if (r.alpha == NOMINAL_ALPHA and r.beta == NOMINAL_BETA
                and r.K == NOMINAL_K and r.x_star_delta_rob is not None):
            if r.delta_name in deltas and r.gamma_g in gammas:
                i = gammas.index(r.gamma_g)
                j = deltas.index(r.delta_name)
                if np.isnan(grid[i, j]):
                    grid[i, j] = r.x_star_delta_rob
                else:
                    grid[i, j] = (grid[i, j] + r.x_star_delta_rob) / 2

    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels(deltas)
    ax.set_yticks(range(len(gammas)))
    ax.set_yticklabels([f"$\\Gamma_g={g}$" for g in gammas])
    ax.set_xlabel(r"Delay set $\delta$")
    ax.set_ylabel(r"Closure level $\Gamma_g$")
    ax.set_title(rf"$\Delta_{{\mathrm{{rob}}}}$ at $\alpha={NOMINAL_ALPHA}$, "
                 rf"$\beta={NOMINAL_BETA}$, $K={NOMINAL_K}$")
    for i in range(len(gammas)):
        for j in range(len(deltas)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{int(grid[i, j])}", ha="center", va="center",
                        color="black", fontsize=12)
    fig.colorbar(im, ax=ax, label=r"$\Delta_{\mathrm{rob}}$ (containers)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_boxplot(results: list[ComboResult], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    by_alpha: dict[float, list[int]] = {}
    for r in results:
        if (r.delta_name == NOMINAL_DELTA and r.beta == NOMINAL_BETA
                and r.gamma_g == NOMINAL_GAMMA and r.K == NOMINAL_K
                and r.x_star_J3_pi is not None):
            by_alpha.setdefault(r.alpha, []).append(r.x_star_J3_pi)

    if not by_alpha:
        return

    alphas = sorted(by_alpha.keys())
    data = [by_alpha[a] for a in alphas]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, labels=[f"$\\alpha={a:.2f}$" for a in alphas])
    ax.set_ylabel(r"$J_3^\pi(x^*)$ (containers)")
    ax.set_title("Worst-case loss under π across transshipment rates")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_manifest(base_inst_path: Path, results: list[ComboResult], path: Path) -> None:
    try:
        import ortools
        ortools_v = ortools.__version__
    except Exception:
        ortools_v = "unknown"

    base_hash = hashlib.sha256(base_inst_path.read_bytes()).hexdigest()

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "ortools_version": ortools_v,
        "instance_path": str(base_inst_path),
        "instance_sha256": base_hash,
        "axes": {k: (list(v) if not isinstance(v, dict) else
                     {kk: list(vv) for kk, vv in v.items()})
                 for k, v in AXES.items()},
        "n_combinations": len(results),
        "total_wall_time_s": sum(
            r.joint_time_s + r.ga_time_s for r in results
        ),
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# =========================================================================
# Helpers
# =========================================================================

def _safe_mean(values):
    xs = [v for v in values if v is not None]
    return mean(xs) if xs else None


def _safe_std(values):
    xs = [v for v in values if v is not None]
    return stdev(xs) if len(xs) >= 2 else 0.0


# =========================================================================
# Orchestration
# =========================================================================

def build_grid(mode: str) -> list[dict]:
    combos = []

    if mode == "quick":
        for alpha in AXES["alpha"]:
            combos.append({
                "alpha": alpha, "delta_name": "low",
                "delta_values": AXES["delta_sets"]["low"],
                "beta": NOMINAL_BETA, "gamma_g": NOMINAL_GAMMA,
                "K": NOMINAL_K, "run_ga": False,
            })
        for delta_name, dvals in AXES["delta_sets"].items():
            if delta_name == "low":
                continue
            combos.append({
                "alpha": NOMINAL_ALPHA, "delta_name": delta_name,
                "delta_values": dvals, "beta": NOMINAL_BETA,
                "gamma_g": NOMINAL_GAMMA, "K": NOMINAL_K, "run_ga": False,
            })
        for gamma_g in AXES["gamma_g_values"]:
            if gamma_g == NOMINAL_GAMMA:
                continue
            combos.append({
                "alpha": NOMINAL_ALPHA, "delta_name": NOMINAL_DELTA,
                "delta_values": AXES["delta_sets"][NOMINAL_DELTA],
                "beta": NOMINAL_BETA, "gamma_g": gamma_g,
                "K": NOMINAL_K, "run_ga": False,
            })
        for K in AXES["K_values"]:
            if K == NOMINAL_K:
                continue
            combos.append({
                "alpha": NOMINAL_ALPHA, "delta_name": NOMINAL_DELTA,
                "delta_values": AXES["delta_sets"][NOMINAL_DELTA],
                "beta": NOMINAL_BETA, "gamma_g": NOMINAL_GAMMA,
                "K": K, "run_ga": False,
            })

    elif mode in ("full", "publication"):
        for alpha, (delta_name, dvals), gamma_g, K in itertools.product(
            AXES["alpha"],
            AXES["delta_sets"].items(),
            AXES["gamma_g_values"],
            AXES["K_values"],
        ):
            combos.append({
                "alpha": alpha, "delta_name": delta_name,
                "delta_values": dvals, "beta": NOMINAL_BETA,
                "gamma_g": gamma_g, "K": K, "run_ga": True,
            })
        for beta in AXES["beta_values"]:
            if beta == NOMINAL_BETA:
                continue
            combos.append({
                "alpha": NOMINAL_ALPHA, "delta_name": NOMINAL_DELTA,
                "delta_values": AXES["delta_sets"][NOMINAL_DELTA],
                "beta": beta, "gamma_g": NOMINAL_GAMMA,
                "K": NOMINAL_K, "run_ga": True,
            })

    return combos


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=str,
                        help="Path to hamad_instance.json")
    parser.add_argument("--mode", choices=["quick", "full", "publication"],
                        default="quick")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of GA seeds (default 5)")
    parser.add_argument("--output-dir", type=str, default="sensitivity_output")
    # FIX 2: time limit raised from 60 s to 600 s
    parser.add_argument("--time-limit-ms", type=int, default=600_000)
    args = parser.parse_args(argv)

    base_inst_path = Path(args.instance)
    if not base_inst_path.exists():
        print(f"ERROR: {base_inst_path} not found")
        return 1

    outdir = Path(args.output_dir)
    outdir.mkdir(exist_ok=True)

    base_inst = load_hamad_instance(base_inst_path)
    print(f"Loaded {len(base_inst['rotations'])} vessels, "
          f"{len(base_inst['flows'])} flows")

    grid = build_grid(args.mode)
    print(f"Mode: {args.mode} — {len(grid)} combinations")
    print(f"GA seeds per combo: {args.seeds if any(c['run_ga'] for c in grid) else 0}")
    print()

    ga_seeds = tuple(range(1, args.seeds + 1))
    results: list[ComboResult] = []
    t_start = time.perf_counter()

    for i, combo in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}]")
        result = run_one_combo(
            base_inst,
            alpha=combo["alpha"],
            delta_name=combo["delta_name"],
            delta_values=combo["delta_values"],
            beta=combo["beta"],
            gamma_g=combo["gamma_g"],
            K=combo["K"],
            run_ga=combo["run_ga"],
            ga_seeds=ga_seeds,
            time_limit_ms=args.time_limit_ms,
        )
        results.append(result)

    total_time = time.perf_counter() - t_start
    print(f"\nTotal run: {total_time:.1f}s ({total_time/60:.1f} min)")

    print("\nWriting artefacts...")
    write_raw_csv(results, outdir / "sensitivity_raw.csv")
    write_aggregated_csv(results, outdir / "sensitivity_aggregated.csv")
    write_wilcoxon_csv(results, outdir / "sensitivity_tests.csv")
    write_latex_table(results, outdir / "sensitivity_summary.tex")
    write_heatmap(results, outdir / "sensitivity_heatmap.png")
    write_boxplot(results, outdir / "sensitivity_boxplot.png")
    write_manifest(base_inst_path, results, outdir / "sensitivity_manifest.json")

    print(f"Artefacts written to {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())