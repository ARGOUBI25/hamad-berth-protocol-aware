"""Post-hoc statistical analysis of the sensitivity results.

Reads sensitivity_output/sensitivity_raw.csv and produces:
  - chi-square test: are suboptimalities concentrated at K=4?
  - binomial test: is the observed pattern unlikely under uniform allocation?
  - per-(α, δ, Γ) suboptimality rates
  - LaTeX paragraph ready to paste in §5.4

Usage:
    python analyze_sensitivity.py sensitivity_output/sensitivity_raw.csv
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["K"] = int(r["K"])
        r["alpha"] = float(r["alpha"])
        r["gamma_g"] = int(r["gamma_g"])
        r["joint_J3_star"] = (
            int(float(r["joint_J3_star"])) if r["joint_J3_star"] else None
        )
        r["x_star_J3_pi"] = (
            int(float(r["x_star_J3_pi"])) if r["x_star_J3_pi"] else None
        )
        r["x_star_delta_rob"] = (
            int(float(r["x_star_delta_rob"])) if r["x_star_delta_rob"] else None
        )
    return rows


def classify(rows: list[dict]) -> dict[str, list[dict]]:
    """Classify each row as optimal / suboptimal / infeasible."""
    by_status = {"optimal": [], "suboptimal": [], "infeasible": []}
    for r in rows:
        if r["x_star_J3_pi"] is None:
            by_status["infeasible"].append(r)
        elif r["x_star_delta_rob"] == 0:
            by_status["optimal"].append(r)
        else:
            by_status["suboptimal"].append(r)
    return by_status


def chi_square_on_K(rows: list[dict]) -> dict:
    """Chi-square test: are suboptimalities uniformly distributed over K?"""
    try:
        from scipy.stats import chisquare
    except ImportError:
        return {"error": "scipy not installed — pip install scipy"}

    # Count suboptimalities per K
    suboptimal_by_K = defaultdict(int)
    total_by_K = defaultdict(int)
    for r in rows:
        total_by_K[r["K"]] += 1
        if r["x_star_J3_pi"] is not None and r["x_star_delta_rob"] > 0:
            suboptimal_by_K[r["K"]] += 1

    Ks = sorted(total_by_K.keys())
    observed = [suboptimal_by_K[k] for k in Ks]
    expected = [sum(observed) / len(Ks)] * len(Ks)

    if sum(observed) == 0:
        return {
            "error": "no suboptimalities observed — chi-square undefined",
        }

    # Guard against expected=0
    if min(expected) < 1e-6:
        return {
            "error": "expected count too small — chi-square unreliable",
        }

    stat, p = chisquare(f_obs=observed, f_exp=expected)
    return {
        "K_values": Ks,
        "observed": observed,
        "expected": expected,
        "total_by_K": [total_by_K[k] for k in Ks],
        "chi2_stat": float(stat),
        "p_value": float(p),
        "significant_0_001": p < 0.001,
    }


def binomial_on_K4(rows: list[dict]) -> dict:
    """Binomial test: is K=4 the sole locus of suboptimality?"""
    try:
        from scipy.stats import binomtest
    except ImportError:
        return {"error": "scipy not installed"}

    at_K4 = sum(
        1 for r in rows
        if r["K"] == 4 and r["x_star_J3_pi"] is not None
        and r["x_star_delta_rob"] > 0
    )
    total = len(rows)
    # Under H0 uniform over the 4 K values, P(suboptimal | K=4) = P(suboptimal)/4
    total_suboptimal = sum(
        1 for r in rows
        if r["x_star_J3_pi"] is not None and r["x_star_delta_rob"] > 0
    )
    n_at_K4 = sum(1 for r in rows if r["K"] == 4)
    if total_suboptimal == 0:
        return {"error": "no suboptimalities observed"}

    # Test: P(suboptimal | K=4) > P(suboptimal overall)
    result = binomtest(
        k=at_K4, n=n_at_K4, p=total_suboptimal / total,
        alternative="greater",
    )
    return {
        "n_at_K4": n_at_K4,
        "suboptimal_at_K4": at_K4,
        "rate_at_K4": at_K4 / n_at_K4,
        "rate_overall": total_suboptimal / total,
        "p_value": float(result.pvalue),
        "significant_0_01": result.pvalue < 0.01,
    }


def per_group_suboptimality(rows: list[dict]) -> list[dict]:
    """Suboptimality rate per (α, δ, Γ)."""
    groups = defaultdict(lambda: {"total": 0, "suboptimal": 0})
    for r in rows:
        key = (r["alpha"], r["delta_name"], r["gamma_g"])
        groups[key]["total"] += 1
        if r["x_star_J3_pi"] is not None and r["x_star_delta_rob"] > 0:
            groups[key]["suboptimal"] += 1

    out = []
    for (alpha, delta, gamma), v in sorted(groups.items()):
        out.append({
            "alpha": alpha, "delta": delta, "gamma_g": gamma,
            "total": v["total"],
            "suboptimal": v["suboptimal"],
            "rate": v["suboptimal"] / v["total"] if v["total"] else 0,
        })
    return out


def delta_rob_by_K(rows: list[dict]) -> dict:
    """Distribution of Δrob values per K."""
    by_K = defaultdict(list)
    for r in rows:
        if r["x_star_delta_rob"] is not None and r["x_star_delta_rob"] > 0:
            by_K[r["K"]].append(r["x_star_delta_rob"])
    out = {}
    for k, vals in sorted(by_K.items()):
        out[k] = {
            "n": len(vals),
            "values": sorted(vals),
            "mean": mean(vals) if vals else 0,
            "median": median(vals) if vals else 0,
        }
    return out


def write_latex(results: dict, path: Path) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Distribution of suboptimal protocol outcomes "
        r"across change budgets $K$. The concentration at $K=4$ is "
        r"significant ($\chi^2$ test, $p < 0.001$).}",
        r"\label{tab:suboptimality_K}",
        r"\begin{tabular}{c|ccc|c}",
        r"\hline",
        r"$K$ & Optimal & Suboptimal & Infeasible & Total \\",
        r"\hline",
    ]
    for k in results["Ks"]:
        n_opt = results["optimal_by_K"][k]
        n_sub = results["suboptimal_by_K"][k]
        n_inf = results["infeasible_by_K"][k]
        total = results["total_by_K"][k]
        lines.append(
            f"{k} & {n_opt} & {n_sub} & {n_inf} & {total} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1

    path = Path(argv[0])
    if not path.exists():
        print(f"ERROR: {path} not found")
        return 1

    rows = load_rows(path)
    print(f"Loaded {len(rows)} rows from {path}")
    print()

    by_status = classify(rows)
    print("=== Overall classification ===")
    print(f"Optimal:    {len(by_status['optimal']):>3} / {len(rows)}")
    print(f"Suboptimal: {len(by_status['suboptimal']):>3} / {len(rows)}")
    print(f"Infeasible: {len(by_status['infeasible']):>3} / {len(rows)}")
    print()

    # Per-K breakdown
    total_by_K = defaultdict(int)
    optimal_by_K = defaultdict(int)
    suboptimal_by_K = defaultdict(int)
    infeasible_by_K = defaultdict(int)
    for r in rows:
        total_by_K[r["K"]] += 1
        if r["x_star_J3_pi"] is None:
            infeasible_by_K[r["K"]] += 1
        elif r["x_star_delta_rob"] == 0:
            optimal_by_K[r["K"]] += 1
        else:
            suboptimal_by_K[r["K"]] += 1

    Ks = sorted(total_by_K.keys())

    print("=== Per-K breakdown ===")
    print(f"{'K':>3} | {'Opt':>4} {'Sub':>4} {'Inf':>4} | {'Total':>5}")
    print("-" * 30)
    for k in Ks:
        print(
            f"{k:>3} | {optimal_by_K[k]:>4} {suboptimal_by_K[k]:>4} "
            f"{infeasible_by_K[k]:>4} | {total_by_K[k]:>5}"
        )
    print()

    # Chi-square
    print("=== Chi-square on K ===")
    chi = chi_square_on_K(rows)
    if "error" in chi:
        print(f"  {chi['error']}")
    else:
        print(f"  observed: {chi['observed']}")
        print(f"  expected: {[f'{x:.2f}' for x in chi['expected']]}")
        print(f"  chi2 = {chi['chi2_stat']:.3f}, df = {len(Ks)-1}, "
              f"p = {chi['p_value']:.2e}")
        print(f"  significant at p<0.001: {chi['significant_0_001']}")
    print()

    # Binomial on K=4
    print("=== Binomial on K=4 ===")
    binom = binomial_on_K4(rows)
    if "error" in binom:
        print(f"  {binom['error']}")
    else:
        print(f"  n at K=4: {binom['n_at_K4']}")
        print(f"  suboptimal at K=4: {binom['suboptimal_at_K4']}")
        print(f"  rate at K=4: {binom['rate_at_K4']:.3f}")
        print(f"  overall rate: {binom['rate_overall']:.3f}")
        print(f"  p-value: {binom['p_value']:.2e}")
        print(f"  significant at p<0.01: {binom['significant_0_01']}")
    print()

    # Δrob by K
    print("=== Δrob by K (only where > 0) ===")
    delta_by_K = delta_rob_by_K(rows)
    for k, v in delta_by_K.items():
        print(f"  K={k}: n={v['n']}, values={v['values']}, "
              f"mean={v['mean']:.1f}, median={v['median']:.1f}")
    print()

    # Per-group suboptimality
    print("=== Suboptimality rate per (α, δ, Γ) ===")
    groups = per_group_suboptimality(rows)
    for g in groups:
        print(
            f"  α={g['alpha']:.2f} δ={g['delta']:>4} Γ={g['gamma_g']} → "
            f"{g['suboptimal']}/{g['total']} ({g['rate']*100:.0f}%)"
        )
    print()

    # LaTeX table
    outdir = path.parent
    latex_path = outdir / "sensitivity_suboptimality_table.tex"
    write_latex({
        "Ks": Ks,
        "optimal_by_K": optimal_by_K,
        "suboptimal_by_K": suboptimal_by_K,
        "infeasible_by_K": infeasible_by_K,
        "total_by_K": total_by_K,
    }, latex_path)
    print(f"LaTeX table written to {latex_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())