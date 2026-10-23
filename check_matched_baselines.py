"""Compare les 3 baselines sur les 10 combinaisons tight (qui existent
aussi dans la grille loose), avec filtrage par méthode."""

import csv

def load(path):
    """Retourne dict[(method, alpha, delta, gamma, beta, K)] = row."""
    data = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["method"], float(r["alpha"]), r["delta"],
                   int(r["gamma_g"]), float(r["beta"]), int(r["K"]))
            data[key] = r
    return data

loose = load("sensitivity_output/baselines_all_loose.csv")
tight = load("sensitivity_output/baselines_all_tight.csv")

matched_keys = [
    (0.40, "low", 1, 25.0, 5),
    (0.53, "low", 1, 25.0, 5),
    (0.60, "low", 1, 25.0, 5),
    (0.53, "high", 1, 25.0, 5),
    (0.53, "wide", 1, 25.0, 5),
    (0.53, "low", 2, 25.0, 5),
    (0.53, "low", 1, 25.0, 2),
    (0.53, "low", 1, 25.0, 3),
    (0.53, "low", 1, 25.0, 4),
    (0.53, "high", 1, 25.0, 4),
]

print(f"{'Method':>4} | {'Loose':>20} | {'Tight':>20} | {'Gap (pts)':>10}")
print("-" * 64)

for method in ["ff", "bf", "det"]:
    results = {}
    for inst_name, data in [("loose", loose), ("tight", tight)]:
        n_exact = n_sub = n_inf = n_missing = 0
        for key in matched_keys:
            row = data.get((method, *key))
            if row is None:
                n_missing += 1
                continue
            if row["status"] == "exact": n_exact += 1
            elif row["status"] == "sub": n_sub += 1
            else: n_inf += 1
        non_exact = (n_sub + n_inf) / (n_exact + n_sub + n_inf) * 100 if (n_exact+n_sub+n_inf) > 0 else 0
        results[inst_name] = {
            "exact": n_exact, "sub": n_sub, "inf": n_inf,
            "missing": n_missing, "non_exact_pct": non_exact,
        }
    gap = results["loose"]["non_exact_pct"] - results["tight"]["non_exact_pct"]
    print(f"{method:>4} | "
          f"{results['loose']['exact']}E/{results['loose']['sub']}S/{results['loose']['inf']}I "
          f"→ {results['loose']['non_exact_pct']:>3.0f}% | "
          f"{results['tight']['exact']}E/{results['tight']['sub']}S/{results['tight']['inf']}I "
          f"→ {results['tight']['non_exact_pct']:>3.0f}% | "
          f"{gap:>+6.0f} pts")
    if results["loose"]["missing"] > 0 or results["tight"]["missing"] > 0:
        print(f"     WARNING: {results['loose']['missing']} loose / "
              f"{results['tight']['missing']} tight missing")