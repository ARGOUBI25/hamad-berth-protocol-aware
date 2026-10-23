import csv

path = "sensitivity_output/sensitivity_aggregated_29mai.csv"
rows = list(csv.DictReader(open(path, encoding="utf-8")))

print(f"{'K':>3} {'delta':>6} {'gamma':>5} | {'pi_drob':>8} {'GA_drob':>8} {'GA_seed':>8} | {'pi_J3pi':>8} {'GA_J3pi':>8}")
print("-" * 80)

found = 0
for r in rows:
    pi_drob = r.get("x_star_delta_rob_mean", "")
    if pi_drob in ("", "0", "0.0", "None"):
        continue
    found += 1
    k = r.get("K", "")
    delta = r.get("delta", "")
    gamma = r.get("gamma_g", "")
    ga_drob = r.get("ga_best_delta_rob", "")
    ga_seed = r.get("ga_best_seed", "")
    pi_j3pi = r.get("x_star_J3_pi", "")
    ga_j3pi = r.get("ga_best_J3_pi", "")
    print(f"{k:>3} {delta:>6} {gamma:>5} | {pi_drob:>8} {ga_drob:>8} {ga_seed:>8} | {pi_j3pi:>8} {ga_j3pi:>8}")

print()
print(f"Total suboptimal cases: {found}")