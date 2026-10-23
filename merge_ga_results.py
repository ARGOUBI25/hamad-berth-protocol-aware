"""Merge GA suboptimal results into the aggregated CSV."""
import csv
from pathlib import Path

agg_path = Path("sensitivity_output/sensitivity_aggregated.csv")
ga_path = Path("sensitivity_output/ga_suboptimal_results.csv")
out_path = Path("sensitivity_output/sensitivity_aggregated.csv")

# Load GA results indexed by (alpha, delta, gamma_g, K)
ga_index = {}
with ga_path.open(encoding="utf-8") as f:
    for r in csv.DictReader(f):
        key = (r["alpha"], r["delta"], r["gamma_g"], r["K"])
        ga_index[key] = r

# Merge into aggregated
with agg_path.open(encoding="utf-8") as f:
    agg_rows = list(csv.DictReader(f))

new_fields = list(agg_rows[0].keys()) + [
    "ga_best_seed", "ga_best_J3_pi", "ga_best_delta_rob",
]
for r in agg_rows:
    key = (r["alpha"], r["delta"], r["gamma_g"], r["K"])
    ga = ga_index.get(key, {})
    r["ga_best_seed"] = ga.get("ga_best_seed", "")
    r["ga_best_J3_pi"] = ga.get("ga_best_J3_pi", "")
    r["ga_best_delta_rob"] = ga.get("ga_best_delta_rob", "")

with out_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=new_fields)
    writer.writeheader()
    writer.writerows(agg_rows)

print(f"Wrote {out_path}")
print(f"Total rows: {len(agg_rows)}")
print(f"Rows with GA data: {sum(1 for r in agg_rows if r['ga_best_delta_rob'])}")