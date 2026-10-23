"""Croise ga_all_74.csv et sensitivity_aggregated_29mai.csv pour
déterminer combien des 42 combinaisons π-infeasible ont un certificat
π-optimal (J_π_GA = J*_MILP) et quel est le prix de π-admissibilité
(J_π_GA - J*_MILP) sur les autres."""

import csv
from collections import defaultdict
from pathlib import Path

# --- 1. Charger J* et status du MILP joint ---
agg_path = "sensitivity_output/sensitivity_aggregated_29mai.csv"
if not Path(agg_path).exists():
    agg_path = "sensitivity_output/sensitivity_aggregated.csv"

milp = {}
with open(agg_path, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        key = (float(r["alpha"]), r["delta"], int(r["gamma_g"]),
               float(r["beta"]), int(r["K"]))
        J_star = r.get("joint_J3_star_mean", "").strip()
        status = r.get("joint_status", "").strip()
        milp[key] = {
            "J_star": int(float(J_star)) if J_star else None,
            "status": status,
        }

# --- 2. Charger le GA-74 ---
ga_path = "sensitivity_output/ga_all_74.csv"
rows = []
with open(ga_path, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        key = (float(r["alpha"]), r["delta"], int(r["gamma_g"]),
               float(r["beta"]), int(r["K"]))
        m = milp.get(key, {})
        rows.append({
            "key": key,
            "milp_cat": r["milp_category"],
            "J_star": m.get("J_star"),
            "milp_status": m.get("status"),
            "J_ga": int(r["J_ga"]) if r["J_ga"] else None,
            "J_pi_ga": int(r["J_pi_ga"]) if r["J_pi_ga"] else None,
            "I_pi_ga": int(r["I_pi_ga"]),
        })

# --- 3. Catégoriser chaque combinaison π-infeasible ---
print("\n=== Analyse des 42 combinaisons π-infeasible (MILP joint) ===\n")

by_cat = defaultdict(lambda: {
    "n": 0, "certified": 0, "not_certified": 0, "no_Jstar": 0,
    "price": [],
})

detail = []
for r in rows:
    cat = r["milp_cat"]
    by_cat[cat]["n"] += 1

    if r["milp_status"] != "OPTIMAL":
        by_cat[cat]["no_Jstar"] += 1
        detail.append({**r, "cert": "no_Jstar", "price": None})
        continue

    if r["J_star"] is None:
        by_cat[cat]["no_Jstar"] += 1
        detail.append({**r, "cert": "no_Jstar", "price": None})
        continue

    if r["J_pi_ga"] == r["J_star"]:
        by_cat[cat]["certified"] += 1
        detail.append({**r, "cert": "yes", "price": 0})
    elif r["J_pi_ga"] is not None and r["J_pi_ga"] > r["J_star"]:
        price = r["J_pi_ga"] - r["J_star"]
        by_cat[cat]["not_certified"] += 1
        by_cat[cat]["price"].append(price)
        detail.append({**r, "cert": "price", "price": price})
    else:
        # J_pi_ga < J* ? impossible mathématiquement, mais à signaler
        detail.append({**r, "cert": "anomaly", "price": None})

# --- 4. Résumé par catégorie ---
print(f"{'MILP cat':>12} | {'n':>3} | {'certified':>9} | {'w/ price':>9} | {'no J*':>6} | {'median price':>12}")
print("-" * 72)
for cat in ["opt", "sub", "inf"]:
    d = by_cat[cat]
    prices = sorted(d["price"])
    med = prices[len(prices)//2] if prices else None
    print(f"{cat:>12} | {d['n']:>3} | {d['certified']:>9} | {d['not_certified']:>9} | {d['no_Jstar']:>6} | {str(med):>12}")

# --- 5. Détail des 42 infeasible ---
print("\n=== Détail des 42 combinaisons π-infeasible ===\n")
print(f"{'α':>5} {'δ':>5} {'Γ':>2} {'K':>2} | {'J*':>5} {'J_GA':>5} {'Jπ_GA':>6} | {'cert':>8} {'price':>5}")
print("-" * 62)
for r in detail:
    if r["milp_cat"] != "inf":
        continue
    key = r["key"]
    print(f"{key[0]:>5.2f} {key[1]:>5} {key[2]:>2} {key[4]:>2} | "
          f"{str(r['J_star']):>5} {str(r['J_ga']):>5} {str(r['J_pi_ga']):>6} | "
          f"{r['cert']:>8} {str(r['price']):>5}")

# --- 6. Écriture CSV détaillé ---
with open("sensitivity_output/certification_74.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "alpha", "delta", "gamma_g", "beta", "K",
        "milp_cat", "milp_status", "J_star",
        "J_ga", "J_pi_ga", "I_pi_ga",
        "cert", "price",
    ])
    writer.writeheader()
    for r in detail:
        key = r["key"]
        writer.writerow({
            "alpha": key[0], "delta": key[1], "gamma_g": key[2],
            "beta": key[3], "K": key[4],
            "milp_cat": r["milp_cat"],
            "milp_status": r["milp_status"],
            "J_star": r["J_star"],
            "J_ga": r["J_ga"],
            "J_pi_ga": r["J_pi_ga"],
            "I_pi_ga": r["I_pi_ga"],
            "cert": r["cert"],
            "price": r["price"],
        })

print(f"\nWritten: sensitivity_output/certification_74.csv")