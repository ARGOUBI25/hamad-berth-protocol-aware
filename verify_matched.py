"""
verify_matched.py -- Check the 10 matched combinations on the loose instance (May 29).

Usage (PyCharm terminal):
    python verify_matched.py path/to/baselines_loose.csv

Step 1: run it once; it prints the column names of the CSV.
Step 2: adjust the six COL_* names below to match those columns, then run again.
"""
import sys
import pandas as pd

# ---- Adjust these names to your CSV columns -------------------------------
COL_METHOD = "method"   # e.g. "greedy_best_fit", "joint_milp", ...
COL_ALPHA  = "alpha"    # 0.40 / 0.53 / 0.60
COL_DELTA  = "delta"    # "low" / "high" / "wide"
COL_GAMMA  = "gamma_g"  # 1 / 2
COL_BETA   = "beta"     # 15 / 25 / 35
COL_K      = "K"        # 2 / 3 / 4 / 5
COL_IPI    = "I_pi"     # number of scenarios where pi fails
COL_DROB   = "Drob"     # robust gap (NaN or empty when I_pi > 0)
METHOD_TO_CHECK = "greedy_best_fit"  # exact label of best-fit in the CSV
# ---------------------------------------------------------------------------

# The 10 combinations evaluated on BOTH instances: (alpha, delta, gamma_g, beta, K)
MATCHED = [
    (0.53, "low",  1, 25, 2),
    (0.53, "low",  1, 25, 3),
    (0.53, "low",  1, 25, 4),
    (0.53, "low",  1, 25, 5),
    (0.40, "low",  1, 25, 5),
    (0.60, "low",  1, 25, 5),
    (0.53, "high", 1, 25, 5),
    (0.53, "wide", 1, 25, 5),
    (0.53, "low",  2, 25, 5),
    (0.53, "high", 1, 25, 4),   # reference combination of Table 4
]

df = pd.read_csv(sys.argv[1])
print("Columns in the CSV:", list(df.columns))
print("Methods found:", df[COL_METHOD].unique() if COL_METHOD in df else "?")
print()

sub = df[df[COL_METHOD] == METHOD_TO_CHECK]
non_exact = 0
for a, d, g, b, k in MATCHED:
    row = sub[(sub[COL_ALPHA].round(2) == a) & (sub[COL_DELTA] == d) &
              (sub[COL_GAMMA] == g) & (sub[COL_BETA] == b) & (sub[COL_K] == k)]
    tag = "REF " if (a, d, g, k) == (0.53, "high", 1, 4) else "    "
    if row.empty:
        print(f"{tag}{(a, d, g, b, k)}  ->  NOT FOUND in CSV")
        continue
    ipi = row[COL_IPI].iloc[0]
    drob = row[COL_DROB].iloc[0]
    exact = (ipi == 0) and (pd.notna(drob) and float(drob) == 0)
    non_exact += 0 if exact else 1
    status = "exact" if exact else ("pi-infeasible" if ipi > 0 else "suboptimal")
    print(f"{tag}{(a, d, g, b, k)}  ->  I_pi={ipi}, Drob={drob}  =>  {status}")

print(f"\n{METHOD_TO_CHECK}: non-exact on {non_exact}/10 matched combinations")
