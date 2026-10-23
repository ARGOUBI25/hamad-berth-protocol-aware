# Protocol-aware plan selection in two-stage berth allocation — code and data

Code, instances and result files accompanying the paper

> *Robust Optimization Is Not Enough: Protocol-Aware Plan Selection in Two-Stage Berth Allocation under Disruption* (submitted to OR Spectrum, special issue "Operations Research Methods for Resilient Networks under Geopolitical Disruptions").

**Archive (DOI):** `[TODO: Zenodo DOI]` · **Repository:** `[TODO: GitHub URL]`

## What is in this repository

| Part | Files |
|---|---|
| Model and protocol | `gate_model.py` (data structures, deterministic protocol π, scenarios, metrics), `hamad_adapter.py` (JSON instance → model objects) |
| Exact second stage | `recourse_milp.py` (exact recourse MILP, dual evaluation J and J_π) |
| Joint robust MILP | `joint_milp.py` (lexicographic z_max / J1 / J2) |
| π-aware GA | `genetic_algorithm.py` (chromosome (ω, σ), OX + uniform crossover, ranking (I^π, J_π, −J1, J2)) |
| Baselines | `baselines.py` (greedy first-fit, greedy best-fit, deterministic MILP) |
| Experiment drivers | `sensitivity_hamad.py` (74-combination grid), `run_baselines.py`, `analyze_sensitivity.py` |
| Post-hoc checks that produced the paper's tables | `check_*.py`, `merge_ga_results.py`, `rerun_ga_suboptimal.py`, `verify_matched.py` |
| Instances | `hamad_instance.json` (loose, 29 May 2025), `hamad_instance_20250516.json` (tight, 16 May 2025) |
| Results | `sensitivity_output/`, `baselines_29mai/`, `baselines_16mai/` |
| Figures | `figures/` (`make_figs.py` regenerates both paper figures from the CSV files) |
| Tests | `test_*.py` (78 tests) |

## Vocabulary: the code base was adapted from a gate-assignment pipeline

The model code was originally written for airport gate assignment and re-used with an adapter. The identifiers keep the original names. Translation:

| In the code | In the paper |
|---|---|
| gate | berth (resource 0 = anchorage, the last-resort resource) |
| rotation | vessel call |
| passenger flow | transshipment container flow (`n` containers, `min_conn` minimum connection time) |
| walking time | transfer time between berths |
| zone | terminal / berth group (closure set Γ_g acts on groups) |
| `stand_id=0` "aggregated infinite stand" | anchorage; the *RuntimeWarning* about it is expected |
| `delta_rob`, `delta_sc`, `I_pi`, `J3_star`, `J3_pi` | Δ_rob, Δ_sc, I^π, J, J_π |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                                    # 78 passed
```

Reported runs: Python 3.11.4 (Windows, AMD64), OR-Tools **9.14.6206** (pinned in `requirements.txt`). With this version, the 78 tests pass, `run_baselines.py` reproduces the stored baseline values and the GA reproduces the stored values on a spot check of four combinations (Python 3.13, Linux). **Use the pinned OR-Tools version for the MILP-based steps:** with OR-Tools 9.15.6755 the deterministic MILP returns a different optimal nominal plan on `hamad_instance.json` (I^π = 8 instead of 5, and a plan without feasible recourse instead of J = 770), because the nominal problem has several optimal solutions and the solver picks among them.

## Reproducing the results

All commands are run from the repository root (scripts use relative paths).

```bash
# 1. Loose-instance grid: 74 combinations, joint MILP + exact recourse + GA (5 seeds, population 60, 50 generations)
#    Long run: 15,899 s (about 4.4 h) in the original run; the joint MILP has a 600 s limit per combination.
python sensitivity_hamad.py hamad_instance.json --mode publication --output-dir sensitivity_output

# 2. Baselines on both instances
python run_baselines.py hamad_instance.json          --output baselines_29mai
python run_baselines.py hamad_instance_20250516.json --output baselines_16mai

# 3. GA on every combination, then certification against the proven MILP optimum
python check_ga_42_infeasible.py        # writes sensitivity_output/ga_all_74.csv (about 5 minutes)
python check_certification_42.py        # certificate: J_pi(GA) = J* implies pi-optimal
python check_ga_10_tight.py             # GA on the tight instance
python check_baselines_all_74.py        # baselines on all 74 + 10 combinations

# 4. Statistics and figures
python analyze_sensitivity.py sensitivity_output/sensitivity_raw.csv
python figures/make_figs.py             # writes figures/fig_*.pdf and .png
```

The joint MILP is the only slow step. Everything that involves the GA runs in seconds per combination.

### Which file feeds which part of the paper

| File | Content |
|---|---|
| `sensitivity_output/sensitivity_aggregated_29mai.csv` | joint MILP + exact recourse, loose instance, 74 combinations |
| `sensitivity_output/ga_all_74.csv` | GA outcome on the same 74 combinations |
| `sensitivity_output/certification_74.csv` | certification of the GA plans against J* |
| `sensitivity_output/sensitivity_raw.csv` | tight instance (16 May), per-combination raw rows |
| `sensitivity_output/ga_10_tight.csv` | GA on the tight instance |
| `sensitivity_output/baselines_all_loose.csv`, `baselines_all_tight.csv` | greedy and deterministic-MILP baselines on all combinations |
| `sensitivity_output/sensitivity_aggregated.csv`, `ga_suboptimal_results.csv` | intermediate versions, kept for traceability (the GA columns for the 8 suboptimal combinations were merged by `merge_ga_results.py`) |
| `sensitivity_output/sensitivity_manifest.json` | run manifest (versions, instance hash, grid axes, wall time) |

## Instances

Both instances are **synthetic, calibrated** instances: 10 vessel calls, 11 berths in 2 terminals plus an anchorage, 12 transshipment flows, `t_min` = 10 min, TEU-to-container ratio θ = 1.5. They use aggregate vessel-call counts from the public *HDX Qatar Daily Port Activity* dataset and throughput and transshipment figures published by Mwani Qatar. Vessel identities, times and container flows are generated; they are not records of real calls. Details: Section 4 of the paper. The transshipment share α is varied by proportional rescaling of the base flows (`regenerate_flows_at_alpha` in `sensitivity_hamad.py`).

## Known limitations and a discrepancy to be aware of

* **Instance hash.** `sensitivity_manifest.json` records `instance_sha256 = 89c673da…539b3c` for `hamad_instance.json`. The file shipped here has SHA-256 `8e3d445e…e37e9fd`, because it was saved again after the manifest was written (file dates: manifest 21 Oct 2026 11:38, instances 21 Oct 2026 15:20). The shipped files are the ones used by every later script (`check_*`, `run_baselines.py`). A spot check of the GA on four combinations of the shipped loose instance, including the uncertified one, returned exactly the values stored in `ga_all_74.csv`. The joint-MILP step (about 4.4 h) was not re-run for this archive. Hashes of the shipped files are in `SHA256SUMS.txt`.
* The manifest covers the 74-combination loose run only; the tight-instance rows come from separate runs (`check_16mai_high.py`, `check_ga_10_tight.py`).
* The uncertified combination is (α = 0.53, wide, Γ_g = 2, K = 4, β = 25): the joint MILP stopped at the 600 s limit (incumbent 1150, gap 43%); the GA plan has J = J_π = 870 and no certificate is claimed.
* `GA_DEFAULT` in `config.py` (population 150, 400 generations) is the library default and was **not** used for the paper; the reported runs use population 60, 50 generations, seeds 1–5.
* MILP results that depend on tie-breaking between equally optimal solutions (deterministic MILP, joint MILP) can change with the solver version; the GA and the exact evaluation of a given plan do not depend on it in the checks above.
* Wall times depend on the machine and on the MILP time limit.

## How to cite

See `CITATION.cff`. Please cite the paper and the archived release (DOI above).

## Licence

Code: MIT (`LICENSE`). Data, results and figures: CC BY 4.0 (`LICENSE-DATA.md`).
