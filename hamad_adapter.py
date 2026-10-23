"""Adaptateur Hamad Port (maritime) → pipeline aérien existant.

Charge hamad_instance.json et produit les objets attendus par gate_model.py :
  Rotation, PassengerFlow, gates, zones, compatibility, walking_times.

Aucune modification de gate_model.py n'est nécessaire.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from gate_model import PassengerFlow, Rotation


def load_hamad_instance(path: str | Path) -> dict[str, Any]:
    """Charge le JSON et le convertit en structures compatibles pipeline."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    # --- 1. Convertir vessel_calls en Rotation -------------------------
    # Les timestamps ISO sont convertis en minutes depuis minuit du 2025-05-29.
    base_date = datetime(2025, 5, 29)
    rotations: list[Rotation] = []
    vessel_meta: dict[str, dict] = {}
    for v in raw["vessel_calls"]:
        arr_dt = datetime.fromisoformat(v["arr_time_utc"].replace("Z", ""))
        dep_dt = datetime.fromisoformat(v["dep_time_utc"].replace("Z", ""))
        arr_min = (arr_dt - base_date).total_seconds() / 60.0
        dep_min = (dep_dt - base_date).total_seconds() / 60.0
        rotations.append(Rotation(
            id=v["id"],
            arr=arr_min,
            dep=dep_min,
            gabarit=v.get("vessel_type"),
            wave="W1",  # une seule vague dans cette instance
        ))
        vessel_meta[v["id"]] = v

    # --- 2. Convertir transshipment_flows en PassengerFlow -------------
    flows: list[PassengerFlow] = []
    for f in raw["transshipment_flows"]:
        flows.append(PassengerFlow(
            id=f["id"],
            arr_rot=f["arr_vessel"],
            dep_rot=f["dep_vessel"],
            n=int(f["n"]),
            min_conn=float(f["min_conn"]),
        ))

    # --- 3. Gates et zones ---------------------------------------------
    gates = [b["id"] for b in raw["berths"]]
    zones = {terminal: tuple(ids) for terminal, ids in raw["zones"].items()}

    # --- 4. Compatibilité depuis LOA/draft + false_pairs --------------
    berth_meta = {b["id"]: b for b in raw["berths"]}
    false_pairs = {tuple(p) for p in raw["compatibility"]["false_pairs"]}
    compatibility: dict[tuple, bool] = {}
    for v in raw["vessel_calls"]:
        for b in raw["berths"]:
            berth_id = b["id"]
            # Vérifier false_pairs explicites
            if (v["id"], berth_id) in false_pairs:
                compatibility[(v["id"], berth_id)] = False
                continue
            # Vérifier LOA/draft
            loa_ok = v["loa_m"] <= b["length_m"]
            draft_ok = v["draft_m"] <= b["draft_m"]
            compatibility[(v["id"], berth_id)] = loa_ok and draft_ok

    # Stand 0 (anchorage) : compatible avec tous les navires sauf V006 si désiré.
    # V006 (dry bulk) n'a PAS de compatibilité false vers 0, donc compatible.
    # C'est OK : il reste à l'anchorage car incompatible avec les berths 1-11.
    for v in raw["vessel_calls"]:
        compatibility[(v["id"], 0)] = True

    # --- 5. Walking times aplati ---------------------------------------
    wt_spec = raw["walking_times"]
    same = float(wt_spec.get("default_same", 0))
    diff = float(wt_spec.get("default_diff", 15))
    all_resources = [0] + gates
    walking_times: dict[tuple, float] = {
        (a, b): (same if a == b else diff)
        for a in all_resources for b in all_resources
    }
    for triple in wt_spec.get("overrides", []):
        if len(triple) == 3:
            a, b, w = triple
            walking_times[(a, b)] = float(w)

    # --- 6. Paramètres de scénarios ------------------------------------
    t_min = float(raw.get("t_min", 10))
    K = int(raw.get("K", 4))
    gamma_g = int(raw.get("gamma_g", 1))
    delta_values = tuple(raw.get("delta_values", [30, 60, 90]))
    beta_max = float(raw.get("beta_max", 25))

    return {
        "rotations": rotations,
        "flows": flows,
        "gates": gates,
        "zones": zones,
        "compatibility": compatibility,
        "walking_times": walking_times,
        "t_min": t_min,
        "K": K,
        "gamma_g": gamma_g,
        "delta_values": delta_values,
        "beta_max": beta_max,
        "vessel_meta": vessel_meta,
        "berth_meta": berth_meta,
    }


def print_summary(inst: dict) -> None:
    """Affiche un résumé de l'instance chargée."""
    print("=" * 72)
    print("HAMAD PORT INSTANCE — CHARGED")
    print("=" * 72)
    print(f"Vessel calls: {len(inst['rotations'])}")
    for r in inst["rotations"]:
        meta = inst["vessel_meta"][r.id]
        print(
            f"  {r.id}: {meta['vessel_type']:20s} "
            f"LOA={meta['loa_m']:>4.0f}m draft={meta['draft_m']:>4.1f}m "
            f"arr={r.arr:>6.0f}min dep={r.dep:>6.0f}min"
        )
    print(f"Berths: {len(inst['gates'])} (ids: {inst['gates']})")
    print(f"Zones: { {k: list(v) for k, v in inst['zones'].items()} }")
    print(f"Transshipment flows: {len(inst['flows'])}")
    total_containers = sum(f.n for f in inst["flows"])
    print(f"  Total containers: {total_containers}")
    print(f"Parameters: t_min={inst['t_min']}, K={inst['K']}, "
          f"gamma_g={inst['gamma_g']}, beta_max={inst['beta_max']}")
    print(f"Delta values: {inst['delta_values']}")
    print("=" * 72)