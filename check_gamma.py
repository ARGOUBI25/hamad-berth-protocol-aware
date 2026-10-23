from hamad_adapter import load_hamad_instance
from gate_model import build_stress_sets, compute_q_scores

for path in ["hamad_instance.json", "hamad_instance_20250516.json"]:
    inst = load_hamad_instance(path)
    q = compute_q_scores(inst["rotations"], inst["flows"])
    gf, stress = build_stress_sets(inst["rotations"], q)
    rot_ids = [r.id for r in inst["rotations"]]
    waves = {}
    for r in inst["rotations"]:
        if r.wave is not None:
            waves.setdefault(r.wave, []).append(r.id)
    print(f"=== {path} ===")
    print(f"  Total vessels in M: {len(rot_ids)}")
    print(f"  Rotations: {rot_ids}")
    print(f"  Waves: { {w: len(ids) for w, ids in waves.items()} }")
    print(f"  Gamma_f: {gf}")
    print(f"  Stress set: {stress}")
    print()