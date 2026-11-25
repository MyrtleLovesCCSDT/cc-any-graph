import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.circuit_eq import evaluate_truth_table, pair_index_map, solve_general_equilibrium
from src.graph_design import _parse_config

base_dir = ROOT / "results" / "graph_designs"
for jf in sorted(base_dir.glob("*/*_full.json")):
    with open(jf) as f:
        bundle = json.load(f)
    cfg = bundle["config"]
    parsed = _parse_config(cfg, allow_5mer=True)
    mode = parsed["mode"]
    names = parsed["node_names"]
    n = len(names)
    print(f"=== {parsed['name']} | mode={mode} | nodes={names} | file={jf.name} | n_des={len(bundle['designs'])} ===")
    if mode != "truth_table":
        d0 = bundle["designs"][0]
        print(f"  D01 metrics={d0['metrics']}")
        continue

    obs = parsed["observables"][0]
    print(f"  obs0: kind={obs['obj'].kind}, nodes={[names[i] for i in obs['obj'].nodes]}, ref={[names[i] for i in obs['obj'].ref_nodes]}, tgt={obs['targets']}, min_d={obs['min_delta']}, min_f={obs['min_fold']}")
    print(f"  edges: {[(e['u_name'], e['v_name']) for e in parsed['edges']]}")

    # Check top-3 + best valid slip/homo candidate across all saved designs
    picks = [0]
    for idx, d in enumerate(bundle["designs"]):
        m = d["metrics"]
        if idx > 0 and m.get("min_slip_margin", -999) >= 0 and m.get("worst_homo_pKd", 99) <= 4.32:
            picks.append(idx)
            break

    for idx in sorted(set(picks)):
        d = bundle["designs"][idx]
        m = d["metrics"]
        c0_uM = np.array(d["c0_uM"], dtype=float)
        c0_m = c0_uM * 1e-6
        kd_m = 10.0 ** np.minimum(-np.array(d["pkd_matrix"], dtype=float), -2.0)
        eq = solve_general_equilibrium(c0_m, kd_m)
        vals = evaluate_truth_table(kd_m, c0_m, obs["obj"], obs["targets"])["values"]
        ref_idx = obs["obj"].ref_nodes[0]
        ref_uM = c0_uM[:, ref_idx]
        out_uM = vals * ref_uM

        print(f"  [{d['design_id']}] MC={m.get('MC_Pass_rate', np.nan):.3f} D={m.get('Delta', np.nan):.3f} F={m.get('Fold', np.nan):.1f} "
              f"off={m.get('worst_off_pKd', np.nan):.2f} homo={m.get('worst_homo_pKd', np.nan):.2f} slip={m.get('min_slip_margin', np.nan):.1f}")
        print(f"    c0_uM(S0..): {np.round(c0_uM, 3).tolist()}")
        print(f"    out_frac: {np.round(vals, 3).tolist()} | out_uM: {np.round(out_uM, 4).tolist()}")

        # Special check: can and_5node serve as NAND via bound_frac / dimer_frac?
        if parsed["name"] == "and_5node":
            free_s = eq[:, ref_idx] * 1e6
            bound_s = ref_uM - free_s
            print(f"    [NAND check on and_5node] bound_S_uM={np.round(bound_s, 4).tolist()}, frac={np.round(1 - vals, 3).tolist()}")
