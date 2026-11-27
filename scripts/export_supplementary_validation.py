#!/usr/bin/env python3
"""Export predicted affinity matrices and exact equilibrium performance for Supplementary Data & manual validation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import openpyxl
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
except ImportError as exc:
    raise SystemExit("openpyxl is required. Install with: pip install openpyxl") from exc

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.circuit_eq import (
    Observable,
    dimer_pairs,
    evaluate_curve,
    evaluate_truth_table,
    pair_index_map,
    solve_general_equilibrium,
)


GRAPH_CIRCUIT_ORDER: List[str] = [
    "yes_3node", "not_3node", "or_4node", "nor_4node", "nor_5node",
    "and_5node", "and_6node", "nand_6node", "imply_5node", "nimply_4node",
    "nimply_6node", "xor_4node", "xnor_4node", "xnor_5node", "or3_5node", "ortho2_6node",
    "analog_bandpass", "dome_3node", "analog_neg_x2", "threshold_decoy_4node",
    "custom_graph_xor_topo", "cycle_c4", "bipartite_k23", "cascade_5node", "barbell_6node",
]

EXTRA_PICKS: Dict[str, List[str]] = {
    "dome_3node": ["D08"],
}

LEGACY_DES_BOUNDS: Set[str] = {
    "circle_s1",
    "cubic_s5",
    "cubic_s5_kmin2",
    "cubic_s10_seed",
    "cubic_yd_4mer_s10_kmin0.5",
    "cubic_yd_5mer_s10_kmin0.5",
    "quadratic_s3",
    "quadratic_s5_kmin2",
}

DES_BOUNDS_SHEET_NAMES: Dict[str, str] = {
    "line_s1": "2D_line_s1",
    "circle_s5_kmin2": "2D_circle_s5",
    "circle_cfit_s5_kmin2": "2D_circle_cfit",
    "quadratic_s10_seed": "2D_quadratic_s10",
    "quadratic_cfit_s10_seed": "2D_quadratic_cfit",
    "cubic_yd_4mer_cfit_s15_kmin0.5": "2D_cubic_4mer",
    "cubic_yd_5mer_cfit_s15_kmin0.5": "2D_cubic_5mer",
    "angle_cfit_s15_seed": "2D_angle_cfit",
    "angle_2_cfit_s10_seed": "2D_angle2_cfit",
    "bandpass_cfit_s10_seed": "2D_bandpass_cfit",
    "bandpass_2_cfit_s10_seed": "2D_bandpass2_cfit",
    "full_circle_cfit_s15_seed": "2D_full_circle",
    "full_circle_2_cfit_s15_seed": "2D_full_circle2",
}

BANANA_SHEET_NAMES: Dict[str, str] = {
    "dimer_3_compact_4mer_s15": "BN_dimer3_4mer_s15",
    "dimer_4_compact_5mer_s10": "BN_dimer4_5mer_s10",
    "monomer_3_5mer_s10": "BN_mono3_5mer_s10",
    "monomer_3_5mer_s8": "BN_mono3_5mer_s8",
    "monomer_4_5mer_s10": "BN_mono4_5mer_s10",
}

BANANA_BANDS: Dict[str, int] = {
    "monomer_3": 3,
    "monomer_4": 4,
    "dimer_3_compact": 3,
    "dimer_4_compact": 4,
    "dimer_3": 3,
    "dimer_4": 4,
}

BANANA_GEOMETRY = {
    "center_x_nM": 81.60262722518003,
    "center_y_nM": -61.35139230766126,
    "x_shear": -0.026983573718282847,
    "y_scale": 0.4952714824652137,
    "radii_nM": [60.90715850262229, 95.56248405294735, 125.34507669381321],
}


# ---------------------------------------------------------------------------
# Pure-NumPy 2D boundary & equilibrium solvers (matches design_2d_boundary / design_banana)
# ---------------------------------------------------------------------------

def boundary_fn_2d(name: str, x: np.ndarray, y: np.ndarray, max_in: float) -> np.ndarray:
    b = name[:-2] if name.endswith("_2") else name
    s = max_in / 200.0
    if b == "line":
        return y > x
    if b == "circle":
        return x ** 2 + (y - max_in) ** 2 > max_in ** 2
    if b == "quadratic":
        return y > 4.0 * x * (1.0 - x / max_in)
    if b == "cubic":
        return y > 0.5 * max_in * (1.0 + (2.0 * x / max_in - 1.0) ** 3)
    if b == "full_circle":
        c, r = 100.0 * s, 30.0 * s
        return (x - c) ** 2 + (y - c) ** 2 > r ** 2
    if b == "angle":
        return (y > 0.5 * x) & (y < 2.0 * x)
    if b == "bandpass":
        return (y > 1.25 * x + 10.0 * s) & (y < 1.25 * x + 60.0 * s)
    raise ValueError(f"Unknown 2D boundary: {name}")


def banana_band_labels(points_native: np.ndarray, bands: int) -> np.ndarray:
    x, y = points_native.T
    g = BANANA_GEOMETRY
    dist = np.hypot(x - g["center_x_nM"] - g["x_shear"] * y, g["y_scale"] * (y - g["center_y_nM"]))
    idx = np.searchsorted(g["radii_nM"][: bands - 1], dist)
    return (idx % 2).astype(int)


def solve_eq_pos_np(c0_nM: np.ndarray, kd_nM: np.ndarray, max_newton: int = 50) -> Tuple[np.ndarray, float]:
    """Damped fixed-point + positivity-preserving Newton equilibrium solver in nM."""
    a_mat = 1.0 / np.maximum(kd_nM, 1e-6)
    a_mod = a_mat.copy()
    np.fill_diagonal(a_mod, 2.0 * np.diag(a_mat))
    eye = np.eye(c0_nM.shape[1], dtype=float)[None, :, :]

    x = c0_nM.copy()
    for _ in range(40):
        x = 0.5 * x + 0.5 * (c0_nM / (1.0 + x @ a_mod))

    active = c0_nM > 0.0
    for _ in range(max_newton):
        factor = 1.0 + x @ a_mod
        f = x * factor - c0_nM
        if np.max(np.abs(f)) < 1e-9:
            break
        jac = x[:, :, None] * a_mod[None, :, :] + factor[:, :, None] * eye
        try:
            delta = np.linalg.solve(jac, f[..., None])[..., 0]
        except np.linalg.LinAlgError:
            delta = np.zeros_like(f)
            for b in range(len(f)):
                delta[b] = np.linalg.lstsq(jac[b], f[b], rcond=None)[0]

        x_next = x - delta
        neg_mask = np.any(x_next < 0.0, axis=1)
        if np.any(neg_mask):
            step_scale = np.ones((x.shape[0], 1), dtype=float)
            for b in np.where(neg_mask)[0]:
                bad = (delta[b] > 0.0) & (x_next[b] <= 0.0)
                if np.any(bad):
                    alpha = 0.9 * np.min(x[b, bad] / delta[b, bad])
                    step_scale[b, 0] = min(1.0, max(1e-4, float(alpha)))
            x_next = x - step_scale * delta
        x = np.where(active, np.maximum(x_next, 1e-18), 0.0)

    final_res = x * (1.0 + x @ a_mod) - c0_nM
    max_rel_err = float(np.max(np.abs(final_res[active]) / np.maximum(c0_nM[active], 1e-9))) if np.any(active) else 0.0
    return x, max_rel_err


def split_edge_nodes(edge_str: str, nodes: List[str]) -> Optional[Tuple[str, str]]:
    if ":" in edge_str:
        u, v = edge_str.split(":", 1)
        if u in nodes and v in nodes:
            return u, v
    matches = [(u, v) for u in nodes for v in nodes if u != v and (u + v) == edge_str]
    if len(matches) == 1:
        return matches[0]
    for u in nodes:
        if edge_str.startswith(u):
            rem = edge_str[len(u) :]
            if rem in nodes:
                return u, rem
    return None


def mass_balance_rel_err_eqtk(eq_m: np.ndarray, c0_m: np.ndarray, n: int) -> np.ndarray:
    pmap = pair_index_map(n)
    tot_calc = eq_m[:, :n].copy()
    for i, j in dimer_pairs(n):
        col = pmap[(i, j)]
        if i == j:
            tot_calc[:, i] += 2.0 * eq_m[:, col]
        else:
            tot_calc[:, i] += eq_m[:, col]
            tot_calc[:, j] += eq_m[:, col]
    rel = np.zeros(c0_m.shape[0], dtype=float)
    for r in range(c0_m.shape[0]):
        active = c0_m[r] > 0.0
        if np.any(active):
            rel[r] = float(np.max(np.abs(tot_calc[r, active] - c0_m[r, active]) / c0_m[r, active]))
    return rel


def readout_conc_uM(obs: Observable, eq_m: np.ndarray, c0_m: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    pmap = pair_index_map(n)
    refs = obs.ref_nodes if obs.ref_nodes is not None else obs.nodes
    ref_uM = c0_m[:, refs[0]] * 1e6
    if obs.kind == "free_frac":
        return eq_m[:, obs.nodes[0]] * 1e6, ref_uM
    if obs.kind == "bound_frac":
        return (c0_m[:, refs[0]] - eq_m[:, obs.nodes[0]]) * 1e6, ref_uM
    if obs.kind == "dimer_frac":
        u, v = obs.nodes[0], obs.nodes[1]
        mult = 2.0 if (u == v and refs[0] == u) else 1.0
        return mult * eq_m[:, pmap[(u, v)]] * 1e6, ref_uM
    if obs.kind == "sum_free_frac":
        mean_free = np.mean([eq_m[:, u] * 1e6 for u in obs.nodes], axis=0)
        mean_ref = np.mean([c0_m[:, r] * 1e6 for r in refs], axis=0)
        return mean_free, mean_ref
    raise ValueError(f"Unknown observable kind: {obs.kind}")


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

def collect_graph_designs(results_dir: Path, top_k: int) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    gd_dir = results_dir / "graph_designs"
    adam_dir = results_dir / "adam_tuning"
    if not gd_dir.exists():
        return [], [], [], []

    available = {p.name: p for p in gd_dir.iterdir() if p.is_dir() and (p / f"{p.name}_full.json").exists()}
    ordered_names = [n for n in GRAPH_CIRCUIT_ORDER if n in available]
    ordered_names += sorted(n for n in available if n not in ordered_names)

    pairwise_rows: List[dict] = []
    matrix_blocks: List[dict] = []
    tt_rows: List[dict] = []
    curve_rows: List[dict] = []

    for name in ordered_names:
        data = json.loads((available[name] / f"{name}_full.json").read_text(encoding="utf-8"))
        cfg = data.get("config", {})
        mode = cfg.get("mode", "unknown")
        designs = data.get("designs", [])
        if not designs:
            continue

        adam_path = adam_dir / f"{name}_adam.json"
        adam_data = json.loads(adam_path.read_text(encoding="utf-8")) if adam_path.exists() else None

        selected_indices = list(range(min(top_k, len(designs))))
        seen_ids = {designs[i].get("design_id") for i in selected_indices}
        extra_ids = list(EXTRA_PICKS.get(name, []))
        if adam_data and adam_data.get("design_id"):
            extra_ids.append(adam_data["design_id"])
        for eid in extra_ids:
            if eid not in seen_ids:
                for idx, d in enumerate(designs):
                    if d.get("design_id") == eid:
                        selected_indices.append(idx)
                        seen_ids.add(eid)
                        break

        nodes_cfg = [n["name"] for n in cfg.get("nodes", [])]
        states_cfg = cfg.get("states_uM", [])
        input_nodes = [
            nd for nd in nodes_cfg
            if states_cfg and len({round(float(st.get(nd, 0.0)), 6) for st in states_cfg}) > 1
        ]

        for idx in selected_indices:
            design = designs[idx]
            rank = idx + 1
            did = design.get("design_id", f"D{rank:02d}")
            seqs = design.get("seqs", {})
            nodes = nodes_cfg if (nodes_cfg and set(nodes_cfg) == set(seqs.keys())) else list(seqs.keys())
            node_idx = {nd: i for i, nd in enumerate(nodes)}
            n = len(nodes)
            pkd_mat = np.array(design["pkd_matrix"], dtype=float)
            kd_nM_mat = 10.0 ** (9.0 - pkd_mat)
            kd_m_mat = 10.0 ** np.minimum(-pkd_mat, -2.0)
            edges_dict = design.get("edges", {})

            # Map target edges in both directions for lookup
            target_lookup: Dict[Tuple[int, int], Tuple[str, dict]] = {}
            for ename, einfo in edges_dict.items():
                uv = split_edge_nodes(ename, nodes)
                if uv:
                    u_i, v_i = node_idx[uv[0]], node_idx[uv[1]]
                    a, b = (u_i, v_i) if u_i <= v_i else (v_i, u_i)
                    target_lookup[(a, b)] = (ename, einfo)

            for i in range(n):
                for j in range(i, n):
                    pkd_val = float(pkd_mat[i, j])
                    kd_nM_val = float(kd_nM_mat[i, j])
                    if i == j:
                        etype = "Homodimer"
                        orient, shift, slip_m, slip_t = "", "", None, ""
                    elif (i, j) in target_lookup:
                        etype = "Target"
                        _, einfo = target_lookup[(i, j)]
                        orient = einfo.get("orient", "")
                        shift = einfo.get("shift", "")
                        slip_m = einfo.get("slip_margin")
                        slip_t = einfo.get("slip_type", "")
                    else:
                        etype = "Off-target"
                        orient, shift, slip_m, slip_t = "", "", None, ""

                    pairwise_rows.append({
                        "Category": mode,
                        "System": name,
                        "Design_ID": did,
                        "Rank": rank,
                        "Node_i": nodes[i],
                        "Node_j": nodes[j],
                        "Edge_Type": etype,
                        "pKd": round(pkd_val, 4),
                        "Kd_nM": round(kd_nM_val, 4),
                        "Orient": orient,
                        "Shift": shift,
                        "Slip_Margin": round(float(slip_m), 3) if slip_m is not None else None,
                        "Slip_Type": slip_t,
                    })

            matrix_blocks.append({
                "category": mode,
                "system": name,
                "design_id": did,
                "rank": rank,
                "nodes": nodes,
                "pkd_mat": pkd_mat,
                "kd_nM_mat": kd_nM_mat,
            })

            if mode not in ("truth_table", "curve"):
                continue

            # Concentration matrix (prefer ADAM if tuned for this design_id)
            use_adam = adam_data is not None and adam_data.get("design_id") == did
            c0_uM = np.array(design["c0_uM"], dtype=float)
            notes_str = ""
            if use_adam:
                adam_names = adam_data.get("node_names", nodes)
                adam_map = {nm: k for k, nm in enumerate(adam_names)}
                c0_adam_raw = np.array(adam_data["c0_uM_after"], dtype=float)
                c0_uM = np.column_stack([c0_adam_raw[:, adam_map[nm]] for nm in nodes])
                notes_str = "ADAM c0"
            if name == "dome_3node" and did == "D08":
                notes_str = "Low-homodimer pick" if not notes_str else f"{notes_str}; Low-homodimer pick"

            c0_m = c0_uM * 1e-6
            eq_m = solve_general_equilibrium(c0_m, kd_m_mat)
            mb_err = mass_balance_rel_err_eqtk(eq_m, c0_m, n)
            pmap = pair_index_map(n)

            # Precompute state strings
            state_free_strs: List[str] = []
            state_dimer_strs: List[str] = []
            state_input_strs: List[str] = []
            for s_idx in range(c0_uM.shape[0]):
                state_free_strs.append("; ".join(f"{nodes[k]}={eq_m[s_idx, k]*1e6:.4g}" for k in range(n)))
                d_parts = []
                for (a, b), (ename, _) in target_lookup.items():
                    d_parts.append(f"{ename}={eq_m[s_idx, pmap[(a, b)]]*1e6:.4g}")
                state_dimer_strs.append("; ".join(d_parts))
                if input_nodes:
                    state_input_strs.append("; ".join(f"{nd}={c0_uM[s_idx, node_idx[nd]]:g}" for nd in input_nodes))
                else:
                    state_input_strs.append("")

            obs_list = cfg.get("observables", [])
            for obs_idx, obs_cfg in enumerate(obs_list):
                obs_nodes_idx = tuple(node_idx[nd] for nd in obs_cfg["nodes"])
                obs_refs_idx = (
                    tuple(node_idx[nd] for nd in obs_cfg["ref_nodes"])
                    if obs_cfg.get("ref_nodes")
                    else None
                )
                obs = Observable(kind=obs_cfg["kind"], nodes=obs_nodes_idx, ref_nodes=obs_refs_idx)
                targets = obs_cfg["targets"]
                obs_nodes_str = ":".join(obs_cfg["nodes"])
                read_uM, ref_uM = readout_conc_uM(obs, eq_m, c0_m, n)

                if mode == "truth_table":
                    ev = evaluate_truth_table(kd_m_mat, c0_m, obs, targets)
                    vals = ev["values"]
                    for s_idx in range(c0_uM.shape[0]):
                        if input_nodes:
                            s_label = "".join("1" if c0_uM[s_idx, node_idx[nd]] > 0.0 else "0" for nd in input_nodes)
                        else:
                            s_label = f"S{s_idx}"
                        tt_rows.append({
                            "System": name,
                            "Design_ID": did,
                            "Rank": rank,
                            "State_Idx": s_idx,
                            "State_Label": s_label,
                            "Obs_Idx": obs_idx,
                            "Obs_Kind": obs.kind,
                            "Obs_Nodes": obs_nodes_str,
                            "Target_Bit": int(targets[s_idx]),
                            "Readout_Signal": round(float(vals[s_idx]), 6),
                            "Readout_Conc_uM": round(float(read_uM[s_idx]), 6),
                            "Ref_Conc_uM": round(float(ref_uM[s_idx]), 6),
                            "Delta": round(float(ev["Delta"]), 4),
                            "Fold_Uncensored": round(float(ev["Fold_uncensored"]), 2) if np.isfinite(ev["Fold_uncensored"]) else None,
                            "Inputs_uM": state_input_strs[s_idx],
                            "Free_Monomers_uM": state_free_strs[s_idx],
                            "Target_Dimers_uM": state_dimer_strs[s_idx],
                            "Max_MB_RelErr": float(f"{mb_err[s_idx]:.2e}"),
                            "Notes": notes_str,
                        })
                else:
                    aff = bool(obs_cfg.get("affine_fit", False))
                    ev = evaluate_curve(kd_m_mat, c0_m, obs, targets, affine_fit=aff)
                    x_col = node_idx[input_nodes[0]] if input_nodes else 0
                    for s_idx in range(c0_uM.shape[0]):
                        curve_rows.append({
                            "System": name,
                            "Design_ID": did,
                            "Rank": rank,
                            "Point_Idx": s_idx,
                            "Input_X_uM": round(float(c0_uM[s_idx, x_col]), 6),
                            "Obs_Kind": obs.kind,
                            "Obs_Nodes": obs_nodes_str,
                            "Target_Y": round(float(targets[s_idx]), 6),
                            "Raw_Signal": round(float(ev["raw"][s_idx]), 6),
                            "Y_Pred": round(float(ev["y_pred"][s_idx]), 6),
                            "Readout_Conc_uM": round(float(read_uM[s_idx]), 6),
                            "Ref_Conc_uM": round(float(ref_uM[s_idx]), 6),
                            "R2": round(float(ev["R2"]), 4),
                            "RMSE": round(float(ev["RMSE"]), 4),
                            "Span": round(float(ev["Span"]), 4),
                            "Free_Monomers_uM": state_free_strs[s_idx],
                            "Target_Dimers_uM": state_dimer_strs[s_idx],
                            "Max_MB_RelErr": float(f"{mb_err[s_idx]:.2e}"),
                            "Notes": notes_str,
                        })

    return pairwise_rows, matrix_blocks, tt_rows, curve_rows


def collect_2d_and_banana(
    results_dir: Path,
    all_des_bounds: bool,
) -> Tuple[List[dict], List[dict], List[dict]]:
    pairwise_rows: List[dict] = []
    matrix_blocks: List[dict] = []
    grid_jobs: List[dict] = []

    # 1. des_bounds_designs
    db_dir = results_dir / "des_bounds_designs"
    if db_dir.exists():
        ordered_db = [k for k in DES_BOUNDS_SHEET_NAMES if (db_dir / k).is_dir()]
        for p in sorted(db_dir.iterdir()):
            if not p.is_dir() or p.name in ordered_db:
                continue
            if not all_des_bounds and p.name in LEGACY_DES_BOUNDS:
                continue
            ordered_db.append(p.name)

        for folder_name in ordered_db:
            folder_path = db_dir / folder_name
            reports = sorted(folder_path.glob("*_report.json"))
            if not reports:
                continue
            data = json.loads(reports[0].read_text(encoding="utf-8"))
            if not data.get("seqs") or "pkd_matrix" not in data:
                continue

            nodes: List[str] = data.get("nodes", []) or list(data["seqs"].keys())
            n = len(nodes)
            node_idx = {nd: i for i, nd in enumerate(nodes)}
            pkd_mat = np.array(data["pkd_matrix"], dtype=float)
            kd_nM_mat = 10.0 ** (9.0 - pkd_mat)
            raw_edges = data.get("edges", [])
            edge_names = list(raw_edges.keys()) if isinstance(raw_edges, dict) else list(raw_edges)
            target_set: Set[Tuple[int, int]] = set()
            for ename in edge_names:
                uv = split_edge_nodes(ename, nodes)
                if uv:
                    a, b = node_idx[uv[0]], node_idx[uv[1]]
                    target_set.add((min(a, b), max(a, b)))

            did = str(data.get("boundary", folder_name))
            for i in range(n):
                for j in range(i, n):
                    etype = "Homodimer" if i == j else ("Target" if (i, j) in target_set else "Off-target")
                    pairwise_rows.append({
                        "Category": "2d_boundary",
                        "System": folder_name,
                        "Design_ID": did,
                        "Rank": 1,
                        "Node_i": nodes[i],
                        "Node_j": nodes[j],
                        "Edge_Type": etype,
                        "pKd": round(float(pkd_mat[i, j]), 4),
                        "Kd_nM": round(float(kd_nM_mat[i, j]), 4),
                        "Orient": "PAR" if etype == "Target" else "",
                        "Shift": "",
                        "Slip_Margin": None,
                        "Slip_Type": "",
                    })

            matrix_blocks.append({
                "category": "2d_boundary",
                "system": folder_name,
                "design_id": did,
                "rank": 1,
                "nodes": nodes,
                "pkd_mat": pkd_mat,
                "kd_nM_mat": kd_nM_mat,
            })

            # Solve exact 121x121 equilibrium grid
            s = float(data.get("conc_scale", 1.0))
            max_in = 200.0 * s
            thresh = float(data.get("threshold_nM", 50.0 * s))
            fitted_tot = data.get("fitted_total_nM", {})
            axis = np.linspace(0.0, max_in, 121)
            gx, gy = np.meshgrid(axis, axis)
            c0 = np.zeros((gx.size, n), dtype=float)
            for idx_n, nd in enumerate(nodes):
                if nd == "X":
                    c0[:, idx_n] = gx.ravel()
                elif nd == "Y":
                    c0[:, idx_n] = gy.ravel()
                else:
                    c0[:, idx_n] = float(fitted_tot[nd])

            x_eq, mb_err = solve_eq_pos_np(c0, kd_nM_mat)
            free_a = x_eq[:, node_idx["A"]]
            target_cls = boundary_fn_2d(did, gx, gy, max_in=max_in).ravel().astype(int)
            pred_cls = (free_a > thresh).astype(int)
            correct = (pred_cls == target_cls).astype(int)
            rep_acc = data.get("fitted_metrics", {}).get("accuracy")
            comp_acc = float(np.mean(correct))

            grid_jobs.append({
                "kind": "2d_boundary",
                "system": folder_name,
                "sheet_name": DES_BOUNDS_SHEET_NAMES.get(folder_name, f"2D_{folder_name}"[:31]),
                "grid_shape": "121x121",
                "scale": s,
                "threshold_nM": thresh,
                "computed_acc": comp_acc,
                "reported_acc": rep_acc,
                "max_mb_err": mb_err,
                "headers": ["X_nM", "Y_nM", "Free_A_nM", "Threshold_nM", "Margin_nM", "Pred_Class", "Target_Class", "Correct"],
                "columns": [
                    gx.ravel(),
                    gy.ravel(),
                    free_a,
                    np.full(gx.size, thresh),
                    free_a - thresh,
                    pred_cls,
                    target_cls,
                    correct,
                ],
            })

    # 2. banana_designs
    bn_dir = results_dir / "banana_designs"
    if bn_dir.exists():
        for folder_path in sorted(p for p in bn_dir.iterdir() if p.is_dir()):
            reports = sorted(folder_path.glob("*_report.json"))
            if not reports:
                continue
            data = json.loads(reports[0].read_text(encoding="utf-8"))
            if not data.get("seqs") or "pkd_matrix" not in data:
                continue

            nodes: List[str] = data.get("nodes", []) or list(data["seqs"].keys())
            n = len(nodes)
            node_idx = {nd: i for i, nd in enumerate(nodes)}
            pkd_mat = np.array(data["pkd_matrix"], dtype=float)
            kd_nM_mat = 10.0 ** (9.0 - pkd_mat)
            raw_edges = data.get("edges", {})
            topo = str(data.get("topology", folder_path.name))

            target_lookup: Dict[Tuple[int, int], dict] = {}
            if isinstance(raw_edges, dict):
                for ename, einfo in raw_edges.items():
                    uv = split_edge_nodes(ename, nodes)
                    if uv:
                        a, b = node_idx[uv[0]], node_idx[uv[1]]
                        target_lookup[(min(a, b), max(a, b))] = einfo

            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        etype = "Homodimer"
                        orient, shift, slip_m, slip_t = "", "", None, ""
                    elif (i, j) in target_lookup:
                        etype = "Target"
                        einfo = target_lookup[(i, j)]
                        orient = einfo.get("orient", "")
                        shift = einfo.get("shift", "")
                        slip_m = einfo.get("slip_margin")
                        slip_t = einfo.get("slip_type", "")
                    else:
                        etype = "Off-target"
                        orient, shift, slip_m, slip_t = "", "", None, ""

                    pairwise_rows.append({
                        "Category": "banana",
                        "System": folder_path.name,
                        "Design_ID": topo,
                        "Rank": 1,
                        "Node_i": nodes[i],
                        "Node_j": nodes[j],
                        "Edge_Type": etype,
                        "pKd": round(float(pkd_mat[i, j]), 4),
                        "Kd_nM": round(float(kd_nM_mat[i, j]), 4),
                        "Orient": orient,
                        "Shift": shift,
                        "Slip_Margin": round(float(slip_m), 3) if slip_m is not None else None,
                        "Slip_Type": slip_t,
                    })

            matrix_blocks.append({
                "category": "banana",
                "system": folder_path.name,
                "design_id": topo,
                "rank": 1,
                "nodes": nodes,
                "pkd_mat": pkd_mat,
                "kd_nM_mat": kd_nM_mat,
            })

            # Solve exact 160x160 uniform_test grid in scaled nM
            s = float(data.get("conc_scale", 1.0))
            tots_scaled = data.get("fitted_total_scaled_nM") or {}
            if not tots_scaled and data.get("fitted_total_native_nM"):
                tots_scaled = {k: float(v) * s for k, v in data["fitted_total_native_nM"].items()}

            readouts = list(data.get("outputs", []))
            bands = BANANA_BANDS.get(topo, 4 if "4" in topo else 3)

            axis_nat = (np.arange(160) + 0.5) * 200.0 / 160.0
            gx_nat, gy_nat = np.meshgrid(axis_nat, axis_nat)
            xy_nat = np.column_stack((gx_nat.ravel(), gy_nat.ravel()))
            xy_real = xy_nat * s

            c0_real = np.zeros((xy_real.shape[0], n), dtype=float)
            for idx_n, nd in enumerate(nodes):
                if nd == "X":
                    c0_real[:, idx_n] = xy_real[:, 0]
                elif nd == "Y":
                    c0_real[:, idx_n] = xy_real[:, 1]
                else:
                    c0_real[:, idx_n] = float(tots_scaled[nd])

            x_eq_real, mb_err = solve_eq_pos_np(c0_real, kd_nM_mat)
            out_cols = []
            for rname in readouts:
                if len(rname) == 1:
                    out_cols.append(x_eq_real[:, node_idx[rname]])
                else:
                    i0, i1 = node_idx[rname[0]], node_idx[rname[1]]
                    out_cols.append(x_eq_real[:, i0] * x_eq_real[:, i1] / kd_nM_mat[i0, i1])

            out0, out1 = out_cols[0], out_cols[1]
            delta = out1 - out0
            pred_cls = (out1 > out0).astype(int)
            target_cls = banana_band_labels(xy_nat, bands)
            correct = (pred_cls == target_cls).astype(int)
            rep_acc = data.get("stages", {}).get("optimized", {}).get("uniform_test", {}).get("accuracy")
            comp_acc = float(np.mean(correct))

            grid_jobs.append({
                "kind": "banana",
                "system": folder_path.name,
                "sheet_name": BANANA_SHEET_NAMES.get(folder_path.name, f"BN_{folder_path.name}"[:31]),
                "grid_shape": "160x160",
                "scale": s,
                "threshold_nM": 0.0,
                "computed_acc": comp_acc,
                "reported_acc": rep_acc,
                "max_mb_err": mb_err,
                "headers": [
                    "X_nM", "Y_nM", f"Out0_{readouts[0]}_nM", f"Out1_{readouts[1]}_nM",
                    "Delta_Out1_minus_Out0_nM", "Pred_Class", "Target_Class", "Correct",
                ],
                "columns": [xy_real[:, 0], xy_real[:, 1], out0, out1, delta, pred_cls, target_cls, correct],
            })

    return pairwise_rows, matrix_blocks, grid_jobs


# ---------------------------------------------------------------------------
# Excel writers
# ---------------------------------------------------------------------------

FONT_HEADER = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
FONT_SUB = Font(name="Calibri", size=10, bold=True, color="1F3864")
FONT_BODY = Font(name="Calibri", size=10, color="111111")
FILL_HEADER = PatternFill("solid", fgColor="1F3864")
FILL_SUB = PatternFill("solid", fgColor="D9E1F2")
FILL_ZEBRA = PatternFill("solid", fgColor="F2F4F8")
THIN_SIDE = Side(style="thin", color="D0D7DE")
BORDER_CELL = Border(left=THIN_SIDE, right=THIN_SIDE, top=THIN_SIDE, bottom=THIN_SIDE)

TAB_COLORS = {
    "master": "1F3864",
    "truth_table": "2F5597",
    "curve": "2E75B6",
    "graph": "BF9000",
    "2d_boundary": "7030A0",
    "banana": "C55A11",
}


def style_cell(cell: Any, font: Font, fill: Optional[PatternFill], align: str) -> None:
    cell.font = font
    cell.border = BORDER_CELL
    cell.alignment = Alignment(horizontal=align, vertical="center")
    if fill is not None:
        cell.fill = fill


def write_flat_sheet(wb: openpyxl.Workbook, title: str, tab_color: str, dict_rows: List[dict]) -> None:
    ws = wb.create_sheet(title)
    ws.sheet_properties.tabColor = tab_color
    ws.freeze_panes = "A2"
    if not dict_rows:
        return
    headers = list(dict_rows[0].keys())
    for col_idx, h in enumerate(headers, start=1):
        style_cell(ws.cell(row=1, column=col_idx, value=h), FONT_HEADER, FILL_HEADER, "center")

    prev_group = None
    group_toggle = 0
    for r_idx, row in enumerate(dict_rows, start=2):
        group_key = (row.get("System"), row.get("Design_ID"))
        if group_key != prev_group:
            if prev_group is not None:
                group_toggle ^= 1
            prev_group = group_key
        fill = FILL_ZEBRA if group_toggle else None
        for col_idx, h in enumerate(headers, start=1):
            val = row[h]
            align = "right" if isinstance(val, (int, float)) else "left"
            style_cell(ws.cell(row=r_idx, column=col_idx, value=val), FONT_BODY, fill, align)

    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(dict_rows) + 1}"
    for col_idx in range(1, len(headers) + 1):
        max_len = max(
            (len(str(ws.cell(row=r, column=col_idx).value or "")) for r in range(1, min(len(dict_rows) + 2, 500))),
            default=8,
        )
        ws.column_dimensions[get_column_letter(col_idx)].width = max(9, min(48, max_len + 3))


def write_matrix_sheet(wb: openpyxl.Workbook, title: str, tab_color: str, blocks: List[dict]) -> None:
    ws = wb.create_sheet(title)
    ws.sheet_properties.tabColor = tab_color
    row_cursor = 1
    for blk in blocks:
        nodes = blk["nodes"]
        n = len(nodes)
        pkd_mat = blk["pkd_mat"]
        kd_mat = blk["kd_nM_mat"]

        # Header row for this block: System | Design_ID | Rank | pKd | nodes... | <blank> | Kd_nM | nodes...
        col_pkd_start = 4
        col_kd_start = col_pkd_start + n + 2
        style_cell(ws.cell(row=row_cursor, column=1, value=blk["system"]), FONT_HEADER, FILL_HEADER, "left")
        style_cell(ws.cell(row=row_cursor, column=2, value=blk["design_id"]), FONT_HEADER, FILL_HEADER, "center")
        style_cell(ws.cell(row=row_cursor, column=3, value=f"Rank {blk['rank']}"), FONT_HEADER, FILL_HEADER, "center")
        style_cell(ws.cell(row=row_cursor, column=col_pkd_start, value="pKd"), FONT_HEADER, FILL_HEADER, "center")
        style_cell(ws.cell(row=row_cursor, column=col_kd_start, value="Kd_nM"), FONT_HEADER, FILL_HEADER, "center")
        for j, nd in enumerate(nodes):
            style_cell(ws.cell(row=row_cursor, column=col_pkd_start + 1 + j, value=nd), FONT_HEADER, FILL_HEADER, "center")
            style_cell(ws.cell(row=row_cursor, column=col_kd_start + 1 + j, value=nd), FONT_HEADER, FILL_HEADER, "center")

        for i, nd_i in enumerate(nodes):
            r = row_cursor + 1 + i
            fill = FILL_ZEBRA if i % 2 == 1 else None
            style_cell(ws.cell(row=r, column=col_pkd_start, value=nd_i), FONT_SUB, FILL_SUB, "center")
            style_cell(ws.cell(row=r, column=col_kd_start, value=nd_i), FONT_SUB, FILL_SUB, "center")
            for j in range(n):
                style_cell(ws.cell(row=r, column=col_pkd_start + 1 + j, value=round(float(pkd_mat[i, j]), 3)), FONT_BODY, fill, "right")
                style_cell(ws.cell(row=r, column=col_kd_start + 1 + j, value=round(float(kd_mat[i, j]), 3)), FONT_BODY, fill, "right")

        row_cursor += n + 2

    for col_idx in range(1, ws.max_column + 1):
        max_len = max(
            (len(str(ws.cell(row=r, column=col_idx).value or "")) for r in range(1, ws.max_row + 1)),
            default=8,
        )
        ws.column_dimensions[get_column_letter(col_idx)].width = max(8, min(28, max_len + 3))


def write_2d_grids_workbook(
    out_path: Path,
    grid_jobs: List[dict],
    csv_dir: Optional[Path] = None,
) -> None:
    # Ensure unique <=31 char sheet names
    used = {"Grid_Summary"}
    for job in grid_jobs:
        base = job["sheet_name"].replace(":", "_").replace("/", "_")[:31]
        cand = base
        k = 2
        while cand in used:
            suf = f"_{k}"
            cand = base[: 31 - len(suf)] + suf
            k += 1
        used.add(cand)
        job["sheet_name"] = cand

    wb = openpyxl.Workbook(write_only=True)

    # Summary verification tab
    ws_sum = wb.create_sheet("Grid_Summary")
    ws_sum.sheet_properties.tabColor = TAB_COLORS["master"]
    ws_sum.freeze_panes = "A2"
    sum_headers = [
        "Category", "System", "Sheet_Name", "Grid_Shape", "Scale",
        "Threshold_nM", "Computed_Acc_pct", "Reported_Acc_pct", "Acc_Diff_pct", "Max_MB_RelErr",
    ]
    for i in range(1, len(sum_headers) + 1):
        ws_sum.column_dimensions[get_column_letter(i)].width = 18
    ws_sum.column_dimensions["B"].width = 30
    ws_sum.column_dimensions["C"].width = 22

    hdr_cells = []
    for h in sum_headers:
        c = WriteOnlyCell(ws_sum, value=h)
        c.font = FONT_HEADER
        c.fill = FILL_HEADER
        c.alignment = Alignment(horizontal="center")
        hdr_cells.append(c)
    ws_sum.append(hdr_cells)

    for job in grid_jobs:
        rep_acc = job["reported_acc"]
        comp_acc = job["computed_acc"]
        diff_pct = round(100.0 * (comp_acc - rep_acc), 4) if rep_acc is not None else None
        ws_sum.append([
            job["kind"],
            job["system"],
            job["sheet_name"],
            job["grid_shape"],
            job["scale"],
            round(job["threshold_nM"], 3),
            round(100.0 * comp_acc, 3),
            round(100.0 * rep_acc, 3) if rep_acc is not None else None,
            diff_pct,
            float(f"{job['max_mb_err']:.2e}"),
        ])

    if csv_dir is not None:
        csv_dir.mkdir(parents=True, exist_ok=True)

    # Individual 2D grid tabs
    for job in grid_jobs:
        ws = wb.create_sheet(job["sheet_name"])
        ws.sheet_properties.tabColor = TAB_COLORS.get(job["kind"], "7030A0")
        ws.freeze_panes = "A2"
        headers = job["headers"]
        for i in range(1, len(headers) + 1):
            ws.column_dimensions[get_column_letter(i)].width = 16

        h_cells = []
        for h in headers:
            c = WriteOnlyCell(ws, value=h)
            c.font = FONT_HEADER
            c.fill = FILL_HEADER
            c.alignment = Alignment(horizontal="center")
            h_cells.append(c)
        ws.append(h_cells)

        cols = job["columns"]
        n_rows = len(cols[0])
        c0, c1, c2, c3, c4, c5, c6, c7 = (
            np.round(cols[0], 4).tolist(),
            np.round(cols[1], 4).tolist(),
            np.round(cols[2], 6).tolist(),
            np.round(cols[3], 6).tolist(),
            np.round(cols[4], 6).tolist(),
            cols[5].tolist(),
            cols[6].tolist(),
            cols[7].tolist(),
        )
        for r in range(n_rows):
            ws.append([c0[r], c1[r], c2[r], c3[r], c4[r], c5[r], c6[r], c7[r]])

        if csv_dir is not None:
            csv_path = csv_dir / f"{job['system']}_grid.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(headers)
                for r in range(n_rows):
                    writer.writerow([c0[r], c1[r], c2[r], c3[r], c4[r], c5[r], c6[r], c7[r]])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export predicted matrices & equilibrium outputs to Excel.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--out-1d",
        type=Path,
        default=Path("results/Supplementary_Data_2_Matrices_and_1D_Equilibrium.xlsx"),
    )
    parser.add_argument(
        "--out-2d",
        type=Path,
        default=Path("results/Supplementary_Data_3_2D_Grids.xlsx"),
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-K designs per graph_designs system.")
    parser.add_argument("--all-des-bounds", action="store_true", help="Include legacy/ablation des_bounds folders.")
    parser.add_argument("--csv-grids", action="store_true", help="Also write 2D grids as CSVs under results/2d_grids_csv/.")
    args = parser.parse_args()

    pw_1d, mat_1d, tt_rows, curve_rows = collect_graph_designs(args.results_dir, top_k=args.top_k)
    pw_2d, mat_2d, grid_jobs = collect_2d_and_banana(args.results_dir, all_des_bounds=args.all_des_bounds)

    all_pairwise = pw_1d + pw_2d
    all_matrices = mat_1d + mat_2d

    # Workbook 2: Matrices & 1D equilibrium
    wb2 = openpyxl.Workbook()
    wb2.remove(wb2.active)
    write_flat_sheet(wb2, "Affinities_Pairwise", TAB_COLORS["master"], all_pairwise)
    write_flat_sheet(wb2, "TruthTables_Equilibrium", TAB_COLORS["truth_table"], tt_rows)
    write_flat_sheet(wb2, "Curves_Equilibrium", TAB_COLORS["curve"], curve_rows)

    for cat, sheet_name in [
        ("truth_table", "Matrices_Logic"),
        ("curve", "Matrices_Curves"),
        ("graph", "Matrices_Graphs"),
        ("2d_boundary", "Matrices_2D"),
        ("banana", "Matrices_Banana"),
    ]:
        sub = [b for b in all_matrices if b["category"] == cat]
        if sub:
            write_matrix_sheet(wb2, sheet_name, TAB_COLORS[cat], sub)

    args.out_1d.parent.mkdir(parents=True, exist_ok=True)
    wb2.save(args.out_1d)
    print(
        f"Saved {args.out_1d}: {len(all_pairwise)} pairwise rows, "
        f"{len(tt_rows)} truth-table state rows, {len(curve_rows)} curve point rows."
    )

    # Workbook 3: 2D & Banana equilibrium grids
    if grid_jobs:
        csv_dir = (args.results_dir / "2d_grids_csv") if args.csv_grids else None
        write_2d_grids_workbook(args.out_2d, grid_jobs, csv_dir=csv_dir)
        print(f"Saved {args.out_2d}: {len(grid_jobs)} 2D grid sheets.")
        for job in grid_jobs:
            rep = job["reported_acc"]
            rep_s = f"{rep:.2%}" if rep is not None else "n/a"
            print(
                f"  [{job['system']}] {job['grid_shape']}: "
                f"computed_acc={job['computed_acc']:.2%} vs report={rep_s}, max_mb_err={job['max_mb_err']:.2e}"
            )


if __name__ == "__main__":
    main()
