#!/usr/bin/env python3
"""Export top-K designed coiled-coil sequences into clean, flat Excel tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
except ImportError as exc:
    raise SystemExit("openpyxl is required. Install with: pip install openpyxl") from exc


# Canonical order from README (extra folders on disk are appended automatically)
GRAPH_CIRCUIT_ORDER: List[str] = [
    "yes_3node", "not_3node", "or_4node", "nor_4node", "nor_5node",
    "and_5node", "and_6node", "nand_6node", "imply_5node", "nimply_4node",
    "nimply_6node", "xor_4node", "xnor_4node", "xnor_5node", "or3_5node", "ortho2_6node",
    "analog_bandpass", "dome_3node", "analog_neg_x2", "threshold_decoy_4node",
    "custom_graph_xor_topo", "cycle_c4", "bipartite_k23", "cascade_5node", "barbell_6node",
]

# Extra candidate IDs to always include even if rank > top_k
EXTRA_PICKS: Dict[str, List[str]] = {
    "dome_3node": ["D08"],
}

# Early small-scale / superseded 2D runs skipped by default (use --all-des-bounds to include)
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


# ---------------------------------------------------------------------------
# Biochemical calculations
# ---------------------------------------------------------------------------

AA_RESIDUE_MW: Dict[str, float] = {
    "A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 103.1388,
    "E": 129.1155, "Q": 128.1307, "G": 57.0519,  "H": 137.1411, "I": 113.1594,
    "L": 113.1594, "K": 128.1741, "M": 131.1926, "F": 147.1766, "P": 97.1167,
    "S": 87.0782,  "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326,
}
H2O_MW = 18.01528
PKA_POS = {"N_term": 8.6, "K": 10.8, "R": 12.5, "H": 6.5}
PKA_NEG = {"C_term": 3.6, "D": 3.9, "E": 4.1, "C": 8.5, "Y": 10.1}


def clean_seq(seq: str) -> str:
    return seq.strip().lstrip("-").upper()


def calc_mw_da(seq: str) -> Optional[float]:
    s = clean_seq(seq)
    if not s:
        return None
    try:
        return round(sum(AA_RESIDUE_MW[aa] for aa in s) + H2O_MW, 2)
    except KeyError:
        return None


def calc_net_charge(seq: str) -> int:
    s = clean_seq(seq)
    return s.count("K") + s.count("R") - s.count("E") - s.count("D")


def calc_pi(seq: str) -> Optional[float]:
    s = clean_seq(seq)
    if not s:
        return None

    def charge_at(ph: float) -> float:
        q = 1.0 / (1.0 + 10.0 ** (ph - PKA_POS["N_term"]))
        q -= 1.0 / (1.0 + 10.0 ** (PKA_NEG["C_term"] - ph))
        for aa, pka in PKA_POS.items():
            if aa != "N_term":
                n = s.count(aa)
                if n:
                    q += n / (1.0 + 10.0 ** (ph - pka))
        for aa, pka in PKA_NEG.items():
            if aa != "C_term":
                n = s.count(aa)
                if n:
                    q -= n / (1.0 + 10.0 ** (pka - ph))
        return q

    lo, hi = 0.0, 14.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if charge_at(mid) > 0:
            lo = mid
        else:
            hi = mid
    return round(0.5 * (lo + hi), 2)


def detect_length_mode(seq: str, pad: str = "") -> str:
    s = clean_seq(seq)
    if pad.startswith("-----") or len(s) == 24:
        return "35mer"
    if len(s) == 21:
        return "3mer"
    if len(s) == 28:
        return "4mer"
    if len(s) == 35:
        return "5mer"
    return f"{len(s)}aa"


def format_heptads(seq: str, length_mode: str) -> str:
    s = clean_seq(seq)
    if length_mode == "35mer" and len(s) == 24:
        return f"{s[:3]}-{s[3:10]}-{s[10:17]}-{s[17:24]}"
    if len(s) % 7 == 0 and len(s) > 0:
        return "-".join(s[i : i + 7] for i in range(0, len(s), 7))
    return s


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_fold_uncensored(path: Path) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[(row["circuit"], row["design_id"])] = float(row["Fold_uncensored"])
    return out


def summarize_c0_range(
    c0_matrix: Any,
    node_idx: int,
    node_name: str,
    tunable_nodes: Set[str],
    mode: str,
) -> Tuple[str, Optional[float], Optional[float]]:
    if not c0_matrix:
        return ("Topology only", None, None)
    if isinstance(c0_matrix[0], (int, float)):
        c0_matrix = [c0_matrix]
    vals = [float(row[node_idx]) for row in c0_matrix]
    vmin = round(min(vals), 5)
    vmax = round(max(vals), 5)
    uniq = sorted({round(v, 6) for v in vals})
    if len(uniq) == 1:
        role = "Scaffold" if node_name in tunable_nodes else "Fixed"
        return (role, vmin, vmax)
    if mode == "curve" and len(uniq) > 2:
        return ("Input sweep", vmin, vmax)
    return ("Input", vmin, vmax)


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


def parse_graph_designs(
    results_dir: Path,
    fold_uncensored: Dict[Tuple[str, str], float],
    top_k: int,
) -> List[Dict[str, Any]]:
    gd_dir = results_dir / "graph_designs"
    adam_dir = results_dir / "adam_tuning"
    if not gd_dir.exists():
        return []

    available = {p.name: p for p in gd_dir.iterdir() if p.is_dir() and (p / f"{p.name}_full.json").exists()}
    ordered_names = [n for n in GRAPH_CIRCUIT_ORDER if n in available]
    ordered_names += sorted(n for n in available if n not in ordered_names)

    systems: List[Dict[str, Any]] = []
    for name in ordered_names:
        full_path = available[name] / f"{name}_full.json"
        data = json.loads(full_path.read_text(encoding="utf-8"))
        cfg = data.get("config", {})
        mode = cfg.get("mode", "unknown")
        designs = data.get("designs", [])
        if not designs:
            continue

        adam_path = adam_dir / f"{name}_adam.json"
        adam_data = json.loads(adam_path.read_text(encoding="utf-8")) if adam_path.exists() else None

        # Select top_k + explicit extra picks + ADAM design_id if outside top_k
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
        tunable_nodes = {t["node"] for t in cfg.get("tunable_scaffolds", [])}

        sys_candidates: List[Dict[str, Any]] = []
        for idx in selected_indices:
            design = designs[idx]
            rank = idx + 1
            did = design.get("design_id", f"D{rank:02d}")
            seqs = design.get("seqs", {})
            nodes = nodes_cfg if (nodes_cfg and set(nodes_cfg) == set(seqs.keys())) else list(seqs.keys())
            pads = design.get("pads", {})
            pkd_mat = design.get("pkd_matrix", [])
            metrics = dict(design.get("metrics", {}))
            edges_dict = design.get("edges", {})

            use_adam = adam_data is not None and adam_data.get("design_id") == did
            notes: List[str] = []
            if name == "dome_3node" and did == "D08":
                notes.append("Low-homodimer pick")

            # Use ADAM concentrations & updated metrics directly when available
            if use_adam:
                after_m = adam_data.get("after", {})
                delta_val = after_m.get("Delta", metrics.get("Delta"))
                fold_val = after_m.get("Fold_uncensored", after_m.get("Fold", metrics.get("Fold")))
                mc_pass_val = after_m.get("MC_Pass_rate", metrics.get("MC_Pass_rate"))
                mc_p10_val = after_m.get("MC_Delta_p10", metrics.get("MC_Delta_p10"))
                c0_active = adam_data.get("c0_uM_after", design.get("c0_uM", []))
                active_node_idx = {n: i for i, n in enumerate(adam_data.get("node_names", nodes))}
                notes.append("ADAM c0")
            else:
                delta_val = metrics.get("Delta")
                fold_val = fold_uncensored.get((name, did))
                if fold_val is None:
                    min_on = metrics.get("min_ON")
                    max_off = metrics.get("max_OFF")
                    if min_on is not None and max_off is not None and max_off > 0:
                        fold_val = float(min_on) / float(max_off)
                    else:
                        fold_val = metrics.get("Fold")
                mc_pass_val = metrics.get("MC_Pass_rate")
                mc_p10_val = metrics.get("MC_Delta_p10")
                c0_active = design.get("c0_uM", [])
                active_node_idx = {n: i for i, n in enumerate(nodes)}

            # Target partners per node + compact edge string
            node_partners: Dict[str, List[str]] = {n: [] for n in nodes}
            edge_summaries: List[str] = []
            for ename, einfo in edges_dict.items():
                pkd_val = einfo.get("pKd")
                orient = einfo.get("orient", "")
                shift = einfo.get("shift", "")
                slip = einfo.get("slip_margin")
                slip_txt = f", slip={slip:.1f}" if slip is not None else ""
                pkd_txt = f"{pkd_val:.2f}" if pkd_val is not None else "?"
                edge_summaries.append(f"{ename}={pkd_txt}")
                uv = split_edge_nodes(ename, nodes)
                if uv:
                    u, v = uv
                    node_partners[u].append(f"{v}:{pkd_txt} ({orient}, sh={shift}{slip_txt})")
                    node_partners[v].append(f"{u}:{pkd_txt} ({orient}, sh={shift}{slip_txt})")

            worst_off = metrics.get("worst_off_pKd")
            if worst_off is not None and worst_off <= -50:
                worst_off = None

            cand_metrics = {
                "category": mode,
                "system": name,
                "design_id": did,
                "rank": rank,
                "n_peptides": len(nodes),
                "lengths": ", ".join(sorted({detect_length_mode(seqs[n], pads.get(n, "")) for n in nodes})),
                "delta": round(delta_val, 4) if delta_val is not None else None,
                "fold": round(fold_val, 2) if fold_val is not None else None,
                "r2": round(metrics["R2"], 4) if "R2" in metrics else None,
                "rmse": round(metrics["RMSE"], 4) if "RMSE" in metrics else None,
                "span": round(metrics["Span"], 4) if "Span" in metrics else None,
                "graph_loss": round(metrics["graph_loss"], 3) if "graph_loss" in metrics else None,
                "acc_pct": None,
                "contour_err_pct": None,
                "mc_pass": round(mc_pass_val, 4) if mc_pass_val is not None else None,
                "mc_delta_p10": round(mc_p10_val, 4) if mc_p10_val is not None else None,
                "worst_off_pkd": round(worst_off, 3) if worst_off is not None else None,
                "worst_homo_pkd": round(metrics["worst_homo_pKd"], 3) if "worst_homo_pKd" in metrics else None,
                "min_slip_margin": round(metrics["min_slip_margin"], 2) if "min_slip_margin" in metrics else None,
                "target_edges": "; ".join(edge_summaries),
                "notes": "; ".join(notes),
            }

            peptides: List[Dict[str, Any]] = []
            for i, node in enumerate(nodes):
                raw_seq = clean_seq(seqs[node])
                pad_seq = pads.get(node, "")
                lm = detect_length_mode(raw_seq, pad_seq)
                n_idx = active_node_idx.get(node, i)
                role, c_min, c_max = summarize_c0_range(c0_active, n_idx, node, tunable_nodes, mode)
                homo_pkd = pkd_mat[i][i] if pkd_mat and i < len(pkd_mat) else None

                peptides.append({
                    "node": node,
                    "length_mode": lm,
                    "length_aa": len(raw_seq),
                    "sequence": raw_seq,
                    "heptad_formatted": format_heptads(raw_seq, lm),
                    "register_pad": pad_seq,
                    "mw_da": calc_mw_da(raw_seq),
                    "net_charge": calc_net_charge(raw_seq),
                    "pi": calc_pi(raw_seq),
                    "conc_role": role,
                    "conc_min_uM": c_min,
                    "conc_max_uM": c_max,
                    "homo_pkd": round(homo_pkd, 3) if homo_pkd is not None else None,
                    "target_partners": "; ".join(node_partners[node]) if node_partners[node] else "",
                })

            sys_candidates.append({"metrics": cand_metrics, "peptides": peptides})

        systems.append({
            "category": mode,
            "system": name,
            "sheet_name": name[:31],
            "candidates": sys_candidates,
        })

    return systems


def parse_des_bounds(results_dir: Path, all_folders: bool = False) -> List[Dict[str, Any]]:
    db_dir = results_dir / "des_bounds_designs"
    if not db_dir.exists():
        return []

    ordered_names = [k for k in DES_BOUNDS_SHEET_NAMES if (db_dir / k).is_dir()]
    for p in sorted(db_dir.iterdir()):
        if not p.is_dir() or p.name in ordered_names:
            continue
        if not all_folders and p.name in LEGACY_DES_BOUNDS:
            continue
        ordered_names.append(p.name)

    systems: List[Dict[str, Any]] = []
    for folder_name in ordered_names:
        folder_path = db_dir / folder_name
        reports = sorted(folder_path.glob("*_report.json"))
        if not reports:
            reports = sorted(folder_path.glob("molecules_*.json"))
        if not reports:
            continue

        data = json.loads(reports[0].read_text(encoding="utf-8"))
        seqs: Dict[str, str] = data.get("seqs", {})
        if not seqs:
            continue
        nodes: List[str] = data.get("nodes", []) or list(seqs.keys())
        pads: Dict[str, str] = data.get("pads", {})
        pkd_mat = data.get("pkd_matrix", [])
        conc_scale = float(data.get("conc_scale", 1.0))
        fitted_nM = data.get("fitted_total_nM", {})

        raw_edges = data.get("edges", [])
        edge_names = list(raw_edges.keys()) if isinstance(raw_edges, dict) else list(raw_edges)
        node_idx = {n: i for i, n in enumerate(nodes)}
        node_partners: Dict[str, List[str]] = {n: [] for n in nodes}
        edge_summaries: List[str] = []

        for ename in edge_names:
            uv = split_edge_nodes(ename, nodes)
            if uv and pkd_mat:
                u, v = uv
                pkd_val = pkd_mat[node_idx[u]][node_idx[v]]
                pkd_txt = f"{pkd_val:.2f}"
                node_partners[u].append(f"{v}:{pkd_txt}")
                node_partners[v].append(f"{u}:{pkd_txt}")
                edge_summaries.append(f"{ename}={pkd_txt}")

        fm = data.get("fitted_metrics", {})
        fme = data.get("fitted_metrics_extra", {})
        acc = fm.get("accuracy")
        contour_pct = fme.get("contour_error_pct_of_domain", fm.get("contour_error_pct_of_domain"))
        err_nM = fm.get("error_nM")
        notes: List[str] = [f"scale={conc_scale:g}"]
        if data.get("threshold_nM") is not None:
            notes.append(f"thresh={data['threshold_nM']} nM")
        if err_nM is not None:
            notes.append(f"err={err_nM:.2f} nM")
        if fm.get("loss") is not None:
            notes.append(f"BCE={fm['loss']:.4f}")

        did = str(data.get("boundary", folder_name))
        cand_metrics = {
            "category": "2d_boundary",
            "system": folder_name,
            "design_id": did,
            "rank": 1,
            "n_peptides": len(nodes),
            "lengths": ", ".join(sorted({detect_length_mode(seqs[n], pads.get(n, "")) for n in nodes})),
            "delta": None,
            "fold": None,
            "r2": None,
            "rmse": None,
            "span": None,
            "graph_loss": None,
            "acc_pct": round(100.0 * acc, 2) if acc is not None else None,
            "contour_err_pct": round(contour_pct, 2) if contour_pct is not None else None,
            "mc_pass": None,
            "mc_delta_p10": None,
            "worst_off_pkd": round(data["worst_off_pKd"], 3) if data.get("worst_off_pKd") is not None else None,
            "worst_homo_pkd": round(data["worst_homo_pKd"], 3) if data.get("worst_homo_pKd") is not None else None,
            "min_slip_margin": None,
            "target_edges": "; ".join(edge_summaries),
            "notes": "; ".join(notes),
        }

        peptides: List[Dict[str, Any]] = []
        for i, node in enumerate(nodes):
            raw_seq = clean_seq(seqs[node])
            pad_seq = pads.get(node, "")
            lm = detect_length_mode(raw_seq, pad_seq)
            if node in fitted_nM:
                role = "Fixed (fitted)"
                val = round(float(fitted_nM[node]) / 1000.0, 5)
                c_min, c_max = val, val
            else:
                role = "2D Input grid"
                c_min, c_max = 0.0, round(0.2 * conc_scale, 5)
            homo_pkd = pkd_mat[i][i] if pkd_mat and i < len(pkd_mat) else None

            peptides.append({
                "node": node,
                "length_mode": lm,
                "length_aa": len(raw_seq),
                "sequence": raw_seq,
                "heptad_formatted": format_heptads(raw_seq, lm),
                "register_pad": pad_seq,
                "mw_da": calc_mw_da(raw_seq),
                "net_charge": calc_net_charge(raw_seq),
                "pi": calc_pi(raw_seq),
                "conc_role": role,
                "conc_min_uM": c_min,
                "conc_max_uM": c_max,
                "homo_pkd": round(homo_pkd, 3) if homo_pkd is not None else None,
                "target_partners": "; ".join(node_partners[node]) if node_partners[node] else "",
            })

        short_name = DES_BOUNDS_SHEET_NAMES.get(folder_name, f"2D_{folder_name}"[:31])
        systems.append({
            "category": "2d_boundary",
            "system": folder_name,
            "sheet_name": short_name,
            "candidates": [{"metrics": cand_metrics, "peptides": peptides}],
        })

    return systems


def parse_banana_designs(results_dir: Path) -> List[Dict[str, Any]]:
    bn_dir = results_dir / "banana_designs"
    if not bn_dir.exists():
        return []

    systems: List[Dict[str, Any]] = []
    for folder_path in sorted(p for p in bn_dir.iterdir() if p.is_dir()):
        reports = sorted(folder_path.glob("*_report.json"))
        if not reports:
            reports = sorted(folder_path.glob("peptide_kd_*.json"))
        if not reports:
            continue

        data = json.loads(reports[0].read_text(encoding="utf-8"))
        seqs: Dict[str, str] = data.get("seqs", {})
        if not seqs:
            continue
        nodes: List[str] = data.get("nodes", []) or list(seqs.keys())
        pads: Dict[str, str] = data.get("pads", {})
        pkd_mat = data.get("pkd_matrix", [])
        conc_scale = float(data.get("conc_scale", 1.0))

        conc_scaled = data.get("fitted_total_scaled_nM") or data.get("fitted_total_nM") or {}
        if not conc_scaled and data.get("fitted_total_native_nM"):
            conc_scaled = {k: float(v) * conc_scale for k, v in data["fitted_total_native_nM"].items()}

        raw_edges = data.get("edges", {})
        node_partners: Dict[str, List[str]] = {n: [] for n in nodes}
        edge_summaries: List[str] = []
        slip_vals: List[float] = []

        if isinstance(raw_edges, dict):
            for ename, einfo in raw_edges.items():
                pkd_val = einfo.get("pKd")
                orient = einfo.get("orient", "")
                shift = einfo.get("shift", "")
                slip = einfo.get("slip_margin")
                if slip is not None:
                    slip_vals.append(float(slip))
                pkd_txt = f"{pkd_val:.2f}" if pkd_val is not None else "?"
                slip_txt = f", slip={slip:.1f}" if slip is not None else ""
                edge_summaries.append(f"{ename}={pkd_txt}")
                uv = split_edge_nodes(ename, nodes)
                if uv:
                    u, v = uv
                    node_partners[u].append(f"{v}:{pkd_txt} ({orient}, sh={shift}{slip_txt})")
                    node_partners[v].append(f"{u}:{pkd_txt} ({orient}, sh={shift}{slip_txt})")

        stages = data.get("stages", {})
        opt_test = (
            stages.get("optimized", {}).get("uniform_test")
            or stages.get("optimized", {}).get("uniform_validation")
            or {}
        )
        acc = opt_test.get("accuracy")
        mc_val = data.get("mc_uniform_val") or {}
        mc_acc = mc_val.get("MC_Acc_med")
        outputs = ", ".join(data.get("outputs", []))
        notes: List[str] = [f"scale={conc_scale:g}"]
        if outputs:
            notes.append(f"outputs={outputs}")

        did = str(data.get("topology", folder_path.name))
        cand_metrics = {
            "category": "banana",
            "system": folder_path.name,
            "design_id": did,
            "rank": 1,
            "n_peptides": len(nodes),
            "lengths": ", ".join(sorted({detect_length_mode(seqs[n], pads.get(n, "")) for n in nodes})),
            "delta": None,
            "fold": None,
            "r2": None,
            "rmse": None,
            "span": None,
            "graph_loss": None,
            "acc_pct": round(100.0 * acc, 2) if acc is not None else None,
            "contour_err_pct": None,
            "mc_pass": round(mc_acc, 4) if mc_acc is not None else None,
            "mc_delta_p10": None,
            "worst_off_pkd": round(data["worst_off_pKd"], 3) if data.get("worst_off_pKd") is not None else None,
            "worst_homo_pkd": round(data["worst_homo_pKd"], 3) if data.get("worst_homo_pKd") is not None else None,
            "min_slip_margin": round(min(slip_vals), 2) if slip_vals else None,
            "target_edges": "; ".join(edge_summaries),
            "notes": "; ".join(notes),
        }

        peptides: List[Dict[str, Any]] = []
        for i, node in enumerate(nodes):
            raw_seq = clean_seq(seqs[node])
            pad_seq = pads.get(node, "")
            lm = detect_length_mode(raw_seq, pad_seq)
            if node in conc_scaled:
                role = "Fixed (scaled)"
                val = round(float(conc_scaled[node]) / 1000.0, 5)
                c_min, c_max = val, val
            else:
                role = "Input grid"
                c_min, c_max = 0.0, round(0.2 * conc_scale, 5)
            homo_pkd = pkd_mat[i][i] if pkd_mat and i < len(pkd_mat) else None

            peptides.append({
                "node": node,
                "length_mode": lm,
                "length_aa": len(raw_seq),
                "sequence": raw_seq,
                "heptad_formatted": format_heptads(raw_seq, lm),
                "register_pad": pad_seq,
                "mw_da": calc_mw_da(raw_seq),
                "net_charge": calc_net_charge(raw_seq),
                "pi": calc_pi(raw_seq),
                "conc_role": role,
                "conc_min_uM": c_min,
                "conc_max_uM": c_max,
                "homo_pkd": round(homo_pkd, 3) if homo_pkd is not None else None,
                "target_partners": "; ".join(node_partners[node]) if node_partners[node] else "",
            })

        short_name = BANANA_SHEET_NAMES.get(folder_path.name, f"BN_{folder_path.name}"[:31])
        systems.append({
            "category": "banana",
            "system": folder_path.name,
            "sheet_name": short_name,
            "candidates": [{"metrics": cand_metrics, "peptides": peptides}],
        })

    return systems


# ---------------------------------------------------------------------------
# Deduplication & Excel writing
# ---------------------------------------------------------------------------

def ensure_unique_sheet_names(systems: List[Dict[str, Any]]) -> None:
    used = {"Master", "Summary"}
    for sys in systems:
        base = sys["sheet_name"].replace(":", "_").replace("/", "_")[:31]
        cand = base
        idx = 2
        while cand in used:
            suffix = f"_{idx}"
            cand = base[: 31 - len(suffix)] + suffix
            idx += 1
        used.add(cand)
        sys["sheet_name"] = cand


def assign_unique_seq_ids(systems: List[Dict[str, Any]]) -> Tuple[int, int]:
    seq_to_id: Dict[str, str] = {}
    seq_counts: Dict[str, int] = {}
    total = 0
    for sys in systems:
        for cand in sys["candidates"]:
            for pep in cand["peptides"]:
                total += 1
                s = pep["sequence"]
                if s not in seq_to_id:
                    seq_to_id[s] = f"PEP_{len(seq_to_id) + 1:03d}"
                seq_counts[s] = seq_counts.get(s, 0) + 1
    for sys in systems:
        for cand in sys["candidates"]:
            for pep in cand["peptides"]:
                s = pep["sequence"]
                pep["unique_seq_id"] = seq_to_id[s]
                pep["seq_occurrences"] = seq_counts[s]
    return total, len(seq_to_id)


FONT_HEADER = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
FONT_BODY = Font(name="Calibri", size=10, color="111111")
FONT_MONO = Font(name="Consolas", size=10, color="111111")
FILL_HEADER = PatternFill("solid", fgColor="1F3864")
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

MONO_COLS = {"Sequence", "Heptad_Formatted", "Register_Pad"}


def style_cell(cell: Any, font: Font, fill: Optional[PatternFill], align: str) -> None:
    cell.font = font
    cell.border = BORDER_CELL
    cell.alignment = Alignment(horizontal=align, vertical="center")
    if fill is not None:
        cell.fill = fill


def write_flat_table(
    ws: Any,
    headers: List[str],
    rows: List[Tuple[List[Any], Optional[PatternFill]]],
) -> None:
    ws.freeze_panes = "A2"
    for col_idx, h in enumerate(headers, start=1):
        style_cell(ws.cell(row=1, column=col_idx, value=h), FONT_HEADER, FILL_HEADER, "center")

    for r_idx, (vals, fill) in enumerate(rows, start=2):
        for col_idx, val in enumerate(vals, start=1):
            h = headers[col_idx - 1]
            font = FONT_MONO if h in MONO_COLS else FONT_BODY
            align = "right" if isinstance(val, (int, float)) else "left"
            style_cell(ws.cell(row=r_idx, column=col_idx, value=val), font, fill, align)

    if rows:
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(rows) + 1}"

    for col_idx in range(1, len(headers) + 1):
        col_letter = get_column_letter(col_idx)
        max_len = 0
        for row_idx in range(1, len(rows) + 2):
            v = ws.cell(row=row_idx, column=col_idx).value
            if v is not None:
                max_len = max(max_len, len(str(v)))
        ws.column_dimensions[col_letter].width = max(8, min(46, max_len + 3))


def category_metric_cols(category: str) -> List[Tuple[str, str]]:
    if category == "truth_table":
        return [
            ("Delta", "delta"), ("Fold", "fold"), ("MC_Pass", "mc_pass"), ("MC_Delta_p10", "mc_delta_p10"),
            ("Worst_Off_pKd", "worst_off_pkd"), ("Worst_Homo_pKd", "worst_homo_pkd"), ("Min_Slip_Margin", "min_slip_margin"),
        ]
    if category == "curve":
        return [
            ("R2", "r2"), ("RMSE", "rmse"), ("Span", "span"), ("MC_Pass", "mc_pass"),
            ("Worst_Off_pKd", "worst_off_pkd"), ("Worst_Homo_pKd", "worst_homo_pkd"), ("Min_Slip_Margin", "min_slip_margin"),
        ]
    if category == "graph":
        return [
            ("Graph_Loss", "graph_loss"), ("Worst_Off_pKd", "worst_off_pkd"),
            ("Worst_Homo_pKd", "worst_homo_pkd"), ("Min_Slip_Margin", "min_slip_margin"),
        ]
    if category == "2d_boundary":
        return [
            ("Acc_pct", "acc_pct"), ("Contour_Err_pct", "contour_err_pct"),
            ("Worst_Off_pKd", "worst_off_pkd"), ("Worst_Homo_pKd", "worst_homo_pkd"),
        ]
    return [
        ("Acc_pct", "acc_pct"), ("MC_Pass", "mc_pass"),
        ("Worst_Off_pKd", "worst_off_pkd"), ("Worst_Homo_pKd", "worst_homo_pkd"), ("Min_Slip_Margin", "min_slip_margin"),
    ]


def write_master_sheet(wb: openpyxl.Workbook, systems: List[Dict[str, Any]]) -> None:
    ws = wb.create_sheet("Master")
    ws.sheet_properties.tabColor = TAB_COLORS["master"]
    headers = [
        "Category", "System", "Design_ID", "Rank", "Node", "Unique_Seq_ID", "Seq_Occurrences",
        "Length_Mode", "Length_AA", "Sequence", "Heptad_Formatted", "Register_Pad",
        "MW_Da", "Net_Charge", "pI", "Conc_Role", "Conc_Min_uM", "Conc_Max_uM",
        "Homodimer_pKd", "Target_Partners", "Delta", "Fold", "R2", "RMSE", "Acc_pct",
        "MC_Pass", "Worst_Off_pKd", "Worst_Homo_pKd", "Min_Slip_Margin", "Notes",
    ]
    rows: List[Tuple[List[Any], Optional[PatternFill]]] = []
    group_idx = 0
    for sys in systems:
        for cand in sys["candidates"]:
            m = cand["metrics"]
            fill = FILL_ZEBRA if group_idx % 2 == 1 else None
            for p in cand["peptides"]:
                rows.append(([
                    m["category"], m["system"], m["design_id"], m["rank"], p["node"],
                    p["unique_seq_id"], p["seq_occurrences"], p["length_mode"], p["length_aa"],
                    p["sequence"], p["heptad_formatted"], p["register_pad"], p["mw_da"],
                    p["net_charge"], p["pi"], p["conc_role"], p["conc_min_uM"], p["conc_max_uM"],
                    p["homo_pkd"], p["target_partners"], m["delta"],
                    m["fold"], m["r2"], m["rmse"], m["acc_pct"], m["mc_pass"],
                    m["worst_off_pkd"], m["worst_homo_pkd"], m["min_slip_margin"], m["notes"],
                ], fill))
            group_idx += 1
    write_flat_table(ws, headers, rows)


def write_summary_sheet(wb: openpyxl.Workbook, systems: List[Dict[str, Any]]) -> None:
    ws = wb.create_sheet("Summary")
    ws.sheet_properties.tabColor = TAB_COLORS["master"]
    headers = [
        "Category", "System", "Design_ID", "Rank", "N_Peptides", "Lengths",
        "Delta", "Fold", "R2", "RMSE", "Span", "Graph_Loss",
        "Acc_pct", "Contour_Err_pct", "MC_Pass", "MC_Delta_p10",
        "Worst_Off_pKd", "Worst_Homo_pKd", "Min_Slip_Margin", "Target_Edges_pKd", "Notes",
    ]
    rows: List[Tuple[List[Any], Optional[PatternFill]]] = []
    for idx, sys in enumerate(systems):
        fill = FILL_ZEBRA if idx % 2 == 1 else None
        for cand in sys["candidates"]:
            m = cand["metrics"]
            rows.append(([
                m["category"], m["system"], m["design_id"], m["rank"], m["n_peptides"], m["lengths"],
                m["delta"], m["fold"], m["r2"], m["rmse"], m["span"], m["graph_loss"],
                m["acc_pct"], m["contour_err_pct"], m["mc_pass"], m["mc_delta_p10"],
                m["worst_off_pkd"], m["worst_homo_pkd"], m["min_slip_margin"], m["target_edges"], m["notes"],
            ], fill))
    write_flat_table(ws, headers, rows)


def write_system_sheet(wb: openpyxl.Workbook, sys: Dict[str, Any]) -> None:
    ws = wb.create_sheet(sys["sheet_name"])
    ws.sheet_properties.tabColor = TAB_COLORS.get(sys["category"], "808080")

    metric_cols = category_metric_cols(sys["category"])
    headers = [
        "Design_ID", "Rank", "Node", "Unique_Seq_ID", "Length_Mode", "Length_AA",
        "Sequence", "Heptad_Formatted", "Register_Pad", "MW_Da", "Net_Charge",
        "pI", "Conc_Role", "Conc_Min_uM", "Conc_Max_uM", "Homodimer_pKd",
        "Target_Partners",
    ] + [h for h, _ in metric_cols] + ["Notes"]

    rows: List[Tuple[List[Any], Optional[PatternFill]]] = []
    for c_idx, cand in enumerate(sys["candidates"]):
        m = cand["metrics"]
        fill = FILL_ZEBRA if c_idx % 2 == 1 else None
        for p in cand["peptides"]:
            vals = [
                m["design_id"], m["rank"], p["node"], p["unique_seq_id"], p["length_mode"],
                p["length_aa"], p["sequence"], p["heptad_formatted"], p["register_pad"],
                p["mw_da"], p["net_charge"], p["pi"], p["conc_role"],
                p["conc_min_uM"], p["conc_max_uM"], p["homo_pkd"], p["target_partners"],
            ] + [m[k] for _, k in metric_cols] + [m["notes"]]
            rows.append((vals, fill))
    write_flat_table(ws, headers, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export top-K CC-any-graph sequences to Excel.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Path to results directory.")
    parser.add_argument("--out", type=Path, default=Path("results/Supplementary_Sequences.xlsx"), help="Output .xlsx path.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of top candidates per graph_designs circuit.")
    parser.add_argument("--all-des-bounds", action="store_true", help="Include legacy/ablation folders in des_bounds_designs.")
    args = parser.parse_args()

    fold_unc = load_fold_uncensored(args.results_dir / "fold_uncensored.csv")

    systems: List[Dict[str, Any]] = []
    systems.extend(parse_graph_designs(args.results_dir, fold_unc, top_k=args.top_k))
    systems.extend(parse_des_bounds(args.results_dir, all_folders=args.all_des_bounds))
    systems.extend(parse_banana_designs(args.results_dir))

    if not systems:
        raise SystemExit(f"No designs found under {args.results_dir}")

    ensure_unique_sheet_names(systems)
    total_peps, unique_peps = assign_unique_seq_ids(systems)

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    write_master_sheet(wb, systems)
    write_summary_sheet(wb, systems)
    for sys in systems:
        write_system_sheet(wb, sys)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.out)
    print(f"Saved {args.out}: {len(systems)} sheets, {total_peps} peptide rows ({unique_peps} unique sequences).")


if __name__ == "__main__":
    main()
