"""Generate Figure 2 (16 logic gates + topologies + compact-vs-extended panel) and Supplement (MC diagnostics)."""

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.circuit_eq import Observable, solve_general_equilibrium
from src.consensus import DEFAULT_CALIB_PATH
from src.graph_design import _parse_config


def set_params():
    mpl.rcParams["figure.figsize"] = (210 / 25.4, 297 / 25.4)
    mpl.rcParams["figure.facecolor"] = "none"
    mpl.rcParams["axes.facecolor"] = "none"
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["axes.labelsize"] = 6
    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["ytick.labelsize"] = 6
    mpl.rcParams["legend.fontsize"] = 7


COL_ON = "#0F766E"
COL_OFF = "#94A3B8"
COL_ON_ALT = "#0284C7"
COL_FAIL = "#DC2626"
COL_FAIL_ALT = "#F59E0B"
COL_THRESH = "#334155"
COL_IN1 = "#1D4ED8"
COL_IN2 = "#7C3AED"
COL_IN3 = "#D97706"
COL_MED = "#E2E8F0"
COL_REP = "#FEF3C7"
COL_REP_EDGE = "#B45309"


def _load_sigma(calib_path: Path) -> float:
    with open(calib_path, "r") as f:
        return float(json.load(f).get("loo_rmse", 0.643))


def _pick_best_design(bundle: dict, override_id: Optional[str] = None) -> dict:
    designs = bundle["designs"]
    if override_id:
        for d in designs:
            if d["design_id"] == override_id:
                return d
    valid = [
        d for d in designs
        if d["metrics"].get("min_slip_margin", 0.0) >= 0.0
        and d["metrics"].get("worst_homo_pKd", 0.0) <= 4.35
    ]
    pool = valid if valid else designs
    return max(pool, key=lambda d: (d["metrics"].get("MC_Pass_rate", 0.0), -d["metrics"].get("total_loss", 0.0)))


def _run_mc_truth_table(
    parsed: dict,
    des: dict,
    sigma_kd: float,
    n_mc: int,
    conc_std: float,
    seed: int,
    obs_override: Optional[Observable] = None,
    targets_override: Optional[List[int]] = None,
    invert_vals: bool = False,
    min_delta_override: Optional[float] = None,
    min_fold_override: Optional[float] = None,
    uniform_criterion: Optional[Tuple[float, float]] = None,
) -> dict:
    names = parsed["node_names"]
    n = len(names)
    c0_m_base = np.array(des["c0_uM"], dtype=float) * 1e-6
    base_log10_kd = -np.array(des["pkd_matrix"], dtype=float)
    kd_nom_m = 10.0 ** np.minimum(base_log10_kd, -2.0)

    obs_list = parsed["observables"]
    if obs_override is not None:
        obs_objs = [obs_override]
        tgt_list = [np.asarray(targets_override, dtype=int)]
        md = min_delta_override if min_delta_override is not None else obs_list[0]["min_delta"]
        mf = min_fold_override if min_fold_override is not None else obs_list[0]["min_fold"]
        min_deltas = [md * 0.85]
        min_folds = [mf * 0.85]
    else:
        obs_objs = [o["obj"] for o in obs_list]
        tgt_list = [np.asarray(o["targets"], dtype=int) for o in obs_list]
        min_deltas = [o["min_delta"] * 0.85 for o in obs_list]
        min_folds = [o["min_fold"] * 0.85 for o in obs_list]

    if uniform_criterion is not None:
        min_deltas = [uniform_criterion[0] for _ in obs_objs]
        min_folds = [uniform_criterion[1] for _ in obs_objs]

    eq_nom = solve_general_equilibrium(c0_m_base, kd_nom_m)
    nom_fracs, nom_uM = [], []
    for obj in obs_objs:
        f_nom = obj.evaluate(eq_nom, c0_m_base, n)
        if invert_vals:
            f_nom = 1.0 - f_nom
        ref_idx = obj.ref_nodes[0]
        nom_fracs.append(f_nom)
        nom_uM.append(f_nom * (c0_m_base[:, ref_idx] * 1e6))

    rng = np.random.default_rng(seed)
    tri_i, tri_j = np.triu_indices(n)
    n_states = c0_m_base.shape[0]
    n_obs = len(obs_objs)

    mc_uM = [np.empty((n_mc, n_states), dtype=float) for _ in range(n_obs)]
    deltas_all = np.empty((n_mc, n_obs), dtype=float)
    folds_all = np.empty((n_mc, n_obs), dtype=float)
    pass_all = np.ones(n_mc, dtype=bool)
    categories = []

    for m in range(n_mc):
        pert = base_log10_kd.copy()
        noise = rng.normal(0.0, sigma_kd, size=tri_i.size)
        pert[tri_i, tri_j] += noise
        pert[tri_j, tri_i] = pert[tri_i, tri_j]
        kd_m = 10.0 ** np.minimum(pert, -2.0)

        if conc_std > 0.0:
            c0_run = c0_m_base * np.maximum(1.0 + rng.normal(0.0, conc_std, size=c0_m_base.shape), 0.1)
        else:
            c0_run = c0_m_base

        eq = solve_general_equilibrium(c0_run, kd_m)
        row_fail_reason = None
        for oi, obj in enumerate(obs_objs):
            vals = obj.evaluate(eq, c0_run, n)
            if invert_vals:
                vals = 1.0 - vals
            ref_idx = obj.ref_nodes[0]
            mc_uM[oi][m] = vals * (c0_run[:, ref_idx] * 1e6)

            bits = tgt_list[oi]
            on_v = vals[bits == 1]
            off_v = vals[bits == 0]
            min_on = float(np.min(on_v))
            max_off = float(np.max(off_v))
            d_val = min_on - max_off
            f_val = min_on / max(max_off, 0.005)
            deltas_all[m, oi] = d_val
            folds_all[m, oi] = f_val

            ok = (d_val >= min_deltas[oi]) and (f_val >= min_folds[oi])
            pass_all[m] &= ok
            if not ok and row_fail_reason is None:
                off_idx = np.where(bits == 0)[0]
                on_idx = np.where(bits == 1)[0]
                w_off = int(off_idx[np.argmax(vals[off_idx])])
                w_on = int(on_idx[np.argmin(vals[on_idx])])
                if vals[w_off] > 0.35:
                    row_fail_reason = f"Leak OFF (S{w_off})"
                elif vals[w_on] < 0.55:
                    row_fail_reason = f"Weak ON (S{w_on})"
                else:
                    row_fail_reason = "Low Δ / Fold"

        categories.append("Pass" if pass_all[m] else (row_fail_reason or "Fail"))

    ref_s0 = float(c0_m_base[0, obs_objs[0].ref_nodes[0]] * 1e6)
    return {
        "c0_uM": c0_m_base * 1e6,
        "ref_s0_uM": ref_s0,
        "nom_uM": nom_uM,
        "mc_uM": mc_uM,
        "targets": tgt_list,
        "deltas": np.min(deltas_all, axis=1),
        "folds": np.min(folds_all, axis=1),
        "min_delta_thresh": float(min(min_deltas)),
        "min_fold_thresh": float(min(min_folds)),
        "pass_rate": float(np.mean(pass_all)),
        "categories": categories,
    }


def _state_order_2in(c0_uM: np.ndarray, idx1: int, idx2: Optional[int], swap: bool = False) -> List[int]:
    if idx2 is None:
        s0 = int(np.argmin(c0_uM[:, idx1]))
        s1 = int(np.argmax(c0_uM[:, idx1]))
        return [s0, s1, s0, s1] if swap else [s0, s0, s1, s1]
    a, b = (idx2, idx1) if swap else (idx1, idx2)
    state_map = {}
    for r in range(c0_uM.shape[0]):
        state_map[(int(c0_uM[r, a] > 0.0), int(c0_uM[r, b] > 0.0))] = r
    return [state_map[(0, 0)], state_map[(0, 1)], state_map[(1, 0)], state_map[(1, 1)]]


def _state_order_3in(c0_uM: np.ndarray, idx1: int, idx2: int, idx3: int) -> List[int]:
    state_map = {}
    for r in range(c0_uM.shape[0]):
        state_map[(int(c0_uM[r, idx1] > 0.0), int(c0_uM[r, idx2] > 0.0), int(c0_uM[r, idx3] > 0.0))] = r
    return [state_map[(i, j, k)] for i in (0, 1) for j in (0, 1) for k in (0, 1)]


TOPO_COORDS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "yes_3node": {"I": (0.14, 0.50), "Q": (0.50, 0.50), "S": (0.86, 0.50)},
    "not_3node": {"I": (0.14, 0.50), "A": (0.50, 0.50), "S": (0.86, 0.50)},
    "or_4node": {"I1": (0.14, 0.78), "I2": (0.14, 0.22), "Q": (0.52, 0.50), "S": (0.86, 0.50)},
    "nor_4node": {"I1": (0.14, 0.78), "I2": (0.14, 0.22), "Q": (0.52, 0.50), "S": (0.86, 0.50)},
    "xor_4node": {"A": (0.15, 0.78), "B": (0.15, 0.22), "Q": (0.54, 0.50), "S": (0.86, 0.50)},
    "xnor_4node": {"A": (0.15, 0.78), "B": (0.15, 0.22), "Q": (0.54, 0.50), "S": (0.86, 0.50)},
    "nimply_4node": {"I1": (0.14, 0.78), "I2": (0.14, 0.22), "Q": (0.50, 0.76), "S": (0.86, 0.48)},
    "and_5node": {"I1": (0.13, 0.78), "I2": (0.13, 0.22), "Q4": (0.50, 0.78), "Q3": (0.50, 0.22), "S": (0.86, 0.50)},
    "imply_5node": {"I1": (0.13, 0.78), "I2": (0.13, 0.22), "M3": (0.39, 0.78), "Q3": (0.61, 0.44), "S": (0.87, 0.44)},
    "nand_6node": {"I1": (0.12, 0.78), "I2": (0.12, 0.22), "M1": (0.37, 0.78), "M2": (0.37, 0.22), "Q1": (0.64, 0.50), "S": (0.88, 0.50)},
    "nor_5node": {"I1": (0.12, 0.78), "I2": (0.12, 0.22), "M5": (0.38, 0.50), "Q1": (0.64, 0.50), "S": (0.88, 0.50)},
    "and_6node": {"I1": (0.13, 0.78), "Q1": (0.48, 0.78), "SA": (0.84, 0.75), "I2": (0.13, 0.22), "Q2": (0.48, 0.22), "SB": (0.84, 0.25)},
    "nimply_6node": {"I1": (0.12, 0.78), "Q4": (0.48, 0.78), "S": (0.88, 0.50), "I2": (0.12, 0.22), "M2": (0.37, 0.22), "Q1": (0.62, 0.22)},
    "xnor_5node": {"A": (0.12, 0.80), "B": (0.12, 0.20), "M": (0.40, 0.50), "Q": (0.65, 0.50), "S": (0.88, 0.50)},
    "or3_5node": {"I1": (0.14, 0.84), "I2": (0.14, 0.50), "I3": (0.14, 0.16), "Q": (0.54, 0.50), "S": (0.86, 0.50)},
    "ortho2_6node": {"I1": (0.14, 0.76), "Q1": (0.50, 0.76), "S1": (0.85, 0.76), "I2": (0.14, 0.24), "Q2": (0.50, 0.24), "S2": (0.85, 0.24)},
}


def _draw_mini_topology(ax, circuit_name: str, parsed: dict, des: dict, swap: bool = False, readout_kind: str = "free") -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    coords = TOPO_COORDS.get(circuit_name)
    names = parsed["node_names"]
    if coords is None:
        return
    pkd_mat = np.array(des["pkd_matrix"], dtype=float)
    is_ab_inputs = circuit_name in ("xor_4node", "xnor_4node", "xnor_5node")

    disp_label = {}
    for nm in names:
        if nm == "I":
            disp_label[nm] = "I2" if swap else "I1"
        elif nm == "I1" or (is_ab_inputs and nm == "A"):
            disp_label[nm] = "I2" if swap else "I1"
        elif nm == "I2" or (is_ab_inputs and nm == "B"):
            disp_label[nm] = "I1" if swap else "I2"
        else:
            disp_label[nm] = nm

    for ed in parsed["edges"]:
        u_nm, v_nm = ed["u_name"], ed["v_name"]
        x1, y1 = coords[u_nm]
        x2, y2 = coords[v_nm]
        pkd = pkd_mat[ed["u"], ed["v"]]
        lw = np.clip(0.5 + 0.18 * (pkd - 5.0), 0.6, 1.8)
        ax.plot([x1, x2], [y1, y2], color="#475569", lw=lw, zorder=1, solid_capstyle="round")
        if readout_kind == "bound" and (u_nm == "S" or v_nm == "S"):
            ax.plot([x1, x2], [y1, y2], color=COL_REP_EDGE, lw=2.3, alpha=0.38, zorder=1)

    if readout_kind == "dimer":
        obs_nodes = parsed["observables"][0]["obj"].nodes
        u_nm, v_nm = names[obs_nodes[0]], names[obs_nodes[1]]
        x1, y1 = coords[u_nm]
        x2, y2 = coords[v_nm]
        ax.plot([x1, x2], [y1, y2], color=COL_REP_EDGE, lw=2.4, alpha=0.40, zorder=1)

    for nm in names:
        x, y = coords[nm]
        lbl = disp_label[nm]
        is_rep = nm.startswith("S")
        is_in = lbl in ("I1", "I2", "I3")
        if is_in:
            fc = COL_IN1 if lbl == "I1" else (COL_IN2 if lbl == "I2" else COL_IN3)
            ec, tc, r = "white", "white", 0.115
        elif is_rep:
            fc, ec, tc, r = COL_REP, COL_REP_EDGE, "#78350F", 0.122
        else:
            fc, ec, tc, r = COL_MED, "#475569", "#1E293B", 0.11
        ax.add_patch(Circle((x, y), radius=r, facecolor=fc, edgecolor=ec, lw=0.8, zorder=3))
        ax.text(x, y, lbl, ha="center", va="center", fontsize=4.2, fontweight="bold", color=tc, zorder=4)


def build_figure2(
    bundles: Dict[str, dict],
    out_prefix: Path,
    unit: str = "uM",
    nand_mode: str = "bound5",
    layout: str = "scaled16",
    n_mc: int = 400,
    sigma_kd: float = 0.643,
    conc_std: float = 0.015,
    seed: int = 42,
) -> None:
    set_params()
    scale = 1000.0 if unit == "nM" else 1.0
    unit_lbl = "nM" if unit == "nM" else "µM"
    y_max = 0.56 * scale

    if nand_mode == "bound5":
        nand_spec = ("NAND", "and_5node", False, "[S]bound", "bound5")
    else:
        nand_spec = ("NAND", "nand_6node", False, "[S]free", "normal")

    # Consistent gate + Boolean formula naming across Figure 2 & Supplement
    base_14 = [
        ("YES (I1)", "yes_3node", False, "[S]free", "1in"),
        ("YES (I2)", "yes_3node", True, "[S]free", "1in"),
        ("NOT (I1)", "not_3node", False, "[A:S] dimer", "1in_dimer"),
        ("NOT (I2)", "not_3node", True, "[A:S] dimer", "1in_dimer"),
        ("OR", "or_4node", False, "[S]free", "normal"),
        ("NOR", "nor_4node", False, "[Q:S] dimer", "dimer"),
        ("AND", "and_5node", False, "[S]free", "normal"),
        nand_spec,
        ("NIMPLY (I1 & ¬I2)", "nimply_4node", False, "[S]free", "normal"),
        ("NIMPLY (I2 & ¬I1)", "nimply_4node", True, "[S]free", "normal"),
        ("IMPLY (¬I1 OR I2)", "imply_5node", False, "[S]free", "normal"),
        ("IMPLY (¬I2 OR I1)", "imply_5node", True, "[S]free", "normal"),
        ("XOR", "xor_4node", False, "[S]free", "normal"),
        ("XNOR", "xnor_4node", False, "[Q:S] dimer", "dimer"),
    ]
    if layout == "scaled16":
        panels = base_14 + [
            ("3-input OR", "or3_5node", False, "[S]free", "or3"),
            ("Dual YES", "ortho2_6node", False, "[S1],[S2]", "ortho2"),
        ]
    else:
        panels = base_14 + [
            ("FALSE (0)", "yes_3node", False, "ref", "const0"),
            ("TRUE (1)", "yes_3node", False, "ref", "const1"),
        ]

    sim_cache: Dict[Tuple[str, str], Tuple[dict, dict, dict]] = {}
    for _, ckey, _, _, smode in panels:
        cache_key = (ckey, smode)
        if cache_key in sim_cache:
            continue
        bundle = bundles[ckey]
        parsed = _parse_config(bundle["config"], allow_5mer=True)
        des = _pick_best_design(bundle)
        if smode == "bound5":
            obs0 = parsed["observables"][0]
            inv_tgt = [1 - int(t) for t in obs0["targets"]]
            sim = _run_mc_truth_table(
                parsed, des, sigma_kd, n_mc, conc_std, seed,
                obs_override=obs0["obj"], targets_override=inv_tgt, invert_vals=True,
                min_delta_override=0.50, min_fold_override=4.5,
            )
        else:
            sim = _run_mc_truth_table(parsed, des, sigma_kd, n_mc, conc_std, seed)
        sim_cache[cache_key] = (parsed, des, sim)

    fig = plt.figure()
    outer_grid = fig.add_gridspec(
        4, 4, left=0.09, right=0.94, top=0.955, bottom=0.310, wspace=0.22, hspace=0.37,
    )
    fig.text(0.03, 0.96, "A", fontsize=9.5, fontweight="bold", color="#0F172A")

    for idx, (title, ckey, swap, rtag, smode) in enumerate(panels):
        row_i, col_i = divmod(idx, 4)
        inner = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_grid[row_i, col_i],
            height_ratios=[0.39, 0.61], hspace=0.05,
        )
        ax_top = fig.add_subplot(inner[0])
        ax_bar = fig.add_subplot(inner[1])

        parsed, des, sim = sim_cache[(ckey, smode)]
        names = parsed["node_names"]
        s0_uM = sim["ref_s0_uM"]
        panel_thresh = 0.5 * s0_uM * scale

        if smode in ("const0", "const1"):
            ax_top.axis("off")
            ax_top.text(0.0, 0.68, title, fontsize=5.8, fontweight="bold", color="#0F172A", va="center")
            ax_top.text(0.0, 0.25, "Control baseline", fontsize=4.6, color="#64748B", va="center")
        else:
            ax_top.set_xlim(0, 1)
            ax_top.set_ylim(0, 1)
            ax_top.axis("off")
            t_fs = 4.9 if len(title) > 13 else 5.5
            ax_top.text(0.0, 0.87, title, fontsize=t_fs, fontweight="bold", color="#0F172A", va="center")
            pass_pct = 100.0 * sim["pass_rate"]
            mir_sfx = " (I1↔I2)" if swap else ""
            ax_top.text(0.0, 0.58, f"{rtag} | {des['design_id']}{mir_sfx}", fontsize=4.1, fontweight="bold", color="#1E293B", va="center")
            ax_top.text(0.0, 0.32, f"[S0]={s0_uM:.2g}{unit_lbl} | Δ≥{sim['min_delta_thresh']:.2f}", fontsize=4.1, color="#334155", va="center")
            ax_top.text(0.0, 0.07, f"MC: {pass_pct:.0f}%", fontsize=4.4, fontweight="bold",
                        color=(COL_ON if pass_pct >= 65 else "#B45309"), va="center")

            ax_topo = ax_top.inset_axes([0.47, 0.02, 0.53, 0.94])
            r_kind = "dimer" if "dimer" in smode else ("bound" if smode == "bound5" else "free")
            _draw_mini_topology(ax_topo, ckey, parsed, des, swap=swap, readout_kind=r_kind)

        ax_bar.axhline(s0_uM * scale, color="#CBD5E1", ls=":", lw=0.65, zorder=1)
        ax_bar.axhline(panel_thresh, color=COL_THRESH, ls="--", lw=0.70, zorder=1)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.spines["left"].set_linewidth(0.6)
        ax_bar.spines["bottom"].set_linewidth(0.6)
        ax_bar.tick_params(width=0.6, length=2.0, pad=1.2)

        if smode == "const0":
            ax_bar.bar(np.arange(4), np.full(4, 0.01 * scale), width=0.58, color=COL_OFF, zorder=2)
            ax_bar.set_xticks(range(4))
            ax_bar.set_xticklabels(["00", "01", "10", "11"], fontsize=5.0)
        elif smode == "const1":
            ax_bar.bar(np.arange(4), np.full(4, 0.50 * scale), width=0.58, color=COL_ON, zorder=2)
            ax_bar.set_xticks(range(4))
            ax_bar.set_xticklabels(["00", "01", "10", "11"], fontsize=5.0)
        elif smode == "or3":
            i1, i2, i3 = names.index("I1"), names.index("I2"), names.index("I3")
            order = _state_order_3in(sim["c0_uM"], i1, i2, i3)
            nom_v = sim["nom_uM"][0][order] * scale
            mc_v = sim["mc_uM"][0][:, order] * scale
            tgts = sim["targets"][0][order]
            p10, p50, p90 = np.percentile(mc_v, [10, 50, 90], axis=0)
            cols = [COL_ON if t == 1 else COL_OFF for t in tgts]
            xs = np.arange(8)
            ax_bar.bar(xs, nom_v, width=0.62, color=cols, alpha=0.90, zorder=2)
            ax_bar.errorbar(xs, p50, yerr=[np.maximum(0, p50 - p10), np.maximum(0, p90 - p50)],
                            fmt="o", ms=1.3, color="#0F172A", elinewidth=0.55, capsize=1.0, capthick=0.45, zorder=3)
            ax_bar.set_xticks(xs)
            ax_bar.set_xticklabels(["000", "001", "010", "011", "100", "101", "110", "111"],
                                   fontsize=3.8, rotation=45, ha="right")
        elif smode == "ortho2":
            i1, i2 = names.index("I1"), names.index("I2")
            order = _state_order_2in(sim["c0_uM"], i1, i2, swap=False)
            xs = np.arange(4)
            w = 0.34
            for oi, (offset, on_col) in enumerate([(-w / 2, COL_ON), (w / 2, COL_ON_ALT)]):
                nom_v = sim["nom_uM"][oi][order] * scale
                mc_v = sim["mc_uM"][oi][:, order] * scale
                tgts = sim["targets"][oi][order]
                p10, p50, p90 = np.percentile(mc_v, [10, 50, 90], axis=0)
                cols = [on_col if t == 1 else COL_OFF for t in tgts]
                ax_bar.bar(xs + offset, nom_v, width=w * 0.9, color=cols, alpha=0.90, zorder=2)
                ax_bar.errorbar(xs + offset, p50, yerr=[np.maximum(0, p50 - p10), np.maximum(0, p90 - p50)],
                                fmt="o", ms=1.2, color="#0F172A", elinewidth=0.5, capsize=0.9, capthick=0.45, zorder=3)
            ax_bar.set_xticks(xs)
            ax_bar.set_xticklabels(["00", "01", "10", "11"], fontsize=5.0)
        else:
            if smode.startswith("1in"):
                idx1, idx2 = names.index("I"), None
            else:
                n1 = "I1" if "I1" in names else "A"
                n2 = "I2" if "I2" in names else "B"
                idx1, idx2 = names.index(n1), names.index(n2)
            order = _state_order_2in(sim["c0_uM"], idx1, idx2, swap=swap)
            nom_v = sim["nom_uM"][0][order] * scale
            mc_v = sim["mc_uM"][0][:, order] * scale
            tgts = sim["targets"][0][order]
            p10, p50, p90 = np.percentile(mc_v, [10, 50, 90], axis=0)
            cols = [COL_ON if t == 1 else COL_OFF for t in tgts]
            xs = np.arange(4)
            ax_bar.bar(xs, nom_v, width=0.56, color=cols, alpha=0.90, zorder=2)
            ax_bar.errorbar(xs, p50, yerr=[np.maximum(0, p50 - p10), np.maximum(0, p90 - p50)],
                            fmt="o", ms=1.6, color="#0F172A", elinewidth=0.6, capsize=1.4, capthick=0.5, zorder=3)
            ax_bar.set_xticks(xs)
            ax_bar.set_xticklabels(["00", "01", "10", "11"], fontsize=5.0)

        ax_bar.set_ylim(0.0, y_max)
        yticks = [0, 0.25 * scale, 0.50 * scale]
        ax_bar.set_yticks(yticks)
        if col_i == 0:
            ax_bar.set_yticklabels([f"{y:.2g}" for y in yticks], fontsize=5.2)
            ax_bar.set_ylabel(f"Output ({unit_lbl})", fontsize=5.5, labelpad=2.0)
        else:
            ax_bar.set_yticklabels([])

    leg_bars = [
        FancyBboxPatch((0, 0), 1, 1, facecolor=COL_ON, edgecolor="none", label="ON (nom.)"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=COL_OFF, edgecolor="none", label="OFF (nom.)"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=COL_ON_ALT, edgecolor="none", label="S2 ON"),
        Line2D([0], [0], marker="o", color="#0F172A", ms=2.2, lw=0.65, label="MC med (10–90%)"),
        Line2D([0], [0], color=COL_THRESH, ls="--", lw=0.75, label="0.5·[S0] thresh"),
        Line2D([0], [0], color="#CBD5E1", ls=":", lw=0.75, label="[S0] ceiling"),
    ]
    leg_topo = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_IN1, ms=4.2, label="Input I1"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_IN2, ms=4.2, label="Input I2"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_MED, markeredgecolor="#475569", ms=4.2, label="Mediator"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_REP, markeredgecolor=COL_REP_EDGE, ms=4.4, label="Reporter S"),
        Line2D([0], [0], color="#475569", lw=1.4, label="Edge width ~ pKd"),
        Line2D([0], [0], color=COL_REP_EDGE, lw=2.2, alpha=0.5, label="Readout complex"),
    ]
    fig.legend(handles=leg_bars, loc="center", bbox_to_anchor=(0.51, 0.278),
               ncol=6, frameon=False, fontsize=4.8, handlelength=1.2, columnspacing=0.9)
    fig.legend(handles=leg_topo, loc="center", bbox_to_anchor=(0.51, 0.256),
               ncol=6, frameon=False, fontsize=4.8, handlelength=1.2, columnspacing=0.9)

    # --- Panel B: Compact CC-native vs Extended Topologies ---
    fig.text(0.03, 0.222, "B", fontsize=9.5, fontweight="bold", color="#0F172A")
    bot_grid = fig.add_gridspec(1, 2, left=0.11, right=0.94, top=0.212, bottom=0.052, wspace=0.30)
    ax_b1 = fig.add_subplot(bot_grid[0])
    ax_b2 = fig.add_subplot(bot_grid[1])

    pairs_cmp = [
        ("NOR", ("nor_4node", False), ("nor_5node", False), 4, 5),
        ("NIMPLY", ("nimply_4node", False), ("nimply_6node", False), 4, 6),
        ("AND", ("and_5node", False), ("and_6node", False), 5, 6),
        ("NAND", ("and_5node", True), ("nand_6node", False), 5, 6),
        ("XNOR", ("xnor_4node", False), ("xnor_5node", False), 4, 5),
    ]
    uni = (0.425, 3.825)  # Modal config pass criterion: 0.85 * (min_delta=0.50, min_fold=4.5)
    comp_pass, ext_pass, comp_weak, ext_weak, x_lbls = [], [], [], [], []
    for gname, (ck_c, inv_c), (ck_e, inv_e), nc, ne in pairs_cmp:
        ids_pair = []
        for ck, inv, out_p, out_w in [(ck_c, inv_c, comp_pass, comp_weak), (ck_e, inv_e, ext_pass, ext_weak)]:
            b = bundles[ck]
            pr = _parse_config(b["config"], allow_5mer=True)
            ds = _pick_best_design(b)
            ids_pair.append(ds["design_id"])
            obs0 = pr["observables"][0]
            tgts = [1 - int(t) for t in obs0["targets"]] if inv else obs0["targets"]
            res = _run_mc_truth_table(
                pr, ds, sigma_kd, n_mc, conc_std, seed,
                obs_override=obs0["obj"], targets_override=tgts, invert_vals=inv,
                uniform_criterion=uni,
            )
            out_p.append(100.0 * res["pass_rate"])
            weak_pct = 100.0 * sum(1 for c in res["categories"] if c.startswith("Weak ON")) / n_mc
            out_w.append(weak_pct)
        x_lbls.append(f"{gname}\n({nc}[{ids_pair[0]}] vs {ne}[{ids_pair[1]}])")
        print(f"[Compact vs Extended | {gname:6s}] {ids_pair[0]}={comp_pass[-1]:5.1f}% (WeakON={comp_weak[-1]:4.1f}%) "
              f"| {ids_pair[1]}={ext_pass[-1]:5.1f}% (WeakON={ext_weak[-1]:4.1f}%)")

    xs = np.arange(len(pairs_cmp))
    w = 0.34
    b1_c = ax_b1.bar(xs - w / 2, comp_pass, width=w, color=COL_ON, alpha=0.90, label="Compact CC-native")
    b1_e = ax_b1.bar(xs + w / 2, ext_pass, width=w, color=COL_OFF, alpha=0.90, label="Extended (+1–2 nodes)")
    for b, v in list(zip(b1_c, comp_pass)) + list(zip(b1_e, ext_pass)):
        ax_b1.text(b.get_x() + b.get_width() / 2, v + 1.8, f"{v:.0f}%", ha="center", va="bottom", fontsize=4.3)
    ax_b1.set_xticks(xs)
    ax_b1.set_xticklabels(x_lbls, fontsize=4.3)
    ax_b1.set_ylim(0, 114)
    ax_b1.set_ylabel("MC Pass Rate (%)", fontsize=5.6)
    ax_b1.set_title("Compact vs Extended (modal thresh Δ ≥ 0.425, Fold ≥ 3.8×)", fontsize=5.2, fontweight="bold", pad=3.0)
    ax_b1.legend(loc="upper right", frameon=False, fontsize=4.6)
    ax_b1.spines["top"].set_visible(False)
    ax_b1.spines["right"].set_visible(False)
    ax_b1.tick_params(labelsize=4.9, width=0.6, length=2.0)

    b2_c = ax_b2.bar(xs - w / 2, comp_weak, width=w, color=COL_ON, alpha=0.90, label="Compact CC-native")
    b2_e = ax_b2.bar(xs + w / 2, ext_weak, width=w, color=COL_OFF, alpha=0.90, label="Extended (+1–2 nodes)")
    for b, v in list(zip(b2_c, comp_weak)) + list(zip(b2_e, ext_weak)):
        ax_b2.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=4.3)
    ax_b2.set_xticks(xs)
    ax_b2.set_xticklabels(x_lbls, fontsize=4.3)
    ax_b2.set_ylim(0, max(65.0, max(ext_weak) + 12.0))
    ax_b2.set_ylabel("Weak-ON Failure Rate (%)", fontsize=5.6)
    ax_b2.set_title("Weak-ON Failure Rate (Added Kd Constraints)", fontsize=5.3, fontweight="bold", pad=3.0)
    ax_b2.legend(loc="upper right", frameon=False, fontsize=4.6)
    ax_b2.spines["top"].set_visible(False)
    ax_b2.spines["right"].set_visible(False)
    ax_b2.tick_params(labelsize=4.9, width=0.6, length=2.0)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), transparent=True)
    fig.savefig(out_prefix.with_suffix(".svg"), transparent=True)
    fig.savefig(out_prefix.with_suffix(".png"), facecolor="white", dpi=300)
    plt.close(fig)
    print(f"Saved Figure 2 to {out_prefix}.{{pdf,svg,png}}")


def build_supplement_mc(
    bundles: Dict[str, dict],
    out_prefix: Path,
    n_mc: int = 400,
    sigma_kd: float = 0.643,
    conc_std: float = 0.015,
    seed: int = 42,
) -> None:
    set_params()
    supp_specs = [
        ("yes_3node", "YES (3-node)", False),
        ("not_3node", "NOT (3-node)", False),
        ("or_4node", "OR (4-node)", False),
        ("nor_4node", "NOR (4-node)", False),
        ("nor_5node", "NOR (5-node)", False),
        ("and_5node", "AND (5-node)", False),
        ("and_6node", "AND (6-node split)", False),
        ("and_5node", "NAND (5-node [S]bound)", True),
        ("nand_6node", "NAND (6-node)", False),
        ("imply_5node", "IMPLY (5-node, ¬I1 OR I2)", False),
        ("nimply_4node", "NIMPLY (4-node, I1 & ¬I2)", False),
        ("nimply_6node", "NIMPLY (6-node, I1 & ¬I2)", False),
        ("xor_4node", "XOR (4-node)", False),
        ("xnor_4node", "XNOR (4-node)", False),
        ("xnor_5node", "XNOR (5-node)", False),
        ("or3_5node", "3-input OR (5-node)", False),
        ("ortho2_6node", "Dual YES (6-node)", False),
    ]
    available = [(k, lbl, inv) for k, lbl, inv in supp_specs if k in bundles]
    n_rows = 9
    fig = plt.figure()
    outer_grid = fig.add_gridspec(
        n_rows, 2, left=0.09, right=0.94, top=0.96, bottom=0.05, wspace=0.26, hspace=0.52,
    )

    sims = []
    max_bin_pct = 0.0
    bins = np.linspace(-0.25, 1.0, 21)
    for ckey, label, inv in available:
        bundle = bundles[ckey]
        parsed = _parse_config(bundle["config"], allow_5mer=True)
        des = _pick_best_design(bundle)
        if inv:
            obs0 = parsed["observables"][0]
            sim = _run_mc_truth_table(
                parsed, des, sigma_kd, n_mc, conc_std, seed,
                obs_override=obs0["obj"], targets_override=[1 - int(t) for t in obs0["targets"]],
                invert_vals=True, min_delta_override=0.50, min_fold_override=4.5,
            )
        else:
            sim = _run_mc_truth_table(parsed, des, sigma_kd, n_mc, conc_std, seed)
        h_all, _ = np.histogram(sim["deltas"], bins=bins)
        max_bin_pct = max(max_bin_pct, float(np.max(100.0 * h_all / n_mc)))
        sims.append((label, des, sim))

    y_hist_max = min(100.0, np.ceil(max_bin_pct / 10.0) * 10.0 + 5.0)

    for idx, (label, des, sim) in enumerate(sims):
        row_i, pair_col = divmod(idx, 2)
        inner = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer_grid[row_i, pair_col],
            width_ratios=[1.05, 1.0], wspace=0.44,
        )
        ax_hist = fig.add_subplot(inner[0])
        ax_fail = fig.add_subplot(inner[1])

        deltas = sim["deltas"]
        cats = np.array(sim["categories"])
        pass_d = deltas[cats == "Pass"]
        fail_d = deltas[cats != "Pass"]
        thresh = sim["min_delta_thresh"]
        fold_th = sim["min_fold_thresh"]

        w_pass = np.full(pass_d.size, 100.0 / n_mc)
        w_fail = np.full(fail_d.size, 100.0 / n_mc)
        ax_hist.hist(pass_d, bins=bins, weights=w_pass, color=COL_ON, alpha=0.80)
        ax_hist.hist(fail_d, bins=bins, weights=w_fail, color=COL_FAIL, alpha=0.70)
        ax_hist.axvline(thresh, color=COL_THRESH, ls="--", lw=0.8)
        ax_hist.text(0.03, 0.90, f"Δ≥{thresh:.2f}, F≥{fold_th:.1f}×", transform=ax_hist.transAxes,
                     fontsize=4.1, color=COL_THRESH, va="top")
        ax_hist.set_title(f"{label} [{des['design_id']}] ({100*sim['pass_rate']:.1f}%)",
                          fontsize=4.9, fontweight="bold", loc="left", pad=2.2)
        ax_hist.set_xlim(-0.25, 1.02)
        ax_hist.set_ylim(0.0, y_hist_max)
        ax_hist.set_xticks([-0.25, 0.0, 0.25, 0.50, 0.75, 1.00])
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)
        ax_hist.tick_params(labelsize=4.0, width=0.5, length=1.6, pad=0.8)
        if row_i == n_rows - 1 or idx == len(sims) - 2:
            ax_hist.set_xlabel("MC Δ (min ON - max OFF)", fontsize=4.7)
        if pair_col == 0:
            ax_hist.set_ylabel("% of trials", fontsize=4.6)

        c_all = Counter(sim["categories"])
        pass_cnt = c_all.pop("Pass", 0)
        top_fails = c_all.most_common(3)
        used_fail = sum(v for _, v in top_fails)
        other_cnt = sum(c_all.values()) - used_fail
        items = [("Pass", pass_cnt)] + top_fails + ([("Other fails", other_cnt)] if other_cnt > 0 else [])
        items = [(k, v) for k, v in items if v > 0][::-1]

        lbls = [c[0] for c in items]
        pcts = [100.0 * c[1] / n_mc for c in items]
        cols = [COL_ON if l == "Pass" else (COL_FAIL_ALT if l == "Other fails" else COL_FAIL) for l in lbls]
        bars = ax_fail.barh(range(len(lbls)), pcts, color=cols, alpha=0.85, height=0.65)
        ax_fail.set_yticks(range(len(lbls)))
        ax_fail.set_yticklabels(lbls, fontsize=4.0)
        ax_fail.set_xlim(0, 120)
        ax_fail.set_xticks([0, 25, 50, 75, 100])
        ax_fail.spines["top"].set_visible(False)
        ax_fail.spines["right"].set_visible(False)
        ax_fail.tick_params(labelsize=4.0, width=0.5, length=1.6, pad=0.8)
        if row_i == n_rows - 1 or idx == len(sims) - 2:
            ax_fail.set_xlabel("MC trials (%)", fontsize=4.7)
        for b, p in zip(bars, pcts):
            txt = f"{p:.0f}%" if p >= 1.0 else "<1%"
            ax_fail.text(p + 2.0, b.get_y() + b.get_height() / 2, txt, va="center", fontsize=4.0)

    ax_note = fig.add_subplot(outer_grid[8, 1])
    ax_note.axis("off")
    ax_note.add_patch(FancyBboxPatch((0.02, 0.08), 0.96, 0.84, boxstyle="round,pad=0.02,rounding_size=0.04",
                                     facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.7))
    ax_note.text(0.06, 0.78, "Monte Carlo Audit Summary", fontsize=5.5, fontweight="bold", color="#0F172A")
    ax_note.text(0.06, 0.44,
                 f"• n = {n_mc} trials per circuit (σ_pKd = {sigma_kd:.3f}, conc. std = {100*conc_std:.1f}%)\n"
                 "• Uniform histogram y-axis (% of trials) across all 17 circuits\n"
                 "• Config pass thresholds (Δ = 0.85·min_Δ, F = 0.85·min_F) shown\n"
                 "• Outcome bars sum to 100% (Pass + top-3 failure modes + Other)",
                 fontsize=4.5, color="#334155", va="center", linespacing=1.35)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), transparent=True)
    fig.savefig(out_prefix.with_suffix(".svg"), transparent=True)
    fig.savefig(out_prefix.with_suffix(".png"), facecolor="white", dpi=300)
    plt.close(fig)
    print(f"Saved Supplement MC figure to {out_prefix}.{{pdf,svg,png}}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build publication Figure 2 & Supplement MC figure.")
    parser.add_argument("--designs-dir", type=Path, default=ROOT / "results" / "graph_designs")
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "figures")
    parser.add_argument("--unit", choices=["uM", "nM"], default="uM")
    parser.add_argument("--nand-mode", choices=["bound5", "free6"], default="bound5")
    parser.add_argument("--layout", choices=["scaled16", "strict16"], default="scaled16")
    parser.add_argument("--n-mc", type=int, default=400)
    parser.add_argument("--conc-std", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bundles: Dict[str, dict] = {}
    for jf in sorted(args.designs_dir.glob("*/*_full.json")):
        with open(jf, "r") as f:
            b = json.load(f)
        if b["config"].get("mode") == "truth_table":
            bundles[b["config"]["name"]] = b

    sigma_kd = _load_sigma(args.calib)
    build_figure2(
        bundles,
        args.out_dir / "Figure2_gates",
        unit=args.unit,
        nand_mode=args.nand_mode,
        layout=args.layout,
        n_mc=args.n_mc,
        sigma_kd=sigma_kd,
        conc_std=args.conc_std,
        seed=args.seed,
    )
    build_supplement_mc(
        bundles,
        args.out_dir / "Supp_MC_gates",
        n_mc=args.n_mc,
        sigma_kd=sigma_kd,
        conc_std=args.conc_std,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
