"""Generate Figure 2 (workflow + 15 unified free_frac(S) gates in 3x5) and Supplement MC figures."""

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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.circuit_eq import solve_general_equilibrium
from src.consensus import DEFAULT_CALIB_PATH
from src.graph_design import _parse_config


def set_params():
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
    mpl.rcParams["legend.fontsize"] = 6


COL_ON = "#0F766E"
COL_OFF = "#94A3B8"
COL_FAIL = "#DC2626"
COL_FAIL_ALT = "#F59E0B"
COL_THRESH = "#334155"
COL_IN1 = "#1D4ED8"
COL_IN2 = "#7C3AED"
COL_IN3 = "#D97706"
COL_MED = "#E2E8F0"
COL_REP = "#FEF3C7"
COL_REP_EDGE = "#B45309"
COL_DARK = "#0F172A"
COL_BLUE = "#1D4ED8"
COL_AMBER = "#D97706"


def _load_sigma(calib_path: Path) -> float:
    with open(calib_path, "r") as f:
        return float(json.load(f).get("loo_rmse", 0.643))


def _load_gate_bundles(designs_dir: Path) -> Dict[str, Tuple[dict, bool]]:
    bundles: Dict[str, Tuple[dict, bool]] = {}
    for cdir in sorted(p for p in designs_dir.iterdir() if p.is_dir()):
        nm = cdir.name
        adam_p = cdir / f"{nm}_adam_full.json"
        full_p = cdir / f"{nm}_full.json"
        src_p = adam_p if adam_p.exists() else full_p
        if not src_p.exists():
            continue
        with open(src_p, "r") as f:
            b = json.load(f)
        if b["config"].get("mode") == "truth_table":
            bundles[b["config"]["name"]] = (b, src_p == adam_p)
    return bundles


def _pick_best_design(bundle: dict, is_adam: bool = False) -> dict:
    designs = bundle["designs"]
    if is_adam:
        for d in designs:
            if d["design_id"] == "D01":
                return d
    valid = [d for d in designs if d["metrics"].get("min_slip_margin", 0.0) >= 0.0]
    pool = valid if valid else designs
    return max(pool, key=lambda d: (d["metrics"].get("MC_Pass_rate", 0.0), -d["metrics"].get("total_loss", 0.0)))


def _run_mc_truth_table(
    parsed: dict,
    des: dict,
    sigma_kd: float,
    n_mc: int,
    conc_std: float,
    seed: int,
    uniform_criterion: Tuple[float, float] = (0.425, 3.825),
) -> dict:
    names = parsed["node_names"]
    n = len(names)
    c0_m_base = np.array(des["c0_uM"], dtype=float) * 1e-6
    base_log10_kd = -np.array(des["pkd_matrix"], dtype=float)
    kd_nom_m = 10.0 ** np.minimum(base_log10_kd, -2.0)

    obs_list = parsed["observables"]
    obs_objs = [o["obj"] for o in obs_list]
    tgt_list = [np.asarray(o["targets"], dtype=int) for o in obs_list]
    min_deltas = [uniform_criterion[0] for _ in obs_objs]
    min_folds = [uniform_criterion[1] for _ in obs_objs]

    eq_nom = solve_general_equilibrium(c0_m_base, kd_nom_m)
    nom_uM = []
    for obj in obs_objs:
        f_nom = obj.evaluate(eq_nom, c0_m_base, n)
        ref_idx = obj.ref_nodes[0]
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

        c0_run = (
            c0_m_base * np.maximum(1.0 + rng.normal(0.0, conc_std, size=c0_m_base.shape), 0.1)
            if conc_std > 0.0
            else c0_m_base
        )

        eq = solve_general_equilibrium(c0_run, kd_m)
        row_fail_reason = None
        for oi, obj in enumerate(obs_objs):
            vals = obj.evaluate(eq, c0_run, n)
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
    "not_4node": {"I": (0.10, 0.50), "M1": (0.37, 0.50), "Q1": (0.64, 0.50), "S": (0.90, 0.50)},
    "or_4node": {"I1": (0.15, 0.78), "I2": (0.15, 0.22), "Q": (0.52, 0.50), "S": (0.87, 0.50)},
    "nor_5node": {"I1": (0.11, 0.78), "I2": (0.11, 0.22), "M5": (0.38, 0.50), "Q1": (0.65, 0.50), "S": (0.90, 0.50)},
    "and_5node": {"I1": (0.13, 0.78), "I2": (0.13, 0.22), "Q4": (0.50, 0.78), "Q3": (0.50, 0.22), "S": (0.88, 0.50)},
    "nand_6node": {"I1": (0.11, 0.78), "I2": (0.11, 0.22), "M1": (0.38, 0.78), "M2": (0.38, 0.22), "Q1": (0.65, 0.50), "S": (0.90, 0.50)},
    "nimply_4node_sink": {"I1": (0.15, 0.78), "I2": (0.15, 0.22), "Q": (0.53, 0.78), "S": (0.87, 0.48)},
    "nimply_6node": {"I1": (0.11, 0.78), "Q4": (0.48, 0.78), "S": (0.89, 0.50), "I2": (0.11, 0.22), "M2": (0.38, 0.22), "Q1": (0.64, 0.22)},
    "imply_5node": {"I1": (0.12, 0.78), "I2": (0.12, 0.22), "M3": (0.38, 0.78), "Q3": (0.62, 0.44), "S": (0.88, 0.44)},
    "xor_4node": {"A": (0.16, 0.78), "B": (0.16, 0.22), "Q": (0.54, 0.50), "S": (0.88, 0.50)},
    "xnor_5node": {"A": (0.12, 0.80), "B": (0.12, 0.20), "M": (0.40, 0.50), "Q": (0.66, 0.50), "S": (0.90, 0.50)},
    "or3_5node": {"I1": (0.14, 0.84), "I2": (0.14, 0.50), "I3": (0.14, 0.16), "Q": (0.54, 0.50), "S": (0.88, 0.50)},
}


def _draw_mini_topology(ax, circuit_name: str, parsed: dict, des: dict, swap: bool = False) -> None:
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    coords = TOPO_COORDS.get(circuit_name)
    names = parsed["node_names"]
    if coords is None:
        return
    pkd_mat = np.array(des["pkd_matrix"], dtype=float)
    is_ab_inputs = circuit_name in ("xor_4node", "xnor_5node")

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
        lw = np.clip(0.55 + 0.17 * (pkd - 5.0), 0.6, 1.7)
        ax.plot([x1, x2], [y1, y2], color="#475569", lw=lw, zorder=1, solid_capstyle="round")

    r_base = 0.115 if len(names) <= 3 else 0.102
    for nm in names:
        x, y = coords[nm]
        lbl = disp_label[nm]
        is_rep = nm.startswith("S")
        is_in = lbl in ("I1", "I2", "I3")
        if is_in:
            fc = COL_IN1 if lbl == "I1" else (COL_IN2 if lbl == "I2" else COL_IN3)
            ec, tc = "white", "white"
        elif is_rep:
            fc, ec, tc = COL_REP, COL_REP_EDGE, "#78350F"
        else:
            fc, ec, tc = COL_MED, "#475569", "#1E293B"
        ax.add_patch(Circle((x, y), radius=r_base, facecolor=fc, edgecolor=ec, lw=0.75, zorder=3))
        fs = 3.8 if len(lbl) >= 2 else 4.2
        ax.text(x, y, lbl, ha="center", va="center", fontsize=fs, fontweight="bold", color=tc, zorder=4)


def _draw_circuit_pipeline(ax) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 16)
    ax.axis("off")

    steps = [
        ("1. Target Graph", "Interaction topology\n& affinity hierarchy", "#EFF6FF", COL_BLUE),
        ("2. Sequence Design", "Multi-start optimization\nwith consensus pKd", "#F0FDFA", COL_ON),
        ("3. Conc. Tuning", "Mass-action equilibrium\n+ robustness tuning", "#F8FAFC", "#475569"),
        ("4. Noise & 3D Check", "pKd + pipetting noise;\nBoltz-2 + Rosetta", "#FFFBEB", COL_AMBER),
    ]
    xs = [1.0, 26.2, 51.4, 76.6]
    w, h, y0 = 22.0, 13.0, 1.5
    for i, (title, body, fc, ec) in enumerate(steps):
        x0 = xs[i]
        ax.add_patch(FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.3,rounding_size=1.1",
            facecolor=fc, edgecolor=ec, linewidth=0.9, zorder=2,
        ))
        ax.text(x0 + w / 2, y0 + h - 3.4, title, ha="center", va="center",
                fontsize=5.8, fontweight="bold", color=ec, zorder=3)
        ax.text(x0 + w / 2, y0 + 4.3, body, ha="center", va="center",
                fontsize=4.8, color=COL_DARK, linespacing=1.30, zorder=3)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch(
                (x0 + w + 0.4, y0 + h / 2), (xs[i + 1] - 0.4, y0 + h / 2),
                arrowstyle="-|>,head_length=2.8,head_width=1.8",
                color="#475569", lw=0.85, zorder=3,
            ))


def build_figure2(
    bundles: Dict[str, Tuple[dict, bool]],
    out_prefix: Path,
    unit: str = "uM",
    n_mc: int = 400,
    sigma_kd: float = 0.643,
    conc_std: float = 0.015,
    uniform_thresh: Tuple[float, float] = (0.425, 3.825),
    seed: int = 42,
) -> None:
    set_params()
    mpl.rcParams["figure.figsize"] = (210 / 25.4, 178 / 25.4)

    scale = 1000.0 if unit == "nM" else 1.0
    unit_lbl = "nM" if unit == "nM" else "µM"
    y_max = 0.56 * scale

    panels = [
        ("YES (I1)", "yes_3node", False, "1in"),
        ("YES (I2)", "yes_3node", True, "1in"),
        ("NOT (I1)", "not_4node", False, "1in"),
        ("NOT (I2)", "not_4node", True, "1in"),
        ("OR", "or_4node", False, "normal"),
        ("NOR", "nor_5node", False, "normal"),
        ("AND", "and_5node", False, "normal"),
        ("NAND", "nand_6node", False, "normal"),
        ("NIMPLY (I1 & ¬I2)", "nimply_4node_sink", False, "normal"),
        ("NIMPLY (I2 & ¬I1)", "nimply_4node_sink", True, "normal"),
        ("IMPLY (¬I1 OR I2)", "imply_5node", False, "normal"),
        ("IMPLY (¬I2 OR I1)", "imply_5node", True, "normal"),
        ("XOR", "xor_4node", False, "normal"),
        ("XNOR", "xnor_5node", False, "normal"),
        ("3-input OR", "or3_5node", False, "or3"),
    ]

    sim_cache: Dict[str, Tuple[dict, dict, dict]] = {}
    for _, ckey, _, _ in panels:
        if ckey in sim_cache:
            continue
        bundle, is_adam = bundles[ckey]
        parsed = _parse_config(bundle["config"], allow_5mer=True)
        des = _pick_best_design(bundle, is_adam=is_adam)
        sim = _run_mc_truth_table(
            parsed, des, sigma_kd, n_mc, conc_std, seed,
            uniform_criterion=uniform_thresh,
        )
        sim_cache[ckey] = (parsed, des, sim)

    fig = plt.figure()
    top_pipe = fig.add_gridspec(1, 1, left=0.075, right=0.96, top=0.96, bottom=0.83)
    ax_pipe = fig.add_subplot(top_pipe[0])
    _draw_circuit_pipeline(ax_pipe)
    ax_pipe.text(-0.04, 1.03, "A", transform=ax_pipe.transAxes, fontsize=9.5, fontweight="bold", color=COL_DARK)

    fig.text(0.035, 0.796, "B", fontsize=9.5, fontweight="bold", color=COL_DARK)
    gate_grid = fig.add_gridspec(
        3, 5, left=0.08, right=0.96, top=0.790, bottom=0.085, wspace=0.18, hspace=0.34,
    )

    for idx, (title, ckey, swap, smode) in enumerate(panels):
        row_i, col_i = divmod(idx, 5)
        inner = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gate_grid[row_i, col_i],
            height_ratios=[0.36, 0.64], hspace=0.04,
        )
        ax_top = fig.add_subplot(inner[0])
        ax_bar = fig.add_subplot(inner[1])

        parsed, des, sim = sim_cache[ckey]
        names = parsed["node_names"]
        s0_uM = sim["ref_s0_uM"]
        panel_thresh = 0.5 * s0_uM * scale
        pass_pct = 100.0 * sim["pass_rate"]

        ax_top.set_xlim(0, 1)
        ax_top.set_ylim(0, 1)
        ax_top.axis("off")
        t_fs = 4.5 if len(title) > 13 else 5.2
        ax_top.text(0.0, 0.72, title, fontsize=t_fs, fontweight="bold", color=COL_DARK, va="center")
        ax_top.text(0.0, 0.25, f"MC: {pass_pct:.0f}%", fontsize=4.5, fontweight="bold",
                    color=(COL_ON if pass_pct >= 65 else "#B45309"), va="center")

        ax_topo = ax_top.inset_axes([0.45, 0.02, 0.55, 0.96])
        _draw_mini_topology(ax_topo, ckey, parsed, des, swap=swap)

        ax_bar.axhline(s0_uM * scale, color="#CBD5E1", ls=":", lw=0.65, zorder=1)
        ax_bar.axhline(panel_thresh, color=COL_THRESH, ls="--", lw=0.70, zorder=1)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.spines["left"].set_linewidth(0.6)
        ax_bar.spines["bottom"].set_linewidth(0.6)
        ax_bar.tick_params(width=0.6, length=2.0, pad=1.2)

        if smode == "or3":
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
                            fmt="o", ms=1.2, color=COL_DARK, elinewidth=0.5, capsize=0.9, capthick=0.45, zorder=3)
            ax_bar.set_xticks(xs)
            ax_bar.set_xticklabels(["000", "001", "010", "011", "100", "101", "110", "111"],
                                   fontsize=3.5, rotation=45, ha="right")
        else:
            if smode == "1in":
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
                            fmt="o", ms=1.4, color=COL_DARK, elinewidth=0.55, capsize=1.2, capthick=0.45, zorder=3)
            ax_bar.set_xticks(xs)
            ax_bar.set_xticklabels(["00", "01", "10", "11"], fontsize=4.8)

        ax_bar.set_ylim(0.0, y_max)
        yticks = [0, 0.25 * scale, 0.50 * scale]
        ax_bar.set_yticks(yticks)
        if col_i == 0:
            ax_bar.set_yticklabels([f"{y:.2g}" for y in yticks], fontsize=5.0)
            ax_bar.set_ylabel(f"[S]free ({unit_lbl})", fontsize=5.2, labelpad=2.0)
        else:
            ax_bar.set_yticklabels([])

    leg_items = [
        FancyBboxPatch((0, 0), 1, 1, facecolor=COL_ON, edgecolor="none", label="ON (nom.)"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=COL_OFF, edgecolor="none", label="OFF (nom.)"),
        Line2D([0], [0], marker="o", color=COL_DARK, ms=2.0, lw=0.65, label="MC med (10–90%)"),
        Line2D([0], [0], color=COL_THRESH, ls="--", lw=0.75, label="0.5·[S0]"),
        Line2D([0], [0], color="#CBD5E1", ls=":", lw=0.75, label="[S0]"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_IN1, ms=4.0, label="I1"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_IN2, ms=4.0, label="I2"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_IN3, ms=4.0, label="I3"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_MED, markeredgecolor="#475569", ms=4.0, label="Mediator"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_REP, markeredgecolor=COL_REP_EDGE, ms=4.2, label="Reporter S"),
    ]
    fig.legend(handles=leg_items, loc="center", bbox_to_anchor=(0.52, 0.030),
               ncol=10, frameon=False, fontsize=4.6, handlelength=1.1, columnspacing=0.8)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), transparent=True)
    fig.savefig(out_prefix.with_suffix(".svg"), transparent=True)
    fig.savefig(out_prefix.with_suffix(".png"), facecolor="white", dpi=300)
    plt.close(fig)
    print(f"Saved Figure 2 to {out_prefix}.{{pdf,svg,png}}")


def build_supplement_mc(
    bundles: Dict[str, Tuple[dict, bool]],
    out_prefix: Path,
    n_mc: int = 400,
    sigma_kd: float = 0.643,
    conc_std: float = 0.015,
    uniform_thresh: Tuple[float, float] = (0.425, 3.825),
    seed: int = 42,
) -> Dict[str, dict]:
    set_params()
    mpl.rcParams["figure.figsize"] = (210 / 25.4, 240 / 25.4)

    supp_specs = [
        ("yes_3node", "YES (3-node)"),
        ("not_4node", "NOT (4-node)"),
        ("or_4node", "OR (4-node)"),
        ("nor_5node", "NOR (5-node)"),
        ("and_5node", "AND (5-node)"),
        ("nand_6node", "NAND (6-node)"),
        ("imply_5node", "IMPLY (5-node)"),
        ("nimply_4node_sink", "NIMPLY (4-node)"),
        ("xor_4node", "XOR (4-node)"),
        ("xnor_5node", "XNOR (5-node)"),
        ("or3_5node", "3-input OR (5-node)"),
    ]
    available = [(k, lbl) for k, lbl in supp_specs if k in bundles]
    n_rows = 6
    fig = plt.figure()
    outer_grid = fig.add_gridspec(
        n_rows, 2, left=0.09, right=0.95, top=0.96, bottom=0.06, wspace=0.26, hspace=0.50,
    )

    sims = []
    sim_by_key: Dict[str, dict] = {}
    max_bin_pct = 0.0
    bins = np.linspace(-0.25, 1.0, 21)
    for ckey, label in available:
        bundle, is_adam = bundles[ckey]
        parsed = _parse_config(bundle["config"], allow_5mer=True)
        des = _pick_best_design(bundle, is_adam=is_adam)
        sim = _run_mc_truth_table(
            parsed, des, sigma_kd, n_mc, conc_std, seed,
            uniform_criterion=uniform_thresh,
        )
        h_all, _ = np.histogram(sim["deltas"], bins=bins)
        max_bin_pct = max(max_bin_pct, float(np.max(100.0 * h_all / n_mc)))
        sims.append((label, sim))
        sim_by_key[ckey] = sim

    y_hist_max = min(100.0, np.ceil(max_bin_pct / 10.0) * 10.0 + 5.0)

    for idx, (label, sim) in enumerate(sims):
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

        ax_hist.hist(pass_d, bins=bins, weights=np.full(pass_d.size, 100.0 / n_mc), color=COL_ON, alpha=0.80)
        ax_hist.hist(fail_d, bins=bins, weights=np.full(fail_d.size, 100.0 / n_mc), color=COL_FAIL, alpha=0.70)
        ax_hist.axvline(thresh, color=COL_THRESH, ls="--", lw=0.8)
        ax_hist.set_title(label, fontsize=5.3, fontweight="bold", loc="left", pad=2.2)
        ax_hist.set_xlim(-0.25, 1.02)
        ax_hist.set_ylim(0.0, y_hist_max)
        ax_hist.set_xticks([-0.25, 0.0, 0.25, 0.50, 0.75, 1.00])
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)
        ax_hist.tick_params(labelsize=4.2, width=0.5, length=1.6, pad=0.8)
        if row_i == n_rows - 1 or idx == len(sims) - 2:
            ax_hist.set_xlabel("MC Δ (min ON − max OFF)", fontsize=4.8)
        if pair_col == 0:
            ax_hist.set_ylabel("% of trials", fontsize=4.8)

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
        ax_fail.set_yticklabels(lbls, fontsize=4.2)
        ax_fail.set_xlim(0, 120)
        ax_fail.set_xticks([0, 25, 50, 75, 100])
        ax_fail.spines["top"].set_visible(False)
        ax_fail.spines["right"].set_visible(False)
        ax_fail.tick_params(labelsize=4.2, width=0.5, length=1.6, pad=0.8)
        if row_i == n_rows - 1 or idx == len(sims) - 2:
            ax_fail.set_xlabel("MC trials (%)", fontsize=4.8)
        for b, p in zip(bars, pcts):
            txt = f"{p:.0f}%" if p >= 1.0 else "<1%"
            ax_fail.text(p + 2.0, b.get_y() + b.get_height() / 2, txt, va="center", fontsize=4.2)

    # Clean minimal legend in empty 12th cell (details belong in manuscript figure caption)
    ax_leg = fig.add_subplot(outer_grid[5, 1])
    ax_leg.axis("off")
    supp_leg = [
        FancyBboxPatch((0, 0), 1, 1, facecolor=COL_ON, alpha=0.80, edgecolor="none", label="Pass"),
        FancyBboxPatch((0, 0), 1, 1, facecolor=COL_FAIL, alpha=0.70, edgecolor="none", label="Fail"),
        Line2D([0], [0], color=COL_THRESH, ls="--", lw=0.9, label=f"Threshold (Δ ≥ {uniform_thresh[0]:.3f})"),
    ]
    ax_leg.legend(handles=supp_leg, loc="center", frameon=False, fontsize=5.2, handlelength=1.4)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), transparent=True)
    fig.savefig(out_prefix.with_suffix(".svg"), transparent=True)
    fig.savefig(out_prefix.with_suffix(".png"), facecolor="white", dpi=300)
    plt.close(fig)
    print(f"Saved Supplement MC figure to {out_prefix}.{{pdf,svg,png}}")
    return sim_by_key


def build_supplement_node_cmp(
    bundles: Dict[str, Tuple[dict, bool]],
    sim_by_key: Dict[str, dict],
    out_prefix: Path,
    n_mc: int = 400,
    sigma_kd: float = 0.643,
    conc_std: float = 0.015,
    uniform_thresh: Tuple[float, float] = (0.425, 3.825),
    seed: int = 42,
) -> None:
    """Supplement figure: Compact I1:I2-sink vs strict 6-7-node Fig. S4 (both free_frac(S))."""
    set_params()
    mpl.rcParams["figure.figsize"] = (118 / 25.4, 84 / 25.4)

    fig = plt.figure()
    ax_b = fig.add_axes([0.14, 0.19, 0.81, 0.69])

    cmp_pairs = [
        ("NIMPLY", "nimply_4node_sink", "nimply_6node", "4 vs 6"),
        ("XOR", "xor_4node", "xor_7node", "4 vs 7"),
        ("XNOR", "xnor_5node", "xnor_7node", "5 vs 7"),
    ]
    c_rates, s4_rates, x_lbls = [], [], []
    for gname, kc, ks4, nstr in cmp_pairs:
        for k, out_list in [(kc, c_rates), (ks4, s4_rates)]:
            if k in sim_by_key:
                res = sim_by_key[k]
            else:
                b, is_adam = bundles[k]
                pr = _parse_config(b["config"], allow_5mer=True)
                ds = _pick_best_design(b, is_adam=is_adam)
                res = _run_mc_truth_table(pr, ds, sigma_kd, n_mc, conc_std, seed, uniform_criterion=uniform_thresh)
            out_list.append(100.0 * res["pass_rate"])
        x_lbls.append(f"{gname}\n({nstr} nodes)")

    xb = np.arange(len(cmp_pairs))
    wb = 0.34
    bc = ax_b.bar(xb - wb / 2, c_rates, width=wb, color=COL_ON, alpha=0.90, label="Compact (I1:I2 sink)")
    bs = ax_b.bar(xb + wb / 2, s4_rates, width=wb, color=COL_OFF, alpha=0.90, label="Strict Fig. S4 (no I1:I2)")
    for b, v in list(zip(bc, c_rates)) + list(zip(bs, s4_rates)):
        ax_b.text(b.get_x() + b.get_width() / 2, v + 1.8, f"{v:.0f}%", ha="center", va="bottom", fontsize=5.2)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(x_lbls, fontsize=5.4)
    ax_b.set_ylim(0, 114)
    ax_b.set_ylabel("MC Pass Rate (%)", fontsize=6.0)
    ax_b.set_title("Compact I1:I2-Sink vs 6–7-Node Fig. S4", fontsize=6.4, fontweight="bold", pad=5.0)
    ax_b.legend(loc="upper right", frameon=False, fontsize=5.2)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.tick_params(labelsize=5.4, width=0.6, length=2.0)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), transparent=True)
    fig.savefig(out_prefix.with_suffix(".svg"), transparent=True)
    fig.savefig(out_prefix.with_suffix(".png"), facecolor="white", dpi=300)
    plt.close(fig)
    print(f"Saved Supplement comparison figure to {out_prefix}.{{pdf,svg,png}}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build publication Figure 2 & Supplement MC figures.")
    parser.add_argument("--designs-dir", type=Path, default=ROOT / "results" / "graph_designs")
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "figures")
    parser.add_argument("--unit", choices=["uM", "nM"], default="uM")
    parser.add_argument("--n-mc", type=int, default=400)
    parser.add_argument("--conc-std", type=float, default=0.015)
    parser.add_argument("--thresh-delta", type=float, default=0.425)
    parser.add_argument("--thresh-fold", type=float, default=3.825)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bundles = _load_gate_bundles(args.designs_dir)
    sigma_kd = _load_sigma(args.calib)
    uni = (args.thresh_delta, args.thresh_fold)

    build_figure2(
        bundles,
        args.out_dir / "Figure2_gates",
        unit=args.unit,
        n_mc=args.n_mc,
        sigma_kd=sigma_kd,
        conc_std=args.conc_std,
        uniform_thresh=uni,
        seed=args.seed,
    )
    sim_by_key = build_supplement_mc(
        bundles,
        args.out_dir / "Supp_MC_gates",
        n_mc=args.n_mc,
        sigma_kd=sigma_kd,
        conc_std=args.conc_std,
        uniform_thresh=uni,
        seed=args.seed,
    )
    build_supplement_node_cmp(
        bundles,
        sim_by_key,
        args.out_dir / "Supp_Compact_vs_FigS4",
        n_mc=args.n_mc,
        sigma_kd=sigma_kd,
        conc_std=args.conc_std,
        uniform_thresh=uni,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
