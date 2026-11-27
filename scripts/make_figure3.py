"""Generate Figure 3: 2D decision boundaries + Banana (3 bands) with gate-style topologies."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from design_2d_boundary import curves, evaluate_full_2d
from design_banana import (
    GEOMETRY,
    arc_samples,
    eval_readouts,
    parse_readouts,
    solve_eq_fast_np,
)


def set_params():
    mpl.rcParams["figure.figsize"] = (210 / 25.4, 152 / 25.4)
    mpl.rcParams["figure.facecolor"] = "none"
    mpl.rcParams["axes.facecolor"] = "none"
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["axes.labelsize"] = 5.5
    mpl.rcParams["xtick.labelsize"] = 5.0
    mpl.rcParams["ytick.labelsize"] = 5.0
    mpl.rcParams["legend.fontsize"] = 5.0


COL_IN1 = "#1D4ED8"
COL_IN2 = "#7C3AED"
COL_MED = "#E2E8F0"
COL_REP = "#FEF3C7"
COL_REP_EDGE = "#B45309"
COL_BAN_A = "#DBEAFE"
COL_BAN_A_EDGE = "#1D4ED8"
COL_DARK = "#0F172A"
COL_TARGET = "#15803D"
COL_BAN_BLUE = "#DCEAF7"
COL_BAN_PEACH = "#FBE6D4"


TOPO_COORDS_2D: Dict[str, Dict[str, Tuple[float, float]]] = {
    "line": {"X": (0.12, 0.78), "Y": (0.12, 0.22), "A": (0.86, 0.78), "B": (0.86, 0.22)},
    "circle": {
        "X": (0.12, 0.72), "Y": (0.12, 0.20),
        "D": (0.42, 0.88), "C": (0.50, 0.50), "B": (0.50, 0.16),
        "A": (0.88, 0.54),
    },
    "quadratic": {
        "X": (0.12, 0.78), "Y": (0.12, 0.22),
        "B": (0.48, 0.84), "C": (0.52, 0.44), "A": (0.88, 0.44),
    },
    "cubic": {
        "X": (0.12, 0.78), "Y": (0.12, 0.22),
        "C": (0.36, 0.52), "D": (0.60, 0.48), "B": (0.66, 0.16),
        "A": (0.86, 0.74),
    },
    "angle": {
        "X": (0.12, 0.78), "Y": (0.12, 0.22),
        "C": (0.48, 0.78), "E": (0.48, 0.22), "A": (0.86, 0.50),
    },
    "bandpass_2": {
        "X": (0.12, 0.78), "Y": (0.12, 0.20),
        "B": (0.48, 0.50), "C": (0.48, 0.16), "A": (0.86, 0.72),
    },
    "full_circle": {
        "X": (0.12, 0.78), "Y": (0.12, 0.22),
        "B": (0.48, 0.86), "D": (0.48, 0.14), "C": (0.52, 0.50),
        "A": (0.88, 0.50),
    },
    "monomer_3": {
        "X": (0.12, 0.72), "Y": (0.12, 0.32),
        "D": (0.42, 0.88), "H": (0.28, 0.12), "C": (0.50, 0.54), "E": (0.64, 0.14),
        "A": (0.88, 0.76), "B": (0.88, 0.40),
    },
}


def _fmt_kd(kd_nm: float) -> str:
    if kd_nm >= 100.0:
        return f"{kd_nm:.0f}"
    if kd_nm >= 10.0:
        return f"{kd_nm:.1f}"
    return f"{kd_nm:.1f}"


def _draw_2d_topology(
    ax,
    key: str,
    nodes: Sequence[str],
    edges_pairs: Sequence[str],
    pkd_mat: np.ndarray,
    kd_mat_nM: np.ndarray,
    is_banana: bool = False,
    show_kd: bool = True,
) -> None:
    ax.set_xlim(-0.16, 1.16)
    ax.set_ylim(-0.16, 1.16)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    coords = TOPO_COORDS_2D.get(key)
    if coords is None or any(nm not in coords for nm in nodes):
        angs = np.linspace(0.0, 2.0 * np.pi, len(nodes), endpoint=False)
        coords = {nm: (0.5 + 0.38 * float(np.cos(a)), 0.5 + 0.38 * float(np.sin(a))) for nm, a in zip(nodes, angs)}

    cx = float(np.mean([coords[nm][0] for nm in nodes]))
    cy = float(np.mean([coords[nm][1] for nm in nodes]))

    for pair in edges_pairs:
        u_nm, v_nm = pair[0], pair[1]
        u, v = nodes.index(u_nm), nodes.index(v_nm)
        x1, y1 = coords[u_nm]
        x2, y2 = coords[v_nm]
        pkd = float(pkd_mat[u, v])
        kd = float(kd_mat_nM[u, v])
        lw = float(np.clip(0.75 + 0.16 * (pkd - 5.0), 0.75, 1.70))
        ax.plot([x1, x2], [y1, y2], color="#475569", lw=lw, zorder=1, solid_capstyle="round")
        if show_kd:
            xm, ym = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            dx, dy = x2 - x1, y2 - y1
            elen = float(np.hypot(dx, dy))
            if elen > 1e-6:
                nx, ny = -dy / elen, dx / elen
                if nx * (xm - cx) + ny * (ym - cy) < 0.0:
                    nx, ny = -nx, -ny
                tx, ty = xm + 0.072 * nx, ym + 0.072 * ny
            else:
                tx, ty = xm, ym
            ax.text(
                tx, ty, _fmt_kd(kd), fontsize=2.7, ha="center", va="center", color="#334155",
                bbox=dict(boxstyle="round,pad=0.03", facecolor="white", edgecolor="none", alpha=0.80),
                zorder=2,
            )

    r_node = 0.076 if len(nodes) >= 7 else 0.086
    for nm in nodes:
        x, y = coords[nm]
        if nm == "X":
            fc, ec, tc = COL_IN1, "white", "white"
        elif nm == "Y":
            fc, ec, tc = COL_IN2, "white", "white"
        elif is_banana and nm == "A":
            fc, ec, tc = COL_BAN_A, COL_BAN_A_EDGE, "#1E3A8A"
        elif nm == "B" and is_banana:
            fc, ec, tc = COL_REP, COL_REP_EDGE, "#78350F"
        elif nm == "A" and not is_banana:
            fc, ec, tc = COL_REP, COL_REP_EDGE, "#78350F"
        else:
            fc, ec, tc = COL_MED, "#475569", "#1E293B"
        ax.add_patch(Circle((x, y), radius=r_node, facecolor=fc, edgecolor=ec, lw=0.75, zorder=3))
        ax.text(x, y, nm, ha="center", va="center", fontsize=3.9, fontweight="bold", color=tc, zorder=4)


def _axis_ticks(max_in: float) -> List[float]:
    if max_in <= 250.0:
        return [0.0, 100.0, 200.0]
    if max_in <= 1100.0:
        return [0.0, 500.0, 1000.0]
    if max_in <= 1700.0:
        return [0.0, 800.0, 1600.0]
    if max_in <= 2200.0:
        return [0.0, 1000.0, 2000.0]
    return [0.0, 1500.0, 3000.0]


def _load_2d_system(report_path: Path) -> dict:
    rep = json.loads(report_path.read_text())
    name = rep["boundary"]
    s = float(rep["conc_scale"])
    nodes = rep["nodes"]
    edges = rep["edges"]
    pkd_mat = np.array(rep["pkd_matrix"], dtype=float)
    kd_mat_nM = 10.0 ** (9.0 - pkd_mat)
    tots_nM = rep["fitted_total_nM"]
    metrics, values, val_grid = evaluate_full_2d(name, kd_mat_nM, tots_nM, nodes, s, use_jax=True)
    return {
        "kind": "2d",
        "key": name,
        "nodes": nodes,
        "edges": edges,
        "pkd_mat": pkd_mat,
        "kd_mat_nM": kd_mat_nM,
        "conc_scale": s,
        "max_in": 200.0 * s,
        "thresh": float(rep["threshold_nM"]),
        "metrics": metrics,
        "values": values,
        "val_grid": val_grid,
    }


def _load_banana_system(report_path: Path) -> dict:
    rep = json.loads(report_path.read_text())
    s = float(rep["conc_scale"])
    max_in = 200.0 * s
    nodes = rep["nodes"]
    edges = list(rep["edges"].keys()) if isinstance(rep["edges"], dict) else list(rep["edges"])
    readouts = rep["outputs"]
    pkd_mat = np.array(rep["pkd_matrix"], dtype=float)
    kd_real_nM = 10.0 ** (9.0 - pkd_mat)
    tots_scaled = rep["fitted_total_scaled_nM"]
    internal = list(nodes[2:])

    grid = np.linspace(0.0, max_in, 161)
    gx, gy = np.meshgrid(grid, grid)
    xy = np.column_stack((gx.ravel(), gy.ravel()))
    c0 = np.column_stack((xy, np.broadcast_to([tots_scaled[k] for k in internal], (xy.shape[0], len(internal)))))
    free = solve_eq_fast_np(c0, kd_real_nM)
    vals = eval_readouts(free, parse_readouts(nodes, kd_real_nM, readouts))
    delta = (vals[:, 1] - vals[:, 0]).reshape(gx.shape)

    acc = float(rep["stages"]["optimized"]["uniform_test"]["accuracy"])
    return {
        "kind": "banana",
        "key": rep["topology"],
        "nodes": nodes,
        "edges": edges,
        "pkd_mat": pkd_mat,
        "kd_mat_nM": kd_real_nM,
        "conc_scale": s,
        "max_in": max_in,
        "bands": 3,
        "acc": acc,
        "delta": delta,
        "val_grid": (gx, gy),
    }


def build_figure3(
    systems: List[Tuple[str, dict]],
    out_prefix: Path,
    show_kd: bool = True,
    legend_in_empty_cell: bool = False,
) -> None:
    set_params()
    fig = plt.figure()
    n_cols = 4
    n_rows = 2
    bottom_m = 0.065 if legend_in_empty_cell else 0.105
    outer = fig.add_gridspec(
        n_rows, n_cols, left=0.06, right=0.985, top=0.96, bottom=bottom_m,
        wspace=0.24, hspace=0.33,
    )

    for idx, (title, sys_data) in enumerate(systems):
        r_i, c_i = divmod(idx, n_cols)
        inner = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[r_i, c_i],
            height_ratios=[0.36, 0.64], hspace=0.07,
        )
        ax_top = fig.add_subplot(inner[0])
        ax_map = fig.add_subplot(inner[1])

        ax_top.set_xlim(0, 1)
        ax_top.set_ylim(0, 1)
        ax_top.axis("off")

        is_banana = (sys_data["kind"] == "banana")
        if is_banana:
            sub_txt = f"Grid Acc: {sys_data['acc']:.1%}\n3 bands (s={sys_data['conc_scale']:g})"
        else:
            m = sys_data["metrics"]
            err_pct = m.get("contour_error_pct_of_domain")
            err_str = f"{err_pct:.2f}%" if err_pct is not None else "n/a"
            sub_txt = f"Acc: {m['accuracy']:.1%}\nErr: {err_str}"

        ax_top.text(0.0, 0.78, title, fontsize=6.0, fontweight="bold", color=COL_DARK, va="center")
        ax_top.text(0.0, 0.34, sub_txt, fontsize=4.6, color="#334155", va="center", linespacing=1.35)

        ax_topo = ax_top.inset_axes([0.33, -0.06, 0.70, 1.15])
        _draw_2d_topology(
            ax_topo,
            sys_data["key"],
            sys_data["nodes"],
            sys_data["edges"],
            sys_data["pkd_mat"],
            sys_data["kd_mat_nM"],
            is_banana=is_banana,
            show_kd=show_kd,
        )

        max_in = sys_data["max_in"]
        gx, gy = sys_data["val_grid"]
        if is_banana:
            delta = sys_data["delta"]
            ax_map.contourf(gx, gy, delta, levels=[-np.inf, 0.0, np.inf], colors=(COL_BAN_BLUE, COL_BAN_PEACH))
            if delta.min() < 0.0 < delta.max():
                ax_map.contour(gx, gy, delta, levels=[0.0], colors="#191919", linewidths=1.0)
            for arc in arc_samples(GEOMETRY, sys_data["bands"], 501):
                ax_map.plot(arc[:, 0] * sys_data["conc_scale"], arc[:, 1] * sys_data["conc_scale"],
                            "--", color=COL_TARGET, lw=0.95)
        else:
            vals = sys_data["values"]
            thresh = sys_data["thresh"]
            vmin = min(float(np.min(vals)), thresh - 1.0)
            vmax = max(float(np.max(vals)), thresh + 1.0)
            norm = TwoSlopeNorm(vcenter=thresh, vmin=vmin, vmax=vmax)
            ax_map.pcolormesh(gx, gy, vals, shading="nearest", cmap="RdBu_r", norm=norm, rasterized=True)
            if vals.min() < thresh < vals.max():
                ax_map.contour(gx, gy, vals, levels=[thresh], colors="#191919", linewidths=1.0)
            for branch in curves(sys_data["key"], max_in):
                ax_map.plot(branch[:, 0], branch[:, 1], "--", color=COL_TARGET, lw=0.95)

        ticks = _axis_ticks(max_in)
        ax_map.set_xlim(0.0, max_in)
        ax_map.set_ylim(0.0, max_in)
        ax_map.set_xticks(ticks)
        ax_map.set_yticks(ticks)
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.set_xlabel("X (nM)", fontsize=5.0, labelpad=1.5)
        ax_map.set_ylabel("Y (nM)", fontsize=5.0, labelpad=1.5)
        ax_map.tick_params(labelsize=4.6, width=0.55, length=1.8, pad=1.2)
        for sp in ax_map.spines.values():
            sp.set_linewidth(0.6)

    leg_items = [
        FancyBboxPatch((0, 0), 1, 1, facecolor="#60A5FA", edgecolor="none", label="Low Free A / Class −1"),
        FancyBboxPatch((0, 0), 1, 1, facecolor="#F87171", edgecolor="none", label="High Free A / Class +1"),
        Line2D([0], [0], color="#191919", lw=1.05, label="Decision boundary"),
        Line2D([0], [0], color=COL_TARGET, ls="--", lw=1.0, label="Target"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_IN1, ms=4.2, label="X"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_IN2, ms=4.2, label="Y"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_MED, markeredgecolor="#475569", ms=4.2, label="Mediator"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_REP, markeredgecolor=COL_REP_EDGE, ms=4.2, label="Out A (2D) / B (+1)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_BAN_A, markeredgecolor=COL_BAN_A_EDGE, ms=4.2, label="Out A (−1, banana)"),
    ]
    if show_kd:
        leg_items.append(Line2D([0], [0], color="none", label="Edge labels: Kd (nM)"))

    if legend_in_empty_cell and len(systems) < n_rows * n_cols:
        ax_leg = fig.add_subplot(outer[1, 3])
        ax_leg.axis("off")
        ax_leg.legend(handles=leg_items, loc="center", frameon=False, fontsize=5.0, handlelength=1.3, labelspacing=0.65)
    else:
        fig.legend(
            handles=leg_items, loc="center", bbox_to_anchor=(0.52, 0.036),
            ncol=5, frameon=False, fontsize=4.8, handlelength=1.2, columnspacing=1.0,
        )

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), transparent=True)
    fig.savefig(out_prefix.with_suffix(".svg"), transparent=True)
    fig.savefig(out_prefix.with_suffix(".png"), facecolor="white", dpi=300)
    plt.close(fig)
    print(f"Saved {out_prefix}.{{pdf,svg,png}}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Figure 3 (2D boundaries + Banana) in two variants.")
    parser.add_argument("--des-dir", type=Path, default=ROOT / "results" / "des_bounds_designs")
    parser.add_argument("--ban-dir", type=Path, default=ROOT / "results" / "banana_designs")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "figures")
    parser.add_argument("--cubic-5mer", action="store_true", help="Use 5mer cubic instead of uniform 4mer cubic.")
    parser.add_argument("--no-kd", action="store_true", help="Hide Kd numbers on topology edges.")
    args = parser.parse_args()

    cubic_folder = "cubic_yd_5mer_cfit_s15_kmin0.5" if args.cubic_5mer else "cubic_yd_4mer_cfit_s15_kmin0.5"

    sys_line = ("Line", _load_2d_system(args.des_dir / "line_s1" / "line_report.json"))
    sys_circle = ("Circle", _load_2d_system(args.des_dir / "circle_cfit_s5_kmin2" / "circle_report.json"))
    sys_quad = ("Quadratic", _load_2d_system(args.des_dir / "quadratic_cfit_s10_seed" / "quadratic_report.json"))
    sys_cubic = ("Cubic", _load_2d_system(args.des_dir / cubic_folder / "cubic_report.json"))
    sys_angle = ("Angle", _load_2d_system(args.des_dir / "angle_cfit_s15_seed" / "angle_report.json"))
    sys_band = ("Bandpass", _load_2d_system(args.des_dir / "bandpass_2_cfit_s15_seed" / "bandpass_2_report.json"))
    sys_fc = ("Full circle", _load_2d_system(args.des_dir / "full_circle_cfit_s15_seed" / "full_circle_report.json"))
    sys_ban = ("Banana (3 bands)", _load_banana_system(args.ban_dir / "monomer_3_5mer_s8" / "monomer_3_report.json"))

    show_kd = not args.no_kd
    with_cubic = [sys_line, sys_circle, sys_quad, sys_cubic, sys_angle, sys_band, sys_fc, sys_ban]
    no_cubic = [sys_line, sys_circle, sys_quad, sys_angle, sys_band, sys_fc, sys_ban]

    build_figure3(
        with_cubic,
        args.out_dir / "Figure3_boundaries_banana_with_cubic",
        show_kd=show_kd,
        legend_in_empty_cell=False,
    )
    build_figure3(
        no_cubic,
        args.out_dir / "Figure3_boundaries_banana_no_cubic",
        show_kd=show_kd,
        legend_in_empty_cell=True,
    )


if __name__ == "__main__":
    main()
