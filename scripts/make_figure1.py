"""Generate Figure 1: Design pipeline, 1D consensus calibration, cross-dataset & independent validation."""

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.anchors import (
    ARONSSON_PAIRS,
    ARONSSON_PEPTIDES,
    CROOKS_PAIRS,
    CROOKS_PEPTIDES,
    THOMAS_PAIRS,
    THOMAS_PEPTIDES,
)
from src.consensus import CONSENSUS_MODELS, detect_score_polarity, pad_register, score_raw_models


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
    mpl.rcParams["legend.fontsize"] = 6


COL_TEAL = "#0F766E"
COL_BLUE = "#1D4ED8"
COL_AMBER = "#D97706"
COL_SLATE = "#64748B"
COL_DARK = "#0F172A"


def _fisher_ci95(r: float, n: int):
    if n <= 3 or abs(r) >= 0.9999:
        return (r, r)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3.0)
    return (float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se)))


def compute_anchor_df() -> pd.DataFrame:
    lower = detect_score_polarity()
    rows = []
    sources = [
        ("Thomas_2013", THOMAS_PEPTIDES, THOMAS_PAIRS),
        ("Aronsson_2015", ARONSSON_PEPTIDES, ARONSSON_PAIRS),
        ("Crooks_2017", CROOKS_PEPTIDES, CROOKS_PAIRS),
    ]
    for ds, pdict, plist in sources:
        for item in plist:
            if ds == "Crooks_2017":
                p1, p2, tm, grp = item
                kd = None
            else:
                p1, p2, tm, kd, grp = item
            s1, r1 = pdict[p1]
            s2, r2 = pdict[p2]
            sc = score_raw_models(pad_register(s1, r1), pad_register(s2, r2), lower, scan_shifts=True)
            rows.append({
                "dataset": ds,
                "group": grp,
                "p1": p1,
                "p2": p2,
                "tm_exp": np.nan if tm is None else float(tm),
                "log10_kd": np.log10(kd) if (kd is not None and kd > 0) else np.nan,
                "pKd_exp": -np.log10(kd) if (kd is not None and kd > 0) else np.nan,
                **sc,
            })
    return pd.DataFrame(rows)


def _draw_pipeline(ax, r_loo: float, rmse_loo: float) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 16)
    ax.axis("off")

    steps = [
        ("1. Sequence Space", "Harbury core + E/K barcodes\n3–5 heptad barrels", "#EFF6FF", COL_BLUE),
        ("2. 1D Consensus", f"4-model Ridge (LOO)\nr={r_loo:.3f}, RMSE={rmse_loo:.2f}", "#F0FDFA", COL_TEAL),
        ("3. Circuit Design", "Multi-start SA + EQTK\nGraphs, gates, curves", "#F8FAFC", "#475569"),
        ("4. MC Stress Test", "σ(pKd)=0.643 + 1.5% conc.\nFailure-mode audit", "#FFFBEB", COL_AMBER),
        ("5. 3D Filtering", "Boltz-2 + PyRosetta\nShifts, homo, trimers", "#F8FAFC", COL_DARK),
    ]

    xs = [0.8, 20.9, 41.0, 61.1, 81.2]
    w, h, y0 = 17.6, 13.0, 1.5
    for i, (title, body, fc, ec) in enumerate(steps):
        x0 = xs[i]
        box = FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.3,rounding_size=1.2",
            facecolor=fc, edgecolor=ec, linewidth=0.9, zorder=2,
        )
        ax.add_patch(box)
        ax.text(x0 + w / 2, y0 + h - 3.3, title, ha="center", va="center",
                fontsize=5.8, fontweight="bold", color=ec, zorder=3)
        ax.text(x0 + w / 2, y0 + 4.3, body, ha="center", va="center",
                fontsize=4.7, color=COL_DARK, linespacing=1.28, zorder=3)

        if i < len(steps) - 1:
            arr = FancyArrowPatch(
                (x0 + w + 0.35, y0 + h / 2), (xs[i + 1] - 0.35, y0 + h / 2),
                arrowstyle="-|>,head_length=2.8,head_width=1.8",
                color="#475569", lw=0.85, zorder=3,
            )
            ax.add_patch(arr)


def build_figure1(out_prefix: Path) -> None:
    set_params()
    df = compute_anchor_df()
    kd_df = df[df["pKd_exp"].notna()].copy().reset_index(drop=True)
    cr_df = df[df["dataset"] == "Crooks_2017"].copy().reset_index(drop=True)

    X_kd = kd_df[CONSENSUS_MODELS].values
    y_log = kd_df["log10_kd"].values
    y_pkd = kd_df["pKd_exp"].values

    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 40)))
    y_loo_pkd = -cross_val_predict(pipe, X_kd, y_log, cv=len(kd_df))
    pipe.fit(X_kd, y_log)

    r_loo, _ = stats.pearsonr(y_pkd, y_loo_pkd)
    rmse_loo = float(np.sqrt(np.mean((y_pkd - y_loo_pkd) ** 2)))

    kd_df["Consensus_LOO_pKd"] = y_loo_pkd
    cr_df["Consensus_pKd"] = -pipe.predict(cr_df[CONSENSUS_MODELS].values)

    th_mask = (kd_df["dataset"] == "Thomas_2013").values
    ar_mask = (kd_df["dataset"] == "Aronsson_2015").values
    pipe_th = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 40))).fit(X_kd[th_mask], y_log[th_mask])
    ar_from_th_pkd = -pipe_th.predict(X_kd[ar_mask])
    pipe_ar = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 40))).fit(X_kd[ar_mask], y_log[ar_mask])
    th_from_ar_pkd = -pipe_ar.predict(X_kd[th_mask])

    all_single_models = ["iCipa_nter", "iCipa_vert", "bCIPA", "qCIPA", "Potapov"]
    single_loo_pkd: Dict[str, np.ndarray] = {}
    for m in all_single_models:
        xm = kd_df[[m]].values
        pred_loo = -cross_val_predict(LinearRegression(), xm, y_log, cv=len(kd_df))
        single_loo_pkd[m] = pred_loo
        kd_df[f"{m}_LOO_pKd"] = pred_loo

    hom_idx = int(np.where((kd_df["group"] == "N_homo").values)[0][0])
    print(f"BN4:BN4 exp pKd = {y_pkd[hom_idx]:.2f} | Consensus LOO = {y_loo_pkd[hom_idx]:.2f}")
    for m in CONSENSUS_MODELS:
        print(f"  {m:12s} LOO on BN4:BN4 = {single_loo_pkd[m][hom_idx]:.2f}")

    fig = plt.figure()
    outer = fig.add_gridspec(
        3, 1, left=0.08, right=0.95, top=0.96, bottom=0.065,
        height_ratios=[0.115, 0.435, 0.450], hspace=0.25,
    )

    ax_pipe = fig.add_subplot(outer[0])
    _draw_pipeline(ax_pipe, r_loo, rmse_loo)
    ax_pipe.text(-0.02, 1.02, "A", transform=ax_pipe.transAxes, fontsize=9, fontweight="bold", color=COL_DARK)

    # Panel B
    mid_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], width_ratios=[1.05, 1.0], wspace=0.26)
    sub_2x2 = GridSpecFromSubplotSpec(2, 2, subplot_spec=mid_grid[0], wspace=0.30, hspace=0.38)

    is_th_het = (kd_df["dataset"] == "Thomas_2013") & (kd_df["group"] != "N_homo")
    is_th_hom = (kd_df["dataset"] == "Thomas_2013") & (kd_df["group"] == "N_homo")
    is_ar = kd_df["dataset"] == "Aronsson_2015"

    all_preds = np.concatenate([y_pkd, y_loo_pkd] + [single_loo_pkd[m] for m in CONSENSUS_MODELS])
    lims_single = [min(1.4, float(np.floor(np.min(all_preds) * 2) / 2) - 0.3), 10.6]
    lims_cons = [3.2, 10.6]

    for i, m in enumerate(CONSENSUS_MODELS):
        r_i, c_i = divmod(i, 2)
        ax = fig.add_subplot(sub_2x2[r_i, c_i])
        yp = single_loo_pkd[m]
        r_raw, _ = stats.pearsonr(y_pkd, kd_df[m].values)
        rm, _ = stats.pearsonr(y_pkd, yp)
        rm_rmse = float(np.sqrt(np.mean((y_pkd - yp) ** 2)))

        ax.plot(lims_single, lims_single, color="#94A3B8", ls="--", lw=0.7, zorder=1)
        ax.scatter(y_pkd[is_th_het], yp[is_th_het], s=14, color=COL_BLUE, edgecolors="white", lw=0.3, zorder=3)
        ax.scatter(y_pkd[is_th_hom], yp[is_th_hom], s=28, marker="*", color=COL_AMBER, edgecolors="white", lw=0.2, zorder=4)
        ax.scatter(y_pkd[is_ar], yp[is_ar], s=14, marker="D", color=COL_TEAL, edgecolors="white", lw=0.3, zorder=3)

        # Annotate BN4 homodimer prediction directly next to orange star
        hx, hy = float(y_pkd[hom_idx]), float(yp[hom_idx])
        dx, dy = (0.35, 0.35) if m == "Potapov" else (0.35, -0.15)
        ax.text(hx + dx, hy + dy, f"BN4: {hy:.1f}", fontsize=4.0, color="#B45309", fontweight="bold", va="center")

        ax.set_xlim(lims_single)
        ax.set_ylim(lims_single)
        ticks_s = [t for t in (2, 4, 6, 8, 10) if t >= lims_single[0] - 0.1]
        ax.set_xticks(ticks_s)
        ax.set_yticks(ticks_s)
        ax.set_title(m.replace("_", " "), fontsize=5.8, fontweight="bold", pad=2.5)
        ax.text(0.05, 0.91, f"r = {r_raw:.2f} (LOO {rm:.2f})\nLOO RMSE = {rm_rmse:.2f}", transform=ax.transAxes,
                fontsize=4.4, va="top", color=COL_DARK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=5.1, width=0.5, length=2.0)
        if r_i == 1:
            ax.set_xlabel("Exp. pKd", fontsize=5.5)
        if c_i == 0:
            ax.set_ylabel("Pred. pKd (LOO)", fontsize=5.5)
        if i == 0:
            ax.text(-0.24, 1.14, "B", transform=ax.transAxes, fontsize=9, fontweight="bold", color=COL_DARK)

    ax_cons = fig.add_subplot(mid_grid[1])
    ax_cons.fill_between(lims_cons, [l - rmse_loo for l in lims_cons], [l + rmse_loo for l in lims_cons],
                         color=COL_TEAL, alpha=0.10, label=f"±RMSE ({rmse_loo:.2f})")
    ax_cons.plot(lims_cons, lims_cons, color="#475569", ls="--", lw=0.8, zorder=1)
    ax_cons.scatter(y_pkd[is_th_het], y_loo_pkd[is_th_het], s=26, color=COL_BLUE, edgecolors="white", lw=0.4,
                    label="Thomas 2013 hetero (n=9)", zorder=3)
    ax_cons.scatter(y_pkd[is_th_hom], y_loo_pkd[is_th_hom], s=48, marker="*", color=COL_AMBER, edgecolors="white", lw=0.3,
                    label="Thomas BN4 homo (n=1)", zorder=4)
    ax_cons.scatter(y_pkd[is_ar], y_loo_pkd[is_ar], s=26, marker="D", color=COL_TEAL, edgecolors="white", lw=0.4,
                    label="Aronsson 2015 hetero (n=4)", zorder=3)

    r_ar_cross = float(stats.pearsonr(y_pkd[ar_mask], ar_from_th_pkd)[0])
    r_th_cross = float(stats.pearsonr(y_pkd[th_mask], th_from_ar_pkd)[0])

    ax_cons.set_xlim(lims_cons)
    ax_cons.set_ylim(lims_cons)
    ax_cons.set_xticks([4, 6, 8, 10])
    ax_cons.set_yticks([4, 6, 8, 10])
    ax_cons.set_xlabel("Experimental pKd", fontsize=6.0)
    ax_cons.set_ylabel("Consensus Predicted pKd (LOO-CV)", fontsize=6.0)
    ax_cons.set_title("4-Model Ridge Consensus (LOO-CV, n = 14)", fontsize=6.4, fontweight="bold", pad=3.5)
    ax_cons.text(0.05, 0.94,
                 f"LOO r = {r_loo:.3f} (R² = {r_loo**2:.2f})\n"
                 f"LOO RMSE = {rmse_loo:.3f} pKd\n"
                 f"Cross-fit r (Th→Ar / Ar→Th): {r_ar_cross:.2f} / {r_th_cross:.2f}\n"
                 f"BN4 homo (4.3): singles 1.9–6.8 → Cons. {y_loo_pkd[hom_idx]:.1f}",
                 transform=ax_cons.transAxes, fontsize=4.8, va="top", color=COL_DARK)
    ax_cons.legend(loc="lower right", frameon=False, fontsize=4.8, handletextpad=0.3)
    ax_cons.spines["top"].set_visible(False)
    ax_cons.spines["right"].set_visible(False)
    ax_cons.tick_params(labelsize=5.5, width=0.6, length=2.2)

    ax_w = ax_cons.inset_axes([0.62, 0.25, 0.34, 0.22])
    coefs = -pipe.named_steps["ridgecv"].coef_
    w_norm = coefs / np.sum(np.abs(coefs))
    ax_w.barh(range(len(CONSENSUS_MODELS)), w_norm[::-1], color=COL_TEAL, alpha=0.82, height=0.65)
    ax_w.set_yticks(range(len(CONSENSUS_MODELS)))
    ax_w.set_yticklabels([m.replace("_", " ") for m in CONSENSUS_MODELS[::-1]], fontsize=4.2)
    ax_w.set_xlim(0, 0.6)
    ax_w.set_xticks([0, 0.25, 0.5])
    ax_w.tick_params(labelsize=4.0, width=0.4, length=1.5, pad=1.0)
    ax_w.set_title("Norm. |Ridge weight|", fontsize=4.3, pad=1.5)
    ax_w.spines["top"].set_visible(False)
    ax_w.spines["right"].set_visible(False)

    # Bottom row: Panel C & D
    bot_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], width_ratios=[1.05, 1.0], wspace=0.26)

    ax_bench = fig.add_subplot(bot_grid[0])
    all_models = ["iCipa_nter", "iCipa_vert", "bCIPA", "qCIPA", "Potapov", "Consensus"]
    ds_labels = [
        ("Thomas\n(Kd, n=10)", kd_df[th_mask], "pKd_exp", 10, 1.0),
        ("Aronsson\n(Kd, n=4*)", kd_df[ar_mask], "pKd_exp", 4, 0.50),
        ("Combined\n(Kd, n=14)", kd_df, "pKd_exp", 14, 1.0),
        ("Crooks 2017\n(Tm, n=158)", cr_df, "tm_exp", len(cr_df), 1.0),
    ]
    mat_r = np.zeros((len(all_models), len(ds_labels)), dtype=float)
    for mi, m in enumerate(all_models):
        for di, (dlbl, sub, ycol, _, _) in enumerate(ds_labels):
            if m == "Consensus":
                x_vals = sub["Consensus_LOO_pKd"].values if "Consensus_LOO_pKd" in sub.columns else sub["Consensus_pKd"].values
            else:
                x_vals = sub[m].values
            mat_r[mi, di] = float(stats.pearsonr(x_vals, sub[ycol].values)[0])
        print(f"Bench r | {m:10s}: " + ", ".join(f"{mat_r[mi, di]:+.3f}" for di in range(len(ds_labels))))

    # Identify best single model per dataset for symmetric 95% CI plotting
    best_single_by_ds = [int(np.argmax(mat_r[:-1, di])) for di in range(len(ds_labels))]

    x_pos = np.arange(len(ds_labels))
    bar_w = 0.125
    m_cols = ["#93C5FD", "#60A5FA", "#A78BFA", "#CBD5E1", "#FBBF24", COL_TEAL]
    for mi, (m, col) in enumerate(zip(all_models, m_cols)):
        offs = (mi - (len(all_models) - 1) / 2.0) * bar_w
        lw = 0.65 if m == "Consensus" else 0.0
        for di, (_, _, _, n_pts, alp) in enumerate(ds_labels):
            r_val = mat_r[mi, di]
            ax_bench.bar(
                x_pos[di] + offs, r_val, width=bar_w * 0.92, color=col, alpha=alp,
                edgecolor=COL_DARK, linewidth=lw,
                label=(m.replace("_", " ") if di == 0 else None), zorder=2,
            )
            if n_pts >= 10 and (m == "Consensus" or mi == best_single_by_ds[di]):
                ci_lo, ci_hi = _fisher_ci95(r_val, n_pts)
                ax_bench.errorbar(
                    x_pos[di] + offs, r_val,
                    yerr=[[max(0.0, r_val - ci_lo)], [max(0.0, ci_hi - r_val)]],
                    fmt="none", ecolor=COL_DARK, elinewidth=0.60, capsize=1.2, capthick=0.50, zorder=4,
                )

    ax_bench.axhline(0, color="#94A3B8", lw=0.55)
    ax_bench.set_xticks(x_pos)
    ax_bench.set_xticklabels([d[0] for d in ds_labels], fontsize=4.8)
    y_min_b = min(-0.13, float(np.min(mat_r)) - 0.08)
    ax_bench.set_ylim(y_min_b, 1.15)
    ax_bench.set_ylabel("Pearson correlation (r)", fontsize=6.0)
    ax_bench.set_title("Single-Model (Raw) vs Consensus (LOO / Zero-Shot)", fontsize=6.0, fontweight="bold", pad=3.5)
    ax_bench.text(0.02, 0.02, "*n=4 (wide CI); Consensus = LOO-CV (Kd) / Zero-shot (Crooks); CI = 95% (n≥10)",
                  transform=ax_bench.transAxes, fontsize=3.9, color="#475569")
    ax_bench.legend(loc="upper right", ncol=2, frameon=False, fontsize=4.5, handlelength=1.1, columnspacing=0.8)
    ax_bench.spines["top"].set_visible(False)
    ax_bench.spines["right"].set_visible(False)
    ax_bench.tick_params(labelsize=5.3, width=0.6, length=2.2)
    ax_bench.text(-0.14, 1.04, "C", transform=ax_bench.transAxes, fontsize=9, fontweight="bold", color=COL_DARK)

    # Panel D: Independent Crooks 2017
    ax_cr = fig.add_subplot(bot_grid[1])
    is_des = (cr_df["group"] == "Desired").values
    is_hom = (cr_df["group"] == "Homodimer").values
    is_off = ~(is_des | is_hom)
    is_noncog = ~is_des

    x_cr = cr_df["Consensus_pKd"].values
    y_cr = cr_df["tm_exp"].values
    r_cr, _ = stats.pearsonr(x_cr, y_cr)
    r_noncog, _ = stats.pearsonr(x_cr[is_noncog], y_cr[is_noncog])
    rho_cr, _ = stats.spearmanr(x_cr, y_cr)
    sl, ic, _, _, _ = stats.linregress(x_cr, y_cr)
    gap_pkd = float(np.min(x_cr[is_des]) - np.max(x_cr[is_noncog]))
    n_clipped = int(np.sum(x_cr < 2.0))
    print(f"Crooks 2017: r_all={r_cr:.3f}, r_noncog={r_noncog:.3f}, rho={rho_cr:.3f}, gap={gap_pkd:+.2f}, n_clipped={n_clipped}")

    x_min_cr = float(np.min(x_cr)) - 0.5
    ax_cr.axvspan(x_min_cr, 2.0, color="#E2E8F0", alpha=0.45, zorder=0)
    ax_cr.axvline(2.0, color="#94A3B8", ls=":", lw=0.7, zorder=1)
    ax_cr.text(1.80, 68, f"pKd<2\n(n={n_clipped},\nclipped)", ha="right", va="top", fontsize=4.2, color="#64748B")

    ax_cr.scatter(x_cr[is_off], y_cr[is_off], s=12, color=COL_SLATE, alpha=0.65, edgecolors="white", lw=0.25,
                  label=f"Off-target (n={np.sum(is_off)})", zorder=2)
    ax_cr.scatter(x_cr[is_hom], y_cr[is_hom], s=18, marker="^", color=COL_AMBER, alpha=0.85, edgecolors="white", lw=0.25,
                  label=f"Homodimer (n={np.sum(is_hom)})", zorder=3)
    ax_cr.scatter(x_cr[is_des], y_cr[is_des], s=28, color=COL_TEAL, edgecolors=COL_DARK, lw=0.45,
                  label=f"Cognate pair (n={np.sum(is_des)})", zorder=4)

    x_line = np.linspace(x_min_cr + 0.2, float(np.max(x_cr)) + 0.3, 50)
    ax_cr.plot(x_line, sl * x_line + ic, color=COL_DARK, ls="--", lw=0.85, zorder=3)
    ax_cr.set_xlim(x_min_cr, float(np.max(x_cr)) + 0.6)

    ax_cr.set_xlabel("Consensus Predicted pKd (Zero-Shot)", fontsize=6.0)
    ax_cr.set_ylabel("Experimental Tm (°C)", fontsize=6.0)
    ax_cr.set_title(f"Independent Validation: Crooks 2017 (n = {len(cr_df)})", fontsize=6.3, fontweight="bold", pad=3.5)
    ax_cr.text(0.30, 0.95,
               f"r (all / non-cog) = {r_cr:.2f} / {r_noncog:.2f}\n"
               f"Spearman ρ = {rho_cr:.3f}\n"
               f"12/12 cognates > 146 non-cog\n"
               f"Min separation gap = +{gap_pkd:.2f} pKd",
               transform=ax_cr.transAxes, fontsize=4.8, va="top", color=COL_DARK)
    ax_cr.legend(loc="lower right", frameon=False, fontsize=4.7, handletextpad=0.3)
    ax_cr.spines["top"].set_visible(False)
    ax_cr.spines["right"].set_visible(False)
    ax_cr.tick_params(labelsize=5.5, width=0.6, length=2.2)
    ax_cr.text(-0.14, 1.04, "D", transform=ax_cr.transAxes, fontsize=9, fontweight="bold", color=COL_DARK)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), transparent=True)
    fig.savefig(out_prefix.with_suffix(".svg"), transparent=True)
    fig.savefig(out_prefix.with_suffix(".png"), facecolor="white", dpi=300)
    plt.close(fig)
    print(f"Saved Figure 1 to {out_prefix}.{{pdf,svg,png}}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Figure 1 (pipeline + 1D consensus validation).")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "figures" / "Figure1_predictor")
    args = parser.parse_args()
    build_figure1(args.out)


if __name__ == "__main__":
    main()
