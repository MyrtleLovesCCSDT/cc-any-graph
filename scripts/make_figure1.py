"""Generate Figure 1: 1D consensus calibration, model comparison & independent validation."""

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
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
    mpl.rcParams["figure.figsize"] = (210 / 25.4, 126 / 25.4)
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
    mpl.rcParams["legend.fontsize"] = 5.5


COL_TEAL = "#0F766E"
COL_BLUE = "#1D4ED8"
COL_AMBER = "#D97706"
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


def build_figure1(out_prefix: Path) -> None:
    set_params()
    df = compute_anchor_df()
    kd_df = df[df["pKd_exp"].notna()].copy().reset_index(drop=True)
    cr_df = df[df["dataset"] == "Crooks_2017"].copy().reset_index(drop=True)

    X_kd = kd_df[CONSENSUS_MODELS].values
    y_log = kd_df["log10_kd"].values
    y_pkd = kd_df["pKd_exp"].values

    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 40)))
    y_cv_pkd = -cross_val_predict(pipe, X_kd, y_log, cv=len(kd_df))
    pipe.fit(X_kd, y_log)

    r_cv, _ = stats.pearsonr(y_pkd, y_cv_pkd)
    rmse_cv = float(np.sqrt(np.mean((y_pkd - y_cv_pkd) ** 2)))

    kd_df["Consensus_CV_pKd"] = y_cv_pkd
    cr_df["Consensus_pKd"] = -pipe.predict(cr_df[CONSENSUS_MODELS].values)

    th_mask = (kd_df["dataset"] == "Thomas_2013").values
    ar_mask = (kd_df["dataset"] == "Aronsson_2015").values

    single_cv_pkd: Dict[str, np.ndarray] = {}
    for m in CONSENSUS_MODELS:
        xm = kd_df[[m]].values
        pred_cv = -cross_val_predict(LinearRegression(), xm, y_log, cv=len(kd_df))
        single_cv_pkd[m] = pred_cv
        kd_df[f"{m}_CV_pKd"] = pred_cv

    fig = plt.figure()
    outer = fig.add_gridspec(
        2, 1, left=0.075, right=0.96, top=0.945, bottom=0.09,
        height_ratios=[0.52, 0.48], hspace=0.33,
    )

    # Panel A: 2x2 single predictors + Consensus scatter
    mid_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[1.04, 1.0], wspace=0.25)
    sub_2x2 = GridSpecFromSubplotSpec(2, 2, subplot_spec=mid_grid[0], wspace=0.28, hspace=0.40)

    is_th_het = (kd_df["dataset"] == "Thomas_2013") & (kd_df["group"] != "N_homo")
    is_th_hom = (kd_df["dataset"] == "Thomas_2013") & (kd_df["group"] == "N_homo")
    is_ar = kd_df["dataset"] == "Aronsson_2015"

    all_preds = np.concatenate([y_pkd, y_cv_pkd] + [single_cv_pkd[m] for m in CONSENSUS_MODELS])
    lims_single = [min(1.4, float(np.floor(np.min(all_preds) * 2) / 2) - 0.3), 10.6]
    lims_cons = [3.2, 10.6]

    for i, m in enumerate(CONSENSUS_MODELS):
        r_i, c_i = divmod(i, 2)
        ax = fig.add_subplot(sub_2x2[r_i, c_i])
        yp = single_cv_pkd[m]
        rm, _ = stats.pearsonr(y_pkd, yp)
        rm_rmse = float(np.sqrt(np.mean((y_pkd - yp) ** 2)))

        ax.plot(lims_single, lims_single, color="#94A3B8", ls="--", lw=0.7, zorder=1)
        ax.scatter(y_pkd[is_th_het], yp[is_th_het], s=13, color=COL_BLUE, edgecolors="white", lw=0.3, zorder=3)
        ax.scatter(y_pkd[is_th_hom], yp[is_th_hom], s=26, marker="*", color=COL_AMBER, edgecolors="white", lw=0.2, zorder=4)
        ax.scatter(y_pkd[is_ar], yp[is_ar], s=13, marker="D", color=COL_TEAL, edgecolors="white", lw=0.3, zorder=3)

        ax.set_xlim(lims_single)
        ax.set_ylim(lims_single)
        ticks_s = [t for t in (2, 4, 6, 8, 10) if t >= lims_single[0] - 0.1]
        ax.set_xticks(ticks_s)
        ax.set_yticks(ticks_s)
        ax.set_title(m.replace("_", " "), fontsize=5.6, fontweight="bold", pad=2.0)
        ax.text(0.05, 0.92, f"r = {rm:.2f}\nRMSE = {rm_rmse:.2f}", transform=ax.transAxes,
                fontsize=4.3, va="top", color=COL_DARK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=5.0, width=0.5, length=2.0)
        if r_i == 1:
            ax.set_xlabel("Measured pKd", fontsize=5.3)
        if c_i == 0:
            ax.set_ylabel("Predicted pKd", fontsize=5.3)
        if i == 0:
            ax.text(-0.24, 1.14, "A", transform=ax.transAxes, fontsize=9, fontweight="bold", color=COL_DARK)

    ax_cons = fig.add_subplot(mid_grid[1])
    ax_cons.fill_between(lims_cons, [l - rmse_cv for l in lims_cons], [l + rmse_cv for l in lims_cons],
                         color=COL_TEAL, alpha=0.10, zorder=0)
    ax_cons.plot(lims_cons, lims_cons, color="#475569", ls="--", lw=0.8, zorder=1)
    ax_cons.scatter(y_pkd[is_th_het], y_cv_pkd[is_th_het], s=24, color=COL_BLUE, edgecolors="white", lw=0.4,
                    label="Thomas 2013 hetero (n=9)", zorder=3)
    ax_cons.scatter(y_pkd[is_th_hom], y_cv_pkd[is_th_hom], s=44, marker="*", color=COL_AMBER, edgecolors="white", lw=0.3,
                    label="Thomas BN4 homo (n=1)", zorder=4)
    ax_cons.scatter(y_pkd[is_ar], y_cv_pkd[is_ar], s=24, marker="D", color=COL_TEAL, edgecolors="white", lw=0.4,
                    label="Aronsson 2015 hetero (n=4)", zorder=3)

    ax_cons.set_xlim(lims_cons)
    ax_cons.set_ylim(lims_cons)
    ax_cons.set_xticks([4, 6, 8, 10])
    ax_cons.set_yticks([4, 6, 8, 10])
    ax_cons.set_xlabel("Measured pKd", fontsize=5.8)
    ax_cons.set_ylabel("Cross-validated predicted pKd", fontsize=5.8)
    ax_cons.set_title("4-Model Consensus Predictor (n = 14)", fontsize=6.2, fontweight="bold", pad=3.0)
    ax_cons.text(0.04, 0.94,
                 f"r = {r_cv:.2f} (R² = {r_cv**2:.2f})\nRMSE = {rmse_cv:.2f} pKd",
                 transform=ax_cons.transAxes, fontsize=5.0, va="top", color=COL_DARK)
    ax_cons.legend(loc="upper left", bbox_to_anchor=(0.01, 0.79), frameon=False,
                   fontsize=4.6, handletextpad=0.3, labelspacing=0.35)
    ax_cons.spines["top"].set_visible(False)
    ax_cons.spines["right"].set_visible(False)
    ax_cons.tick_params(labelsize=5.2, width=0.6, length=2.0)

    # Inset cleanly in empty lower-right corner (no overlap with data points)
    ax_w = ax_cons.inset_axes([0.65, 0.09, 0.31, 0.24])
    coefs = -pipe.named_steps["ridgecv"].coef_
    w_norm = coefs / np.sum(np.abs(coefs))
    ax_w.barh(range(len(CONSENSUS_MODELS)), w_norm[::-1], color=COL_TEAL, alpha=0.82, height=0.65)
    ax_w.set_yticks(range(len(CONSENSUS_MODELS)))
    ax_w.set_yticklabels([m.replace("_", " ") for m in CONSENSUS_MODELS[::-1]], fontsize=4.1)
    ax_w.set_xlim(0, 0.6)
    ax_w.set_xticks([0, 0.25, 0.5])
    ax_w.tick_params(labelsize=4.0, width=0.4, length=1.5, pad=1.0)
    ax_w.set_title("Model weight", fontsize=4.3, pad=1.5)
    ax_w.spines["top"].set_visible(False)
    ax_w.spines["right"].set_visible(False)

    # Row 2: Panel B & C
    bot_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], width_ratios=[1.04, 1.0], wspace=0.25)

    ax_bench = fig.add_subplot(bot_grid[0])
    all_models = ["iCipa_nter", "iCipa_vert", "bCIPA", "Potapov", "Consensus"]
    ds_labels = [
        ("Thomas\n(Kd, n=10)", kd_df[th_mask], "pKd_exp", 10, 1.0),
        ("Aronsson\n(Kd, n=4)", kd_df[ar_mask], "pKd_exp", 4, 1.0),
        ("Combined\n(Kd, n=14)", kd_df, "pKd_exp", 14, 1.0),
        ("Crooks 2017\n(Tm, n=158)", cr_df, "tm_exp", len(cr_df), 1.0),
    ]
    mat_r = np.zeros((len(all_models), len(ds_labels)), dtype=float)
    for mi, m in enumerate(all_models):
        for di, (_, sub, ycol, _, _) in enumerate(ds_labels):
            if m == "Consensus":
                x_vals = sub["Consensus_CV_pKd"].values if "Consensus_CV_pKd" in sub.columns else sub["Consensus_pKd"].values
            else:
                x_vals = sub[m].values
            mat_r[mi, di] = float(stats.pearsonr(x_vals, sub[ycol].values)[0])

    best_single_by_ds = [int(np.argmax(mat_r[:-1, di])) for di in range(len(ds_labels))]
    x_pos = np.arange(len(ds_labels))
    bar_w = 0.145
    m_cols = ["#93C5FD", "#60A5FA", "#A78BFA", "#FBBF24", COL_TEAL]
    for mi, (m, col) in enumerate(zip(all_models, m_cols)):
        offs = (mi - (len(all_models) - 1) / 2.0) * bar_w
        lw = 0.65 if m == "Consensus" else 0.0
        for di, (_, _, _, n_pts, alp) in enumerate(ds_labels):
            r_val = mat_r[mi, di]
            ax_bench.bar(
                x_pos[di] + offs, r_val, width=bar_w * 0.90, color=col, alpha=alp,
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
    ax_bench.set_xticklabels([d[0] for d in ds_labels], fontsize=4.9)
    ax_bench.set_ylim(0.0, 1.15)
    ax_bench.set_ylabel("Pearson correlation (r)", fontsize=5.8)
    ax_bench.set_title("Individual Models vs Consensus", fontsize=6.2, fontweight="bold", pad=3.0)
    ax_bench.legend(loc="upper right", ncol=2, frameon=False, fontsize=4.5, handlelength=1.1, columnspacing=0.8)
    ax_bench.spines["top"].set_visible(False)
    ax_bench.spines["right"].set_visible(False)
    ax_bench.tick_params(labelsize=5.2, width=0.6, length=2.0)
    ax_bench.text(-0.14, 1.05, "B", transform=ax_bench.transAxes, fontsize=9, fontweight="bold", color=COL_DARK)

    # Panel C: Independent Crooks 2017
    ax_cr = fig.add_subplot(bot_grid[1])
    x_cr = cr_df["Consensus_pKd"].values
    y_cr = cr_df["tm_exp"].values
    r_cr, _ = stats.pearsonr(x_cr, y_cr)
    rho_cr, _ = stats.spearmanr(x_cr, y_cr)
    sl, ic, _, _, _ = stats.linregress(x_cr, y_cr)

    ax_cr.scatter(x_cr, y_cr, s=16, color=COL_TEAL, alpha=0.72, edgecolors="white", lw=0.3, zorder=3)
    x_min_cr = float(np.min(x_cr)) - 0.4
    x_max_cr = float(np.max(x_cr)) + 0.5
    x_line = np.linspace(x_min_cr + 0.2, x_max_cr - 0.2, 50)
    ax_cr.plot(x_line, sl * x_line + ic, color=COL_DARK, ls="--", lw=0.85, zorder=4)
    ax_cr.set_xlim(x_min_cr, x_max_cr)

    ax_cr.set_xlabel("Consensus Predicted pKd", fontsize=5.8)
    ax_cr.set_ylabel("Experimental Tm (°C)", fontsize=5.8)
    ax_cr.set_title(f"Independent Test: Crooks 2017 (n = {len(cr_df)})", fontsize=6.2, fontweight="bold", pad=3.0)
    ax_cr.text(0.05, 0.93,
               f"Pearson r = {r_cr:.2f}\nSpearman ρ = {rho_cr:.2f}",
               transform=ax_cr.transAxes, fontsize=5.0, va="top", color=COL_DARK)
    ax_cr.spines["top"].set_visible(False)
    ax_cr.spines["right"].set_visible(False)
    ax_cr.tick_params(labelsize=5.2, width=0.6, length=2.0)
    ax_cr.text(-0.14, 1.05, "C", transform=ax_cr.transAxes, fontsize=9, fontweight="bold", color=COL_DARK)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".pdf"), transparent=True)
    fig.savefig(out_prefix.with_suffix(".svg"), transparent=True)
    fig.savefig(out_prefix.with_suffix(".png"), facecolor="white", dpi=300)
    plt.close(fig)
    print(f"Saved Figure 1 to {out_prefix}.{{pdf,svg,png}}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact Figure 1 (1D consensus validation).")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "figures" / "Figure1_predictor")
    args = parser.parse_args()
    build_figure1(args.out)


if __name__ == "__main__":
    main()
