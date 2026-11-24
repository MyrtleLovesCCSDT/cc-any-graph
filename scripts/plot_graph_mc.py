import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="3-panel MC diagnostic plot for mode='graph' designs.")
    ap.add_argument("--full-json", required=True, help="Path to <name>_full.json")
    ap.add_argument("--design-id", default="D01", help="Design ID to plot (default: D01)")
    ap.add_argument("--n-mc", type=int, default=400)
    ap.add_argument("--sigma", type=float, default=0.643)
    ap.add_argument("--min-gap", type=float, default=1.0, help="Target minus worst non-target pKd margin")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    full_path = Path(args.full_json)
    data = json.loads(full_path.read_text())
    cfg = data["config"]
    designs = data["designs"]
    design = next((d for d in designs if d.get("design_id") == args.design_id), designs[0])
    did = design.get("design_id", args.design_id)

    node_names = [n["name"] for n in cfg["nodes"]]
    n = len(node_names)
    idx = {name: i for i, name in enumerate(node_names)}

    pkd = np.array(design["pkd_matrix"], dtype=float)
    pkd = 0.5 * (pkd + pkd.T)

    target_pairs = set()
    for e in cfg.get("edges", []):
        i, j = idx[e["u"]], idx[e["v"]]
        target_pairs.add((min(i, j), max(i, j)))

    target_list = sorted(target_pairs)
    off_list = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in target_pairs]
    homo_list = [(i, i) for i in range(n)]
    non_target_list = off_list + homo_list
    hierarchies = cfg.get("hierarchies", [])

    # Monte Carlo symmetric perturbations of unique pairs (i <= j)
    rng = np.random.default_rng(args.seed)
    iu = np.triu_indices(n)
    mc_mats = np.empty((args.n_mc, n, n), dtype=float)

    cats = [
        "Pass (gap >= 1.0 & hier)",
        "Marginal pass (0 < gap < 1.0)",
        "Fail: Hierarchy inversion",
        "Fail: Off/homo >= target",
        "Fail: Weak target (<5.0 pKd)",
    ]
    counts = {c: 0 for c in cats}

    for m in range(args.n_mc):
        mat = pkd.copy()
        mat[iu] += rng.normal(0.0, args.sigma, size=len(iu[0]))
        mat = np.triu(mat) + np.triu(mat, 1).T
        mc_mats[m] = mat

        min_t = min(mat[i, j] for i, j in target_list)
        max_nt = max(mat[i, j] for i, j in non_target_list) if non_target_list else -99.0
        gap = min_t - max_nt

        hier_ok = True
        for h in hierarchies:
            si, sj = idx[h["strong"][0]], idx[h["strong"][1]]
            wi, wj = idx[h["weak"][0]], idx[h["weak"][1]]
            if mat[si, sj] <= mat[wi, wj]:
                hier_ok = False
                break

        if min_t < 5.0:
            counts["Fail: Weak target (<5.0 pKd)"] += 1
        elif gap <= 0.0:
            counts["Fail: Off/homo >= target"] += 1
        elif not hier_ok:
            counts["Fail: Hierarchy inversion"] += 1
        elif gap < args.min_gap:
            counts["Marginal pass (0 < gap < 1.0)"] += 1
        else:
            counts["Pass (gap >= 1.0 & hier)"] += 1

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), gridspec_kw={"width_ratios": [1.05, 1.25, 1.0]})

    # Panel 1: Nominal pKd matrix with target edge outlines
    ax = axes[0]
    vmax = max(10.5, float(np.max(pkd)))
    im = ax.imshow(pkd, cmap="YlGnBu", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(node_names, fontweight="bold")
    ax.set_yticklabels(node_names, fontweight="bold")
    ax.set_title(f"{cfg['name']} {did} | Nominal pKd Matrix")

    for i in range(n):
        for j in range(n):
            val = pkd[i, j]
            is_target = (min(i, j), max(i, j)) in target_pairs
            txt_color = "white" if val > vmax * 0.58 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=txt_color, fontsize=9,
                    fontweight="bold" if is_target else "normal")
            if is_target:
                rect = patches.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                         linewidth=2.0, edgecolor="#ff9800", facecolor="none")
                ax.add_patch(rect)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Predicted pKd")

    # Panel 2: Pairwise MC distributions (targets vs worst off-targets & homodimers)
    ax2 = axes[1]
    t_sorted = sorted(target_list, key=lambda p: pkd[p], reverse=True)
    off_sorted = sorted(off_list, key=lambda p: pkd[p], reverse=True)[:6]
    homo_sorted = sorted(homo_list, key=lambda p: pkd[p], reverse=True)

    rows = []
    for i, j in t_sorted:
        rows.append((f"{node_names[i]}:{node_names[j]} (target)", i, j, "#1b9e77"))
    for i, j in off_sorted:
        rows.append((f"{node_names[i]}:{node_names[j]} (off)", i, j, "#d95f02"))
    for i, j in homo_sorted:
        rows.append((f"{node_names[i]}:{node_names[j]} (homo)", i, j, "#e41a1c"))

    y_pos = np.arange(len(rows))[::-1]
    for y, (label, i, j, col) in zip(y_pos, rows):
        vals = mc_mats[:, i, j]
        q05, q25, q50, q75, q95 = np.percentile(vals, [5, 25, 50, 75, 95])
        ax2.plot([q05, q95], [y, y], color=col, lw=1.6, alpha=0.45)
        ax2.plot([q25, q75], [y, y], color=col, lw=4.2, alpha=0.75)
        ax2.scatter([pkd[i, j]], [y], color=col, edgecolor="black", zorder=4, s=36)

    off_cap = cfg.get("off_target_max_pkd", 4.5)
    ax2.axvline(off_cap, color="#e41a1c", ls="--", lw=1.2, alpha=0.8, label=f"Off-target cap ({off_cap})")
    min_t_nom = min(pkd[i, j] for i, j in target_list)
    ax2.axvline(min_t_nom, color="#1b9e77", ls=":", lw=1.2, alpha=0.8, label=f"Min target ({min_t_nom:.2f})")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax2.set_xlabel("pKd (5–95% / 25–75% MC + nominal dot)")
    ax2.set_title(f"Pairwise MC Spread (n={args.n_mc}, sigma={args.sigma})")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(axis="x", alpha=0.25)

    # Panel 3: MC Outcome breakdown
    ax3 = axes[2]
    bar_colors = ["#2ca25f", "#e6ab02", "#fdae6b", "#e34a33", "#b10026"]
    pcts = [100.0 * counts[c] / args.n_mc for c in cats]
    y3 = np.arange(len(cats))[::-1]
    ax3.barh(y3, pcts, color=bar_colors, height=0.6)
    for y, p in zip(y3, pcts):
        ax3.text(min(p + 1.5, 98), y, f"{p:.1f}%", va="center", fontsize=9)
    ax3.set_yticks(y3)
    ax3.set_yticklabels(cats, fontsize=8.5)
    ax3.set_xlim(0, 108)
    ax3.set_xlabel("MC realizations (%)")
    ax3.set_title("MC Graph Topology & Hierarchy Outcomes")
    ax3.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    if args.out:
        out_path = Path(args.out)
    else:
        stem = full_path.stem.replace("_full", "")
        out_path = full_path.parent / f"{stem}_{did}_mc_sim.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)

    pass_pct = pcts[0]
    marg_pct = pcts[0] + pcts[1]
    print(f"{cfg['name']} ({did}): Pass(gap>=1)={pass_pct:.1f}% | Order-preserved={marg_pct:.1f}% -> {out_path}")


if __name__ == "__main__":
    main()
