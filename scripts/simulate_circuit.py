"""Dense simulation and Monte Carlo failure-mode visualization for any designed circuit."""

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.circuit_eq import evaluate_truth_table, pair_index_map, solve_general_equilibrium
from src.consensus import DEFAULT_CALIB_PATH
from src.graph_design import _parse_config


def _load_sigma(calib_path: Path) -> float:
    with open(calib_path, "r") as f:
        return float(json.load(f).get("loo_rmse", 0.643))


def simulate_curve_design(cfg: dict, parsed: dict, des: dict, args, sigma_kd: float, obs_idx: int = 0) -> None:
    names = parsed["node_names"]
    n = len(names)
    obs = parsed["observables"][obs_idx]
    c0_states_m = np.array(des["c0_uM"], dtype=float) * 1e-6
    base_log10_kd = -np.array(des["pkd_matrix"], dtype=float)
    kd_nom_m = 10.0 ** np.minimum(base_log10_kd, -2.0)

    # Detect swept concentration column
    col_ptp = np.ptp(c0_states_m, axis=0)
    sweep_idx = int(np.argmax(col_ptp))
    xs_states = c0_states_m[:, sweep_idx] * 1e6
    pos_xs = xs_states[xs_states > 0.0]
    use_log_x = bool(cfg.get("log_x", False) and pos_xs.size > 0)

    if use_log_x:
        x_dense = np.geomspace(float(np.min(pos_xs)), float(np.max(pos_xs)), args.n_grid)
    else:
        x_dense = np.linspace(float(np.min(xs_states)), float(np.max(xs_states)), args.n_grid)

    c0_dense_m = np.repeat(c0_states_m[:1], args.n_grid, axis=0)
    c0_dense_m[:, sweep_idx] = x_dense * 1e-6

    # Nominal smooth curve + species partitioning
    eq_dense_nom = solve_general_equilibrium(c0_dense_m, kd_nom_m)
    y_dense_nom = obs["obj"].evaluate(eq_dense_nom, c0_dense_m, n)

    y_tgt = np.asarray(obs["targets"], dtype=float)
    is_dome = bool(np.max(y_tgt) > y_tgt[0] + 0.15 and np.max(y_tgt) > y_tgt[-1] + 0.15)
    max_rmse = obs["max_rmse"]
    min_span = obs["min_span"] * 0.85

    # Monte Carlo on stacked (states + dense) grid
    c0_stack = np.vstack([c0_states_m, c0_dense_m])
    n_st = c0_states_m.shape[0]
    rng = np.random.default_rng(args.seed)
    tri_i, tri_j = np.triu_indices(n)
    conc_std = args.conc_std if args.conc_std is not None else parsed["mc_conc_rel_std"]

    y_mc_dense = np.empty((args.n_mc, args.n_grid), dtype=float)
    categories = []

    for m in range(args.n_mc):
        pert = base_log10_kd.copy()
        noise = rng.normal(0.0, sigma_kd, size=tri_i.size)
        pert[tri_i, tri_j] += noise
        pert[tri_j, tri_i] = pert[tri_i, tri_j]
        kd_m = 10.0 ** np.minimum(pert, -2.0)

        if conc_std > 0.0:
            mult = np.maximum(1.0 + rng.normal(0.0, conc_std, size=c0_stack.shape), 0.1)
            c0_run = c0_stack * mult
        else:
            c0_run = c0_stack

        eq = solve_general_equilibrium(c0_run, kd_m)
        y_all = obs["obj"].evaluate(eq, c0_run, n)
        y_st = y_all[:n_st]
        y_dn = y_all[n_st:]
        y_mc_dense[m] = y_dn

        if obs["affine_fit"]:
            x_var = np.var(y_st)
            a = max(0.0, float(np.mean((y_st - np.mean(y_st)) * (y_tgt - np.mean(y_tgt)))) / max(x_var, 1e-12)) if x_var > 1e-12 else 0.0
            b = float(np.mean(y_tgt) - a * np.mean(y_st))
            y_fit = a * y_st + b
        else:
            y_fit = y_st

        rmse = float(np.sqrt(np.mean((y_fit - y_tgt) ** 2)))
        span = float(np.max(y_st) - np.min(y_st))

        if rmse <= max_rmse and span >= min_span:
            categories.append("Exact pass")
        elif is_dome:
            peak = float(np.max(y_dn))
            rise = peak - float(y_dn[0])
            fall = peak - float(y_dn[-1])
            if peak >= 0.35 and rise >= 0.25 and fall >= 0.25:
                categories.append("Dome (shifted/scaled)")
            elif y_dn[0] > 0.35 and rise < 0.25:
                categories.append("Fail: High baseline (weak Q:S)")
            elif peak < 0.35 or rise < 0.25:
                categories.append("Fail: Collapsed peak (weak X:Q)")
            elif fall < 0.25:
                categories.append("Fail: Weak shutdown (weak X:S)")
            else:
                categories.append("Fail: Other shape error")
        else:
            ss_tot = float(np.sum((y_tgt - np.mean(y_tgt)) ** 2))
            corr = float(np.corrcoef(y_st, y_tgt)[0, 1]) if np.std(y_st) > 1e-9 else 0.0
            if corr >= 0.90 and span >= 0.35:
                categories.append("Shape ok (shifted scale)")
            elif span < 0.35:
                categories.append("Fail: Flat / low span")
            else:
                categories.append("Fail: Shape distortion")

    counts = Counter(categories)
    print(f"\n=== MC Diagnostics for {parsed['name']} | {des['design_id']} (n_mc={args.n_mc}, sigma_kd={sigma_kd:.3f}) ===")
    for k, v in counts.most_common():
        print(f"  {k:34s}: {v:4d} ({100.0 * v / args.n_mc:5.1f}%)")

    # Plot 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=160)

    # Panel 1: Smooth curve + MC bands + sample colored trajectories
    ax = axes[0]
    p05, p25, p50, p75, p95 = np.percentile(y_mc_dense, [5, 25, 50, 75, 95], axis=0)
    ax.fill_between(x_dense, p05, p95, color="#4daf4a", alpha=0.14, label="MC 5-95%")
    ax.fill_between(x_dense, p25, p75, color="#4daf4a", alpha=0.25, label="MC 25-75%")

    sample_idx = rng.choice(args.n_mc, size=min(30, args.n_mc), replace=False)
    seen_lbl = set()
    for idx in sample_idx:
        cat = categories[idx]
        if cat == "Exact pass":
            col, lbl = "#1b9e77", "MC: Exact pass"
        elif "shifted" in cat.lower() or "shape ok" in cat.lower():
            col, lbl = "#e6ab02", "MC: Shifted/scaled dome"
        else:
            col, lbl = "#e41a1c", "MC: Failed"
        ax.plot(x_dense, y_mc_dense[idx], color=col, lw=0.9, alpha=0.35, label=(lbl if lbl not in seen_lbl else None))
        seen_lbl.add(lbl)

    ax.plot(xs_states, y_tgt, "k--", lw=1.6, label="Target")
    ax.plot(x_dense, y_dense_nom, color="#00441b", lw=2.5, label="Nominal (1D pred)")
    if use_log_x:
        ax.set_xscale("log")
    ax.set_xlabel(f"[{names[sweep_idx]}]0 (uM)")
    ax.set_ylabel("Normalized output")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"{parsed['name']} {des['design_id']} | MC Trajectories")
    ax.legend(fontsize=7.5, loc="best")

    # Panel 2: Mechanistic species partitioning of reporter node
    ax = axes[1]
    pmap = pair_index_map(n)
    rep = obs["obj"].ref_nodes[0]
    denom = np.maximum(c0_dense_m[:, rep], 1e-18)
    ax.plot(x_dense, eq_dense_nom[:, rep] / denom, lw=2.3, label=f"[{names[rep]}]_free")
    for j in range(n):
        col_idx = pmap[(rep, j)]
        mult = 2.0 if j == rep else 1.0
        lbl = f"[{names[rep]}:{names[j]}]" if j != rep else f"2*[{names[rep]}:{names[rep]}] (homo)"
        frac = mult * eq_dense_nom[:, col_idx] / denom
        if np.max(frac) >= 0.01 or j == rep:
            ax.plot(x_dense, frac, lw=1.8, ls=("--" if j == rep else "-"), label=lbl)
    if use_log_x:
        ax.set_xscale("log")
    ax.set_xlabel(f"[{names[sweep_idx]}]0 (uM)")
    ax.set_ylabel(f"Fraction of total [{names[rep]}]0")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Species Partitioning (Nominal)")
    ax.legend(fontsize=8, loc="best")

    # Panel 3: Failure mode breakdown
    ax = axes[2]
    items = counts.most_common()
    labels = [it[0] for it in items][::-1]
    pcts = [100.0 * it[1] / args.n_mc for it in items][::-1]
    bar_cols = [
        "#1b9e77" if "Exact pass" in l else ("#e6ab02" if ("shifted" in l.lower() or "shape ok" in l.lower()) else "#e41a1c")
        for l in labels
    ]
    bars = ax.barh(range(len(labels)), pcts, color=bar_cols, alpha=0.88)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, max(pcts) * 1.22 + 5)
    ax.set_xlabel("MC realizations (%)")
    ax.set_title("MC Outcome & Failure Modes")
    for b, p in zip(bars, pcts):
        ax.text(p + 1.0, b.get_y() + b.get_height() / 2, f"{p:.1f}%", va="center", fontsize=8.5)

    plt.tight_layout()
    n_obs = len(parsed["observables"])
    sfx = f"_obs{obs_idx + 1}" if n_obs > 1 else ""
    out_path = (
        args.out.with_stem(f"{args.out.stem}{sfx}")
        if (args.out is not None and n_obs > 1)
        else (args.out or (args.full_json.parent / f"{parsed['name']}_{des['design_id']}{sfx}_mc_sim.png"))
    )
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved simulation plot to {out_path}")


def simulate_truth_table_design(cfg: dict, parsed: dict, des: dict, args, sigma_kd: float, obs_idx: int = 0) -> None:
    names = parsed["node_names"]
    n = len(names)
    obs = parsed["observables"][obs_idx]
    targets = np.asarray(obs["targets"], dtype=int)
    c0_m = np.array(des["c0_uM"], dtype=float) * 1e-6
    base_log10_kd = -np.array(des["pkd_matrix"], dtype=float)
    kd_nom_m = 10.0 ** np.minimum(base_log10_kd, -2.0)

    tt_nom = evaluate_truth_table(kd_nom_m, c0_m, obs["obj"], targets)
    min_delta = obs["min_delta"] * 0.85
    min_fold = obs["min_fold"] * 0.85

    rng = np.random.default_rng(args.seed)
    tri_i, tri_j = np.triu_indices(n)
    conc_std = args.conc_std if args.conc_std is not None else parsed["mc_conc_rel_std"]

    vals_mc = np.empty((args.n_mc, c0_m.shape[0]), dtype=float)
    deltas = np.empty(args.n_mc, dtype=float)
    categories = []

    for m in range(args.n_mc):
        pert = base_log10_kd.copy()
        noise = rng.normal(0.0, sigma_kd, size=tri_i.size)
        pert[tri_i, tri_j] += noise
        pert[tri_j, tri_i] = pert[tri_i, tri_j]
        kd_m = 10.0 ** np.minimum(pert, -2.0)

        c0_run = c0_m * np.maximum(1.0 + rng.normal(0.0, conc_std, size=c0_m.shape), 0.1) if conc_std > 0.0 else c0_m
        tt = evaluate_truth_table(kd_m, c0_run, obs["obj"], targets)
        vals = tt["values"]
        vals_mc[m] = vals
        deltas[m] = tt["Delta"]

        if tt["Delta"] >= min_delta and tt["Fold"] >= min_fold:
            categories.append("Pass")
        else:
            off_idx = np.where(targets == 0)[0]
            on_idx = np.where(targets == 1)[0]
            worst_off_i = int(off_idx[np.argmax(vals[off_idx])])
            worst_on_i = int(on_idx[np.argmin(vals[on_idx])])
            if vals[worst_off_i] > 0.35:
                categories.append(f"Fail: Leak in S{worst_off_i} (OFF)")
            elif vals[worst_on_i] < 0.55:
                categories.append(f"Fail: Weak ON in S{worst_on_i}")
            else:
                categories.append("Fail: Low Delta / Fold")

    counts = Counter(categories)
    print(f"\n=== MC Diagnostics for {parsed['name']} | {des['design_id']} (n_mc={args.n_mc}, sigma_kd={sigma_kd:.3f}, conc_std={conc_std:.2f}) ===")
    for k, v in counts.most_common():
        print(f"  {k:30s}: {v:4d} ({100.0 * v / args.n_mc:5.1f}%)")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=160)

    # Panel 1: Truth table with MC 10-90% intervals and sample trajectories
    ax = axes[0]
    x_idx = np.arange(c0_m.shape[0])
    p10, p50, p90 = np.percentile(vals_mc, [10, 50, 90], axis=0)
    bar_cols = ["#1b9e77" if t == 1 else "#377eb8" for t in targets]
    ax.bar(x_idx, tt_nom["values"], width=0.55, color=bar_cols, alpha=0.80, label="Nominal output")
    ax.errorbar(x_idx, p50, yerr=[p50 - p10, p90 - p50], fmt="o", color="black", capsize=4, lw=1.4, label="MC median (10-90%)")

    sample_idx = rng.choice(args.n_mc, size=min(35, args.n_mc), replace=False)
    seen_lbl = set()
    for idx in sample_idx:
        is_pass = categories[idx] == "Pass"
        col = "#1b9e77" if is_pass else "#e41a1c"
        lbl = "MC: Pass" if is_pass else "MC: Fail"
        ax.plot(x_idx, vals_mc[idx], color=col, lw=0.8, alpha=0.25, label=(lbl if lbl not in seen_lbl else None))
        seen_lbl.add(lbl)

    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"S{k}\n(tgt={targets[k]})" for k in x_idx])
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Normalized output")
    ax.set_title(f"{parsed['name']} {des['design_id']} | Truth Table + MC")
    ax.legend(fontsize=7.5, loc="best")

    # Panel 2: Distribution of MC_Delta
    ax = axes[1]
    pass_d = deltas[np.array([c == "Pass" for c in categories])]
    fail_d = deltas[np.array([c != "Pass" for c in categories])]
    bins = np.linspace(min(-0.3, float(np.min(deltas))), 1.0, 28)
    ax.hist(pass_d, bins=bins, color="#1b9e77", alpha=0.75, label=f"Pass ({len(pass_d)})")
    ax.hist(fail_d, bins=bins, color="#e41a1c", alpha=0.70, label=f"Fail ({len(fail_d)})")
    ax.axvline(min_delta, color="black", ls="--", lw=1.4, label=f"Min Delta ({min_delta:.2f})")
    ax.set_xlabel("MC Delta (min_ON - max_OFF)")
    ax.set_ylabel("Count")
    ax.set_title(f"Contrast Distribution (Pass={100.0*len(pass_d)/args.n_mc:.1f}%)")
    ax.legend(fontsize=8)

    # Panel 3: Failure mode breakdown
    ax = axes[2]
    items = counts.most_common()
    labels = [it[0] for it in items][::-1]
    pcts = [100.0 * it[1] / args.n_mc for it in items][::-1]
    cols = ["#1b9e77" if l == "Pass" else "#e41a1c" for l in labels]
    bars = ax.barh(range(len(labels)), pcts, color=cols, alpha=0.88)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, max(pcts) * 1.22 + 5)
    ax.set_xlabel("MC realizations (%)")
    ax.set_title("MC Outcome & Failure Modes")
    for b, p in zip(bars, pcts):
        ax.text(p + 1.0, b.get_y() + b.get_height() / 2, f"{p:.1f}%", va="center", fontsize=8.5)

    plt.tight_layout()
    n_obs = len(parsed["observables"])
    sfx = f"_obs{obs_idx + 1}" if n_obs > 1 else ""
    out_path = (
        args.out.with_stem(f"{args.out.stem}{sfx}")
        if (args.out is not None and n_obs > 1)
        else (args.out or (args.full_json.parent / f"{parsed['name']}_{des['design_id']}{sfx}_mc_sim.png"))
    )
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved simulation plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate designed circuit with dense sweep & MC failure modes.")
    parser.add_argument("--full-json", type=Path, required=True)
    parser.add_argument("--design-id", type=str, default="D01")
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB_PATH)
    parser.add_argument("--n-mc", type=int, default=400)
    parser.add_argument("--n-grid", type=int, default=120)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--conc-std", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.full_json, "r") as f:
        bundle = json.load(f)
    cfg = bundle["config"]
    parsed = _parse_config(cfg, allow_5mer=True)

    des_map = {d["design_id"]: d for d in bundle["designs"]}
    if args.design_id not in des_map:
        raise KeyError(f"Design '{args.design_id}' not found in {args.full_json}")
    des = des_map[args.design_id]
    sigma_kd = _load_sigma(args.calib) if args.sigma is None else args.sigma

    n_obs = len(parsed["observables"])
    if parsed["mode"] == "curve":
        for oi in range(n_obs):
            simulate_curve_design(cfg, parsed, des, args, sigma_kd, obs_idx=oi)
    elif parsed["mode"] == "truth_table":
        for oi in range(n_obs):
            simulate_truth_table_design(cfg, parsed, des, args, sigma_kd, obs_idx=oi)
    else:
        raise ValueError(f"Simulation requires mode='curve' or 'truth_table', got '{parsed['mode']}'.")


if __name__ == "__main__":
    main()
