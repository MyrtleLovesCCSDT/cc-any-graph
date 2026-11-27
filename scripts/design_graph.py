"""CLI for arbitrary coiled-coil graph, logic-gate, and analog-function design."""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.circuit_eq import solve_general_equilibrium
from src.consensus import DEFAULT_CALIB_PATH
from src.graph_design import _parse_config, run_graph_design


def make_preset(name: str) -> dict:
    if name == "yes":
        return {
            "name": "yes_3node",
            "mode": "truth_table",
            "nodes": [
                {"name": "I", "charge_mode": "negative", "lengths": ["4mer"]},
                {"name": "Q", "charge_mode": "positive", "lengths": ["4mer"]},
                {"name": "S", "charge_mode": "negative", "lengths": ["4mer", "35mer", "3mer"]},
            ],
            "edges": [
                {"u": "I", "v": "Q", "pkd_min": 7.4, "pkd_max": 11.5, "require_par": True, "require_shift0": True},
                {"u": "Q", "v": "S", "pkd_min": 5.6, "pkd_max": 6.85, "require_par": True, "require_shift0": True},
            ],
            "hierarchies": [
                {"strong": ["I", "Q"], "weak": ["Q", "S"], "min_dpkd": 1.5, "min_agree": 4},
                {"strong": ["Q", "S"], "weak": ["I", "S"], "min_dpkd": 1.0, "min_agree": 3},
            ],
            "off_target_max_pkd": 4.8,
            "homodimer_max_pkd": 4.5,
            "states_uM": [
                {"I": 0.0, "Q": 5.0, "S": 0.5},
                {"I": 12.5, "Q": 5.0, "S": 0.5},
            ],
            "observables": [
                {"kind": "free_frac", "nodes": ["S"], "targets": [0, 1], "min_delta": 0.45, "min_fold": 3.0},
            ],
        }

    if name == "not":
        return {
            "name": "not_4node",
            "mode": "truth_table",
            "nodes": [
                {"name": "I", "charge_mode": "mixed", "lengths": ["4mer"], "allow_5mer_if_enabled": True},
                {"name": "M1", "charge_mode": "mixed", "lengths": ["4mer"], "allow_5mer_if_enabled": True},
                {"name": "Q1", "charge_mode": "mixed", "lengths": ["4mer"]},
                {"name": "S", "charge_mode": "mixed", "lengths": ["4mer", "35mer", "3mer"]},
            ],
            "edges": [
                {"u": "I", "v": "M1", "pkd_min": 8.4, "pkd_max": 12.0, "require_par": True, "require_shift0": True, "min_slippage": 5.0},
                {"u": "M1", "v": "Q1", "pkd_min": 7.0, "pkd_max": 8.6, "require_par": True, "require_shift0": True, "min_slippage": 5.0},
                {"u": "Q1", "v": "S", "pkd_min": 5.8, "pkd_max": 7.0, "require_par": True, "require_shift0": False},
            ],
            "hierarchies": [
                {"strong": ["I", "M1"], "weak": ["M1", "Q1"], "min_dpkd": 1.4, "min_agree": 3},
                {"strong": ["M1", "Q1"], "weak": ["Q1", "S"], "min_dpkd": 1.3, "min_agree": 3},
            ],
            "off_target_max_pkd": 4.5,
            "homodimer_max_pkd": 4.3,
            "states_uM": [
                {"I": 0.0, "M1": 3.5, "Q1": 2.0, "S": 0.5},
                {"I": 12.0, "M1": 3.5, "Q1": 2.0, "S": 0.5},
            ],
            "observables": [
                {"kind": "free_frac", "nodes": ["S"], "targets": [1, 0], "min_delta": 0.50, "min_fold": 4.5},
            ],
            "tunable_scaffolds": [
                {"node": "Q1", "min_uM": 1.0, "max_uM": 4.5},
                {"node": "M1", "min_uM": 1.5, "max_uM": 7.5},
            ],
        }

    if name == "or":
        return {
            "name": "or_4node",
            "mode": "truth_table",
            "nodes": [
                {"name": "I1", "charge_mode": "mixed", "lengths": ["4mer"]},
                {"name": "I2", "charge_mode": "mixed", "lengths": ["4mer"]},
                {"name": "Q", "charge_mode": "mixed", "lengths": ["4mer"]},
                {"name": "S", "charge_mode": "mixed", "lengths": ["4mer", "35mer", "3mer"]},
            ],
            "edges": [
                {"u": "I1", "v": "Q", "pkd_min": 7.3, "pkd_max": 11.5, "require_par": True, "require_shift0": True},
                {"u": "I2", "v": "Q", "pkd_min": 7.3, "pkd_max": 11.5, "require_par": True, "require_shift0": True},
                {"u": "Q", "v": "S", "pkd_min": 5.5, "pkd_max": 6.7, "require_par": True, "require_shift0": True},
            ],
            "hierarchies": [
                {"strong": ["I1", "Q"], "weak": ["Q", "S"], "min_dpkd": 1.3, "min_agree": 3},
                {"strong": ["I2", "Q"], "weak": ["Q", "S"], "min_dpkd": 1.3, "min_agree": 3},
            ],
            "off_target_max_pkd": 4.8,
            "homodimer_max_pkd": 4.5,
            "states_uM": [
                {"I1": 0.0, "I2": 0.0, "Q": 5.0, "S": 0.5},
                {"I1": 12.5, "I2": 0.0, "Q": 5.0, "S": 0.5},
                {"I1": 0.0, "I2": 12.5, "Q": 5.0, "S": 0.5},
                {"I1": 12.5, "I2": 12.5, "Q": 5.0, "S": 0.5},
            ],
            "observables": [
                {"kind": "free_frac", "nodes": ["S"], "targets": [0, 1, 1, 1], "min_delta": 0.45, "min_fold": 3.0},
            ],
        }

    if name == "and":
        return {
            "name": "and_5node",
            "mode": "truth_table",
            "nodes": [
                {"name": "I1", "charge_mode": "mixed", "lengths": ["4mer"], "allow_5mer_if_enabled": True},
                {"name": "I2", "charge_mode": "mixed", "lengths": ["4mer"], "allow_5mer_if_enabled": True},
                {"name": "Q3", "charge_mode": "mixed", "lengths": ["4mer"]},
                {"name": "Q4", "charge_mode": "mixed", "lengths": ["4mer"]},
                {"name": "S", "charge_mode": "mixed", "lengths": ["4mer", "35mer", "3mer"]},
            ],
            "edges": [
                {"u": "I1", "v": "Q4", "pkd_min": 7.8, "pkd_max": 11.5, "require_par": True, "require_shift0": True, "min_slippage": 5.0},
                {"u": "I2", "v": "Q3", "pkd_min": 7.8, "pkd_max": 11.5, "require_par": True, "require_shift0": True, "min_slippage": 5.0},
                {"u": "Q4", "v": "S", "pkd_min": 6.0, "pkd_max": 7.4, "require_par": True, "require_shift0": True},
                {"u": "Q3", "v": "S", "pkd_min": 6.0, "pkd_max": 7.4, "require_par": True, "require_shift0": True},
            ],
            "hierarchies": [
                {"strong": ["I1", "Q4"], "weak": ["Q4", "S"], "min_dpkd": 1.4, "min_agree": 3},
                {"strong": ["I2", "Q3"], "weak": ["Q3", "S"], "min_dpkd": 1.4, "min_agree": 3},
                {"strong": ["Q4", "S"], "weak": ["I1", "Q3"], "min_dpkd": 1.2, "min_agree": 3},
                {"strong": ["Q3", "S"], "weak": ["I2", "Q4"], "min_dpkd": 1.2, "min_agree": 3},
            ],
            "off_target_max_pkd": 4.5,
            "homodimer_max_pkd": 4.3,
            "states_uM": [
                {"I1": 0.0,  "I2": 0.0,  "Q3": 3.0, "Q4": 3.0, "S": 0.5},
                {"I1": 10.0, "I2": 0.0,  "Q3": 3.0, "Q4": 3.0, "S": 0.5},
                {"I1": 0.0,  "I2": 10.0, "Q3": 3.0, "Q4": 3.0, "S": 0.5},
                {"I1": 10.0, "I2": 10.0, "Q3": 3.0, "Q4": 3.0, "S": 0.5},
            ],
            "observables": [
                {"kind": "free_frac", "nodes": ["S"], "targets": [0, 0, 0, 1], "min_delta": 0.65, "min_fold": 8.0},
            ],
            "tunable_scaffolds": [
                {"node": "Q3", "min_uM": 1.5, "max_uM": 6.0},
                {"node": "Q4", "min_uM": 1.5, "max_uM": 6.0},
            ],
        }

    if name == "xor":
        return {
            "name": "xor_4node",
            "mode": "truth_table",
            "nodes": [
                {"name": "A", "charge_mode": "mixed", "lengths": ["4mer"], "allow_5mer_if_enabled": True},
                {"name": "B", "charge_mode": "mixed", "lengths": ["4mer"], "allow_5mer_if_enabled": True},
                {"name": "Q", "charge_mode": "mixed", "lengths": ["4mer"]},
                {"name": "S", "charge_mode": "mixed", "lengths": ["4mer", "35mer", "3mer"]},
            ],
            "edges": [
                {"u": "A", "v": "B", "pkd_min": 8.8, "pkd_max": 12.5, "require_par": True, "require_shift0": True},
                {"u": "A", "v": "Q", "pkd_min": 6.6, "pkd_max": 7.8, "require_par": True, "require_shift0": True},
                {"u": "B", "v": "Q", "pkd_min": 6.6, "pkd_max": 7.8, "require_par": True, "require_shift0": True},
                {"u": "Q", "v": "S", "pkd_min": 5.4, "pkd_max": 6.5, "require_par": True, "require_shift0": False},
            ],
            "hierarchies": [
                {"strong": ["A", "B"], "weak": ["A", "Q"], "min_dpkd": 1.5, "min_agree": 3},
                {"strong": ["A", "B"], "weak": ["B", "Q"], "min_dpkd": 1.5, "min_agree": 3},
                {"strong": ["A", "Q"], "weak": ["Q", "S"], "min_dpkd": 0.8, "min_agree": 3},
                {"strong": ["B", "Q"], "weak": ["Q", "S"], "min_dpkd": 0.8, "min_agree": 3},
            ],
            "off_target_max_pkd": 4.7,
            "homodimer_max_pkd": 4.5,
            "states_uM": [
                {"A": 0.0, "B": 0.0, "Q": 6.0, "S": 0.5},
                {"A": 12.0, "B": 0.0, "Q": 6.0, "S": 0.5},
                {"A": 0.0, "B": 12.0, "Q": 6.0, "S": 0.5},
                {"A": 12.0, "B": 12.0, "Q": 6.0, "S": 0.5},
            ],
            "observables": [
                {"kind": "free_frac", "nodes": ["S"], "targets": [0, 1, 1, 0], "min_delta": 0.35, "min_fold": 2.5},
            ],
            "tunable_scaffolds": [{"node": "Q", "min_uM": 2.0, "max_uM": 12.0}],
            "mc_conc_rel_std": 0.05,
        }

    if name == "ortho2":
        return {
            "name": "ortho2_6node",
            "mode": "truth_table",
            "nodes": [
                {"name": "I1", "charge_mode": "mixed", "lengths": ["4mer"]},
                {"name": "Q1", "charge_mode": "mixed", "lengths": ["4mer"]},
                {"name": "S1", "charge_mode": "mixed", "lengths": ["4mer", "35mer", "3mer"]},
                {"name": "I2", "charge_mode": "mixed", "lengths": ["4mer"]},
                {"name": "Q2", "charge_mode": "mixed", "lengths": ["4mer"]},
                {"name": "S2", "charge_mode": "mixed", "lengths": ["4mer", "35mer", "3mer"]},
            ],
            "edges": [
                {"u": "I1", "v": "Q1", "pkd_min": 7.4, "pkd_max": 11.5, "require_par": True, "require_shift0": True},
                {"u": "Q1", "v": "S1", "pkd_min": 5.6, "pkd_max": 6.8, "require_par": True, "require_shift0": False},
                {"u": "I2", "v": "Q2", "pkd_min": 7.4, "pkd_max": 11.5, "require_par": True, "require_shift0": True},
                {"u": "Q2", "v": "S2", "pkd_min": 5.6, "pkd_max": 6.8, "require_par": True, "require_shift0": False},
            ],
            "hierarchies": [
                {"strong": ["I1", "Q1"], "weak": ["Q1", "S1"], "min_dpkd": 1.4, "min_agree": 3},
                {"strong": ["I2", "Q2"], "weak": ["Q2", "S2"], "min_dpkd": 1.4, "min_agree": 3},
            ],
            "off_target_max_pkd": 4.8,
            "homodimer_max_pkd": 4.5,
            "states_uM": [
                {"I1": 0.0, "Q1": 5.0, "S1": 0.5, "I2": 0.0, "Q2": 5.0, "S2": 0.5},
                {"I1": 12.5, "Q1": 5.0, "S1": 0.5, "I2": 0.0, "Q2": 5.0, "S2": 0.5},
                {"I1": 0.0, "Q1": 5.0, "S1": 0.5, "I2": 12.5, "Q2": 5.0, "S2": 0.5},
                {"I1": 12.5, "Q1": 5.0, "S1": 0.5, "I2": 12.5, "Q2": 5.0, "S2": 0.5},
            ],
            "observables": [
                {"kind": "free_frac", "nodes": ["S1"], "targets": [0, 1, 0, 1], "min_delta": 0.45, "min_fold": 3.0},
                {"kind": "free_frac", "nodes": ["S2"], "targets": [0, 0, 1, 1], "min_delta": 0.45, "min_fold": 3.0},
            ],
        }

    if name == "neg_x2":
        xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        target_y = [round(0.95 * (1.0 - (x / 8.5) ** 2), 4) for x in xs]
        return {
            "name": "analog_neg_x2",
            "mode": "curve",
            "nodes": [
                {"name": "X", "charge_mode": "mixed", "lengths": ["4mer"]},
                {"name": "B", "charge_mode": "mixed", "lengths": ["4mer"]},
                {"name": "A", "charge_mode": "mixed", "lengths": ["4mer"]},
                {"name": "S", "charge_mode": "mixed", "lengths": ["4mer", "35mer"]},
            ],
            "edges": [
                {"u": "X", "v": "B", "pkd_min": 7.5, "pkd_max": 10.5, "require_par": True, "require_shift0": True},
                {"u": "X", "v": "A", "pkd_min": 6.8, "pkd_max": 8.5, "require_par": True, "require_shift0": True},
                {"u": "A", "v": "S", "pkd_min": 6.4, "pkd_max": 7.8, "require_par": True, "require_shift0": True},
            ],
            "hierarchies": [
                {"strong": ["X", "B"], "weak": ["X", "A"], "min_dpkd": 0.4, "min_agree": 2},
                {"strong": ["X", "A"], "weak": ["A", "S"], "min_dpkd": 0.3, "min_agree": 2},
            ],
            "off_target_max_pkd": 4.8,
            "homodimer_max_pkd": 4.5,
            "states_uM": [{"X": x, "B": 3.5, "A": 2.5, "S": 0.5} for x in xs],
            "observables": [
                {
                    "kind": "dimer_frac",
                    "nodes": ["A", "S"],
                    "ref_nodes": ["S"],
                    "targets": target_y,
                    "affine_fit": False,
                    "min_span": 0.55,
                    "max_rmse": 0.10,
                }
            ],
            "tunable_scaffolds": [
                {"node": "B", "min_uM": 1.0, "max_uM": 8.0},
                {"node": "A", "min_uM": 1.0, "max_uM": 6.0},
            ],
        }

    if name == "bandpass":
        xs = [0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 35.0, 55.0]
        target_y = [0.10, 0.25, 0.52, 0.72, 0.72, 0.48, 0.25, 0.14]
        return {
            "name": "analog_bandpass",
            "mode": "curve",
            "nodes": [
                {"name": "X", "charge_mode": "mixed", "lengths": ["4mer"]},
                {"name": "Q", "charge_mode": "mixed", "lengths": ["4mer"]},
                {"name": "S", "charge_mode": "mixed", "lengths": ["4mer", "35mer", "3mer"]},
            ],
            "edges": [
                {"u": "X", "v": "Q", "pkd_min": 6.8, "pkd_max": 8.5, "require_par": True, "require_shift0": True},
                {"u": "Q", "v": "S", "pkd_min": 5.6, "pkd_max": 6.6, "require_par": True, "require_shift0": False},
                {"u": "X", "v": "S", "pkd_min": 4.8, "pkd_max": 5.7, "require_par": True, "require_shift0": False},
            ],
            "hierarchies": [
                {"strong": ["X", "Q"], "weak": ["Q", "S"], "min_dpkd": 1.0, "min_agree": 3},
                {"strong": ["Q", "S"], "weak": ["X", "S"], "min_dpkd": 0.5, "min_agree": 2},
            ],
            "off_target_max_pkd": 4.6,
            "homodimer_max_pkd": 4.5,
            "states_uM": [{"X": x, "Q": 4.0, "S": 0.5} for x in xs],
            "observables": [
                {
                    "kind": "free_frac",
                    "nodes": ["S"],
                    "targets": target_y,
                    "affine_fit": False,
                    "min_span": 0.35,
                    "max_rmse": 0.12,
                }
            ],
            "tunable_scaffolds": [{"node": "Q", "min_uM": 1.5, "max_uM": 10.0}],
        }

    raise ValueError(f"Unknown preset: {name}")


def plot_best_design(cfg: dict, best: dict, out_path: Path, allow_5mer: bool) -> None:
    parsed = _parse_config(cfg, allow_5mer=allow_5mer)
    names = parsed["node_names"]
    n = len(names)
    pkd_mat = np.array(best["pkd_matrix"], dtype=float)
    c0_m = np.array(best["c0_uM"], dtype=float) * 1e-6
    kd_m = 10.0 ** (-pkd_mat)

    n_cols = 2 if (parsed["mode"] in ("truth_table", "curve") and c0_m.size > 0) else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6.0 * n_cols, 4.8), dpi=150)
    if n_cols == 1:
        axes = [axes]

    ax_h = axes[0]
    im = ax_h.imshow(pkd_mat, cmap="viridis", vmin=3.5, vmax=max(9.5, float(np.max(pkd_mat))))
    ax_h.set_xticks(range(n))
    ax_h.set_yticks(range(n))
    ax_h.set_xticklabels(names)
    ax_h.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            ax_h.text(j, i, f"{pkd_mat[i, j]:.1f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04, label="Predicted pKd")
    ax_h.set_title(f"{parsed['name']} | {best['design_id']} Pairwise pKd")

    if n_cols == 2:
        ax_r = axes[1]
        eq = solve_general_equilibrium(c0_m, kd_m)
        if parsed["mode"] == "truth_table":
            x_idx = np.arange(c0_m.shape[0])
            width = 0.72 / max(len(parsed["observables"]), 1)
            for o_i, obs in enumerate(parsed["observables"]):
                vals = obs["obj"].evaluate(eq, c0_m, n)
                off = (o_i - 0.5 * (len(parsed["observables"]) - 1)) * width
                lbl = f"Out{o_i + 1}: {'-'.join(names[k] for k in obs['obj'].nodes)}"
                ax_r.bar(x_idx + off, vals, width=width * 0.9, label=f"{lbl} (tgt={obs['targets']})")
            ax_r.set_xticks(x_idx)
            ax_r.set_xticklabels([f"S{k}" for k in range(len(x_idx))])
            ax_r.set_ylim(0.0, 1.05)
            ax_r.set_ylabel("Normalized output")
            mc_p = best["metrics"].get("MC_Pass_rate", np.nan)
            ax_r.set_title(f"Truth Table (Delta={best['metrics'].get('Delta', np.nan):.2f}, MC={mc_p:.0%})")
            ax_r.legend(fontsize=8)
        else:
            obs = parsed["observables"][0]
            vals = obs["obj"].evaluate(eq, c0_m, n)
            sweep_idx = int(np.argmax(np.ptp(c0_m, axis=0)))
            xs = (c0_m[:, sweep_idx] * 1e6).tolist()
            ax_r.plot(xs, obs["targets"], "k--", lw=1.8, label="Target curve")
            ax_r.plot(xs, vals, "o-", lw=2.2, color="#1b9e77", label="Designed EQTK")
            if cfg.get("log_x", False):
                ax_r.set_xscale("log")
            ax_r.set_xlabel(f"[{names[sweep_idx]}]0 (uM)")
            ax_r.set_ylabel("Normalized output")
            ax_r.set_ylim(-0.02, 1.05)
            ax_r.set_title(f"Curve Fit (RMSE={best['metrics'].get('RMSE', np.nan):.3f}, R2={best['metrics'].get('R2', np.nan):.3f})")
            ax_r.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Design coiled-coil sequences for arbitrary graphs & gates.")
    parser.add_argument("--preset", type=str, default="yes", choices=["yes", "not", "or", "and", "xor", "ortho2", "neg_x2", "bandpass"])
    parser.add_argument("--config", type=Path, default=None, help="Path to custom JSON circuit/graph config.")
    parser.add_argument("--dump-preset", type=str, default=None, help="Print preset JSON to stdout and exit.")
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB_PATH)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "graph_designs")
    parser.add_argument("--allow-5mer", action="store_true", help="Allow 35aa 5-mers where configured (e.g., XOR).")
    parser.add_argument("--tune-conc", action="store_true", help="Optimize tunable scaffold concentrations via Nelder-Mead.")
    parser.add_argument("--restarts", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top-mc-audit", type=int, default=80)
    parser.add_argument("--n-mc", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dump_preset:
        print(json.dumps(make_preset(args.dump_preset), indent=2))
        return

    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = json.load(f)
    else:
        cfg = make_preset(args.preset)

    name = cfg.get("name", args.preset)
    out_dir = args.out_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"=== Designing '{name}' (mode={cfg.get('mode', 'graph')}, allow_5mer={args.allow_5mer}, workers={args.workers}) ===")
    df, full_designs = run_graph_design(
        cfg=cfg,
        calib_path=args.calib,
        allow_5mer=args.allow_5mer,
        n_restarts=args.restarts,
        n_steps=args.steps,
        n_workers=args.workers,
        top_mc_audit=args.top_mc_audit,
        n_mc=args.n_mc,
        tune_conc=args.tune_conc,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    if df.empty:
        print("No valid candidates found.")
        return

    df.to_csv(out_dir / f"{name}_designs.csv", index=False)
    with open(out_dir / f"{name}_full.json", "w") as f:
        json.dump({"config": cfg, "designs": full_designs}, f, indent=2)

    plot_best_design(cfg, full_designs[0], out_dir / f"{name}_top1.png", allow_5mer=args.allow_5mer)

    show_cols = [c for c in [
        "Design_ID", "total_loss", "graph_loss", "Delta", "Fold", "RMSE", "R2",
        "MC_Pass_rate", "MC_R2_med", "MC_RMSE_med", "worst_off_pKd", "worst_homo_pKd", "min_slip_margin",
    ] if c in df.columns]
    print(f"\nDone in {elapsed:.1f}s. Top 10 candidates:")
    print(df[show_cols].head(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nTop-1 sequences:")
    for nm, seq in full_designs[0]["seqs"].items():
        print(f"  {nm:4s} ({len(seq)}aa, chg={df.iloc[0][f'{nm}_chg']:+d}): {seq}")
    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
