"""3D Boltz-2 + PyRosetta scorecard and 3D-EQTK verification for arbitrary graph designs."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.circuit_eq import evaluate_curve, evaluate_truth_table
from src.graph_design import _parse_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Score 3D Rosetta/Boltz results for arbitrary coiled-coil graphs.")
    parser.add_argument("--full-json", type=Path, required=True, help="Path to <name>_full.json from design_graph.py")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results" / "3d_inputs" / "manifest.csv")
    parser.add_argument("--rosetta-summary", type=Path, default=ROOT / "results" / "rosetta_3d_summary.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--ref-homo-dg", type=float, default=None)
    parser.add_argument("--homo-dg-tol", type=float, default=2.0)
    parser.add_argument("--min-gap-hier", type=float, default=2.0, help="Min Rosetta dG gap (REU) for hierarchy pairs.")
    parser.add_argument("--min-gap-off", type=float, default=2.5, help="Min Rosetta dG gap between weakest edge and worst off-target.")
    parser.add_argument("--max-dimer-unsat", type=float, default=1.5)
    parser.add_argument("--min-trimer-contacts", type=int, default=25)
    parser.add_argument("--min-trimer-iptm", type=float, default=0.70)
    parser.add_argument("--max-clean-trimer-unsat", type=float, default=1.0)
    parser.add_argument("--min-clean-trimer-pack", type=float, default=0.60)
    parser.add_argument("--min-clean-trimer-sc", type=float, default=0.78)
    parser.add_argument("--min-3d-delta", type=float, default=0.30, help="Min 3D-EQTK Delta for truth_table circuits.")
    parser.add_argument("--min-3d-fold", type=float, default=2.0, help="Min 3D-EQTK Fold for truth_table circuits.")
    parser.add_argument("--max-3d-rmse", type=float, default=0.15, help="Max 3D-EQTK RMSE for curve circuits.")
    parser.add_argument("--min-3d-r2", type=float, default=0.65, help="Min 3D-EQTK R2 for curve circuits.")
    args = parser.parse_args()

    with open(args.full_json, "r") as f:
        bundle = json.load(f)
    cfg = bundle["config"]
    parsed = _parse_config(cfg, allow_5mer=True)
    names = parsed["node_names"]
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    n = len(names)
    design_lookup = {d["design_id"]: d for d in bundle["designs"]}

    man = pd.read_csv(args.manifest)
    ros = pd.read_csv(args.rosetta_summary)
    ros_idx = ros.set_index("job_id").to_dict(orient="index")

    # 1. Calibrate Rosetta dG_separated -> log10(Kd) using experimental controls
    ctrl_rows = man[(man["design_id"] == "CTRL") & man["exp_log10_kd"].notna()][["job_id", "role", "exp_log10_kd"]].drop_duplicates("job_id")
    c_data = []
    ref_homo_dg = args.ref_homo_dg
    for _, cr in ctrl_rows.iterrows():
        jid = cr["job_id"]
        if jid in ros_idx and pd.notna(ros_idx[jid].get("dG_separated")):
            dg = float(ros_idx[jid]["dG_separated"])
            c_data.append({"job_id": jid, "role": cr["role"], "exp_log10Kd": float(cr["exp_log10_kd"]), "dG": dg})
            if ref_homo_dg is None and cr["role"] == "BN4:BN4":
                ref_homo_dg = dg

    cdf = pd.DataFrame(c_data)
    has_calib = len(cdf) >= 3
    sl_ros, in_ros, r_ros, rho_ros = np.nan, np.nan, np.nan, np.nan
    if has_calib:
        sl_ros, in_ros, r_ros, _, _ = stats.linregress(cdf["dG"], cdf["exp_log10Kd"])
        rho_ros, _ = stats.spearmanr(cdf["dG"], cdf["exp_log10Kd"])
        print(f"Rosetta calibration (n={len(cdf)}): r={r_ros:+.3f}, rho={rho_ros:+.3f} | log10(Kd) = {sl_ros:.3f}*dG {in_ros:+.2f}")
    else:
        print("Warning: <3 Rosetta controls found; 3D-EQTK verification skipped (reporting Rosetta REU gaps only).")

    des_man = man[man["design_id"] != "CTRL"].copy()
    if "circuit" in des_man.columns:
        sub_c = des_man[des_man["circuit"] == parsed["name"]]
        if not sub_c.empty:
            des_man = sub_c

    master = []
    for dtag, grp in des_man.groupby("design_id", sort=False):
        if dtag not in design_lookup:
            continue
        des_obj = design_lookup[dtag]

        # Collect 2-chain Rosetta values indexed by sorted (u, v) and verify job completeness
        pair_dg: Dict[Tuple[int, int], float] = {}
        pair_unsat: Dict[Tuple[int, int], float] = {}
        trimer_rows = []
        expected_jids = set(grp["job_id"].astype(str))
        expected_trimer_jids = set(grp[grp["pair_type"] == "trimer"]["job_id"].astype(str))
        found_jids = set()

        for _, mr in grp.iterrows():
            jid = str(mr["job_id"])
            if jid not in ros_idx:
                continue
            found_jids.add(jid)
            rdat = ros_idx[jid]
            ptype = str(mr.get("pair_type", ""))
            if ptype == "trimer":
                trimer_rows.append((str(mr["role"]), rdat))
            elif str(mr["u"]) in name_to_idx and str(mr["v"]) in name_to_idx:
                u = name_to_idx[str(mr["u"])]
                v = name_to_idx[str(mr["v"])]
                key = (min(u, v), max(u, v))
                if pd.notna(rdat.get("dG_separated")):
                    pair_dg[key] = float(rdat["dG_separated"])
                if pd.notna(rdat.get("delta_unsatHbonds")):
                    pair_unsat[key] = float(rdat["delta_unsatHbonds"])

        if not pair_dg:
            continue

        all_targets_present = all((ed["u"], ed["v"]) in pair_dg for ed in parsed["edges"])
        all_hiers_present = all((h["su"], h["sv"]) in pair_dg and (h["wu"], h["wv"]) in pair_dg for h in parsed["hierarchies"])
        complete_3d = bool(all_targets_present and all_hiers_present and expected_jids.issubset(found_jids))

        target_dgs = [pair_dg[(ed["u"], ed["v"])] for ed in parsed["edges"] if (ed["u"], ed["v"]) in pair_dg]
        off_dgs = [
            dg for (u, v), dg in pair_dg.items()
            if u != v and (u, v) not in parsed["target_pairs"] and (u, v) not in parsed["ignored_pairs"]
        ]
        homo_dgs = [
            dg for (u, v), dg in pair_dg.items()
            if u == v and (u, v) not in parsed["target_pairs"]
        ]

        weakest_target_dg = max(target_dgs) if target_dgs else np.nan
        best_off_dg = min(off_dgs) if off_dgs else np.nan
        best_homo_dg = min(homo_dgs) if homo_dgs else np.nan
        max_target_unsat = max(
            [pair_unsat[(ed["u"], ed["v"])] for ed in parsed["edges"] if (ed["u"], ed["v"]) in pair_unsat],
            default=np.nan,
        )

        # Hierarchy gaps (dG_weak - dG_strong > 0 since stronger binding has more negative dG)
        hier_gaps = []
        for h in parsed["hierarchies"]:
            ks = (h["su"], h["sv"])
            kw = (h["wu"], h["wv"])
            if ks in pair_dg and kw in pair_dg:
                hier_gaps.append(pair_dg[kw] - pair_dg[ks])
        min_hier_gap = min(hier_gaps) if hier_gaps else np.nan

        gap_target_off = (best_off_dg - weakest_target_dg) if (pd.notna(best_off_dg) and pd.notna(weakest_target_dg)) else np.nan
        gap_target_homo = (best_homo_dg - weakest_target_dg) if (pd.notna(best_homo_dg) and pd.notna(weakest_target_dg)) else np.nan

        # Trimer trap check (fails if any expected trimer is missing or forms a clean assembled trap)
        if not expected_trimer_jids.issubset(found_jids):
            trimer_safe = False
        else:
            trimer_traps = []
            for _trole, rdat in trimer_rows:
                c_cont = rdat.get("C_to_AB_contacts")
                m_piptm = rdat.get("min_pair_iptm")
                t_unsat = rdat.get("delta_unsatHbonds")
                t_pack = rdat.get("packstat")
                t_sc = rdat.get("sc_value")
                if any(pd.isna(v) for v in (c_cont, m_piptm, t_unsat, t_pack, t_sc)):
                    trimer_traps.append(True)
                    continue
                assembled = (float(c_cont) >= args.min_trimer_contacts) and (float(m_piptm) >= args.min_trimer_iptm)
                clean_core = (
                    (float(t_unsat) <= args.max_clean_trimer_unsat)
                    and (float(t_pack) >= args.min_clean_trimer_pack)
                    and (float(t_sc) >= args.min_clean_trimer_sc)
                )
                trimer_traps.append(bool(assembled and clean_core))
            trimer_safe = not any(trimer_traps)

        # Optional 3D-EQTK verification using calibrated Rosetta Kd matrix
        eq_3d_metric = np.nan
        eq_3d_fold_or_r2 = np.nan
        eq_3d_ok = True
        eq_3d_bonus = 0.0
        if has_calib and parsed["mode"] in ("truth_table", "curve") and parsed["observables"]:
            c0_m = np.array(des_obj["c0_uM"], dtype=float) * 1e-6
            # Fallback to 1D prediction for any unexported ultra-weak off-target pair
            log10_kd_3d = -np.array(des_obj["pkd_matrix"], dtype=float)
            for (u, v), dg in pair_dg.items():
                l10 = float(sl_ros * dg + in_ros)
                log10_kd_3d[u, v] = l10
                log10_kd_3d[v, u] = l10
            kd_3d_m = 10.0 ** np.minimum(log10_kd_3d, -2.0)

            if parsed["mode"] == "truth_table":
                deltas, folds = [], []
                for obs in parsed["observables"]:
                    tt = evaluate_truth_table(kd_3d_m, c0_m, obs["obj"], obs["targets"])
                    deltas.append(tt["Delta"])
                    folds.append(tt["Fold"])
                eq_3d_metric = float(min(deltas))
                eq_3d_fold_or_r2 = float(min(folds))
                eq_3d_ok = (eq_3d_metric >= args.min_3d_delta) and (eq_3d_fold_or_r2 >= args.min_3d_fold)
                eq_3d_bonus = 8.0 * eq_3d_metric
            else:
                rmses, r2s = [], []
                for obs in parsed["observables"]:
                    cv = evaluate_curve(kd_3d_m, c0_m, obs["obj"], obs["targets"], affine_fit=obs["affine_fit"])
                    rmses.append(cv["RMSE"])
                    r2s.append(cv["R2"])
                eq_3d_metric = float(max(rmses))
                eq_3d_fold_or_r2 = float(min(r2s))
                eq_3d_ok = (eq_3d_metric <= args.max_3d_rmse) and (eq_3d_fold_or_r2 >= args.min_3d_r2)
                eq_3d_bonus = 6.0 * max(eq_3d_fold_or_r2, -1.0) - 15.0 * eq_3d_metric

        homo_ok = True
        if ref_homo_dg is not None and pd.notna(best_homo_dg):
            homo_ok = best_homo_dg >= (ref_homo_dg - args.homo_dg_tol)

        passed_3d = bool(
            complete_3d
            and trimer_safe
            and homo_ok
            and eq_3d_ok
            and (not parsed["hierarchies"] or (pd.notna(min_hier_gap) and min_hier_gap >= args.min_gap_hier))
            and (not off_dgs or (pd.notna(gap_target_off) and gap_target_off >= args.min_gap_off))
            and (pd.notna(max_target_unsat) and max_target_unsat <= args.max_dimer_unsat)
        )

        score_3d = (
            2.0 * (min_hier_gap if pd.notna(min_hier_gap) else 0.0)
            + 1.5 * (gap_target_off if pd.notna(gap_target_off) else 0.0)
            + 1.0 * (gap_target_homo if pd.notna(gap_target_homo) else 0.0)
            - 2.0 * (max_target_unsat if pd.notna(max_target_unsat) else 0.0)
            + 10.0 * float(des_obj["metrics"].get("MC_Pass_rate", 0.0))
            + eq_3d_bonus
        )

        rec = {
            "Design_ID": dtag,
            "Passed_3D": passed_3d,
            "Complete_3D": complete_3d,
            "Trimer_Safe": trimer_safe,
            "Score_3D": score_3d,
            "1D_MC_Pass": des_obj["metrics"].get("MC_Pass_rate", np.nan),
            "Weakest_Target_dG": weakest_target_dg,
            "Best_OffTarget_dG": best_off_dg,
            "Best_Homo_dG": best_homo_dg,
            "Min_Hier_Gap_REU": min_hier_gap,
            "Gap_Target_Off_REU": gap_target_off,
            "Max_Target_unsatHB": max_target_unsat,
        }
        if parsed["mode"] == "truth_table":
            rec["3D_EQTK_Delta"] = eq_3d_metric
            rec["3D_EQTK_Fold"] = eq_3d_fold_or_r2
        elif parsed["mode"] == "curve":
            rec["3D_EQTK_RMSE"] = eq_3d_metric
            rec["3D_EQTK_R2"] = eq_3d_fold_or_r2

        for ed in parsed["edges"]:
            rec[f"dG_{ed['u_name']}:{ed['v_name']}"] = pair_dg.get((ed["u"], ed["v"]), np.nan)
        for nm, sq in des_obj["seqs"].items():
            rec[f"{nm}_seq"] = sq
        master.append(rec)

    mdf = pd.DataFrame(master)
    if mdf.empty:
        print("No matching design Rosetta scores found.")
        return

    mdf = mdf.sort_values(["Passed_3D", "Score_3D"], ascending=[False, False]).reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = parsed["name"]
    out_csv = args.out_dir / f"{tag}_3d_scorecard.csv"
    mdf.to_csv(out_csv, index=False)

    show_cols = [c for c in [
        "Design_ID", "Passed_3D", "Complete_3D", "Trimer_Safe", "1D_MC_Pass", "Weakest_Target_dG",
        "Best_OffTarget_dG", "Best_Homo_dG", "Min_Hier_Gap_REU", "Gap_Target_Off_REU",
        "3D_EQTK_Delta", "3D_EQTK_Fold", "3D_EQTK_RMSE", "3D_EQTK_R2", "Score_3D",
    ] if c in mdf.columns]
    print("\n" + mdf[show_cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # Plot summary figure
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=150)
    ax = axes[0]
    if has_calib:
        ax.scatter(cdf["dG"], cdf["exp_log10Kd"], s=70, color="#2b8cbe", zorder=4, label="Exp. controls")
        xs = np.linspace(cdf["dG"].min() - 3, cdf["dG"].max() + 3, 50)
        ax.plot(xs, sl_ros * xs + in_ros, "k--", lw=1.2, label=f"r={r_ros:.2f}, rho={rho_ros:.2f}")
    top1 = mdf.iloc[0]
    for ed in parsed["edges"]:
        col = f"dG_{ed['u_name']}:{ed['v_name']}"
        if pd.notna(top1.get(col)):
            ax.axvline(top1[col], ls=":", lw=1.5, label=f"{ed['u_name']}:{ed['v_name']} ({top1[col]:.1f})")
    if pd.notna(top1["Best_OffTarget_dG"]):
        ax.axvline(top1["Best_OffTarget_dG"], color="#e41a1c", ls="--", lw=1.5, label=f"Worst off ({top1['Best_OffTarget_dG']:.1f})")
    ax.set_title(f"{tag} | Rosetta Calibration & {top1['Design_ID']}")
    ax.set_xlabel("Rosetta dG_separated (REU)")
    ax.set_ylabel("Experimental log10(Kd [M])")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    top_k = mdf.head(5)
    x_pos = np.arange(len(top_k))
    ax.bar(x_pos - 0.25, -top_k["Weakest_Target_dG"], width=0.24, color="#1b9e77", label="Weakest target edge")
    if top_k["Best_OffTarget_dG"].notna().any():
        ax.bar(x_pos, -top_k["Best_OffTarget_dG"], width=0.24, color="#e41a1c", label="Strongest off-target")
    if top_k["Best_Homo_dG"].notna().any():
        ax.bar(x_pos + 0.25, -top_k["Best_Homo_dG"], width=0.24, color="#377eb8", label="Strongest homodimer")
    if ref_homo_dg is not None:
        ax.axhline(-ref_homo_dg, color="purple", ls="--", lw=1, label=f"BN4 homo ({ref_homo_dg:.1f})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_k["Design_ID"])
    ax.set_ylabel("-dG_separated (REU, higher = stronger)")
    ax.set_title("Top Candidates: Target vs Off-target Gap")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out_dir / f"{tag}_3d_summary.png")
    plt.close(fig)
    print(f"\nSaved scorecard to {out_csv}")


if __name__ == "__main__":
    main()
