"""3D Boltz-2 + PyRosetta physical scorecard for arbitrary coiled-coil graph designs."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
    parser = argparse.ArgumentParser(description="Score 3D Rosetta/Boltz physical filters for arbitrary coiled-coil graphs.")
    parser.add_argument("--full-json", type=Path, required=True, help="Path to <name>_full.json from design_graph.py")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results" / "3d_inputs" / "manifest.csv")
    parser.add_argument("--rosetta-summary", type=Path, default=ROOT / "results" / "rosetta_3d_summary.csv")
    parser.add_argument("--boltz-summary", type=Path, default=ROOT / "results" / "boltz_3d_summary.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    # Target dimer structural gates (benchmarked on 14 experimental controls)
    parser.add_argument("--min-cos-ab", type=float, default=0.50)
    parser.add_argument("--min-target-iptm", type=float, default=0.90)
    parser.add_argument("--min-helix-frac", type=float, default=0.85)
    parser.add_argument("--max-asn-ca", type=float, default=7.5)
    parser.add_argument("--max-dimer-unsat", type=float, default=2.0)
    parser.add_argument("--min-target-pack", type=float, default=0.595)
    parser.add_argument("--min-target-sc", type=float, default=0.80)
    # Non-target / hierarchy Rosetta REU & SASA-normalized gates
    parser.add_argument("--min-gap-hier", type=float, default=-2.0)
    parser.add_argument("--min-gap-off", type=float, default=-2.0)
    parser.add_argument("--min-gap-homo", type=float, default=-2.0)
    parser.add_argument("--min-gap-sasa", type=float, default=-0.15)
    parser.add_argument("--max-5mer-nontarget-dg", type=float, default=-64.0)
    parser.add_argument("--max-5mer-sasa-dg", type=float, default=-3.80)
    # Trimer trap detector
    parser.add_argument("--min-trimer-contacts", type=int, default=25)
    parser.add_argument("--min-trimer-iptm", type=float, default=0.70)
    parser.add_argument("--max-clean-trimer-unsat", type=float, default=1.5)
    parser.add_argument("--min-clean-trimer-pack", type=float, default=0.58)
    parser.add_argument("--min-clean-trimer-sc", type=float, default=0.76)
    args = parser.parse_args()

    with open(args.full_json, "r") as f:
        bundle = json.load(f)
    cfg = bundle["config"]
    parsed = _parse_config(cfg, allow_5mer=True)
    names = parsed["node_names"]
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    design_lookup = {d["design_id"]: d for d in bundle["designs"]}

    man = pd.read_csv(args.manifest)
    ros = pd.read_csv(args.rosetta_summary)
    if args.boltz_summary is not None and args.boltz_summary.exists():
        boltz = pd.read_csv(args.boltz_summary)
        boltz_cols = ["job_id"] + [
            c for c in ("iptm", "plddt", "helix_frac", "cos_AB", "min_Asn_CA", "min_pair_iptm", "C_to_AB_contacts")
            if c in boltz.columns and c not in ros.columns
        ]
        if len(boltz_cols) > 1:
            ros = ros.merge(boltz[boltz_cols].drop_duplicates("job_id"), on="job_id", how="left")
    ros_idx = ros.drop_duplicates("job_id").set_index("job_id").to_dict(orient="index")

    ctrl_rows = man[(man["design_id"] == "CTRL") & man["exp_log10_kd"].notna()][["job_id", "role", "exp_log10_kd"]].drop_duplicates("job_id")
    c_data = []
    ref_homo_dg = None
    for _, cr in ctrl_rows.iterrows():
        jid = cr["job_id"]
        if jid in ros_idx and pd.notna(ros_idx[jid].get("dG_separated")):
            dg = float(ros_idx[jid]["dG_separated"])
            c_data.append({"job_id": jid, "role": cr["role"], "exp_log10Kd": float(cr["exp_log10_kd"]), "dG": dg})
            if cr["role"] == "BN4:BN4":
                ref_homo_dg = dg

    cdf = pd.DataFrame(c_data)
    has_calib = len(cdf) >= 3
    sl_ros, in_ros, r_ros, rho_ros = np.nan, np.nan, np.nan, np.nan
    if has_calib:
        sl_ros, in_ros, r_ros, _, _ = stats.linregress(cdf["dG"], cdf["exp_log10Kd"])
        rho_ros, _ = stats.spearmanr(cdf["dG"], cdf["exp_log10Kd"])
        print(f"Rosetta calibration (n={len(cdf)}, diag only): r={r_ros:+.3f}, rho={rho_ros:+.3f}")

    des_man = man[man["design_id"] != "CTRL"].copy()
    if "circuit" in des_man.columns:
        des_man = des_man[des_man["circuit"] == parsed["name"]]
    if des_man.empty:
        print(f"No manifest entries for circuit {parsed['name']}; skipping.")
        return

    master = []
    for dtag, grp in des_man.groupby("design_id", sort=False):
        if dtag not in design_lookup:
            continue
        des_obj = design_lookup[dtag]

        def _seq_len(idx: int) -> int:
            return len(str(des_obj["seqs"][names[idx]]).strip().lstrip("-"))

        def _pair_sum_len(k: Tuple[int, int]) -> int:
            return _seq_len(k[0]) + _seq_len(k[1])

        pair_data: Dict[Tuple[int, int], Dict[str, float]] = {}
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
                pair_data[(min(u, v), max(u, v))] = rdat

        if not pair_data:
            continue

        pair_dg = {k: float(v["dG_separated"]) for k, v in pair_data.items() if pd.notna(v.get("dG_separated"))}
        pair_sasa = {k: float(v["dG_per_SASA100"]) for k, v in pair_data.items() if pd.notna(v.get("dG_per_SASA100"))}
        all_targets_present = all((ed["u"], ed["v"]) in pair_dg for ed in parsed["edges"])
        all_hiers_present = all((h["su"], h["sv"]) in pair_dg and (h["wu"], h["wv"]) in pair_dg for h in parsed["hierarchies"])
        complete_3d = bool(all_targets_present and all_hiers_present and expected_jids.issubset(found_jids))

        fail_reasons: List[str] = []
        if not complete_3d:
            fail_reasons.append("INCOMPLETE_3D")

        target_unsats, target_packs, target_scs = [], [], []
        anti_targets = 0
        for ed in parsed["edges"]:
            key = (ed["u"], ed["v"])
            ename = f"{ed['u_name']}:{ed['v_name']}"
            rdat = pair_data.get(key)
            if rdat is None:
                continue
            cos_v = rdat.get("cos_AB")
            iptm_v = rdat.get("iptm")
            hel_v = rdat.get("helix_frac")
            asn_v = rdat.get("min_Asn_CA")
            uns_v = rdat.get("delta_unsatHbonds")
            pck_v = rdat.get("packstat")
            sc_v = rdat.get("sc_value")

            if pd.notna(cos_v) and float(cos_v) < args.min_cos_ab:
                anti_targets += 1
                fail_reasons.append(f"ANTI({ename}={float(cos_v):.2f})")
            if pd.notna(iptm_v) and float(iptm_v) < args.min_target_iptm:
                fail_reasons.append(f"LOW_IPTM({ename}={float(iptm_v):.2f})")
            if pd.notna(hel_v) and float(hel_v) < args.min_helix_frac:
                fail_reasons.append(f"LOW_HELIX({ename}={float(hel_v):.2f})")
            if pd.notna(asn_v) and float(asn_v) > args.max_asn_ca:
                fail_reasons.append(f"BAD_ASN({ename}={float(asn_v):.1f})")
            if pd.notna(uns_v):
                target_unsats.append(float(uns_v))
                if float(uns_v) > args.max_dimer_unsat:
                    fail_reasons.append(f"UNSAT({ename}={float(uns_v):.0f})")
            if pd.notna(pck_v):
                target_packs.append(float(pck_v))
                if float(pck_v) < args.min_target_pack:
                    fail_reasons.append(f"LOW_PACK({ename}={float(pck_v):.3f})")
            if pd.notna(sc_v):
                target_scs.append(float(sc_v))
                if float(sc_v) < args.min_target_sc:
                    fail_reasons.append(f"LOW_SC({ename}={float(sc_v):.2f})")

        target_keys = [(ed["u"], ed["v"]) for ed in parsed["edges"] if (ed["u"], ed["v"]) in pair_dg]
        off_keys = [(u, v) for (u, v) in pair_dg if u != v and (u, v) not in parsed["target_pairs"] and (u, v) not in parsed["ignored_pairs"]]
        homo_keys = [(u, v) for (u, v) in pair_dg if u == v and (u, v) not in parsed["target_pairs"]]

        weak_t_key = max(target_keys, key=lambda k: pair_dg[k]) if target_keys else None
        best_off_key = min(off_keys, key=lambda k: pair_dg[k]) if off_keys else None
        best_homo_key = min(homo_keys, key=lambda k: pair_dg[k]) if homo_keys else None

        weakest_target_dg = pair_dg[weak_t_key] if weak_t_key else np.nan
        best_off_dg = pair_dg[best_off_key] if best_off_key else np.nan
        best_homo_dg = pair_dg[best_homo_key] if best_homo_key else np.nan

        weak_t_sasa = max([pair_sasa[k] for k in target_keys if k in pair_sasa], default=np.nan)
        best_off_sasa = min([pair_sasa[k] for k in off_keys if k in pair_sasa], default=np.nan)
        best_homo_sasa = min([pair_sasa[k] for k in homo_keys if k in pair_sasa], default=np.nan)

        gap_sasa_off = (best_off_sasa - weak_t_sasa) if (pd.notna(best_off_sasa) and pd.notna(weak_t_sasa)) else np.nan
        gap_sasa_homo = (best_homo_sasa - weak_t_sasa) if (pd.notna(best_homo_sasa) and pd.notna(weak_t_sasa)) else np.nan

        max_target_unsat = max(target_unsats) if target_unsats else np.nan
        min_target_pack = min(target_packs) if target_packs else np.nan
        min_target_sc = min(target_scs) if target_scs else np.nan

        hier_gaps = []
        for h in parsed["hierarchies"]:
            ks, kw = (h["su"], h["sv"]), (h["wu"], h["wv"])
            if ks in pair_dg and kw in pair_dg:
                hier_gaps.append(pair_dg[kw] - pair_dg[ks])
        min_hier_gap = min(hier_gaps) if hier_gaps else np.nan

        gap_target_off = (best_off_dg - weakest_target_dg) if (pd.notna(best_off_dg) and pd.notna(weakest_target_dg)) else np.nan
        gap_target_homo = (best_homo_dg - weakest_target_dg) if (pd.notna(best_homo_dg) and pd.notna(weakest_target_dg)) else np.nan

        if parsed["hierarchies"] and (pd.isna(min_hier_gap) or min_hier_gap < args.min_gap_hier):
            fail_reasons.append(f"HIER_GAP({min_hier_gap:.1f})")

        # Off-target gap: raw REU for same-length pairs, SASA-normalized for mixed-length pairs
        if off_keys:
            same_len_off = weak_t_key and best_off_key and abs(_pair_sum_len(best_off_key) - _pair_sum_len(weak_t_key)) <= 4
            if pd.notna(gap_sasa_off):
                if gap_sasa_off < args.min_gap_sasa:
                    fail_reasons.append(f"OFF_SASA({gap_sasa_off:+.2f})")
                elif same_len_off and gap_target_off < args.min_gap_off:
                    fail_reasons.append(f"OFF_GAP({gap_target_off:.1f})")
            elif pd.isna(gap_target_off) or gap_target_off < args.min_gap_off:
                fail_reasons.append(f"OFF_GAP({gap_target_off:.1f})")

        # Homodimer gap: raw REU for same-length pairs, SASA-normalized for mixed-length pairs
        if homo_keys:
            same_len_homo = weak_t_key and best_homo_key and abs(_pair_sum_len(best_homo_key) - _pair_sum_len(weak_t_key)) <= 4
            if pd.notna(gap_sasa_homo):
                if gap_sasa_homo < args.min_gap_sasa:
                    fail_reasons.append(f"HOMO_SASA({gap_sasa_homo:+.2f})")
                elif same_len_homo and gap_target_homo < args.min_gap_homo:
                    fail_reasons.append(f"HOMO_GAP({gap_target_homo:.1f})")
            elif pd.isna(gap_target_homo) or gap_target_homo < args.min_gap_homo:
                fail_reasons.append(f"HOMO_GAP({gap_target_homo:.1f})")

        # 5-mer monster check: only flag if BOTH raw dG <= -64 AND SASA energy density <= -3.80 (above BN4:BN4=-3.62)
        for u, v in off_keys + homo_keys:
            rdat = pair_data[(u, v)]
            dg = pair_dg[(u, v)]
            sdg = pair_sasa.get((u, v), np.nan)
            lmax = max(_seq_len(u), _seq_len(v))
            iptm_v = rdat.get("iptm")
            sc_v = rdat.get("sc_value")
            sasa_hot = pd.isna(sdg) or (sdg <= args.max_5mer_sasa_dg)
            if lmax >= 32 and dg <= args.max_5mer_nontarget_dg and sasa_hot and (pd.isna(iptm_v) or float(iptm_v) >= 0.85) and (pd.isna(sc_v) or float(sc_v) >= 0.80):
                tag_nt = f"{names[u]}:{names[v]}" if u != v else f"homo_{names[u]}"
                fail_reasons.append(f"5MER_TRAP({tag_nt}={dg:.1f},{sdg:.2f})")

        # Trimer trap check
        if not expected_trimer_jids.issubset(found_jids):
            trimer_safe = False
            fail_reasons.append("MISSING_TRIMERS")
        else:
            trimer_traps = []
            for trole, rdat in trimer_rows:
                c_cont = rdat.get("C_to_AB_contacts")
                m_piptm = rdat.get("min_pair_iptm")
                t_unsat = rdat.get("delta_unsatHbonds")
                t_pack = rdat.get("packstat")
                t_sc = rdat.get("sc_value")
                if any(pd.isna(v) for v in (c_cont, m_piptm, t_unsat, t_pack, t_sc)):
                    trimer_traps.append(True)
                    fail_reasons.append(f"TRIMER_NAN({trole})")
                    continue
                assembled = (float(c_cont) >= args.min_trimer_contacts) and (float(m_piptm) >= args.min_trimer_iptm)
                clean_core = (
                    (float(t_unsat) <= args.max_clean_trimer_unsat)
                    and (float(t_pack) >= args.min_clean_trimer_pack)
                    and (float(t_sc) >= args.min_clean_trimer_sc)
                )
                if assembled and clean_core:
                    trimer_traps.append(True)
                    fail_reasons.append(f"TRIMER_TRAP({trole})")
                else:
                    trimer_traps.append(False)
            trimer_safe = not any(trimer_traps)

        # Diagnostic 3D-EQTK only
        eq_3d_metric, eq_3d_fold_or_r2 = np.nan, np.nan
        if has_calib and parsed["mode"] in ("truth_table", "curve") and parsed["observables"]:
            c0_m = np.array(des_obj["c0_uM"], dtype=float) * 1e-6
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
                eq_3d_metric, eq_3d_fold_or_r2 = float(min(deltas)), float(min(folds))
            else:
                rmses, r2s = [], []
                for obs in parsed["observables"]:
                    cv = evaluate_curve(kd_3d_m, c0_m, obs["obj"], obs["targets"], affine_fit=obs["affine_fit"])
                    rmses.append(cv["RMSE"])
                    r2s.append(cv["R2"])
                eq_3d_metric, eq_3d_fold_or_r2 = float(max(rmses)), float(min(r2s))

        passed_3d = len(fail_reasons) == 0

        mc_val = des_obj["metrics"].get("MC_Pass_rate", 0.0)
        mc_clean = float(mc_val) if (mc_val is not None and np.isfinite(mc_val)) else 0.0
        score_3d = (
            15.0 * (gap_sasa_off if pd.notna(gap_sasa_off) else 0.0)
            + 15.0 * (gap_sasa_homo if pd.notna(gap_sasa_homo) else 0.0)
            + 1.0 * (min_hier_gap if pd.notna(min_hier_gap) else 0.0)
            - 2.0 * (max_target_unsat if pd.notna(max_target_unsat) else 0.0)
            + 15.0 * (min_target_pack - 0.60 if pd.notna(min_target_pack) else 0.0)
            + 15.0 * mc_clean
        )

        rec = {
            "Design_ID": dtag,
            "Passed_3D": passed_3d,
            "Complete_3D": complete_3d,
            "Trimer_Safe": trimer_safe,
            "Anti_Targets": anti_targets,
            "Score_3D": score_3d,
            "1D_MC_Pass": des_obj["metrics"].get("MC_Pass_rate", np.nan),
            "Weakest_Target_dG": weakest_target_dg,
            "Best_OffTarget_dG": best_off_dg,
            "Best_Homo_dG": best_homo_dg,
            "Gap_Target_Off_REU": gap_target_off,
            "Gap_Target_Homo_REU": gap_target_homo,
            "Gap_SASA_Off": gap_sasa_off,
            "Gap_SASA_Homo": gap_sasa_homo,
            "Min_Hier_Gap_REU": min_hier_gap,
            "Max_Target_unsatHB": max_target_unsat,
            "Min_Target_pack": min_target_pack,
            "Min_Target_sc": min_target_sc,
            "Fail_Reasons": "; ".join(fail_reasons) if fail_reasons else "OK",
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
        "Design_ID", "Passed_3D", "Trimer_Safe", "Anti_Targets", "1D_MC_Pass",
        "Gap_Target_Off_REU", "Gap_Target_Homo_REU", "Gap_SASA_Off", "Gap_SASA_Homo",
        "Min_Hier_Gap_REU", "Max_Target_unsatHB", "Min_Target_pack", "Min_Target_sc", "Fail_Reasons",
    ] if c in mdf.columns]
    print("\n" + mdf[show_cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

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
    ax.set_title(f"{tag} | Rosetta Controls & {top1['Design_ID']} ({'PASS' if top1['Passed_3D'] else 'FAIL'})")
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
    ax.set_title("Top Candidates: Target vs Off/Homo Gap")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out_dir / f"{tag}_3d_summary.png")
    plt.close(fig)
    print(f"\nSaved scorecard to {out_csv}")


if __name__ == "__main__":
    main()
