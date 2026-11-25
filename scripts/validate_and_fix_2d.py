#!/usr/bin/env python3
"""Validation & post-processing for 2D boundary designs (matched to design_2d_boundary.py v3):
1. Re-evaluate and re-plot line_s1 with v3 plotting layout (no SA re-run).
2. Check iCipa_slippage_margin, orientation, and shift on all target edges.
3. Run Monte-Carlo stress test (sigma_pKd=0.643, conc_std=0.015) on 2D designs.
4. Fast diagnostic of fit_ideal_gate on the bipartite cubic topology (YD instead of XD).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import design_2d_boundary as d2b
from src.consensus import ConsensusModel


DEFAULT_DESIGNS = [
    ROOT / "results/des_bounds_designs/line_s1/line_report.json",
    ROOT / "results/des_bounds_designs/circle_s5_kmin2/circle_report.json",
    ROOT / "results/des_bounds_designs/quadratic_s10_seed/quadratic_report.json",
    ROOT / "results/des_bounds_designs/cubic_s5/cubic_report.json",
]


def solve_eq_pos_np(c0_nM: np.ndarray, kd_nM: np.ndarray, max_newton: int = 30) -> tuple[np.ndarray, float]:
    """Damped fixed-point + positivity-preserving Newton solver."""
    a_mat = 1.0 / np.maximum(kd_nM, 1e-3)
    a_mod = a_mat.copy()
    np.fill_diagonal(a_mod, 2.0 * np.diag(a_mat))
    eye = np.eye(c0_nM.shape[1], dtype=float)[None, :, :]

    x = c0_nM.copy()
    for _ in range(40):
        x = 0.5 * x + 0.5 * (c0_nM / (1.0 + x @ a_mod))

    for _ in range(max_newton):
        factor = 1.0 + x @ a_mod
        f = x * factor - c0_nM
        if np.max(np.abs(f)) < 1e-9:
            break
        jac = x[:, :, None] * a_mod[None, :, :] + factor[:, :, None] * eye
        try:
            delta = np.linalg.solve(jac, f[..., None])[..., 0]
        except np.linalg.LinAlgError:
            delta = np.zeros_like(f)
            for b in range(len(f)):
                delta[b] = np.linalg.lstsq(jac[b], f[b], rcond=None)[0]

        x_next = x - delta
        neg_mask = np.any(x_next < 0.0, axis=1)
        if np.any(neg_mask):
            step_scale = np.ones((x.shape[0], 1), dtype=float)
            for b in np.where(neg_mask)[0]:
                bad = (delta[b] > 0.0) & (x_next[b] <= 0.0)
                if np.any(bad):
                    alpha = 0.9 * np.min(x[b, bad] / delta[b, bad])
                    step_scale[b, 0] = min(1.0, max(1e-4, float(alpha)))
            x_next = x - step_scale * delta
        x = np.where(c0_nM > 0.0, np.maximum(x_next, 1e-18), 0.0)

    final_res = x * (1.0 + x @ a_mod) - c0_nM
    max_rel_err = float(np.max(np.abs(final_res) / np.maximum(c0_nM, 1.0)))
    return x, max_rel_err


def replot_line_v3(report_path: Path) -> None:
    print(f"\n=== 1. Re-evaluating & re-plotting {report_path} with v3 ===")
    rep = json.loads(report_path.read_text())
    out_dir = report_path.parent
    name = rep["boundary"]
    s = float(rep["conc_scale"])
    nodes = rep["nodes"]
    edges = rep["edges"]

    for stem in (f"{name}_contours", f"{name}_performance"):
        for ext in (".png", ".pdf"):
            src = out_dir / f"{stem}{ext}"
            bak = out_dir / f"{stem}_v1_backup{ext}"
            if src.exists() and not bak.exists():
                shutil.copy2(src, bak)

    pkd_mat = np.array(rep["pkd_matrix"], dtype=float)
    kd_mat_nM = 10.0 ** (9.0 - pkd_mat)

    ideal_kd_mat = np.full((len(nodes), len(nodes)), 1e9, dtype=float)
    for e, val in rep["ideal_Kd_nM"].items():
        i, j = nodes.index(e[0]), nodes.index(e[1])
        ideal_kd_mat[i, j] = ideal_kd_mat[j, i] = float(val)

    ideal_eval = d2b.evaluate_full_2d(name, ideal_kd_mat, rep["ideal_total_nM"], nodes, s)
    init_eval = d2b.evaluate_full_2d(name, kd_mat_nM, rep["ideal_total_nM"], nodes, s)
    fit_eval = d2b.evaluate_full_2d(name, kd_mat_nM, rep["fitted_total_nM"], nodes, s)

    old_m = rep["fitted_metrics"]
    fit_m = fit_eval[0]
    print(f"  v1 JSON fitted : loss={old_m['loss']:.4f}, acc={old_m['accuracy']:.4%}, err={old_m['error_nM']:.2f} nM")
    print(f"  v3 re-evaluated: loss={fit_m['loss']:.4f}, acc={fit_m['accuracy']:.4%}, err={fit_m['error_nM']:.2f} nM")

    stages = {
        "Ideal": (ideal_eval[0], ideal_eval[1]),
        "Selected peptides": (init_eval[0], init_eval[1]),
        "Fitted concentrations": (fit_eval[0], fit_eval[1]),
    }
    val_grid = fit_eval[2]

    for ext in (".pdf", ".png"):
        d2b.plot_topology_and_contour(
            name,
            nodes,
            edges,
            kd_mat_nM,
            rep["fitted_total_nM"],
            fit_eval[1],
            val_grid,
            s,
            fit_eval[0],
            float(rep["worst_off_pKd"]),
            float(rep["worst_homo_pKd"]),
            out_dir / f"{name}_contours{ext}",
        )
        d2b.plot_performance_stages(name, stages, val_grid, s, out_dir / f"{name}_performance{ext}")

    print(f"  Saved v3 {name}_contours.png/.pdf and {name}_performance.png/.pdf in {out_dir}")


def check_slippage_and_mc(
    report_paths: list[Path],
    calib_path: Path,
    sigma_pkd: float,
    conc_std: float,
    n_mc: int,
    grid_n: int,
    seed: int,
) -> None:
    print(
        f"\n=== 2. Slippage & MC Stress Test "
        f"(n_mc={n_mc}, grid={grid_n}x{grid_n}, sigma_pKd={sigma_pkd:.3f}, conc_std={conc_std:.3f}) ==="
    )
    model = ConsensusModel(str(calib_path))

    for rp in report_paths:
        if not rp.exists():
            print(f"\n[skip] not found: {rp}")
            continue
        rep = json.loads(rp.read_text())
        bname = rep["boundary"]
        bkey = "cubic" if bname == "cubic_yd" else bname
        s = float(rep["conc_scale"])
        max_in = 200.0 * s
        nodes = rep["nodes"]
        edges = rep["edges"]
        pads = rep["pads"]
        pkd_mat = np.array(rep["pkd_matrix"], dtype=float)
        totals_nM = rep["fitted_total_nM"]
        fm = rep["fitted_metrics"]
        err_pct = 100.0 * fm["error_nM"] / max_in

        print(f"\n--- {rp.parent.name} ({bname}, scale={s}x) ---")
        print(
            f"  Nominal 121x121: acc={fm['accuracy']:.2%}, loss={fm['loss']:.4f}, "
            f"err={fm['error_nM']:.2f} nM ({err_pct:.2f}% of domain)"
        )
        print("  Target edges (orient / shift / iCipa_slippage_margin):")
        for e in edges:
            u, v = e[0], e[1]
            iu, iv = nodes.index(u), nodes.index(v)
            sc = model.predict_pair(pads[u], pads[v])
            pkd_pred = float(sc["pKd"])
            orient = sc.get("iCipa_best_orient", "?")
            shift_raw = sc.get("iCipa_best_shift", None)
            shift_str = f"{int(shift_raw):+d}" if shift_raw is not None else "?"
            margin = float(sc.get("iCipa_slippage_margin", float("nan")))
            flag = ""
            if orient != "PAR" or (shift_raw is not None and int(shift_raw) != 0):
                flag = "  <-- WARN: orient/shift!"
            elif np.isfinite(margin) and margin < 1.0:
                flag = "  <-- tight margin (<1.0)"
            print(
                f"    {e}: len=({len(pads[u])-1},{len(pads[v])-1})  "
                f"pKd={pkd_mat[iu, iv]:.2f} (pred={pkd_pred:.2f})  "
                f"{orient} shift={shift_str}  margin={margin:.2f}{flag}"
            )

        gx, gy = np.meshgrid(*[np.linspace(0.0, max_in, grid_n)] * 2)
        target = d2b.boundary_fn(bkey, gx, gy, max_in=max_in).ravel()
        thresh = 50.0 * s
        trans = 5.0 * s
        idx_a = nodes.index("A")

        c0_base = np.zeros((gx.size, len(nodes)), dtype=float)
        for i, node in enumerate(nodes):
            if node == "X":
                c0_base[:, i] = gx.ravel()
            elif node == "Y":
                c0_base[:, i] = gy.ravel()
            else:
                c0_base[:, i] = float(totals_nM[node])

        kd_nom = 10.0 ** (9.0 - np.clip(pkd_mat, 2.0, 12.0))
        x_nom, nom_mb = solve_eq_pos_np(c0_base, kd_nom)
        fa_nom = x_nom[:, idx_a]
        logits_nom = (fa_nom - thresh) / trans
        nom_acc = float(np.mean((fa_nom > thresh) == target))
        nom_loss = float(np.mean(np.logaddexp(0.0, logits_nom) - target * logits_nom))
        print(f"  Nominal {grid_n}x{grid_n}: acc={nom_acc:.2%}, loss={nom_loss:.4f}, max_mb_err={nom_mb:.2e}")

        rng = np.random.default_rng(seed)
        n = len(nodes)
        accs, losses, max_mb_errs = [], [], []
        for _ in range(n_mc):
            noise = rng.normal(0.0, sigma_pkd, size=(n, n))
            noise = np.triu(noise)
            noise = noise + noise.T - np.diag(np.diag(noise))
            pkd_pert = np.clip(pkd_mat + noise, 2.0, 12.0)
            kd_pert = 10.0 ** (9.0 - pkd_pert)

            if conc_std > 0.0:
                c_mult = np.clip(rng.normal(1.0, conc_std, size=(1, n)), 0.0, None)
                c0_pert = c0_base * c_mult
            else:
                c0_pert = c0_base

            x_eq, mb_err = solve_eq_pos_np(c0_pert, kd_pert)
            fa = x_eq[:, idx_a]
            logits = (fa - thresh) / trans
            accs.append(float(np.mean((fa > thresh) == target)))
            losses.append(float(np.mean(np.logaddexp(0.0, logits) - target * logits)))
            max_mb_errs.append(mb_err)

        accs_arr = np.array(accs)
        losses_arr = np.array(losses)
        print(
            f"  MC Acc : med={np.median(accs_arr):.2%}, p10={np.percentile(accs_arr, 10):.2%}, "
            f"p90={np.percentile(accs_arr, 90):.2%}, min={np.min(accs_arr):.2%}, "
            f">=90%:{np.mean(accs_arr >= 0.90):.1%}, >=85%:{np.mean(accs_arr >= 0.85):.1%}"
        )
        print(
            f"  MC Loss: med={np.median(losses_arr):.4f}, p90={np.percentile(losses_arr, 90):.4f}, "
            f"max_mb_err={np.max(max_mb_errs):.2e}"
        )


def test_cubic_yd_ideal() -> None:
    print("\n=== 3. Fast fit_ideal_gate diagnostic on bipartite cubic (YD instead of XD) ===")
    orig_edges = d2b.TOPOLOGIES["cubic"]["edges"]
    yd_edges = ("AB", "AD", "CD", "XA", "XC", "YD", "YB")
    nodes = d2b.TOPOLOGIES["cubic"]["nodes"]
    try:
        d2b.TOPOLOGIES["cubic"]["edges"] = yd_edges
        for kmin in (0.5, 1.0, 2.0):
            for sd in (42, 123):
                ideal_kd_nat, ideal_tot_nat = d2b.fit_ideal_gate("cubic", kd_min_native=kmin, kd_max_native=1000.0, seed=sd)
                s_eval = 5.0
                ideal_kd_s5 = {k: v * s_eval for k, v in ideal_kd_nat.items()}
                ideal_tot_s5 = {k: v * s_eval for k, v in ideal_tot_nat.items()}

                kd_mat = np.full((len(nodes), len(nodes)), 1e9, dtype=float)
                for e, val in ideal_kd_s5.items():
                    i, j = nodes.index(e[0]), nodes.index(e[1])
                    kd_mat[i, j] = kd_mat[j, i] = float(val)

                m, vals, _ = d2b.evaluate_full_2d("cubic", kd_mat, ideal_tot_s5, nodes, s_eval)
                pkd_s5 = {e: round(9.0 - np.log10(v * 5.0), 2) for e, v in ideal_kd_nat.items()}
                pkd_s10 = {e: round(9.0 - np.log10(v * 10.0), 2) for e, v in ideal_kd_nat.items()}
                tots_nat = {k: round(v, 1) for k, v in ideal_tot_nat.items()}
                print(f"\n  [cubic_yd | kd_min_nat={kmin} | seed={sd}]")
                print(
                    f"    acc={m['accuracy']:.2%}, loss={m['loss']:.4f}, "
                    f"err_pct={100*m['error_nM']/(200*s_eval):.2f}%, "
                    f"FreeA_nat=[{vals.min()/s_eval:.1f}, {vals.max()/s_eval:.1f}] (thr=50)"
                )
                print(f"    pKd @ scale=5x : {pkd_s5}")
                print(f"    pKd @ scale=10x: {pkd_s10}")
                print(f"    Totals (nat)   : {tots_nat}")
    finally:
        d2b.TOPOLOGIES["cubic"]["edges"] = orig_edges


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replot-line", action="store_true")
    p.add_argument("--mc-and-slippage", action="store_true")
    p.add_argument("--test-cubic-yd", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--reports", nargs="*", default=[str(x) for x in DEFAULT_DESIGNS])
    p.add_argument("--calib", default=str(ROOT / "configs/consensus_calibration.json"))
    p.add_argument("--n-mc", type=int, default=200)
    p.add_argument("--grid-n", type=int, default=51)
    p.add_argument("--conc-std", type=float, default=0.015)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not (args.replot_line or args.mc_and_slippage or args.test_cubic_yd or args.all):
        args.all = True

    if args.all or args.replot_line:
        line_rep = ROOT / "results/des_bounds_designs/line_s1/line_report.json"
        if line_rep.exists():
            replot_line_v3(line_rep)
        else:
            print(f"[skip] not found: {line_rep}")

    if args.all or args.mc_and_slippage:
        calib = json.loads(Path(args.calib).read_text())
        sigma = float(calib.get("loo_rmse", 0.643))
        check_slippage_and_mc(
            [Path(r) for r in args.reports],
            Path(args.calib),
            sigma_pkd=sigma,
            conc_std=args.conc_std,
            n_mc=args.n_mc,
            grid_n=args.grid_n,
            seed=args.seed,
        )

    if args.all or args.test_cubic_yd:
        test_cubic_yd_ideal()


if __name__ == "__main__":
    main()
