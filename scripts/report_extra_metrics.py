#!/usr/bin/env python3
"""Post-hoc metrics for the committed designs. Does not re-run any design.

1. For every results/des_bounds_designs/*/*_report.json: recompute the 121x121
   classification from the saved pkd_matrix + fitted_total_nM with EQTK, verify it
   against the stored numbers, and add a "fitted_metrics_extra" block with the
   class-imbalance context (majority baseline, balanced accuracy, MCC,
   sensitivity/specificity, contour error as % of domain).
2. For every results/graph_designs/*/*_full.json: report the uncensored
   Fold = min_ON / max_OFF next to the floored Fold = min_ON / max(max_OFF, 0.005)
   into results/fold_uncensored.csv, so censored table entries can be labelled.

Needs only numpy, scipy, eqtk and contourpy; it does not import the repo's design
code, so it doubles as an independent check of the published numbers.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import eqtk
from contourpy import contour_generator
from scipy.spatial import cKDTree

FOLD_FLOOR = 0.005


def boundary_fn(name, x, y, max_in):
    if name == "line":
        return y > x
    if name == "circle":
        return x ** 2 + (y - max_in) ** 2 > max_in ** 2
    if name == "quadratic":
        return y > 4.0 * x * (1.0 - x / max_in)
    if name == "cubic":
        return y > 0.5 * max_in * (1.0 + (2.0 * x / max_in - 1.0) ** 3)
    raise ValueError(name)


def curve_points(name, max_in, n=1001):
    t = np.linspace(0.0, 1.0, n)
    if name == "line":
        return max_in * t, max_in * t
    if name == "circle":
        return max_in * np.sin(np.pi * t / 2), max_in * (1 - np.cos(np.pi * t / 2))
    if name == "quadratic":
        x = max_in * t
        return x, 4.0 * x * (1 - x / max_in)
    if name == "cubic":
        x = max_in * t
        return x, 0.5 * max_in * (1 + (2 * x / max_in - 1) ** 3)
    raise ValueError(name)


def _pairs(n):
    return [(i, j) for i in range(n) for j in range(i, n)]


def solve_free_nM(c0_nM, kd_nM):
    """Free monomer concentrations (nM) for the all-pairs dimer equilibrium, via EQTK."""
    n = c0_nM.shape[1]
    prs = _pairs(n)
    stoich = np.zeros((len(prs), n + len(prs)))
    for r, (i, j) in enumerate(prs):
        if i == j:
            stoich[r, i] = -2.0
        else:
            stoich[r, i] = -1.0
            stoich[r, j] = -1.0
        stoich[r, n + r] = 1.0
    c0_full = np.zeros((c0_nM.shape[0], n + len(prs)))
    c0_full[:, :n] = c0_nM * 1e-9
    k_vec = np.array([1.0 / (kd_nM[i, j] * 1e-9) for i, j in prs])
    sol = np.asarray(eqtk.solve(c0_full, N=stoich, K=k_vec))
    free = sol[:, :n]
    tot = free.copy()
    for r, (i, j) in enumerate(prs):
        d = sol[:, n + r]
        if i == j:
            tot[:, i] += 2.0 * d
        else:
            tot[:, i] += d
            tot[:, j] += d
    resid = float(np.max(np.abs(tot - c0_full[:, :n]) / np.maximum(c0_full[:, :n], 1e-30)))
    return free * 1e9, resid


def contour_err_nM(delta, gx, gy, tree, conc_scale):
    lines = contour_generator(x=gx[0], y=gy[:, 0], z=delta).lines(0)
    if not lines:
        return float("inf")
    step = max(0.5, 0.5 * conc_scale)
    pts = []
    for line in lines:
        for a, b in zip(line, line[1:]):
            cnt = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
            pts.extend(np.linspace(a, b, cnt))
    pts = np.asarray(pts)
    return float((cKDTree(pts).query(tree.data)[0].mean() + tree.query(pts)[0].mean()) / 2.0)


def mcc(pred, lab):
    tp = float(np.sum(pred & lab))
    tn = float(np.sum(~pred & ~lab))
    fp = float(np.sum(pred & ~lab))
    fn = float(np.sum(~pred & lab))
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def do_2d(report_paths, tol, write):
    rows = []
    for rp in sorted(report_paths):
        rep = json.loads(rp.read_text())
        name = rep["boundary"]
        key = "cubic" if name == "cubic_yd" else name
        s = float(rep["conc_scale"])
        max_in, thresh, trans = 200.0 * s, 50.0 * s, 5.0 * s
        nodes = rep["nodes"]
        kd_nM = 10.0 ** (9.0 - np.asarray(rep["pkd_matrix"], float))
        tots = rep["fitted_total_nM"]

        g = np.linspace(0.0, max_in, 121)
        gx, gy = np.meshgrid(g, g)
        c0 = np.column_stack([
            gx.ravel() if nd == "X" else (gy.ravel() if nd == "Y" else np.full(gx.size, float(tots[nd])))
            for nd in nodes
        ])
        free, resid = solve_free_nM(c0, kd_nM)
        vals = free[:, nodes.index("A")].reshape(gx.shape)
        lab = boundary_fn(key, gx, gy, max_in)
        pred = vals > thresh
        acc = float(np.mean(pred == lab))
        logits = (vals - thresh) / trans
        loss = float(np.logaddexp(0.0, np.where(lab, -logits, logits)).mean())
        tree = cKDTree(np.column_stack(curve_points(key, max_in)))
        err = contour_err_nM(vals - thresh, gx, gy, tree, s)

        stored = rep["fitted_metrics"]
        d_acc = abs(acc - float(stored["accuracy"]))
        d_loss = abs(loss - float(stored["loss"]))
        d_err = abs(err - float(stored["error_nM"]))
        ok = (d_acc <= tol) and (d_loss <= tol) and (d_err <= max(0.05, 0.001 * max_in))

        pos = float(lab.mean())
        tpr = float(np.mean(pred[lab])) if lab.any() else 0.0
        tnr = float(np.mean(~pred[~lab])) if (~lab).any() else 0.0
        extra = {
            "recomputed_with": "eqtk, independent of design_2d_boundary.py",
            "recomputed_accuracy": acc,
            "recomputed_loss": loss,
            "recomputed_error_nM": err,
            "max_relative_mass_balance_residual": resid,
            "matches_stored_metrics": bool(ok),
            "positive_class_fraction": pos,
            "majority_baseline_accuracy": max(pos, 1.0 - pos),
            "sensitivity": tpr,
            "specificity": tnr,
            "balanced_accuracy": 0.5 * (tpr + tnr),
            "mcc": mcc(pred, lab),
            "contour_error_pct_of_domain": 100.0 * err / max_in,
        }

        status = "OK" if ok else "MISMATCH"
        print(f"  {rp.parent.name:34s} acc={acc:.4%} (stored {stored['accuracy']:.4%}) "
              f"bal_acc={extra['balanced_accuracy']:.4%} MCC={extra['mcc']:+.3f} "
              f"baseline={extra['majority_baseline_accuracy']:.1%}  [{status}]")

        if not ok:
            print(f"    !! not writing {rp.name}: recomputation disagrees with stored metrics "
                  f"(d_acc={d_acc:.2e}, d_loss={d_loss:.2e}, d_err={d_err:.2e} nM)")
        elif write:
            rep["fitted_metrics_extra"] = extra
            rp.write_text(json.dumps(rep, indent=2) + "\n")

        rows.append((rp.parent.name, name, acc, extra["majority_baseline_accuracy"],
                     extra["balanced_accuracy"], extra["mcc"],
                     err, extra["contour_error_pct_of_domain"], ok))
    return rows


def observable_vals(kind, nodes, refs, free, dimers, c0_M):
    if kind == "free_frac":
        return free[:, nodes[0]] / np.maximum(c0_M[:, refs[0]], 1e-18)
    if kind == "bound_frac":
        return 1.0 - free[:, nodes[0]] / np.maximum(c0_M[:, refs[0]], 1e-18)
    if kind == "dimer_frac":
        u, v = nodes[0], nodes[1]
        mult = 2.0 if (u == v and refs[0] == u) else 1.0
        return mult * dimers[(min(u, v), max(u, v))] / np.maximum(c0_M[:, refs[0]], 1e-18)
    if kind == "sum_free_frac":
        acc = np.zeros(free.shape[0])
        for u, r in zip(nodes, refs):
            acc += free[:, u] / np.maximum(c0_M[:, r], 1e-18)
        return acc / float(len(nodes))
    raise ValueError(kind)


def do_fold(full_jsons, out_csv, write):
    rows = []
    for fp in sorted(full_jsons):
        bundle = json.loads(fp.read_text())
        cfg = bundle["config"]
        if cfg.get("mode") != "truth_table":
            continue
        des = bundle["designs"][0]
        names = [nd["name"] for nd in cfg["nodes"]]
        idx = {nm: i for i, nm in enumerate(names)}
        n = len(names)
        kd_nM = np.minimum(10.0 ** (9.0 - np.asarray(des["pkd_matrix"], float)), 1e7)
        c0_nM = np.asarray(des["c0_uM"], float) * 1e3
        free_nM, _ = solve_free_nM(c0_nM, kd_nM)
        free_M, c0_M, kd_M = free_nM * 1e-9, c0_nM * 1e-9, kd_nM * 1e-9
        dimers = {(i, j): free_M[:, i] * free_M[:, j] / kd_M[i, j]
                  for i in range(n) for j in range(i, n)}

        worst = None
        for ob in cfg["observables"]:
            nodes_i = tuple(idx[x] for x in ob["nodes"])
            refs_i = tuple(idx[x] for x in ob.get("ref_nodes", ob["nodes"]))
            vals = observable_vals(ob["kind"], nodes_i, refs_i, free_M, dimers, c0_M)
            bits = np.asarray(ob["targets"], int)
            min_on = float(vals[bits == 1].min())
            max_off = float(vals[bits == 0].max())
            cand = (min_on / max(max_off, FOLD_FLOOR),
                    (min_on / max_off) if max_off > 0 else float("inf"),
                    min_on, max_off)
            if worst is None or cand[0] < worst[0]:
                worst = cand

        fold_floored, fold_true, min_on, max_off = worst
        censored = max_off < FOLD_FLOOR
        stored = des["metrics"].get("Fold")
        rows.append({
            "circuit": cfg["name"],
            "design_id": des.get("design_id", "D01"),
            "min_ON": round(min_on, 6),
            "max_OFF": round(max_off, 6),
            "Fold_reported": None if stored is None else round(float(stored), 3),
            "Fold_recomputed_floored": round(fold_floored, 3),
            "Fold_uncensored": ("inf" if not np.isfinite(fold_true) else round(fold_true, 3)),
            "censored_by_floor": "YES" if censored else "no",
        })
        mark = "  <-- CENSORED by the 0.005 floor" if censored else ""
        print(f"  {cfg['name']:22s} max_OFF={max_off:.4f}  Fold reported={fold_floored:8.2f}  "
              f"uncensored={fold_true:10.2f}{mark}")

    if rows and write:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  Wrote {out_csv}")
    return rows


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=root)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    args = ap.parse_args()
    write = not args.dry_run

    reports = sorted((args.root / "results" / "des_bounds_designs").glob("*/*_report.json"))
    fulls = sorted((args.root / "results" / "graph_designs").glob("*/*_full.json"))

    print(f"\n=== 1. 2D boundary reports ({len(reports)} found) ===")
    rows2d = do_2d(reports, args.tol, write=write)

    print(f"\n=== 2. Uncensored Fold for logic gates ({len(fulls)} bundles scanned) ===")
    do_fold(fulls, args.root / "results" / "fold_uncensored.csv", write=write)

    print("\n=== Paste-ready README rows for the 2D section ===")
    print("| Boundary | Fitted Acc | Majority baseline | Balanced Acc | MCC | Contour error |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for nm, _b, acc, base, bal, m, err, pct, _ok in rows2d:
        print(f"| `{nm}` | {acc:.2%} | {base:.1%} | {bal:.2%} | {m:+.3f} | {err:.2f} nM ({pct:.2f}%) |")

    bad = [r for r in rows2d if not r[-1]]
    if bad:
        print(f"\n!! {len(bad)} report(s) did not reproduce and were left untouched.")
        return 1
    print("\nAll 2D reports reproduced from saved data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
