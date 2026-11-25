#!/usr/bin/env bash
# =============================================================================
#  cc-any-graph : apply code-review fixes (P0 + P2)
#
#  Run from anywhere inside the repo:   bash apply_review_fixes.sh
#  Nothing is committed. Every edit is idempotent: re-running is a no-op.
#  Undo everything with:               git checkout -- . && git clean -fd
# =============================================================================
set -euo pipefail

BRANCH="${BRANCH:-review-fixes}"
NO_BRANCH="${NO_BRANCH:-0}"

# --- locate repo root --------------------------------------------------------
if ! ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  echo "ERROR: not inside a git repository. cd into cc-any-graph first." >&2
  exit 1
fi
cd "$ROOT"
if [[ ! -f src/circuit_eq.py || ! -f scripts/validate_and_fix_2d.py ]]; then
  echo "ERROR: $ROOT does not look like cc-any-graph (src/circuit_eq.py missing)." >&2
  exit 1
fi
echo "repo: $ROOT"
echo "head: $(git log --oneline -1)"
echo

# --- work on a branch if the tree is clean -----------------------------------
if [[ "$NO_BRANCH" != "1" ]]; then
  if [[ -z "$(git status --porcelain)" ]]; then
    if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
      git checkout "$BRANCH"; echo "switched to existing branch $BRANCH"
    else
      git checkout -b "$BRANCH"; echo "created branch $BRANCH"
    fi
  else
    echo "NOTE: working tree is dirty -> patching in place on $(git rev-parse --abbrev-ref HEAD)."
  fi
  echo
fi

# =============================================================================
#  1. new file: scripts/report_extra_metrics.py
# =============================================================================
cat > scripts/report_extra_metrics.py <<'PYFILE'
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
PYFILE
echo "wrote scripts/report_extra_metrics.py"

# =============================================================================
#  2. source patches (idempotent)
# =============================================================================
python3 - <<'PATCHER'
import sys
from pathlib import Path

applied, skipped, failed = [], [], []


def patch(rel, marker, edits):
    """edits: list of (old, new). Skips if marker already present."""
    p = Path(rel)
    if not p.exists():
        failed.append(f"{rel}: file not found")
        return
    src = p.read_text()
    if marker in src:
        skipped.append(f"{rel} [{marker}]")
        return
    out = src
    for old, new in edits:
        if old not in out:
            failed.append(f"{rel}: anchor not found -> {old.strip().splitlines()[0][:70]!r}")
            return
        if out.count(old) != 1:
            failed.append(f"{rel}: anchor appears {out.count(old)}x (expected 1) -> "
                          f"{old.strip().splitlines()[0][:70]!r}")
            return
        out = out.replace(old, new, 1)
    p.write_text(out)
    applied.append(f"{rel} [{marker}]")


# --- A. circuit_eq.py : expose the uncensored Fold -------------------------
patch(
    "src/circuit_eq.py",
    "Fold_uncensored",
    [(
        '''    return {
        "values": vals,
        "min_ON": min_on,
        "max_OFF": max_off,
        "Delta": min_on - max_off,
        "Fold": min_on / max(max_off, 0.005),
    }''',
        '''    fold_floor = 0.005
    # "Fold" keeps the 0.005 floor, so every number already published and every MC
    # pass criterion is unchanged. "Fold_uncensored" is the true ratio and
    # "Fold_censored" marks the rows where the floor, not the design, set the value.
    return {
        "values": vals,
        "min_ON": min_on,
        "max_OFF": max_off,
        "Delta": min_on - max_off,
        "Fold": min_on / max(max_off, fold_floor),
        "Fold_uncensored": (min_on / max_off) if max_off > 0.0 else float("inf"),
        "Fold_censored": bool(max_off < fold_floor),
    }''',
    )],
)

# --- B. design_2d_boundary.py : class-imbalance metrics --------------------
patch(
    "scripts/design_2d_boundary.py",
    "classification_extras",
    [
        (
            "def evaluate_full_2d(name: str, kd_mat_nM: np.ndarray",
            '''def classification_extras(pred: np.ndarray, labels: np.ndarray, err_nM: float, max_in: float) -> dict:
    """Class-imbalance context for a thresholded 2D field.

    Raw accuracy on a uniform grid is not interpretable on its own: the circle and
    quadratic boundaries put ~78% and ~66% of the domain in one class, so a constant
    classifier already scores that. Reviewers ask for the baseline, the balanced
    accuracy and MCC alongside it.
    """
    pred = np.asarray(pred, dtype=bool).ravel()
    lab = np.asarray(labels, dtype=bool).ravel()
    tp = float(np.sum(pred & lab))
    tn = float(np.sum(~pred & ~lab))
    fp = float(np.sum(pred & ~lab))
    fn = float(np.sum(~pred & lab))
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    pos = float(lab.mean())
    return {
        "positive_class_fraction": pos,
        "majority_baseline_accuracy": max(pos, 1.0 - pos),
        "sensitivity": tpr,
        "specificity": tnr,
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "mcc": float((tp * tn - fp * fn) / den) if den > 0 else 0.0,
        "contour_error_pct_of_domain": (100.0 * err_nM / max_in) if np.isfinite(err_nM) else None,
    }


def evaluate_full_2d(name: str, kd_mat_nM: np.ndarray''',
        ),
        (
            '''        error_nM=err if np.isfinite(err) else None,
    )
    return metrics, values, val_grid''',
            '''        error_nM=err if np.isfinite(err) else None,
    )
    metrics.update(classification_extras(delta > 0.0, labels, err, max_in))
    return metrics, values, val_grid''',
        ),
    ],
)

# --- C. validate_and_fix_2d.py : real slippage margin ---------------------
patch(
    "scripts/validate_and_fix_2d.py",
    "slip_to_dlog10kd",
    [
        (
            "    model = ConsensusModel(str(calib_path))\n",
            '''    model = ConsensusModel(str(calib_path))
    # iCipa_nter score units -> log10(Kd) units, read off the ridge calibration, so the
    # register margin can be judged in affinity terms instead of arbitrary score points.
    _icipa_i = list(model.models).index("iCipa_nter")
    slip_to_dlog10kd = abs(
        float(model.calib["ridge_coef_log10_kd"][_icipa_i])
        / float(model.calib["scaler_scale"][_icipa_i])
    )
    print(f"  1 iCipa_nter score unit = {slip_to_dlog10kd:.4f} log10 Kd units")
''',
        ),
        (
            '''            margin = float(sc.get("iCipa_slippage_margin", float("nan")))
            flag = ""
            if orient != "PAR" or (shift_raw is not None and int(shift_raw) != 0):
                flag = "  <-- WARN: orient/shift!"
            elif np.isfinite(margin) and margin < 1.0:
                flag = "  <-- tight margin (<1.0)"
            print(
                f"    {e}: len=({len(pads[u])-1},{len(pads[v])-1})  "
                f"pKd={pkd_mat[iu, iv]:.2f} (pred={pkd_pred:.2f})  "
                f"{orient} shift={shift_str}  margin={margin:.2f}{flag}"
            )''',
            '''            # predict_pair never returns "iCipa_slippage_margin"; the old sc.get() made this
            # column silently nan, so the register check never actually ran.
            margin_val, margin_type = model.slippage_margin(pads[u], pads[v])
            margin = float(margin_val)
            dlog10 = margin * slip_to_dlog10kd
            flag = ""
            if orient != "PAR" or (shift_raw is not None and int(shift_raw) != 0):
                flag = "  <-- WARN: orient/shift!"
            elif dlog10 < 0.0:
                flag = "  <-- WARN: wrong pose is preferred!"
            elif dlog10 < 0.30:
                flag = "  <-- weak register preference (<0.30 log10 Kd)"
            print(
                f"    {e}: len=({len(pads[u])-1},{len(pads[v])-1})  "
                f"pKd={pkd_mat[iu, iv]:.2f} (pred={pkd_pred:.2f})  "
                f"{orient} shift={shift_str}  margin={margin:.2f} "
                f"({dlog10:+.2f} log10 Kd vs {margin_type}){flag}"
            )''',
        ),
    ],
)

# --- D. validate_and_fix_2d.py : DEFAULT_DESIGNS points at a missing dir ---
patch(
    "scripts/validate_and_fix_2d.py",
    "cubic_yd_4mer_cfit_s15_kmin0.5/cubic_report.json",
    [(
        '    ROOT / "results/des_bounds_designs/cubic_s5/cubic_report.json",\n]',
        '    ROOT / "results/des_bounds_designs/cubic_yd_4mer_cfit_s15_kmin0.5/cubic_report.json",\n'
        '    ROOT / "results/des_bounds_designs/cubic_yd_5mer_cfit_s15_kmin0.5/cubic_report.json",\n]',
    )],
)

# --- E. check_eqconc_and_adam.py : persist the tuning result ---------------
patch(
    "scripts/check_eqconc_and_adam.py",
    "--save-json",
    [
        (
            '    p.add_argument("--train-seed", type=int, default=123)',
            '    p.add_argument("--train-seed", type=int, default=123)\n'
            '    p.add_argument("--save-json", type=Path, default=None,\n'
            '                   help="Write the before/after tuning report (c0 levels and metrics) to this JSON.")',
        ),
        (
            '''    print(f"EQTK After : Nom Delta={tt_new['Delta']:.3f}, Fold={tt_new['Fold']:.1f}x | Sim MC Pass={mc_new['MC_Pass_rate']*100:.1f}%, p10={mc_new['MC_Delta_p10']:.3f}")''',
            '''    print(f"EQTK After : Nom Delta={tt_new['Delta']:.3f}, Fold={tt_new['Fold']:.1f}x | Sim MC Pass={mc_new['MC_Pass_rate']*100:.1f}%, p10={mc_new['MC_Delta_p10']:.3f}")

    if getattr(args, "save_json", None) is not None:
        def _snap(tt, mc):
            return {
                "Delta": float(tt["Delta"]),
                "Fold": float(tt["Fold"]),
                "Fold_uncensored": float(tt.get("Fold_uncensored", tt["Fold"])),
                "min_ON": float(tt["min_ON"]),
                "max_OFF": float(tt["max_OFF"]),
                "MC_Pass_rate": float(mc["MC_Pass_rate"]),
                "MC_Delta_p10": float(mc["MC_Delta_p10"]),
                "MC_Delta_med": float(mc["MC_Delta_med"]),
                "MC_Fold_med": float(mc["MC_Fold_med"]),
            }

        payload = {
            "circuit": parsed["name"],
            "design_id": des.get("design_id"),
            "settings": {
                "sigma_pKd": float(args.sigma),
                "conc_std": float(args.conc_std),
                "eval_mc": int(args.eval_mc),
                "batch_mc": int(args.batch_mc),
                "adam_steps": int(args.steps),
                "con_min_uM": float(args.con_min),
                "con_max_uM": float(args.con_max),
                "train_seed": int(args.train_seed),
                "eval_seed": 42,
                "fix_reporter": bool(args.fix_reporter),
            },
            "node_names": [str(x) for x in names],
            "tuned_nodes": [str(names[int(j)]) for j in np.atleast_1d(tunable_idx).tolist()],
            "levels_uM_before": np.asarray(nom_levels, dtype=float).tolist(),
            "levels_uM_after": np.asarray(new_levels, dtype=float).tolist(),
            "c0_uM_before": np.asarray(c0_uM_base, dtype=float).tolist(),
            "c0_uM_after": np.asarray(new_c0_uM, dtype=float).tolist(),
            "before": _snap(tt_old, mc_old),
            "after": _snap(tt_new, mc_new),
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(payload, indent=2) + "\\n")
        print(f"Saved ADAM tuning report to {args.save_json}")''',
        ),
    ],
)

# --- F. README : the censored Fold cell -----------------------------------
patch(
    "README.md",
    "731.5x",
    [
        ("| 0.975 | 195.3x |", "| 0.975 | 731.5x† |"),
        (
            "(`shift=-7`, `slip=-33.9`).*",
            "(`shift=-7`, `slip=-33.9`).*\n\n"
            "*†Note on `not_3node` Fold: `Fold` in the code is `min_ON / max(max_OFF, 0.005)`, "
            "so any design with `max_OFF < 0.005` is reported at the ≈200x floor rather than at its "
            "true ratio. `not_3node` has `max_OFF = 0.0013`, i.e. a true fold of `731.5x`; it is the "
            "only row affected. See [`results/fold_uncensored.csv`](results/fold_uncensored.csv).*",
        ),
    ],
)

for tag, items in (("APPLIED", applied), ("ALREADY DONE", skipped), ("FAILED", failed)):
    if items:
        print(f"\n{tag}:")
        for it in items:
            print(f"  - {it}")

if failed:
    print("\nERROR: some patches could not be applied; nothing in those files was changed.")
    sys.exit(1)
PATCHER
echo

# =============================================================================
#  3. hygiene: .DS_Store, .gitignore, requirements.txt
# =============================================================================
DS="results/des_bounds_designs/quadratic_s10_seed/.DS_Store"
if git ls-files --error-unmatch "$DS" >/dev/null 2>&1; then
  git rm --cached --quiet "$DS"; rm -f "$DS"; echo "untracked and deleted $DS"
else
  echo "$DS already untracked"
fi

python3 - <<'GITIGNORE'
from pathlib import Path
p = Path(".gitignore")
txt = p.read_text()
needed = [
    "!results/des_bounds_designs/",
    "!results/des_bounds_designs/*/",
    "!results/des_bounds_designs/*/*",
    "!results/fold_uncensored.csv",
    "!results/adam_tuning/",
    "!results/adam_tuning/*.json",
]
missing = [ln for ln in needed if ln not in txt.splitlines()]
if missing:
    if not txt.endswith("\n"):
        txt += "\n"
    txt += "\n# keep the committed 2D designs and the post-hoc metric artifacts\n"
    txt += "\n".join(missing) + "\n"
    p.write_text(txt)
    print(f"added {len(missing)} negation(s) to .gitignore")
else:
    print(".gitignore already covers the committed results")
GITIGNORE

python3 - <<'REQS'
from pathlib import Path
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

# import name -> distribution name
mods = {
    "numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "sklearn": "scikit-learn",
    "matplotlib": "matplotlib", "contourpy": "contourpy", "eqtk": "eqtk",
    "jax": "jax", "jaxlib": "jaxlib", "optax": "optax", "gemmi": "gemmi",
}
lines = [
    "# Generated by apply_review_fixes.sh from the environment that produced the results.",
    "# Pinned to what was actually installed; relax if you need a wider range.",
    "",
]
missing = []
for mod, dist in sorted(mods.items()):
    try:
        lines.append(f"{dist}=={version(dist)}")
    except PackageNotFoundError:
        missing.append(dist)
lines += [
    "",
    "# Not on PyPI -- install separately:",
    "#   pyccscore   (1D coiled-coil scoring: iCipa / bCIPA / qCIPA / Potapov)",
    "#   pyrosetta   (scripts/score_rosetta.py only; academic licence)",
    "#   boltz       (scripts/analyze_boltz.py consumes its output; not imported here)",
]
if missing:
    lines += ["", "# Imported by the repo but not installed in this environment:"] + \
             [f"#   {d}" for d in missing]
Path("requirements.txt").write_text("\n".join(lines) + "\n")
print("wrote requirements.txt" + (f" (not installed here: {', '.join(missing)})" if missing else ""))
REQS
echo

# =============================================================================
#  4. syntax check
# =============================================================================
echo "byte-compiling changed sources..."
python3 -m py_compile \
  src/circuit_eq.py \
  scripts/design_2d_boundary.py \
  scripts/validate_and_fix_2d.py \
  scripts/check_eqconc_and_adam.py \
  scripts/report_extra_metrics.py
echo "syntax OK"
echo

# =============================================================================
#  5. post-hoc metrics on the committed results
# =============================================================================
echo "running scripts/report_extra_metrics.py ..."
if python3 scripts/report_extra_metrics.py; then
  echo
  echo "post-hoc metrics done"
else
  echo
  echo "WARNING: report_extra_metrics.py failed (needs numpy, scipy, eqtk, contourpy)."
  echo "         The source patches above are still applied."
fi
echo

# =============================================================================
#  6. what is left for a human
# =============================================================================
cat > REVIEW_TODO.md <<'TODO'
# Left to do by hand

Code fixes from the review are applied. These are wording and extra-run items.

## 1. Re-run the 2D register check (the fix changes its output)

```bash
python3 scripts/validate_and_fix_2d.py --mc-and-slippage \
  --reports results/des_bounds_designs/*/*_report.json
```

The `margin=` column used to print `nan` for every edge, so the register check never
ran. It now prints the margin in iCipa units *and* in log10 Kd units. Watch for
`WARN: wrong pose is preferred!` and `weak register preference`. Anything flagged needs a
footnote in the 2D table, the way `nand_6node` and `xnor_5node` are footnoted in the
logic table.

## 2. Regenerate the ADAM section with saved artifacts

For each of the five circuits in that README table:

```bash
for c in xor_4node xnor_4node imply_5node and_5node nand_6node; do
  python3 scripts/check_eqconc_and_adam.py \
    --full-json results/graph_designs/$c/${c}_full.json \
    --tune-adam --fix-reporter \
    --save-json results/adam_tuning/${c}_adam.json
done
```

Then rewrite the table from the JSONs. Right now those 20 numbers exist only in old
terminal output, which means nobody -- including you in a month -- can check them.

Note `xnor_4node` is listed as `D02` in the README, so add `--design-id D02` for it.

## 3. Two README wording changes

**(a) The two MC stages.** The current text implies `Sim MC` is lower than `Design MC`
because of 1.5% pipetting noise. It is not: at 1.5% the concentration noise changes the
pass rate by ~0.5 pp on average. The gap is the selection effect -- `Design MC` is the
statistic the 40 candidates were *ranked* by, so the winner's value is optimistic by
~2.5 pp on average and up to 10 pp on the weak gates. Suggested replacement for the
`Sim MC` bullet:

> 2. **`Sim MC` (`simulate_circuit.py`, `n_mc = 400`, `sigma_pKd = 0.643`,
>    `conc_std = 0.015`)**: an independent re-evaluation with a fresh seed, including
>    1.5% nanoliter-dispenser pipetting noise. Because `Design MC` is the statistic the
>    post-SA ranking selects on, it is optimistic by construction (~2.5 pp on average,
>    up to 10 pp for the lowest-margin gates); `Sim MC` is unbiased and is the figure to
>    quote. The 1.5% concentration noise itself accounts for <1 pp of the difference.

**(b) Analog "Shape Preserved".** State the criterion inline, because it is much softer
than `Exact`: a dome counts as preserved if peak >= 0.35 with rise and fall >= 0.25, and a
monotone curve if `corr >= 0.90` and span >= 0.35. Keep `Exact` as the headline number
(13-24%) and present the shape column as a separate, explicitly-defined metric.

## 4. Supplementary: correlated model error

The MC assumes the 0.643 log-unit pKd error is i.i.d. across pairs. It is not -- the
four scoring models share features, so a common-mode component is expected. Splitting
the variance into a shared and an independent part *raises* pass rates: at rho = 0.5,
+7.6 pp on average. This argues the current noise model is the conservative one, so it
is worth a supplementary figure (rho = 0, 0.25, 0.5, 1.0).

## 5. Mark the six edges outside the calibration range

Calibration spans pKd 4.32-10.0 (n = 14). These target edges are extrapolations:

| Circuit | Edge | pKd | Kd |
| :--- | :--- | :---: | :---: |
| `nor_4node` | I2:Q | 10.05 | 89 pM |
| `nor_5node` | I1:M5 | 10.74 | 18 pM |
| `nor_5node` | I2:M5 | 10.34 | 45 pM |
| `threshold_decoy_4node` | D:I | 10.57 | 27 pM |
| `xor_4node` | A:B | 11.30 | 5.0 pM |
| `xnor_5node` | A:B | 12.08 | 0.8 pM |

Sub-picomolar affinity for a 28-residue coiled-coil dimer is not physically credible.
Either lower `pkd_max` to 10.0 in those configs and re-run, or flag the cells. Note
`xor_4node` and `xnor_5node` are also the two designs with negative slippage margins and
the lowest pass rates -- the XOR family is the weak link on three independent axes.

## 6. Methods paragraph that needs to exist

- Calibration: n = 14 Kd anchors (Thomas 2013, Aronsson 2015), a single homodimer anchor
  (`BN4:BN4`, pKd 4.32), LOO r = 0.917, LOO RMSE = 0.643. Crooks 2017 is a zero-shot Tm
  check only, not part of the fit.
- Equilibrium: monomers and dimers only; no trimers or higher oligomers are represented,
  so a trimer-forming design is mispredicted qualitatively, not just quantitatively. The
  core alphabet (Ile/Val at *a*, Leu at *d*) is the Harbury dimer-specifying choice,
  which is what justifies the restriction.
- Kd is capped at 10 mM (pKd >= 2) throughout.
- 2D: trained on a 31x31 grid (11x11 inside SA) plus contour samples, reported on 121x121,
  whose points include the training grid. The contour error is the leak-free metric.
  The reported design is the best of 40 candidates under the same metric.
- The two concentration-noise models differ: `src/circuit_eq.py` draws independently per
  state and species; `scripts/validate_and_fix_2d.py` draws one multiplier per species
  shared across the grid, including the X and Y input axes.

## 7. One stale stored metric

`xnor_5node` D01 stores `Delta = 0.5511`, and README reports `0.551`. Recomputing from
that design's own saved `pkd_matrix` and `c0_uM` gives `0.5537`, with EQTK and with an
independent solver agreeing. It looks like a metric written before the final `c0` tune.
The gap is 0.5% and changes no conclusion, but if anyone re-derives the table from the
JSONs they will get a different number, so either refresh that one entry or note it.

Every other Delta, Fold, min_ON, max_OFF, R2, RMSE and Span across all 21 logic and
analog designs reproduces exactly, as do all five 2D reports.
TODO
echo "wrote REVIEW_TODO.md"
echo

echo "============================================================"
echo " done. nothing committed."
echo
git status --short
echo
echo " review the diff:    git diff"
echo " then commit:        git add -A && git commit -m 'Fix slippage check, persist ADAM tuning, add class-imbalance and uncensored-Fold metrics'"
echo " or undo everything: git checkout -- . && git clean -fd"
echo "============================================================"
