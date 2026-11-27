"""De novo coiled-coil sequence design for korenkov-es/banana topologies."""

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.optimize import least_squares, minimize
from scipy.special import expit, log_softmax

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.consensus import ConsensusModel, DEFAULT_CALIB_PATH
from src.sequence_space import (
    PeptideSpec,
    build_sequence,
    calc_eps280,
    generate_node_pool,
    mutate_spec,
    net_charge,
)

INPUT_MAX_NM = 200.0
KD_BOUNDS_NATIVE = (1.0, 1000.0)
TOTAL_BOUNDS_NATIVE = (1.0, 400.0)

GEOMETRY = {
    "center_x_nM": 81.60262722518003,
    "center_y_nM": -61.35139230766126,
    "x_shear": -0.026983573718282847,
    "y_scale": 0.4952714824652137,
    "radii_nM": [60.90715850262229, 95.56248405294735, 125.34507669381321],
}

UNIFORM_GRID = {"train_side": 41, "validation_side": 80, "test_side": 160}

# Exact fitted models from korenkov-es/banana/results/fixed/*/parameters.json
FIXED_TOPOLOGIES = {
    "monomer_3": {
        "bands": 3,
        "outputs": ["A", "B"],
        "Kd_nM": {
            "AC": 499.99999988736414,
            "BE": 53.99999990442276,
            "CE": 13.199999978146812,
            "BC": 50.00000009262788,
            "XC": 5.446685925157823,
            "YC": 1.0,
            "YH": 3.859176663621489,
            "XD": 1.6380434573105336,
            "CD": 7.751217994791379,
        },
        "total_nM": {
            "A": 27.000000072442706,
            "B": 41.74736308642819,
            "C": 59.414207336832106,
            "D": 92.92059016710962,
            "E": 46.69796806363618,
            "H": 175.84703210586733,
        },
        "training_scale_nM": 0.0014411993198139902,
        "free_C_at_switch_nM": [7.348469228349534, 4.898979485566356],
    },
    "monomer_4": {
        "bands": 4,
        "outputs": ["A", "B"],
        "Kd_nM": {
            "AC": 499.9989345376492,
            "BE": 8.99989893587604,
            "CE": 13.19983773410407,
            "AF": 18.000131113576742,
            "CF": 2.099993458772492,
            "BC": 35.70899174570164,
            "XC": 5.5984501134272415,
            "YC": 6.9054371645824375,
            "YH": 16.18095765000964,
            "XD": 1.942207998516307,
            "CD": 11.584826914040478,
        },
        "total_nM": {
            "A": 4.499946503336959,
            "B": 9.002844289067959,
            "C": 55.669749181797435,
            "D": 101.89912569338972,
            "E": 17.129191894752744,
            "F": 4.999958602819114,
            "H": 133.6398307694902,
        },
        "training_scale_nM": 0.00014711042812769616,
        "free_C_at_switch_nM": [9.0, 6.0, 4.0],
    },
    "dimer_3_compact": {
        "bands": 3,
        "outputs": ["AC", "BD"],
        "Kd_nM": {
            "YA": 46.51544666232592,
            "YB": 32.55766244507443,
            "XA": 6.309190278921238,
            "XB": 4.416184554744517,
            "AC": 1.0000001337147808,
            "BD": 1.452284593924423,
            "CE": 35.030211763527966,
            "DF": 7.595995479430785,
            "XE": 1.2293809305034922,
            "XF": 1.0000009161437011,
        },
        "total_nM": {
            "A": 63.05157647451341,
            "B": 103.04597362280563,
            "C": 63.00076098014036,
            "D": 56.100663199139554,
            "E": 67.70052363019579,
            "F": 21.204396063325408,
        },
        "training_scale_nM": 0.0034768511291373865,
    },
    "dimer_4_compact": {
        "bands": 4,
        "outputs": ["CG", "DH"],
        "Kd_nM": {
            "YA": 26.811277256807355,
            "YB": 26.67457385350718,
            "XA": 12.277499474214418,
            "XB": 12.21492569756244,
            "AC": 1.2704948648932406,
            "BD": 1.1419076897415037,
            "CE": 999.9999973232672,
            "DF": 806.1355372265114,
            "XE": 1.3637526353572822,
            "XF": 1.361365430518396,
            "CG": 4.663875550152643,
            "DH": 4.216331791392929,
        },
        "total_nM": {
            "A": 30.3285688210901,
            "B": 28.894830550665507,
            "C": 37.36512207628375,
            "D": 35.73663494506497,
            "E": 55.75564100144513,
            "F": 46.46321016980778,
            "G": 10.991532698102178,
            "H": 10.93434343251797,
        },
        "training_scale_nM": 4.3812889961120224e-07,
    },
    "dimer_3": {
        "bands": 3,
        "outputs": ["EI", "FI"],
        "Kd_nM": {
            "YA": 19.5537914539098,
            "YB": 13.343139149992878,
            "XC": 6.661295249271672,
            "XD": 5.368796437357061,
            "AC": 1.2562828080688606,
            "BD": 1.0000230916264452,
            "EI": 1.4249270975188957,
            "FI": 1.9221935825239522,
            "CE": 1.000006522910618,
            "DF": 6.934341800203538,
            "XY": 999.9999999999989,
            "XI": 1.0000140910181008,
        },
        "total_nM": {
            "A": 55.901072042544854,
            "B": 226.0867301118904,
            "C": 189.1242363768561,
            "D": 338.4159413456021,
            "E": 171.3375654598585,
            "F": 113.84422735824671,
            "I": 155.5527920665002,
        },
        "training_scale_nM": 0.002081005491813272,
    },
    "dimer_4": {
        "bands": 4,
        "outputs": ["AG", "BH"],
        "Kd_nM": {
            "XA": 4.450082268319659,
            "XB": 3.6788990592906687,
            "YA": 63.10833010988525,
            "YB": 52.1615835370229,
            "AC": 20.74173597673457,
            "BD": 29.569761379116002,
            "CE": 5.212877394132419,
            "DF": 5.857669214460509,
            "XE": 1.0299206006789345,
            "XF": 1.0303592807822617,
            "AG": 6.392323635097527,
            "BH": 3.866509176019415,
            "CI": 26.795530340744374,
            "DJ": 47.036829461328765,
            "GI": 35.97168847875019,
            "HJ": 37.905369834790385,
        },
        "total_nM": {
            "A": 79.50427764863275,
            "B": 57.814523492588755,
            "C": 50.84880196790205,
            "D": 44.77858410829475,
            "E": 40.85765024463221,
            "F": 46.29264991684453,
            "G": 23.88117652709971,
            "H": 23.063481133420048,
            "I": 16.285247836143352,
            "J": 8.407972045467858,
        },
        "training_scale_nM": 7.603652879271756e-06,
    },
}


def band_indices(points: np.ndarray, arcs: dict, bands: int) -> np.ndarray:
    x, y = points.T
    dist = np.hypot(x - arcs["center_x_nM"] - arcs["x_shear"] * y, arcs["y_scale"] * (y - arcs["center_y_nM"]))
    return np.searchsorted(arcs["radii_nM"][: bands - 1], dist)


def band_labels(points: np.ndarray, bands: int, arcs: dict = GEOMETRY) -> np.ndarray:
    return (band_indices(points, arcs, bands) % 2).astype(int)


def arc_samples(arcs: dict, bands: int, count: int = 801) -> List[np.ndarray]:
    theta = np.linspace(0.0, np.pi, count)
    nx = np.cos(theta)
    ny = arcs["y_scale"] * np.sin(theta) - arcs["x_shear"] * nx
    out = []
    for r in arcs["radii_nM"][: bands - 1]:
        y = arcs["center_y_nM"] + r * np.sin(theta) / arcs["y_scale"]
        x = arcs["center_x_nM"] + arcs["x_shear"] * y + r * nx
        inside = (x >= 0.0) & (x <= INPUT_MAX_NM) & (y >= 0.0) & (y <= INPUT_MAX_NM)
        out.append(np.column_stack((x, y, nx / np.hypot(nx, ny), ny / np.hypot(nx, ny)))[inside])
    return out


def square_grid(axis: np.ndarray) -> np.ndarray:
    return np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)


def make_uniform_samples(bands: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    tr = square_grid(np.linspace(0.0, INPUT_MAX_NM, UNIFORM_GRID["train_side"]))
    va = square_grid((np.arange(UNIFORM_GRID["validation_side"]) + 0.5) * INPUT_MAX_NM / UNIFORM_GRID["validation_side"])
    te = square_grid((np.arange(UNIFORM_GRID["test_side"]) + 0.5) * INPUT_MAX_NM / UNIFORM_GRID["test_side"])
    return {
        "uniform_train": (tr, band_labels(tr, bands)),
        "uniform_validation": (va, band_labels(va, bands)),
        "uniform_test": (te, band_labels(te, bands)),
    }


def load_banana_mat(mat_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    banana = loadmat(mat_path, simplify_cells=True)["banana"]
    x, targets = banana["x"], banana["t"]
    labels = (targets == 1).astype(int)
    train_idx = banana["train"][0] - 1
    test_idx = banana["test"][0] - 1
    rng = np.random.default_rng(42)
    groups = [rng.permutation(train_idx[labels[train_idx] == lab]) for lab in (0, 1)]
    counts = [round(len(g) * 0.2) for g in groups]
    val_idx = np.concatenate([g[:n] for g, n in zip(groups, counts)])
    tr_idx = np.concatenate([g[n:] for g, n in zip(groups, counts)])
    low, high = x[tr_idx].min(axis=0), x[tr_idx].max(axis=0)
    conc = INPUT_MAX_NM * np.clip((x - low) / (high - low), 0.0, 1.0)
    return {
        "banana_train": (conc[tr_idx], labels[tr_idx]),
        "banana_validation": (conc[val_idx], labels[val_idx]),
        "test": (conc[test_idx], labels[test_idx]),
    }


def solve_eq_pos_with_jac(c0_nM: np.ndarray, kd_mat_nM: np.ndarray, max_newton: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    n = c0_nM.shape[1]
    affinity = 1.0 / np.maximum(kd_mat_nM, 1e-6)
    binding = affinity + np.diag(np.diag(affinity))
    eye = np.eye(n, dtype=float)[None, :, :]

    x = c0_nM.copy()
    for _ in range(36):
        x = 0.5 * x + 0.5 * (c0_nM / (1.0 + x @ binding))

    active = c0_nM > 0.0
    for _ in range(max_newton):
        factor = 1.0 + x @ binding
        f = x * factor - c0_nM
        if np.max(np.abs(f)) < 1e-9:
            break
        jac = x[:, :, None] * binding[None, :, :] + factor[:, :, None] * eye
        try:
            dx = np.linalg.solve(jac, -f[..., None])[..., 0]
        except np.linalg.LinAlgError:
            dx = np.zeros_like(f)
            for b in range(len(f)):
                dx[b] = np.linalg.lstsq(jac[b], -f[b], rcond=None)[0]
        neg = (dx < 0.0) & active
        alpha = np.where(neg, -0.92 * x / np.minimum(dx, -1e-30), 1.0)
        step = np.minimum(1.0, np.min(alpha, axis=1, keepdims=True))
        x = np.where(active, np.maximum(x + step * dx, 1e-18), 0.0)

    factor = 1.0 + x @ binding
    jac = x[:, :, None] * binding[None, :, :] + factor[:, :, None] * eye
    return x, jac


def solve_eq_fast_np(c0_nM: np.ndarray, kd_mat_nM: np.ndarray) -> np.ndarray:
    x, _ = solve_eq_pos_with_jac(c0_nM, kd_mat_nM, max_newton=14)
    return x


def parse_readouts(nodes: Sequence[str], kd_mat_nM: np.ndarray, readouts: Sequence[str]):
    spec = []
    for r in readouts:
        if len(r) == 1:
            spec.append((nodes.index(r), None, None))
        else:
            i, j = nodes.index(r[0]), nodes.index(r[1])
            spec.append((i, j, float(kd_mat_nM[i, j])))
    return spec


def eval_readouts(free: np.ndarray, readout_spec) -> np.ndarray:
    cols = []
    for i, j, k_ij in readout_spec:
        if j is None:
            cols.append(free[:, i])
        else:
            cols.append(free[:, i] * free[:, j] / k_ij)
    return np.column_stack(cols)


def cross_entropy(signals: np.ndarray, labels: np.ndarray, scale_nM: float) -> Tuple[float, np.ndarray]:
    log_probs = log_softmax(signals / scale_nM, axis=1)
    return float(-log_probs[np.arange(len(labels)), labels].mean()), log_probs


def eval_banana_sample(
    kd_mat_nM: np.ndarray,
    totals_dict: Dict[str, float],
    nodes: Sequence[str],
    readouts: Sequence[str],
    sample: Tuple[np.ndarray, np.ndarray],
    scale_nM: float,
) -> dict:
    points, labels = sample
    internal = list(nodes[2:])
    c0 = np.column_stack((points, np.broadcast_to([totals_dict[k] for k in internal], (len(points), len(internal)))))
    free, _ = solve_eq_pos_with_jac(c0, kd_mat_nM)
    rspec = parse_readouts(nodes, kd_mat_nM, readouts)
    vals = eval_readouts(free, rspec)
    pred = vals.argmax(axis=1)
    ce1, _ = cross_entropy(vals, labels, 1.0)
    ces, _ = cross_entropy(vals, labels, scale_nM)
    return {
        "accuracy": float(np.mean(pred == labels)),
        "correct": int(np.sum(pred == labels)),
        "n": int(len(labels)),
        "cross_entropy_1nM": ce1,
        "training_scale_CE": ces,
        "median_gap_nM": float(np.median(np.abs(vals[:, 1] - vals[:, 0]))),
    }


def fit_totals_ce(
    kd_mat_nM: np.ndarray,
    init_totals: Dict[str, float],
    nodes: Sequence[str],
    readouts: Sequence[str],
    data: Tuple[np.ndarray, np.ndarray],
    scale_nM: float,
    maxiter: int = 220,
    trainable: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    points, labels = data
    internal = list(nodes[2:])
    rspec = parse_readouts(nodes, kd_mat_nM, readouts)
    init_vec = np.array([init_totals[k] for k in internal], dtype=float)
    if trainable is None:
        active = np.ones(len(internal), dtype=bool)
    else:
        tset = set(trainable)
        active = np.array([k in tset for k in internal], dtype=bool)
    if not np.any(active):
        return dict(init_totals)

    bounds = [np.log(TOTAL_BOUNDS_NATIVE)] * int(np.sum(active))

    def objective(log_p):
        tots = init_vec.copy()
        tots[active] = np.exp(log_p)
        c0 = np.column_stack((points, np.broadcast_to(tots, (len(points), len(internal)))))
        free, jac = solve_eq_pos_with_jac(c0, kd_mat_nM)
        signals = eval_readouts(free, rspec)
        loss, log_probs = cross_entropy(signals, labels, scale_nM)
        probs = np.exp(log_probs)
        probs[np.arange(len(labels)), labels] -= 1.0
        rhs = np.zeros_like(free)
        for k_out, (i, j, k_ij) in enumerate(rspec):
            w = probs[:, k_out] / scale_nM
            if j is None:
                rhs[:, i] += w
            else:
                rhs[:, i] += w * free[:, j] / k_ij
                rhs[:, j] += w * free[:, i] / k_ij
        adjoint = np.linalg.solve(jac.swapaxes(1, 2), rhs[..., None])[..., 0]
        grad_t = adjoint[:, 2:].mean(axis=0) * tots
        return loss, grad_t[active]

    start = np.clip(np.log(init_vec[active]), np.log(TOTAL_BOUNDS_NATIVE[0]), np.log(TOTAL_BOUNDS_NATIVE[1]))
    res = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=maxiter))
    out = init_vec.copy()
    out[active] = np.exp(res.x)
    return dict(zip(internal, out))


def fit_totals_dimer_lsq(
    kd_mat_nM: np.ndarray,
    init_totals: Dict[str, float],
    nodes: Sequence[str],
    readouts: Sequence[str],
    data: Tuple[np.ndarray, np.ndarray],
    scale_nM: float,
    max_nfev: int = 125,
) -> Dict[str, float]:
    points, labels = data
    sign = 2 * labels - 1
    internal = list(nodes[2:])
    rspec = parse_readouts(nodes, kd_mat_nM, readouts)
    i0, j0, k0 = rspec[0]
    i1, j1, k1 = rspec[1]
    last, cached = None, None

    def residual(log_p):
        nonlocal last, cached
        if last is not None and np.array_equal(log_p, last):
            return cached
        tots = np.exp(log_p)
        c0 = np.column_stack((points, np.broadcast_to(tots, (len(points), len(internal)))))
        free, jac = solve_eq_pos_with_jac(c0, kd_mat_nM)
        b0 = free[:, i0] * free[:, j0] / k0
        b1 = free[:, i1] * free[:, j1] / k1
        delta = b1 - b0

        rhs = np.zeros_like(free)
        rhs[:, i1] += free[:, j1] / k1
        rhs[:, j1] += free[:, i1] / k1
        rhs[:, i0] -= free[:, j0] / k0
        rhs[:, j0] -= free[:, i0] / k0
        adjoint = np.linalg.solve(jac.swapaxes(1, 2), rhs[..., None])[..., 0]
        grad_t = adjoint[:, 2:] * tots[None, :]

        z = -sign * delta / scale_nM
        r = np.sqrt(2.0 * np.logaddexp(0.0, z))
        factor = -sign * expit(z) / (scale_nM * np.maximum(r, 1e-200))
        cached = (r, grad_t * factor[:, None])
        last = log_p.copy()
        return cached

    start = np.clip(
        np.log([init_totals[k] for k in internal]),
        np.log(TOTAL_BOUNDS_NATIVE[0]),
        np.log(TOTAL_BOUNDS_NATIVE[1]),
    )
    res = least_squares(
        lambda v: residual(v)[0],
        start,
        jac=lambda v: residual(v)[1],
        bounds=(np.log(TOTAL_BOUNDS_NATIVE[0]), np.log(TOTAL_BOUNDS_NATIVE[1])),
        x_scale="jac",
        max_nfev=max_nfev,
        ftol=1e-8,
        xtol=1e-10,
    )
    return dict(zip(internal, np.exp(res.x)))


def refit_totals_banana(
    kd_mat_nM: np.ndarray,
    ideal_model: dict,
    nodes: Sequence[str],
    uniform_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_restarts: int = 8,
    seed: int = 915,
    fast_mode: bool = False,
) -> Dict[str, float]:
    readouts = ideal_model["outputs"]
    initial = dict(ideal_model["total_nM"])
    scale = float(ideal_model["training_scale_nM"])
    train = uniform_data["uniform_train"]
    val = uniform_data["uniform_validation"]
    side = UNIFORM_GRID["train_side"]
    coarse_idx = np.arange(side ** 2).reshape(side, side)[::2, ::2].ravel()
    coarse = (train[0][coarse_idx], train[1][coarse_idx])
    is_monomer = (len(readouts[0]) == 1)

    def rank(tots):
        m = eval_banana_sample(kd_mat_nM, tots, nodes, readouts, val, scale)
        return (m["correct"], -m["training_scale_CE"], tots)

    rng = np.random.default_rng(seed)
    starts = [initial]
    if is_monomer and "free_C_at_switch_nM" in ideal_model:
        levels = np.array(ideal_model["free_C_at_switch_nM"], dtype=float)
        # Analytical switch-level alignment matching banana/core/refitting.py
        idx_a, idx_b, idx_c, idx_e = nodes.index("A"), nodes.index("B"), nodes.index("C"), nodes.index("E")
        k_ac, k_bc = kd_mat_nM[idx_a, idx_c], kd_mat_nM[idx_b, idx_c]
        k_be, k_ce = kd_mat_nM[idx_b, idx_e], kd_mat_nM[idx_c, idx_e]
        has_f = "F" in initial
        if has_f:
            idx_f = nodes.index("F")
            k_af, k_cf = kd_mat_nM[idx_a, idx_f], kd_mat_nM[idx_c, idx_f]
            names_al = ["B", "E", "F"]
        else:
            names_al = ["B", "E"]

        def free_comp(tot, fac, comp, half):
            off = fac * half + comp - tot
            return 2.0 * tot * half / (np.sqrt(off ** 2 + 4.0 * fac * tot * half) + off)

        def mismatch(log_t):
            td = dict(initial)
            for nm, v in zip(names_al, np.exp(log_t)):
                td[nm] = v
            fac_a = 1.0 + levels / k_ac
            a_v = free_comp(td["A"], fac_a, td["F"], k_af * (1.0 + levels / k_cf)) if has_f else (td["A"] / fac_a)
            b_v = free_comp(td["B"], 1.0 + levels / k_bc, td["E"], k_be * (1.0 + levels / k_ce))
            return b_v - a_v

        s_al = np.log([initial[nm] for nm in names_al])
        res_al = least_squares(mismatch, s_al, bounds=np.log(TOTAL_BOUNDS_NATIVE), gtol=1e-12, max_nfev=300)
        aligned = dict(initial)
        for nm, v in zip(names_al, np.exp(res_al.x)):
            aligned[nm] = float(v)
        sensor = [nm for nm in initial if nm not in ("A", "B", "E", "F")]
        aligned = fit_totals_ce(kd_mat_nM, aligned, nodes, readouts, coarse, scale, maxiter=200, trainable=sensor)
        starts.append(aligned)

    for idx in range(n_restarts):
        if idx % 2 == 0:
            starts.append({k: float(np.clip(v * np.exp(rng.normal(0.0, 0.85)), *TOTAL_BOUNDS_NATIVE)) for k, v in initial.items()})
        else:
            starts.append({k: float(np.exp(rng.uniform(*np.log(TOTAL_BOUNDS_NATIVE)))) for k in initial})

    candidates = [rank(initial)]
    temps = (0.1, scale) if fast_mode else (0.4, 0.04, max(5e-4, scale), scale)
    m_it = 70 if fast_mode else 140
    d_nf = 55 if fast_mode else 100
    for st in starts:
        cur = st
        candidates.append(rank(cur))
        for temp in temps:
            if is_monomer:
                cur = fit_totals_ce(kd_mat_nM, cur, nodes, readouts, coarse, temp, maxiter=m_it)
            else:
                cur = fit_totals_dimer_lsq(kd_mat_nM, cur, nodes, readouts, coarse, temp, max_nfev=d_nf)
            candidates.append(rank(cur))

    best_acc = max(candidates, key=lambda r: (r[0], r[1]))[2]
    best_ce = max(candidates, key=lambda r: r[1])[2]
    polish_list = (best_acc,) if fast_mode else (best_acc, best_ce)
    p_it = 140 if fast_mode else 350
    p_nf = 100 if fast_mode else 220
    for base_t in polish_list:
        if is_monomer:
            polished = fit_totals_ce(kd_mat_nM, base_t, nodes, readouts, train, scale, maxiter=p_it)
        else:
            polished = fit_totals_dimer_lsq(kd_mat_nM, base_t, nodes, readouts, train, scale, max_nfev=p_nf)
        candidates.append(rank(polished))

    return max(candidates, key=lambda r: (r[0], r[1]))[2]


def _hinge_sq(deficit: float, w_sq: float = 25.0, w_lin: float = 5.0) -> float:
    if deficit <= 0.0:
        return 0.0
    return w_sq * (deficit * deficit) + w_lin * deficit


def _score_banana_sa(
    pair_mat: List[List[dict]],
    nodes: Sequence[str],
    target_edges: Sequence[Tuple[int, int, float]],
    hierarchies: Sequence[Tuple[int, int, int, int, float]],
    off_max_pkd: float,
    homo_max_pkd: float,
    c0_train_native: np.ndarray,
    labels_train: np.ndarray,
    readouts: Sequence[str],
    conc_scale: float,
) -> Tuple[float, dict]:
    n = len(nodes)
    target_set = {(min(u, v), max(u, v)) for u, v, _ in target_edges}

    graph_loss = 0.0
    for u, v, ideal_pkd in target_edges:
        sc = pair_mat[u][v]
        pkd = sc["pKd"]
        graph_loss += _hinge_sq((ideal_pkd - 0.12) - pkd, 45.0, 9.0)
        graph_loss += _hinge_sq(pkd - (ideal_pkd + 0.12), 45.0, 9.0)
        graph_loss += 28.0 * ((pkd - ideal_pkd) ** 2)
        if sc.get("iCipa_best_orient", "PAR") != "PAR":
            graph_loss += 50.0
        if sc.get("iCipa_best_shift", 0) != 0:
            graph_loss += 80.0

    for su, sv, wu, wv, min_gap in hierarchies:
        dpkd = pair_mat[su][sv]["pKd"] - pair_mat[wu][wv]["pKd"]
        graph_loss += _hinge_sq(min_gap - dpkd, 22.0, 4.0)

    worst_off = -99.0
    worst_homo = -99.0
    for i in range(n):
        for j in range(i, n):
            if (i, j) in target_set:
                continue
            pkd = pair_mat[i][j]["pKd"]
            cap = homo_max_pkd if i == j else off_max_pkd
            if i == j:
                worst_homo = max(worst_homo, pkd)
                graph_loss += _hinge_sq(pkd - cap, 42.0, 8.0)
            else:
                worst_off = max(worst_off, pkd)
                graph_loss += _hinge_sq(pkd - cap, 28.0, 5.0)

    log10_kd = np.array([[pair_mat[i][j]["log10_kd"] for j in range(n)] for i in range(n)], dtype=float)
    kd_native_nM = ((10.0 ** np.minimum(log10_kd, -2.0)) * 1e9) / conc_scale

    free = solve_eq_fast_np(c0_train_native, kd_native_nM)
    rspec = parse_readouts(nodes, kd_native_nM, readouts)
    vals = eval_readouts(free, rspec)
    delta = vals[:, 1] - vals[:, 0]
    sign = 2 * labels_train - 1

    # Smooth normalized margin + binary cross-entropy so monomer and dimer outputs are well-conditioned
    norm_delta = delta / np.maximum(vals[:, 0] + vals[:, 1], 1e-3)
    logits = np.clip(-sign * norm_delta / 0.04, -50.0, 50.0)
    bce_norm = float(np.logaddexp(0.0, logits).mean())
    acc = float(np.mean((delta > 0.0) == labels_train))
    circuit_loss = 3.0 * bce_norm - 1.5 * acc

    total_loss = graph_loss + circuit_loss
    return float(total_loss), {
        "total_loss": float(total_loss),
        "graph_loss": float(graph_loss),
        "circuit_loss": float(circuit_loss),
        "acc_train": acc,
        "worst_off_pKd": float(worst_off),
        "worst_homo_pKd": float(worst_homo),
    }


def _worker_sa_banana(args_tuple):
    (
        calib_path_str,
        pools_by_node,
        nodes,
        target_edges,
        hierarchies,
        lengths_by_node,
        off_max_pkd,
        homo_max_pkd,
        c0_train_native,
        labels_train,
        readouts,
        conc_scale,
        worker_seed,
        n_restarts,
        n_steps,
    ) = args_tuple

    rng = np.random.default_rng(worker_seed)
    model = ConsensusModel(Path(calib_path_str))
    n = len(nodes)
    t_start, t_end = 9.0, 0.15
    hof = {}

    for _ in range(n_restarts):
        specs: List[Optional[PeptideSpec]] = [None] * n
        pads: List[str] = [""] * n
        raws: List[str] = [""] * n
        used = set()
        for i in range(n):
            pool = pools_by_node[i]
            for t in range(50):
                sp = pool[int(rng.integers(0, len(pool)))]
                p_i, r_i = build_sequence(sp)
                if r_i not in used or t == 49:
                    specs[i], pads[i], raws[i] = sp, p_i, r_i
                    used.add(r_i)
                    break

        mat = [[model.predict_pair(pads[i], pads[j]) for j in range(n)] for i in range(n)]
        cur_loss, cur_m = _score_banana_sa(
            mat, nodes, target_edges, hierarchies, off_max_pkd, homo_max_pkd,
            c0_train_native, labels_train, readouts, conc_scale,
        )
        hof[tuple(raws)] = (cur_loss, tuple(raws), tuple(pads), cur_m)

        for step in range(n_steps):
            frac = step / max(n_steps - 1, 1)
            temp = t_start * ((t_end / t_start) ** frac)
            k = int(rng.integers(0, n))
            pool_k = pools_by_node[k]
            if rng.random() < 0.32 and pool_k:
                cand_spec = pool_k[int(rng.integers(0, len(pool_k)))]
            else:
                cand_spec = mutate_spec(specs[k], lengths_by_node[k], "mixed", rng)

            cand_pad, cand_raw = build_sequence(cand_spec)
            if cand_raw == raws[k] or any(cand_raw == raws[j] for j in range(n) if j != k):
                continue

            row_k = [model.predict_pair(cand_pad, cand_pad) if j == k else model.predict_pair(cand_pad, pads[j]) for j in range(n)]
            new_mat = [r[:] for r in mat]
            for j in range(n):
                new_mat[k][j] = row_k[j]
                new_mat[j][k] = row_k[j]

            new_loss, new_m = _score_banana_sa(
                new_mat, nodes, target_edges, hierarchies, off_max_pkd, homo_max_pkd,
                c0_train_native, labels_train, readouts, conc_scale,
            )
            delta = new_loss - cur_loss
            if delta <= 0.0 or rng.random() < math.exp(-delta / max(temp, 1e-6)):
                specs[k], pads[k], raws[k] = cand_spec, cand_pad, cand_raw
                mat, cur_loss, cur_m = new_mat, new_loss, new_m
                rk = tuple(raws)
                if rk not in hof or new_loss < hof[rk][0]:
                    hof[rk] = (new_loss, rk, tuple(pads), new_m)

        if len(hof) > 200:
            top = sorted(hof.values(), key=lambda x: x[0])[:100]
            hof = {item[1]: item for item in top}

    top_all = sorted(hof.values(), key=lambda x: x[0])[:40]
    return [(item[0], item[2], item[3]) for item in top_all]


def build_combined_pool(model: ConsensusModel, lengths: Sequence[str], homo_min_nM: float, seed: int) -> List[PeptideSpec]:
    pool: List[PeptideSpec] = []
    seen = set()
    for idx, chg_mode in enumerate(("positive", "negative", "mixed")):
        sub = generate_node_pool(
            model,
            charge_mode=chg_mode,
            lengths=lengths,
            homo_min_nM=homo_min_nM,
            max_mixed_per_len=800,
            seed=seed + 31 * idx,
        )
        for sp in sub:
            _, raw = build_sequence(sp)
            if raw not in seen:
                seen.add(raw)
                pool.append(sp)
    return pool


def mc_stress_test_banana(
    pkd_mat: np.ndarray,
    totals_dict: Dict[str, float],
    nodes: Sequence[str],
    readouts: Sequence[str],
    conc_scale: float,
    sample: Tuple[np.ndarray, np.ndarray],
    sigma_pkd: float = 0.643,
    conc_std: float = 0.015,
    n_mc: int = 200,
    seed: int = 42,
) -> dict:
    points, labels = sample
    n = len(nodes)
    internal = list(nodes[2:])
    c0_base = np.column_stack((points, np.broadcast_to([totals_dict[k] for k in internal], (len(points), len(internal)))))
    rng = np.random.default_rng(seed)
    accs = np.empty(n_mc, dtype=float)
    for m in range(n_mc):
        noise = rng.normal(0.0, sigma_pkd, size=(n, n))
        noise = np.triu(noise)
        noise = noise + noise.T - np.diag(np.diag(noise))
        pkd_pert = np.clip(pkd_mat + noise, 2.0, 12.0)
        kd_native = (10.0 ** (9.0 - pkd_pert)) / conc_scale

        if conc_std > 0.0:
            mult = np.maximum(rng.normal(1.0, conc_std, size=(1, n)), 0.1)
            c0_run = c0_base * mult
        else:
            c0_run = c0_base

        free = solve_eq_fast_np(c0_run, kd_native)
        rspec = parse_readouts(nodes, kd_native, readouts)
        vals = eval_readouts(free, rspec)
        accs[m] = float(np.mean(vals.argmax(axis=1) == labels))

    return {
        "MC_Acc_med": float(np.median(accs)),
        "MC_Acc_p10": float(np.percentile(accs, 10)),
        "MC_Acc_p90": float(np.percentile(accs, 90)),
        "MC_Pass_80": float(np.mean(accs >= 0.80)),
        "MC_Pass_85": float(np.mean(accs >= 0.85)),
    }


def plot_banana_stages(
    topology_name: str,
    nodes: Sequence[str],
    ideal_model: dict,
    kd_native_mat: np.ndarray,
    init_totals: Dict[str, float],
    fitted_totals: Dict[str, float],
    bands: int,
    test_sample: Optional[Tuple[np.ndarray, np.ndarray]],
    stage_metrics: dict,
    out_path: Path,
):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), dpi=250)
    grid = np.linspace(0.0, INPUT_MAX_NM, 101)
    xx, yy = np.meshgrid(grid, grid)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    internal = list(nodes[2:])
    readouts = ideal_model["outputs"]

    ideal_kd_mat = np.full((len(nodes), len(nodes)), 1e9, dtype=float)
    for pair, v in ideal_model["Kd_nM"].items():
        i, j = nodes.index(pair[0]), nodes.index(pair[1])
        ideal_kd_mat[i, j] = ideal_kd_mat[j, i] = float(v)

    configs = [
        ("Ideal", ideal_kd_mat, ideal_model["total_nM"], stage_metrics["ideal"]),
        ("Selected peptides", kd_native_mat, init_totals, stage_metrics["initial"]),
        ("Refitted totals", kd_native_mat, fitted_totals, stage_metrics["optimized"]),
    ]
    colors = ("#38669b", "#b85c24")

    for ax, (title, kmat, tots, metrics) in zip(axes, configs):
        c0 = np.column_stack((xy, np.broadcast_to([tots[k] for k in internal], (xy.shape[0], len(internal)))))
        free = solve_eq_fast_np(c0, kmat)
        vals = eval_readouts(free, parse_readouts(nodes, kmat, readouts))
        delta = (vals[:, 1] - vals[:, 0]).reshape(xx.shape)
        ax.contourf(xx, yy, delta, levels=[-np.inf, 0.0, np.inf], colors=("#e6eff9", "#faedde"))
        if delta.min() < 0.0 < delta.max():
            ax.contour(xx, yy, delta, levels=[0.0], colors="black", linewidths=1.0)
        for arc in arc_samples(GEOMETRY, bands, 501):
            ax.plot(arc[:, 0], arc[:, 1], "--", color="#13783b", lw=1.0)
        if test_sample is not None:
            pts, lbls = test_sample
            for c_idx, col in enumerate(colors):
                ax.scatter(*pts[lbls == c_idx].T, s=1.8, c=col, alpha=0.35, linewidths=0, rasterized=True)
            acc_str = f"Grid {metrics['uniform_test']['accuracy']:.1%} | Banana {metrics['banana_test']['accuracy']:.1%}"
        else:
            acc_str = f"Grid {metrics['uniform_test']['accuracy']:.1%}"
        ax.set(xlabel="Total X (nM)", aspect="equal", xlim=(0, INPUT_MAX_NM), ylim=(0, INPUT_MAX_NM))
        ax.set_title(f"{title}\n{acc_str}", fontsize=8.5)
    axes[0].set_ylabel("Total Y (nM)")
    fig.suptitle(f"{topology_name} ({len(nodes)} species, {len(ideal_model['Kd_nM'])} target edges; -1->{readouts[0]}, +1->{readouts[1]})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _worker_refit_banana(job):
    calib_path_str, idx, pads_t, sa_m, ideal, nodes, readouts, s, uniform_data, n_restarts, fast_mode = job
    model = ConsensusModel(Path(calib_path_str))
    n = len(nodes)
    pads = list(pads_t)
    raws = [p.lstrip("-") for p in pads]
    mat = [[model.predict_pair(pads[i], pads[j]) for j in range(n)] for i in range(n)]
    pkd_mat = np.array([[mat[i][j]["pKd"] for j in range(n)] for i in range(n)], dtype=float)
    kd_real_nM = 10.0 ** (9.0 - pkd_mat)
    kd_native_nM = kd_real_nM / s

    init_val = eval_banana_sample(kd_native_nM, ideal["total_nM"], nodes, readouts, uniform_data["uniform_validation"], ideal["training_scale_nM"])
    fitted_tot = refit_totals_banana(
        kd_native_nM, ideal, nodes, uniform_data, n_restarts=n_restarts, seed=915 + idx, fast_mode=fast_mode
    )
    opt_val = eval_banana_sample(kd_native_nM, fitted_tot, nodes, readouts, uniform_data["uniform_validation"], ideal["training_scale_nM"])
    if (opt_val["correct"], -opt_val["training_scale_CE"]) <= (init_val["correct"], -init_val["training_scale_CE"]):
        fitted_tot = dict(ideal["total_nM"])
        opt_val = init_val

    score_key = (opt_val["correct"], -opt_val["training_scale_CE"], -sa_m["graph_loss"])
    return {
        "score_key": score_key,
        "idx": idx,
        "pads_t": pads_t,
        "pads": pads,
        "raws": raws,
        "mat": mat,
        "pkd_mat": pkd_mat,
        "kd_real_nM": kd_real_nM,
        "kd_native_nM": kd_native_nM,
        "fitted_tot": fitted_tot,
        "sa_m": sa_m,
    }


def main():
    p = argparse.ArgumentParser(description="De novo coiled-coil peptide design for Banana classifier topologies.")
    p.add_argument("--topology", type=str, default="dimer_3_compact", choices=list(FIXED_TOPOLOGIES.keys()))
    p.add_argument("--conc-scale", type=float, default=None, help="Scale [0,200] nM & Kd by s (default: 15 for dimer_3_compact, 10 otherwise).")
    p.add_argument("--allow-5mer", action="store_true")
    p.add_argument("--off-max-pkd", type=float, default=4.1)
    p.add_argument("--homo-max-pkd", type=float, default=4.0)
    p.add_argument("--restarts", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--top-fit", type=int, default=30)
    p.add_argument("--n-mc", type=int, default=200)
    p.add_argument("--conc-std", type=float, default=0.015)
    p.add_argument("--banana-mat", type=Path, default=None, help="Optional path to korenkov-es/banana/data/banana.mat for 4900-pt test eval.")
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB_PATH)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "banana_designs")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    topo_name = args.topology
    ideal = FIXED_TOPOLOGIES[topo_name]
    s = float(args.conc_scale if args.conc_scale is not None else (15.0 if topo_name == "dimer_3_compact" else 10.0))
    mer_tag = "5mer" if args.allow_5mer else "4mer"
    out_dir = args.out_dir / f"{topo_name}_{mer_tag}_s{s:g}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Exact node ordering matching banana/core/model.py: ["X", "Y", *sorted(ideal["total_nM"])]
    nodes = ["X", "Y", *sorted(ideal["total_nM"].keys())]
    n = len(nodes)
    bands = int(ideal["bands"])
    readouts = list(ideal["outputs"])
    edges_pairs = list(ideal["Kd_nM"].keys())

    uniform_data = make_uniform_samples(bands)
    banana_data = load_banana_mat(args.banana_mat) if (args.banana_mat is not None and args.banana_mat.exists()) else None

    ideal_kd_mat = np.full((n, n), 1e9, dtype=float)
    for pair, val in ideal["Kd_nM"].items():
        i, j = nodes.index(pair[0]), nodes.index(pair[1])
        ideal_kd_mat[i, j] = ideal_kd_mat[j, i] = float(val)
    ideal_uni_test = eval_banana_sample(ideal_kd_mat, ideal["total_nM"], nodes, readouts, uniform_data["uniform_test"], ideal["training_scale_nM"])
    print(f"=== [{topo_name}] (nodes={n}, edges={len(edges_pairs)}, conc_scale={s}, readouts={readouts}) ===")
    print(f"Ideal uniform_test: acc={ideal_uni_test['accuracy']:.2%}, gap_med={ideal_uni_test['median_gap_nM']:.3g} nM")
    if banana_data is not None:
        ideal_ban_test = eval_banana_sample(ideal_kd_mat, ideal["total_nM"], nodes, readouts, banana_data["test"], ideal["training_scale_nM"])
        print(f"Ideal banana_test : acc={ideal_ban_test['accuracy']:.2%}")

    model = ConsensusModel(args.calib)
    lens = ["4mer", "5mer"] if args.allow_5mer else ["4mer"]
    lengths_by_node = [lens for _ in nodes]
    homo_cutoff_nM = (10.0 ** (-args.homo_max_pkd)) * 1e9
    pool = build_combined_pool(model, lens, homo_min_nM=min(homo_cutoff_nM, 45_000.0), seed=args.seed)
    pools_by_node = [pool for _ in nodes]
    print(f"Combined pool size per node: {len(pool)} (lengths={lens})")

    target_edges = []
    for pair in edges_pairs:
        u, v = nodes.index(pair[0]), nodes.index(pair[1])
        if u > v:
            u, v = v, u
        kd_real_nM = float(ideal["Kd_nM"][pair]) * s
        ideal_pkd = 9.0 - math.log10(max(kd_real_nM, 1e-3))
        target_edges.append((u, v, ideal_pkd))

    hierarchies = []
    for a in range(len(target_edges)):
        for b_idx in range(len(target_edges)):
            if a == b_idx:
                continue
            u1, v1, p1 = target_edges[a]
            u2, v2, p2 = target_edges[b_idx]
            if p1 - p2 >= 0.35:
                hierarchies.append((u1, v1, u2, v2, min(0.75 * (p1 - p2), 1.4)))

    # Fast SA training set: 15x15 uniform + arc boundary offsets (+/- 5 nM)
    gx_sa, gy_sa = np.meshgrid(*[np.linspace(0.0, INPUT_MAX_NM, 15)] * 2)
    xy_sa = np.column_stack((gx_sa.ravel(), gy_sa.ravel()))
    arc_pts = []
    for arc in arc_samples(GEOMETRY, bands, 121):
        sub = arc[::6]
        arc_pts.append(np.clip(sub[:, :2] + 5.0 * sub[:, 2:4], 0.0, INPUT_MAX_NM))
        arc_pts.append(np.clip(sub[:, :2] - 5.0 * sub[:, 2:4], 0.0, INPUT_MAX_NM))
    xy_train_sa = np.vstack([xy_sa, *arc_pts])
    labels_train_sa = band_labels(xy_train_sa, bands)
    internal = list(nodes[2:])
    c0_train_sa = np.column_stack((
        xy_train_sa,
        np.broadcast_to([ideal["total_nM"][k] for k in internal], (xy_train_sa.shape[0], len(internal))),
    ))

    t0 = time.time()
    workers = max(1, min(args.workers, args.restarts))
    r_per_w = [args.restarts // workers + (1 if i < args.restarts % workers else 0) for i in range(workers)]
    jobs = [
        (
            str(args.calib), pools_by_node, nodes, target_edges, hierarchies, lengths_by_node,
            args.off_max_pkd, args.homo_max_pkd, c0_train_sa, labels_train_sa, readouts, s,
            args.seed + 1009 * idx, r_cnt, args.steps,
        )
        for idx, r_cnt in enumerate(r_per_w) if r_cnt > 0
    ]

    raw_results = []
    if workers == 1:
        raw_results = _worker_sa_banana(jobs[0])
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for chunk in ex.map(_worker_sa_banana, jobs):
                raw_results.extend(chunk)

    dedup = {}
    for loss, pads_t, m in raw_results:
        if pads_t not in dedup or loss < dedup[pads_t][0]:
            dedup[pads_t] = (loss, pads_t, m)
    top_cands = sorted(dedup.values(), key=lambda x: x[0])[: args.top_fit]
    print(f"SA finished in {time.time() - t0:.1f}s ({len(dedup)} unique designs). Refitting totals on top {len(top_cands)}...")

    t_ref = time.time()
    refit_jobs = [
        (str(args.calib), idx, pads_t, sa_m, ideal, nodes, readouts, s, uniform_data, 2, True)
        for idx, (_, pads_t, sa_m) in enumerate(top_cands)
    ]
    if workers == 1:
        stage1 = [_worker_refit_banana(j) for j in refit_jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            stage1 = list(ex.map(_worker_refit_banana, refit_jobs))
    stage1.sort(key=lambda b: b["score_key"], reverse=True)
    polish_jobs = [
        (str(args.calib), b["idx"], b["pads_t"], b["sa_m"], ideal, nodes, readouts, s, uniform_data, 8, False)
        for b in stage1[:min(3, len(stage1))]
    ]
    if workers == 1:
        stage2 = [_worker_refit_banana(j) for j in polish_jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(polish_jobs))) as ex:
            stage2 = list(ex.map(_worker_refit_banana, polish_jobs))
    best_bundle = max(stage1[:1] + stage2, key=lambda b: b["score_key"])
    print(f"Parallel refit finished in {time.time() - t_ref:.1f}s.")

    b = best_bundle
    stages = {
        "ideal": {k: eval_banana_sample(ideal_kd_mat, ideal["total_nM"], nodes, readouts, uniform_data[k], ideal["training_scale_nM"]) for k in uniform_data},
        "initial": {k: eval_banana_sample(b["kd_native_nM"], ideal["total_nM"], nodes, readouts, uniform_data[k], ideal["training_scale_nM"]) for k in uniform_data},
        "optimized": {k: eval_banana_sample(b["kd_native_nM"], b["fitted_tot"], nodes, readouts, uniform_data[k], ideal["training_scale_nM"]) for k in uniform_data},
    }
    test_sample = None
    if banana_data is not None:
        test_sample = banana_data["test"]
        stages["ideal"]["banana_test"] = eval_banana_sample(ideal_kd_mat, ideal["total_nM"], nodes, readouts, test_sample, ideal["training_scale_nM"])
        stages["initial"]["banana_test"] = eval_banana_sample(b["kd_native_nM"], ideal["total_nM"], nodes, readouts, test_sample, ideal["training_scale_nM"])
        stages["optimized"]["banana_test"] = eval_banana_sample(b["kd_native_nM"], b["fitted_tot"], nodes, readouts, test_sample, ideal["training_scale_nM"])

    print(f"\n=== Best CC Design for Banana '{topo_name}' (conc_scale={s}, lengths={lens}) ===")
    print(f"Grid 160x160 : Ideal={stages['ideal']['uniform_test']['accuracy']:.2%} | Init={stages['initial']['uniform_test']['accuracy']:.2%} | Refitted={stages['optimized']['uniform_test']['accuracy']:.2%}")
    if banana_data is not None:
        print(f"Banana 4900  : Ideal={stages['ideal']['banana_test']['accuracy']:.2%} | Init={stages['initial']['banana_test']['accuracy']:.2%} | Refitted={stages['optimized']['banana_test']['accuracy']:.2%}")
    print(f"Median gap   : {stages['optimized']['uniform_test']['median_gap_nM']:.4g} nM (native) = {stages['optimized']['uniform_test']['median_gap_nM']*s:.4g} nM (scaled)")
    print(f"Worst off/homo pKd: {b['sa_m']['worst_off_pKd']:.2f} / {b['sa_m']['worst_homo_pKd']:.2f}")
    print("Refitted Totals (native nM):", {k: round(float(v), 2) for k, v in b["fitted_tot"].items()})
    print("Refitted Totals (scaled nM):", {k: round(float(v) * s, 1) for k, v in b["fitted_tot"].items()})

    print("Target edges (orient / shift / slippage):")
    edge_audit = {}
    for pair in edges_pairs:
        u, v = nodes.index(pair[0]), nodes.index(pair[1])
        sc = b["mat"][u][v]
        slip, slip_type = model.slippage_margin(b["pads"][u], b["pads"][v])
        orient = sc.get("iCipa_best_orient", "PAR")
        shift = int(sc.get("iCipa_best_shift", 0))
        edge_audit[pair] = {
            "Kd_real_nM": float(b["kd_real_nM"][u, v]),
            "Kd_native_nM": float(b["kd_native_nM"][u, v]),
            "ideal_native_nM": float(ideal["Kd_nM"][pair]),
            "pKd": float(b["pkd_mat"][u, v]),
            "orient": orient,
            "shift": shift,
            "slip_margin": float(slip),
            "slip_type": slip_type,
        }
        print(f"  {pair:3s}: Kd_nat={b['kd_native_nM'][u, v]:6.2f} nM (ideal={ideal['Kd_nM'][pair]:6.2f}), pKd={b['pkd_mat'][u, v]:.2f}, {orient} shift={shift:+d}, slip={slip:+.2f} ({slip_type})")

    print("Sequences:")
    for i, nm in enumerate(nodes):
        print(f"  {nm:2s} ({len(b['raws'][i])}aa, chg={net_charge(b['raws'][i]):+d}, eps={calc_eps280(b['raws'][i])}): {b['raws'][i]}")

    mc_grid = mc_stress_test_banana(
        b["pkd_mat"], b["fitted_tot"], nodes, readouts, s, uniform_data["uniform_validation"],
        sigma_pkd=model.loo_rmse, conc_std=args.conc_std, n_mc=args.n_mc, seed=args.seed,
    )
    print(f"MC Grid (n={args.n_mc}, sigma={model.loo_rmse:.3f}, c_std={args.conc_std}): med={mc_grid['MC_Acc_med']:.2%}, p10={mc_grid['MC_Acc_p10']:.2%}, p90={mc_grid['MC_Acc_p90']:.2%}")
    mc_ban = None
    if test_sample is not None:
        mc_ban = mc_stress_test_banana(
            b["pkd_mat"], b["fitted_tot"], nodes, readouts, s, test_sample,
            sigma_pkd=model.loo_rmse, conc_std=args.conc_std, n_mc=args.n_mc, seed=args.seed,
        )
        print(f"MC Banana (4900): med={mc_ban['MC_Acc_med']:.2%}, p10={mc_ban['MC_Acc_p10']:.2%}, p90={mc_ban['MC_Acc_p90']:.2%}")

    for ext in (".pdf", ".png"):
        plot_banana_stages(
            topo_name, nodes, ideal, b["kd_native_nM"], ideal["total_nM"], b["fitted_tot"],
            bands, test_sample, stages, out_dir / f"{topo_name}_performance{ext}",
        )

    # Export full N*(N+1)/2 matrix in native [0, 200] nM scale for korenkov-es/banana/refit.py
    kd_export = {
        f"{nodes[i]}{nodes[j]}": float(b["kd_native_nM"][i, j])
        for i in range(n)
        for j in range(i, n)
    }
    peptide_kd_payload = {
        "source": f"CC-any-graph de novo design ({topo_name}, {mer_tag}, conc_scale={s})",
        "Kd_nM": {topo_name: kd_export},
    }
    (out_dir / f"peptide_kd_{topo_name}.json").write_text(json.dumps(peptide_kd_payload, indent=2) + "\n")

    report = {
        "topology": topo_name,
        "conc_scale": s,
        "lengths": lens,
        "nodes": nodes,
        "outputs": readouts,
        "seqs": dict(zip(nodes, b["raws"])),
        "pads": dict(zip(nodes, b["pads"])),
        "pkd_matrix": b["pkd_mat"].tolist(),
        "fitted_total_native_nM": {k: float(v) for k, v in b["fitted_tot"].items()},
        "fitted_total_scaled_nM": {k: float(v) * s for k, v in b["fitted_tot"].items()},
        "edges": edge_audit,
        "worst_off_pKd": b["sa_m"]["worst_off_pKd"],
        "worst_homo_pKd": b["sa_m"]["worst_homo_pKd"],
        "stages": stages,
        "mc_uniform_val": mc_grid,
        "mc_banana_test": mc_ban,
    }
    (out_dir / f"{topo_name}_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved report, plots, and peptide_kd_{topo_name}.json to {out_dir}")


if __name__ == "__main__":
    main()
