"""De novo coiled-coil sequence design for 2D boundaries from korenkov-es/des_boundaries (v4)."""

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

from contourpy import contour_generator
import jax
from jax import lax, jit
import jax.numpy as jnp
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial import cKDTree

jax.config.update("jax_enable_x64", True)

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


@partial(jit, static_argnames=("it",))
def eq_conc(c0, A, it=1000):
    diag = jnp.arange(A.shape[-1])
    A = A.at[..., diag, diag].multiply(2.0)

    def _body(_, c):
        return c0 / (1.0 + jnp.einsum("...ij,...j->...i", A, c))

    return lax.fori_loop(0, it, _body, c0)


def set_params():
    mpl.rcParams["figure.figsize"] = (210 / 25.4, 297 / 25.4)
    mpl.rcParams["figure.facecolor"] = "none"
    mpl.rcParams["axes.facecolor"] = "none"
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["axes.labelsize"] = 6
    mpl.rcParams["xtick.labelsize"] = 6
    mpl.rcParams["ytick.labelsize"] = 6
    mpl.rcParams["legend.fontsize"] = 7


def create_fig(xsize=None, ysize=None, n_rows=1, n_cols=1):
    set_params()
    if xsize is None:
        xsize = 0.85 if n_cols > 1 else 1 / 4
    if ysize is None:
        ysize = 0.85 if n_rows > 1 else 1 / 6
    fig = plt.figure()
    grid = fig.add_gridspec(n_rows, n_cols, left=0.1, bottom=0.1, right=0.1 + xsize, top=0.1 + ysize)
    return fig, grid.subplots()


ALL_GATES = [
    "line",
    "circle",
    "quadratic",
    "cubic",
    "full_circle",
    "full_circle_2",
    "angle",
    "angle_2",
    "bandpass",
    "bandpass_2",
]

TOPOLOGIES = {
    "line": dict(nodes=("X", "Y", "A", "B"), edges=("AB", "XA", "YB")),
    "circle": dict(nodes=("X", "Y", "A", "B", "C", "D"), edges=("AC", "BC", "XC", "XD", "YB")),
    "quadratic": dict(nodes=("X", "Y", "A", "B", "C"), edges=("AC", "BC", "XB", "XC", "YC")),
    "cubic": dict(nodes=("X", "Y", "A", "B", "C", "D"), edges=("AB", "AD", "CD", "XA", "XC", "XD", "YB")),
    "full_circle": dict(nodes=("X", "Y", "A", "B", "C", "D"), edges=("XB", "XC", "YC", "YD", "AC", "BC", "CD")),
    "full_circle_2": dict(
        nodes=("X", "Y", "A", "C", "D", "E", "G"),
        edges=("XD", "XG", "YC", "YE", "AC", "AG", "CE", "DG"),
    ),
    "angle": dict(nodes=("X", "Y", "A", "C", "E"), edges=("XC", "YE", "AC", "AE", "CE")),
    "angle_2": dict(
        nodes=("X", "Y", "A", "B", "E", "F", "G"),
        edges=("XY", "XE", "YF", "AB", "AE", "BG", "EF", "FG"),
    ),
    "bandpass": dict(nodes=("X", "Y", "A", "B", "C"), edges=("XA", "XB", "YC", "AB", "AC", "BC")),
    "bandpass_2": dict(nodes=("X", "Y", "A", "B", "C"), edges=("XY", "XA", "XB", "YC", "AB", "AC", "BC")),
}

STARTING_SEEDS = {
    "quadratic": (
        {"XB": 1.0010652542114258, "BC": 2.281172752380371, "XC": 3.5668487548828125,
         "YC": 11.817999839782715, "AC": 1.0663700103759766},
        {"A": 231.3270263671875, "B": 97.80352020263672, "C": 256.8213806152344},
    ),
    "cubic": (
        {"XA": 91.08134460449219, "AB": 90.83689880371094, "CD": 2.6216490268707275, "YD": 1.0358837842941284,
         "XC": 1.155128002166748, "AD": 4.06572151184082, "YB": 1.1823222637176514},
        {"A": 162.6712188720703, "B": 20.12062644958496, "C": 60.63164138793945, "D": 183.9221954345703},
    ),
    "full_circle": (
        {"XB": 1.0, "XC": 6.5619658207637315, "YC": 6.5619658207641836, "YD": 1.000000000000002,
         "AC": 1.0000000000000002, "BC": 6.8086315365657555, "CD": 6.808631536566877},
        {"A": 324.9227986910585, "B": 103.79024815914606, "C": 315.85145665091954, "D": 103.7902481591466},
    ),
    "full_circle_2": (
        {"XD": 1.0, "XG": 3.4682258511743993, "YC": 3.467787909332605, "YE": 1.0, "AC": 1.0000000000000073,
         "AG": 1.0, "CE": 3.6762066181612996, "DG": 3.6766395586815643},
        {"A": 353.45203911854117, "C": 172.9761826445396, "D": 104.1326629887872,
         "E": 104.13292799810259, "G": 172.986042314218},
    ),
    "angle": (
        {"XC": 1.7498635399722011, "YE": 1.7498635399719276, "AC": 79.15394957585218,
         "AE": 79.15394957584402, "CE": 1.0000000000000038},
        {"A": 66.41112440901443, "C": 191.47934304115967, "E": 191.47934304115898},
    ),
    "angle_2": (
        {"XY": 4.211920825851432, "XE": 1.234885049779596, "YF": 1.1154574942272528, "AB": 2.2023142053814535,
         "AE": 336.9917823381263, "BG": 1.0000000000000042, "EF": 1.000000000000011, "FG": 24.300308243276362},
        {"A": 330.1040524367706, "B": 348.0877793262765, "E": 60.03574221555958,
         "F": 59.56207486917893, "G": 62.72052078086879},
    ),
    "bandpass": (
        {"XA": 50.68336067224629, "XB": 1.0000000000047613, "YC": 1.0, "AB": 33.44078126840161,
         "AC": 185.20229340647523, "BC": 3.726120209205433},
        {"A": 84.67795090431774, "B": 219.7827364670203, "C": 299.51165875817435},
    ),
    "bandpass_2": (
        {"XY": 3.0756601299814568, "XA": 41.57373229720732, "XB": 1.000000001836519, "YC": 1.0000000161825586,
         "AB": 31.079879675683074, "AC": 162.6459717613409, "BC": 3.986718692771153},
        {"A": 84.18489562291869, "B": 168.2695100923247, "C": 237.95434643567992},
    ),
}

FIT_SETTINGS = {
    "line": dict(starts=4, max_nfev=160, min_slope=0.55),
    "circle": dict(starts=5, max_nfev=180, min_slope=0.40),
    "quadratic": dict(starts=6, max_nfev=220, min_slope=0.10),
    "cubic": dict(starts=8, max_nfev=250, min_slope=0.10),
    "full_circle": dict(starts=4, max_nfev=220, min_slope=0.12),
    "angle": dict(starts=4, max_nfev=220, min_slope=0.05),
    "bandpass": dict(starts=4, max_nfev=220, min_slope=0.12),
}


def base_boundary_name(name: str) -> str:
    return name[:-2] if name.endswith("_2") else name


def boundary_fn(name: str, x, y, max_in: float = 200.0):
    b = base_boundary_name(name)
    s = max_in / 200.0
    if b == "line":
        return y > x
    if b == "circle":
        return x ** 2 + (y - max_in) ** 2 > max_in ** 2
    if b == "quadratic":
        return y > 4.0 * x * (1.0 - x / max_in)
    if b == "cubic":
        return y > 0.5 * max_in * (1.0 + (2.0 * x / max_in - 1.0) ** 3)
    if b == "full_circle":
        c, r = 100.0 * s, 30.0 * s
        return (x - c) ** 2 + (y - c) ** 2 > r ** 2
    if b == "angle":
        return (y > 0.5 * x) & (y < 2.0 * x)
    if b == "bandpass":
        return (y > 1.25 * x + 10.0 * s) & (y < 1.25 * x + 60.0 * s)
    raise ValueError(f"Unknown boundary: {name}")


def curves(name: str, max_in: float = 200.0, n_pts: int = 1001) -> Tuple[np.ndarray, ...]:
    b = base_boundary_name(name)
    s = max_in / 200.0
    t = np.linspace(0.0, 1.0, n_pts)
    if b == "line":
        return (np.column_stack((max_in * t, max_in * t)),)
    if b == "circle":
        return (np.column_stack((max_in * np.sin(np.pi * t / 2.0), max_in * (1.0 - np.cos(np.pi * t / 2.0)))),)
    if b == "quadratic":
        x = max_in * t
        return (np.column_stack((x, 4.0 * x * (1.0 - x / max_in))),)
    if b == "cubic":
        x = max_in * t
        return (np.column_stack((x, 0.5 * max_in * (1.0 + (2.0 * x / max_in - 1.0) ** 3))),)
    if b == "full_circle":
        c, r = 100.0 * s, 30.0 * s
        semicircle = r * np.column_stack((-np.cos(np.pi * t), np.sin(np.pi * t)))
        return (c + semicircle, c - semicircle)
    if b == "angle":
        return (
            np.column_stack((max_in * t, 0.5 * max_in * t)),
            np.column_stack((max_in * (1.0 - t) / 2.0, max_in * (1.0 - t))),
        )
    if b == "bandpass":
        bx_lo = (max_in - 10.0 * s) * t / 1.25
        bx_hi = (max_in - 60.0 * s) * (1.0 - t) / 1.25
        return (
            np.column_stack((bx_lo, 1.25 * bx_lo + 10.0 * s)),
            np.column_stack((bx_hi, 1.25 * bx_hi + 60.0 * s)),
        )
    raise ValueError(f"Unknown boundary: {name}")


def contour_geometry(name: str, max_in: float = 200.0, step: int = 25, interior: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    b = base_boundary_name(name)
    points, normals = [], []
    for branch in curves(b, max_in):
        curve = branch[::step]
        if interior and len(curve) > 2:
            curve = curve[1:-1]
        if b == "angle":
            curve = curve[np.any(curve != 0.0, axis=1)]
        tangent = np.gradient(curve, axis=0)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        points.append(curve)
        normals.append(-normal if b == "circle" else normal)
    return np.vstack(points), np.vstack(normals)


def contour_samples(name: str, max_in: float = 200.0, step: int = 25, offset: float = 3.0, interior: bool = False) -> np.ndarray:
    curve, normal = contour_geometry(name, max_in=max_in, step=step, interior=interior)
    return np.vstack([
        curve,
        np.clip(curve + offset * normal, 0.0, max_in),
        np.clip(curve - offset * normal, 0.0, max_in),
    ])


def fit_ideal_gate(name: str, kd_min_native: float = 1.0, kd_max_native: float = 1000.0, seed: int = 42):
    graph = TOPOLOGIES[name]
    edges = graph["edges"]
    inputs = ("X", "Y")
    internal = [nd for nd in graph["nodes"] if nd not in inputs]
    nodes = [*inputs, *internal]
    row, col = np.array([[nodes.index(nd) for nd in pair] for pair in edges]).T
    out_idx = nodes.index("A")
    b = base_boundary_name(name)
    opts = FIT_SETTINGS[b]

    pts = jnp.asarray(contour_samples(b, max_in=200.0, step=25, offset=3.0, interior=True))

    def residual(log_params):
        kd = jnp.exp(log_params[: len(edges)])
        totals = jnp.broadcast_to(jnp.exp(log_params[len(edges) :]), (len(pts), len(internal)))
        aff = jnp.zeros((len(nodes), len(nodes))).at[row, col].set(1.0 / kd).at[col, row].set(1.0 / kd)
        free = eq_conc(jnp.column_stack((pts, totals)), aff, 180)
        center, plus, minus = jnp.split(free[:, out_idx], 3)
        slope = (plus - minus) / 6.0
        offset = (center - 50.0) / jnp.maximum(slope, 0.03)
        return jnp.r_[offset, 25.0 * jnp.minimum(slope - opts["min_slope"], 0.0)]

    fun = jax.jit(residual)
    jac = jax.jit(jax.jacfwd(residual))
    kd_b = (kd_min_native, kd_max_native)
    tot_b = (1.0, 400.0)
    bounds = np.log([kd_b] * len(edges) + [tot_b] * len(internal)).T
    rng = np.random.default_rng(seed)
    fits = []
    for index in range(opts["starts"]):
        log_kd = rng.uniform(np.log(kd_b[0]), np.log(min(100.0, kd_b[1])), len(edges))
        log_tot = rng.uniform(np.log(60.0), np.log(360.0), len(internal))
        start = np.clip(np.r_[log_kd, log_tot], *bounds)
        if index == 0 and name in STARTING_SEEDS:
            skd, stot = STARTING_SEEDS[name]
            start = np.clip(np.log([skd[e] for e in edges] + [stot[nd] for nd in internal]), *bounds)
        fits.append(least_squares(lambda z: np.asarray(fun(z)), start, jac=lambda z: np.asarray(jac(z)), bounds=bounds, max_nfev=opts["max_nfev"]))
    best_x = np.exp(min(fits, key=lambda f: f.cost).x)
    return dict(zip(edges, best_x[: len(edges)])), dict(zip(internal, best_x[len(edges) :]))


def equilibrium_matrix_jax(kd_mat_nM: np.ndarray, totals_nM: Dict[str, float], gx: np.ndarray, gy: np.ndarray, nodes: Sequence[str], output: str = "A") -> np.ndarray:
    affinity = 1.0 / np.maximum(kd_mat_nM, 1e-6)
    c0 = np.column_stack((gx.ravel(), gy.ravel(), *[np.full(gx.size, totals_nM[nm]) for nm in nodes[2:]]))
    free = np.asarray(eq_conc(jnp.asarray(c0), jnp.asarray(affinity)))
    binding = affinity + np.diag(affinity.diagonal())
    eye = np.eye(len(nodes))
    for _ in range(50):
        factor = 1.0 + free @ binding
        if np.allclose(free * factor, c0, atol=1e-9, rtol=1e-10):
            return free[:, nodes.index(output)].reshape(gx.shape)
        jac = free[:, :, None] * binding + factor[:, :, None] * eye
        step = np.linalg.solve(jac, (free * factor - c0)[..., None])[..., 0]
        free = np.clip(free - step, 0.0, c0)
    raise RuntimeError("Equilibrium concentrations did not satisfy mass balance.")


def solve_eq_batch_np(c0_nM: np.ndarray, kd_mat_nM: np.ndarray) -> np.ndarray:
    n = c0_nM.shape[1]
    k_mat = 1.0 / np.maximum(kd_mat_nM, 1e-6)
    a_mod = k_mat.copy()
    np.fill_diagonal(a_mod, 2.0 * np.diag(k_mat))

    x = c0_nM.copy()
    for _ in range(36):
        x = 0.5 * x + 0.5 * (c0_nM / (1.0 + x @ a_mod))

    eye = np.eye(n, dtype=float)[None, :, :]
    active = c0_nM > 0.0
    for _ in range(12):
        factor = 1.0 + x @ a_mod
        f = x * factor - c0_nM
        if np.max(np.abs(f)) < 1e-8:
            break
        jac = x[:, :, None] * a_mod[None, :, :] + factor[:, :, None] * eye
        dx = np.linalg.solve(jac, -f[..., None])[..., 0]
        neg = (dx < 0.0) & active
        alpha = np.where(neg, -0.95 * x / np.minimum(dx, -1e-30), 1.0)
        step = np.minimum(1.0, np.min(alpha, axis=1, keepdims=True))
        x = np.where(active, np.maximum(x + step * dx, 1e-18), 0.0)
    return x


def bce_loss(values: np.ndarray, labels: np.ndarray, thresh_nM: float, trans_nM: float) -> float:
    logits = (values - thresh_nM) / trans_nM
    return float(np.logaddexp(0.0, np.where(labels, -logits, logits)).mean())


def contour_error(delta: np.ndarray, gx: np.ndarray, gy: np.ndarray, tree: cKDTree, conc_scale: float = 1.0) -> float:
    lines = contour_generator(x=gx[0], y=gy[:, 0], z=delta).lines(0)
    if not lines:
        return float(np.inf)
    step_nm = max(0.5, 0.5 * conc_scale)
    samples = []
    for line in lines:
        for a, b in zip(line, line[1:]):
            count = max(2, int(np.ceil(np.linalg.norm(b - a) / step_nm)) + 1)
            samples.extend(np.linspace(a, b, count))
    return float((cKDTree(samples).query(tree.data)[0].mean() + tree.query(samples)[0].mean()) / 2.0)


def classification_extras(pred: np.ndarray, labels: np.ndarray, err_nM: float, max_in: float) -> dict:
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


def evaluate_full_2d(name: str, kd_mat_nM: np.ndarray, totals_nM: Dict[str, float], nodes: Sequence[str], conc_scale: float, use_jax: bool = True):
    max_in = 200.0 * conc_scale
    thresh = 50.0 * conc_scale
    trans = 5.0 * conc_scale
    val_grid = np.meshgrid(*[np.linspace(0.0, max_in, 121)] * 2)
    if use_jax:
        values = equilibrium_matrix_jax(kd_mat_nM, totals_nM, *val_grid, nodes, output="A")
    else:
        gx, gy = val_grid
        c0 = np.column_stack((gx.ravel(), gy.ravel(), *[np.full(gx.size, totals_nM[nm]) for nm in nodes[2:]]))
        values = solve_eq_batch_np(c0, kd_mat_nM)[:, nodes.index("A")].reshape(gx.shape)
    labels = boundary_fn(name, *val_grid, max_in=max_in)
    delta = values - thresh
    tree = cKDTree(np.vstack(curves(name, max_in)))
    err = contour_error(delta, *val_grid, tree, conc_scale=conc_scale)
    metrics = dict(
        loss=bce_loss(values, labels, thresh, trans),
        accuracy=float(np.mean((delta > 0.0) == labels)),
        error_nM=err if np.isfinite(err) else None,
    )
    metrics.update(classification_extras(delta > 0.0, labels, err, max_in))
    return metrics, values, val_grid


def fit_concentrations_full(
    name: str,
    kd_mat_nM: np.ndarray,
    init_totals_nM: Dict[str, float],
    nodes: Sequence[str],
    conc_scale: float,
    contour_fit: bool = False,
    tot_max_native: float = 400.0,
):
    max_in = 200.0 * conc_scale
    thresh = 50.0 * conc_scale
    trans = 5.0 * conc_scale
    train_grid = np.meshgrid(*[np.linspace(0.0, max_in, 31)] * 2)
    gx, gy = train_grid
    labels = boundary_fn(name, gx, gy, max_in=max_in).ravel()
    internal = list(nodes[2:])
    out_idx = nodes.index("A")
    xy = np.column_stack((gx.ravel(), gy.ravel()))
    n_grid = xy.shape[0]

    if contour_fit:
        pts = contour_samples(name, max_in=max_in, step=25, offset=4.0 * conc_scale, interior=True)
        n_c = len(pts) // 3
        plus_xy = pts[n_c : 2 * n_c]
        minus_xy = pts[2 * n_c :]
        plus_lbl = boundary_fn(name, plus_xy[:, 0], plus_xy[:, 1], max_in=max_in)
        minus_lbl = boundary_fn(name, minus_xy[:, 0], minus_xy[:, 1], max_in=max_in)
        all_xy = np.vstack((xy, pts))
    else:
        plus_lbl = minus_lbl = None
        all_xy = xy

    def objective(log_totals):
        tots = np.exp(log_totals)
        c0 = np.column_stack((all_xy, np.broadcast_to(tots, (all_xy.shape[0], len(internal)))))
        free = solve_eq_batch_np(c0, kd_mat_nM)[:, out_idx]
        if not contour_fit:
            return bce_loss(free, labels, thresh, trans)
        grid_v = free[:n_grid]
        center, plus, minus = np.split(free[n_grid:], 3)
        c_loss = float(np.mean(((center - thresh) / trans) ** 2))
        band_bce = 0.5 * (bce_loss(plus, plus_lbl, thresh, trans) + bce_loss(minus, minus_lbl, thresh, trans))
        return c_loss + 0.8 * band_bce + 0.25 * bce_loss(grid_v, labels, thresh, trans)

    bounds = [(np.log(1.0 * conc_scale), np.log(tot_max_native * conc_scale))] * len(internal)
    best_res = None
    for scale_mult in (1.0, 0.5, 2.0):
        start = np.clip(np.log([init_totals_nM[k] * scale_mult for k in internal]), [b[0] for b in bounds], [b[1] for b in bounds])
        res = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        if best_res is None or res.fun < best_res.fun:
            best_res = res
    return dict(zip(internal, np.exp(best_res.x)))


def _hinge_sq(deficit: float, w_sq: float = 25.0, w_lin: float = 5.0) -> float:
    if deficit <= 0.0:
        return 0.0
    return w_sq * (deficit * deficit) + w_lin * deficit


def _score_2d_sa(
    pair_mat: List[List[dict]],
    nodes: Sequence[str],
    target_edges: Sequence[Tuple[int, int, float]],
    hierarchies: Sequence[Tuple[int, int, int, int, float]],
    off_max_pkd: float,
    homo_max_pkd: float,
    c0_train_nM: np.ndarray,
    labels_train: np.ndarray,
    n_grid_pts: int,
    thresh_nM: float,
    trans_nM: float,
    cfit_mode: bool,
) -> Tuple[float, dict]:
    n = len(nodes)
    target_set = {(min(u, v), max(u, v)) for u, v, _ in target_edges}

    graph_loss = 0.0
    tol_lo = 0.22 if cfit_mode else 0.35
    tol_hi = 0.28 if cfit_mode else 0.45
    w_pkd = 4.0 if cfit_mode else 1.5
    w_shift = 40.0 if cfit_mode else 16.0
    for u, v, ideal_pkd in target_edges:
        sc = pair_mat[u][v]
        pkd = sc["pKd"]
        graph_loss += _hinge_sq((ideal_pkd - tol_lo) - pkd, 28.0, 5.0)
        graph_loss += _hinge_sq(pkd - (ideal_pkd + tol_hi), 28.0, 5.0)
        graph_loss += w_pkd * ((pkd - ideal_pkd) ** 2)
        if sc.get("iCipa_best_orient", "PAR") != "PAR":
            graph_loss += 35.0
        if sc.get("iCipa_best_shift", 0) != 0:
            graph_loss += w_shift

    for su, sv, wu, wv, min_gap in hierarchies:
        dpkd = pair_mat[su][sv]["pKd"] - pair_mat[wu][wv]["pKd"]
        graph_loss += _hinge_sq(min_gap - dpkd, 25.0, 5.0)

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
                graph_loss += _hinge_sq(pkd - cap, 45.0, 8.0)
            else:
                worst_off = max(worst_off, pkd)
                graph_loss += _hinge_sq(pkd - cap, 25.0, 5.0)

    log10_kd_mat = np.array([[pair_mat[i][j]["log10_kd"] for j in range(n)] for i in range(n)], dtype=float)
    kd_mat_nM = (10.0 ** np.minimum(log10_kd_mat, -2.0)) * 1e9
    free = solve_eq_batch_np(c0_train_nM, kd_mat_nM)
    a_vals = free[:, nodes.index("A")]

    grid_vals = a_vals[:n_grid_pts]
    curve_vals = a_vals[n_grid_pts:]
    bce = bce_loss(grid_vals, labels_train, thresh_nM, trans_nM)
    acc = float(np.mean((grid_vals > thresh_nM) == labels_train))
    curve_pen = float(np.mean(((curve_vals - thresh_nM) / trans_nM) ** 2)) if curve_vals.size else 0.0
    if cfit_mode:
        circuit_loss = 10.0 * bce + 18.0 * curve_pen
    else:
        circuit_loss = 35.0 * bce + 1.5 * curve_pen
    total_loss = graph_loss + circuit_loss

    return float(total_loss), {
        "total_loss": float(total_loss),
        "graph_loss": float(graph_loss),
        "bce_train": float(bce),
        "acc_train": float(acc),
        "worst_off_pKd": float(worst_off),
        "worst_homo_pKd": float(worst_homo),
    }


def _worker_sa_2d(args_tuple):
    (
        calib_path_str,
        pools_by_node,
        nodes,
        target_edges,
        hierarchies,
        lengths_by_node,
        off_max_pkd,
        homo_max_pkd,
        c0_train_nM,
        labels_train,
        n_grid_pts,
        thresh_nM,
        trans_nM,
        cfit_mode,
        worker_seed,
        n_restarts,
        n_steps,
    ) = args_tuple

    rng = np.random.default_rng(worker_seed)
    model = ConsensusModel(Path(calib_path_str))
    n = len(nodes)
    t_start, t_end = 8.0, 0.15
    hof = {}

    for _ in range(n_restarts):
        specs: List[Optional[PeptideSpec]] = [None] * n
        pads: List[str] = [""] * n
        raws: List[str] = [""] * n
        used = set()
        for i in range(n):
            pool = pools_by_node[i]
            for t in range(40):
                sp = pool[int(rng.integers(0, len(pool)))]
                p_i, r_i = build_sequence(sp)
                if r_i not in used or t == 39:
                    specs[i], pads[i], raws[i] = sp, p_i, r_i
                    used.add(r_i)
                    break

        mat = [[model.predict_pair(pads[i], pads[j]) for j in range(n)] for i in range(n)]
        cur_loss, cur_m = _score_2d_sa(
            mat, nodes, target_edges, hierarchies, off_max_pkd, homo_max_pkd,
            c0_train_nM, labels_train, n_grid_pts, thresh_nM, trans_nM, cfit_mode,
        )
        hof[tuple(raws)] = (cur_loss, tuple(raws), tuple(pads), cur_m)

        for step in range(n_steps):
            frac = step / max(n_steps - 1, 1)
            temp = t_start * ((t_end / t_start) ** frac)
            k = int(rng.integers(0, n))
            pool_k = pools_by_node[k]
            if rng.random() < 0.30 and pool_k:
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

            new_loss, new_m = _score_2d_sa(
                new_mat, nodes, target_edges, hierarchies, off_max_pkd, homo_max_pkd,
                c0_train_nM, labels_train, n_grid_pts, thresh_nM, trans_nM, cfit_mode,
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


def plot_topology_and_contour(
    name: str,
    nodes: Sequence[str],
    edges_pairs: Sequence[str],
    kd_mat_nM: np.ndarray,
    totals_nM: Dict[str, float],
    field: np.ndarray,
    val_grid,
    conc_scale: float,
    metrics: dict,
    worst_off: float,
    worst_homo: float,
    out_path: Path,
):
    max_in = 200.0 * conc_scale
    thresh = 50.0 * conc_scale
    fig, (network, ax) = create_fig(xsize=0.72, ysize=0.22, n_cols=2)
    angles = np.linspace(0.0, 2.0 * np.pi, len(nodes), endpoint=False)
    positions = dict(zip(nodes, np.column_stack((np.cos(angles), np.sin(angles)))))

    for pair in edges_pairs:
        a, b = pair[0], pair[1]
        p1, p2 = positions[a], positions[b]
        network.plot([p1[0], p2[0]], [p1[1], p2[1]], color="0.5", lw=1.4)
        mid = 0.44 * p1 + 0.56 * p2
        kd_val = kd_mat_nM[nodes.index(a), nodes.index(b)]
        network.text(
            mid[0], mid[1], f"{kd_val:.1f} nM", fontsize=4.8, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.85),
        )

    for nd, pt in positions.items():
        lbl = f"{nd}\n{totals_nM[nd]:.0f}" if nd in totals_nM else nd
        fc = "#dce8f7" if nd in ("X", "Y") else ("#fdebd0" if nd == "A" else "white")
        network.text(*pt, lbl, fontsize=5.5, ha="center", va="center", bbox=dict(boxstyle="circle,pad=0.25", facecolor=fc, lw=0.8))
    network.set(aspect="equal")
    network.set_title(f"Topology (off={worst_off:.1f}, homo={worst_homo:.1f} pKd)", fontsize=6.5)
    network.margins(0.22)
    network.set_axis_off()

    if field.min() < thresh < field.max():
        ax.contour(*val_grid, field, levels=[thresh], colors="#191919", linewidths=1.2)
        ax.plot([], [], color="#191919", label=f"Free A = {thresh:.0f} nM")
    for i, branch in enumerate(curves(name, max_in)):
        ax.plot(*branch.T, "--", color="#198438", lw=1.2, label="Target" if i == 0 else None)
    err_str = f"{metrics['error_nM']:.1f} nM" if metrics["error_nM"] is not None else "n/a"
    ax.set(xlabel="X (nM)", ylabel="Y (nM)", aspect="equal", xlim=(0, max_in), ylim=(0, max_in))
    ax.set_title(f"{name.capitalize()} (Acc {metrics['accuracy']:.1%}, err {err_str})", fontsize=7)
    ax.legend(loc="best")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_performance_stages(name: str, stages: dict, val_grid, conc_scale: float, out_path: Path):
    max_in = 200.0 * conc_scale
    thresh = 50.0 * conc_scale
    fig, axes = create_fig(xsize=0.86 if len(stages) > 1 else 0.34, ysize=0.22, n_cols=len(stages))
    axes = np.atleast_1d(axes)

    all_vals = np.concatenate([v.ravel() for _, v in stages.values()])
    vmin = min(float(np.min(all_vals)), thresh - 1.0)
    vmax = max(float(np.max(all_vals)), thresh + 1.0)
    norm = TwoSlopeNorm(vcenter=thresh, vmin=vmin, vmax=vmax)

    for ax, (title, (metrics, values)) in zip(axes, stages.items()):
        mesh = ax.pcolormesh(*val_grid, values, shading="nearest", cmap="RdBu_r", norm=norm, rasterized=True)
        if values.min() < thresh < values.max():
            ax.contour(*val_grid, values, levels=[thresh], colors="#191919", linewidths=1.2)
            ax.plot([], [], color="#191919", label=f"Free A = {thresh:.0f} nM")
        for i, branch in enumerate(curves(name, max_in)):
            ax.plot(*branch.T, "--", color="#198438", lw=1.2, label="Target" if i == 0 else None)
        ax.set(xlabel="X (nM)", aspect="equal", xlim=(0, max_in), ylim=(0, max_in))
        ax.set_title(f"{title}\nLoss {metrics['loss']:.3f}   Acc {metrics['accuracy']:.1%}", fontsize=7.5)
    axes[0].set_ylabel("Y (nM)")
    axes[len(axes) // 2].legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.colorbar(mesh, ax=list(axes), label="Free A (nM)", shrink=0.74, ticks=[norm.vmin, thresh, norm.vmax], format="%.1f")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _worker_fit_2d(job):
    calib_path_str, pads_t, sa_m, name, ideal_tot_nM, nodes, s, contour_fit, tot_max_native = job
    model = ConsensusModel(Path(calib_path_str))
    pads = list(pads_t)
    raws = [p.lstrip("-") for p in pads]
    mat = [[model.predict_pair(pads[i], pads[j]) for j in range(len(nodes))] for i in range(len(nodes))]
    pkd_mat = np.array([[mat[i][j]["pKd"] for j in range(len(nodes))] for i in range(len(nodes))], dtype=float)
    kd_mat_nM = 10.0 ** (9.0 - pkd_mat)

    def _err_val(m):
        return float("inf") if m["error_nM"] is None else float(m["error_nM"])

    init_eval = evaluate_full_2d(name, kd_mat_nM, ideal_tot_nM, nodes, s, use_jax=False)
    fitted_tot = fit_concentrations_full(
        name, kd_mat_nM, ideal_tot_nM, nodes, s,
        contour_fit=contour_fit, tot_max_native=tot_max_native,
    )
    fit_eval = evaluate_full_2d(name, kd_mat_nM, fitted_tot, nodes, s, use_jax=False)
    if contour_fit:
        if (_err_val(fit_eval[0]), -fit_eval[0]["accuracy"]) >= (_err_val(init_eval[0]), -init_eval[0]["accuracy"]):
            fitted_tot, fit_eval = ideal_tot_nM, init_eval
        score_key = (_err_val(fit_eval[0]), -fit_eval[0]["accuracy"], fit_eval[0]["loss"])
    else:
        if fit_eval[0]["loss"] >= init_eval[0]["loss"]:
            fitted_tot, fit_eval = ideal_tot_nM, init_eval
        score_key = (fit_eval[0]["loss"], -fit_eval[0]["accuracy"])
    return {
        "score_key": score_key,
        "pads": pads,
        "raws": raws,
        "pkd_mat": pkd_mat,
        "kd_mat_nM": kd_mat_nM,
        "fitted_tot": fitted_tot,
        "sa_m": sa_m,
    }


def main():
    p = argparse.ArgumentParser(description="Design coiled-coil sequences for 2D boundaries from des_boundaries.")
    p.add_argument("--boundary", type=str, default="line", choices=ALL_GATES)
    p.add_argument("--conc-scale", type=float, default=1.0, help="Scale [0, 200] nM grid & Kd by s (weakens Kd s-fold).")
    p.add_argument("--kd-min-native", type=float, default=1.0, help="Lower Kd bound in native [0,200] nM space.")
    p.add_argument("--kd-max-native", type=float, default=1000.0, help="Upper Kd bound in native [0,200] nM space.")
    p.add_argument("--tot-max-native", type=float, default=400.0, help="Upper total conc bound in native [0,200] nM space.")
    p.add_argument("--use-seed", action="store_true", help="Use starting_models from des_boundaries/threshold_seeds.py.")
    p.add_argument("--allow-5mer", action="store_true")
    p.add_argument("--cubic-yd", action="store_true", help="Use bipartite cubic topology (YD instead of XD).")
    p.add_argument("--contour-fit", action="store_true", help="Optimize contour shape instead of raw grid BCE.")
    p.add_argument("--restarts", type=int, default=20)
    p.add_argument("--steps", type=int, default=1400)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--top-fit", type=int, default=25)
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB_PATH)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "des_bounds_designs")
    p.add_argument("--off-max-pkd", type=float, default=4.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    name = args.boundary
    s = float(args.conc_scale)
    max_in = 200.0 * s
    thresh = 50.0 * s
    trans = 5.0 * s
    use_seed = bool(args.use_seed and name in STARTING_SEEDS)
    tag = "seed" if use_seed else f"kmin{args.kd_min_native:g}"
    mer_tag = "_5mer" if args.allow_5mer else "_4mer"
    cf_tag = "_cfit" if args.contour_fit else ""

    if name == "cubic" and args.cubic_yd:
        TOPOLOGIES["cubic"]["edges"] = ("AB", "AD", "CD", "XA", "XC", "YD", "YB")
        out_dir = args.out_dir / f"cubic_yd{mer_tag}{cf_tag}_s{s:g}_{tag}"
    else:
        out_dir = args.out_dir / f"{name}{mer_tag if args.allow_5mer else ''}{cf_tag}_s{s:g}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = TOPOLOGIES[name]
    nodes = list(graph["nodes"])
    if use_seed:
        print(f"=== [{name}] Step 1: Using des_boundaries STARTING_SEEDS (conc_scale={s}) ===")
        ideal_kd_nat, ideal_tot_nat = STARTING_SEEDS[name]
        edges_pairs = list(graph["edges"]) if name in ("line", "circle") else list(ideal_kd_nat.keys())
    else:
        print(f"=== [{name}] Step 1: Fitting ideal parameters (conc_scale={s}, native Kd=[{args.kd_min_native}, {args.kd_max_native}] -> eff min={args.kd_min_native*s:.1f} nM) ===")
        ideal_kd_nat, ideal_tot_nat = fit_ideal_gate(name, kd_min_native=args.kd_min_native, kd_max_native=args.kd_max_native, seed=args.seed)
        edges_pairs = list(graph["edges"])

    ideal_kd_nM = {k: v * s for k, v in ideal_kd_nat.items()}
    ideal_tot_nM = {k: v * s for k, v in ideal_tot_nat.items()}

    ideal_mat_nM = np.full((len(nodes), len(nodes)), 1e9, dtype=float)
    for pair, val in ideal_kd_nM.items():
        u, v = nodes.index(pair[0]), nodes.index(pair[1])
        ideal_mat_nM[u, v] = ideal_mat_nM[v, u] = val
    ideal_eval = evaluate_full_2d(name, ideal_mat_nM, ideal_tot_nM, nodes, s)
    print(f"Ideal: loss={ideal_eval[0]['loss']:.4f}, acc={ideal_eval[0]['accuracy']:.2%}, err_nM={ideal_eval[0]['error_nM']}")
    print("  Ideal Kd (nM):", {k: round(float(v), 2) for k, v in ideal_kd_nM.items()})
    print("  Ideal Totals (nM):", {k: round(float(v), 1) for k, v in ideal_tot_nM.items()})

    model = ConsensusModel(args.calib)
    lens = ["4mer", "5mer"] if args.allow_5mer else ["4mer"]
    lengths_by_node = [lens for _ in nodes]
    pool = generate_node_pool(model, charge_mode="mixed", lengths=lens, homo_min_nM=45_000.0, seed=args.seed)
    pools_by_node = [pool for _ in nodes]
    print(f"Pool size per node: {len(pool)} (lengths={lens})")

    target_edges = []
    for pair in edges_pairs:
        u, v = nodes.index(pair[0]), nodes.index(pair[1])
        if u > v:
            u, v = v, u
        ideal_pkd = 9.0 - math.log10(max(ideal_kd_nM[pair], 1e-3))
        target_edges.append((u, v, ideal_pkd))

    hierarchies = []
    for a in range(len(target_edges)):
        for b_idx in range(len(target_edges)):
            if a == b_idx:
                continue
            u1, v1, p1 = target_edges[a]
            u2, v2, p2 = target_edges[b_idx]
            if p1 - p2 >= 0.30:
                hierarchies.append((u1, v1, u2, v2, min(0.75 * (p1 - p2), 1.5)))
    if name == "circle":
        ac_u, ac_v = min(nodes.index("A"), nodes.index("C")), max(nodes.index("A"), nodes.index("C"))
        bc_u, bc_v = min(nodes.index("B"), nodes.index("C")), max(nodes.index("B"), nodes.index("C"))
        hierarchies.append((bc_u, bc_v, ac_u, ac_v, 0.05))

    gx_c, gy_c = np.meshgrid(*[np.linspace(0.0, max_in, 11)] * 2)
    labels_c = boundary_fn(name, gx_c, gy_c, max_in=max_in).ravel()
    stride = 20 if args.contour_fit else 50
    curve_c, _ = contour_geometry(name, max_in=max_in, step=stride, interior=True)
    xy_all = np.vstack((np.column_stack((gx_c.ravel(), gy_c.ravel())), curve_c))
    internal = nodes[2:]
    c0_train_nM = np.column_stack((xy_all, np.broadcast_to([ideal_tot_nM[k] for k in internal], (xy_all.shape[0], len(internal)))))

    t0 = time.time()
    workers = max(1, min(args.workers, args.restarts))
    r_per_w = [args.restarts // workers + (1 if i < args.restarts % workers else 0) for i in range(workers)]
    jobs = [
        (
            str(args.calib), pools_by_node, nodes, target_edges, hierarchies, lengths_by_node,
            args.off_max_pkd, 4.35, c0_train_nM, labels_c, gx_c.size, thresh, trans, bool(args.contour_fit),
            args.seed + 1009 * idx, r_cnt, args.steps,
        )
        for idx, r_cnt in enumerate(r_per_w) if r_cnt > 0
    ]

    raw_results = []
    if workers == 1:
        raw_results = _worker_sa_2d(jobs[0])
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for chunk in ex.map(_worker_sa_2d, jobs):
                raw_results.extend(chunk)

    dedup = {}
    for loss, pads_t, m in raw_results:
        if pads_t not in dedup or loss < dedup[pads_t][0]:
            dedup[pads_t] = (loss, pads_t, m)
    top_cands = sorted(dedup.values(), key=lambda x: x[0])[: args.top_fit]
    print(f"SA finished in {time.time() - t0:.1f}s ({len(dedup)} unique designs). Fitting concentrations on top {len(top_cands)}...")

    t_ref = time.time()
    fit_jobs = [
        (str(args.calib), pads_t, sa_m, name, ideal_tot_nM, nodes, s, bool(args.contour_fit), args.tot_max_native)
        for _, pads_t, sa_m in top_cands
    ]
    if workers == 1:
        fit_bundles = [_worker_fit_2d(j) for j in fit_jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fit_bundles = list(ex.map(_worker_fit_2d, fit_jobs))
    best_bundle = min(fit_bundles, key=lambda b: b["score_key"])
    best_bundle["init_eval"] = evaluate_full_2d(name, best_bundle["kd_mat_nM"], ideal_tot_nM, nodes, s, use_jax=True)
    best_bundle["fit_eval"] = evaluate_full_2d(name, best_bundle["kd_mat_nM"], best_bundle["fitted_tot"], nodes, s, use_jax=True)
    print(f"Parallel concentration fit finished in {time.time() - t_ref:.1f}s.")

    b = best_bundle
    print(f"\n=== Best CC Design for '{name}' (conc_scale={s}, use_seed={use_seed}) ===")
    print(
        f"Initial totals : loss={b['init_eval'][0]['loss']:.4f}, acc={b['init_eval'][0]['accuracy']:.2%}, "
        f"bal_acc={b['init_eval'][0]['balanced_accuracy']:.2%}, err_nM={b['init_eval'][0]['error_nM']}"
    )
    print(
        f"Fitted totals  : loss={b['fit_eval'][0]['loss']:.4f}, acc={b['fit_eval'][0]['accuracy']:.2%}, "
        f"bal_acc={b['fit_eval'][0]['balanced_accuracy']:.2%}, mcc={b['fit_eval'][0]['mcc']:.3f}, err_nM={b['fit_eval'][0]['error_nM']}"
    )
    print(f"Worst off/homo pKd: {b['sa_m']['worst_off_pKd']:.2f} / {b['sa_m']['worst_homo_pKd']:.2f}")
    print("Fitted Totals (nM):", {k: round(float(v), 1) for k, v in b["fitted_tot"].items()})
    for pair in edges_pairs:
        u, v = nodes.index(pair[0]), nodes.index(pair[1])
        print(f"  Edge {pair}: Kd={b['kd_mat_nM'][u, v]:.2f} nM (pKd={b['pkd_mat'][u, v]:.2f}, ideal={ideal_kd_nM[pair]:.2f} nM)")
    for i, nm in enumerate(nodes):
        print(f"  {nm:2s} ({len(b['raws'][i])}aa, chg={net_charge(b['raws'][i]):+d}, eps={calc_eps280(b['raws'][i])}): {b['raws'][i]}")

    stages = {
        "Ideal": (ideal_eval[0], ideal_eval[1]),
        "Selected peptides": (b["init_eval"][0], b["init_eval"][1]),
        "Fitted concentrations": (b["fit_eval"][0], b["fit_eval"][1]),
    }
    val_grid = b["fit_eval"][2]
    for ext in (".pdf", ".png"):
        plot_topology_and_contour(
            name, nodes, edges_pairs, b["kd_mat_nM"], b["fitted_tot"], b["fit_eval"][1], val_grid, s,
            b["fit_eval"][0], b["sa_m"]["worst_off_pKd"], b["sa_m"]["worst_homo_pKd"], out_dir / f"{name}_contours{ext}",
        )
        plot_performance_stages(name, stages, val_grid, s, out_dir / f"{name}_performance{ext}")

    report = {
        "boundary": name,
        "conc_scale": s,
        "use_seed": use_seed,
        "cubic_yd": bool(name == "cubic" and args.cubic_yd),
        "contour_fit": bool(args.contour_fit),
        "threshold_nM": thresh,
        "nodes": nodes,
        "edges": edges_pairs,
        "ideal_Kd_nM": {k: float(v) for k, v in ideal_kd_nM.items()},
        "ideal_total_nM": {k: float(v) for k, v in ideal_tot_nM.items()},
        "ideal_metrics": ideal_eval[0],
        "seqs": dict(zip(nodes, b["raws"])),
        "pads": dict(zip(nodes, b["pads"])),
        "pkd_matrix": b["pkd_mat"].tolist(),
        "fitted_total_nM": {k: float(v) for k, v in b["fitted_tot"].items()},
        "initial_metrics": b["init_eval"][0],
        "fitted_metrics": b["fit_eval"][0],
        "worst_off_pKd": b["sa_m"]["worst_off_pKd"],
        "worst_homo_pKd": b["sa_m"]["worst_homo_pKd"],
    }
    (out_dir / f"{name}_report.json").write_text(json.dumps(report, indent=2) + "\n")

    # Export full N*(N+1)/2 Kd matrix in native [0, 200] nM scale for korenkov-es/des_boundaries/molecular_performance.py
    mol_kd = {
        f"{nodes[i]}{nodes[j]}": float(b["kd_mat_nM"][i, j] / s)
        for i in range(len(nodes))
        for j in range(i, len(nodes))
    }
    (out_dir / f"molecules_{name}.json").write_text(json.dumps({"boundary": name, "Kd_nM": mol_kd}, indent=2) + "\n")
    print(f"Saved plots, report, and molecules_{name}.json to {out_dir}")


if __name__ == "__main__":
    main()
