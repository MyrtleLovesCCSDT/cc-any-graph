"""General N-peptide thermodynamic circuit simulator via EQTK and Monte Carlo."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import eqtk


def dimer_pairs(n_species: int) -> List[Tuple[int, int]]:
    """Canonical ordering of all N*(N+1)/2 homo- and heterodimers (i <= j)."""
    return [(i, j) for i in range(n_species) for j in range(i, n_species)]


def pair_index_map(n_species: int) -> Dict[Tuple[int, int], int]:
    """Map symmetric pair (i, j) -> column index in full species vector."""
    mapping: Dict[Tuple[int, int], int] = {}
    for idx, (i, j) in enumerate(dimer_pairs(n_species)):
        col = n_species + idx
        mapping[(i, j)] = col
        mapping[(j, i)] = col
    return mapping


def build_stoich_matrix(n_species: int) -> np.ndarray:
    """Build EQTK stoichiometry matrix for N monomers and all N*(N+1)/2 dimers."""
    pairs = dimer_pairs(n_species)
    n_dimers = len(pairs)
    stoich = np.zeros((n_dimers, n_species + n_dimers), dtype=float)
    for r, (i, j) in enumerate(pairs):
        if i == j:
            stoich[r, i] = -2.0
        else:
            stoich[r, i] = -1.0
            stoich[r, j] = -1.0
        stoich[r, n_species + r] = 1.0
    return stoich


def _check_eqtk() -> bool:
    try:
        stoich2 = build_stoich_matrix(2)
        c_test = np.array([[1e-6, 1e-6, 0.0, 0.0, 0.0]], dtype=float)
        # pairs for N=2: (0,0), (0,1), (1,1) -> col 3 is (0,1)
        sol = np.asarray(eqtk.solve(c_test, N=stoich2, K=np.array([1.0, 1e6, 1.0])))
        return abs(sol[0, 3] - 0.38196601e-6) < 1e-9
    except Exception:
        return False


_USE_EQTK = _check_eqtk()
_STOICH_CACHE: Dict[int, np.ndarray] = {}


def get_stoich(n_species: int) -> np.ndarray:
    if n_species not in _STOICH_CACHE:
        _STOICH_CACHE[n_species] = build_stoich_matrix(n_species)
    return _STOICH_CACHE[n_species]


def _newton_dimers_general(c0_rows: np.ndarray, k_mat: np.ndarray) -> np.ndarray:
    """Vectorized damped-Newton equilibrium solver for arbitrary N dimers."""
    n = k_mat.shape[0]
    pairs = dimer_pairs(n)
    out = np.zeros((c0_rows.shape[0], n + len(pairs)), dtype=float)

    for row_idx, row in enumerate(c0_rows):
        tot = np.asarray(row[:n], dtype=float)
        active = np.where(tot > 0.0)[0]
        if active.size == 0:
            continue

        t_act = tot[active]
        k_sub = k_mat[np.ix_(active, active)]
        a_mat = k_sub.copy()
        np.fill_diagonal(a_mat, 2.0 * np.diag(k_sub))

        x = t_act.copy()
        m = active.size
        diag_idx = np.diag_indices(m)
        for _ in range(100):
            ax = a_mat @ x
            f_vec = x * (1.0 + ax) - t_act
            rel_err = np.max(np.abs(f_vec) / np.maximum(t_act, 1e-18))
            if rel_err < 1e-12 and np.max(np.abs(f_vec)) < 1e-16:
                break
            jac = x[:, None] * a_mat
            jac[diag_idx] += 1.0 + ax
            dx = np.linalg.solve(jac, -f_vec)
            neg = dx < 0.0
            if np.any(neg):
                alpha_max = float(np.min(-x[neg] / dx[neg]))
                step = min(1.0, 0.95 * alpha_max)
            else:
                step = 1.0
            x = np.maximum(x + step * dx, 1e-24)

        full_x = np.zeros(n, dtype=float)
        full_x[active] = x
        out[row_idx, :n] = full_x
        for p_idx, (i, j) in enumerate(pairs):
            out[row_idx, n + p_idx] = k_mat[i, j] * full_x[i] * full_x[j]

    return out


def solve_general_equilibrium(c0_monomers_m: np.ndarray, kd_matrix_m: np.ndarray) -> np.ndarray:
    """
    Solve coupled dimer equilibrium for N species across M condition rows.

    Parameters
    ----------
    c0_monomers_m : array (M, N)
        Initial total monomer concentrations in Molar.
    kd_matrix_m : array (N, N)
        Symmetric matrix of dissociation constants Kd in Molar.
    """
    c0_monomers_m = np.asarray(c0_monomers_m, dtype=float)
    n = c0_monomers_m.shape[1]
    pairs = dimer_pairs(n)
    kd_safe = np.maximum(np.asarray(kd_matrix_m, dtype=float), 1e-12)
    kd_sym = 0.5 * (kd_safe + kd_safe.T)

    if _USE_EQTK:
        c0_full = np.zeros((c0_monomers_m.shape[0], n + len(pairs)), dtype=float)
        c0_full[:, :n] = c0_monomers_m
        k_vec = np.array([1.0 / kd_sym[i, j] for i, j in pairs], dtype=float)
        try:
            return np.asarray(eqtk.solve(c0_full, N=get_stoich(n), K=k_vec))
        except Exception:
            pass

    return _newton_dimers_general(c0_monomers_m, 1.0 / kd_sym)


@dataclass
class Observable:
    """
    Defines how to read out a scalar signal (0..1) from equilibrium species.

    kind:
      - 'free_frac':   [nodes[0]]_free / [ref_node]_0
      - 'bound_frac':  1 - [nodes[0]]_free / [ref_node]_0
      - 'dimer_frac':  [nodes[0]:nodes[1]] / [ref_node]_0
      - 'sum_free_frac': sum_k ([nodes[k]]_free / [ref_nodes[k]]_0) / len(nodes)
    """
    kind: str
    nodes: Tuple[int, ...]
    ref_nodes: Optional[Tuple[int, ...]] = None

    def evaluate(self, eq: np.ndarray, c0_m: np.ndarray, n_species: int) -> np.ndarray:
        pmap = pair_index_map(n_species)
        refs = self.ref_nodes if self.ref_nodes is not None else self.nodes

        if self.kind == "free_frac":
            u = self.nodes[0]
            denom = np.maximum(c0_m[:, refs[0]], 1e-18)
            return eq[:, u] / denom
        if self.kind == "bound_frac":
            u = self.nodes[0]
            denom = np.maximum(c0_m[:, refs[0]], 1e-18)
            return 1.0 - (eq[:, u] / denom)
        if self.kind == "dimer_frac":
            u, v = self.nodes[0], self.nodes[1]
            col = pmap[(u, v)]
            denom = np.maximum(c0_m[:, refs[0]], 1e-18)
            mult = 2.0 if (u == v and refs[0] == u) else 1.0
            return mult * eq[:, col] / denom
        if self.kind == "sum_free_frac":
            acc = np.zeros(eq.shape[0], dtype=float)
            for u, r in zip(self.nodes, refs):
                acc += eq[:, u] / np.maximum(c0_m[:, r], 1e-18)
            return acc / float(len(self.nodes))
        raise ValueError(f"Unknown observable kind: {self.kind}")


def evaluate_truth_table(
    kd_matrix_m: np.ndarray,
    states_c0_m: np.ndarray,
    observable: Observable,
    target_bits: Sequence[int],
) -> dict:
    """Evaluate Boolean gate performance across input states."""
    n = kd_matrix_m.shape[0]
    eq = solve_general_equilibrium(states_c0_m, kd_matrix_m)
    vals = observable.evaluate(eq, states_c0_m, n)
    bits = np.asarray(target_bits, dtype=int)

    on_vals = vals[bits == 1]
    off_vals = vals[bits == 0]
    min_on = float(np.min(on_vals))
    max_off = float(np.max(off_vals))
    fold_floor = 0.005
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
    }


def evaluate_curve(
    kd_matrix_m: np.ndarray,
    states_c0_m: np.ndarray,
    observable: Observable,
    target_y: Sequence[float],
    affine_fit: bool = False,
) -> dict:
    """Evaluate analog transfer function (e.g., 1 - (X/Xmax)^2) across concentration grid."""
    n = kd_matrix_m.shape[0]
    eq = solve_general_equilibrium(states_c0_m, kd_matrix_m)
    raw = observable.evaluate(eq, states_c0_m, n)
    y_tgt = np.asarray(target_y, dtype=float)

    if affine_fit:
        # Fit y_tgt ~= a * raw + b with a >= 0
        x_var = np.var(raw)
        if x_var < 1e-12:
            a, b = 0.0, float(np.mean(y_tgt))
        else:
            cov = float(np.mean((raw - np.mean(raw)) * (y_tgt - np.mean(y_tgt))))
            a = max(0.0, cov / x_var)
            b = float(np.mean(y_tgt) - a * np.mean(raw))
        y_pred = a * raw + b
    else:
        y_pred = raw

    rmse = float(np.sqrt(np.mean((y_pred - y_tgt) ** 2)))
    ss_tot = float(np.sum((y_tgt - np.mean(y_tgt)) ** 2))
    r2 = 1.0 - float(np.sum((y_pred - y_tgt) ** 2)) / max(ss_tot, 1e-12)
    span = float(np.max(raw) - np.min(raw))
    return {
        "raw": raw,
        "y_pred": y_pred,
        "RMSE": rmse,
        "R2": r2,
        "Span": span,
    }


def mc_robustness_truth_table(
    log10_kd_matrix: np.ndarray,
    states_c0_m: np.ndarray,
    observable: Observable,
    target_bits: Sequence[int],
    n_mc: int = 200,
    sigma_kd: float = 0.643,
    conc_rel_std: float = 0.0,
    min_delta: float = 0.35,
    min_fold: float = 2.5,
    seed: int = 42,
    return_arrays: bool = False,
) -> dict:
    """Monte Carlo stress test perturbing symmetric log10(Kd) matrix (and optionally C0)."""
    rng = np.random.default_rng(seed)
    n = log10_kd_matrix.shape[0]
    tri_i, tri_j = np.triu_indices(n)
    base = np.minimum(np.asarray(log10_kd_matrix, dtype=float), -2.0)

    deltas = np.empty(n_mc, dtype=float)
    folds = np.empty(n_mc, dtype=float)
    for m in range(n_mc):
        pert = base.copy()
        noise = rng.normal(0.0, sigma_kd, size=tri_i.size)
        pert[tri_i, tri_j] += noise
        pert[tri_j, tri_i] = pert[tri_i, tri_j]
        kd_mat = 10.0 ** np.minimum(pert, -2.0)

        if conc_rel_std > 0.0:
            c_mult = np.maximum(1.0 + rng.normal(0.0, conc_rel_std, size=states_c0_m.shape), 0.1)
            c0_m = states_c0_m * c_mult
        else:
            c0_m = states_c0_m

        res = evaluate_truth_table(kd_mat, c0_m, observable, target_bits)
        deltas[m] = res["Delta"]
        folds[m] = res["Fold"]

    pass_mask = (deltas >= min_delta) & (folds >= min_fold)
    out = {
        "MC_Delta_med": float(np.median(deltas)),
        "MC_Delta_p10": float(np.percentile(deltas, 10)),
        "MC_Fold_med": float(np.median(folds)),
        "MC_Pass_rate": float(np.mean(pass_mask)),
    }
    if return_arrays:
        out["_pass_mask"] = pass_mask
    return out


def mc_robustness_curve(
    log10_kd_matrix: np.ndarray,
    states_c0_m: np.ndarray,
    observable: Observable,
    target_y: Sequence[float],
    n_mc: int = 200,
    sigma_kd: float = 0.643,
    conc_rel_std: float = 0.0,
    affine_fit: bool = False,
    max_rmse: float = 0.12,
    min_span: float = 0.40,
    seed: int = 42,
    return_arrays: bool = False,
) -> dict:
    """Monte Carlo stress test for analog transfer functions."""
    rng = np.random.default_rng(seed)
    n = log10_kd_matrix.shape[0]
    tri_i, tri_j = np.triu_indices(n)
    base = np.minimum(np.asarray(log10_kd_matrix, dtype=float), -2.0)

    rmses = np.empty(n_mc, dtype=float)
    r2s = np.empty(n_mc, dtype=float)
    spans = np.empty(n_mc, dtype=float)
    for m in range(n_mc):
        pert = base.copy()
        noise = rng.normal(0.0, sigma_kd, size=tri_i.size)
        pert[tri_i, tri_j] += noise
        pert[tri_j, tri_i] = pert[tri_i, tri_j]
        kd_mat = 10.0 ** np.minimum(pert, -2.0)

        c0_m = (
            states_c0_m * np.maximum(1.0 + rng.normal(0.0, conc_rel_std, size=states_c0_m.shape), 0.1)
            if conc_rel_std > 0.0
            else states_c0_m
        )
        res = evaluate_curve(kd_mat, c0_m, observable, target_y, affine_fit=affine_fit)
        rmses[m] = res["RMSE"]
        r2s[m] = res["R2"]
        spans[m] = res["Span"]

    pass_mask = (rmses <= max_rmse) & (spans >= min_span)
    out = {
        "MC_RMSE_med": float(np.median(rmses)),
        "MC_R2_med": float(np.median(r2s)),
        "MC_Span_med": float(np.median(spans)),
        "MC_Pass_rate": float(np.mean(pass_mask)),
    }
    if return_arrays:
        out["_pass_mask"] = pass_mask
    return out
