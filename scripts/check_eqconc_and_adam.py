"""Verify EQTK vs JAX eq_conc (van Dorp 2011) and tune c0 via optimize_batch_c0_ADAM on MC ensembles."""

import argparse
from collections import namedtuple
from functools import partial
import json
import sys
from pathlib import Path

import jax
from jax import lax, jit
from jax import random as jax_random
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.circuit_eq import evaluate_truth_table, mc_robustness_truth_table, solve_general_equilibrium
from src.consensus import DEFAULT_CALIB_PATH
from src.graph_design import _parse_config


# --- 1. Advisor's exact solver (unmodified) + damped version ---

@partial(jit, static_argnames=("it",))
def eq_conc(c0, A, it=1000):
    diag = jnp.arange(A.shape[-1])
    A_mod = A.at[..., diag, diag].multiply(2.0)

    def _body(_, c):
        return c0 / (1.0 + jnp.einsum("...ij,...j->...i", A_mod, c))

    return lax.fori_loop(0, it, _body, c0)


@partial(jit, static_argnames=("it", "damping"))
def eq_conc_damped(c0, A, it=300, damping=0.5):
    diag = jnp.arange(A.shape[-1])
    A_mod = A.at[..., diag, diag].multiply(2.0)

    def _body(_, c):
        c_new = c0 / (1.0 + jnp.einsum("...ij,...j->...i", A_mod, c))
        return (1.0 - damping) * c + damping * c_new

    return lax.fori_loop(0, it, _body, c0)


# --- 2. Advisor's exact Adam optimizer (unmodified) ---

@partial(jit, static_argnames=("func", "c0_shape", "args"))
def optimize_batch_c0_ADAM(func, c0_shape, args, data, c0=None):
    key = jax_random.PRNGKey(args.seed)
    key, subkey = jax_random.split(key)

    minv = jnp.full(c0_shape, args.con_min, dtype=jnp.float32)
    maxv = jnp.full(c0_shape, args.con_max, dtype=jnp.float32)

    c = c0 if c0 is not None else jax_random.uniform(
        subkey, shape=c0_shape, minval=minv, maxval=maxv, dtype=jnp.float32
    )

    lower = jnp.full_like(c, args.con_min, dtype=jnp.float32)
    upper = jnp.full_like(c, args.con_max, dtype=jnp.float32)

    lr_schedule = optax.cosine_decay_schedule(
        init_value=args.lr_init,
        decay_steps=args.num_steps,
        alpha=args.lr_final / args.lr_init,
    )

    optimizer = optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.999, eps=1e-8)
    opt_state = optimizer.init(c)
    grad_fn = jax.grad(func)

    def step(_, carry):
        c, opt_state = carry
        grads = grad_fn(c, *data)
        updates, opt_state = optimizer.update(grads, opt_state)
        c = optax.apply_updates(c, updates)
        c = jnp.clip(c, lower, upper)
        return c, opt_state

    return lax.fori_loop(0, args.num_steps, step, (c, opt_state))[0]


AdamArgs = namedtuple("AdamArgs", ["seed", "con_min", "con_max", "lr_init", "lr_final", "num_steps"])


def eval_obs_jax(c_eq, c0_uM, A_uM, kind, nodes, ref_nodes):
    if kind == "free_frac":
        u, r = nodes[0], ref_nodes[0]
        return c_eq[..., u] / jnp.maximum(c0_uM[..., r], 1e-12)
    if kind == "bound_frac":
        u, r = nodes[0], ref_nodes[0]
        return 1.0 - c_eq[..., u] / jnp.maximum(c0_uM[..., r], 1e-12)
    if kind == "dimer_frac":
        u, v, r = nodes[0], nodes[1], ref_nodes[0]
        mult = 2.0 if (u == v and r == u) else 1.0
        d_uv = A_uM[..., u, v] * c_eq[..., u] * c_eq[..., v]
        return mult * d_uv / jnp.maximum(c0_uM[..., r], 1e-12)
    if kind == "sum_free_frac":
        acc = 0.0
        for u, r in zip(nodes, ref_nodes):
            acc = acc + c_eq[..., u] / jnp.maximum(c0_uM[..., r], 1e-12)
        return acc / float(len(nodes))
    raise ValueError(f"Unsupported observable kind: {kind}")


def mass_balance_rel_err(c_eq, c0_uM, A_uM):
    A_mod = A_uM.copy()
    idx = np.arange(A_uM.shape[-1])
    A_mod[..., idx, idx] *= 2.0
    rhs = c_eq * (1.0 + np.einsum("...ij,...j->...i", A_mod, c_eq))
    mask = c0_uM > 0.0
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(rhs[mask] - c0_uM[mask]) / c0_uM[mask]))


def pkd_to_A_uM(pkd_mat, clip_max_pkd=12.0):
    # circuit_eq clips log10_kd to <= -2.0 and kd >= 1e-12 M (pKd in [2, 12])
    pkd_safe = np.clip(pkd_mat, 2.0, clip_max_pkd)
    return (10.0 ** (pkd_safe - 6.0)).astype(np.float32)


def generate_mc_batches(pkd_mat, n_mc, sigma_kd, seed):
    rng = np.random.default_rng(seed)
    n = pkd_mat.shape[0]
    tri_i, tri_j = np.triu_indices(n)
    base_log10 = np.minimum(-np.asarray(pkd_mat, dtype=float), -2.0)

    kd_m_list = []
    A_uM_list = []
    for _ in range(n_mc):
        pert = base_log10.copy()
        noise = rng.normal(0.0, sigma_kd, size=tri_i.size)
        pert[tri_i, tri_j] += noise
        pert[tri_j, tri_i] = pert[tri_i, tri_j]
        kd_m = 10.0 ** np.minimum(pert, -2.0)
        kd_safe_m = np.maximum(kd_m, 1e-12)
        kd_m_list.append(kd_m)
        A_uM_list.append((1e-6 / kd_safe_m).astype(np.float32))
    return np.stack(kd_m_list), np.stack(A_uM_list)


def compare_solvers(parsed, des, sigma_kd=0.643, n_mc=400, conc_std=0.015, seed=42):
    names = parsed["node_names"]
    n = len(names)
    obs = parsed["observables"][0]
    targets = np.asarray(obs["targets"], dtype=int)
    c0_uM = np.array(des["c0_uM"], dtype=float)
    c0_m = c0_uM * 1e-6
    pkd_mat = np.array(des["pkd_matrix"], dtype=float)
    kd_nom_m = 10.0 ** np.minimum(-pkd_mat, -2.0)
    A_nom_uM = pkd_to_A_uM(pkd_mat)

    # 1. Nominal comparison
    eq_tk = solve_general_equilibrium(c0_m, kd_nom_m)
    vals_eqtk = obs["obj"].evaluate(eq_tk, c0_m, n)

    c0_j = jnp.asarray(c0_uM, dtype=jnp.float32)
    A_j = jnp.asarray(A_nom_uM, dtype=jnp.float32)
    c_raw = np.asarray(eq_conc(c0_j, A_j, it=1000))
    c_dmp = np.asarray(eq_conc_damped(c0_j, A_j, it=300, damping=0.5))

    vals_raw = np.asarray(eval_obs_jax(c_raw, c0_j, A_j, obs["obj"].kind, obs["obj"].nodes, obs["obj"].ref_nodes))
    vals_dmp = np.asarray(eval_obs_jax(c_dmp, c0_j, A_j, obs["obj"].kind, obs["obj"].nodes, obs["obj"].ref_nodes))

    mono_tk_uM = eq_tk[:, :n] * 1e6
    print(f"\n=== Solver Comparison: {parsed['name']} ({des['design_id']}) ===")
    print(f"Nominal max |eqtk - eq_conc(1000)|      output={np.max(np.abs(vals_eqtk - vals_raw)):.2e}, monomer={np.max(np.abs(mono_tk_uM - c_raw)):.2e} uM, mb_err={mass_balance_rel_err(c_raw, c0_uM, A_nom_uM):.2e}")
    print(f"Nominal max |eqtk - eq_conc_dmp(300)|   output={np.max(np.abs(vals_eqtk - vals_dmp)):.2e}, monomer={np.max(np.abs(mono_tk_uM - c_dmp)):.2e} uM, mb_err={mass_balance_rel_err(c_dmp, c0_uM, A_nom_uM):.2e}")

    # 2. MC comparison on identical perturbations
    rng = np.random.default_rng(seed)
    tri_i, tri_j = np.triu_indices(n)
    base_log10 = np.minimum(-pkd_mat, -2.0)
    min_d, min_f = obs["min_delta"] * 0.85, obs["min_fold"] * 0.85

    A_mc = []
    c0_mc = []
    d_tk, f_tk = [], []
    for _ in range(n_mc):
        pert = base_log10.copy()
        noise = rng.normal(0.0, sigma_kd, size=tri_i.size)
        pert[tri_i, tri_j] += noise
        pert[tri_j, tri_i] = pert[tri_i, tri_j]
        kd_m = 10.0 ** np.minimum(pert, -2.0)
        c0_run_m = c0_m * np.maximum(1.0 + rng.normal(0.0, conc_std, size=c0_m.shape), 0.1) if conc_std > 0 else c0_m
        tt = evaluate_truth_table(kd_m, c0_run_m, obs["obj"], targets)
        d_tk.append(tt["Delta"])
        f_tk.append(tt["Fold"])
        A_mc.append((1e-6 / np.maximum(kd_m, 1e-12)).astype(np.float32))
        c0_mc.append((c0_run_m * 1e6).astype(np.float32))

    A_mc_j = jnp.asarray(np.stack(A_mc), dtype=jnp.float32)[:, None, :, :]
    c0_mc_j = jnp.asarray(np.stack(c0_mc), dtype=jnp.float32)

    for label, solver_fn in [("eq_conc(it=1000)", lambda c, a: eq_conc(c, a, it=1000)), ("eq_conc_damped(it=300)", lambda c, a: eq_conc_damped(c, a, it=300, damping=0.5))]:
        c_sol = np.asarray(solver_fn(c0_mc_j, A_mc_j))
        vals = np.asarray(eval_obs_jax(c_sol, c0_mc_j, A_mc_j, obs["obj"].kind, obs["obj"].nodes, obs["obj"].ref_nodes))
        on_v = np.min(vals[:, targets == 1], axis=1)
        off_v = np.max(vals[:, targets == 0], axis=1)
        deltas = on_v - off_v
        folds = on_v / np.maximum(off_v, 0.005)
        pass_r = float(np.mean((deltas >= min_d) & (folds >= min_f)))
        mb = mass_balance_rel_err(c_sol, np.asarray(c0_mc_j), np.asarray(A_mc_j))
        print(f"MC (n={n_mc}) {label:24s}: Pass={pass_r*100:5.1f}% | med_Delta={np.median(deltas):.3f} | p10={np.percentile(deltas, 10):.3f} | max_mb_err={mb:.2e}")

    d_tk = np.array(d_tk)
    f_tk = np.array(f_tk)
    pass_tk = float(np.mean((d_tk >= min_d) & (f_tk >= min_f)))
    print(f"MC (n={n_mc}) {'EQTK':24s}: Pass={pass_tk*100:5.1f}% | med_Delta={np.median(d_tk):.3f} | p10={np.percentile(d_tk, 10):.3f}")


def tune_c0_mc_adam(parsed, des, args):
    names = parsed["node_names"]
    n = len(names)
    c0_uM_base = np.array(des["c0_uM"], dtype=np.float32)
    pkd_mat = np.array(des["pkd_matrix"], dtype=float)
    active_mask = (c0_uM_base > 0.0).astype(np.float32)

    # Identify per-node baseline levels and which nodes are tunable
    nom_levels = np.array(
        [float(np.median(c0_uM_base[:, j][c0_uM_base[:, j] > 0.0])) if np.any(c0_uM_base[:, j] > 0.0) else 0.0 for j in range(n)],
        dtype=np.float32,
    )
    rep_nodes = {r for obs in parsed["observables"] for r in obs["obj"].ref_nodes}
    tunable_idx = [j for j in range(n) if nom_levels[j] > 0.0 and not (args.fix_reporter and j in rep_nodes)]

    # Parameter vector theta maps onto tunable nodes
    param_to_node = np.zeros((n, len(tunable_idx)), dtype=np.float32)
    for k, j in enumerate(tunable_idx):
        param_to_node[j, k] = 1.0
    fixed_vec = np.array([0.0 if j in tunable_idx else nom_levels[j] for j in range(n)], dtype=np.float32)

    param_to_node_j = jnp.asarray(param_to_node)
    fixed_vec_j = jnp.asarray(fixed_vec)
    active_mask_j = jnp.asarray(active_mask)

    # Generate training MC batch (separate seed from evaluation seed=42!)
    _, A_train_uM = generate_mc_batches(pkd_mat, n_mc=args.batch_mc, sigma_kd=args.sigma, seed=args.train_seed)
    A_nom_uM = pkd_to_A_uM(pkd_mat)
    A_train_all = jnp.asarray(np.concatenate([A_nom_uM[None], A_train_uM], axis=0))[:, None, :, :]

    rng = np.random.default_rng(args.train_seed + 1)
    if args.conc_std > 0.0:
        c_noise = np.maximum(1.0 + rng.normal(0.0, args.conc_std, size=(args.batch_mc, c0_uM_base.shape[0], n)), 0.1).astype(np.float32)
        c_noise = np.concatenate([np.ones((1, c0_uM_base.shape[0], n), dtype=np.float32), c_noise], axis=0)
    else:
        c_noise = np.ones((args.batch_mc + 1, c0_uM_base.shape[0], n), dtype=np.float32)
    c_noise_j = jnp.asarray(c_noise)

    obs_specs = []
    for obs in parsed["observables"]:
        t_arr = np.asarray(obs["targets"], dtype=int)
        obs_specs.append((
            obs["obj"].kind,
            obs["obj"].nodes,
            obs["obj"].ref_nodes,
            tuple(np.where(t_arr == 1)[0].tolist()),
            tuple(np.where(t_arr == 0)[0].tolist()),
            float(obs["min_delta"] * 0.85),
            float(obs["min_fold"] * 0.85),
        ))

    def mc_loss(theta, A_batch, noise_batch):
        levels = fixed_vec_j + param_to_node_j @ theta
        c0_states = active_mask_j * levels[None, :]
        c0_batch = c0_states[None, :, :] * noise_batch
        c_eq = eq_conc_damped(c0_batch, A_batch, it=250, damping=0.5)

        total_loss = 0.0
        for kind, nodes, ref_nodes, on_idx, off_idx, min_d, min_f in obs_specs:
            vals = eval_obs_jax(c_eq, c0_batch, A_batch, kind, nodes, ref_nodes)
            on_vals = vals[:, list(on_idx)]
            off_vals = vals[:, list(off_idx)]

            # Smooth min(ON) and max(OFF) + exact min/max for margin loss
            tau = 0.04
            smin_on = -tau * jax.scipy.special.logsumexp(-on_vals / tau, axis=1)
            smax_off = tau * jax.scipy.special.logsumexp(off_vals / tau, axis=1)
            delta = smin_on - smax_off
            fold = jnp.min(on_vals, axis=1) / jnp.maximum(jnp.max(off_vals, axis=1), 0.005)

            # Maximize expected contrast & strongly penalize failing MC trials
            loss_d = -jnp.mean(delta)
            fail_d = jnp.mean(jnp.maximum(0.0, (min_d + 0.08) - delta) ** 2)
            fail_f = jnp.mean(jnp.maximum(0.0, (min_f + 0.5) - fold) ** 2)
            # Keep nominal (index 0) strong
            nom_pen = jnp.maximum(0.0, (min_d + 0.15) - delta[0]) ** 2
            total_loss = total_loss + loss_d + 25.0 * fail_d + 0.4 * fail_f + 10.0 * nom_pen
        return total_loss

    theta0 = jnp.asarray(nom_levels[tunable_idx], dtype=jnp.float32)
    adam_args = AdamArgs(
        seed=args.train_seed,
        con_min=args.con_min,
        con_max=args.con_max,
        lr_init=args.lr_init,
        lr_final=args.lr_final,
        num_steps=args.steps,
    )
    theta_opt = np.asarray(
        optimize_batch_c0_ADAM(mc_loss, theta0.shape, adam_args, (A_train_all, c_noise_j), c0=theta0)
    )

    new_levels = fixed_vec.copy()
    new_levels[tunable_idx] = theta_opt
    new_c0_uM = active_mask * new_levels[None, :]

    print("\n--- ADAM MC Concentration Tuning ---")
    for j in range(n):
        tag = "(fixed)" if j not in tunable_idx else "(tuned)"
        print(f"  {names[j]:4s} {tag:7s}: {nom_levels[j]:6.3f} -> {new_levels[j]:6.3f} uM")

    # Independent evaluation on seed=42 with EQTK
    log10_kd_mat = np.minimum(-pkd_mat, -2.0)
    kd_nom_m = 10.0 ** log10_kd_mat
    obs0 = parsed["observables"][0]
    tt_old = evaluate_truth_table(kd_nom_m, c0_uM_base * 1e-6, obs0["obj"], obs0["targets"])
    tt_new = evaluate_truth_table(kd_nom_m, new_c0_uM * 1e-6, obs0["obj"], obs0["targets"])

    mc_old = mc_robustness_truth_table(
        log10_kd_mat, c0_uM_base * 1e-6, obs0["obj"], obs0["targets"],
        n_mc=args.eval_mc, sigma_kd=args.sigma, conc_rel_std=args.conc_std,
        min_delta=obs0["min_delta"] * 0.85, min_fold=obs0["min_fold"] * 0.85, seed=42,
    )
    mc_new = mc_robustness_truth_table(
        log10_kd_mat, new_c0_uM * 1e-6, obs0["obj"], obs0["targets"],
        n_mc=args.eval_mc, sigma_kd=args.sigma, conc_rel_std=args.conc_std,
        min_delta=obs0["min_delta"] * 0.85, min_fold=obs0["min_fold"] * 0.85, seed=42,
    )
    print(f"EQTK Before: Nom Delta={tt_old['Delta']:.3f}, Fold={tt_old['Fold']:.1f}x | Sim MC Pass={mc_old['MC_Pass_rate']*100:.1f}%, p10={mc_old['MC_Delta_p10']:.3f}")
    print(f"EQTK After : Nom Delta={tt_new['Delta']:.3f}, Fold={tt_new['Fold']:.1f}x | Sim MC Pass={mc_new['MC_Pass_rate']*100:.1f}%, p10={mc_new['MC_Delta_p10']:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full-json", type=Path, required=True)
    p.add_argument("--design-id", type=str, default="D01")
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB_PATH)
    p.add_argument("--tune-adam", action="store_true")
    p.add_argument("--fix-reporter", action="store_true", help="Keep reporter S concentration fixed.")
    p.add_argument("--con-min", type=float, default=0.1)
    p.add_argument("--con-max", type=float, default=20.0)
    p.add_argument("--lr-init", type=float, default=0.25)
    p.add_argument("--lr-final", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-mc", type=int, default=256)
    p.add_argument("--eval-mc", type=int, default=400)
    p.add_argument("--conc-std", type=float, default=0.015)
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--train-seed", type=int, default=123)
    args = p.parse_args()

    if args.sigma is None:
        with open(args.calib, "r") as f:
            args.sigma = float(json.load(f).get("loo_rmse", 0.643))

    with open(args.full_json, "r") as f:
        bundle = json.load(f)
    parsed = _parse_config(bundle["config"], allow_5mer=True)
    des = {d["design_id"]: d for d in bundle["designs"]}[args.design_id]

    compare_solvers(parsed, des, sigma_kd=args.sigma, n_mc=args.eval_mc, conc_std=args.conc_std, seed=42)
    if args.tune_adam and parsed["mode"] == "truth_table":
        tune_c0_mc_adam(parsed, des, args)


if __name__ == "__main__":
    main()
