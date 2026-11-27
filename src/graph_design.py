"""Arbitrary coiled-coil graph, logic-gate, and analog-function optimizer."""

from concurrent.futures import ProcessPoolExecutor
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .circuit_eq import (
    Observable,
    evaluate_curve,
    evaluate_truth_table,
    mc_robustness_curve,
    mc_robustness_truth_table,
)
from .consensus import ConsensusModel, ensure_padded
from .sequence_space import (
    PeptideSpec,
    build_sequence,
    calc_eps280,
    generate_node_pool,
    mutate_spec,
    net_charge,
)


def _hinge_sq(deficit: float, w_sq: float = 25.0, w_lin: float = 5.0) -> float:
    if deficit <= 0.0:
        return 0.0
    return w_sq * (deficit * deficit) + w_lin * deficit


def _parse_config(cfg: dict, allow_5mer: bool = False) -> dict:
    nodes = cfg["nodes"]
    name_to_idx = {nd["name"]: idx for idx, nd in enumerate(nodes)}
    n = len(nodes)

    node_lengths: List[List[str]] = []
    node_charges: List[str] = []
    fixed_pads: List[Optional[str]] = []
    for nd in nodes:
        lens = list(nd.get("lengths", ["4mer"]))
        if allow_5mer and nd.get("allow_5mer_if_enabled", False) and "5mer" not in lens:
            lens.append("5mer")
        if not allow_5mer:
            lens = [x for x in lens if x != "5mer"] or ["4mer"]
        node_lengths.append(lens)
        node_charges.append(nd.get("charge_mode", "mixed"))
        if nd.get("fixed_seq"):
            fixed_pads.append(ensure_padded(nd["fixed_seq"], nd.get("reg")))
        else:
            fixed_pads.append(None)

    edges = []
    target_pairs = set()
    for ed in cfg.get("edges", []):
        u = name_to_idx[ed["u"]]
        v = name_to_idx[ed["v"]]
        if u > v:
            u, v = v, u
        target_pairs.add((u, v))
        edges.append({
            "u": u,
            "v": v,
            "u_name": nodes[u]["name"],
            "v_name": nodes[v]["name"],
            "pkd_min": float(ed.get("pkd_min", 5.5)),
            "pkd_max": float(ed.get("pkd_max", 12.0)),
            "require_par": bool(ed.get("require_par", True)),
            "require_shift0": bool(ed.get("require_shift0", True)),
            "min_agree": int(ed.get("min_agree", 3)),
            "min_slippage": float(ed.get("min_slippage", 0.0)),
        })

    ignored_pairs = set()
    for p in cfg.get("ignored_pairs", []):
        u, v = name_to_idx[p[0]], name_to_idx[p[1]]
        ignored_pairs.add((min(u, v), max(u, v)))

    hierarchies = []
    for h in cfg.get("hierarchies", []):
        su, sv = name_to_idx[h["strong"][0]], name_to_idx[h["strong"][1]]
        wu, wv = name_to_idx[h["weak"][0]], name_to_idx[h["weak"][1]]
        hierarchies.append({
            "su": min(su, sv),
            "sv": max(su, sv),
            "wu": min(wu, wv),
            "wv": max(wu, wv),
            "min_dpkd": float(h.get("min_dpkd", 1.2)),
            "min_agree": int(h.get("min_agree", 3)),
        })

    mode = cfg.get("mode", "graph")
    states_list = cfg.get("states_uM", [])
    states_c0_m = np.zeros((len(states_list), n), dtype=float)
    for r_idx, st in enumerate(states_list):
        for k_name, val in st.items():
            states_c0_m[r_idx, name_to_idx[k_name]] = float(val) * 1e-6

    observables = []
    for obs_cfg in cfg.get("observables", []):
        o_nodes = tuple(name_to_idx[x] for x in obs_cfg["nodes"])
        r_nodes = tuple(name_to_idx[x] for x in obs_cfg.get("ref_nodes", obs_cfg["nodes"]))
        obs_obj = Observable(kind=obs_cfg["kind"], nodes=o_nodes, ref_nodes=r_nodes)
        observables.append({
            "obj": obs_obj,
            "targets": obs_cfg.get("targets", obs_cfg.get("target_y", [])),
            "min_delta": float(obs_cfg.get("min_delta", 0.45)),
            "min_fold": float(obs_cfg.get("min_fold", 3.0)),
            "min_on": float(obs_cfg.get("min_on", 0.60)),
            "max_off": float(obs_cfg.get("max_off", 0.20)),
            "affine_fit": bool(obs_cfg.get("affine_fit", False)),
            "min_span": float(obs_cfg.get("min_span", 0.45)),
            "max_rmse": float(obs_cfg.get("max_rmse", 0.12)),
        })

    tunable_scaffolds = []
    for ts in cfg.get("tunable_scaffolds", []):
        tunable_scaffolds.append({
            "idx": name_to_idx[ts["node"]],
            "min_uM": float(ts.get("min_uM", 0.5)),
            "max_uM": float(ts.get("max_uM", 20.0)),
        })

    return {
        "name": cfg.get("name", "custom_circuit"),
        "mode": mode,
        "n_nodes": n,
        "node_names": [nd["name"] for nd in nodes],
        "node_lengths": node_lengths,
        "node_charges": node_charges,
        "fixed_pads": fixed_pads,
        "edges": edges,
        "target_pairs": target_pairs,
        "ignored_pairs": ignored_pairs,
        "hierarchies": hierarchies,
        "off_target_max_pkd": float(cfg.get("off_target_max_pkd", 4.8)),
        "homodimer_max_pkd": float(cfg.get("homodimer_max_pkd", 4.5)),
        "states_c0_m": states_c0_m,
        "observables": observables,
        "tunable_scaffolds": tunable_scaffolds,
        "mc_conc_rel_std": float(cfg.get("mc_conc_rel_std", 0.0)),
    }


def _cached_slippage(
    model: ConsensusModel,
    pad_u: str,
    pad_v: str,
    slip_cache: Optional[Dict[Tuple[str, str], Tuple[float, str]]],
) -> Tuple[float, str]:
    a, b = (pad_u, pad_v) if pad_u <= pad_v else (pad_v, pad_u)
    if slip_cache is not None and (a, b) in slip_cache:
        return slip_cache[(a, b)]
    res = model.slippage_margin(a, b)
    if slip_cache is not None:
        slip_cache[(a, b)] = res
    return res


def _score_matrix(
    pair_mat: List[List[dict]],
    parsed: dict,
    model_names: Sequence[str],
    c0_override_m: Optional[np.ndarray] = None,
    model: Optional[ConsensusModel] = None,
    pads: Optional[Sequence[str]] = None,
    slip_cache: Optional[Dict[Tuple[str, str], Tuple[float, str]]] = None,
) -> Tuple[float, dict]:
    n = parsed["n_nodes"]
    target_pairs = parsed["target_pairs"]
    ignored_pairs = parsed["ignored_pairs"]
    off_max = parsed["off_target_max_pkd"]
    homo_max = parsed["homodimer_max_pkd"]

    graph_loss = 0.0

    # 1. Target edges
    for ed in parsed["edges"]:
        u, v = ed["u"], ed["v"]
        sc = pair_mat[u][v]
        pkd = sc["pKd"]
        graph_loss += _hinge_sq(ed["pkd_min"] - pkd, 30.0, 6.0)
        graph_loss += _hinge_sq(pkd - ed["pkd_max"], 30.0, 6.0)
        if ed["require_par"] and sc.get("iCipa_best_orient", "PAR") != "PAR":
            graph_loss += 60.0
        if ed["require_shift0"] and sc.get("iCipa_best_shift", 0) != 0:
            graph_loss += 45.0
        if ed["min_slippage"] > 0.0 and model is not None and pads is not None:
            slip, _ = _cached_slippage(model, pads[u], pads[v], slip_cache)
            graph_loss += _hinge_sq(ed["min_slippage"] - slip, 0.5, 1.5)

    # 2. Pairwise hierarchies & multi-model consensus
    hier_margin_bonus = 0.0
    for h in parsed["hierarchies"]:
        sc_s = pair_mat[h["su"]][h["sv"]]
        sc_w = pair_mat[h["wu"]][h["wv"]]
        dpkd = sc_s["pKd"] - sc_w["pKd"]
        graph_loss += _hinge_sq(h["min_dpkd"] - dpkd, 25.0, 5.0)
        if dpkd > h["min_dpkd"]:
            hier_margin_bonus += min(dpkd - h["min_dpkd"], 1.5)
        agree = sum(sc_s[m] > sc_w[m] for m in model_names)
        if agree < h["min_agree"]:
            graph_loss += 8.0 * float(h["min_agree"] - agree)

    graph_loss -= 0.4 * hier_margin_bonus

    # 3. Off-target & homodimer suppression (including all shifts & ANTI via predict_pair)
    worst_off_pkd = -99.0
    worst_homo_pkd = -99.0
    off_margin_bonus = 0.0
    for i in range(n):
        for j in range(i, n):
            if (i, j) in target_pairs or (i, j) in ignored_pairs:
                continue
            pkd = pair_mat[i][j]["pKd"]
            cap = homo_max if i == j else off_max
            if i == j:
                worst_homo_pkd = max(worst_homo_pkd, pkd)
            else:
                worst_off_pkd = max(worst_off_pkd, pkd)
            graph_loss += _hinge_sq(pkd - cap, 25.0, 5.0)
            if pkd < cap:
                off_margin_bonus += min(cap - pkd, 1.5)

    graph_loss -= 0.15 * off_margin_bonus

    # 4. EQTK thermodynamic objective
    circuit_loss = 0.0
    metrics: Dict[str, float] = {
        "graph_loss": float(graph_loss),
        "worst_off_pKd": float(worst_off_pkd),
        "worst_homo_pKd": float(worst_homo_pkd),
    }
    mode = parsed["mode"]
    c0_m = parsed["states_c0_m"] if c0_override_m is None else c0_override_m

    if mode in ("truth_table", "curve") and c0_m.size > 0 and parsed["observables"]:
        log10_kd_mat = np.array([[pair_mat[i][j]["log10_kd"] for j in range(n)] for i in range(n)], dtype=float)
        kd_mat_m = 10.0 ** np.minimum(log10_kd_mat, -2.0)

        if mode == "truth_table":
            deltas, folds = [], []
            for o_idx, obs in enumerate(parsed["observables"]):
                tt = evaluate_truth_table(kd_mat_m, c0_m, obs["obj"], obs["targets"])
                deltas.append(tt["Delta"])
                folds.append(tt["Fold"])
                circuit_loss += _hinge_sq(obs["min_delta"] - tt["Delta"], 90.0, 18.0)
                circuit_loss += _hinge_sq(obs["min_on"] - tt["min_ON"], 50.0, 8.0)
                circuit_loss += _hinge_sq(tt["max_OFF"] - obs["max_off"], 50.0, 8.0)
                circuit_loss += _hinge_sq(obs["min_fold"] - tt["Fold"], 4.0, 1.0)
                circuit_loss -= 12.0 * tt["Delta"]
                if o_idx == 0:
                    metrics["min_ON"] = tt["min_ON"]
                    metrics["max_OFF"] = tt["max_OFF"]
            metrics["Delta"] = float(min(deltas))
            metrics["Fold"] = float(min(folds))
        else:
            rmses, r2s, spans = [], [], []
            for obs in parsed["observables"]:
                cv = evaluate_curve(kd_mat_m, c0_m, obs["obj"], obs["targets"], affine_fit=obs["affine_fit"])
                rmses.append(cv["RMSE"])
                r2s.append(cv["R2"])
                spans.append(cv["Span"])
                circuit_loss += 180.0 * (cv["RMSE"] ** 2) + 45.0 * cv["RMSE"]
                circuit_loss += _hinge_sq(obs["min_span"] - cv["Span"], 80.0, 15.0)
                circuit_loss -= 4.0 * min(cv["Span"], 0.85)
            metrics["RMSE"] = float(max(rmses))
            metrics["R2"] = float(min(r2s))
            metrics["Span"] = float(min(spans))

    total_loss = graph_loss + circuit_loss
    metrics["circuit_loss"] = float(circuit_loss)
    metrics["total_loss"] = float(total_loss)
    return float(total_loss), metrics


def _build_pair_matrix(model: ConsensusModel, pads: Sequence[str]) -> List[List[dict]]:
    n = len(pads)
    mat: List[List[dict]] = [[{} for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            sc = model.predict_pair(pads[i], pads[j])
            mat[i][j] = sc
            mat[j][i] = sc
    return mat


def _worker_sa(args_tuple) -> List[Tuple[float, Tuple[str, ...], dict]]:
    (
        cfg_dict,
        calib_path_str,
        pools_by_node,
        worker_seed,
        n_restarts,
        n_steps,
        t_start,
        t_end,
        allow_5mer,
    ) = args_tuple

    rng = np.random.default_rng(worker_seed)
    model = ConsensusModel(Path(calib_path_str))
    parsed = _parse_config(cfg_dict, allow_5mer=allow_5mer)
    n = parsed["n_nodes"]
    mutable_nodes = [i for i in range(n) if parsed["fixed_pads"][i] is None]
    slip_cache: Dict[Tuple[str, str], Tuple[float, str]] = {}

    # Hall of fame: raw_tuple -> (loss, raw_tuple, pads_tuple, metrics)
    hof: Dict[Tuple[str, ...], Tuple[float, Tuple[str, ...], Tuple[str, ...], dict]] = {}

    for _restart in range(n_restarts):
        specs: List[Optional[PeptideSpec]] = [None] * n
        pads: List[str] = [""] * n
        raws: List[str] = [""] * n
        used_raws = set()

        for i in range(n):
            if parsed["fixed_pads"][i] is not None:
                pads[i] = str(parsed["fixed_pads"][i])
                raws[i] = pads[i].lstrip("-")
                used_raws.add(raws[i])
            else:
                pool = pools_by_node[i]
                for _t in range(40):
                    sp = pool[int(rng.integers(0, len(pool)))]
                    p_i, r_i = build_sequence(sp)
                    if r_i not in used_raws or _t == 39:
                        specs[i] = sp
                        pads[i] = p_i
                        raws[i] = r_i
                        used_raws.add(r_i)
                        break

        cur_mat = _build_pair_matrix(model, pads)
        cur_loss, cur_metrics = _score_matrix(
            cur_mat, parsed, model.models, model=model, pads=pads, slip_cache=slip_cache
        )

        best_key = tuple(raws)
        hof[best_key] = (cur_loss, best_key, tuple(pads), cur_metrics)

        if not mutable_nodes:
            continue

        for step in range(n_steps):
            frac = step / max(n_steps - 1, 1)
            temp = t_start * ((t_end / t_start) ** frac)

            k = mutable_nodes[int(rng.integers(0, len(mutable_nodes)))]
            pool_k = pools_by_node[k]

            if rng.random() < 0.30 and pool_k:
                cand_spec = pool_k[int(rng.integers(0, len(pool_k)))]
            else:
                cand_spec = mutate_spec(
                    specs[k],
                    parsed["node_lengths"][k],
                    parsed["node_charges"][k],
                    rng,
                )

            cand_pad, cand_raw = build_sequence(cand_spec)
            if cand_raw == raws[k] or any(cand_raw == raws[j] for j in range(n) if j != k):
                continue

            # Incremental O(N) update: only pairs touching node k
            row_k = [
                model.predict_pair(cand_pad, cand_pad) if j == k else model.predict_pair(cand_pad, pads[j])
                for j in range(n)
            ]

            new_mat = [row[:] for row in cur_mat]
            for j in range(n):
                new_mat[k][j] = row_k[j]
                new_mat[j][k] = row_k[j]

            cand_pads = pads[:]
            cand_pads[k] = cand_pad
            new_loss, new_metrics = _score_matrix(
                new_mat, parsed, model.models, model=model, pads=cand_pads, slip_cache=slip_cache
            )
            delta = new_loss - cur_loss
            if delta <= 0.0 or rng.random() < math.exp(-delta / max(temp, 1e-6)):
                specs[k] = cand_spec
                pads[k] = cand_pad
                raws[k] = cand_raw
                cur_mat = new_mat
                cur_loss = new_loss
                cur_metrics = new_metrics

                rk = tuple(raws)
                if rk not in hof or new_loss < hof[rk][0]:
                    hof[rk] = (new_loss, rk, tuple(pads), new_metrics)

        if len(hof) > 250:
            top_items = sorted(hof.values(), key=lambda x: x[0])[:120]
            hof = {item[1]: item for item in top_items}

    sorted_hof = sorted(hof.values(), key=lambda x: x[0])[:60]
    return [(item[0], item[2], item[3]) for item in sorted_hof]


def _polish_scaffold_concentrations(
    log10_kd_mat: np.ndarray,
    parsed: dict,
) -> Optional[np.ndarray]:
    tunables = parsed["tunable_scaffolds"]
    if not tunables or parsed["states_c0_m"].size == 0 or not parsed["observables"]:
        return None

    base_c0 = parsed["states_c0_m"].copy()
    kd_mat_m = 10.0 ** np.minimum(log10_kd_mat, -2.0)

    x0 = []
    bounds = []
    for ts in tunables:
        idx = ts["idx"]
        nonzero = base_c0[:, idx][base_c0[:, idx] > 0.0]
        cur_um = float(np.median(nonzero) * 1e6) if nonzero.size else 5.0
        x0.append(cur_um)
        bounds.append((ts["min_uM"], ts["max_uM"]))

    def _obj(x_vec):
        c0_try = base_c0.copy()
        bound_pen = 0.0
        for val_um, ts in zip(x_vec, tunables):
            val_um_clamped = float(np.clip(val_um, ts["min_uM"], ts["max_uM"]))
            if val_um != val_um_clamped:
                bound_pen += 50.0 * ((val_um - val_um_clamped) ** 2) + 5.0 * abs(val_um - val_um_clamped)
            mask = base_c0[:, ts["idx"]] > 0.0
            c0_try[mask, ts["idx"]] = val_um_clamped * 1e-6

        if parsed["mode"] == "truth_table":
            pen = bound_pen
            for obs in parsed["observables"]:
                tt = evaluate_truth_table(kd_mat_m, c0_try, obs["obj"], obs["targets"])
                pen += _hinge_sq(obs["min_delta"] - tt["Delta"], 90.0, 18.0)
                pen += _hinge_sq(obs["min_on"] - tt["min_ON"], 50.0, 8.0)
                pen += _hinge_sq(tt["max_OFF"] - obs["max_off"], 50.0, 8.0)
                pen += _hinge_sq(obs["min_fold"] - tt["Fold"], 4.0, 1.0)
                pen -= 12.0 * tt["Delta"]
            return pen

        pen = bound_pen
        for obs in parsed["observables"]:
            cv = evaluate_curve(kd_mat_m, c0_try, obs["obj"], obs["targets"], affine_fit=obs["affine_fit"])
            pen += 180.0 * (cv["RMSE"] ** 2) + 45.0 * cv["RMSE"]
            pen += _hinge_sq(obs["min_span"] - cv["Span"], 80.0, 15.0)
            pen -= 4.0 * min(cv["Span"], 0.85)
        return pen

    res = minimize(_obj, x0=np.array(x0, dtype=float), method="Nelder-Mead", options={"maxiter": 150})
    best_c0 = base_c0.copy()
    for val_um, ts in zip(res.x, tunables):
        val_um_clamped = float(np.clip(val_um, ts["min_uM"], ts["max_uM"]))
        mask = base_c0[:, ts["idx"]] > 0.0
        best_c0[mask, ts["idx"]] = val_um_clamped * 1e-6
    return best_c0


def run_graph_design(
    cfg: dict,
    calib_path: Path,
    allow_5mer: bool = False,
    n_restarts: int = 16,
    n_steps: int = 1500,
    n_workers: int = 4,
    top_mc_audit: int = 80,
    n_mc: int = 200,
    tune_conc: bool = False,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Run multi-start Simulated Annealing + EQTK MC audit for an arbitrary graph config."""
    model = ConsensusModel(calib_path)
    parsed = _parse_config(cfg, allow_5mer=allow_5mer)
    n = parsed["n_nodes"]

    # Build/cache candidate pools keyed by (charge_mode, tuple(lengths))
    pool_cache: Dict[Tuple[str, Tuple[str, ...]], List[PeptideSpec]] = {}
    pools_by_node: List[List[PeptideSpec]] = []
    homo_cutoff_nM = (10.0 ** (-parsed["homodimer_max_pkd"])) * 1e9
    for i in range(n):
        if parsed["fixed_pads"][i] is not None:
            pools_by_node.append([])
            continue
        key = (parsed["node_charges"][i], tuple(parsed["node_lengths"][i]))
        if key not in pool_cache:
            pool_cache[key] = generate_node_pool(
                model,
                charge_mode=key[0],
                lengths=key[1],
                homo_min_nM=min(homo_cutoff_nM, 15_000.0),
                seed=seed + len(pool_cache) * 17,
            )
            if not pool_cache[key]:
                raise RuntimeError(f"Empty pool generated for node config {key}")
        pools_by_node.append(pool_cache[key])

    pool_sizes = {parsed["node_names"][i]: len(pools_by_node[i]) for i in range(n)}
    print(f"Pools ready: {pool_sizes}")

    t_start, t_end = 8.0, 0.15
    workers = max(1, min(n_workers, n_restarts))
    worker_results: List[Tuple[float, Tuple[str, ...], dict]] = []

    if workers == 1:
        worker_results = _worker_sa((
            cfg, str(calib_path), pools_by_node, seed, n_restarts, n_steps, t_start, t_end, allow_5mer
        ))
    else:
        restarts_per_w = [n_restarts // workers + (1 if i < n_restarts % workers else 0) for i in range(workers)]
        jobs = [
            (cfg, str(calib_path), pools_by_node, seed + 1009 * idx, r_cnt, n_steps, t_start, t_end, allow_5mer)
            for idx, r_cnt in enumerate(restarts_per_w)
            if r_cnt > 0
        ]
        with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
            for chunk in ex.map(_worker_sa, jobs):
                worker_results.extend(chunk)

    # Deduplicate across workers
    dedup: Dict[Tuple[str, ...], Tuple[float, Tuple[str, ...], dict]] = {}
    for loss, pads_t, metrics in worker_results:
        if pads_t not in dedup or loss < dedup[pads_t][0]:
            dedup[pads_t] = (loss, pads_t, metrics)

    candidates = sorted(dedup.values(), key=lambda x: x[0])[:top_mc_audit]
    if not candidates:
        return pd.DataFrame(), []

    # Full audit: exact slippage margins, optional concentration tuning, and Monte Carlo robustness
    records: List[dict] = []
    full_designs: List[dict] = []
    mode = parsed["mode"]
    names = parsed["node_names"]

    slip_cache: Dict[Tuple[str, str], Tuple[float, str]] = {}
    for rank_idx, (loss, pads_t, base_metrics) in enumerate(candidates, 1):
        pads = list(pads_t)
        raws = [p.lstrip("-") for p in pads]
        mat = _build_pair_matrix(model, pads)
        log10_kd_mat = np.array([[mat[i][j]["log10_kd"] for j in range(n)] for i in range(n)], dtype=float)
        pkd_mat = -log10_kd_mat

        c0_m = parsed["states_c0_m"]
        if tune_conc and parsed["tunable_scaffolds"]:
            tuned = _polish_scaffold_concentrations(log10_kd_mat, parsed)
            if tuned is not None:
                c0_m = tuned
        loss, base_metrics = _score_matrix(
            mat, parsed, model.models, c0_override_m=c0_m, model=model, pads=pads, slip_cache=slip_cache
        )

        # Edge slippage check
        edge_details = {}
        min_slip = np.inf
        for ed in parsed["edges"]:
            u, v = ed["u"], ed["v"]
            slip, slip_type = _cached_slippage(model, pads[u], pads[v], slip_cache)
            if ed["require_par"] and ed["require_shift0"]:
                min_slip = min(min_slip, slip)
            edge_details[f"{ed['u_name']}:{ed['v_name']}"] = {
                "pKd": float(pkd_mat[u, v]),
                "Kd_nM": float(mat[u][v]["Kd_nM"]),
                "orient": mat[u][v].get("iCipa_best_orient", "PAR"),
                "shift": int(mat[u][v].get("iCipa_best_shift", 0)),
                "slip_margin": float(slip),
                "slip_type": slip_type,
            }
        if np.isinf(min_slip):
            min_slip = np.nan

        mc_summary: Dict[str, float] = {}
        if mode == "truth_table" and c0_m.size > 0 and parsed["observables"]:
            mc_seed = seed + rank_idx * 13
            mc_list = [
                mc_robustness_truth_table(
                    log10_kd_mat,
                    c0_m,
                    obs["obj"],
                    obs["targets"],
                    n_mc=n_mc,
                    sigma_kd=model.loo_rmse,
                    conc_rel_std=parsed["mc_conc_rel_std"],
                    min_delta=obs["min_delta"] * 0.85,
                    min_fold=obs["min_fold"] * 0.85,
                    seed=mc_seed,
                    return_arrays=True,
                )
                for obs in parsed["observables"]
            ]
            joint_pass = np.logical_and.reduce([m["_pass_mask"] for m in mc_list])
            mc_summary = {
                "MC_Pass_rate": float(np.mean(joint_pass)),
                "MC_Delta_med": float(min(m["MC_Delta_med"] for m in mc_list)),
                "MC_Delta_p10": float(min(m["MC_Delta_p10"] for m in mc_list)),
                "MC_Fold_med": float(min(m["MC_Fold_med"] for m in mc_list)),
            }
        elif mode == "curve" and c0_m.size > 0 and parsed["observables"]:
            mc_seed = seed + rank_idx * 13
            mc_list = [
                mc_robustness_curve(
                    log10_kd_mat,
                    c0_m,
                    obs["obj"],
                    obs["targets"],
                    n_mc=n_mc,
                    sigma_kd=model.loo_rmse,
                    conc_rel_std=parsed["mc_conc_rel_std"],
                    affine_fit=obs["affine_fit"],
                    max_rmse=obs["max_rmse"],
                    min_span=obs["min_span"] * 0.85,
                    seed=mc_seed,
                    return_arrays=True,
                )
                for obs in parsed["observables"]
            ]
            joint_pass = np.logical_and.reduce([m["_pass_mask"] for m in mc_list])
            mc_summary = {
                "MC_Pass_rate": float(np.mean(joint_pass)),
                "MC_RMSE_med": float(max(m["MC_RMSE_med"] for m in mc_list)),
                "MC_R2_med": float(min(m["MC_R2_med"] for m in mc_list)),
                "MC_Span_med": float(min(m["MC_Span_med"] for m in mc_list)),
            }

        row: Dict[str, object] = {
            "total_loss": float(loss),
            "graph_loss": float(base_metrics["graph_loss"]),
            "circuit_loss": float(base_metrics["circuit_loss"]),
            "worst_off_pKd": float(base_metrics["worst_off_pKd"]),
            "worst_homo_pKd": float(base_metrics["worst_homo_pKd"]),
            "min_slip_margin": float(min_slip) if not np.isnan(min_slip) else np.nan,
        }
        for k in ("Delta", "Fold", "min_ON", "max_OFF", "RMSE", "R2", "Span"):
            if k in base_metrics:
                row[k] = float(base_metrics[k])
        row.update(mc_summary)

        for e_lbl, e_info in edge_details.items():
            row[f"pKd_{e_lbl}"] = e_info["pKd"]
        for i, nm in enumerate(names):
            row[f"{nm}_seq"] = raws[i]
            row[f"{nm}_pad"] = pads[i]
            row[f"{nm}_len"] = len(raws[i])
            row[f"{nm}_chg"] = net_charge(raws[i])
            row[f"{nm}_eps280"] = calc_eps280(raws[i])

        records.append(row)
        full_designs.append({
            "seqs": {nm: raws[i] for i, nm in enumerate(names)},
            "pads": {nm: pads[i] for i, nm in enumerate(names)},
            "pkd_matrix": pkd_mat.tolist(),
            "c0_uM": (c0_m * 1e6).tolist(),
            "edges": edge_details,
            "metrics": {k: v for k, v in row.items() if isinstance(v, (int, float))},
        })

    df = pd.DataFrame(records)
    # Sort primarily by MC_Pass_rate (if present), then lowest total_loss
    if "MC_Pass_rate" in df.columns:
        df["_orig_idx"] = np.arange(len(df))
        sort_cols = ["MC_Pass_rate", "total_loss"]
        asc = [False, True]
        df = df.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
        order = df["_orig_idx"].astype(int).tolist()
        df = df.drop(columns=["_orig_idx"])
        full_designs = [full_designs[i] for i in order]

    df.insert(0, "Design_ID", [f"D{i + 1:02d}" for i in range(len(df))])
    for i, d in enumerate(full_designs):
        d["design_id"] = f"D{i + 1:02d}"

    return df, full_designs
