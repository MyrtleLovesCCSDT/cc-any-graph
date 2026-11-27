"""Export Boltz-2 / ColabFold 3D validation jobs for arbitrary graph designs."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.anchors import ARONSSON_PAIRS, ARONSSON_PEPTIDES, THOMAS_PAIRS, THOMAS_PEPTIDES
from src.graph_design import _parse_config


def _clean_seq(seq: str) -> str:
    return str(seq).strip().lstrip("-")


def _safe_tag(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def write_3d_job(out_dir: Path, job_id: str, chains: List[str]) -> None:
    fasta_dir = out_dir / "colabfold_fasta"
    yaml_dir = out_dir / "boltz_yaml"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    yaml_dir.mkdir(parents=True, exist_ok=True)

    clean = [_clean_seq(s) for s in chains]
    with open(fasta_dir / f"{job_id}.fasta", "w") as f:
        f.write(f">{job_id}\n" + ":".join(clean) + "\n")

    chain_ids = ["A", "B", "C", "D"]
    with open(yaml_dir / f"{job_id}.yaml", "w") as f:
        f.write("version: 1\nsequences:\n")
        for cid, seq in zip(chain_ids, clean):
            f.write(f"  - protein:\n      id: {cid}\n      sequence: {seq}\n      msa: empty\n")


def _parse_design_map(items: List[str]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid --design-map entry '{item}', expected circuit:D02,D05")
        circ, dids = item.split(":", 1)
        out[circ.strip()] = {d.strip() for d in dids.split(",") if d.strip()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 3D dimer/trimer jobs from one or more *_full.json files.")
    parser.add_argument("--full-json", type=Path, nargs="+", required=True, help="One or more <name>_full.json paths from design_graph.py")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "3d_inputs")
    parser.add_argument("--top-n", type=int, default=3, help="Number of top designs per circuit to export (ignored for circuits in --design-map).")
    parser.add_argument("--design-map", type=str, nargs="*", default=[], help="Explicit design IDs per circuit, e.g. nor_4node:D05,D13,D31")
    parser.add_argument("--existing-manifest", type=Path, default=None, help="Optional existing manifest.csv to skip already-exported chain tuples.")
    parser.add_argument("--max-offtargets", type=int, default=-1, help="Worst 1D off-target heterodimers per design (-1 for all).")
    parser.add_argument("--trimer-mode", type=str, default="targets", choices=["none", "targets", "hubs"])
    parser.add_argument("--max-trimers", type=int, default=12, help="Max trimer jobs per design (-1 for all).")
    parser.add_argument("--no-controls", action="store_true")
    args = parser.parse_args()

    design_map = _parse_design_map(args.design_map)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seen_jobs: Dict[Tuple[str, ...], str] = {}
    reused_count = 0
    new_count = 0

    if args.existing_manifest is not None and args.existing_manifest.exists():
        old_man = pd.read_csv(args.existing_manifest)
        for _, r in old_man.iterrows():
            if pd.notna(r.get("seqs")) and pd.notna(r.get("job_id")):
                key = tuple(_clean_seq(s) for s in str(r["seqs"]).split(":"))
                seen_jobs.setdefault(key, str(r["job_id"]))

    for json_path in args.full_json:
        with open(json_path, "r") as f:
            bundle = json.load(f)

        cfg = bundle["config"]
        parsed = _parse_config(cfg, allow_5mer=True)
        raw_name = parsed["name"]
        circ_name = _safe_tag(raw_name)

        if design_map:
            if raw_name in design_map:
                wanted = design_map[raw_name]
            elif circ_name in design_map:
                wanted = design_map[circ_name]
            else:
                continue
            designs = [d for d in bundle["designs"] if d["design_id"] in wanted]
        else:
            designs = bundle["designs"][: args.top_n]

        names = parsed["node_names"]
        n = len(names)
        target_pairs = parsed["target_pairs"]

        for des in designs:
            dtag = des["design_id"]
            seqs = [_clean_seq(des["seqs"][nm]) for nm in names]
            pkd_mat = np.array(des["pkd_matrix"], dtype=float)

            jobs: List[Tuple[str, str, int, int, int, List[str], float]] = []

            for u, v in sorted(target_pairs):
                role = f"edge_{names[u]}__{names[v]}"
                jobs.append((role, "target_dimer", u, v, -1, [seqs[u], seqs[v]], float(pkd_mat[u, v])))

            for u in range(n):
                if (u, u) in target_pairs:
                    continue
                role = f"homo_{names[u]}"
                jobs.append((role, "homodimer", u, u, -1, [seqs[u], seqs[u]], float(pkd_mat[u, u])))

            off_list = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) in target_pairs or (u, v) in parsed["ignored_pairs"]:
                        continue
                    off_list.append((float(pkd_mat[u, v]), u, v))
            off_list.sort(reverse=True)
            if args.max_offtargets >= 0:
                off_list = off_list[: args.max_offtargets]
            for pkd_v, u, v in off_list:
                role = f"off_{names[u]}__{names[v]}"
                jobs.append((role, "off_target", u, v, -1, [seqs[u], seqs[v]], pkd_v))

            if args.trimer_mode != "none":
                trimer_cands = []
                for u, v in sorted(target_pairs):
                    if u != v:
                        trimer_cands.append((float(pkd_mat[u, v]), u, u, v, f"trimer_{names[u]}_{names[u]}_{names[v]}"))
                        trimer_cands.append((float(pkd_mat[u, v]), u, v, v, f"trimer_{names[u]}_{names[v]}_{names[v]}"))
                for u in range(n):
                    for v in range(u + 1, n):
                        for w in range(v + 1, n):
                            n_edges = int((u, v) in target_pairs) + int((u, w) in target_pairs) + int((v, w) in target_pairs)
                            if n_edges >= 2:
                                top2 = sorted([float(pkd_mat[u, v]), float(pkd_mat[u, w]), float(pkd_mat[v, w])])[1:]
                                score = 0.5 * sum(top2)
                                trimer_cands.append((score, u, v, w, f"trimer_{names[u]}_{names[v]}_{names[w]}"))
                if args.trimer_mode == "hubs":
                    for u in range(n):
                        trimer_cands.append((float(pkd_mat[u, u]), u, u, u, f"trimer_{names[u]}_{names[u]}_{names[u]}"))
                trimer_cands.sort(reverse=True)
                if args.max_trimers >= 0:
                    trimer_cands = trimer_cands[: args.max_trimers]
                for pkd_v, u, v, w, role in trimer_cands:
                    jobs.append((role, "trimer", u, v, w, [seqs[u], seqs[v], seqs[w]], pkd_v))

            for role, pair_type, u, v, w, chains, pred_pkd in jobs:
                key = tuple(chains)
                if key not in seen_jobs:
                    job_id = _safe_tag(f"{circ_name}_{dtag}__{role}")
                    seen_jobs[key] = job_id
                    write_3d_job(args.out_dir, job_id, chains)
                    new_count += 1
                else:
                    job_id = seen_jobs[key]
                    reused_count += 1

                rows.append({
                    "job_id": job_id,
                    "circuit": raw_name,
                    "design_id": dtag,
                    "role": role,
                    "pair_type": pair_type,
                    "u": names[u],
                    "v": names[v],
                    "w": names[w] if w >= 0 else "",
                    "category": "design",
                    "seqs": ":".join(chains),
                    "pred_1d_pKd": pred_pkd,
                    "exp_log10_kd": np.nan,
                })

    if not args.no_controls:
        for dataset, pep_dict, pair_list in [
            ("Thomas_2013", THOMAS_PEPTIDES, THOMAS_PAIRS),
            ("Aronsson_2015", ARONSSON_PEPTIDES, ARONSSON_PAIRS),
        ]:
            for p1, p2, _, kd, _ in pair_list:
                if kd is None or kd <= 0:
                    continue
                s1 = _clean_seq(pep_dict[p1][0])
                s2 = _clean_seq(pep_dict[p2][0])
                chains = [s1, s2]
                key = tuple(chains)
                if key not in seen_jobs:
                    job_id = _safe_tag(f"CTRL_{dataset}_{p1}__{p2}")
                    seen_jobs[key] = job_id
                    write_3d_job(args.out_dir, job_id, chains)
                    new_count += 1
                else:
                    job_id = seen_jobs[key]
                    reused_count += 1

                rows.append({
                    "job_id": job_id,
                    "circuit": "CTRL",
                    "design_id": "CTRL",
                    "role": f"{p1}:{p2}",
                    "pair_type": "control",
                    "u": p1,
                    "v": p2,
                    "w": "",
                    "category": dataset,
                    "seqs": ":".join(chains),
                    "pred_1d_pKd": np.nan,
                    "exp_log10_kd": float(np.log10(kd)),
                })

    manifest = pd.DataFrame(rows)
    manifest_path = args.out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Exported {new_count} NEW 3D jobs (reused {reused_count}, {len(manifest)} manifest entries) to {args.out_dir}")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
