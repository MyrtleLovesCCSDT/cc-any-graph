# CC-any-graph

Universal 1D + EQTK + 3D validation framework for *de novo* coiled-coil peptide design across arbitrary interaction graphs, Boolean logic gates, and analog concentration-response curves.

Extended from [coiled-coil-consensus](https://github.com/MyrtleLovesCCSDT/coiled-coil-consensus) (YES-gate pipeline) to support:
- **`mode: "graph"`**: arbitrary target interaction topologies, edge hierarchies, and off-target/homodimer suppression.
- **`mode: "truth_table"`**: all 10 non-trivial input-permutation equivalence classes of two-input Boolean logic (covering all 16 two-input functions up to `I1`/`I2` relabeling and trivial constants `0`/`1`), plus 3-input `OR` (`or3_5node`) and dual-output `ortho2_6node`.
- **`mode: "curve"`**: quantitative concentration-response curves (quadratic decay, prozone bandpass domes, and decoy-buffered sharp threshold switches).

---

## Summary of 1D + EQTK Results

All pairwise 1D affinities are computed with symmetric `(A,B)/(B,A)` scoring and clean buffer resets in `src/consensus.py`, and final design metrics are recomputed directly from the saved `pkd_matrix` and `c0_uM`.

Two Monte Carlo stages are reported separately:
1. **`Design MC` (`design_graph.py`, `n_mc = 200`, `sigma_pKd = 0.643`)**: post-SA ranking stage with `mc_conc_rel_std = 0.0` for all configs except `xor_4node` (`0.05`).
2. **`Sim MC` (`simulate_circuit.py`, `n_mc = 400`, `sigma_pKd = 0.643`, `conc_std = 0.015`)**: dense diagnostic sweep modeling `1.5%` nanoliter dispenser pipetting noise (matching all diagnostic plots in `assets/`).

### Logic Circuits (`mode: "truth_table"`)

| Circuit | Topology / Readout | Delta | Fold | Thresh (`0.85*min_Δ`) | `MC_Delta_p10` | Design MC (`n=200`) | Sim MC (`n=400, std=0.015`) | Worst Off / Homo pKd | Min Slip Margin |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `yes_3node` | Buffer (`I-Q-S`, `free_frac(S)`) | 0.959 | 27.7x | 0.383 | 0.795 | 99.5% | 96.2% | 1.70 / 3.30 | 3.3 |
| `not_3node` | Inverter (`I-A-S`, `dimer_frac(A,S)`) | 0.975 | 195.3x | 0.383 | 0.855 | 99.5% | 99.8% | 1.56 / 2.78 | 27.9 |
| `or_4node` | OR (`I1,I2-Q-S`, `free_frac(S)`) | 0.936 | 21.9x | 0.383 | 0.663 | 98.0% | 92.8% | 2.52 / 3.05 | 13.2 |
| `nor_4node` | Compact NOR (`dimer_frac(Q,S)`) | 0.981 | 192.3x | 0.552 | 0.874 | 96.0% | 93.8% | 2.46 / 3.49 | 21.2 |
| `nor_5node` | Fig. S4 NOR (`free_frac(S)`, 5-mer) | 0.952 | 34.9x | 0.425 | 0.702 | 93.5% | 92.2% | 3.88 / 3.66 | 23.7 |
| `and_5node` | Fig. S4 AND (`free_frac(S)`, 4-mer) | 0.910 | 34.0x | 0.552 | 0.476 | 71.0% | 67.5% | 4.36 / 3.61 | 22.8 |
| `and_6node` | Split-reporter AND (`SA:SB`, 4-mer) | 0.502 | 10.4x | 0.425 | 0.139 | 31.5% | 26.2% | 4.97 / 3.65 | 23.2 |
| `nand_6node` | Fig. S4 NAND (`free_frac(S)`, 5-mer) | 0.730 | 5.6x | 0.425 | 0.199 | 41.5% | 35.2% | 4.27 / 3.99 | -22.8* |
| `imply_5node` | Fig. S4 `NOT I1 OR I2` (`free_frac(S)`) | 0.827 | 13.5x | 0.425 | 0.352 | 71.5% | 64.2% | 4.33 / 3.78 | 27.9 |
| `nimply_4node` | Compact `I1 AND NOT I2` (`free_frac(S)`) | 0.978 | 66.9x | 0.552 | 0.843 | 93.0% | 90.5% | 4.27 / 4.03 | 1.7 |
| `nimply_6node` | Fig. S4 `I1 AND NOT I2` (`free_frac(S)`) | 0.792 | 15.0x | 0.425 | 0.187 | 51.0% | 54.0% | 4.61 / 3.97 | 23.3 |
| `xor_4node` | Competitive sink (`A:B >> A,B:Q > Q:S`) | 0.748 | 5.9x | 0.297 | 0.184 | 68.0% (`std=0.05`) | 71.0% | 2.91 / 3.97 | 19.9 |
| `xnor_4node` | Dimer readout of XOR (`dimer_frac(Q,S)`) | 0.602 | 8.3x | 0.425 | 0.135 | 40.5% | 34.8% | 3.12 / 4.58 | 25.7 |
| `xnor_5node` | Mediator XNOR (`A:B > A,B:M > M:Q > Q:S`) | 0.551 | 3.6x | 0.383 | 0.094 | 23.0% | 19.2% | 4.05 / 4.46 | -33.9* |
| `or3_5node` | 3-input OR (8 states, 4-mer) | 0.964 | 39.0x | 0.510 | 0.774 | 93.5% | 87.8% | 3.44 / 3.91 | 22.8 |
| `ortho2_6node` | Dual orthogonal YES outputs | 0.919 | 14.0x | 0.383 | 0.645 | 89.5% (joint) | 93.2% / 90.5% (per-obs) | 3.28 / 3.98 | 32.7 |

*\*Note on negative `min_slip_margin`: computed over `require_par=true` and `require_shift0=true` edges. Negative margins occur on 5-mer/4-mer shift-7 edges: `nand_6node` D01 `M1:Q1` (`shift=7`, `slip=-22.8`), `xnor_5node` D01 `M:Q` (`shift=-7`, `slip=-33.9`).*

### Analog Curves (`mode: "curve"`)

| Design (Source) | Objective | R2 | RMSE | Span | Design MC Exact (`n=200, std=0`) | Sim MC Exact (`n=400, std=0.015`) | Sim Shape Preserved (`Exact + Shifted/Scaled`) | Worst Off / Homo pKd | Min Slip |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `analog_bandpass` (preset `bandpass`) | Prozone bandpass dome | 0.890 | 0.077 | 0.661 | 17.5% | 13.0% | 80.2% (13.0% + 67.2%) | - / 4.02 | 35.3 |
| `dome_3node` (`dome_3node.json`, D01) | Prozone bandpass dome | 0.890 | 0.070 | 0.745 | 24.0% | 15.8% | 91.6% (15.8% + 75.8%) | - / 4.46 | 32.4 |
| `analog_neg_x2` (preset `neg_x2`) | `0.95*(1-(X/8.5)^2)` | 0.957 | 0.059 | 0.747 | 23.5% | 19.8% | 77.3% (19.8% + 57.5%) | 4.29 / 3.90 | 21.2 |
| `threshold_decoy_4node` (`threshold_decoy_4node.json`) | Decoy-buffered sharp switch | 0.991 | 0.031 | 0.926 | 69.5% | 69.2% | 97.4% (69.2% + 28.2%) | 3.09 / 4.11 | 22.5 |

*Note on `dome_3node`: D01 (`worst_homo = 4.46`) is shown in `assets/dome_3node.png`; D08 (`R2 = 0.883`, `RMSE = 0.072`, `worst_homo = 2.64`) is the preferred low-homodimer candidate.*

### Arbitrary Interaction Graphs (`mode: "graph"`)

| Design (Config File) | Topology | `graph_loss` | Graph MC (`gap>=1 / order`, `n=400`) | Worst Off / Homo pKd | Min Slip Margin |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `custom_graph_xor_topo` (`configs/circuits/custom_graph_test.json`) | Triangle + tail (`A:B > A,B:Q > Q:S`) | -1.110 | 65.8% / 72.2% | 3.09 / 3.74 | 16.8 |
| `cycle_c4` (`configs/circuits/cycle_c4.json`) | 4-cycle (`A-B-C-D-A`, no diagonals) | -1.350 | 100.0% / 100.0% | 2.72 / 2.79 | 32.7 |
| `bipartite_k23` (`configs/circuits/bipartite_k23.json`) | `K2,3` bipartite graph | -2.025 | 100.0% / 100.0% | 2.66 / 2.39 | 23.9 |
| `cascade_5node` (`configs/circuits/cascade_5node.json`) | Weighted chain (`A:B > B:C > C:D > D:E`) | -2.107 | 38.2% / 55.8% | 4.06 / 3.08 | 13.8 |
| `barbell_6node` (`configs/circuits/barbell_6node.json`) | Two triangles + bridge (`C1:C2`) | 3.715 | 37.2% / 89.0% | 4.35 / 4.49 | 16.4 |

---

## Diagnostic Examples (`simulate_circuit.py`, `n_mc = 400`, `conc_std = 0.015`)

| `and_5node` (Sim MC Pass: 67.5%) | `xor_4node` (Sim MC Pass: 71.0%) |
| :---: | :---: |
| ![and_5node](assets/and_5node.png) | ![xor_4node](assets/xor_4node.png) |

| `dome_3node` D01 (Exact: 15.8%, Dome: 91.6%) | `threshold_decoy_4node` (Exact: 69.2%, Shape: 97.4%) |
| :---: | :---: |
| ![dome_3node](assets/dome_3node.png) | ![threshold_decoy_4node](assets/threshold_decoy_4node.png) |

---

## Designed Sequences & Data

All top candidates per circuit (`*_full.json`, `*_designs.csv`) are committed under `results/graph_designs/<name>/`, alongside a consolidated top-3 summary table in [`designs_top3_summary.csv`](designs_top3_summary.csv).

---

## Quickstart

### 1. Design from preset or custom JSON
```bash
python3 scripts/design_graph.py --preset xor --allow-5mer --tune-conc --restarts 32 --steps 2500
python3 scripts/design_graph.py --config configs/circuits/and_5node.json --out-dir results/graph_designs --tune-conc --restarts 32 --steps 2500
```

### 2. Dense sweep & Monte Carlo failure diagnostics
```bash
python3 scripts/simulate_circuit.py --full-json results/graph_designs/and_5node/and_5node_full.json --conc-std 0.015 --n-mc 400
python3 scripts/plot_graph_mc.py --full-json results/graph_designs/cycle_c4/cycle_c4_full.json
```

### 3. 3D validation pipeline (Boltz-2 + PyRosetta)
```bash
python3 scripts/export_3d_jobs_graph.py --full-json results/graph_designs/*/*_full.json --top-n 1 --trimer-mode hubs --max-offtargets -1 --max-trimers 12
python3 scripts/analyze_boltz.py --boltz-dir results/boltz_out --manifest results/3d_inputs/manifest.csv --out results/boltz_3d_summary.csv
python3 scripts/score_rosetta.py --boltz-summary results/boltz_3d_summary.csv --manifest results/3d_inputs/manifest.csv --out-dir results/rosetta --out results/rosetta_3d_summary.csv --skip-existing
python3 scripts/scorecard_3d_graph.py --full-json results/graph_designs/and_5node/and_5node_full.json --manifest results/3d_inputs/manifest.csv --rosetta-summary results/rosetta_3d_summary.csv --out-dir results/3d_scorecards
```

---

## Repository Structure

```text
CC-any-graph/
├── configs/
│   ├── consensus_calibration.json   # 4-model Ridge fit (loo_r=0.917, loo_rmse=0.643)
│   └── circuits/                    # Arbitrary graph, logic gate & analog curve JSON specs
├── src/
│   ├── anchors.py                   # Experimental anchor datasets
│   ├── consensus.py                 # Symmetric + buffer-clean ConsensusModel & slippage_margin
│   ├── sequence_space.py            # PeptideSpec, mixed E/K barcodes, 3/35/4/5-mer spaces
│   ├── circuit_eq.py                # Generalized EQTK equilibrium + observables + MC
│   ├── graph_design.py              # Multi-start SA + Nelder-Mead concentration tuning
│   └── structure_3d.py              # Boltz-2 CIF structural metrics
├── scripts/
│   ├── calibrate.py                 # Refits consensus_calibration.json
│   ├── design_graph.py              # Main CLI (presets + --config)
│   ├── simulate_circuit.py          # Dense concentration sweep & MC failure modes
│   ├── make_extra_configs.py        # Reproduces extra gate/graph JSON configs
│   ├── plot_graph_mc.py             # 3-panel MC robustness plots for mode='graph'
│   ├── export_3d_jobs_graph.py      # Multi-circuit Boltz-2/ColabFold job exporter
│   ├── analyze_boltz.py             # Parses Boltz-2 outputs
│   ├── score_rosetta.py             # PyRosetta FastRelax + InterfaceAnalyzerMover
│   └── scorecard_3d_graph.py        # 3D-EQTK consistency & Complete_3D scorecard
├── results/graph_designs/           # Committed *_full.json and *.csv designs per circuit
├── designs_top3_summary.csv         # Consolidated top-3 sequences & metrics across all 25 designs
└── assets/                          # Diagnostic plots
```

## Calibration Scope & Planned 3D Checks
- **1D model domain**: calibrated on 3–4 heptad canonical parallel dimers (`BN4:BN4` Kd = 48 uM, pKd = 4.32 is the sole homodimer anchor).
- **Pre-synthesis checks**: 5-mers, homodimers with pKd > 4.32 (e.g., prefer `dome_3node D08` over `D01`), mixed E/K antiparallel flips, register shifts (`shift != 0`), and trimer traps (`A:B:Q`, `X:Q:S`, `uuu`) require Boltz-2 + PyRosetta filtering via `scorecard_3d_graph.py` prior to peptide synthesis.
