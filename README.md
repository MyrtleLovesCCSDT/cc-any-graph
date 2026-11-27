# CC-any-graph

Universal 1D + EQTK + 3D validation framework for *de novo* coiled-coil peptide design across arbitrary interaction graphs, Boolean logic gates, and analog concentration-response curves.

Extended from [coiled-coil-consensus](https://github.com/MyrtleLovesCCSDT/coiled-coil-consensus) (YES-gate pipeline) to support:
- **`mode: "graph"`**: arbitrary target interaction topologies, edge hierarchies, and off-target/homodimer suppression.
- **`mode: "truth_table"`**: all 10 non-trivial input-permutation equivalence classes of two-input Boolean logic (unified under `free_frac(S)` fluorophore-quencher readout, comparing compact `I1:I2`-sink vs Fig. S4 topologies for `NIMPLY`, `XOR`, `XNOR`), plus 3-input `OR` (`or3_5node`) and dual-output `ortho2_6node`.
- **`mode: "curve"`**: quantitative concentration-response curves (quadratic decay, prozone bandpass domes, and decoy-buffered sharp threshold switches).

---

## Summary of 1D + EQTK Results

All pairwise 1D affinities are computed with symmetric `(A,B)/(B,A)` scoring and clean buffer resets in `src/consensus.py`, and final design metrics are recomputed directly from the saved `pkd_matrix` and `c0_uM`.

Two Monte Carlo stages are reported separately:
1. **`Design MC` (`design_graph.py`, `n_mc = 200`, `sigma_pKd = 0.643`)**: post-SA ranking stage with `mc_conc_rel_std = 0.0` for all configs except `xor_4node` (`0.05`).
2. **`Sim MC` (`simulate_circuit.py`, `n_mc = 400`, `sigma_pKd = 0.643`, `conc_std = 0.015`)**: dense diagnostic sweep modeling `1.5%` nanoliter dispenser pipetting noise (matching all diagnostic plots in `assets/`).

### Logic Circuits (`mode: "truth_table"`, unified `free_frac(S)` readout)

All two-input, 3-input, and dual-output logic gates now share the exact same fluorophore-on-`S` / quencher-on-`Q` readout (`free_frac(S)`, unlabeled `I`/`M` that never directly bind `S`). For `NIMPLY`, `XOR`, and `XNOR`, both compact coiled-coil-native `I1:I2`-sink topologies (`4–5` nodes) and strict Fig. S4 topologies (`6–7` nodes, no `I1:I2` edge) are benchmarked side by side.

| Circuit | Topology / Readout | Delta | Fold | Thresh (`0.85*min_Δ`) | `MC_Delta_p10` | Design MC (`n=200`) | Sim MC (`n=400, std=0.015`) | Worst Off / Homo pKd | Min Slip Margin |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `yes_3node` | Buffer (`I-Q-S`, `free_frac(S)`) | 0.959 | 27.7x | 0.383 | 0.795 | 99.5% | 96.2% | 1.70 / 3.30 | 3.3 |
| `not_4node` | Fig. S4 NOT (`I-M1-Q1-S`, `free_frac(S)`) | 0.910 | 13.3x | 0.425 | 0.665 | 84.5% | 82.8% | 4.61 / 3.18 | 48.1 |
| `or_4node` | OR (`I1,I2-Q-S`, `free_frac(S)`) | 0.936 | 21.9x | 0.383 | 0.663 | 98.0% | 92.8% | 2.52 / 3.05 | 13.2 |
| `nor_5node` | Fig. S4 NOR (`I1,I2-M5-Q1-S`, `free_frac(S)`) | 0.942 | 33.0x | 0.425 | 0.656 | 94.0% | 91.2% | 4.10 / 4.38 | 32.3 |
| `and_5node` | Fig. S4 AND (`I1-Q4-S`, `I2-Q3-S`, `free_frac(S)`) | 0.879 | 33.4x | 0.552 | 0.418 | 58.0% | 56.5% | 4.55 / 4.32 | 27.9 |
| `nand_6node` | Fig. S4 NAND (`free_frac(S)`) | 0.730 | 7.9x | 0.425 | 0.141 | 43.5% | 37.8% | 4.45 / 3.96 | 19.2 |
| `imply_5node` | Fig. S4 `NOT I1 OR I2` (`free_frac(S)`) | 0.905 | 17.5x | 0.425 | 0.574 | 84.0% | 80.0% | 4.26 / 4.01 | 21.6 |
| `nimply_4node_sink` | Compact `I1 AND NOT I2` (`I1:I2` sink, `free_frac(S)`) | 0.925 | 22.6x | 0.468 | 0.681 | 90.5% | 84.0% | 4.45 / 3.13 | 27.9 |
| `nimply_6node` | Fig. S4 `I1 AND NOT I2` (`free_frac(S)`) | 0.768 | 7.4x | 0.425 | 0.210 | 42.0% | 35.8% | 4.63 / 4.11 | 27.9 |
| `xor_4node` | Compact XOR (`A:B >> A,B:Q > Q:S`, `free_frac(S)`) | 0.748 | 5.9x | 0.297 | 0.184 | 68.0% (`std=0.05`) | 71.0% | 2.91 / 3.97 | 19.9 |
| `xor_7node` | Fig. S4 XOR (7-node, `free_frac(S)`) | 0.572 | 4.3x | 0.425 | -0.162 | 4.0% | 2.5% | 5.03 / 4.63 | 9.8 |
| `xnor_5node` | Compact XNOR (`A:B > A,B:M > M:Q > Q:S`, `free_frac(S)`) | 0.621 | 4.1x | 0.383 | 0.130 | 30.0% | 28.7% | 4.18 / 4.56 | 27.6 |
| `xnor_7node` | Fig. S4 XNOR (7-node, `free_frac(S)`) | 0.537 | 3.4x | 0.425 | -0.159 | 2.5% | 1.2% | 5.33 / 3.96 | 17.7 |
| `or3_5node` | 3-input OR (8 states, 4-mer) | 0.964 | 39.0x | 0.510 | 0.774 | 93.5% | 87.8% | 3.44 / 3.91 | 22.8 |
| `ortho2_6node` | Dual orthogonal YES outputs | 0.919 | 14.0x | 0.383 | 0.645 | 89.5% (joint) | 93.2% / 90.5% (per-obs) | 3.28 / 3.98 | 32.7 |

*All redesigned edges have positive slippage margins (`min_slip_margin >= +9.8`, zero `shift != 0` register slips). Note that `not_3node`, `nor_4node`, `and_6node`, `xnor_4node` (dimer/bound readouts), and old `nimply_4node` (`I2:S`) were retired for experimental assay consistency.*

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

### JAX Solver Cross-Check & Robust `c0` Tuning (`check_eqconc_and_adam.py`)

Verified equilibrium equivalence between `EQTK` and the JAX fixed-point solver `eq_conc` (van Dorp et al., *Phys. Rev. E* 84, 036114, 2011; damped variant `damping=0.5` removes stoichiometric oscillations on stiff pairs, e.g. `xor_4node` max mass-balance residual `3.07e-01 -> 9.55e-03`), followed by MC-ensemble `c0` optimization via `optimize_batch_c0_ADAM` (`256` MC + nominal draws, `con_max = 20.0 uM`, fixed reporter) and independent `EQTK` validation (`n_mc = 400`, `sigma_pKd = 0.643`, `conc_std = 0.015`):

| Circuit (Design) | Nominal Delta (`Before -> After`) | Fold (`Before -> After`) | `MC_Delta_p10` (`Before -> After`) | Sim MC Pass (`Before -> After`) |
| :--- | :---: | :---: | :---: | :---: |
| `not_4node` (`D01`) | `0.910 -> 0.947` | `13.3x -> 27.2x` | `0.590 -> 0.674` | `82.8% -> 90.0%` |
| `nor_5node` (`D01`) | `0.942 -> 0.946` | `33.0x -> 40.6x` | `0.631 -> 0.660` | `91.2% -> 93.8%` |
| `and_5node` (`D01`) | `0.879 -> 0.859` | `33.4x -> 70.7x` | `0.396 -> 0.380` | `56.5% -> 63.7%` |
| `imply_5node` (`D01`) | `0.905 -> 0.937` | `17.5x -> 29.4x` | `0.474 -> 0.551` | `80.0% -> 88.2%` |
| `nand_6node` (`D01`) | `0.730 -> 0.830` | `7.9x -> 14.8x` | `0.127 -> 0.193` | `37.8% -> 59.8%` |
| `nimply_4node_sink` (`D01`) | `0.925 -> 0.948` | `22.6x -> 36.9x` | `0.585 -> 0.663` | `84.0% -> 89.2%` |
| `nimply_6node` (`D01`) | `0.768 -> 0.787` | `7.4x -> 13.6x` | `0.190 -> 0.199` | `35.8% -> 53.0%` |
| `xor_4node` (`D01`) | `0.748 -> 0.794` | `5.9x -> 7.0x` | `0.193 -> 0.210` | `71.0% -> 75.2%` |
| `xor_7node` (`D01`) | `0.572 -> 0.628` | `4.3x -> 4.9x` | `-0.086 -> -0.055` | `2.5% -> 5.8%` |
| `xnor_5node` (`D01`) | `0.621 -> 0.731` | `4.1x -> 5.7x` | `0.134 -> 0.173` | `28.7% -> 42.0%` |
| `xnor_7node` (`D01`) | `0.537 -> 0.568` | `3.4x -> 3.9x` | `-0.172 -> -0.128` | `1.2% -> 3.0%` |

### 2D Decision Boundaries (`design_2d_boundary.py`, `des_bounds` integration)

*De novo* coiled-coil sequence designs classifying `(X, Y)` concentrations by `Free A > 50 * conc_scale` nM (`121x121` exact `eq_conc` + Newton verification, SA + `L-BFGS-B` concentration fitting, and `n_mc = 200` stress test at `sigma_pKd = 0.643, conc_std = 0.015`):

| Boundary | Folder (`results/des_bounds_designs/`) | Lengths | Ideal Acc | Fitted Acc | BCE Loss | Contour Error (`% domain`) | Off / Homo pKd | Registers | MC Med Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `line` | `line_s1` (`s=1`) | `4mer` | 99.47% | **98.25%** | 0.0819 | `2.86 nM (1.43%)` | `4.93 / 4.38` | 3/3 `PAR, 0` | 82.2% |
| `circle` | `circle_s5_kmin2` (`s=5`) | `4mer` | 99.77% | **97.05%** | 0.1261 | `46.17 nM (4.62%)` | `4.82 / 4.13` | 5/5 `PAR, 0` | 77.4% |
| `quadratic` | `quadratic_s10_seed` (`s=10`) | `4/5mer` | 94.19% | **91.69%** | 0.3002 | `94.01 nM (4.70%)` | `4.73 / 4.04` | 4/5 `0` (`YC -7`) | 65.1% |
| `cubic` (bipartite `YD`) | `cubic_yd_4mer_cfit_s15_kmin0.5` (`s=15`) | `4mer` | 98.45% | **96.52%** | 0.2500 | `68.97 nM (2.30%)` | `4.82 / 4.03` | 7/7 `PAR, 0` | 70.5% |

*(A mixed `4/5mer` variant `cubic_yd_5mer_cfit_s15_kmin0.5` achieves `97.04%` Acc, `2.06%` contour error, `7/7 PAR shift=0`, `off/homo = 4.47 / 4.44`, `MC med = 70.8%`.)*

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

### 4. JAX solver cross-check, ADAM `c0` tuning, and 2D decision boundaries
```bash
python3 scripts/check_eqconc_and_adam.py --full-json results/graph_designs/xor_4node/xor_4node_full.json --tune-adam --fix-reporter
python3 scripts/design_2d_boundary.py --boundary cubic --cubic-yd --contour-fit --conc-scale 15.0 --kd-min-native 0.5 --restarts 40 --steps 3000 --top-fit 40 --seed 42
python3 scripts/validate_and_fix_2d.py --mc-and-slippage --reports results/des_bounds_designs/cubic_yd_4mer_cfit_s15_kmin0.5/cubic_report.json
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
│   ├── scorecard_3d_graph.py        # 3D-EQTK consistency & Complete_3D scorecard
│   ├── check_eqconc_and_adam.py     # EQTK vs JAX eq_conc verification & MC ADAM c0 tuning
│   ├── design_2d_boundary.py        # De novo 2D boundary design (des_bounds integration)
│   └── validate_and_fix_2d.py       # 2D boundary v3 replot, register check & MC stress test
├── results/graph_designs/           # Committed *_full.json and *.csv designs per circuit
├── results/des_bounds_designs/      # Final 2D boundary designs (line, circle, quadratic, cubic_yd)
├── designs_top3_summary.csv         # Consolidated top-3 sequences & metrics across all 25 designs
└── assets/                          # Diagnostic plots
```

## Calibration Scope & Planned 3D Checks
- **1D model domain**: calibrated on 3–4 heptad canonical parallel dimers (`BN4:BN4` Kd = 48 uM, pKd = 4.32 is the sole homodimer anchor).
- **Pre-synthesis checks**: 5-mers, homodimers with pKd > 4.32 (e.g., prefer `dome_3node D08` over `D01`), mixed E/K antiparallel flips, register shifts (`shift != 0`), and trimer traps (`A:B:Q`, `X:Q:S`, `uuu`) require Boltz-2 + PyRosetta filtering via `scorecard_3d_graph.py` prior to peptide synthesis.
