import json
from pathlib import Path

OUT_DIR = Path("configs/circuits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

dome_path = OUT_DIR / "dome_3node.json"
dome_cfg = json.loads(dome_path.read_text()) if dome_path.exists() else {}


def node(name, charge="mixed", lengths=("4mer",), allow_5mer=False):
    d = {"name": name, "charge_mode": charge, "lengths": list(lengths)}
    if allow_5mer:
        d["allow_5mer_if_enabled"] = True
    return d


def edge(u, v, pkd_min, pkd_max, shift0=True, par=True, min_slippage=0.0):
    d = {
        "u": u,
        "v": v,
        "pkd_min": pkd_min,
        "pkd_max": pkd_max,
        "require_par": par,
        "require_shift0": shift0,
    }
    if min_slippage > 0.0:
        d["min_slippage"] = min_slippage
    return d


def hier(strong, weak, min_dpkd=1.2, min_agree=3):
    return {"strong": list(strong), "weak": list(weak), "min_dpkd": min_dpkd, "min_agree": min_agree}


def free_obs(targets, min_delta=0.50, min_fold=4.5, target_node="S"):
    return [{
        "kind": "free_frac",
        "nodes": [target_node],
        "targets": targets,
        "min_delta": min_delta,
        "min_fold": min_fold,
    }]


S_LEN = ("4mer", "35mer", "3mer")
configs = []

# --- 1. NOT (4-node Figure S4 topology, output = free_frac(S)) ---
configs.append({
    "name": "not_4node",
    "mode": "truth_table",
    "nodes": [
        node("I", allow_5mer=True),
        node("M1", allow_5mer=True),
        node("Q1"),
        node("S", lengths=S_LEN),
    ],
    "edges": [
        edge("I", "M1", 8.4, 12.0, True, min_slippage=5.0),
        edge("M1", "Q1", 7.0, 8.6, True, min_slippage=5.0),
        edge("Q1", "S", 5.8, 7.0, False),
    ],
    "hierarchies": [
        hier(["I", "M1"], ["M1", "Q1"], 1.4, 3),
        hier(["M1", "Q1"], ["Q1", "S"], 1.3, 3),
    ],
    "off_target_max_pkd": 4.5,
    "homodimer_max_pkd": 4.3,
    "states_uM": [
        {"I": 0.0, "M1": 3.5, "Q1": 2.0, "S": 0.5},
        {"I": 12.0, "M1": 3.5, "Q1": 2.0, "S": 0.5},
    ],
    "observables": free_obs([1, 0], min_delta=0.50, min_fold=4.5),
    "tunable_scaffolds": [
        {"node": "Q1", "min_uM": 1.0, "max_uM": 4.5},
        {"node": "M1", "min_uM": 1.5, "max_uM": 7.5},
    ],
})

# --- 2. AND (5-node Figure S4 topology, output = free_frac(S)) ---
configs.append({
    "name": "and_5node",
    "mode": "truth_table",
    "nodes": [
        node("I1", allow_5mer=True),
        node("I2", allow_5mer=True),
        node("Q3"),
        node("Q4"),
        node("S", lengths=S_LEN),
    ],
    "edges": [
        edge("I1", "Q4", 7.8, 11.5, True, min_slippage=5.0),
        edge("I2", "Q3", 7.8, 11.5, True, min_slippage=5.0),
        edge("Q4", "S", 6.0, 7.4, True),
        edge("Q3", "S", 6.0, 7.4, True),
    ],
    "hierarchies": [
        hier(["I1", "Q4"], ["Q4", "S"], 1.4, 3),
        hier(["I2", "Q3"], ["Q3", "S"], 1.4, 3),
        hier(["Q4", "S"], ["I1", "Q3"], 1.2, 3),
        hier(["Q3", "S"], ["I2", "Q4"], 1.2, 3),
    ],
    "off_target_max_pkd": 4.5,
    "homodimer_max_pkd": 4.3,
    "states_uM": [
        {"I1": 0.0,  "I2": 0.0,  "Q3": 3.0, "Q4": 3.0, "S": 0.5},
        {"I1": 10.0, "I2": 0.0,  "Q3": 3.0, "Q4": 3.0, "S": 0.5},
        {"I1": 0.0,  "I2": 10.0, "Q3": 3.0, "Q4": 3.0, "S": 0.5},
        {"I1": 10.0, "I2": 10.0, "Q3": 3.0, "Q4": 3.0, "S": 0.5},
    ],
    "observables": free_obs([0, 0, 0, 1], min_delta=0.65, min_fold=8.0),
    "tunable_scaffolds": [
        {"node": "Q3", "min_uM": 1.5, "max_uM": 6.0},
        {"node": "Q4", "min_uM": 1.5, "max_uM": 6.0},
    ],
})

# --- 3. NOR (5-node Figure S4 topology, output = free_frac(S)) ---
configs.append({
    "name": "nor_5node",
    "mode": "truth_table",
    "nodes": [
        node("I1", allow_5mer=True),
        node("I2", allow_5mer=True),
        node("M5", allow_5mer=True),
        node("Q1"),
        node("S", lengths=S_LEN),
    ],
    "edges": [
        edge("I1", "M5", 8.6, 12.0, True, min_slippage=5.0),
        edge("I2", "M5", 8.6, 12.0, True, min_slippage=5.0),
        edge("M5", "Q1", 7.3, 8.8, True, min_slippage=5.0),
        edge("Q1", "S", 5.9, 7.2, False),
    ],
    "hierarchies": [
        hier(["I1", "M5"], ["M5", "Q1"], 1.2, 3),
        hier(["I2", "M5"], ["M5", "Q1"], 1.2, 3),
        hier(["M5", "Q1"], ["Q1", "S"], 1.2, 3),
    ],
    "off_target_max_pkd": 4.5,
    "homodimer_max_pkd": 4.3,
    "states_uM": [
        {"I1": 0.0,  "I2": 0.0,  "M5": 4.5, "Q1": 2.0, "S": 0.4},
        {"I1": 12.0, "I2": 0.0,  "M5": 4.5, "Q1": 2.0, "S": 0.4},
        {"I1": 0.0,  "I2": 12.0, "M5": 4.5, "Q1": 2.0, "S": 0.4},
        {"I1": 12.0, "I2": 12.0, "M5": 4.5, "Q1": 2.0, "S": 0.4},
    ],
    "observables": free_obs([1, 0, 0, 0], min_delta=0.50, min_fold=4.5),
    "tunable_scaffolds": [
        {"node": "Q1", "min_uM": 1.0, "max_uM": 4.0},
        {"node": "M5", "min_uM": 2.5, "max_uM": 7.5},
    ],
})

# --- 4. NAND (6-node Figure S4 topology, output = free_frac(S), slippage-safe) ---
configs.append({
    "name": "nand_6node",
    "mode": "truth_table",
    "nodes": [
        node("I1", allow_5mer=True),
        node("I2", allow_5mer=True),
        node("M1", allow_5mer=True),
        node("M2", allow_5mer=True),
        node("Q1"),
        node("S", lengths=S_LEN),
    ],
    "edges": [
        edge("I1", "M1", 8.6, 12.0, True, min_slippage=5.0),
        edge("I2", "M2", 8.6, 12.0, True, min_slippage=5.0),
        edge("M1", "Q1", 7.2, 8.6, True, min_slippage=5.0),
        edge("M2", "Q1", 7.2, 8.6, True, min_slippage=5.0),
        edge("Q1", "S", 5.9, 7.1, False),
    ],
    "hierarchies": [
        hier(["I1", "M1"], ["M1", "Q1"], 1.4, 3),
        hier(["I2", "M2"], ["M2", "Q1"], 1.4, 3),
        hier(["M1", "Q1"], ["Q1", "S"], 1.3, 3),
        hier(["M2", "Q1"], ["Q1", "S"], 1.3, 3),
    ],
    "off_target_max_pkd": 4.5,
    "homodimer_max_pkd": 4.3,
    "states_uM": [
        {"I1": 0.0,  "I2": 0.0,  "M1": 4.5, "M2": 4.5, "Q1": 2.5, "S": 0.4},
        {"I1": 12.5, "I2": 0.0,  "M1": 4.5, "M2": 4.5, "Q1": 2.5, "S": 0.4},
        {"I1": 0.0,  "I2": 12.5, "M1": 4.5, "M2": 4.5, "Q1": 2.5, "S": 0.4},
        {"I1": 12.5, "I2": 12.5, "M1": 4.5, "M2": 4.5, "Q1": 2.5, "S": 0.4},
    ],
    "observables": free_obs([1, 1, 1, 0], min_delta=0.50, min_fold=4.5),
    "tunable_scaffolds": [
        {"node": "Q1", "min_uM": 1.0, "max_uM": 5.0},
        {"node": "M1", "min_uM": 2.0, "max_uM": 8.0},
        {"node": "M2", "min_uM": 2.0, "max_uM": 8.0},
    ],
})

# --- 5. IMPLY: NOT I1 OR I2 (5-node Figure S4 topology, output = free_frac(S)) ---
configs.append({
    "name": "imply_5node",
    "mode": "truth_table",
    "nodes": [
        node("I1", allow_5mer=True),
        node("I2"),
        node("M3", allow_5mer=True),
        node("Q3"),
        node("S", lengths=S_LEN),
    ],
    "edges": [
        edge("I1", "M3", 8.6, 12.0, True, min_slippage=5.0),
        edge("M3", "Q3", 7.3, 8.8, True, min_slippage=5.0),
        edge("I2", "Q3", 7.6, 11.5, True, min_slippage=5.0),
        edge("Q3", "S", 5.9, 7.2, False),
    ],
    "hierarchies": [
        hier(["I1", "M3"], ["M3", "Q3"], 1.3, 3),
        hier(["M3", "Q3"], ["Q3", "S"], 1.3, 3),
        hier(["I2", "Q3"], ["Q3", "S"], 1.4, 3),
    ],
    "off_target_max_pkd": 4.5,
    "homodimer_max_pkd": 4.3,
    "states_uM": [
        {"I1": 0.0,  "I2": 0.0, "M3": 4.5, "Q3": 2.0, "S": 0.4},
        {"I1": 12.0, "I2": 0.0, "M3": 4.5, "Q3": 2.0, "S": 0.4},
        {"I1": 0.0,  "I2": 8.0, "M3": 4.5, "Q3": 2.0, "S": 0.4},
        {"I1": 12.0, "I2": 8.0, "M3": 4.5, "Q3": 2.0, "S": 0.4},
    ],
    "observables": free_obs([1, 0, 1, 1], min_delta=0.50, min_fold=4.5),
    "tunable_scaffolds": [
        {"node": "Q3", "min_uM": 1.0, "max_uM": 4.0},
        {"node": "M3", "min_uM": 2.5, "max_uM": 7.5},
    ],
})

# --- 6A. NIMPLY compact CC-native (4-node, physically valid: I1:I2 sink >> I1:Q > Q:S, NO I2:S!) ---
configs.append({
    "name": "nimply_4node_sink",
    "mode": "truth_table",
    "nodes": [
        node("I1", allow_5mer=True),
        node("I2", allow_5mer=True),
        node("Q"),
        node("S", lengths=S_LEN),
    ],
    "edges": [
        edge("I1", "I2", 9.0, 12.5, True, min_slippage=5.0),
        edge("I1", "Q", 7.3, 8.6, True, min_slippage=5.0),
        edge("Q", "S", 5.9, 7.0, False),
    ],
    "hierarchies": [
        hier(["I1", "I2"], ["I1", "Q"], 1.5, 3),
        hier(["I1", "Q"], ["Q", "S"], 1.4, 3),
    ],
    "off_target_max_pkd": 4.5,
    "homodimer_max_pkd": 4.3,
    "states_uM": [
        {"I1": 0.0,  "I2": 0.0,  "Q": 2.5, "S": 0.4},
        {"I1": 10.0, "I2": 0.0,  "Q": 2.5, "S": 0.4},
        {"I1": 0.0,  "I2": 10.0, "Q": 2.5, "S": 0.4},
        {"I1": 10.0, "I2": 10.0, "Q": 2.5, "S": 0.4},
    ],
    "observables": free_obs([0, 1, 0, 0], min_delta=0.55, min_fold=5.0),
    "tunable_scaffolds": [
        {"node": "Q", "min_uM": 1.0, "max_uM": 5.0},
    ],
})

# --- 6B. NIMPLY: I1 AND NOT I2 (6-node Figure S4 topology, I1/I2 isolated from S and each other) ---
configs.append({
    "name": "nimply_6node",
    "mode": "truth_table",
    "nodes": [
        node("I1", allow_5mer=True),
        node("I2", allow_5mer=True),
        node("M2", allow_5mer=True),
        node("Q4"),
        node("Q1"),
        node("S", lengths=S_LEN),
    ],
    "edges": [
        edge("I1", "Q4", 7.8, 11.5, True, min_slippage=5.0),
        edge("Q4", "S", 6.0, 7.2, True),
        edge("I2", "M2", 8.6, 12.0, True, min_slippage=5.0),
        edge("M2", "Q1", 7.3, 8.8, True, min_slippage=5.0),
        edge("Q1", "S", 6.0, 7.2, False),
    ],
    "hierarchies": [
        hier(["I1", "Q4"], ["Q4", "S"], 1.4, 3),
        hier(["I2", "M2"], ["M2", "Q1"], 1.3, 3),
        hier(["M2", "Q1"], ["Q1", "S"], 1.3, 3),
    ],
    "off_target_max_pkd": 4.5,
    "homodimer_max_pkd": 4.3,
    "states_uM": [
        {"I1": 0.0,  "I2": 0.0,  "M2": 4.5, "Q4": 2.5, "Q1": 2.0, "S": 0.4},
        {"I1": 10.0, "I2": 0.0,  "M2": 4.5, "Q4": 2.5, "Q1": 2.0, "S": 0.4},
        {"I1": 0.0,  "I2": 12.0, "M2": 4.5, "Q4": 2.5, "Q1": 2.0, "S": 0.4},
        {"I1": 10.0, "I2": 12.0, "M2": 4.5, "Q4": 2.5, "Q1": 2.0, "S": 0.4},
    ],
    "observables": free_obs([0, 1, 0, 0], min_delta=0.50, min_fold=4.5),
    "tunable_scaffolds": [
        {"node": "Q4", "min_uM": 1.2, "max_uM": 4.5},
        {"node": "Q1", "min_uM": 1.0, "max_uM": 4.0},
        {"node": "M2", "min_uM": 2.5, "max_uM": 7.5},
    ],
})

# --- 7B. XOR (7-node Figure S4 topology, no I1:I2 edge, output = free_frac(S)) ---
configs.append({
    "name": "xor_7node",
    "mode": "truth_table",
    "nodes": [
        node("I1", allow_5mer=True),
        node("I2", allow_5mer=True),
        node("M1", allow_5mer=True),
        node("M2", allow_5mer=True),
        node("Q1"),
        node("Q2"),
        node("S", lengths=S_LEN),
    ],
    "edges": [
        edge("I1", "M1", 8.5, 12.0, True, min_slippage=5.0),
        edge("I2", "M2", 8.5, 12.0, True, min_slippage=5.0),
        edge("M1", "Q1", 7.1, 8.5, True, min_slippage=5.0),
        edge("M2", "Q1", 7.1, 8.5, True, min_slippage=5.0),
        edge("I1", "Q2", 7.1, 8.5, True, min_slippage=5.0),
        edge("I2", "Q2", 7.1, 8.5, True, min_slippage=5.0),
        edge("Q1", "S", 5.8, 7.0, False),
        edge("Q2", "S", 5.8, 7.0, False),
    ],
    "hierarchies": [
        hier(["I1", "M1"], ["M1", "Q1"], 1.3, 3),
        hier(["I2", "M2"], ["M2", "Q1"], 1.3, 3),
        hier(["M1", "Q1"], ["Q1", "S"], 1.2, 3),
        hier(["M2", "Q1"], ["Q1", "S"], 1.2, 3),
        hier(["I1", "Q2"], ["Q2", "S"], 1.3, 3),
        hier(["I2", "Q2"], ["Q2", "S"], 1.3, 3),
    ],
    "off_target_max_pkd": 4.7,
    "homodimer_max_pkd": 4.5,
    "states_uM": [
        {"I1": 0.0,  "I2": 0.0,  "M1": 3.5, "M2": 3.5, "Q1": 2.0, "Q2": 2.0, "S": 0.4},
        {"I1": 12.0, "I2": 0.0,  "M1": 3.5, "M2": 3.5, "Q1": 2.0, "Q2": 2.0, "S": 0.4},
        {"I1": 0.0,  "I2": 12.0, "M1": 3.5, "M2": 3.5, "Q1": 2.0, "Q2": 2.0, "S": 0.4},
        {"I1": 12.0, "I2": 12.0, "M1": 3.5, "M2": 3.5, "Q1": 2.0, "Q2": 2.0, "S": 0.4},
    ],
    "observables": free_obs([0, 1, 1, 0], min_delta=0.50, min_fold=4.5),
    "tunable_scaffolds": [
        {"node": "Q1", "min_uM": 1.0, "max_uM": 4.5},
        {"node": "Q2", "min_uM": 1.0, "max_uM": 4.5},
        {"node": "M1", "min_uM": 2.0, "max_uM": 7.0},
        {"node": "M2", "min_uM": 2.0, "max_uM": 7.0},
    ],
})

# --- 8A. XNOR compact CC-native (5-node free_frac(S) via A:B sink + mediator M, slippage-safe) ---
configs.append({
    "name": "xnor_5node",
    "mode": "truth_table",
    "nodes": [
        node("A", allow_5mer=True),
        node("B", allow_5mer=True),
        node("M", allow_5mer=True),
        node("Q"),
        node("S", lengths=S_LEN),
    ],
    "edges": [
        edge("A", "B", 9.0, 12.5, True, min_slippage=5.0),
        edge("A", "M", 7.6, 8.8, True, min_slippage=5.0),
        edge("B", "M", 7.6, 8.8, True, min_slippage=5.0),
        edge("M", "Q", 6.4, 7.5, True, min_slippage=5.0),
        edge("Q", "S", 5.4, 6.3, False),
    ],
    "hierarchies": [
        hier(["A", "B"], ["A", "M"], 1.3, 3),
        hier(["A", "B"], ["B", "M"], 1.3, 3),
        hier(["A", "M"], ["M", "Q"], 1.1, 3),
        hier(["B", "M"], ["M", "Q"], 1.1, 3),
        hier(["M", "Q"], ["Q", "S"], 1.0, 3),
    ],
    "off_target_max_pkd": 4.5,
    "homodimer_max_pkd": 4.3,
    "states_uM": [
        {"A": 0.0,  "B": 0.0,  "M": 3.5, "Q": 1.5, "S": 0.3},
        {"A": 10.0, "B": 0.0,  "M": 3.5, "Q": 1.5, "S": 0.3},
        {"A": 0.0,  "B": 10.0, "M": 3.5, "Q": 1.5, "S": 0.3},
        {"A": 10.0, "B": 10.0, "M": 3.5, "Q": 1.5, "S": 0.3},
    ],
    "observables": free_obs([1, 0, 0, 1], min_delta=0.45, min_fold=3.5),
    "tunable_scaffolds": [
        {"node": "Q", "min_uM": 0.8, "max_uM": 3.5},
        {"node": "M", "min_uM": 1.5, "max_uM": 6.0},
    ],
})

# --- 8B. XNOR (7-node Figure S4 topology, no I1:I2 edge, output = free_frac(S)) ---
configs.append({
    "name": "xnor_7node",
    "mode": "truth_table",
    "nodes": [
        node("I1", allow_5mer=True),
        node("I2", allow_5mer=True),
        node("M3", allow_5mer=True),
        node("M4", allow_5mer=True),
        node("Q3"),
        node("Q4"),
        node("S", lengths=S_LEN),
    ],
    "edges": [
        edge("I1", "M3", 8.5, 12.0, True, min_slippage=5.0),
        edge("I2", "M4", 8.5, 12.0, True, min_slippage=5.0),
        edge("M3", "Q3", 7.1, 8.5, True, min_slippage=5.0),
        edge("M4", "Q4", 7.1, 8.5, True, min_slippage=5.0),
        edge("I2", "Q3", 7.1, 8.5, True, min_slippage=5.0),
        edge("I1", "Q4", 7.1, 8.5, True, min_slippage=5.0),
        edge("Q3", "S", 5.8, 7.0, False),
        edge("Q4", "S", 5.8, 7.0, False),
    ],
    "hierarchies": [
        hier(["I1", "M3"], ["M3", "Q3"], 1.3, 3),
        hier(["I2", "M4"], ["M4", "Q4"], 1.3, 3),
        hier(["M3", "Q3"], ["Q3", "S"], 1.2, 3),
        hier(["M4", "Q4"], ["Q4", "S"], 1.2, 3),
        hier(["I2", "Q3"], ["Q3", "S"], 1.3, 3),
        hier(["I1", "Q4"], ["Q4", "S"], 1.3, 3),
    ],
    "off_target_max_pkd": 4.7,
    "homodimer_max_pkd": 4.5,
    "states_uM": [
        {"I1": 0.0,  "I2": 0.0,  "M3": 3.5, "M4": 3.5, "Q3": 2.0, "Q4": 2.0, "S": 0.4},
        {"I1": 12.0, "I2": 0.0,  "M3": 3.5, "M4": 3.5, "Q3": 2.0, "Q4": 2.0, "S": 0.4},
        {"I1": 0.0,  "I2": 12.0, "M3": 3.5, "M4": 3.5, "Q3": 2.0, "Q4": 2.0, "S": 0.4},
        {"I1": 12.0, "I2": 12.0, "M3": 3.5, "M4": 3.5, "Q3": 2.0, "Q4": 2.0, "S": 0.4},
    ],
    "observables": free_obs([1, 0, 0, 1], min_delta=0.50, min_fold=4.5),
    "tunable_scaffolds": [
        {"node": "Q3", "min_uM": 1.0, "max_uM": 4.5},
        {"node": "Q4", "min_uM": 1.0, "max_uM": 4.5},
        {"node": "M3", "min_uM": 2.0, "max_uM": 7.0},
        {"node": "M4", "min_uM": 2.0, "max_uM": 7.0},
    ],
})

# --- 9. 3-input OR (5-node) ---
configs.append({
    "name": "or3_5node",
    "mode": "truth_table",
    "nodes": [
        node("I1"), node("I2"), node("I3"),
        node("Q"), node("S", lengths=S_LEN),
    ],
    "edges": [
        edge("I1", "Q", 7.5, 11.5, True),
        edge("I2", "Q", 7.5, 11.5, True),
        edge("I3", "Q", 7.5, 11.5, True),
        edge("Q", "S", 5.8, 7.0, True),
    ],
    "hierarchies": [
        hier(["I1", "Q"], ["Q", "S"], 1.4, 3),
        hier(["I2", "Q"], ["Q", "S"], 1.4, 3),
        hier(["I3", "Q"], ["Q", "S"], 1.4, 3),
    ],
    "off_target_max_pkd": 4.5,
    "homodimer_max_pkd": 4.3,
    "states_uM": [
        {"I1": 8.0 * a, "I2": 8.0 * b, "I3": 8.0 * c, "Q": 2.5, "S": 0.5}
        for a in (0, 1) for b in (0, 1) for c in (0, 1)
    ],
    "observables": free_obs([0, 1, 1, 1, 1, 1, 1, 1], min_delta=0.60, min_fold=6.0),
    "tunable_scaffolds": [{"node": "Q", "min_uM": 1.2, "max_uM": 4.5}],
})

# --- 10. Pure arbitrary graphs (mode = "graph") ---
configs.append({
    "name": "cycle_c4",
    "mode": "graph",
    "nodes": [node("A"), node("B"), node("C"), node("D")],
    "edges": [
        edge("A", "B", 7.0, 10.5, True),
        edge("B", "C", 7.0, 10.5, True),
        edge("C", "D", 7.0, 10.5, True),
        edge("D", "A", 7.0, 10.5, True),
    ],
    "hierarchies": [],
    "off_target_max_pkd": 4.3,
    "homodimer_max_pkd": 4.3,
})

configs.append({
    "name": "bipartite_k23",
    "mode": "graph",
    "nodes": [node("A1"), node("A2"), node("B1"), node("B2"), node("B3")],
    "edges": [
        edge(a, b, 6.8, 10.5, True)
        for a in ("A1", "A2") for b in ("B1", "B2", "B3")
    ],
    "hierarchies": [],
    "off_target_max_pkd": 4.3,
    "homodimer_max_pkd": 4.3,
})

configs.append({
    "name": "barbell_6node",
    "mode": "graph",
    "nodes": [node("A1"), node("B1"), node("C1"), node("A2"), node("B2"), node("C2")],
    "edges": [
        edge("A1", "B1", 6.8, 10.5, True),
        edge("B1", "C1", 6.8, 10.5, True),
        edge("A1", "C1", 6.8, 10.5, True),
        edge("C1", "C2", 6.8, 10.5, True),
        edge("A2", "B2", 6.8, 10.5, True),
        edge("B2", "C2", 6.8, 10.5, True),
        edge("A2", "C2", 6.8, 10.5, True),
    ],
    "hierarchies": [],
    "off_target_max_pkd": 4.4,
    "homodimer_max_pkd": 4.3,
})

configs.append({
    "name": "cascade_5node",
    "mode": "graph",
    "nodes": [node("A", allow_5mer=True), node("B", allow_5mer=True), node("C"), node("D"), node("E")],
    "edges": [
        edge("A", "B", 9.2, 12.2, True, min_slippage=5.0),
        edge("B", "C", 7.8, 9.0, True, min_slippage=5.0),
        edge("C", "D", 6.6, 7.6, True),
        edge("D", "E", 5.5, 6.4, False),
    ],
    "hierarchies": [
        hier(["A", "B"], ["B", "C"], 0.9, 3),
        hier(["B", "C"], ["C", "D"], 0.9, 3),
        hier(["C", "D"], ["D", "E"], 0.9, 3),
    ],
    "off_target_max_pkd": 4.4,
    "homodimer_max_pkd": 4.3,
})

# --- 11. Delayed sharp threshold switch via decoy D (mode = "curve") ---
if "states_uM" in dome_cfg:
    xs = [0.1 * (150.0 ** (i / 16.0)) for i in range(17)]
    targets = [round(0.06 + 0.92 / (1.0 + (6.0 / x) ** 6), 4) for x in xs]
    configs.append({
        "name": "threshold_decoy_4node",
        "mode": "curve",
        "log_x": True,
        "nodes": [
            node("D", allow_5mer=True),
            node("I", allow_5mer=True),
            node("Q"),
            node("S", lengths=S_LEN),
        ],
        "edges": [
            edge("D", "I", 8.8, 12.2, True, min_slippage=5.0),
            edge("I", "Q", 7.3, 8.8, True, min_slippage=5.0),
            edge("Q", "S", 5.8, 7.0, False),
        ],
        "hierarchies": [
            hier(["D", "I"], ["I", "Q"], 1.1, 3),
            hier(["I", "Q"], ["Q", "S"], 1.1, 3),
        ],
        "off_target_max_pkd": 4.5,
        "homodimer_max_pkd": 4.3,
        "states_uM": [{"D": 3.0, "I": round(x, 4), "Q": 3.0, "S": 0.4} for x in xs],
        "observables": [{
            "kind": "free_frac",
            "nodes": ["S"],
            "targets": targets,
            "affine_fit": False,
            "min_span": 0.65,
            "max_rmse": 0.12,
        }],
        "tunable_scaffolds": [
            {"node": "D", "min_uM": 1.5, "max_uM": 5.0},
            {"node": "Q", "min_uM": 1.5, "max_uM": 5.0},
        ],
    })

for cfg in configs:
    p = OUT_DIR / f"{cfg['name']}.json"
    p.write_text(json.dumps(cfg, indent=2) + "\n")
    print("Wrote", p)
