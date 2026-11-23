"""Structured sequence space, pool generation, and heptad-aware mutators."""

from dataclasses import dataclass
import itertools
from typing import Dict, List, Sequence, Tuple
import numpy as np

from .consensus import ConsensusModel

# Internal Asn positions prioritized so Boltz-2 / Rosetta avoid terminal fraying
CORES_BY_LEN: Dict[str, List[Tuple[str, ...]]] = {
    "3mer": [
        ("I", "N", "I"),
        ("V", "N", "V"),
        ("I", "N", "V"),
        ("V", "N", "I"),
    ],
    "35mer": [
        ("I", "N", "I"),
        ("V", "N", "V"),
        ("I", "N", "V"),
        ("V", "N", "I"),
    ],
    "4mer": [
        ("I", "I", "N", "I"),
        ("I", "N", "I", "I"),
        ("V", "V", "N", "V"),
        ("I", "V", "N", "V"),
        ("V", "N", "V", "V"),
        ("V", "I", "N", "I"),
        ("I", "N", "N", "I"),
        ("V", "N", "N", "V"),
    ],
    "5mer": [
        ("I", "I", "N", "I", "I"),
        ("I", "N", "I", "I", "I"),
        ("I", "I", "I", "N", "I"),
        ("I", "N", "I", "N", "I"),
        ("V", "I", "N", "I", "V"),
        ("V", "N", "I", "N", "V"),
    ],
}

BC_3 = [("S", "A"), ("A", "S"), ("S", "A")]
BC_4 = [("A", "S"), ("S", "A"), ("A", "S"), ("S", "A")]
BC_5 = [("A", "S"), ("S", "A"), ("A", "S"), ("S", "A"), ("A", "S")]

EG_ALPHABET: Dict[str, Tuple[str, ...]] = {
    "positive": ("K", "Q"),
    "negative": ("E", "Q", "A"),
    "mixed": ("E", "K", "Q", "A"),
    "any": ("E", "K", "Q", "A"),
}

MIXED_MOTIFS = [
    ("K", "K"),
    ("E", "E"),
    ("K", "Q"),
    ("E", "Q"),
    ("Q", "K"),
    ("Q", "E"),
    ("Q", "Q"),
    ("K", "E"),
    ("E", "K"),
]


def net_charge(seq: str) -> int:
    s = seq.strip().lstrip("-")
    return s.count("K") + s.count("R") - s.count("E") - s.count("D")


def calc_eps280(seq: str) -> int:
    s = seq.strip().lstrip("-")
    return 5500 * s.count("W") + 1490 * s.count("Y")


@dataclass(frozen=True)
class PeptideSpec:
    length_mode: str
    a_core: Tuple[str, ...]
    g_tuple: Tuple[str, ...]
    e_tuple: Tuple[str, ...]
    f_tuple: Tuple[str, ...]
    lead_e: str = "E"
    lead_f: str = "K"


def _w_index(length_mode: str) -> int:
    return 1 if length_mode in ("3mer", "35mer") else 2


def auto_f_tuple(
    g_tuple: Tuple[str, ...],
    e_tuple: Tuple[str, ...],
    length_mode: str,
    charge_mode: str,
) -> Tuple[Tuple[str, ...], str, str]:
    """Choose soluble f-positions (always preserving one W for UV280) and 35mer lead."""
    n_h = len(g_tuple)
    w_idx = _w_index(length_mode)

    if charge_mode in ("positive", "negative"):
        dom_aa = "K" if charge_mode == "positive" else "E"
        opp = "E" if charge_mode == "positive" else "K"
        eg_dom = sum(1 for x in (g_tuple + e_tuple) if x == dom_aa)

        if length_mode in ("3mer", "35mer"):
            f_list = ["S", "W", opp]
            lead_e = "Q" if charge_mode == "positive" else "E"
            lead_f = opp
            return tuple(f_list), lead_e, lead_f

        if n_h == 4:
            f_list = [opp, opp, "W", opp] if eg_dom >= 6 else [opp, "S", "W", opp]
            return tuple(f_list), "E", "K"

        # 5mer
        f_list = [opp, opp, "W", opp, opp] if eg_dom >= 7 else [opp, "S", "W", "S", opp]
        return tuple(f_list), "E", "K"

    # Mixed / any: keep W at w_idx, adjust remaining f positions to keep 2 <= |net_charge| <= 6
    lead_e = "Q" if length_mode == "35mer" else "E"
    lead_f = "S"
    f_list = ["S"] * n_h
    f_list[w_idx] = "W"

    base_chg = sum(1 if x == "K" else (-1 if x == "E" else 0) for x in (g_tuple + e_tuple))
    chg = base_chg
    adj_order = [i for i in range(n_h) if i != w_idx]

    for idx in adj_order:
        if -6 <= chg <= 6 and abs(chg) >= 2:
            break
        if abs(chg) < 2:
            if chg >= 0:
                f_list[idx] = "K"
                chg += 1
            else:
                f_list[idx] = "E"
                chg -= 1
        elif chg > 6:
            f_list[idx] = "E"
            chg -= 1
        elif chg < -6:
            f_list[idx] = "K"
            chg += 1

    return tuple(f_list), lead_e, lead_f


def build_sequence(spec: PeptideSpec) -> Tuple[str, str]:
    """Convert PeptideSpec to (register_padded_seq, raw_seq)."""
    lm = spec.length_mode
    if lm in ("3mer", "35mer"):
        body = "".join(
            f"{spec.g_tuple[h]}{spec.a_core[h]}{BC_3[h][0]}{BC_3[h][1]}L{spec.e_tuple[h]}{spec.f_tuple[h]}"
            for h in range(3)
        )
        if lm == "35mer":
            raw = f"L{spec.lead_e}{spec.lead_f}" + body
            return "-----" + raw, raw
        return "-" + body, body

    bc = BC_4 if lm == "4mer" else BC_5
    n_h = len(spec.a_core)
    raw = "".join(
        f"{spec.g_tuple[h]}{spec.a_core[h]}{bc[h][0]}{bc[h][1]}L{spec.e_tuple[h]}{spec.f_tuple[h]}"
        for h in range(n_h)
    )
    return "-" + raw, raw


def valid_charge(raw: str, charge_mode: str, length_mode: str) -> bool:
    chg = net_charge(raw)
    max_abs = 7 if length_mode == "5mer" else 6
    if charge_mode == "positive":
        return 2 <= chg <= max_abs
    if charge_mode == "negative":
        return -max_abs <= chg <= -2
    return 2 <= abs(chg) <= max_abs


def _make_spec_checked(
    length_mode: str,
    a_core: Tuple[str, ...],
    g_tuple: Tuple[str, ...],
    e_tuple: Tuple[str, ...],
    charge_mode: str,
) -> PeptideSpec:
    f_tup, lead_e, lead_f = auto_f_tuple(g_tuple, e_tuple, length_mode, charge_mode)
    return PeptideSpec(
        length_mode=length_mode,
        a_core=a_core,
        g_tuple=g_tuple,
        e_tuple=e_tuple,
        f_tuple=f_tup,
        lead_e=lead_e,
        lead_f=lead_f,
    )


def generate_node_pool(
    model: ConsensusModel,
    charge_mode: str,
    lengths: Sequence[str],
    homo_min_nM: float = 20_000.0,
    max_mixed_per_len: int = 450,
    seed: int = 42,
) -> List[PeptideSpec]:
    """Generate a diverse, soluble, homodimer-screened PeptideSpec pool."""
    rng = np.random.default_rng(seed)
    pool: List[PeptideSpec] = []
    seen_raw = set()

    def _try_add(spec: PeptideSpec) -> bool:
        pad, raw = build_sequence(spec)
        if raw in seen_raw or not valid_charge(raw, charge_mode, spec.length_mode):
            return False
        sc_homo = model.predict_pair(pad, pad)
        if sc_homo["Kd_nM"] < homo_min_nM:
            return False
        seen_raw.add(raw)
        pool.append(spec)
        return True

    for lm in lengths:
        cores = CORES_BY_LEN[lm]
        n_h = 3 if lm in ("3mer", "35mer") else (4 if lm == "4mer" else 5)

        if charge_mode == "positive":
            if n_h <= 4:
                for a_tup in cores:
                    for comb in itertools.product(["K", "Q"], repeat=2 * n_h):
                        if comb.count("K") < (n_h + 1):
                            continue
                        _try_add(_make_spec_checked(lm, a_tup, comb[:n_h], comb[n_h:], charge_mode))
            else:
                # 5mer: structured motif + random sampling
                cands = []
                for a_tup in cores:
                    for _ in range(180):
                        comb = tuple(rng.choice(["K", "Q"], size=2 * n_h, p=[0.75, 0.25]))
                        if comb.count("K") >= 6:
                            cands.append(_make_spec_checked(lm, a_tup, comb[:n_h], comb[n_h:], charge_mode))
                for sp in cands:
                    _try_add(sp)

        elif charge_mode == "negative":
            if n_h == 3:
                for a_tup in cores:
                    for comb in itertools.product(["E", "Q"], repeat=6):
                        if comb.count("E") >= 4:
                            _try_add(_make_spec_checked(lm, a_tup, comb[:3], comb[3:], charge_mode))
            elif n_h == 4:
                for a_tup in cores[:6]:
                    for comb in itertools.product(["E", "Q", "A"], repeat=8):
                        n_e, n_a = comb.count("E"), comb.count("A")
                        if n_e < 4 or n_a > 2 or ("V" in a_tup and n_e < 5):
                            continue
                        _try_add(_make_spec_checked(lm, a_tup, comb[:4], comb[4:], charge_mode))
            else:
                for a_tup in cores:
                    for _ in range(220):
                        comb = tuple(rng.choice(["E", "Q", "A"], size=10, p=[0.65, 0.25, 0.10]))
                        if comb.count("E") >= 5 and comb.count("A") <= 2:
                            _try_add(_make_spec_checked(lm, a_tup, comb[:5], comb[5:], charge_mode))

        else:
            # Mixed / any: electrostatic barcodes with both E and K patterns + pure edges
            added = 0
            # First add structured heptad-block barcodes (very clean orthogonal patterns)
            block_motifs = [("K", "K"), ("E", "E"), ("K", "Q"), ("E", "Q"), ("Q", "K"), ("Q", "E"), ("Q", "Q")]
            cands: List[PeptideSpec] = []
            for a_tup in cores:
                if n_h == 4:
                    # Sample diverse 4-heptad combinations + pure +/- anchors
                    for _ in range(120):
                        idxs = rng.integers(0, len(block_motifs), size=4)
                        g_t = tuple(block_motifs[i][0] for i in idxs)
                        e_t = tuple(block_motifs[i][1] for i in idxs)
                        if sum(x in ("E", "K") for x in g_t + e_t) >= 4:
                            cands.append(_make_spec_checked(lm, a_tup, g_t, e_t, charge_mode))
                else:
                    for _ in range(160):
                        idxs = rng.integers(0, len(MIXED_MOTIFS), size=n_h)
                        g_t = tuple(MIXED_MOTIFS[i][0] for i in idxs)
                        e_t = tuple(MIXED_MOTIFS[i][1] for i in idxs)
                        if sum(x in ("E", "K") for x in g_t + e_t) >= n_h:
                            cands.append(_make_spec_checked(lm, a_tup, g_t, e_t, charge_mode))

                # Also seed random point combos with Q/A tuning
                for _ in range(120):
                    g_t = tuple(rng.choice(["E", "K", "Q", "A"], size=n_h, p=[0.35, 0.35, 0.22, 0.08]))
                    e_t = tuple(rng.choice(["E", "K", "Q", "A"], size=n_h, p=[0.35, 0.35, 0.22, 0.08]))
                    if sum(x in ("E", "K") for x in g_t + e_t) >= n_h and (g_t + e_t).count("A") <= 2:
                        cands.append(_make_spec_checked(lm, a_tup, g_t, e_t, charge_mode))

            rng.shuffle(cands)
            for sp in cands:
                if _try_add(sp):
                    added += 1
                    if added >= max_mixed_per_len:
                        break

    return pool


def _resize_spec(
    spec: PeptideSpec,
    new_lm: str,
    charge_mode: str,
    rng: np.random.Generator,
) -> PeptideSpec:
    old_lm = spec.length_mode
    if old_lm == new_lm:
        return spec

    def _to_len(seq_t: Tuple[str, ...], old_m: str, new_m: str, fill_opts: Sequence[str]) -> Tuple[str, ...]:
        if old_m in ("3mer", "35mer") and new_m in ("3mer", "35mer"):
            return seq_t
        if new_m in ("3mer", "35mer"):
            # Keep middle Asn-containing region aligned for 35mer (1:4) vs 3mer (0:3)
            return seq_t[1:4] if new_m == "35mer" else seq_t[:3]
        if old_m in ("3mer", "35mer"):
            # Expand 3 -> 4 (and optionally 5)
            pad_aa = str(rng.choice(fill_opts))
            seq4 = ((pad_aa,) + seq_t) if old_m == "35mer" else (seq_t + (pad_aa,))
            if new_m == "4mer":
                return seq4
            pad_aa2 = str(rng.choice(fill_opts))
            return seq4 + (pad_aa2,)
        if old_m == "4mer" and new_m == "5mer":
            return seq_t + (str(rng.choice(fill_opts)),)
        # 5mer -> 4mer
        return seq_t[:4]

    eg_opts = EG_ALPHABET[charge_mode]
    a_new = _to_len(spec.a_core, old_lm, new_lm, ("I", "V"))
    # Ensure resized core stays in CORES_BY_LEN (always internal Asn, never terminal)
    if a_new not in CORES_BY_LEN[new_lm]:
        core_pool = CORES_BY_LEN[new_lm]
        a_new = core_pool[int(rng.integers(0, len(core_pool)))]

    g_new = _to_len(spec.g_tuple, old_lm, new_lm, eg_opts)
    e_new = _to_len(spec.e_tuple, old_lm, new_lm, eg_opts)
    return _make_spec_checked(new_lm, a_new, g_new, e_new, charge_mode)


def mutate_spec(
    spec: PeptideSpec,
    allowed_lengths: Sequence[str],
    charge_mode: str,
    rng: np.random.Generator,
) -> PeptideSpec:
    """Apply a structure-preserving local move (e/g flip, heptad flip, core swap, or length step)."""
    eg_opts = EG_ALPHABET[charge_mode]

    for _ in range(8):
        move = float(rng.random())
        lm = spec.length_mode
        n_h = len(spec.a_core)
        a_l = list(spec.a_core)
        g_l = list(spec.g_tuple)
        e_l = list(spec.e_tuple)

        if len(allowed_lengths) > 1 and move < 0.14:
            other_lens = [x for x in allowed_lengths if x != lm]
            cand = _resize_spec(spec, str(rng.choice(other_lens)), charge_mode, rng)
        elif move < 0.34:
            # Core move: pick another valid core for this length
            cores = CORES_BY_LEN[lm]
            a_new = cores[int(rng.integers(0, len(cores)))]
            cand = _make_spec_checked(lm, a_new, tuple(g_l), tuple(e_l), charge_mode)
        elif move < 0.56 and charge_mode in ("mixed", "any"):
            # Simultaneous heptad charge-block flip (e.g. K/K <-> E/E or K/Q)
            h = int(rng.integers(0, n_h))
            m = MIXED_MOTIFS[int(rng.integers(0, len(MIXED_MOTIFS)))]
            g_l[h], e_l[h] = m[0], m[1]
            cand = _make_spec_checked(lm, tuple(a_l), tuple(g_l), tuple(e_l), charge_mode)
        else:
            # 1-2 point mutations in g / e
            n_mut = 1 if rng.random() < 0.7 else 2
            for _m in range(n_mut):
                h = int(rng.integers(0, n_h))
                aa = str(rng.choice(eg_opts))
                if rng.random() < 0.5:
                    g_l[h] = aa
                else:
                    e_l[h] = aa
            if (g_l + e_l).count("A") > 2:
                continue
            cand = _make_spec_checked(lm, tuple(a_l), tuple(g_l), tuple(e_l), charge_mode)

        _, raw = build_sequence(cand)
        if valid_charge(raw, charge_mode, cand.length_mode):
            return cand

    return spec
