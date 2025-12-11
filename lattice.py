# lattice.py
"""
Spinless fermion chain with PBC (nearest-neighbour t-V model). Helper
functions for geometry and Hilbert-space enumeration.
"""
import itertools
from typing import List, Tuple
import numpy as np
import csv, re, json
import os, time, math, numpy as np
from pathlib import Path
from math import comb as _comb
import itertools 
import jax, jax.numpy as jnp
from datetime import datetime
from loss import _to_complex

# ---------- geometry -------------------------------------------------------- #
def neighbours(i: int, L: int) -> Tuple[int, int]:
    return ((i - 1) % L, (i + 1) % L)

# ---------- Hilbert-space enumeration -------------------------------------- #
def enumerate_fock(L: int, Np: int) -> List[int]:
    """All bit-masks with Np particles on L sites."""
    masks = []
    for occ in itertools.combinations(range(L), Np):
        m = 0
        for j in occ:
            m |= 1 << j
        masks.append(m)
    return masks

def mask_to_array(mask: int, L: int):
    string = np.binary_repr(mask,width=L) 
    return [int(bit) for bit in string]

# ---------- Helpers to grab ED data ------------------------------------------- #
def my_readcsv(path: str):
    """
    Parse a 'wide' ED CSV with columns like:
      delta, Eigenvalue_0, Eigenvalue_1, Eigenvalue_2, Eigenvalue_3,
      GroundStateWaveFunction_* , bitmask, ...

    File layout (as in your upload):
      - row[0]           -> delta (string int)
      - row[1]           -> a single cell containing the 4 eigenvalues as a bracketed string
                            e.g. "[-8.25  -8.44  -8.44  -8.22]"
                            (we'll parse and take E0 = min of these)
      - from some index  -> Nstates complex coeff cells, each like "(a+bj)"
      - next Nstates     -> integer bitmasks

    Returns:
      coeffs : np.ndarray[complex]   shape (Nstates,)
      masks  : np.ndarray[np.uint64] shape (Nstates,)
      E0     : float                 ground-state energy
    """
    p = Path(path)
    with p.open('r', newline='') as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise ValueError("CSV appears empty or header-only.")

    header = rows[0]
    row = rows[1]

    # --- infer Ns, Np -> Nstates from filename e.g. ..._Ns12_Np4_...
    m_ns = re.search(r"Ns(\d+)", p.stem)
    m_np = re.search(r"Np(\d+)", p.stem)
    if not (m_ns and m_np):
        raise ValueError("Filename must include Ns<sites> and Np<particles> (e.g. ..._Ns12_Np4_...).")
    Ns = int(m_ns.group(1))
    Np = int(m_np.group(1))
    Nstates = _comb(Ns, Np) # Ns => 2*Ns for Honeycomb ED data

    # --- parse ground-state energy from the eigenvalue fields
    # Try 1: the typical "one cell holds all 4 eigenvalues in brackets"
    E0 = None
    if len(row) > 1 and row[1].strip().startswith('['):
        inner = row[1].strip().strip('[]')
        # split on whitespace; allow no commas
        try:
            evals = [float(tok) for tok in re.split(r'\s+', inner.strip()) if tok]
            if evals:
                E0 = float(min(evals))
        except Exception:
            E0 = None

    # Try 2: if each eigenvalue is in its own column and is a number
    if E0 is None:
        # find indices of columns named Eigenvalue_*
        ev_idxs = [i for i, h in enumerate(header) if h.startswith("Eigenvalue_")]
        evals = []
        for i in ev_idxs:
            try:
                evals.append(float(row[i]))
            except Exception:
                pass
        if evals:
            E0 = float(min(evals))

    # If still None, leave E0 as None (not present in CSV)
    # --- find the start of the complex coefficient block
    # A complex token like "(<real>+<imag>j)" or "(<real>-<imag>j)"
    cx_pat = re.compile(
        r"^\(\s*[-+]?\d*\.?\d+(?:e[-+]?\d+)?\s*[-+]\s*\d*\.?\d+(?:e[-+]?\d+)?j\s*\)$",
        re.I
    )
    start = None
    for i, cell in enumerate(row):
        if cx_pat.match(cell.replace(" ", "")):
            start = i
            break
    if start is None:
        raise ValueError("Could not detect start of complex coefficients in the CSV row.")

    if len(row) < start + 2 * Nstates:
        raise ValueError(
            f"Row too short: need {2*Nstates} cells after coeff start={start}, "
            f"have {len(row)-start}."
        )

    coeff_cells = row[start:start + Nstates]
    mask_cells  = row[start + Nstates:start + 2 * Nstates]

    # --- parse complex coeffs and integer masks
    def parse_cx(s: str) -> complex:
        s2 = s.strip().strip("()").replace(" ", "")
        return complex(s2)

    coeffs = np.array([parse_cx(s) for s in coeff_cells], dtype=np.complex128)
    masks  = np.array([int(x) for x in mask_cells], dtype=np.uint64)

    # sanity check
    if coeffs.shape[0] != Nstates or masks.shape[0] != Nstates:
        raise ValueError("Parsed lengths do not match Nstates.")

    return coeffs, masks, E0

def per_group_overlap_metrics_host(preds_c: np.ndarray,
                                   target_c: np.ndarray,
                                   gids: np.ndarray,
                                   G: int) -> dict[str, np.ndarray]:
    """Compute |<Ψθ|ΨED>| per gid on CPU (post device_get)."""
    overlaps, losses, sizes = [], [], []
    for g in range(G):
        sel = (gids == g)
        th = preds_c[sel].copy()
        ed = target_c[sel].copy()
        th /= np.linalg.norm(th) + 1e-12
        ed /= np.linalg.norm(ed) + 1e-12
        ov = np.vdot(th, ed)
        overlaps.append(abs(ov))
        losses.append(1.0 - abs(ov)**2)
        sizes.append(int(sel.sum()))
    return {
        "overlap_abs": np.array(overlaps, dtype=float),
        "loss":        np.array(losses,   dtype=float),
        "sizes":       np.array(sizes,    dtype=int),
    }

def write_run_summary_gpu(out_dir: str | Path,
                          run_name: str,
                          config_dict: dict,
                          loss_history_device: jnp.ndarray,
                          preds_device: jnp.ndarray,
                          target_device: jnp.ndarray,
                          gids_device: jnp.ndarray,
                          G: int) -> Path:
    """
    One-row CSV with config + loss stats + per-group overlaps, minimizing
    device->host transfers.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}_summary.csv"

    # Ensure pending device work is done, then single device_get calls
    loss_history_device.block_until_ready()
    preds_device.block_until_ready()
    target_device.block_until_ready()
    gids_device.block_until_ready()

    losses = np.asarray(jax.device_get(loss_history_device), dtype=float)
    preds_c = np.asarray(jax.device_get(_to_complex(preds_device)))
    target_c= np.asarray(jax.device_get(_to_complex(target_device)))
    gids    = np.asarray(jax.device_get(gids_device), dtype=int)

    group_stats = per_group_overlap_metrics_host(preds_c, target_c, gids, G)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **config_dict,
        "epochs": int(losses.shape[0]),
        "loss_final": float(losses[-1]),
        "loss_mean": float(losses.mean()),
        "loss_var":  float(losses.var(ddof=0)),
        "loss_min":  float(losses.min()),
        "loss_max":  float(losses.max()),
        "group_overlap_abs": json.dumps(group_stats["overlap_abs"].tolist()),
        "group_loss":        json.dumps(group_stats["loss"].tolist()),
        "group_sizes":       json.dumps(group_stats["sizes"].tolist()),
    }

    # Fast CSV write without per-row Python overhead: use numpy.savetxt on values order
    # but we want keys too → use a tiny header+single-row csv via Python once.
    import csv
    with out_path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(row.keys()))
        wr.writeheader(); wr.writerow(row)

    return out_path
