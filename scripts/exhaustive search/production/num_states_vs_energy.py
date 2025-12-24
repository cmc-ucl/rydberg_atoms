#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path

# ============================================================
# ========================== CONFIG ==========================
# ============================================================
# Paths
RYDBERG_NPY       = "Ryd_classical_E.npy"     # base energies, shape: [n_states]
CONCENTRATION_NPY = "concentration_all.npy"   # integer concentrations per state, shape: [n_states]

# Physics / binning
MU_EV        = -8.229098090849243e-08     # chemical potential shift (eV). Energy shift: E_mu = E0 + conc * MU_EV
BIN_WIDTH_EV = 10**-10        # eV. If set, overrides NUM_BINS
NUM_BINS     = 100         # used only if BIN_WIDTH_EV is None
E_MIN_EV     = None        # lower bound in eV (None → min(E_mu))
E_MAX_EV     = None        # upper bound in eV (None → max(E_mu))

# Output
OUT_PREFIX   = "num_states_vs_energy"  # creates "<prefix>__mu=<...>.csv/json"
# ============================================================


def build_bins(data, bin_width=None, num_bins=None, E_min=None, E_max=None):
    if E_min is None: E_min = float(np.min(data))
    if E_max is None: E_max = float(np.max(data))
    if E_max <= E_min:
        raise ValueError("E_max must be greater than E_min.")

    if bin_width is not None:
        # align edges to E_min so bins are consistent across runs
        n = int(np.ceil((E_max - E_min) / bin_width))
        edges = E_min + np.arange(n + 1) * bin_width
        # ensure last edge >= E_max
        if edges[-1] < E_max:
            edges = np.append(edges, edges[-1] + bin_width)
    elif num_bins is not None:
        edges = np.linspace(E_min, E_max, num_bins + 1)
    else:
        # sensible default
        num_bins = 100
        edges = np.linspace(E_min, E_max, num_bins + 1)
    return edges


def main():
    # Load data
    E0   = np.load(RYDBERG_NPY, mmap_mode="r")
    conc = np.load(CONCENTRATION_NPY, mmap_mode="r")

    if conc.dtype.kind != "i":
        raise ValueError("`concentration_all` must be integer-valued for exact μ shifts.")

    # Shift energies by μ
    E_mu = E0 + conc * MU_EV  # eV

    # Binning
    edges = build_bins(
        E_mu,
        bin_width=BIN_WIDTH_EV,
        num_bins=NUM_BINS if BIN_WIDTH_EV is None else None,
        E_min=E_MIN_EV,
        E_max=E_MAX_EV,
    )
    counts, edges = np.histogram(E_mu, bins=edges)

    # Prepare outputs
    mu_str   = f"{MU_EV:.8g}"
    base     = f"{OUT_PREFIX}__mu={mu_str}"
    csv_path = Path(f"{base}.csv")
    meta_path= Path(f"{base}.json")

    # Save CSV: left_edge,right_edge,count
    with open(csv_path, "w") as f:
        f.write("edge_left_eV,edge_right_eV,count\n")
        for i in range(len(edges) - 1):
            f.write(f"{edges[i]},{edges[i+1]},{int(counts[i])}\n")

    # Save small metadata
    binwidth = None
    if len(edges) >= 2:
        diffs = np.diff(edges)
        binwidth = float(diffs[0]) if np.allclose(diffs, diffs[0]) else "variable"

    meta = {
        "mu_eV": float(MU_EV),
        "n_states": int(E_mu.size),
        "emin_eV": float(edges[0]),
        "emax_eV": float(edges[-1]),
        "bin_count": int(len(edges) - 1),
        "bin_width_eV": binwidth,
        "files": {"csv": str(csv_path.name)},
        "notes": "Histogram counts per energy bin for shifted energies E + conc*mu."
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()