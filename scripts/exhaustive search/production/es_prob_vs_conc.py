import json
import numpy as np
from multiprocessing import Pool
from functools import partial
from itertools import product
from scipy.special import logsumexp
from joblib import Parallel, delayed
from collections import OrderedDict

# -----------------------
# Partition function (yours)
# -----------------------
def get_partition_function(energy, multiplicity, T=298.15, return_pi=True, N_N=0, N_potential=0.):
    k_b = 8.617333262145e-05  # Boltzmann constant in eV/K
    energy = np.array(energy, dtype=float)
    multiplicity = np.array(multiplicity, dtype=float)

    # Compute log probabilities
    log_p_i = np.log(multiplicity) + (-energy + (N_N * N_potential)) / (k_b * T)
    log_Z = logsumexp(log_p_i)  # log of partition function

    if not np.isfinite(log_Z):
        print("Warning: log_Z is invalid!", log_Z)

    # Normalize in log-space and then exponentiate
    log_pi = log_p_i - log_Z
    pi = np.exp(log_pi)

    if return_pi:
        return np.exp(log_Z), pi
    return np.exp(log_Z)

# -----------------------
# Load state-space (yours)
# -----------------------
# with open("sample_4x3_4um_production.json", "r") as f:
#     data = json.load(f)
# Ryd_classical_E = np.array(data["Rydberg_classical_energy"])
# concentration_all = np.array(data["concentration"])
# np.save("Ryd_classical_E.npy", Ryd_classical_E)
# np.save("concentration_all.npy", concentration_all)

Ryd_classical_E = np.load("Ryd_classical_E.npy", mmap_mode="r")
concentration_all = np.load("concentration_all.npy", mmap_mode="r")

# IMPORTANT: concentration bins rely on integers
if concentration_all.dtype.kind != "i":
    raise ValueError("`concentration_all` must be integer-valued to aggregate exactly by concentration. "
                     "If you have fractional concentrations, ask for a binned version instead.")

num_sites = len(concentration_all)
# If you truly need to override, keep this; otherwise comment it out.
num_sites = 28

# -----------------------
# Ranges (yours)
# -----------------------
eV_to_rad_s = 1.519267447321156e15
Delta_mu_range = -np.linspace(125000000, -62500000, 10)
Delta_mu_range = Delta_mu_range / eV_to_rad_s  # -> eV
T_all = [4.1e-5]

# Grid of (T, Δμ)
T_mu_pairs = list(product(T_all, Delta_mu_range))

# -----------------------
# Average concentration (your original)
# -----------------------
def compute_av_conc_wrapper(T, Delta_mu):
    energy_new = Ryd_classical_E + concentration_all * Delta_mu
    Z, pi = get_partition_function(
        energy_new, [1] * len(Ryd_classical_E), return_pi=True, T=T
    )
    av_conc = np.sum(pi * concentration_all) / num_sites
    return (T, Delta_mu, float(av_conc))

# -----------------------
# NEW: concentration distribution per Δμ
# -----------------------
def concentration_distribution_for_mu(T, Delta_mu, max_conc, renormalize_within_range=True):
    """
    Compute P(conc=k) for k=0..max_conc for a single (T, Δμ).
    Returns a dict: {"T": T, "max_conc": max_conc, "renormalized": bool, "probs": {"0": p0, ...}}
    """
    # Energy shift by chemical potential
    energies_mu = Ryd_classical_E + concentration_all * Delta_mu

    # Microstate probabilities
    _, pi = get_partition_function(energies_mu, [1] * len(Ryd_classical_E), return_pi=True, T=T)
    pi = np.asarray(pi, dtype=float)

    # Aggregate to concentration probabilities via weighted bincount
    K = int(concentration_all.max())
    conc_probs_all = np.bincount(concentration_all, weights=pi, minlength=K + 1)

    # Keep only 0..max_conc
    conc_probs = conc_probs_all[: max_conc + 1].astype(float, copy=True)

    if renormalize_within_range:
        s = conc_probs.sum()
        if s > 0:
            conc_probs /= s

    # Serialize
    probs_dict = {str(k): float(conc_probs[k]) for k in range(max_conc + 1)}

    return {
        "T": float(T),
        "max_conc": int(max_conc),
        "renormalized": bool(renormalize_within_range),
        "probs": probs_dict,
    }

# -----------------------
# Parameters for the new JSON
# -----------------------
max_conc = 28            # <-- set this to whatever upper concentration you want to include
renormalize_within_range = True
outfile_dist = "conc_distribution_by_mu_T_4.1e-5.json"
outfile_av   = "av_conc_4x3_4um_production_T_4.1.json"  # your original output

# -----------------------
# Parallel runs
# -----------------------
# 1) Average concentration (original)
results_av = Parallel(n_jobs=10)(
    delayed(compute_av_conc_wrapper)(T, mu) for T, mu in T_mu_pairs
)

# 2) Probability distribution per Δμ
#    We'll store as an OrderedDict mapping each Δμ -> distribution dict
def _dist_job(pair):
    T, mu = pair
    return (mu, concentration_distribution_for_mu(T, mu, max_conc, renormalize_within_range))

dist_items = Parallel(n_jobs=10)(
    delayed(_dist_job)(pair) for pair in T_mu_pairs
)

# Since T_all likely has one element, multiple identical T for many μ; we key only by μ
# If you later add multiple T values, consider nesting by T then μ.
dist_by_mu = OrderedDict()
for mu, payload in dist_items:
    # Use string keys for stable JSON
    dist_by_mu[str(mu)] = payload

# -----------------------
# Save files
# -----------------------
# A) Your original average concentration results
with open(outfile_av, "w") as f:
    json.dump(results_av, f, indent=4)

# B) New JSON: per-μ concentration distributions
#    Schema: { "<mu_0>": {"T":..., "max_conc":..., "renormalized":..., "probs": { "0": p0, ... } }, ... }
with open(outfile_dist, "w") as f:
    json.dump(dist_by_mu, f, indent=2)

print(f"Saved average concentrations to: {outfile_av}")
print(f"Saved concentration distributions to: {outfile_dist}")
