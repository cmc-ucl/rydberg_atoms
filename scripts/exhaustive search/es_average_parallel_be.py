import json
import numpy as np
from multiprocessing import Pool
from functools import partial
from itertools import product
from scipy.special import logsumexp

import numpy as np

def get_bose_einstein_partition_function(energy, T=298.15, return_pi=True):
    """
    Compute the Bose–Einstein partition function and state probabilities
    (average occupation fractions), assuming the chemical potential has
    already been incorporated into the energy values.

    Args:
        energy (np.ndarray): Effective energy levels (in eV), already including -μ
        T (float): Temperature in Kelvin
        return_pi (bool): Whether to return probabilities

    Returns:
        float: Partition function
        np.ndarray: Occupation probabilities (if return_pi=True)
    """
    k_b = 8.617333262145e-05  # Boltzmann constant in eV/K
    energy = np.array(energy, dtype=float)
    beta = 1.0 / (k_b * T)
    x = beta * energy

    # Prevent invalid (<= 0) arguments for exp in denominator
    if np.any(x <= 0):
        raise ValueError("All effective energy values must be positive for Bose–Einstein stats.")

    # Average occupation numbers
    n_i = 1.0 / (np.exp(x) - 1.0)

    # Partition function is not the same as classical Z; this is a formal total weight
    ln_Z = -np.sum(np.log(1 - np.exp(-x)))  # ln(Z)

    if not np.isfinite(ln_Z):
        print("Warning: log(Z) is not finite!", ln_Z)

    if return_pi:
        total = np.sum(n_i)
        pi = n_i / total if total > 0 else np.zeros_like(n_i)
        return np.exp(ln_Z), pi
    return np.exp(ln_Z)

# Load the energies and concentrations
with open("sample_4x3_4um.json", "r") as f:
    data = json.load(f)

Ryd_classical_E = np.array(data["Rydberg_classical_energy"])
concentration_all = np.array(data["concentration"])
num_sites = 28

# Define Δμ and T ranges
eV_to_rad_s = 1.519267447321156e15
Delta_mu_range = np.linspace(-1.25e8, 1.25e8, 100, endpoint=True)
Delta_mu_range = Delta_mu_range / eV_to_rad_s

T_all = np.linspace(1e-6, 1e-4, 10, endpoint=True)

# Create all combinations of (T, Δμ)
T_mu_pairs = list(product(T_all, Delta_mu_range))

# Function to compute average concentration
def compute_av_conc(T, Delta_mu, energies, concentrations, num_sites):
    energy_new = energies + concentrations * Delta_mu
    Z, pi = get_bose_einstein_partition_function(
        energy_new, return_pi=True, T=T
    )
    av_conc = np.sum(pi * concentrations) / num_sites
    return (T, Delta_mu, float(av_conc))

# Parallel execution
with Pool(processes=128) as pool:
    func = partial(compute_av_conc,
                   energies=Ryd_classical_E,
                   concentrations=concentration_all,
                   num_sites=num_sites)
    flat_results = pool.starmap(func, T_mu_pairs)

# Organize results into the original format
results = {}
for T, Delta_mu, av_conc in flat_results:
    if T not in results:
        results[T] = {
            "Delta_mu": [],
            "avg_conc": []
        }
    results[T]["Delta_mu"].append(Delta_mu)
    results[T]["avg_conc"].append(av_conc)

# Save the averaged concentration results
with open("av_conc_4x3_4um_only.json", "w") as f:
    json.dump(results, f, indent=4)