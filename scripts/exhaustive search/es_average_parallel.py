import json
import numpy as np
from multiprocessing import Pool
from functools import partial
from itertools import product
from scipy.special import logsumexp

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

# Load the energies and concentrations
with open("sample_4x3_4um_production.json", "r") as f:
    data = json.load(f)

Ryd_classical_E = np.array(data["Rydberg_classical_energy"])
concentration_all = np.array(data["concentration"])
num_sites = len(concentration_all)
num_sites = 28 

# Define Δμ and T ranges
eV_to_rad_s = 1.519267447321156e15
Delta_mu_range = np.append(-np.linspace(125000000, -62500000, 10),125000000)
Delta_mu_range = Delta_mu_range / eV_to_rad_s

T_all = np.linspace(1e-6, 1e-4, 100, endpoint=True)

# Create all combinations of (T, Δμ)
T_mu_pairs = list(product(T_all, Delta_mu_range))

# Function to compute average concentration
def compute_av_conc(T, Delta_mu, energies, concentrations, num_sites):
    energy_new = energies + concentrations * Delta_mu
    Z, pi = get_partition_function(
        energy_new, [1] * len(energies), return_pi=True, T=T
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
with open("av_conc_4x3_4um_production.json", "w") as f:
    json.dump(results, f, indent=4)
