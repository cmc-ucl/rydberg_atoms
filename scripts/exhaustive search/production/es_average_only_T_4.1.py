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

#Ryd_classical_E = np.array(data["Rydberg_classical_energy"])
#concentration_all = np.array(data["concentration"])
#np.save("Ryd_classical_E.npy", Ryd_classical_E)
#np.save("concentration_all.npy", concentration_all)
Ryd_classical_E = np.load("Ryd_classical_E.npy", mmap_mode="r")
concentration_all = np.load("concentration_all.npy", mmap_mode="r")
num_sites = len(concentration_all)
num_sites = 28 

# Define Δμ and T ranges
eV_to_rad_s = 1.519267447321156e15
Delta_mu_range = np.linspace(125000000, -125000000,100)
Delta_mu_range = Delta_mu_range / eV_to_rad_s

T_all = [4.1e-5]

# Create all combinations of (T, Δμ)
T_mu_pairs = list(product(T_all, Delta_mu_range))

# Function to compute average concentration
#def compute_av_conc(T, Delta_mu, energies, concentrations, num_sites):
#    energy_new = energies + concentrations * Delta_mu
#    Z, pi = get_partition_function(
#        energy_new, [1] * len(energies), return_pi=True, T=T
#    )
#    av_conc = np.sum(pi * concentrations) / num_sites
#    return (T, Delta_mu, float(av_conc))
from joblib import Parallel, delayed

def compute_av_conc_wrapper(T, Delta_mu):
    energy_new = Ryd_classical_E + concentration_all * Delta_mu
    Z, pi = get_partition_function(
        energy_new, [1] * len(Ryd_classical_E), return_pi=True, T=T
    )
    av_conc = np.sum(pi * concentration_all) / num_sites
    return (T, Delta_mu, float(av_conc))

results = Parallel(n_jobs=10)(
    delayed(compute_av_conc_wrapper)(T, mu) for T, mu in T_mu_pairs
)
# Parallel execution
#with Pool(processes=128) as pool:
#    func = partial(compute_av_conc,
#                   energies=Ryd_classical_E,
#                   concentrations=concentration_all,
#                   num_sites=num_sites)
#    flat_results = pool.starmap(func, T_mu_pairs)

# Organize results into the original format
#results = {}
#for T, Delta_mu, av_conc in flat_results:
#    if T not in results:
#        results[T] = {
#            "Delta_mu": [],
#            "avg_conc": []
#        }
#    results[T]["Delta_mu"].append(Delta_mu)
#    results[T]["avg_conc"].append(av_conc)

# Save the averaged concentration results
with open("av_conc_4x3_4um_production_T_4.1.json", "w") as f:
    json.dump(results, f, indent=4)
