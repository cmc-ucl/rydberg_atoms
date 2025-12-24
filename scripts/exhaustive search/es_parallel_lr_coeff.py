import itertools
import numpy as np
import copy
import json
from scipy.spatial import distance_matrix
from multiprocessing import Pool, cpu_count
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

coordinates = np.array([[ 0.00000000e+00,  0.00000000e+00],
       [-2.00500000e-06,  3.47276187e-06],
       [ 0.00000000e+00,  6.94552374e-06],
       [-2.00500000e-06,  1.04182856e-05],
       [ 0.00000000e+00,  1.38910475e-05],
       [-2.00500000e-06,  1.73638093e-05],
       [ 0.00000000e+00,  2.08365712e-05],
       [ 4.01000000e-06,  0.00000000e+00],
       [ 6.01500000e-06,  3.47276187e-06],
       [ 4.01000000e-06,  6.94552374e-06],
       [ 6.01500000e-06,  1.04182856e-05],
       [ 4.01000000e-06,  1.38910475e-05],
       [ 6.01500000e-06,  1.73638093e-05],
       [ 4.01000000e-06,  2.08365712e-05],
       [ 1.20300000e-05,  0.00000000e+00],
       [ 1.00250000e-05,  3.47276187e-06],
       [ 1.20300000e-05,  6.94552374e-06],
       [ 1.00250000e-05,  1.04182856e-05],
       [ 1.20300000e-05,  1.38910475e-05],
       [ 1.00250000e-05,  1.73638093e-05],
       [ 1.20300000e-05,  2.08365712e-05],
       [ 1.60400000e-05,  0.00000000e+00],
       [ 1.80450000e-05,  3.47276187e-06],
       [ 1.60400000e-05,  6.94552374e-06],
       [ 1.80450000e-05,  1.04182856e-05],
       [ 1.60400000e-05,  1.38910475e-05],
       [ 1.80450000e-05,  1.73638093e-05],
       [ 1.60400000e-05,  2.08365712e-05]])

dm = distance_matrix(coordinates, coordinates)

# Remove zero (self-distances), flatten, and round to avoid floating point noise
flat_dists = np.unique(np.round(dm[dm > 1e-12], decimals=12))

# Sort distances into shells (1 = nearest neighbor, 2 = next, ...)
shell_matrix = np.zeros_like(dm, dtype=int)

for shell_index, dist in enumerate(flat_dists, start=1):
    mask = np.isclose(dm, dist, atol=1e-12)
    shell_matrix[mask] = shell_index

V_nn = 0.00020351
scaling_factors = [0,1,1/27,1/343]

interaction_matrix = np.zeros((len(coordinates),len(coordinates)))
for shell in [1,2,3]:
    idx = np.where(shell_matrix == shell)
    # print(idx)
    interaction_matrix[idx] = V_nn*scaling_factors[shell]


# Write config-energies
Ryd_classical_E = []
concentration_all = []

num_sites = len(coordinates)
max_conc = len(coordinates)


# Generate binary vectors and energy values
def compute_energy(x, interaction_matrix):
    x = np.array(x, dtype=np.uint8)
    xx = np.outer(x, x)
    dm_xx = interaction_matrix * xx
    energy = np.sum(np.triu(dm_xx))
    concentration = int(np.sum(x))
    return energy, concentration

# Prepare data
interaction_matrix = interaction_matrix
num_sites = len(coordinates)
n_bits = num_sites

# Generator of binary configurations
def binary_configs(n):
    return itertools.product([0, 1], repeat=n)

# Run in parallel
with Pool(processes=128) as pool:
    func = partial(compute_energy, interaction_matrix=interaction_matrix)
    results = pool.map(func, binary_configs(n_bits))

# Unpack results
Ryd_classical_E, concentration_all = zip(*results)
Ryd_classical_E = list(Ryd_classical_E)
concentration_all = list(concentration_all)

import json

# Example: save to 'output.json'
data = {
    "concentration": concentration_all,
    "Rydberg_classical_energy": Ryd_classical_E
}

with open(f"sample_4x3_lr_coeff.json", "w") as f:
    json.dump(data, f, indent=4)


# Delta_mu_range = np.linspace(-1.25e8,1.25e8,100,endpoint=True)
# Delta_mu_range = np.array(Delta_mu_range)/eV_to_rad_s

# T_all = np.linspace(1e-6,1e-4,10,endpoint=True)
# # T_all = [1e-6]

# concentration_all = np.array(concentration_all)

# # Initialize result dictionary
# # Create all combinations of (T, Δμ)
# T_mu_pairs = list(product(T_all, Delta_mu_range))

# # Function to compute average concentration
# def compute_av_conc(T, Delta_mu, energies, concentrations, num_sites):
#     energy_new = energies + concentrations * Delta_mu
#     Z, pi = get_partition_function(
#         energy_new, [1] * len(energies), return_pi=True, T=T
#     )
#     av_conc = np.sum(pi * concentrations) / num_sites
#     return (T, Delta_mu, float(av_conc))

# # Parallel execution
# with Pool(processes=128) as pool:
#     func = partial(compute_av_conc,
#                    energies=Ryd_classical_E,
#                    concentrations=concentration_all,
#                    num_sites=num_sites)
#     flat_results = pool.starmap(func, T_mu_pairs)

# # Organize results into the original format
# results = {}
# for T, Delta_mu, av_conc in flat_results:
#     if T not in results:
#         results[T] = {
#             "Delta_mu": [],
#             "avg_conc": []
#         }
#     results[T]["Delta_mu"].append(Delta_mu)
#     results[T]["avg_conc"].append(av_conc)

# # Save the averaged concentration results
# with open("av_conc_4x3_4.01um.json", "w") as f:
#     json.dump(results, f, indent=4)

        



    