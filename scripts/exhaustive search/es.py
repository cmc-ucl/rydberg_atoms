import itertools
import numpy as np
import copy
import json
from scipy.spatial import distance_matrix
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

coordinates = np.array([
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
 [-2.00500000e-06,  3.47276187e-06,  0.00000000e+00],
 [ 0.00000000e+00 , 6.94552374e-06,  0.00000000e+00],
 [-2.00500000e-06,  1.04182856e-05,  0.00000000e+00],
 [ 0.00000000e+00,  1.38910475e-05,  0.00000000e+00],
 [-2.00500000e-06,  1.73638093e-05,  0.00000000e+00],
 [ 0.00000000e+00,  2.08365712e-05,  0.00000000e+00],
 [ 4.01000000e-06,  0.00000000e+00,  0.00000000e+00],
 [ 6.01500000e-06,  3.47276187e-06,  0.00000000e+00],
 [ 4.01000000e-06,  6.94552374e-06,  0.00000000e+00],
 [ 6.01500000e-06,  1.04182856e-05,  0.00000000e+00],
 [ 4.01000000e-06,  1.38910475e-05,  0.00000000e+00],
 [ 6.01500000e-06,  1.73638093e-05,  0.00000000e+00],
 [ 4.01000000e-06,  2.08365712e-05,  0.00000000e+00],
 [ 1.20300000e-05,  0.00000000e+00,  0.00000000e+00],
 [ 1.00250000e-05,  3.47276187e-06,  0.00000000e+00],
 [ 1.20300000e-05,  6.94552374e-06,  0.00000000e+00],
 [ 1.00250000e-05,  1.04182856e-05,  0.00000000e+00],
 [ 1.20300000e-05,  1.38910475e-05,  0.00000000e+00],
 [ 1.00250000e-05,  1.73638093e-05,  0.00000000e+00],
 [ 1.20300000e-05,  2.08365712e-05,  0.00000000e+00],
 [ 1.60400000e-05,  0.00000000e+00,  0.00000000e+00],
 [ 1.80450000e-05,  3.47276187e-06,  0.00000000e+00],
 [ 1.60400000e-05,  6.94552374e-06,  0.00000000e+00],
 [ 1.80450000e-05,  1.04182856e-05,  0.00000000e+00],
 [ 1.60400000e-05,  1.38910475e-05,  0.00000000e+00],
 [ 1.80450000e-05,  1.73638093e-05,  0.00000000e+00],
 [ 1.60400000e-05,  2.08365712e-05,  0.00000000e+00]])

dm = distance_matrix(coordinates, coordinates)

C6 = 5.42e-24  # C6 constant in rad m^6 / s
eV_to_rad_s = 1.519267447321156e15

np.fill_diagonal(dm,1)
dm_inv = 1/dm


dm_inv_6 = dm_inv**6

dm_inv_6_C6 = C6*dm_inv_6



dm_inv_6_C6_eV = dm_inv_6_C6/eV_to_rad_s

np.fill_diagonal(dm_inv_6_C6_eV,0)


# Write config-energies
Ryd_classical_E = []
concentration_all = []

num_sites = len(coordinates)
max_conc = len(coordinates)


# Generate binary vectors and energy values
for x in itertools.product([0, 1], repeat=len(coordinates)):   
    xx = np.outer(x,x)
    dm_xx = dm_inv_6_C6_eV * xx
    energy = np.sum(np.triu(dm_xx))
    Ryd_classical_E.append(energy)
    concentration_all.append(np.sum(x))

import json

# Example: save to 'output.json'
data = {
    "concentration": concentration_all,
    "Rydberg_classical_energy": Ryd_classical_E
}

with open(f"production_sample_4x3_4um_all_energies.json", "w") as f:
    json.dump(data, f, indent=4)


# Delta_mu_range = np.linspace(-1.25e8,1.25e8,100,endpoint=True)
Delta_mu_range = -np.linspace(125000000, -62500000, 10)
Delta_mu_range = np.array(Delta_mu_range)/eV_to_rad_s

T_all = np.linspace(1e-6,1e-4,20,endpoint=True)
# T_all = [1e-6]

concentration_all = np.array(concentration_all)
# Initialize result dictionary
results = {}

for idx, T in enumerate(T_all):
    av_conc_classical = []

    for Delta_mu in Delta_mu_range:
        # Calculate energy for this Δμ
        energy_new = Ryd_classical_E + concentration_all * Delta_mu
        Z, pi = get_partition_function(
            energy_new, [1] * len(Ryd_classical_E), return_pi=True, T=T
        )
        av_conc = np.sum(pi * concentration_all) / num_sites
        av_conc_classical.append(float(av_conc))

    # Store the data under temperature T
    results[T] = {
        "Delta_mu": Delta_mu_range.tolist(),
        "avg_conc": av_conc_classical
    }

# Save to JSON
with open(f"production_av_conc_4x3_4um_all_T.json", "w") as f:
    json.dump(results, f, indent=4)

        



    