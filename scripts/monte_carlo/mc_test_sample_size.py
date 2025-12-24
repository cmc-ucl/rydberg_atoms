import copy
import json
import numpy as np
from scipy.special import logsumexp

from scipy.spatial import distance_matrix

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
       [-2.00500000e-06,  2.43093331e-05],
       [ 0.00000000e+00,  2.77820950e-05],
       [-2.00500000e-06,  3.12548568e-05],
       [ 0.00000000e+00,  3.47276187e-05],
       [-2.00500000e-06,  3.82003806e-05],
       [ 0.00000000e+00,  4.16731424e-05],
       [ 4.01000000e-06,  0.00000000e+00],
       [ 6.01500000e-06,  3.47276187e-06],
       [ 4.01000000e-06,  6.94552374e-06],
       [ 6.01500000e-06,  1.04182856e-05],
       [ 4.01000000e-06,  1.38910475e-05],
       [ 6.01500000e-06,  1.73638093e-05],
       [ 4.01000000e-06,  2.08365712e-05],
       [ 6.01500000e-06,  2.43093331e-05],
       [ 4.01000000e-06,  2.77820950e-05],
       [ 6.01500000e-06,  3.12548568e-05],
       [ 4.01000000e-06,  3.47276187e-05],
       [ 6.01500000e-06,  3.82003806e-05],
       [ 4.01000000e-06,  4.16731424e-05],
       [ 1.20300000e-05,  0.00000000e+00],
       [ 1.00250000e-05,  3.47276187e-06],
       [ 1.20300000e-05,  6.94552374e-06],
       [ 1.00250000e-05,  1.04182856e-05],
       [ 1.20300000e-05,  1.38910475e-05],
       [ 1.00250000e-05,  1.73638093e-05],
       [ 1.20300000e-05,  2.08365712e-05],
       [ 1.00250000e-05,  2.43093331e-05],
       [ 1.20300000e-05,  2.77820950e-05],
       [ 1.00250000e-05,  3.12548568e-05],
       [ 1.20300000e-05,  3.47276187e-05],
       [ 1.00250000e-05,  3.82003806e-05],
       [ 1.20300000e-05,  4.16731424e-05],
       [ 1.60400000e-05,  0.00000000e+00],
       [ 1.80450000e-05,  3.47276187e-06],
       [ 1.60400000e-05,  6.94552374e-06],
       [ 1.80450000e-05,  1.04182856e-05],
       [ 1.60400000e-05,  1.38910475e-05],
       [ 1.80450000e-05,  1.73638093e-05],
       [ 1.60400000e-05,  2.08365712e-05],
       [ 1.80450000e-05,  2.43093331e-05],
       [ 1.60400000e-05,  2.77820950e-05],
       [ 1.80450000e-05,  3.12548568e-05],
       [ 1.60400000e-05,  3.47276187e-05],
       [ 1.80450000e-05,  3.82003806e-05],
       [ 1.60400000e-05,  4.16731424e-05],
       [ 2.40600000e-05,  0.00000000e+00],
       [ 2.20550000e-05,  3.47276187e-06],
       [ 2.40600000e-05,  6.94552374e-06],
       [ 2.20550000e-05,  1.04182856e-05],
       [ 2.40600000e-05,  1.38910475e-05],
       [ 2.20550000e-05,  1.73638093e-05],
       [ 2.40600000e-05,  2.08365712e-05],
       [ 2.20550000e-05,  2.43093331e-05],
       [ 2.40600000e-05,  2.77820950e-05],
       [ 2.20550000e-05,  3.12548568e-05],
       [ 2.40600000e-05,  3.47276187e-05],
       [ 2.20550000e-05,  3.82003806e-05],
       [ 2.40600000e-05,  4.16731424e-05],
       [ 2.80700000e-05,  0.00000000e+00],
       [ 3.00750000e-05,  3.47276187e-06],
       [ 2.80700000e-05,  6.94552374e-06],
       [ 3.00750000e-05,  1.04182856e-05],
       [ 2.80700000e-05,  1.38910475e-05],
       [ 3.00750000e-05,  1.73638093e-05],
       [ 2.80700000e-05,  2.08365712e-05],
       [ 3.00750000e-05,  2.43093331e-05],
       [ 2.80700000e-05,  2.77820950e-05],
       [ 3.00750000e-05,  3.12548568e-05],
       [ 2.80700000e-05,  3.47276187e-05],
       [ 3.00750000e-05,  3.82003806e-05],
       [ 2.80700000e-05,  4.16731424e-05]])

dm = distance_matrix(coordinates, coordinates)

C6 = 5.42e-24  # C6 constant in rad m^6 / s
eV_to_rad_s = 1.5193e+15 

np.fill_diagonal(dm,1)
dm_inv = 1/dm


dm_inv_6 = dm_inv**6

dm_inv_6_C6 = C6*dm_inv_6



dm_inv_6_C6_eV = dm_inv_6_C6/eV_to_rad_s

np.fill_diagonal(dm_inv_6_C6_eV,0)


# Write config-energies
size = 10000000
Ryd_classical_E = []
concentration_all = []

num_sites = len(coordinates)
max_conc = len(coordinates)


concentration = np.random.randint(0, max_conc, size=size)

# Generate binary vectors and energy values

import json

# Example: save to 'output.json'

with open(f"sample_{size}_4um.json", "r") as f:
    data = json.load(f)

# Access the contents
concentration = data["concentration"]
Ryd_classical_E = data["Rydberg_classical_energy"]

concentration_init = np.array(concentration)
Ryd_classical_E_init = np.array(Ryd_classical_E)

V_ratio = 237.18294429260789
#Delta_mu_range = np.array([125000000,62500000,31250000,0,-31250000])#*V_ratio
Delta_mu_range = np.array([125000000, 62500000, 12500, 1250, 0, -1250, -62500000, -125000000])*V_ratio
Delta_mu_range = -Delta_mu_range / eV_to_rad_s

#T_all = np.linspace(1e-6,1e-2,30,endpoint=True)
T_all = np.linspace(1e-6, 1e-4, 10, endpoint=True)*V_ratio
T = 1e-5

# Initialize result dictionary
results = {}

sample_size = [1000,10000,100000,1000000,10000000]

for i in sample_size:
    av_conc_classical = []

    N = len(Ryd_classical_E)  # should be 10,000,000

    # Generate i unique random indices
    indices = np.random.choice(N, size=i, replace=False)

    # Select corresponding elements
    Ryd_classical_E = Ryd_classical_E_init[indices]
    concentration = concentration_init[indices]

    for Delta_mu in Delta_mu_range:
        # Calculate energy for this Δμ

        energy_new = Ryd_classical_E + concentration * Delta_mu
        Z, pi = get_partition_function(
            energy_new, [1] * len(Ryd_classical_E), return_pi=True, T=T
        )
        av_conc = np.sum(pi * concentration) / num_sites
        av_conc_classical.append(float(av_conc))

    # Store the data under temperature T
    results[i] = {
        "Delta_mu": Delta_mu_range.tolist(),
        "avg_conc": av_conc_classical
    }

# Save to JSON
with open(f"av_conc_test_sample_size_4um_hardware_values.json", "w") as f:
    json.dump(results, f, indent=4)

        




