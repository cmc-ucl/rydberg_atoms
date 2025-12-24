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
with open("../sample_4x3_4.01um.json", "r") as f:
    data = json.load(f)

Ryd_classical_E = np.array(data["Rydberg_classical_energy"])
concentration_all = np.array(data["concentration"])
num_sites = len(concentration_all)
print(num_sites)
num_sites = 28 
print(num_sites)

# Define Δμ and T ranges
eV_to_rad_s = 1.519267447321156e15
Delta_mu_range = np.array([125000000,62500000,31250000,15625000,0,-31250000])
Delta_mu_range = -Delta_mu_range / eV_to_rad_s

# Access the contents
concentration = data["concentration"]
Ryd_classical_E = data["Rydberg_classical_energy"]

concentration_init = np.array(concentration)
Ryd_classical_E_init = np.array(Ryd_classical_E)


#T_all = np.linspace(1e-6,1e-2,30,endpoint=True)
#T_all = np.linspace(1e-6, 1e-4, 10, endpoint=True)*V_ratio
T = 1e-5

# Initialize result dictionary
results = {}

sample_size = [int(x) for x in np.linspace(0.1,1,10)*2**28]

for i in sample_size:
    av_conc_classical = []

    N = len(Ryd_classical_E_init)  # should be 10,000,000

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
# Save the averaged concentration results
with open("av_conc_4x3_4.01um_hardware_values_mc_values.json", "w") as f:
    json.dump(results, f, indent=4)
