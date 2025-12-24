import json
import numpy as np
import itertools
from multiprocessing import Pool
from scipy.special import logsumexp

# Load data
with open("sample_4x3_4um.json", "r") as f:
    data = json.load(f)

Ryd_classical_E = np.array(data["Rydberg_classical_energy"])
concentration = np.array(data["concentration"])
num_sites = 28

# Constants
eV_to_rad_s = 1.519267447321156e15
delta_g_all = ['125000000','62500000','31250000','15625000','0','-31250000']
delta_g_all_int = np.array([-int(x) for x in delta_g_all])
Delta_mu_range = delta_g_all_int / eV_to_rad_s
T = 5e-5


def get_partition_function(energy, multiplicity, T=298.15, return_pi=True, N_N=0, N_potential=0.):
    k_b = 8.617333262145e-05
    energy = np.array(energy, dtype=float)
    multiplicity = np.array(multiplicity, dtype=float)
    log_p_i = np.log(multiplicity) + (-energy + (N_N * N_potential)) / (k_b * T)
    log_Z = logsumexp(log_p_i)
    log_pi = log_p_i - log_Z
    pi = np.exp(log_pi)
    return (np.exp(log_Z), pi) if return_pi else np.exp(log_Z)


# Precompute all configurations
configurations = list(itertools.product([0, 1], repeat=num_sites))

def compute_prob_site(args):
    i, x, pi = args
    x_arr = np.array(x)
    if np.sum(x_arr) > 0:
        prob_site = (pi[i] * x_arr) / np.sum(x_arr)
    else:
        prob_site = np.zeros_like(x_arr)
    return prob_site.tolist()


results = {}

for Delta_mu_str, Delta_mu in zip(delta_g_all, Delta_mu_range):
    print(f"Processing Δμ = {Delta_mu_str}...")
    energy_new = Ryd_classical_E + concentration * Delta_mu
    Z, pi = get_partition_function(energy_new, [1] * len(Ryd_classical_E), T=T, return_pi=True)

    with Pool() as pool:
        args = [(i, x, pi) for i, x in enumerate(configurations)]
        prob_site_mu = pool.map(compute_prob_site, args)

    results[Delta_mu_str] = prob_site_mu

# Save to JSON
with open("prob_site_mu_parallel.json", "w") as f:
    json.dump(results, f, indent=2)