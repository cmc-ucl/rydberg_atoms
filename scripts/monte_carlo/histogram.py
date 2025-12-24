import json
import numpy as np
from multiprocessing import Pool
from functools import partial
from itertools import product
from scipy.special import logsumexp


# Load the energies and concentrations
with open("data/Monte_carlo/sample_1000000_4.1um.json", "r") as f:
    data = json.load(f)

Ryd_classical_E = np.array(data["Rydberg_classical_energy"])
concentration_all = np.array(data["concentration"])

# Define Δμ and T ranges
eV_to_rad_s = 1.519267447321156e15
Delta_mu_range = np.array(['-125000000','-62500000','-1250','0','1250','62500000','125000000'])
delta_g_all_int = np.array([-int(x) for x in Delta_mu_range])

Delta_mu_range_eV = -delta_g_all_int / eV_to_rad_s

# Dictionary to hold everything
results = {}

# Loop over chemical potentials
for i,mu in enumerate(Delta_mu_range_eV):
    print(mu)
    energy_new = Ryd_classical_E + concentration_all * mu
    print(np.max(energy_new),np.min(energy_new))
    counts, bins = np.histogram(energy_new, bins=10000, density=True)
    
    # Use str(mu) as key because JSON keys must be strings
    results[Delta_mu_range[i]] = {
        "counts": counts.tolist(),
        "bins": bins.tolist()
    }

# Save to JSON
with open("data/Monte_carlo/histogram_new.json", "w") as f:
    json.dump(results, f, indent=2)


