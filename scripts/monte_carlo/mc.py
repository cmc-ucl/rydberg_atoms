import copy
import json
import numpy as np

from scipy.spatial import distance_matrix

def get_partition_function(energy, multiplicity, T=298.15, return_pi=True, N_N=0, N_potential=0.):
    """
    Calculate the partition function and probabilities for different energy levels.
    
    Args:
        energy (np.ndarray): Array of energy levels (in eV).
        multiplicity (np.ndarray): Array of corresponding multiplicities for energy levels.
        T (float, optional): Temperature in Kelvin. Default is 298.15 K.
        return_pi (bool, optional): Whether to return probabilities alongside the partition function. Default is True.
        N_N (float, optional): Number of N particles to consider. Default is 0.
        N_potential (float, optional): Potential energy contribution for N particles (in eV). Default is 0.

    Returns:
        tuple or float: 
            - If `return_pi` is True: Returns a tuple (partition function, probabilities).
            - If `return_pi` is False: Returns only the partition function.
    """
    # Constants
    k_b = 8.617333262145e-05  # Boltzmann constant in eV/K

    # Ensure inputs are NumPy arrays
    energy = np.array(energy, dtype=float)
    multiplicity = np.array(multiplicity, dtype=float)
    
    # Calculate the weighted probabilities for each energy level
    exponent = (-energy + (N_N * N_potential)) / (k_b * T)
    p_i = multiplicity * np.exp(exponent)
    
    # Compute the partition function
    pf = np.sum(p_i)
    
    # Normalize probabilities
    p_i /= pf
    
    # Return the results based on the flag
    if return_pi:
        return pf, p_i
    return pf

coordinates = np.array([[ 0.00000000e+00,  0.00000000e+00],
       [-2.05000000e-06,  3.55070416e-06],
       [ 0.00000000e+00,  7.10140831e-06],
       [-2.05000000e-06,  1.06521125e-05],
       [ 0.00000000e+00,  1.42028166e-05],
       [-2.05000000e-06,  1.77535208e-05],
       [ 0.00000000e+00,  2.13042249e-05],
       [-2.05000000e-06,  2.48549291e-05],
       [ 0.00000000e+00,  2.84056332e-05],
       [-2.05000000e-06,  3.19563374e-05],
       [ 0.00000000e+00,  3.55070416e-05],
       [-2.05000000e-06,  3.90577457e-05],
       [ 0.00000000e+00,  4.26084499e-05],
       [ 4.10000000e-06,  0.00000000e+00],
       [ 6.15000000e-06,  3.55070416e-06],
       [ 4.10000000e-06,  7.10140831e-06],
       [ 6.15000000e-06,  1.06521125e-05],
       [ 4.10000000e-06,  1.42028166e-05],
       [ 6.15000000e-06,  1.77535208e-05],
       [ 4.10000000e-06,  2.13042249e-05],
       [ 6.15000000e-06,  2.48549291e-05],
       [ 4.10000000e-06,  2.84056332e-05],
       [ 6.15000000e-06,  3.19563374e-05],
       [ 4.10000000e-06,  3.55070416e-05],
       [ 6.15000000e-06,  3.90577457e-05],
       [ 4.10000000e-06,  4.26084499e-05],
       [ 1.23000000e-05,  0.00000000e+00],
       [ 1.02500000e-05,  3.55070416e-06],
       [ 1.23000000e-05,  7.10140831e-06],
       [ 1.02500000e-05,  1.06521125e-05],
       [ 1.23000000e-05,  1.42028166e-05],
       [ 1.02500000e-05,  1.77535208e-05],
       [ 1.23000000e-05,  2.13042249e-05],
       [ 1.02500000e-05,  2.48549291e-05],
       [ 1.23000000e-05,  2.84056332e-05],
       [ 1.02500000e-05,  3.19563374e-05],
       [ 1.23000000e-05,  3.55070416e-05],
       [ 1.02500000e-05,  3.90577457e-05],
       [ 1.23000000e-05,  4.26084499e-05],
       [ 1.64000000e-05,  0.00000000e+00],
       [ 1.84500000e-05,  3.55070416e-06],
       [ 1.64000000e-05,  7.10140831e-06],
       [ 1.84500000e-05,  1.06521125e-05],
       [ 1.64000000e-05,  1.42028166e-05],
       [ 1.84500000e-05,  1.77535208e-05],
       [ 1.64000000e-05,  2.13042249e-05],
       [ 1.84500000e-05,  2.48549291e-05],
       [ 1.64000000e-05,  2.84056332e-05],
       [ 1.84500000e-05,  3.19563374e-05],
       [ 1.64000000e-05,  3.55070416e-05],
       [ 1.84500000e-05,  3.90577457e-05],
       [ 1.64000000e-05,  4.26084499e-05],
       [ 2.46000000e-05,  0.00000000e+00],
       [ 2.25500000e-05,  3.55070416e-06],
       [ 2.46000000e-05,  7.10140831e-06],
       [ 2.25500000e-05,  1.06521125e-05],
       [ 2.46000000e-05,  1.42028166e-05],
       [ 2.25500000e-05,  1.77535208e-05],
       [ 2.46000000e-05,  2.13042249e-05],
       [ 2.25500000e-05,  2.48549291e-05],
       [ 2.46000000e-05,  2.84056332e-05],
       [ 2.25500000e-05,  3.19563374e-05],
       [ 2.46000000e-05,  3.55070416e-05],
       [ 2.25500000e-05,  3.90577457e-05],
       [ 2.46000000e-05,  4.26084499e-05],
       [ 2.87000000e-05,  0.00000000e+00],
       [ 3.07500000e-05,  3.55070416e-06],
       [ 2.87000000e-05,  7.10140831e-06],
       [ 3.07500000e-05,  1.06521125e-05],
       [ 2.87000000e-05,  1.42028166e-05],
       [ 3.07500000e-05,  1.77535208e-05],
       [ 2.87000000e-05,  2.13042249e-05],
       [ 3.07500000e-05,  2.48549291e-05],
       [ 2.87000000e-05,  2.84056332e-05],
       [ 3.07500000e-05,  3.19563374e-05],
       [ 2.87000000e-05,  3.55070416e-05],
       [ 3.07500000e-05,  3.90577457e-05],
       [ 2.87000000e-05,  4.26084499e-05]])

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
size = 1000
Ryd_classical_E = []
concentration_all = []

num_sites = len(coordinates)
max_conc = len(coordinates)


concentration = np.random.randint(0, max_conc, size=size)

# Generate binary vectors and energy values
for conc in concentration:
    ones = np.random.choice(num_sites, conc, replace=False)
    x = np.zeros(num_sites, dtype='int')
    x[ones] = 1
    
    xx = np.outer(x,x)
    dm_xx = dm_inv_6_C6_eV * xx
    energy = np.sum(np.triu(dm_xx))
    Ryd_classical_E.append(energy)

import json

# Example: save to 'output.json'
data = {
    "concentration": concentration.tolist(),
    "Rydberg_classical_energy": Ryd_classical_E
}

with open(f"scripts/monte_carlo/sample_{size}_4.1um.json", "w") as f:
    json.dump(data, f, indent=4)


Delta_mu_range = np.linspace(-1.25e8,1.25e8,100,endpoint=True)
Delta_mu_range = np.array(Delta_mu_range)/eV_to_rad_s

T_all = np.linspace(1e-6,1e-4,10,endpoint=True)
# T_all = [1e-6]


# Initialize result dictionary
results = {}

for idx, T in enumerate(T_all):
    av_conc_classical = []

    for Delta_mu in Delta_mu_range:
        # Calculate energy for this Δμ
        energy_new = Ryd_classical_E + concentration * Delta_mu
        Z, pi = get_partition_function(
            energy_new, [1] * len(Ryd_classical_E), return_pi=True, T=T
        )
        av_conc = np.sum(pi * concentration) / num_sites
        av_conc_classical.append(float(av_conc))

    # Store the data under temperature T
    results[T] = {
        "Delta_mu": Delta_mu_range.tolist(),
        "avg_conc": av_conc_classical
    }

# Save to JSON
with open(f"scripts/monte_carlo/av_conc_{size}_4.1um.json", "w") as f:
    json.dump(results, f, indent=4)

        



