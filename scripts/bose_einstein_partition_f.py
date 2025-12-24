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