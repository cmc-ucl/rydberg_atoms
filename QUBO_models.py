# This is here temporarily while I write the same functions in 
# a standalone package

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def build_ml_qubo(structure, structures, energies, max_neigh=1):
    """
    Build a QUBO matrix using machine learning based on a structure and training data.

    Parameters:
        structure (pymatgen.core.structure.Structure): Reference structure.
        structures (list of np.ndarray): List of training configurations.
        energies (np.ndarray): Training target energies.
        max_neigh (int): Maximum number of neighbor shells to include in the descriptor.

    Returns:
        np.ndarray: QUBO matrix.
    """
    # Step 1: Filter the distance matrix to include only specified neighbor shells
    distance_matrix = np.round(structure.distance_matrix, 5)
    shells = np.unique(distance_matrix)
    num_sites = structure.num_sites

    # Initialize filtered distance matrix
    distance_matrix_filter = np.zeros((num_sites, num_sites), dtype=int)

    for i, shell in enumerate(shells[:max_neigh + 1]):
        row_indices, col_indices = np.where(distance_matrix == shell)

        distance_matrix_filter[row_indices, col_indices] = i

    # Upper triangular filtering and diagonal filling
    distance_matrix_filter = np.triu(distance_matrix_filter, k=0)
    np.fill_diagonal(distance_matrix_filter, 1)

    # Step 2: Build the descriptor for the training set
    upper_tri_indices = np.where(distance_matrix_filter != 0)
    descriptor = []

    for config in structures:
        matrix = np.outer(config, config)
        upper_tri_elements = matrix[upper_tri_indices]
        descriptor.append(upper_tri_elements)

    descriptor = np.array(descriptor)

    # Step 3: Train a linear regression model
    reg = LinearRegression()
    reg.fit(descriptor, energies)

    # Print R^2 score
    print(f"R²: {reg.score(descriptor, energies):.4f}")

    # Step 4: Build the QUBO matrix
    Q = np.zeros((num_sites, num_sites))
    Q[upper_tri_indices] = reg.coef_

    return Q


def classical_energy(x, q):
    """
    Compute the classical energy for a binary vector and QUBO matrix.

    Parameters:
        x (np.ndarray): Binary vector.
        q (np.ndarray): QUBO matrix.

    Returns:
        float: Classical energy.
    """
    return np.dot(x, np.dot(q, x))


def get_qubo_energies(Q, all_configurations):
    """
    Calculate the energies of configurations using a QUBO matrix.

    Parameters:
        Q (np.ndarray): QUBO matrix.
        all_configurations (list of np.ndarray): List of configurations to evaluate.

    Returns:
        list: Predicted energies for each configuration.
    """
    predicted_energies = [
        classical_energy(config, Q) for config in all_configurations
    ]
    return predicted_energies

def test_qubo_energies(y_pred, y_dft):
    """
    Calculate the mean squared error (MSE) between predicted and DFT energies.

    Parameters:
        y_pred (list or np.ndarray): Predicted energies.
        y_dft (list or np.ndarray): DFT reference energies.

    Returns:
        float: Mean squared error.
    """
    return mean_squared_error(y_dft, y_pred)

