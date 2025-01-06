# This is here temporarily while I write the same functions in 
# a standalone package

import copy
import numpy as np

def build_symmetry_equivalent_configurations(atom_indices, N_index):
    """
    Build symmetry-equivalent configurations for given atom indices and selected sites.

    Parameters:
    - atom_indices: Array of atom indices from which configurations are derived.
    - N_index: Indices of the sites to consider for symmetry equivalence.

    Returns:
    - unique_configurations: Array of unique symmetry-equivalent configurations.
    """

    # Handle the case where no sites are selected
    if len(N_index) == 0:
        return np.array([np.zeros(len(atom_indices[0]), dtype='int')])  # Return an array of zeros

    # Initialize configurations based on atom indices
    configurations = (atom_indices == -1)  # Mark invalid or non-existing sites

    # Update configurations with the selected site indices
    for index in N_index:
        configurations += (atom_indices == index)
    configurations = configurations.astype(int)  # Convert to integers

    # Extract unique configurations
    unique_configurations, unique_configurations_index = np.unique(
        configurations, axis=0, return_index=True
    )

    return unique_configurations


def generate_random_structures(
    initial_structure,
    atom_indices,
    N_atoms,
    new_species,
    N_config,
    DFT_config,
    active_sites=False,
    return_multiplicity=False
):
    """
    Generate symmetry-independent structures based on random configurations.

    Parameters:
    - initial_structure: The starting structure object.
    - atom_indices: Indices obtained from `get_all_configurations`.
    - N_atoms: Number of sites to replace.
    - N_config: Number of attempts for generating configurations.
    - DFT_config: Number of final structures to generate.
    - new_species: Atomic number for the new species.
    - active_sites: List of active sites to replace (optional, default: False for all sites).
    - return_multiplicity: Whether to return the multiplicity of configurations (default: False).

    Returns:
    - final_structures: A list of generated structures.
    - config_unique_count (optional): List of counts for each unique configuration.
    """
    # Initialize variables
    all_structures = []
    descriptor_all = []
    structures_all = []
    config_all = []
    config_unique = []
    config_unique_count = []
    n_sic = 0
    N_attempts = 0

    # Determine active sites
    if not active_sites:
        num_sites = initial_structure.num_sites
        active_sites = np.arange(num_sites)
    else:
        num_sites = len(active_sites)

    # Generate random configurations
    while n_sic < DFT_config and N_attempts < N_config:
        N_attempts += 1

        # Randomly select sites to replace
        sites_index = np.random.choice(num_sites, N_atoms, replace=False)
        sites = active_sites[sites_index]
        structure_tmp = copy.deepcopy(initial_structure)

        # Build symmetry-equivalent configurations
        sec = build_symmetry_equivalent_configurations(atom_indices, sites)
        sic = sec[0]  # Symmetry-independent configuration

        # Check if the configuration is unique
        is_in_config_unique = any(np.array_equal(sic, existing_sic) for existing_sic in config_unique)
        if not is_in_config_unique:
            config_unique.append(sic)
            config_unique_count.append(len(sec))
            n_sic += 1

    # Generate final structures based on unique configurations
    final_structures = []
    for config in config_unique:
        structure_tmp = copy.deepcopy(initial_structure)
        N_index = np.where(config == 1)[0]
        for N in N_index:
            structure_tmp.replace(N, new_species)
        final_structures.append(structure_tmp)

    # Return results
    if return_multiplicity:
        return final_structures, config_unique_count
    
    return final_structures

