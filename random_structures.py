# This is here temporarily while I write the same functions in 
# a standalone package

import copy
import numpy as np

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, PointGroupAnalyzer
from pymatgen.core.structure import Structure, Molecule


def build_symmetry_equivalent_configurations(atom_indices, N_index):
    """
    WIP: does this work for more than one atom?
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


def build_test_train_set(structures_train, energies_train, atom_indices, N_atom):
    """
    WIP: add more than one substitution option.
    Build the test and training dataset by generating symmetry-equivalent configurations.

    Parameters:
        structures_train (list): List of training structures.
        energies_train (list): List of energies corresponding to the training structures.
        atom_indices (np.ndarray): Array of atom indices for symmetry generation.
        N_atom (int): Atomic number of the target atom.

    Returns:
        tuple: 
            - all_configurations (np.ndarray): Array of symmetry-equivalent configurations.
            - all_energies (np.ndarray): Array of corresponding energies.
    """
    all_configurations = []
    all_energies = []

    for i, structure in enumerate(structures_train):
        # Identify indices of atoms with the target atomic number
        N_index = np.where(np.array(structure.atomic_numbers) == N_atom)[0]

        # Generate symmetry-equivalent configurations
        symmetry_configs = build_symmetry_equivalent_configurations(atom_indices, N_index).tolist()
        
        # Extend configurations and energies
        all_configurations.extend(symmetry_configs)
        all_energies.extend([energies_train[i]] * len(symmetry_configs))

    # Convert to NumPy arrays
    all_configurations = np.array(all_configurations)
    all_energies = np.array(all_energies)

    return all_configurations, all_energies


def generate_random_structures(
    initial_structure,
    atom_indices,
    N_atoms,
    new_species,
    N_config,
    DFT_config,
    active_sites=None,
    return_multiplicity=False
):
    """
    Generate symmetry-independent structures based on random configurations. 
    WIP: This works for one substitution only, extend it to more (not just binary)

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
    if active_sites is None:
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


def get_all_configurations(structure_pmg, prec=6):
    """
    Generate all symmetry-equivalent atomic configurations for a given structure.

    Parameters:
        structure_pmg (pymatgen.core.structure.Structure): Input structure.
        prec (int): Precision for rounding fractional coordinates.

    Returns:
        np.ndarray: Matrix where each row corresponds to the atom indices for
                    a symmetry-equivalent configuration.
    """
    # Get symmetry operations
    symmops = SpacegroupAnalyzer(structure_pmg).get_symmetry_operations()
    n_symmops = len(symmops)

    # Extract structure data
    coordinates = np.round(np.array(structure_pmg.frac_coords), prec)
    atom_numbers = np.array(structure_pmg.atomic_numbers)
    lattice = structure_pmg.lattice.matrix

    # Prepare original structure for appending atoms
    original_structure_pmg = copy.deepcopy(structure_pmg)

    # Initialize atom indices matrix
    atom_indices = np.ones((n_symmops, structure_pmg.num_sites), dtype=int) * -1

    # Loop over symmetry operations
    for i, symmop in enumerate(symmops):
        # Apply symmetry operation to fractional coordinates
        transformed_coords = [
            np.round(symmop.operate(coord), prec) for coord in coordinates
        ]
        
        # Create a temporary structure with transformed coordinates
        structure_tmp = Structure(
            lattice, atom_numbers, transformed_coords, coords_are_cartesian=False, to_unit_cell=False
        )

        # Append original atoms to avoid missing sites
        for atomic_number, coord in zip(
            original_structure_pmg.atomic_numbers, original_structure_pmg.frac_coords
        ):
            structure_tmp.append(
                atomic_number, coord, coords_are_cartesian=False, validate_proximity=False
            )

        # Map transformed sites to original indices
        for m in range(len(atom_numbers)):
            index = len(atom_numbers) + m
            for n in range(len(atom_numbers)):
                if structure_tmp.sites[n].is_periodic_image(
                    structure_tmp.sites[index], tolerance=0.001
                ):
                    atom_indices[i, m] = n
                    break

    return atom_indices


def get_all_configurations_molecule(mol: Molecule, prec=6, tol=1e-3):
    """
    Generate all symmetry-equivalent atomic configurations for a given molecule.

    Parameters:
        mol (pymatgen.core.structure.Molecule): Input molecule.
        prec (int): Precision for rounding coordinates.
        tol (float): Distance tolerance for identifying equivalent atoms.

    Returns:
        np.ndarray: Matrix where each row corresponds to the atom indices for
                    a symmetry-equivalent configuration.
    """
    mol = mol.get_centered_molecule()
    pga = PointGroupAnalyzer(mol)
    symmops = pga.get_symmetry_operations()

    coords = np.round(np.array(mol.cart_coords), prec)
    atomic_numbers = np.array(mol.atomic_numbers)
    num_atoms = len(coords)

    atom_indices = np.ones((len(symmops), num_atoms), dtype=int) * -1

    for i, symmop in enumerate(symmops):
        transformed_coords = np.round(np.array([symmop.operate(c) for c in coords]), prec)
        used = set()
        for j, trans_coord in enumerate(transformed_coords):
            for k, orig_coord in enumerate(coords):
                if k in used:
                    continue
                if atomic_numbers[j] == atomic_numbers[k] and np.linalg.norm(trans_coord - orig_coord) < tol:
                    atom_indices[i, j] = k
                    used.add(k)
                    break

    return atom_indices

