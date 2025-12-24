import numpy as np
import copy
import os
import json
from pymatgen.core.structure import Structure,Molecule, Lattice
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from braket.ahs.atom_arrangement import AtomArrangement, SiteType
from braket.ahs.driving_field import DrivingField
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation


def cut_graphene_rectangle(structure_bulk,width, height,center=True):
    
    """
    Cut a graphene structure to a specified size using a hexagonal mask.

    This function takes a bulk graphene structure and creates a smaller graphene structure
    by applying a hexagonal mask with the specified size. The resulting structure is a
    smaller graphene sheet centered on the original structure.

    Parameters:
    - structure_bulk (Structure): The bulk graphene structure to be cut.
    - size (float): The size of the hexagonal mask used for cutting.

    Returns:
    - Molecule: A smaller graphene structure after applying the hexagonal mask.

    """
    
    structure = copy.deepcopy(structure_bulk)
    
    
    expansion_matrix = 20*np.eye(3)
    expansion_matrix[2][2] = 1
    structure.make_supercell(expansion_matrix)
    if center == True:
        com = np.mean(structure.frac_coords,axis=0)
        structure.translate_sites(np.arange(structure.num_sites),-com,to_unit_cell=False)

    mask = []
    for i in range(structure.num_sites):
        x = structure.cart_coords[i][0]
        y = structure.cart_coords[i][1]
        mask.append(is_inside_rectangle(width, height,x, y))
        
    keep_site = np.where(np.array(mask) == True)
    
    species = np.array(structure.atomic_numbers)[keep_site]
    coordinates = np.array(structure.cart_coords)[keep_site]
    return Molecule(species,coordinates)


def is_inside_rectangle(width, height, x, y):
    """
    Check if a point (x, y) is inside a rectangle of specified width and height.

    This function determines whether a given point is inside or outside a rectangle
    with specified width and height.

    Parameters:
    - width (float): The width of the rectangle.
    - height (float): The height of the rectangle.
    - x (float): The x-coordinate of the point to be checked.
    - y (float): The y-coordinate of the point to be checked.

    Returns:
    - inside (bool): True if the point is inside the rectangle, False otherwise.
    """
    return 0 <= x <= width and 0 <= y <= height

#########################

#########################

# def get_all_configurations(gui_object):

#     symmops = np.array(gui_object.symmops)
#     coordinates = np.array(gui_object.atom_positions)
#     n_symmops = gui_object.n_symmops
#     atom_numbers = np.array(gui_object.atom_number)
#     lattice = gui_object.lattice
    
#     original_structure = Structure(lattice,atom_numbers,coordinates,coords_are_cartesian=True)

        
        
#     rotations = []
#     translation = []
#     for symmop in symmops:
#         rotations.append(symmop[0:3])
#         translation.append(symmop[3:4])
#     atom_indices = []
#     structures = []
#     for i in range(n_symmops):
#         atom_indices_tmp = []
#         coordinates_new = np.matmul(coordinates,rotations[i])+np.tile(translation[i], (len(atom_numbers),1))

#         #lattice_new = np.matmul(lattice,rotations[i])+np.tile(translation[i], (3,1))
#         structure_tmp = Structure(lattice,atom_numbers,coordinates_new,coords_are_cartesian=True)
#         for k,coord in enumerate(original_structure.frac_coords):
#             structure_tmp.append(original_structure.atomic_numbers[k],coord,coords_are_cartesian=False,validate_proximity=False)
#         for m in range(len(atom_numbers)):
#             index = len(atom_numbers)+m
#             for n in range(len(atom_numbers)):
#                 if structure_tmp.sites[n].is_periodic_image(structure_tmp.sites[index]):
#                     #print(m,n)
#                     atom_indices_tmp.append(n)
#                     break
#         atom_indices.append(atom_indices_tmp)

#     return atom_indices


# def build_test_train_set(structures_train,energies_train,atom_indices,N_atom):
#     all_configurations = []
#     all_energies = []
#     #energies = list(chain(*graphene_allN_cry_energy_norm[1:max_N]))
    
#     for i,structure in enumerate(structures_train):
        
#         N_index = np.where(np.array(structure.atomic_numbers)==N_atom)[0]
        
#         all_configurations.extend(build_symmetry_equivalent_configurations(atom_indices,N_index).tolist())
#         all_energies.extend([energies_train[i]]*len(build_symmetry_equivalent_configurations(atom_indices,N_index)))
#         #print(i,len(all_configurations))
#     all_configurations = np.array(all_configurations)
#     all_energies = np.array(all_energies)
    
#     return all_configurations, all_energies


# def generate_random_structures(initial_structure,atom_indices,N_atoms,new_species,N_config,DFT_config,active_sites=False,return_multiplicity=False,molecule=False):
    
#     #N_atoms: number of sites to replace
#     #N_config: number of attempts
#     #DFT_config: number of final structures generated
#     #new_species: new atomic number
#     #active_sites: sites in the structure to replace (useful for Al/GaN)
#     #atom_indices: indices obtained from get_all_configurations
#     #Returns: symmetry independent structures

#     all_structures = []

    
#     if active_sites is False:
#         num_sites = initial_structure.num_sites
#         active_sites = np.arange(num_sites)
#     else:
#         num_sites = len(active_sites)
        
    

#     # Generate a random configurations
#     descriptor_all = []
#     structures_all = []
#     config_all = []
#     config_unique = []
#     config_unique_count = []
#     n_sic = 0
#     N_attempts= 0
    
#     while n_sic < DFT_config and N_attempts <N_config:
#         N_attempts += 1
#         sites_index = np.random.choice(num_sites,N_atoms,replace=False)
#         sites = active_sites[sites_index]
#         structure_tmp = copy.deepcopy(initial_structure)
#         sec = build_symmetry_equivalent_configurations(atom_indices,sites)
# #         print(sec[0],np.lexsort(sec,axis=0))
# #         print(np.where(np.array(sec)==1))
#         # I don't need this if np.unique returns sorted arrays
# #         sic = sec[np.lexsort(sec,axis=0)][0]
#         sic = sec[0]
        
#         is_in_config_unique = any(np.array_equal(sic, existing_sic) for existing_sic in config_unique)

#         if not is_in_config_unique:  
#             config_unique.append(sic)

#             config_unique_count.append(len(sec))
#             n_sic += 1


#     final_structures = []

#     for config in config_unique:

#         N_index = np.where(config==1)[0]
#         structure_tmp = copy.deepcopy(initial_structure)
        
#         if molecule == True:
#             atom_types = np.array(structure_tmp.atomic_numbers)
#             coordinates = structure_tmp.cart_coords
            
#             atom_types[N_index] = new_species
                
#             structure_tmp = Molecule(atom_types,coordinates)
#         else:
#             for N in N_index:
#                 structure_tmp.replace(N,new_species)
#         final_structures.append(structure_tmp)
#     if return_multiplicity == True:
#         return final_structures,config_unique_count
#     else:
#         return final_structures


# def build_symmetry_equivalent_configurations(atom_indices,N_index):
    
#     """
#     Given a list of atom_indices and position of the dopants generate
#     all the equivalent structures.

#     Args:
#         atom_indices (numpy array): N_sites x N_symmops array of 
#                                     equivalent structures obtained from
#                                     get_all_configurations_no_pbc() or
#                                     get_all_configurations_pbc().
#         N_index (numpy array): N_dopants array of position of dopant atoms
#                                in the structure
        
#     Returns:
#         unique_configurations (numpy array): binary array where x_i == 1 
#                                              means there is a dopant atom in 
#                                              position i
#     """
#     if len(N_index) == 0:

#         #return np.tile(np.zeros(len(atom_indices[0]),dtype='int'), (len(atom_indices), 1))
#         return np.array([np.zeros(len(atom_indices[0]),dtype='int')]) # TEST
#     configurations = atom_indices == -1
#     for index in N_index:
#         configurations += atom_indices == index
#     configurations = configurations.astype(int)

#     unique_configurations,unique_configurations_index = np.unique(configurations,axis=0,return_index=True)
    
#     return unique_configurations


# def find_sic(configurations,atom_indices, energies=None, sort=True):
    
    """
    Find symmetry-inequivalent configurations (SIC) and return their corresponding unique energies.

    This function takes a list of binary configurations and their associated energies,
    as well as atom indices, and returns the unique SIC, unique energies, and the multiplicity
    of each unique configuration.

    Parameters:
    - configurations (numpy array 2D): A list of binary configurations.
    - energies (numpy array 1D): A list of energies (one per configuration).
    - atom_indices (numpy array 1D): Atom indices returned by get_all_configurations_no_pbc or get_all_configurations_pbc.

    Returns:
    - config_unique (List[List[int]]): A list of unique symmetry-inequivalent configurations.
    - unique_energies (List[float]): A list of unique energies corresponding to config_unique.
    - multiplicity (List[int]): A list of multiplicity values for each unique configuration.
    """
    
    config_unique = []
    multiplicity = []
    keep_energy = [] 
    for i,config in enumerate(configurations):
        
        sites = np.where(config == 1)[0] 
        sec = build_symmetry_equivalent_configurations(atom_indices,sites)
        sic = sec[0]
        is_in_config_unique = any(np.array_equal(sic, existing_sic) for existing_sic in config_unique)
        
        if not is_in_config_unique:  

            config_unique.append(sic)

            multiplicity.append(len(sec))
            if energies is not None:
                keep_energy.append(i)
    if sort == True:
        n_ones = np.sum(config_unique,axis=1)
        config_unique = (np.array(config_unique)[np.argsort(n_ones)]).tolist()
        
    
        if energies is not None:
            unique_energies = np.array(energies)[keep_energy]
            unique_energies = (unique_energies[np.argsort(n_ones)]).tolist()
        
            return config_unique, unique_energies, multiplicity
        else:
            return config_unique, multiplicity
    else:
        if energies is not None:
            return config_unique, unique_energies, multiplicity
        else:
            return config_unique, multiplicity

######################### CRYSTAL #########################

def extract_final_energy(file_path):
    """
    Extract the final energy from a CRYSTAL output file.

    Parameters:
        file_path (str): Path to the .out file.

    Returns:
        float: Final energy in atomic units (AU) or None if not found.
    """
    with open(file_path, 'r') as f:
        for line in f:
            if '* OPT END - CONVERGED' in line:
                # Extract the energy from the line
                parts = line.split()
                return float(parts[7])  # Energy is the 7th element in the line
    return None


def read_gui_file(file_path):
    """
    Reads a .gui file and creates a pymatgen Structure object.
    
    Parameters:
        file_path (str): Path to the .gui file.
    
    Returns:
        pymatgen.core.structure.Structure: Pymatgen Structure object.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse lattice matrix (lines 2-4)
    lattice = []
    for i in range(1, 4):
        lattice.append([float(x) for x in lines[i].split()])

    # Skip symmetry operators
    num_symmetry_operators = int(lines[4].strip())
    start_index = 5 + 4 * num_symmetry_operators

    # Parse number of atoms
    num_atoms = int(lines[start_index].strip())
    start_index += 1

    # Parse atomic positions and numbers
    atomic_numbers = []
    cartesian_coords = []
    for i in range(start_index, start_index + num_atoms):
        parts = lines[i].split()
        atomic_numbers.append(int(parts[0]))  # Atomic number
        cartesian_coords.append([float(x) for x in parts[1:4]])  # Cartesian coordinates

    # Create and return the pymatgen Structure object
    lattice_matrix = Lattice(lattice)
    structure = Structure(
        lattice_matrix,
        atomic_numbers,
        cartesian_coords,
        coords_are_cartesian=True
    )
    return structure


def process_dataset(train_folder, test_folder, N_sites, E_N, E_C):
    """
    Processes the CRYSTAL .out and .gui files in the train and test folders,
    and returns structures and normalized energies for both sets.

    Parameters:
        train_folder (str): Path to training set folder.
        test_folder (str): Path to test set folder.
        N_sites (int): Total number of atomic sites in the structure.
        E_N (float): Energy per N atom (reference).
        E_C (float): Energy per C atom (reference).

    Returns:
        (structures_train, energies_train, structures_test, energies_test)
    """
    def process_files(folder):
        structures = []
        energies = []
        for file in os.listdir(folder):
            if file.endswith('.out'):
                file_name = file[:-4]
                file_path = os.path.join(folder, file)

                # Extract the final energy
                energy = extract_final_energy(file_path)
                if energy is None:
                    continue

                # Read the corresponding .gui file
                gui_file_path = os.path.join(folder, file_name + '.gui')
                structure = read_gui_file(gui_file_path)

                # Count atoms
                N_N = np.sum(np.array(structure.atomic_numbers) == 7)
                N_C = N_sites - N_N

                # Normalize energy
                energy_norm = (energy - (E_C * N_C) - (E_N * N_N)) / N_sites

                structures.append(structure)
                energies.append(energy_norm)

        return structures, energies

    structures_train, energies_train = process_files(train_folder)
    structures_test, energies_test = process_files(test_folder)

    return structures_train, energies_train, structures_test, energies_test

def process_dataset_B(train_folder, test_folder, N_sites, E_N, E_C, E_B):
    """
    Processes the CRYSTAL .out and .gui files in the train and test folders,
    and returns structures and normalized energies for both sets.

    Parameters:
        train_folder (str): Path to training set folder.
        test_folder (str): Path to test set folder.
        N_sites (int): Total number of atomic sites in the structure.
        E_N (float): Energy per N atom (reference).
        E_C (float): Energy per C atom (reference).

    Returns:
        (structures_train, energies_train, structures_test, energies_test)
    """
    def process_files(folder):
        structures = []
        energies = []
        for file in os.listdir(folder):
            if file.endswith('.out'):
                file_name = file[:-4]
                file_path = os.path.join(folder, file)

                # Extract the final energy
                energy = extract_final_energy(file_path)
                if energy is None:
                    continue

                # Read the corresponding .gui file
                gui_file_path = os.path.join(folder, file_name + '.gui')
                structure = read_gui_file(gui_file_path)

                # Count atoms
                N_N = np.sum(np.array(structure.atomic_numbers) == 7)
                N_B = np.sum(np.array(structure.atomic_numbers) == 5)
                N_C = N_sites - N_N - N_B

                # Normalize energy
                energy_norm = (energy - (E_C * N_C) - (E_N * N_N) - (E_B * N_B)) / N_sites

                structures.append(structure)
                energies.append(energy_norm)

        return structures, energies

    structures_train, energies_train = process_files(train_folder)
    structures_test, energies_test = process_files(test_folder)

    return structures_train, energies_train, structures_test, energies_test


###################### END CRYSTAL ########################




def get_partition_function(energy, multiplicity, T=298.15, return_pi=True, N_N=0, N_potential=0.):

    """
    Calculate the partition function and probabilities for different energy levels.
    
    Args:
        energy (np.ndarray): Array of energy levels.
        multiplicity (np.ndarray): Array of corresponding multiplicities.
        T (float, optional): Temperature in Kelvin. Default is 298.15 K.
        return_pi (bool, optional): Flag to return probabilities. Default is True.
        N_N (float, optional): Number of N particles. Default is 0.
        N_potential (float, optional): Potential for N particles. Default is 0.

    Returns:
        tuple or float: If return_pi is True, returns a tuple containing partition function and probabilities.
                        Otherwise, returns the partition function.
    """
    k_b = 8.617333262145e-05  # Boltzmann constant in eV/K
    
    energy = np.array(energy)
    multiplicity = np.array(multiplicity)
    p_i = multiplicity * np.exp((-energy + (N_N * N_potential)) / (k_b * T))
    pf = np.sum(p_i)
    
    p_i /= pf
    
    if return_pi:
        return pf, p_i
    else:
        return pf
  
    
def build_adjacency_matrix(structure, max_neigh = 1, diagonal_terms = False, triu = False):

    # structure = pymatgen Structure object
    
    num_sites = structure.num_sites
    distance_matrix_pbc = np.round(structure.distance_matrix,5)

    distance_matrix = np.zeros((num_sites,num_sites),float)
    
    shells = np.unique(distance_matrix_pbc[0])
    
    for i,s in enumerate(shells[0:max_neigh+1]):
        row_index = np.where(distance_matrix_pbc == s)[0]
        col_index = np.where(distance_matrix_pbc == s)[1]
        distance_matrix[row_index,col_index] = i
    
    if triu == True:
        distance_matrix = np.triu(distance_matrix,0)
    
    if diagonal_terms == True:
        np.fill_diagonal(distance_matrix,[1]*num_sites)
    
    return distance_matrix


def extract_pre_post_sequences(json_file):
    """
    Extracts pre_sequence and post_sequence lists separately from a given JSON structure.

    Parameters:
    json_data (dict): JSON dictionary containing "shot_outputs" with "pre_sequence" and "post_sequence".

    Returns:
    tuple: Two lists - one for all pre_sequences and one for all post_sequences.
    """

    with open(json_file, 'r') as f:
        json_data = json.load(f)

    pre_sequences = []
    post_sequences = []

    for shot in json_data.get("shot_outputs", []):
        pre_sequences.append(shot.get("pre_sequence", []))
        post_sequences.append(shot.get("post_sequence", []))
    
    return pre_sequences, post_sequences

# Extract pre and post sequences separately

def read_all_sequences_from_folder(folder_path):
    """
    Reads all .json files in the given folder and extracts combined pre- and post-sequences.

    Parameters:
        folder_path (str): Path to folder containing JSON files.

    Returns:
        tuple: Two flat lists - concatenated pre_sequences and post_sequences from all files.
    """
    all_pre = []
    all_post = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.json') and not filename.startswith('input'):
            full_path = os.path.join(folder_path, filename)
            pre, post = extract_pre_post_sequences(full_path)
            # Flatten and concatenate
            for p in pre:
                all_pre.append(p)
            for q in post:
                all_post.append(q)

    return all_pre, all_post


def get_hamming(preseqs,postseqs): 
    
    preseqs = np.array(preseqs)
    postseqs = np.array(postseqs)
    
    num_atoms = preseqs.shape[1]
    
    keep_config = np.where(np.sum(preseqs,axis=1) == num_atoms)[0]

    hamming = num_atoms-np.sum(postseqs[keep_config],axis=1)
    len_hamming = len(hamming)
    hamming_average = np.average(hamming)
    hamming_std = np.std(hamming)
    hamming_max = np.max(hamming)
    hamming_min = np.min(hamming)
    
    hamming_unique, unique_index = np.unique(hamming,return_counts=True)
#     print(np.average(np.sum(postseqs[keep_config],axis=1)))
    return len_hamming, hamming_average, hamming_std,hamming_max,hamming_min,hamming_unique, unique_index


def stack_chunks_vertically(arr, chunk_size=28):
    """
    Split a (N, M) array into M/chunk_size horizontal chunks,
    and stack them vertically to shape (N * num_chunks, chunk_size).
    
    Parameters:
        arr (ndarray): Input array of shape (N, M)
        chunk_size (int): Width of each chunk (default is 28)
    
    Returns:
        stacked (ndarray): Output array of shape (N * num_chunks, chunk_size)
    """
    if arr.shape[1] % chunk_size != 0:
        raise ValueError("Array width must be divisible by chunk_size.")

    num_chunks = arr.shape[1] // chunk_size
    chunks = [arr[:, i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    stacked = np.vstack(chunks)
    return stacked


def extract_detuning_values(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                values = (
                    data.get('effective_hamiltonian', {})
                        .get('rydberg', {})
                        .get('detuning', {})
                        .get('global_', {})
                        .get('values')
                )

                if values is not None:
                    return values  # Return as soon as found

            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return None  # Nothing found


########################### BUILD MODEL ###############################

def build_H_ryd(structure, binary_occ, energies, neighbor_counts, scaling_factors):
    """
    Build a Rydberg Hamiltonian matrix using machine learning based on a structure and training data.

    Parameters:
        structure (pymatgen.core.structure.Structure): Reference structure.
        binary_occ (list of np.ndarray): List of training configurations.
        energies (np.ndarray): Training target energies.
        neighbor_counts (list of int): Number of neighbors to include in each shell.
        scaling_factors (list of float): Scaling factors for each shell.

    Returns:
        np.ndarray: QUBO matrix.
    """
    # Step 1: Filter the distance matrix to include only specified neighbor shells
    distance_matrix = np.round(structure.distance_matrix, 5)
    num_sites = structure.num_sites

    # Sort the distance matrix and get sorted indices
    sorted_indices = np.argsort(distance_matrix, axis=1)
    sorted_distances = np.sort(distance_matrix, axis=1)

    # Initialize filtered distance matrix
    distance_matrix_filter = np.zeros_like(distance_matrix, dtype=float)

    # Assign scaling factors based on the number of neighbors per shell
    for i, (site_distances, site_indices) in enumerate(zip(sorted_distances, sorted_indices)):
        start_idx = 0
        for shell_idx, num_neighbors in enumerate(neighbor_counts):
            # Determine the indices for this shell
            end_idx = start_idx + num_neighbors
            neighbors = site_indices[start_idx:end_idx]  # Select neighbors
            distance_matrix_filter[i, neighbors] = scaling_factors[shell_idx]
            start_idx = end_idx

    # Upper triangular filtering and diagonal filling
    distance_matrix_filter = np.triu(distance_matrix_filter, k=0)
    np.fill_diagonal(distance_matrix_filter, 1)
 
    # Step 2: Build the descriptor for the training set
    upper_tri_indices = np.where(distance_matrix_filter != 0)
    descriptor = []

    for config in binary_occ:
        # Compute the outer product of the configuration vector
        matrix = np.outer(config, config)

        # Mask to select off-diagonal elements where matrix == 1
        mask = (matrix == 1) & (np.triu(np.ones_like(matrix), k=1) == 1)
        
        # Apply the mask to the distance_matrix_filter
        selected_elements = distance_matrix_filter[mask]

        # Descriptor includes the number of occupied sites and sum of selected elements
        num_occupied_sites = np.sum(config)
        descriptor.append([num_occupied_sites, np.sum(selected_elements)])

    descriptor = np.array(descriptor)

    # Step 3: Train a linear regression model
    reg = LinearRegression()
    reg.fit(descriptor, energies)

    # Print R^2 score
    print(f"R²: {reg.score(descriptor, energies):.6f}")


    return reg.coef_

def calculate_Ryd_energy_pbc(structure, X_test, reg_coef, neighbor_count, scaling_factors):
    """
    Calculate the energy for test configurations using the regression coefficients and neighbor contributions.

    Parameters:
        structure (pymatgen.core.structure.Structure): Reference structure.
        X_test (list of np.ndarray): Test configurations.
        reg_coef (np.ndarray): Regression coefficients from the linear model.
        neighbor_counts (list of int): Number of neighbors to include in each shell.
        scaling_factors (list of float): Scaling factors for each shell.

    Returns:
        np.ndarray: Calculated energies for the test configurations.
    """
    # Step 1: Filter the distance matrix to include only specified neighbor shells
    distance_matrix = np.round(structure.distance_matrix, 5)
    num_sites = structure.num_sites

    # Sort the distance matrix and get sorted indices
    sorted_indices = np.argsort(distance_matrix, axis=1)
    sorted_distances = np.sort(distance_matrix, axis=1)

    # Initialize filtered distance matrix
    distance_matrix_filter = np.zeros_like(distance_matrix, dtype=float)

    # Assign scaling factors based on the number of neighbors per shell
    for i, (site_distances, site_indices) in enumerate(zip(sorted_distances, sorted_indices)):
        start_idx = 0
        for shell_idx, num_neighbors in enumerate(neighbor_count):
            # Determine the indices for this shell
            end_idx = start_idx + num_neighbors
            neighbors = site_indices[start_idx:end_idx]  # Select neighbors
            distance_matrix_filter[i, neighbors] = scaling_factors[shell_idx]
            start_idx = end_idx

    # Upper triangular filtering and diagonal filling
    distance_matrix_filter = np.triu(distance_matrix_filter, k=0)
    np.fill_diagonal(distance_matrix_filter, 1)

    # Step 2: Calculate the energy contributions for each test configuration
    energies = []
    for config in X_test:
        # Compute the outer product of the configuration vector
        matrix = np.outer(config, config)

        # Mask to select off-diagonal elements where matrix == 1
        mask = (matrix == 1) & (np.triu(np.ones_like(matrix), k=1) == 1)

        # Apply the mask to the distance_matrix_filter
        selected_elements = distance_matrix_filter[mask]

        # Calculate the energy contribution
        num_occupied_sites = np.sum(config)  # Contribution from the number of occupied sites
        pairwise_contribution = np.sum(selected_elements * reg_coef[1])  # Contribution from neighbors

        # Add all contributions
        energy = reg_coef[0] * num_occupied_sites + pairwise_contribution
        energies.append(energy)

    return np.array(energies)


def calculate_Ryd_energy_mol(structure, X_test, reg_coef, scaling_factors):

    """
    Calculate the energy for test configurations using the regression coefficients and neighbor contributions.
    For the non periodic structure
    Parameters:
        structure (pymatgen.core.structure.Structure): Reference structure.
        X_test (list of np.ndarray): Test configurations.
        neighbor_counts (list of int): Number of neighbors to include in each shell.
        scaling_factors (list of float): Scaling factors for each shell.

    Returns:
        np.ndarray: Calculated energies for the test configurations.
    """
    # Step 1: Filter the distance matrix to include only specified neighbor shells
    structure = Molecule(structure.atomic_numbers,structure.cart_coords)
    distance_matrix = np.round(structure.distance_matrix, 5)
    num_sites = structure.num_sites

    num_sites = structure.num_sites
    distance_matrix_mol = np.round(structure.distance_matrix, 5)

    # Initialize an empty distance matrix
    distance_matrix_filter = np.zeros((num_sites, num_sites), dtype=float)

    # Get unique distance shells and limit to the maximum number of neighbors
    shells = np.unique(distance_matrix_mol)
    max_neigh = len(scaling_factors)

    # Assign scaled values to the distance matrix based on shells
    for i, shell in enumerate(shells[:max_neigh]):
        # print(i,shell,scaling_factors[i])
        # Find row and column indices where the distance matches the current shell
        row_indices, col_indices = np.where(distance_matrix_mol == shell)
        
        # Apply the scaling factor to the corresponding entries
        distance_matrix_filter[row_indices, col_indices] = scaling_factors[i]

    # # Sort the distance matrix and get sorted indices
    # sorted_indices = np.argsort(distance_matrix, axis=1)
    # sorted_distances = np.sort(distance_matrix, axis=1)

    # # Initialize filtered distance matrix
    # distance_matrix_filter = np.zeros_like(distance_matrix, dtype=float)

    # # Assign scaling factors based on the number of neighbors per shell
    # for i, (site_distances, site_indices) in enumerate(zip(sorted_distances, sorted_indices)):
    #     start_idx = 0
    #     for shell_idx, num_neighbors in enumerate(neighbor_count):
    #         # Determine the indices for this shell
    #         end_idx = start_idx + num_neighbors
    #         neighbors = site_indices[start_idx:end_idx]  # Select neighbors
    #         distance_matrix_filter[i, neighbors] = scaling_factors[shell_idx]
    #         start_idx = end_idx

    # Upper triangular filtering and diagonal filling
    distance_matrix_filter = np.triu(distance_matrix_filter, k=0)
    np.fill_diagonal(distance_matrix_filter, 1)
    
    # Step 2: Calculate the energy contributions for each test configuration
    energies = []
    for config in X_test:
        # Compute the outer product of the configuration vector
        matrix = np.outer(config, config)

        # Mask to select off-diagonal elements where matrix == 1
        mask = (matrix == 1) & (np.triu(np.ones_like(matrix), k=1) == 1)

        # Apply the mask to the distance_matrix_filter
        selected_elements = distance_matrix_filter[mask]

        # Calculate the energy contribution
        num_occupied_sites = np.sum(config)  # Contribution from the number of occupied sites
        pairwise_contribution = np.sum(selected_elements * reg_coef[1])  # Contribution from neighbors

        # Add all contributions
        energy = reg_coef[0] * num_occupied_sites + pairwise_contribution
        energies.append(energy)

    return np.array(energies)

######################### END BUILD MODEL ##############################

######################### BUILD ATOMS ARRANGEMENT #######################


def show_register(
    register: AtomArrangement,
    blockade_radius: float = 0.0,
    what_to_draw: str = "bond",
    show_atom_index: bool = True,
):
    """Plot the given register

    Args:
        register (AtomArrangement): A given register
        blockade_radius (float): The blockade radius for the register. Default is 0
        what_to_draw (str): Either "bond" or "circle" to indicate the blockade region.
            Default is "bond"
        show_atom_index (bool): Whether showing the indices of the atoms. Default is True

    """
    filled_sites = [site.coordinate for site in register if site.site_type == SiteType.FILLED]
    empty_sites = [site.coordinate for site in register if site.site_type == SiteType.VACANT]

    plt.figure(figsize=(7, 7))
    if filled_sites:
        plt.plot(
            np.array(filled_sites)[:, 0], np.array(filled_sites)[:, 1], "r.", ms=15, label="filled"
        )
    if empty_sites:
        plt.plot(
            np.array(empty_sites)[:, 0], np.array(empty_sites)[:, 1], "k.", ms=5, label="empty"
        )
    plt.legend(bbox_to_anchor=(1.1, 1.05))

    if show_atom_index:
        for idx, site in enumerate(register):
            plt.text(*site.coordinate, f"  {idx}", fontsize=12)

    if blockade_radius > 0 and what_to_draw == "bond":
        for i in range(len(filled_sites)):
            for j in range(i + 1, len(filled_sites)):
                dist = np.linalg.norm(np.array(filled_sites[i]) - np.array(filled_sites[j]))
                if dist <= blockade_radius:
                    plt.plot(
                        [filled_sites[i][0], filled_sites[j][0]],
                        [filled_sites[i][1], filled_sites[j][1]],
                        "b",
                    )

    if blockade_radius > 0 and what_to_draw == "circle":
        for site in filled_sites:
            plt.gca().add_patch(
                plt.Circle((site[0], site[1]), blockade_radius / 2, color="b", alpha=0.3)
            )
        plt.gca().set_aspect(1)
    plt.show()


def get_coords(num_x=6, num_y=6, a=1):
    """Get the coordinates of a graphene with certain size

    Args:
        num_x (Int): The number of columns
        num_y (Int): The number of pairs (2*num_y is the length of the columns minus 1)
        a (float): The lattice constant

    Returns:
        The coordinates of the graphene as an np array
    """    
    if np.mod(num_x, 2) != 0:
        raise ValueError("num_x should be an even number")
    
    pair_1 = np.array([[0, 0], [-1/2, np.sqrt(3)/2]]) * a
    chain_1 = []
    for y in range(num_y):
        chain_1 += [atom + np.array([0, np.sqrt(3) * a * y]) for atom in pair_1]
    chain_1.append([0, np.sqrt(3) * a * num_y])
        
    pair_2 = np.array([[1, 0], [3/2, np.sqrt(3)/2]]) * a
    chain_2 = []
    for y in range(num_y):
        chain_2 += [atom + np.array([0, np.sqrt(3) * a * y]) for atom in pair_2]
    chain_2.append([1, np.sqrt(3) * a * num_y])
    
    coords = []
    for x in range(num_x):
        if np.mod(x, 2) == 0:
            coords += [item + np.array([[3 * a * np.floor(x/2), 0]]) for item in chain_1]

        if np.mod(x, 2) == 1:
            coords += [item + np.array([[3 * a * np.floor(x/2), 0]]) for item in chain_2]
            
    coords = [item[0] for item in coords]

    return np.array(coords)

def get_register(separation, coords, show_atom_arrangement=True):
    """Get the register of a graphene for a given coordinates and scale (separation)

    Args:
        separation (float): The scale of the graphene or re-scaled lattice constant
        coords (np.array): The coordinates of the graphene
        show_atom_arrangement (bool): If plot the graphene

    Returns:
        The register for the graphene
    """  
    
    N_sites = coords.shape[0]
    coordinates_atoms = coords*separation
    register = AtomArrangement()
    
    for k in range(N_sites):
        register.add(coordinates_atoms[k])
        
    if show_atom_arrangement:
        show_register(register, show_atom_index=True, blockade_radius=1.01 * separation) # Note that blockade_radius is redefined

    return register


