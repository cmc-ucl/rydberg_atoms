import numpy as np
import copy
from pymatgen.core.structure import Structure



def get_all_configurations(gui_object):

    symmops = np.array(gui_object.symmops)
    coordinates = np.array(gui_object.atom_positions)
    n_symmops = gui_object.n_symmops
    atom_numbers = np.array(gui_object.atom_number)
    lattice = gui_object.lattice
    
    original_structure = Structure(lattice,atom_numbers,coordinates,coords_are_cartesian=True)

        
        
    rotations = []
    translation = []
    for symmop in symmops:
        rotations.append(symmop[0:3])
        translation.append(symmop[3:4])
    atom_indices = []
    structures = []
    for i in range(n_symmops):
        atom_indices_tmp = []
        coordinates_new = np.matmul(coordinates,rotations[i])+np.tile(translation[i], (len(atom_numbers),1))

        #lattice_new = np.matmul(lattice,rotations[i])+np.tile(translation[i], (3,1))
        structure_tmp = Structure(lattice,atom_numbers,coordinates_new,coords_are_cartesian=True)
        for k,coord in enumerate(original_structure.frac_coords):
            structure_tmp.append(original_structure.atomic_numbers[k],coord,coords_are_cartesian=False,validate_proximity=False)
        for m in range(len(atom_numbers)):
            index = len(atom_numbers)+m
            for n in range(len(atom_numbers)):
                if structure_tmp.sites[n].is_periodic_image(structure_tmp.sites[index]):
                    #print(m,n)
                    atom_indices_tmp.append(n)
                    break
        atom_indices.append(atom_indices_tmp)

    return atom_indices


def build_test_train_set(structures_train,energies_train,atom_indices,N_atom):
    all_configurations = []
    all_energies = []
    #energies = list(chain(*graphene_allN_cry_energy_norm[1:max_N]))
    
    for i,structure in enumerate(structures_train):
        
        N_index = np.where(np.array(structure.atomic_numbers)==N_atom)[0]
        
        all_configurations.extend(build_symmetry_equivalent_configurations(atom_indices,N_index).tolist())
        all_energies.extend([energies_train[i]]*len(build_symmetry_equivalent_configurations(atom_indices,N_index)))
        #print(i,len(all_configurations))
    all_configurations = np.array(all_configurations)
    all_energies = np.array(all_energies)
    
    return all_configurations, all_energies


def generate_random_structures(initial_structure,atom_indices,N_atoms,new_species,N_config,DFT_config,active_sites=False,return_multiplicity=False,molecule=False):
    
    #N_atoms: number of sites to replace
    #N_config: number of attempts
    #DFT_config: number of final structures generated
    #new_species: new atomic number
    #active_sites: sites in the structure to replace (useful for Al/GaN)
    #atom_indices: indices obtained from get_all_configurations
    #Returns: symmetry independent structures

    all_structures = []

    
    if active_sites is False:
        num_sites = initial_structure.num_sites
        active_sites = np.arange(num_sites)
    else:
        num_sites = len(active_sites)
        
    

    # Generate a random configurations
    descriptor_all = []
    structures_all = []
    config_all = []
    config_unique = []
    config_unique_count = []
    n_sic = 0
    N_attempts= 0
    
    while n_sic < DFT_config and N_attempts <N_config:
        N_attempts += 1
        sites_index = np.random.choice(num_sites,N_atoms,replace=False)
        sites = active_sites[sites_index]
        structure_tmp = copy.deepcopy(initial_structure)
        sec = build_symmetry_equivalent_configurations(atom_indices,sites)
#         print(sec[0],np.lexsort(sec,axis=0))
#         print(np.where(np.array(sec)==1))
        # I don't need this if np.unique returns sorted arrays
#         sic = sec[np.lexsort(sec,axis=0)][0]
        sic = sec[0]
        
        is_in_config_unique = any(np.array_equal(sic, existing_sic) for existing_sic in config_unique)

        if not is_in_config_unique:  
            config_unique.append(sic)

            config_unique_count.append(len(sec))
            n_sic += 1


    final_structures = []

    for config in config_unique:

        N_index = np.where(config==1)[0]
        structure_tmp = copy.deepcopy(initial_structure)
        
        if molecule == True:
            atom_types = np.array(structure_tmp.atomic_numbers)
            coordinates = structure_tmp.cart_coords
            
            atom_types[N_index] = new_species
                
            structure_tmp = Molecule(atom_types,coordinates)
        else:
            for N in N_index:
                structure_tmp.replace(N,new_species)
        final_structures.append(structure_tmp)
    if return_multiplicity == True:
        return final_structures,config_unique_count
    else:
        return final_structures


def build_symmetry_equivalent_configurations(atom_indices,N_index):
    
    """
    Given a list of atom_indices and position of the dopants generate
    all the equivalent structures.

    Args:
        atom_indices (numpy array): N_sites x N_symmops array of 
                                    equivalent structures obtained from
                                    get_all_configurations_no_pbc() or
                                    get_all_configurations_pbc().
        N_index (numpy array): N_dopants array of position of dopant atoms
                               in the structure
        
    Returns:
        unique_configurations (numpy array): binary array where x_i == 1 
                                             means there is a dopant atom in 
                                             position i
    """
    if len(N_index) == 0:

        #return np.tile(np.zeros(len(atom_indices[0]),dtype='int'), (len(atom_indices), 1))
        return np.array([np.zeros(len(atom_indices[0]),dtype='int')]) # TEST
    configurations = atom_indices == -1
    for index in N_index:
        configurations += atom_indices == index
    configurations = configurations.astype(int)

    unique_configurations,unique_configurations_index = np.unique(configurations,axis=0,return_index=True)
    
    return unique_configurations


def find_sic(configurations,atom_indices, energies=None, sort=True):
    
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