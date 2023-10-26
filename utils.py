import numpy as np


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
    #if len(N_index) == 0:

        #return np.tile(np.zeros(len(atom_indices[0]),dtype='int'), (len(atom_indices), 1))
    configurations = atom_indices == -1
    for index in N_index:
        configurations += atom_indices == index
    configurations = configurations.astype(int)

    unique_configurations,unique_configurations_index = np.unique(configurations,axis=0,return_index=True)
    
    return unique_configurations


def find_sic(configurations,energies,atom_indices):
    
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
            keep_energy.append(i)
    unique_energies = np.array(energies)[keep_energy]
    
    return config_unique, unique_energies, multiplicity

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