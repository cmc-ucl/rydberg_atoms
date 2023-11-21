import numpy as np
from joblib import Parallel, delayed

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

        

# def find_sic(configurations,energies,atom_indices):
    
#     """
#     Find symmetry-inequivalent configurations (SIC) and return their corresponding unique energies.

#     This function takes a list of binary configurations and their associated energies,
#     as well as atom indices, and returns the unique SIC, unique energies, and the multiplicity
#     of each unique configuration.

#     Parameters:
#     - configurations (numpy array 2D): A list of binary configurations.
#     - energies (numpy array 1D): A list of energies (one per configuration).
#     - atom_indices (numpy array 1D): Atom indices returned by get_all_configurations_no_pbc or get_all_configurations_pbc.

#     Returns:
#     - config_unique (List[List[int]]): A list of unique symmetry-inequivalent configurations.
#     - unique_energies (List[float]): A list of unique energies corresponding to config_unique.
#     - multiplicity (List[int]): A list of multiplicity values for each unique configuration.
#     """
    
#     config_unique = []
#     multiplicity = []
#     keep_energy = [] 
#     for i,config in enumerate(configurations):
        
#         sites = np.where(config == 1)[0] 
#         sec = build_symmetry_equivalent_configurations(atom_indices,sites)
#         sic = sec[0]
#         is_in_config_unique = any(np.array_equal(sic, existing_sic) for existing_sic in config_unique)
        
#         if not is_in_config_unique:  

#             config_unique.append(sic)

#             multiplicity.append(len(sec))
#             keep_energy.append(i)

# #     def get_config_multiplicity(input_data):
# #         i, config = input_data[0], input_data[1]
# #         sites = np.where(config == 1)[0] 
# #         sec = build_symmetry_equivalent_configurations(atom_indices,sites)
# #         sic = sec[0]
# #         is_in_config_unique = any(np.array_equal(sic, existing_sic) for existing_sic in config_unique)
        
# #         return (is_in_config_unique, sic, len(sec), i)
    
# #     results = Parallel(n_jobs=-1, backend="loky")(map(delayed(get_config_multiplicity), enumerate(configurations)))
    
# #     for (is_in_config_unique, sic, len_sec, i) in results:
# #         if not is_in_config_unique:  

# #             config_unique.append(sic)

# #             multiplicity.append(len_sec)
# #             keep_energy.append(i)        

#     unique_energies = np.array(energies)[keep_energy]
    
#     return config_unique, unique_energies, multiplicity

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



######## Mao's change

import functools as ft
import numpy as np

from braket.ahs.atom_arrangement import AtomArrangement
from braket.timings.time_series import TimeSeries
from braket.ahs.driving_field import DrivingField
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation

from braket.analog_hamiltonian_simulator.rydberg.constants import (
    RYDBERG_INTERACTION_COEF,
    SPACE_UNIT,
    TIME_UNIT,
)

from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_helpers import (
    get_blockade_configurations,
    _get_sparse_ops,
    _get_coefs
)

from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_unit_converter import (
    convert_unit,
)

import matplotlib.pyplot as plt


def tensor(N, indices):
    """Return the tensor product of a set of binary variables
    
    Example 1: 
        tensor(2, [0, 1]) = array([0, 0, 0, 1]) represents x0x1 with N=2 variables
    
    Example 2: 
        tensor(3, [0, 1]) = array([0, 0, 0, 0, 0, 0, 1, 1]) represents x0x1 with N=3 variables    
        
    Example 3: 
        tensor(3, [0, 2]) = array([0, 0, 0, 0, 0, 1, 0, 1]) represents x0x2 with N=3 variables    
        
    Example 4: 
        tensor(3, [1]) = array([0, 0, 1, 1, 0, 0, 1, 1]) represents x1 with N=3 variables    
        
    Example 5: 
        tensor(3, [0,1,2]) = array([0, 0, 0, 0, 0, 0, 0, 1]) represents x0x1x2 with N=3 variables    
        
    """
    
    list_binary_variables = [(1, 1) for _ in range(N)]
    for ind in indices:
        list_binary_variables[ind] = (0, 1)
        
    return ft.reduce(np.kron, list_binary_variables)

def get_next_nearest_neighbors(neighbors):
    """Return the next nearest neighbors from a given set of neighbors
    
    Example:
        >>> neighbors = [
            (0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6), (6, 7)
        ]
        
        >>> get_next_nearest_neighbors(neighbors)
        >>> [(2, 3), (4, 5), (0, 3, 6), (0, 2, 6), (1, 5, 7), (1, 4, 7), (2, 3), (4, 5)]
    """
    
    
    num_atoms = max(list(sum(neighbors, ()))) + 1
    first_neighbors = [[] for _ in range(num_atoms)]

    for pair in neighbors:
        first_neighbors[pair[0]].append(pair[1])
        first_neighbors[pair[1]].append(pair[0])


    next_nearest_neighbors = [[] for _ in range(num_atoms)]
    for i in range(num_atoms):
        temp = []
        for j in first_neighbors[i]:
            temp += first_neighbors[j]
            
        next_nearest_neighbors[i] = tuple(set(temp) - {i})    
        
    return next_nearest_neighbors
    

def QUBO(neighbors, J1=0.080403, J2=0.019894, J3=0.0):
    """Return the spectrum and solutions for a QUBO problem defined as
    
        H = J1 \sum_i x_i + J2 \sum_{<i,j>}x_ix_j + J3 \sum_{<<i,j>>}x_ix_j
    
    where J1 is the coefficient for the linear term, and J2 and J3 are the coefficients
    for the nearest neighbor and next nearest neighbor interactions.
    
    """
    
    
    J1 = float(J1)
    J2 = float(J2)
    num_atoms = max(list(sum(neighbors, ()))) + 1
    
    H = 0
    # Add the linear tearm or the potential term
    for i in range(num_atoms):
        H += J1 * tensor(num_atoms, [i])    
        
    # Add the nearest neighbor interaction
    for (i,j) in neighbors:
        if i < j:
            H += J2 * tensor(num_atoms, [i, j])
    
    
    # Add the next nearest neighbor interaction
    next_nearest_neighbors = get_next_nearest_neighbors(neighbors)
    for i in range(num_atoms):
        for j in next_nearest_neighbors[i]:
            if i < j:
                H += J3 * tensor(num_atoms, [i, j])    
    
    
    # Get the minimum, and its configuration
    min_val = min(H)
    min_val_indices = [i for i in range(len(H)) if H[i]==min_val]
    configs = [f'{index:0{num_atoms}b}' for index in min_val_indices]
    
    return H, min_val, min_val_indices, configs    
    
def get_final_ryd_Hamiltonian(
    coords, 
    detuning = 125000000.0,
    J1=0.080403, 
    J2=0.019894,
    C6 = 5.42e-24,
    threshold_factor = 1/18    
):
    """
        Return the Rydberg Hamiltonian for a given atom arrangement and a QUBO model.
        
        Args:
            coords: The coordinates of atoms and we assume that the nearest neighbor distance is 1. 
            detuning: The detuning in the Rydberg Hamiltonian
            J1: The linear term in the QUBO
            J2: The nearest neighbor interaction in the QUBO
            C6: The Rydberg interaction constant
            threshold_factor: see below
            
        Notes:
            When J2 * threshold_factor < J1, then the inter-atomic distance R = (C6/detuning * abs(J1)/J2)**(1/6)
            Otherwise, we define R1 = (C6 / 27 / abs(detuning))**(1/6), R2 = (C6 / abs(detuning))**(1/6), and take
            R = (R1+R2)/2
            
    """
    
    num_atoms = len(coords)
    
    if J1 == 0:
        detuning = 0
#         R = 8e-6
    elif J1 < 0:
        detuning = abs(detuning)
    else:
        detuning = -abs(detuning)
        
    if J2 * threshold_factor < abs(J1):
        R = (C6/detuning * -J1/J2)**(1/6)
    else:
        warnings.warn("J1 is too small")
        R1 = (C6 / 27 / abs(detuning))**(1/6)
        R2 = (C6 / abs(detuning))**(1/6)        
        R = (R1+R2)/2    
    
    # We will define the Hamiltonian via a ficticious AHS program
    
    # Define the register 
    register = AtomArrangement()

    for coord in coords:
        register.add(np.array(coord) * R)
        
    # Define a const driving field with zero Rabi frequency, and 
    # max allowed detuning
    t_max = 4e-6
    Omega = TimeSeries().put(0.0, 0.0).put(t_max, 0.0)
    Delta = TimeSeries().put(0.0, detuning).put(t_max, detuning)
    phi = TimeSeries().put(0.0, 0.0).put(t_max, 0.0)
    
    drive = DrivingField(
        amplitude=Omega,
        phase=phi,
        detuning=Delta
    )
    
    program = AnalogHamiltonianSimulation(
        hamiltonian=drive,
        register=register
    )
    
    
    # Now extract the Hamiltonian as a matrix from the program
    
    program = convert_unit(program.to_ir())
        
    configurations = get_blockade_configurations(program.setup.ahs_register, 0.0)

    
    rydberg_interaction_coef = RYDBERG_INTERACTION_COEF / ((SPACE_UNIT**6) / TIME_UNIT)
        
    rabi_ops, detuning_ops, interaction_op, local_detuning_ops = _get_sparse_ops(
        program, configurations, rydberg_interaction_coef
    )
        
    t_max_converted = program.hamiltonian.drivingFields[0].amplitude.time_series.times[-1]
    rabi_coefs, detuning_coefs, local_detuing_coefs = _get_coefs(program, [0, t_max_converted])
    
#     print(f"interaction_op={interaction_op}")
#     print(f"detuning_ops[0]={detuning_ops[0]}")
#     print(f"detuning_coefs[0][-1]={detuning_coefs[0][-1]}")
    
    H = interaction_op - detuning_ops[0] * detuning_coefs[0][-1]
    
    # H is diagonal
    diagH = np.real(H.diagonal())
    
    min_val = min(diagH)
    min_val_indices = [i for i in range(len(diagH)) if diagH[i]==min_val]
    configs = [f'{index:0{num_atoms}b}' for index in min_val_indices]
    
    
    return diagH, min_val, min_val_indices, configs, R


def show_coords(coords, radius=1.0, show_atom_index=True):
    """
    Plot the given coordinates for the atoms
    """
    fig = plt.figure(figsize=(7, 7))
    plt.plot(np.array(coords)[:, 0], np.array(coords)[:, 1], 'r.', ms=15)

    if show_atom_index:
        for idx, coord in enumerate(coords):
            plt.text(coord[0], coord[1], f"  {idx}", fontsize=12)
    
    if radius > 0:
        for site in coords:
            plt.gca().add_patch( plt.Circle((site[0],site[1]), radius/2, color="b", alpha=0.3) )
        plt.gca().set_aspect(1)
    plt.show()


def get_final_ryd_Hamiltonian_v2(
    coords, 
    detuning = 125000000.0,
    J1=0.080403, 
    J2=0.019894,
    C6 = 5.42e-24
):
    """
        Return the Rydberg Hamiltonian for a given atom arrangement and a QUBO model, version 2
        
        
        Args:
            coords: The coordinates of atoms and we assume that the nearest neighbor distance is 1. 
            detuning: The detuning in the Rydberg Hamiltonian
            J1: The linear term in the QUBO
            J2: The nearest neighbor interaction in the QUBO
            C6: The Rydberg interaction constant
            
        Notes:
            In this version, we have only one formula for R, namely (C6/detuning * -J1/J2)**(1/6).
            We return ratio = abs(detuning / J1) to map the spectrums
            
    """
    
    num_atoms = len(coords)
    
    if J1 == 0:
        detuning = 0
#         R = 8e-6
    elif J1 < 0:
        detuning = abs(detuning)
    else:
        detuning = -abs(detuning)
    
    
    
    R = (C6/detuning * -J1/J2)**(1/6)
    ratio = abs(detuning/1e6 / J1)
    
    
    # We will define the Hamiltonian via a ficticious AHS program
    
    # Define the register 
    register = AtomArrangement()

    for coord in coords:
        register.add(np.array(coord) * R)
        
    # Define a const driving field with zero Rabi frequency, and 
    # max allowed detuning
    t_max = 4e-6
    Omega = TimeSeries().put(0.0, 0.0).put(t_max, 0.0)
    Delta = TimeSeries().put(0.0, detuning).put(t_max, detuning)
    phi = TimeSeries().put(0.0, 0.0).put(t_max, 0.0)
    
    drive = DrivingField(
        amplitude=Omega,
        phase=phi,
        detuning=Delta
    )
    
    program = AnalogHamiltonianSimulation(
        hamiltonian=drive,
        register=register
    )
    
    
    # Now extract the Hamiltonian as a matrix from the program
    
    program = convert_unit(program.to_ir())
        
    configurations = get_blockade_configurations(program.setup.ahs_register, 0.0)

    
    rydberg_interaction_coef = RYDBERG_INTERACTION_COEF / ((SPACE_UNIT**6) / TIME_UNIT)
        
    rabi_ops, detuning_ops, interaction_op, local_detuning_ops = _get_sparse_ops(
        program, configurations, rydberg_interaction_coef
    )
        
    t_max_converted = program.hamiltonian.drivingFields[0].amplitude.time_series.times[-1]
    rabi_coefs, detuning_coefs, local_detuing_coefs = _get_coefs(program, [0, t_max_converted])
    
#     print(f"interaction_op={interaction_op}")
#     print(f"detuning_ops[0]={detuning_ops[0]}")
#     print(f"detuning_coefs[0][-1]={detuning_coefs[0][-1]}")
    
    H = interaction_op - detuning_ops[0] * detuning_coefs[0][-1]
    
    # H is diagonal
    diagH = np.real(H.diagonal())
    
    min_val = min(diagH)
    min_val_indices = [i for i in range(len(diagH)) if diagH[i]==min_val]
    configs = [f'{index:0{num_atoms}b}' for index in min_val_indices]
    
    
    return diagH, min_val, min_val_indices, configs, R, ratio
