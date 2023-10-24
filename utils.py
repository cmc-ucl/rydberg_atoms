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