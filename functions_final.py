from pymatgen.core.structure import Structure, Molecule
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, PointGroupAnalyzer
import copy
import numpy as np
from ase.visualize import view
from pymatgen.io.ase import AseAtomsAdaptor
import itertools
from scipy import constants
import math
import shutil as sh

#import sys
#sys.path.insert(1,'/Users/brunocamino/Desktop/Imperial/crystal-code-tools/CRYSTALpytools/CRYSTALpytools/')
from CRYSTALpytools.crystal_io import *
from CRYSTALpytools.convert import *

#from CRYSTALpytools.convert import *
#from CRYSTALpytools.crystal_io import *

import matplotlib.pyplot as plt

k_b = constants.physical_constants['Boltzmann constant in eV/K'][0]
plt.style.use('tableau-colorblind10')
def vview(structure):
    view(AseAtomsAdaptor().get_atoms(structure))
    
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
    _get_coefs,
    _get_rabi_dict,
    _get_detuning_dict,
    _get_sparse_from_dict
)

from braket.analog_hamiltonian_simulator.rydberg.rydberg_simulator_unit_converter import (
    convert_unit,
)

from pyscf import gto, dft

def classical_energy(x,q):

    # x is the binary vector
    # q is the qubo matrix

    E_tmp = np.matmul(x,q)
    E_classical = np.sum(x*E_tmp)
    
    return E_classical


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

    
    energy = np.array(energy)
    multiplicity = np.array(multiplicity)
    p_i = multiplicity * np.exp((-energy + (N_N * N_potential)) / (k_b * T))
    
    pf = np.sum(p_i)
    
    p_i /= pf
    
    if return_pi:
        return pf, p_i
    else:
        return pf


def build_adjacency_matrix_no_pbc(structure_pbc, max_neigh = 1, diagonal_terms = False, triu = False):
    # structure = pymatgen Structure object
    
    from pymatgen.core.structure import Molecule
    structure = Molecule(structure_pbc.atomic_numbers,structure_pbc.cart_coords)
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
        np.fill_diagonal(distance_matrix,[-1]*num_sites)
    
    
    return distance_matrix.astype('int')


def build_qubo_from_J(structure,J):
    num_sites = structure.num_sites
    max_neigh = len(J)-1
    am = build_adjacency_matrix_no_pbc(structure,max_neigh==max_neigh,diagonal_terms=True,triu=True)
    
    Q = np.zeros((num_sites,num_sites))
    
    for i in np.arange(1,len(J)):
        print(i)
        indices = np.where(am == i )
        Q[indices] = J[i]
    np.fill_diagonal(Q,J[0])
    
    return Q


# Function to check if a point is inside the hexagon
def is_inside_hexagon(size,x, y):
    
    """
    Check if a point (x, y) is inside a regular hexagon of a specified size.

    This function determines whether a given point is inside or outside a regular hexagon
    with a specified size.

    Parameters:
    - size (float): The size of the hexagon (distance from the center to a vertex).
    - x (float): The x-coordinate of the point to be checked.
    - y (float): The y-coordinate of the point to be checked.

    Returns:
    - inside (bool): True if the point is inside the hexagon, False otherwise.

    """
    
    hexagon_vertices = size*np.array([(1, 0), (0.5, math.sqrt(3) / 2), (-0.5, math.sqrt(3) / 2), (-1, 0), (-0.5, -math.sqrt(3) / 2), (0.5, -math.sqrt(3) / 2)])

    odd_nodes = False
    j = len(hexagon_vertices) - 1

    for i in range(len(hexagon_vertices)):
        xi, yi = hexagon_vertices[i]
        xj, yj = hexagon_vertices[j]

        if yi < y and yj >= y or yj < y and yi >= y:
            if xi + (y - yi) / (yj - yi) * (xj - xi) < x:
                odd_nodes = not odd_nodes

        j = i

    return odd_nodes


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


def cut_graphene_hexagonal(structure_bulk,size,center=True):
    
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
    
    expansion_matrix = 7*np.eye(3)
    expansion_matrix[2][2] = 1
    structure.make_supercell(expansion_matrix)
    if center == True:
        com = np.mean(structure.frac_coords,axis=0)
        structure.translate_sites(np.arange(structure.num_sites),-com,to_unit_cell=False)

    mask = []
    for i in range(structure.num_sites):
        x = structure.cart_coords[i][0]
        y = structure.cart_coords[i][1]
        mask.append(is_inside_hexagon(size,x, y))
        
    keep_site = np.where(np.array(mask) == True)
    
    species = np.array(structure.atomic_numbers)[keep_site]
    coordinates = np.array(structure.cart_coords)[keep_site]
    return Molecule(species,coordinates)


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


def get_R_from_J(
    J1 = 0.080403, 
    J2 = 0.019894,
    detuning = 125000000.0,
    R_fix = 2.0,
    C6 = 5.42e-24,
    max_C = True
):
    
    num_atoms = len(structure.num_sites)
    
    # MAXIMISE C
    if J1 == 0:
        detuning = 0
#         R = 8e-6
    elif J1 < 0:
        detuning = abs(detuning)
    else:
        detuning = -abs(detuning)
    
    
    
    R = (C6/detuning * -J1/J2)**(1/6)
    ratio = abs(detuning/1e6 / J1)
    
    return R


def get_all_configurations_pmg_mol(structure_pmg,prec=6):
    
    lattice = np.eye(3)*10
    
    structure_pmg = structure_pmg.get_centered_molecule()
    symmops = PointGroupAnalyzer(structure_pmg).get_symmetry_operations()

    coordinates = np.round(np.array(structure_pmg.cart_coords),prec)
    n_symmops = len(symmops)
    atom_numbers = np.array(structure_pmg.atomic_numbers)
    
    original_structure_pmg = copy.deepcopy(structure_pmg)
            
    rotations = []
    translation = []

    atom_indices = np.ones((len(symmops),structure_pmg.num_sites),dtype='int')
    atom_indices *= -1

    structures = []
    for i,symmop in enumerate(symmops):
        
        atom_indices_tmp = []
        coordinates_new = []
        for site in coordinates:
            coordinates_new.append(np.round(symmop.operate(site),prec))

        structure_tmp = Structure(lattice,atom_numbers,coordinates_new,coords_are_cartesian=False,
                                  to_unit_cell=False)

        for k,coord in enumerate(original_structure_pmg.cart_coords):
            structure_tmp.append(original_structure_pmg.atomic_numbers[k],coord,coords_are_cartesian=False,
                                 validate_proximity=False)

        for m in range(len(atom_numbers)):
            index = len(atom_numbers)+m
            for n in range(len(atom_numbers)):

                if structure_tmp.sites[n].is_periodic_image(structure_tmp.sites[index],tolerance=0.001):
                    
                    atom_indices[i,m] = n
                    break


    return atom_indices


#Build all configurations (complete set)
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

def build_ml_qubo(structure,X_train,y_train,max_neigh=1):
    
    #Filter
    distance_matrix = np.round(structure.distance_matrix,5)
    shells = np.unique(np.round(distance_matrix,5))
    num_sites = structure.num_sites
    distance_matrix_filter = np.zeros((num_sites,num_sites),int)

    for i,s in enumerate(shells[0:max_neigh+1]):
        row_index = np.where(distance_matrix == s)[0]
        col_index = np.where(distance_matrix == s)[1]
        distance_matrix_filter[row_index,col_index] = i
    distance_matrix_filter = np.triu(distance_matrix_filter,0)
    np.fill_diagonal(distance_matrix_filter,[1]*num_sites)
    
    #Build the descriptor

    upper_tri_indices = np.where(distance_matrix_filter != 0)
    descriptor = []

    for config in X_train:
        matrix = np.outer(config,config)
        upper_tri_elements = matrix[upper_tri_indices]
        descriptor.append(upper_tri_elements)
        

#     descriptor_all = []
#     for config in all_configurations:
#         matrix = np.outer(config,config)
#         upper_tri_elements = matrix[upper_tri_indices]
#         descriptor_all.append(upper_tri_elements)
    
    descriptor = np.array(descriptor)
    
    from sklearn.linear_model import LinearRegression
    
    
    reg = LinearRegression() #create the object
    reg.fit(descriptor, y_train)
    
    print('R2: ',reg.score(descriptor, y_train))

    Q = np.zeros((num_sites,num_sites))
    Q[upper_tri_indices] = reg.coef_
    
    return Q

def get_qubo_energies(Q,all_configurations):
    
    predicted_energy = []
    
    for i,config in enumerate(all_configurations):
        predicted_energy.append(classical_energy(config,Q))
    
    return predicted_energy


# In[37]:


def test_qubo_energies(y_pred,y_dft):
    
    from sklearn.metrics import mean_squared_error as mse
    
    return mse(y_pred, y_dft)


# In[36]:


def test_qubo_energies_mape(y_pred,y_dft):
    
    from sklearn.metrics import mean_absolute_percentage_error as mse
    
    return mse(y_pred, y_dft)

def build_symmetry_equivalent_configurations(atom_indices,N_index):
    
    if len(N_index) == 0:
        #return np.tile(np.zeros(len(atom_indices[0]),dtype='int'), (len(atom_indices), 1))
        return np.array([np.zeros(len(atom_indices[0]),dtype='int')]) # TEST
    configurations = atom_indices == -1
    #print(configurations)
    for index in N_index:
        configurations += atom_indices == index
    configurations = configurations.astype(int)

    unique_configurations,unique_configurations_index = np.unique(configurations,axis=0,return_index=True)
    
    return unique_configurations


def generate_random_structures_mol(initial_structure,atom_indices,N_atoms,new_species,N_config,DFT_config,active_sites=False,return_multiplicity=False):
    
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
#         structure_tmp = copy.deepcopy(structure)
        sec = build_symmetry_equivalent_configurations(atom_indices,sites)

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
        atomic_numbers = np.array(initial_structure.atomic_numbers)
        atomic_numbers[N_index] = new_species
        structure_tmp = Molecule(atomic_numbers,initial_structure.cart_coords)
        final_structures.append(structure_tmp)
    if return_multiplicity == True:
        return final_structures,config_unique_count
    else:
        return final_structures


#ORIGINAL: THIS WORKS, DO NOT MODIFY
# def build_ml_h_ryd(structure_pbc,X_train,y_train,max_neigh=1):
    
    C = 5.42e-18#5.42e-24
    Delta_g = 1#-1.25e8
    from pymatgen.core.structure import Molecule
    structure = Molecule(structure_pbc.atomic_numbers,structure_pbc.cart_coords)
    #Filter
    distance_matrix = np.round(structure.distance_matrix,5)
    #print(distance_matrix)
    shells = np.unique(np.round(distance_matrix,5))
    num_sites = structure.num_sites
    distance_matrix_filter = np.zeros((num_sites,num_sites),float)
    
    ryd_param = [1,1,1/27]
    ryd_param = [1,1,1/27,1/343]
    
    for i,s in enumerate(shells[0:max_neigh+1]):
        row_index = np.where(distance_matrix == s)[0]
        col_index = np.where(distance_matrix == s)[1]
        distance_matrix_filter[row_index,col_index] = ryd_param[i]
    distance_matrix_filter = np.triu(distance_matrix_filter,0)

    #print(distance_matrix_filter[5])    
    
    #Build the descriptor

    upper_tri_indices = np.where(distance_matrix_filter != 0.)
    descriptor = []
    descriptor_test = []
    for config in X_train:
        matrix = np.outer(config,config)*distance_matrix_filter #matrix[i][j] == 1 if i and j are ==1
        upper_tri_elements = matrix[upper_tri_indices]
        
        descriptor.append(upper_tri_elements)
        diag = np.sum(matrix.diagonal())
        diag_all = matrix.diagonal().tolist()
        all_terms = np.sum(upper_tri_elements)
        diag_all.append(all_terms-diag)
        
        descriptor_test.append([diag,all_terms-diag])
        #descriptor_test.append(diag_all)
    #print(descriptor_test)

#     descriptor_all = []
#     for config in all_configurations:
#         matrix = np.outer(config,config)
#         upper_tri_elements = matrix[upper_tri_indices]
#         descriptor_all.append(upper_tri_elements)
    descriptor  = copy.deepcopy(descriptor_test)
    descriptor = np.array(descriptor)
    
    from sklearn.linear_model import LinearRegression
    
    
    reg = LinearRegression() #create the object
    reg.fit(descriptor, y_train)
    print(reg.coef_)
    print('R2: ',reg.score(descriptor, y_train))
    
    
    ##########QUBO E
    Q_structure = np.zeros((num_sites,num_sites),float)
    distance_matrix = build_adjacency_matrix(structure,max_neigh=2)
    
    #print(Q_structure)
    nn = np.where(distance_matrix==1)

    Q_structure[nn] = reg.coef_[1]-(reg.coef_[1]*1/27)
    #print(reg.coef_[1],Q_structure[nn])
    nnn = np.where(distance_matrix==2)
    Q_structure[nnn] = reg.coef_[1]*1/27
    #print(Q_structure[nnn])
    # Add the chemical potential (still calling it J1)
    #J1 += J1*mu
    np.fill_diagonal(Q_structure,reg.coef_[0])
    
#     Q = np.zeros((num_sites,num_sites))
#     Q[upper_tri_indices] = reg.coef_
    #print(np.unique(Q_structure))
    return Q_structure
