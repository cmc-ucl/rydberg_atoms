import functools as ft
import numpy as np

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
    
    