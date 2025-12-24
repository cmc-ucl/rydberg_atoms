import copy
import json
import numpy as np

from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist

coordinates = np.array([[ 0.        ,  0.        ],
       [-2.005     ,  3.47276187],
       [ 0.        ,  6.94552374],
       [-2.005     , 10.41828561],
       [ 0.        , 13.89104748],
       [-2.005     , 17.36380935],
       [ 0.        , 20.83657122],
       [-2.005     , 24.30933308],
       [ 0.        , 27.78209495],
       [-2.005     , 31.25485682],
       [ 0.        , 34.72761869],
       [-2.005     , 38.20038056],
       [ 0.        , 41.67314243],
       [ 4.01      ,  0.        ],
       [ 6.015     ,  3.47276187],
       [ 4.01      ,  6.94552374],
       [ 6.015     , 10.41828561],
       [ 4.01      , 13.89104748],
       [ 6.015     , 17.36380935],
       [ 4.01      , 20.83657122],
       [ 6.015     , 24.30933308],
       [ 4.01      , 27.78209495],
       [ 6.015     , 31.25485682],
       [ 4.01      , 34.72761869],
       [ 6.015     , 38.20038056],
       [ 4.01      , 41.67314243],
       [12.03      ,  0.        ],
       [10.025     ,  3.47276187],
       [12.03      ,  6.94552374],
       [10.025     , 10.41828561],
       [12.03      , 13.89104748],
       [10.025     , 17.36380935],
       [12.03      , 20.83657122],
       [10.025     , 24.30933308],
       [12.03      , 27.78209495],
       [10.025     , 31.25485682],
       [12.03      , 34.72761869],
       [10.025     , 38.20038056],
       [12.03      , 41.67314243],
       [16.04      ,  0.        ],
       [18.045     ,  3.47276187],
       [16.04      ,  6.94552374],
       [18.045     , 10.41828561],
       [16.04      , 13.89104748],
       [18.045     , 17.36380935],
       [16.04      , 20.83657122],
       [18.045     , 24.30933308],
       [16.04      , 27.78209495],
       [18.045     , 31.25485682],
       [16.04      , 34.72761869],
       [18.045     , 38.20038056],
       [16.04      , 41.67314243],
       [24.06      ,  0.        ],
       [22.055     ,  3.47276187],
       [24.06      ,  6.94552374],
       [22.055     , 10.41828561],
       [24.06      , 13.89104748],
       [22.055     , 17.36380935],
       [24.06      , 20.83657122],
       [22.055     , 24.30933308],
       [24.06      , 27.78209495],
       [22.055     , 31.25485682],
       [24.06      , 34.72761869],
       [22.055     , 38.20038056],
       [24.06      , 41.67314243],
       [28.07      ,  0.        ],
       [30.075     ,  3.47276187],
       [28.07      ,  6.94552374],
       [30.075     , 10.41828561],
       [28.07      , 13.89104748],
       [30.075     , 17.36380935],
       [28.07      , 20.83657122],
       [30.075     , 24.30933308],
       [28.07      , 27.78209495],
       [30.075     , 31.25485682],
       [28.07      , 34.72761869],
       [30.075     , 38.20038056],
       [28.07      , 41.67314243]])
coordinates = coordinates*1e-6
dm = distance_matrix(coordinates, coordinates)

C6 = 5.42e-24  # C6 constant in rad m^6 / s
eV_to_rad_s = 1.519267447321156e15

np.fill_diagonal(dm,1)
dm_inv = 1/dm


dm_inv_6 = dm_inv**6

dm_inv_6_C6 = C6*dm_inv_6



dm_inv_6_C6_eV = dm_inv_6_C6/eV_to_rad_s

np.fill_diagonal(dm_inv_6_C6_eV,0)


# Write config-energies
size = 10000
Ryd_classical_E = []
concentration_all = []
mean_distance = []
min_distance = []

num_sites = len(coordinates)
max_conc = len(coordinates)


rng = np.random.default_rng(seed=2025202)  # create a generator with seed
concentration = rng.integers(0, max_conc, size=size)  # use rng.integers

# Generate binary vectors and energy values
for conc in concentration:
    ones = rng.choice(num_sites, conc, replace=False)
    x = np.zeros(num_sites, dtype='int')
    x[ones] = 1
    # Geometrical
    if conc > 1:
        mean_distance.append(pdist(coordinates[ones]).mean())
        min_distance.append(pdist(coordinates[ones]).min())
    else:
        mean_distance.append(np.nan)
        min_distance.append(np.nan)
    
    xx = np.outer(x,x)
    dm_xx = dm_inv_6_C6_eV * xx
    energy = np.sum(np.triu(dm_xx))
    Ryd_classical_E.append(float(f"{energy:.12g}"))

import json

# Example: save to 'output.json'
data = {
    "concentration": concentration.tolist(),
    "Rydberg_classical_energy": Ryd_classical_E,
    "mean_distance": mean_distance,
    "min_distance":min_distance
}

with open(f"sample_{size}_4um_production_full_range.json", "w") as f:
    json.dump(data, f, indent=4)
