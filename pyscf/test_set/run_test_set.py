import subprocess
import multiprocessing as mp
import sys
import numpy as np

start_ind, end_ind = int(sys.argv[1]), int(sys.argv[2])
print([start_ind, end_ind])

# GKP2: [0, 46]
# GKP1: [47, 93]
# GKP3: [94, 140]
# GKP4: [141, 187]
# us-east-1: [188, 234]
# us-west-2: [235, 245]

def run(ind):
    subprocess.run(['python', f'graphene_mol_r_6_6_h_{ind}.py'])

with mp.Pool(processes=end_ind-start_ind+1, initializer=np.random.seed) as p:
    measurements = p.map(run, range(start_ind, end_ind+1))

