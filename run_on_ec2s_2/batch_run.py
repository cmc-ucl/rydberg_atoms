import os

files = []
for file in os.listdir():
    if 'py' in file and 'graphene_mol_r_6_6_h' in file:
        files.append(file)

print(files)

import multiprocessing as mp
import subprocess
import numpy as np

def run_one(file):
    subprocess.run(['python', file])


with mp.Pool(processes=len(files), initializer=np.random.seed) as p:
    measurements = p.map(run_one, files)

