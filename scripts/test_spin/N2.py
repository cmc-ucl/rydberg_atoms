###### capture the output ##########
import sys
old_stdout = sys.stdout
log_file = open("N2.out","w")
sys.stdout = log_file
###### capture the output ##########

import json
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize
#
#


coords = [[0.000000, 0.000000, 0.000000], [0.000000, 0.000, 1.2]]
species = ["N","N"]

# Convert to PySCF molecule format
mol = gto.Mole()
mol.atom = [[str(species[i]), coord] for i, coord in enumerate(coords)]
mol.basis = 'ccpvtz'
mol.build()

# Perform DFT calculation
mf = dft.RKS(mol)
mf.xc = 'pbe0'

mol_eq = optimize(mf,maxsteps=200)

mf_eq = dft.RKS(mol_eq)
mf_eq.xc = 'pbe0'
energy = mf_eq.kernel()


mf_eq.analyze()
mo_energies = mf_eq.mo_energy.tolist()
mulliken_charges = mf_eq.mulliken_pop()[1].tolist()
dipole_moment = mf_eq.dip_moment().tolist()

# Save results to a JSON file
results = {
    "energy": energy,
    "MO_energies": mo_energies,
    "Mulliken_charges": mulliken_charges,
    "dipole_moment": dipole_moment
}

with open("N2.json", "w") as json_file:
    json.dump(results, json_file, indent=4)

atom_coords = mol_eq.atom_coords()
atom_charges = mol_eq.atom_charges()
num_atoms = len(atom_coords)

with open('N2.xyz', 'w') as f:
    f.write(f"{num_atoms}\n")
    f.write("Optimized geometry\n")
    for i in range(num_atoms):
        symbol = gto.mole._symbol(atom_charges[i])
        x, y, z = atom_coords[i]
        f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")

num_alpha = mol_eq.nelectron // 2 + mol_eq.spin // 2
num_beta = mol_eq.nelectron // 2 - mol_eq.spin // 2

print(f"Number of alpha electrons: {num_alpha}")
print(f"Number of beta electrons: {num_beta}")

# Spin multiplicity
multiplicity = mol_eq.spin + 1
print(f"Spin multiplicity: {multiplicity}")

###### capture the output ##########
sys.stdout = old_stdout
log_file.close()
###### capture the output ##########
