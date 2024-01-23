This folder contains two sets of simulation data for the graphene structure with 54 atoms. The first set of data in "graphene_54_1.01_R_plots" is obtained by truncating the Rydberg Hamiltonian to neareast neighbor followed by running the quantum annealing, and we repeat the simulation for 50 values of J1 between -0.01 and -0.1. The second set of data in "graphene_54_1.74_R_plots" is obtained similarly by truncating the Rydberg Hamiltonian to second nearest neighbor. 

These two folders have similar structures. In the "graphene_54_1.01_R_plots" folder, there are 50 folders "graphene_54_J1" for certain J1 values that contain the raw data of the simulation, including 

1. "ahs_program.json": the json file for the AHS program 

2. "atom_coordinates.csv" and "atom_coordinates.png": the atom coordinates

3. "mps_samples.csv" and "mps_samples.png": the 1000 sampled results at the end of the evolution (Boltzmann distribution)

4. "most_likely_configs.png": The first two lowest lying states

The "graphene_54_plots" folder contains "most_likely_configs_J1.png" and "mps_samples_J1.png" for all the J1s, as well as four plots obtained by aggregating the data

1. "probs_lists.png": The probability of the first two lowest lying states, as a function of J1

2. qubo_energy_lists.png: The qubo energy calculated for `J2=0.019894`,  as a function of J1

3. "ryd_energy_lists.png": The energy calculated with Rydberg Hamiltonian, as a function of J1

4. "rydbergs_lists.png": The number of Rydberg atoms, as a function of J1


The other files are not relevant for now. The "graphene_54_1.74_R_plots" folder is structured similarly.