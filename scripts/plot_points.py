import matplotlib.pyplot as plt
import numpy as np

# Provided JSON-like structure
data = {"nshots": 1, "lattice": {"sites": [[3.39e-05, 4.56e-05], [2.93e-05, 3.99e-05], [3.09e-05, 3.28e-05], [3.75e-05, 2.96e-05], [4.41e-05, 3.28e-05], [4.57e-05, 3.99e-05], [4.12e-05, 4.56e-05]], "filling": [1, 1, 1, 1, 1, 1, 1]}, "effective_hamiltonian": {"rydberg": {"rabi_frequency_amplitude": {"global_": {"times": [0.0, 5e-08, 1e-06, 1.05e-06], "values": [0.0, 3141600.0, 3141600.0, 0.0]}}, "rabi_frequency_phase": {"global_": {"times": [0.0, 1.05e-06], "values": [0.0, 0.0]}}, "detuning": {"global_": {"times": [0.0, 1.05e-06], "values": [0.0, 0.0]}, "local": {"times": [0.0, 5e-08, 1e-06, 1.05e-06], "values": [0, -125000000, -125000000, 0], "lattice_site_coefficients": [1.0, 1.0, 0, 1.0, 0, 1.0, 0]}}}}}

# Extract site coordinates
sites = np.array(data["lattice"]["sites"])

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(sites[:, 0], sites[:, 1], c='blue', s=50)

# Annotate points (optional)
for idx, (x, y) in enumerate(sites):
    plt.text(x, y, str(idx), fontsize=8, ha='right', va='bottom')

plt.xlabel("x position (meters)")
plt.ylabel("y position (meters)")
plt.title("Lattice Sites")
plt.axis('equal')
plt.grid(True)
plt.show()