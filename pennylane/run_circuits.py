#%%
from os import read
import py_compile
import numpy as np
from qufi import execute, save_results, IQFT
import qufi

circuits = []

bv4_p = IQFT.build_circuit(4)
circuits.append((bv4_p, 'Bernstein-Vazirani_4'))

theta_values = [np.pi]
phi_values = [np.pi]

#%%

results = execute(circuits, theta_values, phi_values)

#%%
save_results(results, filename='./prova.p.gz')

# %%

read_results = qufi.read_results_double_fi(["../results/u_gate_15degrees_step_bv_4_pennylane.p.gz"])

# %%

qufi.compute_merged_histogram(read_results)
qufi.compute_circuit_heatmaps(read_results)
qufi.compute_circuit_delta_heatmaps(read_results)
qufi.compute_qubit_histograms(read_results)
qufi.compute_qubit_heatmaps(read_results)

