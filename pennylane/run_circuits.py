#%%
import numpy as np
from qiskit.test.mock import FakeSantiago
from qufi import execute_over_range, BernsteinVazirani, Bell
import qufi

#%%
circuits = []

bv4_p = BernsteinVazirani.build_circuit(3, '101')
circuits.append((bv4_p, 'Bernstein-Vazirani_4'))

angles = {'theta0':[1, 2], 'phi0':[2, 3], 'theta1':[3, 4], 'phi1':[4, 5]}

#%%
device_backend = FakeSantiago()
coupling_map = qufi.get_qiskit_coupling_map(circuits[0][0], device_backend)

#%%
results_names = execute_over_range(circuits, angles, coupling_map=coupling_map, results_folder="./tmp/")

# %%
results = qufi.read_results_directory("./tmp/")

#%%
print(results)

# %%
qufi.compute_merged_histogram(results)
qufi.compute_circuit_heatmaps(results)
qufi.compute_circuit_delta_heatmaps(results)
qufi.compute_qubit_histograms(results)
qufi.compute_qubit_heatmaps(results)
