#%%
import numpy as np
from qufi import execute, BernsteinVazirani, Bell
import qufi

#%%
circuits = []

bv4_p = BernsteinVazirani.build_circuit(3, '101')
circuits.append((bv4_p, 'Bernstein-Vazirani_4'))

angles = {'theta1':[1, 2], 'phi1':[2, 3], 'theta2':[3, 4], 'phi2':[4, 5]}

#%%
circuits = []

c = Bell.build_circuit()
circuits.append((c, 'Bell'))

angles = {'theta1':[1, 2], 'phi1':[2, 3], 'theta2':[3, 4], 'phi2':[4, 5]}

#%%
from qiskit.test.mock import FakeSantiago

device_backend = FakeSantiago()
coupling_map = qufi.get_qiskit_coupling_map(circuits[0][0], device_backend)

#%%
results_names = execute(circuits, angles, coupling_map=coupling_map, results_folder="./tmp/")

# %%
results = qufi.read_results_double_fi(results_names)

# %%
qufi.compute_merged_histogram(results)
qufi.compute_circuit_heatmaps(results)
qufi.compute_circuit_delta_heatmaps(results)
qufi.compute_qubit_histograms(results)
qufi.compute_qubit_heatmaps(results)
