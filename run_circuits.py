#%%
import numpy as np
from qiskit.providers.fake_provider import FakeSantiago
from qufi import execute_over_range, BernsteinVazirani, get_qiskit_coupling_map, read_results_directory, generate_all_statistics

#%%
circuits = []

bv4_p = BernsteinVazirani.build_circuit(3, '101')
circuits.append((bv4_p, 'Bernstein-Vazirani_4'))

angles = {'theta0':[0, 0], 'phi0':[0, 0], 'theta1':[0, 0], 'phi1':[0, 0]}

#%%
device_backend = FakeSantiago()
coupling_map = get_qiskit_coupling_map(circuits[0][0], device_backend)

#%%
results_names = execute_over_range(circuits, angles, coupling_map=coupling_map, results_folder="./tmp/")

# %%
results = read_results_directory("./tmp/")

# %%
generate_all_statistics(results)

# %%
