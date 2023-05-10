#%%
import numpy as np
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
from qiskit.providers.fake_provider import FakeSantiago
from qufi import execute_over_range, BernsteinVazirani, get_qiskit_coupling_map, read_file, generate_all_statistics

#%%
circuits = []

bv4_p = BernsteinVazirani.build_circuit(3, '101')
circuits.append((bv4_p, 'Bernstein-Vazirani_4'))

angles = {'theta0':[0, np.pi/2], 'phi0':[0, np.pi/4], 'theta1':[0, 0], 'phi1':[0, 0]}

#%%
device_backend = FakeSantiago()
coupling_map = get_qiskit_coupling_map(circuits[0][0], device_backend)

#%%
results_names = execute_over_range(circuits, angles, coupling_map=coupling_map, results_folder="./tmp/")
# %%
results = read_file(f"{results_names[1]}")

# %%
generate_all_statistics(results)

# %%
