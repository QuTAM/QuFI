#%%
import numpy as np
import matplotlib
# matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['text.usetex'] = True
from qiskit.providers.fake_provider import FakeSantiago
from qufi import execute_over_range_single, BernsteinVazirani, get_qiskit_coupling_map, read_file, generate_all_statistics

#%%
circuits = []

bv4_p = BernsteinVazirani.build_circuit(3, '101')
circuits.append((bv4_p, 'Bernstein-Vazirani_4'))

angles = {'theta0':[0, np.pi/4], 'phi0':[0, np.pi/4], 'theta1':[0, 0], 'phi1':[0, 0]}

#%%
# device_backend = FakeSantiago()
# coupling_map = get_qiskit_coupling_map(circuits[0][0], device_backend)

#%%
results_names = execute_over_range_single(circuits, angles, noise=False, results_folder="./tmp/")
# %%
results = []
for name in results_names:
    results.append(read_file(f"{name}", single = True))

# %%
for i, res in enumerate(results):
    results[i].to_csv(f'example_result_{i+1}.csv', index=False, header = True)

# %%
