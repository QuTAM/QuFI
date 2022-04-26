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
from qiskit.providers.aer import AerSimulator
from qiskit import transpile, QuantumCircuit
from qiskit.test.mock import FakeSantiago

device_backend = FakeSantiago()

bv4_qasm = circuits[0][0].tape.to_openqasm()
bv4_qiskit = QuantumCircuit.from_qasm_str(bv4_qasm)
simulator = AerSimulator.from_backend(device_backend)
# Transpile the circuit for the noisy basis gates
tcirc = transpile(bv4_qiskit, simulator, optimization_level=3)
mapping = tcirc._layout.get_physical_bits()
coupling_map = {}
coupling_map['topology'] = set(tuple(sorted(l)) for l in device_backend.configuration().to_dict()['coupling_map'])
coupling_map['logical2physical'] = { k._index:v for (k,v) in tcirc._layout.get_virtual_bits().items() if k._register.name == 'q'}
coupling_map['physical2logical'] = { k:v._index for (k, v) in tcirc._layout.get_physical_bits().items() if v._register.name == 'q'}

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
