#%%
import numpy as np
from qufi import execute, save_results, BernsteinVazirani, Bell
import qufi

circuits = []

bv4_p = BernsteinVazirani.build_circuit(3, '101')
circuits.append((bv4_p, 'Bernstein-Vazirani_4'))

angles = {'theta1':[np.pi], 'phi1':[np.pi], 'theta2':[np.pi], 'phi2':[np.pi]}

#%%

results = execute(circuits, angles)

#%%
circuits = []

c = Bell.build_circuit()
circuits.append((c, 'Bell'))

angles = {'theta1':[np.pi], 'phi1':[np.pi], 'theta2':[np.pi], 'phi2':[np.pi]}

from qiskit.providers.aer import AerSimulator
from qiskit import transpile, QuantumCircuit
from qiskit.test.mock import FakeSantiago

device_backend = FakeSantiago()

bv4_qasm = c.tape.to_openqasm()
bv4_qiskit = QuantumCircuit.from_qasm_str(bv4_qasm)
simulator = AerSimulator.from_backend(device_backend)
# Transpile the circuit for the noisy basis gates
tcirc = transpile(bv4_qiskit, simulator, optimization_level=3)
mapping = tcirc._layout.get_physical_bits()
coupling_map = {}
coupling_map['topology'] = set(tuple(sorted(l)) for l in device_backend.configuration().to_dict()['coupling_map'])
coupling_map['logical2physical'] = { k._index:v for (k,v) in tcirc._layout.get_virtual_bits().items() if k._register.name == 'q'}
coupling_map['physical2logical'] = { k:v._index for (k, v) in tcirc._layout.get_physical_bits().items()}

#%%

results = execute(circuits, angles, coupling_map)

#%%
save_results(results, filename="../results/u_gate_15degrees_step_bv_4_pennylane.p.gz")

# %%

read_results = qufi.read_results_double_fi(["../results/u_gate_15degrees_step_bv_4_pennylane.p.gz"])

# %%

qufi.compute_merged_histogram(read_results)
qufi.compute_circuit_heatmaps(read_results)
qufi.compute_circuit_delta_heatmaps(read_results)
qufi.compute_qubit_histograms(read_results)
qufi.compute_qubit_heatmaps(read_results)

