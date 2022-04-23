
import numpy as np
from qufi import execute, save_results, BernsteinVazirani, Bell
import qufi
from qiskit.providers.aer import AerSimulator
from qiskit import transpile, QuantumCircuit
from qiskit.test.mock import FakeSantiago

circuits = []

bv4_p = BernsteinVazirani.build_circuit(3, '101')
circuits.append((bv4_p, 'Bernstein-Vazirani_4'))

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

results = execute(circuits, coupling_map=coupling_map)

save_results(results, filename="../results/bv4_df_pennylane_byobu.p.gz")
