from qufi import execute, BernsteinVazirani
import qufi
from qiskit.test.mock import FakeSantiago

circuits = []

bv4_p = BernsteinVazirani.build_circuit(3, '101')
circuits.append((bv4_p, 'Bernstein-Vazirani_4'))

device_backend = FakeSantiago()
coupling_map = qufi.get_qiskit_coupling_map(circuits[0][0], device_backend)

results_names = execute(circuits, coupling_map=coupling_map)

results = qufi.read_results_double_fi(results_names)

qufi.compute_merged_histogram(results)
qufi.compute_circuit_heatmaps(results)
qufi.compute_circuit_delta_heatmaps(results)
qufi.compute_qubit_histograms(results)
qufi.compute_qubit_heatmaps(results)