import numpy as np
from itertools import product
import pickle, gzip
import datetime
from math import ceil
# Importing standard Qiskit libraries
from qiskit.circuit.quantumcircuit import QuantumCircuit as qiskitQC
from qiskit.test.mock import FakeSantiago
from qiskit.providers.aer.noise import NoiseModel
import pennylane as qml
import sys
sys.path.insert(0,'..')
import networkx as nx

file_logging = False
logging_filename = "./qufi.log"
console_logging = True

def log(content):
    if file_logging:
        fp = open(logging_filename, "a")
        fp.write(content+'\n')
        fp.flush()
        fp.close()
    if console_logging:
        print(content)

def probs_to_counts(probs, nwires):
    res_dict = {}
    shots = 1024
    for p, t in zip(probs, list(product(['0', '1'], repeat=nwires))):
        b = ''.join(t)
        count = int(ceil(shots*float(p)))
        if count != 0:
            res_dict[b] = count
    # debug check for ceil rounding
    if sum(res_dict.values()) != shots:
        log(f"Rounding error! {sum(res_dict.values())} != {shots}")
    return res_dict

def run_circuits(base_circuit, generated_circuits, device_backend=FakeSantiago()):
    # Execute golden circuit simulation without noise
    log('Running circuits')
    gold_device = qml.device('lightning.qubit', wires=base_circuit.device.num_wires)
    gold_qnode = qml.QNode(base_circuit.func, gold_device)
    answer_gold = probs_to_counts(gold_qnode(), base_circuit.device.num_wires)

    # Execute golden circuit simulation with noise
    noise_model = NoiseModel.from_backend(device_backend)
    gold_device_noisy = qml.device('qiskit.aer', wires=base_circuit.device.num_wires, backend='aer_simulator', noise_model=noise_model)
    gold_qnode_noisy = qml.QNode(base_circuit.func, gold_device_noisy)
    answer_gold_noise = probs_to_counts(gold_qnode_noisy(), base_circuit.device.num_wires)

    # Execute injection circuit simulations without noise
    answers = []
    for c, i in zip(generated_circuits, range(0, len(generated_circuits))):
        inj_device = qml.device('lightning.qubit', wires=c.device.num_wires)
        inj_qnode = qml.QNode(c.func, inj_device)
        answer = probs_to_counts(inj_qnode(), base_circuit.device.num_wires)
        answers.append(answer)

    # Execute injection circuit simulations with noise
    answers_noise = []
    for c, i in zip(generated_circuits, range(0, len(generated_circuits))):
        inj_device_noisy = qml.device('qiskit.aer', wires=c.device.num_wires, backend='aer_simulator', noise_model=noise_model)
        inj_qnode_noisy = qml.QNode(c.func, inj_device_noisy)
        answer_noise = probs_to_counts(inj_qnode_noisy(), base_circuit.device.num_wires)
        answers_noise.append(answer_noise)

    return {'output_gold':answer_gold, 'output_injections':answers
            , 'output_gold_noise':answer_gold_noise, 'output_injections_noise':answers_noise
            , 'noise_target':str(device_backend)
            }

def convert_qiskit_circuit(qiskit_circuit):
    shots = 1024
    measure_list = [g[1][0]._index for g in qiskit_circuit[0].data if g[0].name == 'measure']
    qregs = qiskit_circuit[0].num_qubits
    # Avoid qml.load warning on trying to convert measure operators
    qiskit_circuit[0].remove_final_measurements()
    pl_circuit = qml.load(qiskit_circuit[0], format='qiskit')
    device = qml.device("lightning.qubit", wires=qregs, shots=shots)
    @qml.qnode(device)
    def conv_circuit():
        pl_circuit(wires=range(qregs))
        return qml.probs(wires=measure_list) #[qml.expval(qml.PauliZ(i)) for i in range(qregs)]
    # Do NOT remove this evaluation, else the qnode can't bind the function before exiting convert_qiskit_circuit()'s context
    conv_circuit()
    #print(qml.draw(conv_circuit)())
    return conv_circuit

@qml.qfunc_transform
def pl_insert_gate(tape, index, wire, theta, phi, lam):
    i = 0
    for gate in tape.operations + tape.measurements:
        # Ignore barriers and measurement gates
        if i == index:
            # If gate are not using a single qubit, insert one gate after each qubit
            qml.apply(gate)
            qml.U3(theta=theta, phi=phi, delta=lam, wires=wire, id="FAULT")
        else:
            qml.apply(gate)
        i = i + 1

def pl_generate_circuits(base_circuit, name, theta=0, phi=0, lam=0):
    mycircuits = []
    inj_info = []
    with base_circuit.tape as tape:
        index = 0
        for op in tape.operations:
            for wire in op.wires:
                shots = 1024
                transformed_circuit = pl_insert_gate(index, wire, theta, phi, lam)(base_circuit.func)
                device = qml.device('lightning.qubit', wires=len(tape.wires), shots=shots)
                transformed_qnode = qml.QNode(transformed_circuit, device)
                log(f'Generated circuit: {name} with fault on ({op.name}, wire:{wire}), theta = {theta}, phi = {phi}')
                #print(qml.draw(transformed_qnode)())
                mycircuits.append(transformed_qnode)
                inj_info.append(wire)
            index = index + 1
        log(f"{len(mycircuits)} circuits generated\n")
        return mycircuits, inj_info

def pl_inject(circuit, name, theta=0, phi=0, lam=0):
    output = {'name': name, 'base_circuit':circuit, 'theta':theta, 'phi':phi, 'lambda':lam}
    output['pennylane_version'] = qml.version()
    #print(qml.draw(circuit)())
    output['circuits_injections'], wires = pl_generate_circuits(circuit, name, theta, phi, lam)
    output.update(run_circuits( output['base_circuit'], output['circuits_injections'] ) )
    # Remove all qnodes from the output dict since pickle can't process them (they are functions)
    # "Then he turned himself into a pickle, funniest shit I've ever seen!"
    output['base_circuit'] = None
    output['circuits_injections'] = wires
    return output

def execute(circuits, theta_values=np.arange(0, np.pi+0.01, np.pi/12), phi_values=np.arange(0, 2*np.pi, np.pi/12)):
    results = []
    tstart = datetime.datetime.now()
    log(f"Start: {tstart}")
    for circuit in circuits:
        log(f"-"*80+"\n")
        tstartint = datetime.datetime.now()
        log(f"Circuit {circuit[1]} start: {tstartint}")
        angle_values = product(theta_values, phi_values)
        for angles in angle_values:
            # Converting the circuit only once at the start of outer loop causes reference bugs (insight needed)
            if isinstance(circuit[0], qml.QNode):
                target_circuit = circuit[0]
            elif isinstance(circuit[0], qiskitQC):
                target_circuit = convert_qiskit_circuit(circuit)
            else:
                log(f"Unsupported {type(circuit[0])} object, injection stopped.")
                exit()
            log(f"-"*80+"\n"+f"Injecting circuit: {circuit[1]} theta: {angles[0]} phi: {angles[1]}")
            r = pl_inject(target_circuit, circuit[1], theta=angles[0], phi=angles[1])
            results.append(r)
        tendint = datetime.datetime.now()
        log(f"Done: {tendint}\nElapsed time: {tendint-tstartint}\n"+"-"*80+"\n")
    tend = datetime.datetime.now()
    log(f"Done: {tend}\nTotal elapsed time: {tend-tstart}\n")

    return results

def save_results(results, filename='./results.p.gz'):
    pickle.dump(results, gzip.open(filename, 'w'))
    log(f"Files saved to {filename}")