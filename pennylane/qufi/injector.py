from copy import deepcopy
import numpy as np
from itertools import product
import pickle, gzip
import datetime
from math import ceil
from os.path import isdir, dirname
from os import mkdir
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
    # Debug check for ceil rounding (Still bugged somehow, sometimes off by 1-2 shots)
    #if sum(res_dict.values()) != shots:
    #    log(f"Rounding error! {sum(res_dict.values())} != {shots}")
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
def pl_insert_gate(tape, index, wire, theta=0, phi=0, lam=0):
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

@qml.qfunc_transform
def pl_insert_df_gate(tape, index, wire, second_theta=0, second_phi=0, lam=0):
    i = 0
    inserted = False
    for gate in tape.operations:
        # Ignore barriers and measurement gates
        if i > index and not inserted and wire in gate.wires:
            # If gate are not using a single qubit, insert one gate after each qubit
            qml.apply(gate)
            qml.U3(theta=second_theta, phi=second_phi, delta=lam, wires=wire, id="FAULT")
            inserted = True
        else:
            qml.apply(gate)
        i = i + 1
    if not inserted:
        qml.U3(theta=second_theta, phi=second_phi, delta=lam, wires=wire, id="FAULT")
    for meas in tape.measurements:
        qml.apply(meas)

def pl_generate_circuits(base_circuit, name, theta=0, phi=0, lam=0):
    mycircuits = []
    inj_info = []
    index_info = []
    with base_circuit.tape as tape:
        index = 0
        for op in tape.operations:
            for wire in op.wires:
                shots = 1024
                transformed_circuit = pl_insert_gate(index, wire, theta, phi, lam)(base_circuit.func)
                device = qml.device('lightning.qubit', wires=len(tape.wires), shots=shots)
                transformed_qnode = qml.QNode(transformed_circuit, device)
                log(f'Generated single fault circuit: {name} with fault on ({op.name}, wire:{wire}), theta = {theta}, phi = {phi}')
                #print(qml.draw(transformed_qnode)())
                transformed_qnode()
                mycircuits.append(transformed_qnode)
                inj_info.append(wire)
                index_info.append(index)
            index = index + 1
        log(f"{len(mycircuits)} circuits generated\n")
        return mycircuits, inj_info, index_info

def pl_insert(circuit, name, theta=0, phi=0, lam=0):
    output = {'name': name, 'base_circuit':circuit, 'theta1':theta, 'phi1':phi, 'theta2':0, 'phi2':0, 'lambda':lam}
    output['pennylane_version'] = qml.version()
    #print(qml.draw(circuit)())
    generated_circuits, wires, indexes = pl_generate_circuits(circuit, name, theta, phi, lam)
    output['generated_circuits'] = generated_circuits
    # Remove all qnodes from the output dict since pickle can't process them (they are functions)
    # "Then he turned himself into a pickle, funniest shit I've ever seen!"
    output['wires'] = wires
    output['second_wires'] = wires
    output['indexes'] = indexes
    return output

def pl_insert_df(r, name, theta2, phi2, coupling_map):
    shots = 1024
    r['theta2'] = theta2
    r['phi2'] = phi2
    r['second_wires'] = []
    double_fault_circuits = []
    
    for qnode, wire, index in zip(r['generated_circuits'], r['wires'], r['indexes']):
        with qnode.tape as tape:
            for gate in tape.operations:
                if gate.id == 'FAULT':
                    for logical_qubit in tape.wires:
                        physical_qubit = coupling_map['logical2physical'][logical_qubit]
                        neighbouring_qubits = [el[1] for el in coupling_map['topology'] if el[0]==physical_qubit]
                        # Don't loop over all logical qubits, just over the ones connected! Kinda
                        for neighbor in neighbouring_qubits:
                            if neighbor not in coupling_map['physical2logical'].keys() or coupling_map['physical2logical'][neighbor] not in range(len(tape.wires)):
                                continue
                            else:
                                #log(f"-"*80+"\n"+f"Injecting circuit: {circuit[1]} theta1: {angle_pair1[0]} phi1: {angle_pair1[1]} theta2: {angle_pair2[0]} phi2: {angle_pair2[1]}")
                                second_fault_wire = coupling_map['physical2logical'][neighbor]
                                if second_fault_wire in gate.wires:
                                    continue
                                double_fault_circuit = pl_insert_df_gate(index, second_fault_wire, theta2, phi2)(deepcopy(qnode).func)
                                double_fault_device = qml.device('lightning.qubit', wires=len(tape.wires), shots=shots)
                                double_fault_qnode = qml.QNode(double_fault_circuit, double_fault_device)
                                double_fault_qnode()
                                log(f'Generated double fault circuit: {name} with faults on (wire1:{wire}, theta1:{gate.parameters[0]:.2f}, phi1:{gate.parameters[1]:.2f}) and (wire2:{second_fault_wire}, theta2:{theta2:.2f}, phi2:{phi2:.2f})')
                                #print(qml.draw(double_fault_qnode)())
                                # Due to multiple qubit gates, some double faults are repeated.
                                double_fault_circuits.append(double_fault_qnode)
                                r['second_wires'].append(second_fault_wire)
                                r['wires'].append(wire)
    log(f"{len(double_fault_circuits)} double fault circuits generated\n")
    r['generated_circuits'] = double_fault_circuits
    return r

def pl_inject(circuitStruct):
    circuitStruct.update(run_circuits( circuitStruct['base_circuit'], circuitStruct['generated_circuits'] ) )

def execute(circuits,
            angles={'theta1':np.arange(0, np.pi+0.01, np.pi/12), 
                    'phi1':np.arange(0, np.pi+0.01, np.pi/12), 
                    'theta2':np.arange(0, np.pi+0.01, np.pi/12), 
                    'phi2':np.arange(0, np.pi+0.01, np.pi/12)}, 
            coupling_map=None,
            results_folder="./tmp/"):
    results_folder = "./tmp/"
    #results = []
    results_names = []
    tstart = datetime.datetime.now()
    log(f"Start: {tstart}")
    for circuit in circuits:
        log(f"-"*80+"\n")
        tstartint = datetime.datetime.now()
        log(f"Circuit {circuit[1]} start: {tstartint}")
        angle_combinations = product(angles['theta1'], angles['phi1'])
        for angle_pair1 in angle_combinations:
            # Converting the circuit only once at the start of outer loop causes reference bugs (insight needed)
            if isinstance(circuit[0], qml.QNode):
                target_circuit = circuit[0]
            elif isinstance(circuit[0], qiskitQC):
                target_circuit = convert_qiskit_circuit(circuit)
            else:
                log(f"Unsupported {type(circuit[0])} object, injection stopped.")
                exit()

            log(f"-"*80+"\n"+f"Injecting circuit: {circuit[1]} theta1: {angle_pair1[0]} phi1: {angle_pair1[1]}")
            r = pl_insert(deepcopy(target_circuit), circuit[1], theta=angle_pair1[0], phi=angle_pair1[1])
            if coupling_map != None:
                angle_combinations_df = product(angles['theta2'], angles['phi2'])
                for angle_pair2 in angle_combinations_df:
                    s = pl_insert_df(deepcopy(r), circuit[1], angle_pair2[0], angle_pair2[1], coupling_map)
                    pl_inject(s)
                    #results.append(s)
                    tmp_name = f"{results_folder}{circuit[1]}_{angle_pair1[0]}_{angle_pair1[1]}_{angle_pair2[0]}_{angle_pair2[1]}.p.gz"
                    save_results([s], tmp_name)
                    results_names.append(tmp_name)
            else:
                pl_inject(r)
                #results.append(r)
                tmp_name = f"{results_folder}{circuit[1]}_{angle_pair1[0]}_{angle_pair1[1]}_0_0.p.gz"
                save_results([r], tmp_name)
                results_names.append(tmp_name)  
        tendint = datetime.datetime.now()
        log(f"Done: {tendint}\nElapsed time: {tendint-tstartint}\n"+"-"*80+"\n")
    tend = datetime.datetime.now()
    log(f"Done: {tend}\nTotal elapsed time: {tend-tstart}\n")

    # return results
    return results_names

def save_results(results, filename='./results.p.gz'):
    # Temporary fix for pickle.dump
    for circuit in results:
        del circuit['base_circuit']
        del circuit['generated_circuits']
    if not isdir(dirname(filename)):
        mkdir(dirname(filename))
    pickle.dump(results, gzip.open(filename, 'w'))
    log(f"Files saved to {filename}")