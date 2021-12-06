import numpy as np
import pickle, gzip
import copy
import datetime
from itertools import product

# importing Qiskit
from qiskit import *
from qiskit import Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.test.mock import FakeSantiago, FakeCasablanca, FakeSydney
#Santiago 5 qubits
#Casablanca 7 qubits
#Guadalupe 16 qubits
#Sydney 27 qubits

theta_values = np.arange(0, np.pi+0.01, np.pi/12) # 0 <= theta <= pi
phi_values = np.arange(0, np.pi+0.01, np.pi/12) # 0 <= phi <= pi
#phi_values = np.arange(0, 2*np.pi, np.pi/12) # 0 <= phi < 2pi


def insert_gate(base_circuit, index, qubit, theta=0, phi=0, lam=0):
    qregs = len(base_circuit.qubits)
    cregs = len(base_circuit.clbits)
    if base_circuit.metadata is None:
        metadata = {'first_qubit':qubit.index, 'first_gate_index':index, 'gate_inserted':'u', 'first_theta':theta, 'first_phi':phi}
    else:
        metadata = copy.deepcopy(base_circuit.metadata)
        metadata.update({'second_qubit':qubit.index, 'second_gate_index':index, 'second_theta':theta, 'second_phi':phi})
    mycircuit = QuantumCircuit(qregs, cregs, metadata=metadata)

    for i in range(0, len(base_circuit.data)):
        mycircuit.data.append(base_circuit.data[i])
        if i == index and qubit in mycircuit.qubits:
            # mycircuit.barrier(range(0, n+1))
            mycircuit.u(theta, phi, lam, qubit)
            # mycircuit.barrier(range(0, n+1))
    return mycircuit

def generate_circuits(base_circuit, backend, transpiled_circuit, theta=0, phi=0, lam=0):
    # qubit_couples = list(set(tuple(sorted(l)) for l in backend.configuration().to_dict()['coupling_map'])) # to avoid duplicates?
    qubit_couples = backend.configuration().to_dict()['coupling_map'] #actual physical neighboring qubits
    mapping = transpiled_circuit._layout.get_virtual_bits()  #from logical to physical qubits
    mapping_inv = transpiled_circuit._layout.get_physical_bits()  #from physical to logical qubits

    mycircuits = []
    qregs = len(base_circuit.qubits)
    cregs = len(base_circuit.clbits)

    for i in range(0, len(base_circuit.data)):
        gate = base_circuit.data[i][0]
        qubits = base_circuit.data[i][1]

        # Ignore barriers and measurement gates
        if isinstance(gate, qiskit.circuit.measure.Measure) or isinstance(gate, qiskit.circuit.barrier.Barrier):
            continue

        # If gate are not using a single qubit, insert one gate after each qubit
        for logical_qb in qubits:
            physical_qubit = mapping[logical_qb]
            neighbors_qubits = [el[1] for el in qubit_couples if el[0]==physical_qubit] 
            
            single_fault_circuit = insert_gate(base_circuit, i, logical_qb, theta, phi, lam)

            for neighbor in neighbors_qubits:
                if mapping_inv[neighbor] not in base_circuit.qubits:
                    continue
                for second_theta, second_phi in product(np.arange(0, theta+0.01, np.pi/12), np.arange(0, phi+0.01, np.pi/12)):
                    mycircuits.append(insert_gate(single_fault_circuit, i, mapping_inv[neighbor], second_theta, second_phi, lam))


    print('{} circuits generated for theta {} and phi {}'.format(len(mycircuits), theta, phi))
    return mycircuits

def showPercentage(what, so_far, size):
    so_far += 1
    sys.stdout.write(
               "\r%s  %s / %s  (%.2f%%)" % (
                    what, so_far, size,
                    so_far/size*100))
    sys.stdout.flush()

def run_circuits(base_circuit, transpiled_circuit, generated_circuits, backend):
    print('running circuits')
    aer_sim = Aer.get_backend('qasm_simulator')
    shots = 1024
    qobj_gold = assemble(base_circuit)
    #print("running base circuit")
    results_gold = aer_sim.run(qobj_gold).result()
    answer_gold = results_gold.get_counts()
    
    
    simulator = AerSimulator.from_backend(backend)
    # Execute noisy simulation and get counts
    result_noise = simulator.run(transpiled_circuit).result()
    answer_gold_noise = result_noise.get_counts(0)

    answers = []
    #for c, i in zip(generated_circuits, range(0, len(generated_circuits))):
    #    #print("running circuit i={}".format(i))
    #    qobj = assemble(c)
    #    results = aer_sim.run(qobj).result()
    #    answer = results.get_counts()
    #    answers.append(answer)

    answers_noise = []
    totalCirc = len(generated_circuits)
    for c, i in zip(generated_circuits, range(0, len(generated_circuits))):
        showPercentage("Executing circuits with noise", i, totalCirc)
        # Transpile the circuit for the noisy basis gates
        tcirc = transpile(c, simulator, optimization_level=3)
        # Execute noisy simulation and get counts
        result_noise = simulator.run(tcirc).result()
        answer_noise = result_noise.get_counts(0)
        answers_noise.append(answer_noise)

    return {'output_gold':answer_gold, 'output_injections':answers
            , 'output_gold_noise':answer_gold_noise, 'output_injections_noise':answers_noise
            , 'noise_target':str(backend)
            }


def computeQVF(gold, answers):
    # Quantum Vulnerability Factor
    max_good = max(gold, key=gold.get)#check
    good_count = 0
    for a in answers:
        max_key = max(a, key=a.get)
        if max_key==max_good:
            good_count+=1
    
    qvf = (good_count/len(answers))
    return qvf


def computeQSR(gold, answers, threshold):
    # Quantum Success Rate
    max_good = max(gold, key=gold.get)#check
    good_count = 0
    for a in answers:
        max_key = max(a, key=a.get)
        if (max_key == max_good) and (a[max_key]>threshold):
            good_count += 1

    qsr = (good_count/len(answers))
    return qsr

def inject(circuit, name, backend):
    print('running {}'.format(name))
    output = {'name': name, 'base_circuit':circuit}
    output['qiskit_version'] = qiskit.__qiskit_version__
    output['circuits_injections'] = []

    simulator = AerSimulator.from_backend(backend)
    # Transpile the circuit for the noisy basis gates
    tcirc = transpile(circuit, simulator, optimization_level=3)

    angle_values = product(theta_values, phi_values)
    for angles in angle_values:
        theta = angles[0]
        phi = angles[1]
        generated_circuits = generate_circuits(circuit, backend, tcirc, theta, phi)
        output['circuits_injections'].extend(generated_circuits)
    print('{} total circuits generated'.format(len(output['circuits_injections'])))
    print('running:',datetime.datetime.now())
    output.update(run_circuits( output['base_circuit'], tcirc, output['circuits_injections'], backend ) )
    #output.update(run_circuits( output['base_circuit'], tcirc, output['circuits_injections'][:1000], backend ) ) # test how long it takes to run one thousand circuits
    print('done:',datetime.datetime.now())
    print_metadata(output['circuits_injections'][:200])
    check_metadata(output['circuits_injections'])
    return output


def print_metadata(circuits):
    for circ in circuits:
        print(circ.metadata['first_qubit'],circ.metadata['second_qubit'])

def check_metadata(circuits):
    for circ in circuits:
        if circ.metadata['first_qubit'] == circ.metadata['second_qubit']:
            print('\n\tfound qubit pair using the same qubit!!!')
            print(circ.metadata['first_qubit'],circ.metadata['second_qubit'])
