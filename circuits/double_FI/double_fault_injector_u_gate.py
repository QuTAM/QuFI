import numpy as np
import pickle, gzip
import copy
from itertools import product

# importing Qiskit
from qiskit import *
from qiskit import Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.test.mock import FakeSantiago, FakeGuadalupe, FakeCasablanca, FakeSydney
#Santiago 5 qubits
#Casablanca 7 qubits
#Guadalupe 16 qubits
#Sydney 27 qubits

theta_values = np.arange(0, np.pi+0.01, np.pi/12) # 0 <= theta <= pi
phi_values = np.arange(0, np.pi+0.01, np.pi/12) # 0 <= phi <= pi
#phi_values = np.arange(0, 2*np.pi, np.pi/12) # 0 <= phi < 2pi

def insert_second_gate(base_circuit, index, qubit, theta=0, phi=0, lam=0):
    qregs = len(base_circuit.qubits)
    cregs = len(base_circuit.clbits)
    metadata = copy.deepcopy(base_circuit.metadata)
    metadata['second_qubit'] = qubit.index
    metadata['second_gate_index'] = index
    metadata['second_theta'] = theta
    metadata['second_phi'] = phi

    mycircuit = QuantumCircuit(qregs, cregs, metadata=metadata)

    for i in range(0, len(base_circuit.data)):
        mycircuit.data.append(base_circuit.data[i])
        if i == index:
            # mycircuit.barrier(range(0, n+1))
            mycircuit.u(theta, phi, lam, qubit)
            # mycircuit.barrier(range(0, n+1))
    return mycircuit

def generate_second_fault_circuits_same_angle(base_circuit, theta=0, phi=0):
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
        for qb in qubits:
            if qb.index != 1: 
                mycircuits.append(insert_second_gate(base_circuit, i, qb, theta, phi))
    
    return mycircuits


def generate_second_fault_circuits(base_circuit):
    mycircuits = []
    qregs = len(base_circuit.qubits)
    cregs = len(base_circuit.clbits)

    angle_values = product(theta_values, phi_values)

    for angles in angle_values:
        theta = angles[0]
        phi = angles[1]
        for i in range(0, len(base_circuit.data)):
            gate = base_circuit.data[i][0]
            qubits = base_circuit.data[i][1]

            # Ignore barriers and measurement gates
            if isinstance(gate, qiskit.circuit.measure.Measure) or isinstance(gate, qiskit.circuit.barrier.Barrier):
                continue

            # If gate are not using a single qubit, insert one gate after each qubit
            for qb in qubits:
                if qb.index != 1: 
                    mycircuits.append(insert_second_gate(base_circuit, i, qb, theta, phi))
    
    return mycircuits

def insert_gate(base_circuit, index, qubit, theta=0, phi=0, lam=0):
    qregs = len(base_circuit.qubits)
    cregs = len(base_circuit.clbits)
    metadata = {'qubit':qubit.index, 'gate_index':index, 'gate_inserted':'u', 'theta':theta, 'phi':phi}
    mycircuit = QuantumCircuit(qregs, cregs, metadata=metadata)

    for i in range(0, len(base_circuit.data)):
        mycircuit.data.append(base_circuit.data[i])
        if i == index:
            # mycircuit.barrier(range(0, n+1))
            mycircuit.u(theta, phi, lam, qubit)
            # mycircuit.barrier(range(0, n+1))
    return mycircuit

def generate_circuits(base_circuit, theta=0, phi=0, lam=0):
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
        for qb in qubits:
            if qb.index == 1: # insert faults only on qubit 1, second faults will be on qubits 0 and 2
                circuits_double_fault = generate_second_fault_circuits_same_angle(insert_gate(base_circuit, i, qb, theta, phi, lam), theta, phi)
                mycircuits.extend(circuits_double_fault)
    print('{} circuits generated for theta {} and phi {}'.format(len(mycircuits), theta, phi))
    return mycircuits

def run_circuits(base_circuit, generated_circuits):
    print('running circuits')
    aer_sim = Aer.get_backend('qasm_simulator')
    shots = 1024
    qobj_gold = assemble(base_circuit)
    #print("running base circuit")
    results_gold = aer_sim.run(qobj_gold).result()
    answer_gold = results_gold.get_counts()
    
    device_backend = FakeSydney()#FakeSantiago()
    sim_santiago = AerSimulator.from_backend(device_backend)
    # Transpile the circuit for the noisy basis gates
    tcirc = transpile(base_circuit, sim_santiago)
    # Execute noisy simulation and get counts
    result_noise = sim_santiago.run(tcirc).result()
    answer_gold_noise = result_noise.get_counts(0)

    answers = []
    for c, i in zip(generated_circuits, range(0, len(generated_circuits))):
        #print("running circuit i={}".format(i))
        qobj = assemble(c)
        results = aer_sim.run(qobj).result()
        answer = results.get_counts()
        answers.append(answer)

    answers_noise = []
    for c, i in zip(generated_circuits, range(0, len(generated_circuits))):
        # Transpile the circuit for the noisy basis gates
        tcirc = transpile(c, sim_santiago)
        # Execute noisy simulation and get counts
        result_noise = sim_santiago.run(tcirc).result()
        answer_noise = result_noise.get_counts(0)
        answers_noise.append(answer_noise)

    return {'output_gold':answer_gold, 'output_injections':answers
            , 'output_gold_noise':answer_gold_noise, 'output_injections_noise':answers_noise
            , 'noise_target':str(device_backend)
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

def inject(circuit, name ):
    print('running {}'.format(name))
    output = {'name': name, 'base_circuit':circuit}
    output['qiskit_version'] = qiskit.__qiskit_version__
    #output['circuits_injections'] = generate_circuits(circuit, theta, phi, lam)
    #output.update(run_circuits( output['base_circuit'], output['circuits_injections'] ) )
    output['circuits_injections'] = []
    #theta_values = np.arange(0, np.pi+0.01, np.pi/12) # 0 <= theta <= pi
    #phi_values = np.arange(0, 2*np.pi, np.pi/12) # 0 <= phi < 2pi
    angle_values = product(theta_values, phi_values)
    for angles in angle_values:
        theta = angles[0]
        phi = angles[1]
        output['circuits_injections'].extend(generate_circuits(circuit, theta, phi))
    print('{} total circuits generated'.format(len(output['circuits_injections'])))
    #output.update(run_circuits( output['base_circuit'], output['circuits_injections'] ) )
    print_metadata(output['circuits_injections'])
    return output


def print_metadata(circuits):
    for circ in circuits:
        print(circ.metadata)
