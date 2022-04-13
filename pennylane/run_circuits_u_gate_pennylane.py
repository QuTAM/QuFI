#%%
from email.mime import base
import numpy as np
from itertools import product
import pickle, gzip
import datetime
from math import ceil
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, Aer, IBMQ, execute
#from qiskit.tools.jupyter import *
#from qiskit.visualization import *
from qiskit.test.mock import FakeSantiago, FakeCasablanca, FakeSydney
from qiskit.providers.aer.noise import NoiseModel
import pennylane as qml

from fault_injector_u_gate_pennylane import inject, insert_gate


fp=open("./run_circuits_u_gate_logging.txt", "a")

#%%
circuits = []

#import Grover
#grove = Grover.build_circuit()
#circuits.append( (grove, 'Grover') )

import Bernstein_Vazirani_pennylane as Bernstein_Vazirani
bv_4 = Bernstein_Vazirani.build_circuit(3, '101')
circuits.append( (bv_4, 'Bernstein-Vazirani_4') )

#bv_5 = Bernstein_Vazirani.build_circuit(4, '1010')
#circuits.append( (bv_5, 'Bernstein-Vazirani_5') )
#
#bv_6 = Bernstein_Vazirani.build_circuit(5, '10101')
#circuits.append( (bv_6, 'Bernstein-Vazirani_6') )
#
#bv_7 = Bernstein_Vazirani.build_circuit(6, '101010')
#circuits.append( (bv_7, 'Bernstein-Vazirani_7') )


#import Deutsch_Jozsa_pennylane as Deutsch_Jozsa
#dj_4 = Deutsch_Jozsa.build_circuit(3, '101')
#circuits.append( (dj_4, 'Deutsch-Jozsa_4') )
#
#dj_5 = Deutsch_Jozsa.build_circuit(4, '1010')
#circuits.append( (dj_5, 'Deutsch-Jozsa_5') )
#
#dj_6 = Deutsch_Jozsa.build_circuit(5, '10101')
#circuits.append( (dj_6, 'Deutsch-Jozsa_6') )
#
#dj_7 = Deutsch_Jozsa.build_circuit(6, '101010')
#circuits.append( (dj_7, 'Deutsch-Jozsa_7') )

#import inverseQFT_pennylane as inverseQFT
#qft4 = inverseQFT.build_circuit(4)
#circuits.append( (qft4, 'inverseQFT4') )
#qft5 = inverseQFT.build_circuit(5)
#circuits.append( (qft5, 'inverseQFT5') )
#qft6 = inverseQFT.build_circuit(6)
#circuits.append( (qft6, 'inverseQFT6') )
#qft7 = inverseQFT.build_circuit(7)
#circuits.append( (qft7, 'inverseQFT7') )


#%%
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
        print(sum(res_dict.values()), shots)
        a = input()
    return res_dict

def run_circuits(base_circuit, generated_circuits):
    # Execute golden circuit simulation without noise
    print('running circuits')
    gold_device = qml.device('qiskit.aer', wires=base_circuit.device.num_wires, backend='qasm_simulator')
    gold_qnode = qml.QNode(base_circuit.func, gold_device)
    answer_gold = probs_to_counts(gold_qnode(), base_circuit.device.num_wires)

    # Execute golden circuit simulation with noise
    device_backend = FakeSantiago()
    noise_model = NoiseModel.from_backend(device_backend)
    gold_device_noisy = qml.device('qiskit.aer', wires=base_circuit.device.num_wires, backend='aer_simulator', noise_model=noise_model)
    gold_qnode_noisy = qml.QNode(base_circuit.func, gold_device_noisy)
    answer_gold_noise = probs_to_counts(gold_qnode_noisy(), base_circuit.device.num_wires)

    # Execute injection circuit simulations without noise
    answers = []
    for c, i in zip(generated_circuits, range(0, len(generated_circuits))):
        inj_device = qml.device('qiskit.aer', wires=c.device.num_wires, backend='qasm_simulator')
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

#%%
def convert_circuit(qiskit_circuit):
    shots = 1024
    measure_list = [g[1][0].index for g in qiskit_circuit[0].data if g[0].name == 'measure']
    qregs = qiskit_circuit[0].num_qubits
    pl_circuit = qml.load(qiskit_circuit[0], format='qiskit')
    device = qml.device("lightning.qubit", wires=qregs, shots=shots)
    @qml.qnode(device)
    def conv_circuit():
        pl_circuit(wires=range(qregs))
        return qml.probs(wires=measure_list) #[qml.expval(qml.PauliZ(i)) for i in range(qregs)]
    # Do NOT remove this print, else the qnode can't bind the function before exiting convert_circuit()'s context
    print(qml.draw(conv_circuit)())
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
                print('circuit:', name, 'gate', op.name, 'theta:', theta, 'phi:', phi)
                #print(qml.draw(transformed_qnode)())
                mycircuits.append(transformed_qnode)
                inj_info.append(wire)
            index = index + 1
        print('{} circuits generated'.format(len(mycircuits)))
        return mycircuits, inj_info

def pl_inject(circuit, name, theta=0, phi=0, lam=0):
    print('running {}'.format(name))
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

#%%
theta_values = [0, np.pi/4, np.pi/2, np.pi*3/4, np.pi]#np.arange(0, np.pi+0.01, np.pi/12) # 0 <= theta <= pi # [0, np.pi/2]
phi_values = [0, np.pi/2]#np.arange(0, 2*np.pi, np.pi/12) # 0 <= phi < 2pi # [0]
results = []
for qiskit_circuit in circuits:
    print('-'*80)
    fp.write('-'*80)
    fp.write('\n')
    tstart = datetime.datetime.now()
    print('start:',tstart)
    fp.write('start:'+str(tstart))
    fp.write('\n')
    fp.flush()
    angle_values = product(theta_values, phi_values)
    for angles in angle_values:
        # Converting the circuit only once at the start of outer loop causes reference bugs (insight needed)
        conv_circuit = convert_circuit(qiskit_circuit)
        print('-'*80)
        fp.write('-'*80)
        fp.write('\n')
        print("\n")
        print('circuit:', qiskit_circuit[1], 'theta:', angles[0], 'phi:', angles[1])
        fp.write('circuit: '+str(qiskit_circuit[1])+ ' theta: '+str(angles[0]) +' phi: '+str(angles[1]))
        fp.write('\n')
        fp.flush()
        r = pl_inject(conv_circuit, qiskit_circuit[1], theta=angles[0], phi=angles[1])
        results.append(r)
    tend = datetime.datetime.now()
    print('done:',tend)
    fp.write('done:'+str(tend))
    print('Elapsed time:',tend-tstart)
    fp.write('Elapsed time:'+str(tend-tstart))
    fp.write('\n')
    print('-'*80)
    fp.write('-'*80)
    fp.write('\n')

#%%
# Careful! Very verbose
print(results)


#%%
filename_output = '../results/u_gate_15degrees_step_bv_4_pennylane.p.gz'
pickle.dump(results, gzip.open(filename_output, 'w'))
print('files saved to:',filename_output)
fp.write('files saved to:'+str(filename_output))
fp.close()


# %%
