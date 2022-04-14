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
import sys
sys.path.insert(0,'..')


#%%
circuits = []

#################### Qiskit defined circuits ###################

#from circuits import Grover
#grove = Grover.build_circuit()
#circuits.append( (grove, 'Grover') )

#from circuits import Bernstein_Vazirani
#bv_4 = Bernstein_Vazirani.build_circuit(3, '101')
#circuits.append( (bv_4, 'Bernstein-Vazirani_4') )

#bv_5 = Bernstein_Vazirani.build_circuit(4, '1010')
#circuits.append( (bv_5, 'Bernstein-Vazirani_5') )
#bv_6 = Bernstein_Vazirani.build_circuit(5, '10101')
#circuits.append( (bv_6, 'Bernstein-Vazirani_6') )
#bv_7 = Bernstein_Vazirani.build_circuit(6, '101010')
#circuits.append( (bv_7, 'Bernstein-Vazirani_7') )

#from circuits import Deutsch_Jozsa
#dj_4 = Deutsch_Jozsa.build_circuit(3, '101')
#circuits.append( (dj_4, 'Deutsch-Jozsa_4') )
#dj_5 = Deutsch_Jozsa.build_circuit(4, '1010')
#circuits.append( (dj_5, 'Deutsch-Jozsa_5') )
#dj_6 = Deutsch_Jozsa.build_circuit(5, '10101')
#circuits.append( (dj_6, 'Deutsch-Jozsa_6') )
#dj_7 = Deutsch_Jozsa.build_circuit(6, '101010')
#circuits.append( (dj_7, 'Deutsch-Jozsa_7') )

#from circuits import inverseQFT
#qft4 = inverseQFT.build_circuit(4)
#circuits.append( (qft4, 'inverseQFT4') )
#qft5 = inverseQFT.build_circuit(5)
#circuits.append( (qft5, 'inverseQFT5') )
#qft6 = inverseQFT.build_circuit(6)
#circuits.append( (qft6, 'inverseQFT6') )
#qft7 = inverseQFT.build_circuit(7)
#circuits.append( (qft7, 'inverseQFT7') )

################## Pennylane defined circuits ##################

#import Grover_pennylane
#grover = Grover_pennylane.build_circuit()
#circuits.append( (grover, 'Grover') )

import Bernstein_Vazirani_pennylane
bv4_p = Bernstein_Vazirani_pennylane.build_circuit(3, '101')
circuits.append((bv4_p, 'Bernstein-Vazirani_4'))
#bv5_p = Bernstein_Vazirani_pennylane.build_circuit(4, '1010')
#circuits.append((bv5_p, 'Bernstein-Vazirani_5'))
#bv6_p = Bernstein_Vazirani_pennylane.build_circuit(5, '10101')
#circuits.append((bv6_p, 'Bernstein-Vazirani_6'))
#bv7_p = Bernstein_Vazirani_pennylane.build_circuit(6, '101010')
#circuits.append((bv7_p, 'Bernstein-Vazirani_7'))

#import Deutsch_Jozsa_pennylane
#dj_4_p = Deutsch_Jozsa_pennylane.build_circuit(3, '101')
#circuits.append((dj_4_p, 'Deutsch-Jozsa_4'))
#dj_5_p = Deutsch_Jozsa_pennylane.build_circuit(4, '1010')
#circuits.append((dj_5_p, 'Deutsch-Jozsa_5'))
#dj_6_p = Deutsch_Jozsa_pennylane.build_circuit(5, '10101')
#circuits.append((dj_6_p, 'Deutsch-Jozsa_6'))
#dj_7_p = Deutsch_Jozsa_pennylane.build_circuit(6, '101010')
#circuits.append((dj_7_p, 'Deutsch-Jozsa_7'))

#import inverseQFT_pennylane
#iqft4_p = inverseQFT_pennylane.build_circuit(4)
#circuits.append( (iqft4_p, 'inverseQFT4') )
#iqft5_p = inverseQFT_pennylane.build_circuit(5)
#circuits.append( (iqft5_p, 'inverseQFT5') )
#iqft6_p = inverseQFT_pennylane.build_circuit(6)
#circuits.append( (iqft6_p, 'inverseQFT6') )
#iqft7_p = inverseQFT_pennylane.build_circuit(7)
#circuits.append( (iqft7_p, 'inverseQFT7') )

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
        print("Rounding error! ", sum(res_dict.values()), shots)
    return res_dict

def run_circuits(base_circuit, generated_circuits):
    # Execute golden circuit simulation without noise
    print('running circuits')
    gold_device = qml.device('lightning.qubit', wires=base_circuit.device.num_wires)
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

#%%
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
                print(f'Generated circuit: {name} with fault on ({op.name}, wire:{wire}), theta = {theta}, phi = {phi}')
                #print(qml.draw(transformed_qnode)())
                mycircuits.append(transformed_qnode)
                inj_info.append(wire)
            index = index + 1
        print('{} circuits generated'.format(len(mycircuits)))
        return mycircuits, inj_info

def pl_inject(circuit, name, theta=0, phi=0, lam=0):
    output = {'name': name, 'base_circuit':circuit, 'theta':theta, 'phi':phi, 'lambda':lam}
    output['pennylane_version'] = qml.version()
    #print(qml.draw(circuit)())
    output['circuits_injections'], wires = pl_generate_circuits(circuit, name, theta, phi, lam)
    output.update(run_circuits( output['base_circuit'], output['circuits_injections'] ) )
    # Remove all qnodes from the output dict since pickle can't process them (they are functions)
    # "Then he turned himself into a pickle, funniest s.inv()hit I've ever seen!"
    output['base_circuit'] = None
    output['circuits_injections'] = wires
    return output

#%%
theta_values = np.arange(0, np.pi+0.01, np.pi/12) # 0 <= theta <= pi
phi_values = np.arange(0, 2*np.pi, np.pi/12) # 0 <= phi < 2pi
results = []
fp=open("./run_circuits_u_gate_logging.txt", "a")
tstart = datetime.datetime.now()
print('Start:',tstart)
fp.write('Start:'+str(tstart))
for circuit in circuits:
    print('-'*80)
    fp.write('-'*80)
    fp.write('\n')
    tstartint = datetime.datetime.now()
    print('Circuit', circuit[1], 'start:',tstartint)
    fp.write('Circuit '+circuit[1]+' start:'+str(tstartint))
    fp.write('\n')
    fp.flush()
    angle_values = product(theta_values, phi_values)
    for angles in angle_values:
        # Converting the circuit only once at the start of outer loop causes reference bugs (insight needed)
        if isinstance(circuit[0], qml.QNode):
            target_circuit = circuit[0]
        else:
            target_circuit = convert_qiskit_circuit(circuit)
        print('-'*80)
        fp.write('-'*80)
        fp.write('\n')
        print("\n")
        print('Injecting circuit:', circuit[1], 'theta:', angles[0], 'phi:', angles[1])
        fp.write('Injecting circuit: '+str(circuit[1])+ ' theta: '+str(angles[0]) +' phi: '+str(angles[1]))
        fp.write('\n')
        fp.flush()
        r = pl_inject(target_circuit, circuit[1], theta=angles[0], phi=angles[1])
        results.append(r)
    tendint = datetime.datetime.now()
    print('done:',tendint)
    fp.write('done:'+str(tendint))
    print('Elapsed time:',tendint-tstartint)
    fp.write('Elapsed time:'+str(tendint-tstartint))
    fp.write('\n')
    print('-'*80)
    fp.write('-'*80)
    fp.write('\n')
tend = datetime.datetime.now()
fp.write('done:'+str(tend))
print('Total elapsed time:',tend-tstart) 
fp.write('Total elapsed time:'+str(tend-tstart)) 

#%%
# Careful! Very verbose
print(results)

#%%
filename_output = '../results/u_gate_15degrees_step_bv_4_pennylane_lightning.p.gz'
pickle.dump(results, gzip.open(filename_output, 'w'))
print('files saved to:',filename_output)
fp.write('files saved to:'+str(filename_output))
fp.close()
# %%
