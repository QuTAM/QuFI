# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, Aer, IBMQ, execute
from qiskit.tools.jupyter import *
from qiskit.visualization import *

def build_circuit():
    qreg_q = QuantumRegister(2, 'q')
    creg_c = ClassicalRegister(2, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)
    
    circuit.h(qreg_q[0])
    circuit.h(qreg_q[1])
    circuit.h(qreg_q[1])
    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.h(qreg_q[0])
    circuit.h(qreg_q[1])
    circuit.x(qreg_q[0])
    circuit.h(qreg_q[1])
    circuit.x(qreg_q[1])
    circuit.h(qreg_q[1])
    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.x(qreg_q[0])
    circuit.h(qreg_q[1])
    circuit.h(qreg_q[0])
    circuit.x(qreg_q[1])
    circuit.h(qreg_q[1])
    circuit.measure(qreg_q[0], creg_c[0])
    circuit.measure(qreg_q[1], creg_c[1])
    return circuit

