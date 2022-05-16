from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi
import math

def qft_rotations(circuit, n):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(pi/2**(n-qubit), qubit, n)
    # At the end of our function, we call the same function again on
    # the next qubits (we reduced n by one earlier in the function)
    qft_rotations(circuit, n)
    
def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft(circuit, n):
    """QFT on the first n qubits in circuit"""
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit

def inverse_qft(circuit, n):
    """Does the inverse QFT on the first n qubits in circuit"""
    # First we create a QFT circuit of the correct size:
    qft_circ = qft(QuantumCircuit(n), n)
    # Then we take the inverse of this circuit
    invqft_circ = qft_circ.inverse()
    # And add it to the first n qubits in our existing circuit
    circuit.append(invqft_circ, circuit.qubits[:n])
    return circuit.decompose() # .decompose() allows us to see the individual gates

def build_circuit(nqubits):
    number = '10'*math.ceil(nqubits/2)
    number = int(number[0:nqubits], 2) # code a number interleaving 1 and 0 in binary
    powerTwo = 2**(nqubits-1)
    qc = QuantumCircuit(nqubits,nqubits)
    for qubit in range(nqubits):
        qc.h(qubit)
        qc.p(number*pi/powerTwo,qubit)
        powerTwo = powerTwo/2
    qc = inverse_qft(qc, nqubits)
    #qc.measure_all()
    # Measurement
    #qc.barrier()
    for i in range(nqubits):
        print(i)
        qc.measure(i, i)
    return qc

