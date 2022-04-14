import pennylane as qml
from numpy import pi
import math

def qft_rotations(wires):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    if wires == 0:
        return
    wires -= 1
    qml.Hadamard(wires=wires)
    for qubit in range(wires):
        qml.ControlledPhase(pi/2**(wires-qubit), wires=[qubit, wires])
    # At the end of our function, we call the same function again on
    # the next qubits (we reduced n by one earlier in the function)
    qft_rotations(wires)
    
def swap_registers(wires):
    for qubit in range(wires//2):
        qml.SWAP(wires=[qubit, wires-qubit-1])

def qft(wires):
    """QFT on the first n qubits in circuit"""
    qft_rotations(wires)
    swap_registers(wires)

def build_circuit(nqubits):
    number = '10'*math.ceil(nqubits/2)
    number = int(number[0:nqubits], 2) # code a number interleaving 1 and 0 in binary

    dev = qml.device("default.qubit", wires=nqubits)

    @qml.qnode(dev)
    def circuit():
        powerTwo = 2**(nqubits-1)
        for qubit in range(nqubits):
            qml.Hadamard(qubit)
            qml.PhaseShift(phi=number*pi/powerTwo, wires=qubit)
            powerTwo = powerTwo/2
        #qml.QFT(wires=range(nqubits)).inv().decomposition()
        qml.adjoint(qft)(nqubits)

        # Measurement
        return qml.probs(wires=range(nqubits))

    circuit()

    return circuit