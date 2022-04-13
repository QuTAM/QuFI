# Importing standard Qiskit libraries
import pennylane as qml

def build_circuit():
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=1)
        qml.CNOT(wires=[0, 1])
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.PauliX(wires=0)
        qml.Hadamard(wires=1)
        qml.PauliX(wires=1)
        qml.Hadamard(wires=1)
        qml.CNOT(wires=[0, 1])
        qml.PauliX(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=0)
        qml.PauliX(wires=1)
        qml.Hadamard(wires=1)
        
        return qml.probs(wires=range(2))

    return circuit