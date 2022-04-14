# importing Qiskit
import pennylane as qml

def build_circuit(n, s):
    #n = 9 # number of qubits used to represent s
    #s = '011011011'   # the hidden binary string
    
    # We need a circuit with n qubits, plus one auxiliary qubit
    dev = qml.device("default.qubit", wires=n+1)

    @qml.qnode(dev)
    def circuit():
        # put auxiliary in state |->
        qml.Hadamard(wires=n)
        qml.PauliZ(wires=n)

        # Apply Hadamard gates before querying the oracle
        for i in range(n):
            qml.Hadamard(wires=i)
        
        # Apply the inner-product oracle
        si = s[::-1] # reverse s to fit qiskit's qubit ordering
        for q in range(n):
            if si[q] == '0':
                qml.Identity(wires=q)
            else:
                qml.CNOT(wires=[q, n])
        
        #Apply Hadamard gates after querying the oracle
        for i in range(n):
            qml.Hadamard(wires=i)

        return qml.probs(wires=range(n))

    circuit()

    return circuit

