# importing Qiskit
import pennylane as qml

def build_circuit(n, b_str):
    # set the length of the n-bit input string. 
    #n = 9
    
    # We need a circuit with n qubits, plus one auxiliary qubit
    dev = qml.device("default.qubit", wires=n+1)

    @qml.qnode(dev)
    def circuit():        
        # Apply H-gates
        for qubit in range(n):
            qml.Hadamard(wires=qubit)
        
        # Put qubit in state |->
        qml.PauliX(wires=n)
        qml.Hadamard(wires=n)
        
        # Add oracle
        # Place X-gates
        for qubit in range(len(b_str)):
            if b_str[qubit] == '1':
                qml.PauliX(wires=qubit)
        
        # Controlled-NOT gates
        for qubit in range(n):
            qml.CNOT(wires=[qubit, n])
                
        # Place X-gates
        for qubit in range(len(b_str)):
            if b_str[qubit] == '1':
                qml.PauliX(wires=qubit)
        
        # Repeat H-gates
        for qubit in range(n):
            qml.Hadamard(wires=qubit)
        
        return qml.probs(wires=range(n))

    return circuit