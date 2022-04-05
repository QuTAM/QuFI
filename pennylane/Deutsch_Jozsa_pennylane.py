# importing Qiskit
from qiskit import QuantumCircuit, assemble, transpile



def build_circuit(n, b_str):
    # set the length of the n-bit input string. 
    #n = 9
    
    
    balanced_oracle = QuantumCircuit(n+1)
    #b_str = "101101101"
    
    # Place X-gates
    for qubit in range(len(b_str)):
        if b_str[qubit] == '1':
            balanced_oracle.x(qubit)
    
    # Use barrier as divider
    balanced_oracle.barrier()
    
    # Controlled-NOT gates
    for qubit in range(n):
        balanced_oracle.cx(qubit, n)
    
    balanced_oracle.barrier()
    
    # Place X-gates
    for qubit in range(len(b_str)):
        if b_str[qubit] == '1':
            balanced_oracle.x(qubit)
    
    
    dj_circuit = QuantumCircuit(n+1, n)
    
    # Apply H-gates
    for qubit in range(n):
        dj_circuit.h(qubit)
    
    # Put qubit in state |->
    dj_circuit.x(n)
    dj_circuit.h(n)
    
    # Add oracle
    #dj_circuit += balanced_oracle
    dj_circuit.compose(balanced_oracle, inplace=True)
    
    # Repeat H-gates
    for qubit in range(n):
        dj_circuit.h(qubit)
    dj_circuit.barrier()
    
    # Measure
    for i in range(n):
        dj_circuit.measure(i, i)
    
    return dj_circuit

