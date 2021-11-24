# importing Qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile, assemble


def build_circuit(n, s):
    #n = 9 # number of qubits used to represent s
    #s = '011011011'   # the hidden binary string
    
    #%%
    # We need a circuit with n qubits, plus one auxiliary qubit
    # Also need n classical bits to write the output to
    bv_circuit = QuantumCircuit(n+1, n)
    
    # put auxiliary in state |->
    bv_circuit.h(n)
    bv_circuit.z(n)
    
    # Apply Hadamard gates before querying the oracle
    for i in range(n):
        bv_circuit.h(i)
        
    # Apply barrier 
    bv_circuit.barrier()
    
    # Apply the inner-product oracle
    s = s[::-1] # reverse s to fit qiskit's qubit ordering
    for q in range(n):
        if s[q] == '0':
            bv_circuit.i(q)
        else:
            bv_circuit.cx(q, n)
            
    # Apply barrier 
    bv_circuit.barrier()
    
    #Apply Hadamard gates after querying the oracle
    for i in range(n):
        bv_circuit.h(i)
    
    # Measurement
    for i in range(n):
        bv_circuit.measure(i, i)

    return bv_circuit

