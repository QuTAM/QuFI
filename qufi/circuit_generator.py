from numpy import pi
import math
import pennylane as qml

class BernsteinVazirani:

    default_n = 3
    default_s = '101'

    def build_circuit(n=default_n, s=default_s):
        #n = 9 # number of qubits used to represent s
        #s = '011011011'   # the hidden binary string
        
        # We need a circuit with n qubits, plus one auxiliary qubit
        dev = qml.device("lightning.qubit", wires=n+1)

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

class DeutschJozsa:

    default_n = 3
    default_s = '101'

    def build_circuit(n=default_n, s=default_s):
        # set the length of the n-bit input string. 
        #n = 9
        
        # We need a circuit with n qubits, plus one auxiliary qubit
        dev = qml.device("lightning.qubit", wires=n+1)

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
            for qubit in range(len(s)):
                if s[qubit] == '1':
                    qml.PauliX(wires=qubit)
            
            # Controlled-NOT gates
            for qubit in range(n):
                qml.CNOT(wires=[qubit, n])
                    
            # Place X-gates
            for qubit in range(len(s)):
                if s[qubit] == '1':
                    qml.PauliX(wires=qubit)
            
            # Repeat H-gates
            for qubit in range(n):
                qml.Hadamard(wires=qubit)
            
            return qml.probs(wires=range(n))

        circuit()

        return circuit

class Grover:

    def build_circuit():
        dev = qml.device("lightning.qubit", wires=2)

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

        circuit()

        return circuit

class IQFT:

    def qft_rotations(wires):
        """Performs qft on the first n qubits in circuit (without swaps)"""
        if wires == 0:
            return
        wires -= 1
        qml.Hadamard(wires=wires)
        for qubit in range(wires):
            qml.ControlledPhaseShift(pi/2**(wires-qubit), wires=[qubit, wires])
        # At the end of our function, we call the same function again on
        # the next qubits (we reduced n by one earlier in the function)
        IQFT.qft_rotations(wires)
        
    def swap_registers(wires):
        for qubit in range(wires//2):
            qml.SWAP(wires=[qubit, wires-qubit-1])

    def qft(wires):
        """QFT on the first n qubits in circuit"""
        IQFT.qft_rotations(wires)
        IQFT.swap_registers(wires)

    def build_circuit(nqubits=4):
        number = '10'*math.ceil(nqubits/2)
        number = int(number[0:nqubits], 2) # code a number interleaving 1 and 0 in binary

        dev = qml.device("lightning.qubit", wires=nqubits)

        @qml.qnode(dev)
        def circuit():
            powerTwo = 2**(nqubits-1)
            for qubit in range(nqubits):
                qml.Hadamard(qubit)
                qml.PhaseShift(phi=number*pi/powerTwo, wires=qubit)
                powerTwo = powerTwo/2
            qml.adjoint(IQFT.qft)(nqubits)

            # Measurement
            return qml.probs(wires=range(nqubits))

        circuit()

        return circuit

class Bell:

    def build_circuit():
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])

            # Measurement
            return qml.probs(wires=1)

        circuit()

        return circuit