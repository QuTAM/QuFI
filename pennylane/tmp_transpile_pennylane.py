#%%
import numpy as np
from itertools import product
import pickle
import gzip
import datetime
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, Aer, IBMQ, execute
from qiskit.test.mock import FakeSantiago, FakeCasablanca, FakeSydney, FakeBogota, FakeLima, FakeMelbourne
from qiskit.visualization import plot_circuit_layout
#from qiskit.tools.jupyter import *
#from qiskit.visualization import *
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.visualization import dag_drawer

from fault_injector_u_gate import inject


#%%
circuits = []

import Grover
grover = Grover.build_circuit()
# circuits.append( (grover, 'Grover') )

# provider = IBMQ.load_account()
# backend = provider.backend.ibmq_bogota
backend = FakeBogota()

new_circ_lv0 = transpile(grover, backend=backend, optimization_level=0)
plot_circuit_layout(new_circ_lv0, backend)

#%%
myset = set(tuple(sorted(l)) for l in backend.configuration().to_dict()['coupling_map'])

# %%
for qa,qb in myset:
    print(qb)
# %%
