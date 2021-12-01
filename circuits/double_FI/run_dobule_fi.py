#%%
import numpy as np
from itertools import product
import pickle, gzip
import datetime
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, Aer, IBMQ, execute
#from qiskit.tools.jupyter import *
#from qiskit.visualization import *

from double_fault_injector_u_gate import inject


fp=open("./run_double_fi.log", "a")

#%%
circuits = []


import inverseQFT
qft3 = inverseQFT.build_circuit(3)
circuits.append( (qft3, 'inverseQFT3') )



#%%
theta_values = np.arange(0, np.pi+0.01, np.pi/12) # 0 <= theta <= pi
phi_values = np.arange(0, 2*np.pi, np.pi/12) # 0 <= phi < 2pi
results = []
for circ in circuits:
    print('-'*80)
    fp.write('-'*80)
    fp.write('\n')
    print('start:',datetime.datetime.now())
    fp.write('start:'+str(datetime.datetime.now()))
    fp.write('\n')
    fp.flush()
    r = inject(circ[0], circ[1])
    results.append(r)
    print('done:',datetime.datetime.now())
    fp.write('done:'+str(datetime.datetime.now()))
    fp.write('\n')
    print('-'*80)
    fp.write('-'*80)
    fp.write('\n')
#%%
filename_output = '../double_u_gate_15degrees_step_qft_3.p.gz'
pickle.dump(results, gzip.open(filename_output, 'w'))
print('files saved to:',filename_output)
fp.write('files saved to:'+str(filename_output))
fp.close()

