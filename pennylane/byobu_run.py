#%%
import os
import sys
from copy import deepcopy
import numpy as np
from qufi import execute, BernsteinVazirani
import qufi
from qiskit.test.mock import FakeSantiago

index = int(sys.argv[1])
splits = int(sys.argv[2])
circuits = []

bv4_p = BernsteinVazirani.build_circuit(3, '101')
circuits.append((bv4_p, 'Bernstein-Vazirani_4'))

angles={'theta0':np.arange(0, 2*np.pi+0.01, np.pi/12), 
        'phi0':np.arange(0, np.pi+0.01, np.pi/12), 
        'theta1':np.arange(0, 2*np.pi+0.01, np.pi/12), 
        'phi1':np.arange(0, np.pi+0.01, np.pi/12)}

# Custom coupling map
coupling_map = {'topology': {(0, 1), (1, 2), (2, 3), (3, 4)}, 'logical2physical': {0: 3, 1: 4, 3: 2, 2: 0}, 'physical2logical': {3: 0, 4: 1, 2: 3, 0: 2}}

#%%
for theta0_subgroup, group in zip(np.array_split(angles['theta0'], splits), range(splits)):
    if group == index:
        subset_angles = deepcopy(angles)
        subset_angles['theta0'] = theta0_subgroup
        print(subset_angles)
        results_names = execute(circuits, subset_angles, coupling_map=coupling_map)
        print(results_names)
        input
