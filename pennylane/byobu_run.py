#%%
import os
import sys
from copy import deepcopy
import numpy as np
from qufi import execute, BernsteinVazirani
import qufi
from qiskit.test.mock import FakeSantiago
from itertools import product

index = int(sys.argv[1])
splits = int(sys.argv[2])
circuits = []

bv4_p = BernsteinVazirani.build_circuit(3, '101')
circuits.append((bv4_p, 'Bernstein-Vazirani_4'))

angles={'theta0':np.arange(0, np.pi+0.01, np.pi/12), 
        'phi0':np.arange(0, 2*np.pi+0.01, np.pi/12), 
        'theta1':np.arange(0, np.pi+0.01, np.pi/12), 
        'phi1':np.arange(0, 2*np.pi+0.01, np.pi/12)}

# Custom coupling map
coupling_map = {'topology': {(0, 1), (1, 2), (2, 3), (3, 4)}, 'logical2physical': {0: 3, 1: 4, 3: 2, 2: 0}, 'physical2logical': {3: 0, 4: 1, 2: 3, 0: 2}}

#%%
angle_injections = []
angle_combinations = product(angles['theta0'], angles['phi0'])
for angle_pair1 in angle_combinations:
    angle_combinations_df = product(np.arange(0, angle_pair1[0]+0.01, np.pi/12), np.arange(0, angle_pair1[1]+0.01, np.pi/12))
    for angle_pair2 in angle_combinations_df:
        angle_injections.append((angle_pair1[0], angle_pair1[1], angle_pair2[0], angle_pair2[1]))

#%%
for batch, group in zip(np.array_split(angle_injections, splits), range(splits)):
    if group == index:
        print(len(batch))
        results_names = execute(circuits, batch, coupling_map=coupling_map)
        print(len(results_names))
        input()
