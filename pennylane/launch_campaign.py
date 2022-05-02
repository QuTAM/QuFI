#%%
from os import system
from numpy import pi, arange, array_split
from itertools import product
from multiprocessing import cpu_count
from dill import dump, HIGHEST_PROTOCOL
from qufi import BernsteinVazirani

angles={'theta0':arange(0, pi+0.01, pi/12), 
        'phi0':arange(0, 2*pi+0.01, pi/12), 
        'theta1':arange(0, pi+0.01, pi/12), 
        'phi1':arange(0, 2*pi+0.01, pi/12)}
splits = cpu_count()

angle_injections = []
angle_combinations = product(angles['theta0'], angles['phi0'])
for angle_pair1 in angle_combinations:
    angle_combinations_df = product(arange(0, angle_pair1[0]+0.01, pi/12), arange(0, angle_pair1[1]+0.01, pi/12))
    for angle_pair2 in angle_combinations_df:
        angle_injections.append((angle_pair1[0], angle_pair1[1], angle_pair2[0], angle_pair2[1]))

circuits = []

bv4_p = BernsteinVazirani.build_circuit(3, '101')
circuits.append((bv4_p, 'Bernstein-Vazirani_4'))

# Custom coupling map
coupling_map = {'topology': {(0, 1), (1, 2), (2, 3), (3, 4)}, 'logical2physical': {0: 3, 1: 4, 3: 2, 2: 0}, 'physical2logical': {3: 0, 4: 1, 2: 3, 0: 2}}

with open('angle_injections.pickle', 'wb') as handle:
    dump(angle_injections, handle, protocol=HIGHEST_PROTOCOL)
with open('coupling_map.pickle', 'wb') as handle:
    dump(coupling_map, handle, protocol=HIGHEST_PROTOCOL)
with open('circuits.pickle', 'wb') as handle:
    dump(circuits, handle, protocol=HIGHEST_PROTOCOL)

print("angle_injections len", len(angle_injections), "splits", splits)
#%%
system(f"byobu new-session -d -s qufi \"htop\"")
system("tmux set-option remain-on-exit on")
counter = 0
for batch, group in zip(array_split(angle_injections, splits), range(splits)):
    print(f"Launching shell for batch:{group} size:{len(batch)}")
    system(f"byobu new-window -t qufi -n \"run_{counter}\" \"python3 run.py {counter}\"")
    counter = counter + 1

# %%
