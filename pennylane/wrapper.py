#%%
import os
import numpy as np
from itertools import product

angles={'theta0':np.arange(0, np.pi+0.01, np.pi/12), 
        'phi0':np.arange(0, 2*np.pi+0.01, np.pi/12), 
        'theta1':np.arange(0, np.pi+0.01, np.pi/12), 
        'phi1':np.arange(0, 2*np.pi+0.01, np.pi/12)}
splits = len(angles['phi0'])

angle_injections = []
angle_combinations = product(angles['theta0'], angles['phi0'])
for angle_pair1 in angle_combinations:
    angle_combinations_df = product(np.arange(0, angle_pair1[0]+0.01, np.pi/12), np.arange(0, angle_pair1[1]+0.01, np.pi/12))
    for angle_pair2 in angle_combinations_df:
        angle_injections.append((angle_pair1[0], angle_pair1[1], angle_pair2[0], angle_pair2[1]))

print("angle_injections len", len(angle_injections))
#%%
os.system(f"byobu new-session -d -s qufi \"htop\"")
os.system("tmux set-option remain-on-exit on")
counter = 0
for batch, group in zip(np.array_split(angle_injections, splits), range(splits)):
    print(f"Launching shell for batch:{group} size:{len(batch)}")
    os.system(f"byobu new-window -t qufi -n \"th{counter}\" \"python3 byobu_run.py {counter} {splits}\"")
    counter = counter + 1

# %%
