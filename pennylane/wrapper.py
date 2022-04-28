#%%
import os
import numpy as np

angles={'theta0':np.arange(0, np.pi+0.01, np.pi/12), 
        'phi0':np.arange(0, 2*np.pi+0.01, np.pi/12), 
        'theta1':np.arange(0, np.pi+0.01, np.pi/12), 
        'phi1':np.arange(0, 2*np.pi+0.01, np.pi/12)}
splits = len(angles['theta0'])
#%%
os.system(f"byobu new-session -d -s qufi \"htop\"")
os.system("tmux set-option remain-on-exit on")
counter = 0
for theta0_subgroup in np.array_split(angles['theta0'], splits):
    print(f"Launching shell for theta0:{theta0_subgroup}")
    os.system(f"byobu new-window -t qufi -n \"th{counter}\" \"python3 byobu_run.py {counter} {splits}\"")
    counter = counter + 1
