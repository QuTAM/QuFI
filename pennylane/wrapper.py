#%%
import os
import numpy as np

angles={'theta1':np.arange(0, np.pi+0.01, np.pi/12), 
        'phi1':np.arange(0, np.pi+0.01, np.pi/12), 
        'theta2':np.arange(0, np.pi+0.01, np.pi/12), 
        'phi2':np.arange(0, np.pi+0.01, np.pi/12)}
splits = len(angles['theta1'])
#%%
os.system(f"byobu new-session -d -s qufi \"htop\"")
os.system("tmux set-option remain-on-exit on")
counter = 0
for theta1_subgroup in np.array_split(angles['theta1'], splits):
    print(theta1_subgroup)
    #os.system(f"byobu new-window -t qufi -n \"th{counter}\" \"conda run -n dev python3 byobu_run.py {counter} {splits}\"")
    os.system(f"byobu new-window -t qufi -n \"th{counter}\" \"python3 byobu_run.py {counter} {splits}\"")
    counter = counter + 1
# %%
