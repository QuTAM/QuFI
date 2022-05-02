#%%
from sys import argv
from numpy import array_split
from qufi import execute
from dill import load
from multiprocessing import cpu_count

index = argv[1]
splits = cpu_count()

#%%
with open('angle_injections.pickle', 'rb') as handle:
    angle_injections = load(handle)
with open('coupling_map.pickle', 'rb') as handle:
    coupling_map = load(handle)
with open('circuits.pickle', 'rb') as handle:
    circuits = load(handle)

#%%
for batch, group in zip(array_split(angle_injections, splits), range(splits)):
    if group == index:
        print(len(batch))
        results_names = execute(circuits, batch, coupling_map=coupling_map)
        print(len(results_names))
        input()
