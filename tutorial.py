#%%
import numpy as np
from qufi import execute_over_range_single, BernsteinVazirani, read_file
from itertools import product

#%%
circuits = []

bv4_p = BernsteinVazirani.build_circuit(3, '101')
circuits.append((bv4_p, 'Bernstein-Vazirani_4'))

angles = {'theta0':[0, round(np.pi/4, 3)], 'phi0':[0, round(np.pi/4, 3)]}


#%%
results_names = execute_over_range_single(circuits, angles, noise=False, 
                                          results_folder="./tmp/")

# %%
results = []
for name in results_names:
    results.append(read_file(f"{name}", single = True))

# %%
combinations = product(angles['theta0'], angles['phi0'])
for i, comb in enumerate(combinations):
    results[i].to_csv(f'example_result_{comb}.csv', index=False, header=True)

# %%
