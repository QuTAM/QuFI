#%%
import py_compile
import numpy as np
from qufi import execute, save_results, IQFT

circuits = []

bv4_p = IQFT.build_circuit(4)
circuits.append((bv4_p, 'Bernstein-Vazirani_4'))

theta_values = [np.pi]
phi_values = [np.pi]

#%%

results = execute(circuits, theta_values, phi_values)

#%%
save_results(results)

# %%
