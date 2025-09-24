import numpy as np

expected_values_nom_values = np.array([
    [0,0,0],
    [0,0,0],
    [0,0,0],
],dtype=np.float32)

expected_values_den_values = np.array([
    [0,0,0],
    [0,0,0],
    [0,0,0],
],dtype=np.float32)

def expected_values():
    a,b = expected_values_nom_values[0,:], expected_values_den_values[0,:]
    # print(f"a: {a}")
    # print(f"b: {b}")
    return np.divide(a, np.maximum(2 ** -30, b))

print(expected_values())

# https://github.com/numpy/numpy/issues/29804