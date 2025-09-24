import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return e_x / e_x.sum(axis=0)
def hopfield_dict(input : np.ndarray, keys : np.ndarray, values : np.ndarray, temperature = 0.01):
    return np.matmul(softmax(np.matmul(keys, input) / temperature), values)

keys = np.array([
    [1,0,0],
    [1,0,0],
    [0,1,0],
    [0,0,1],
    ])
values = np.array([
    [10,20,-10],
    [10,40,-10],
    [20,10,-10],
    [30,10,-10],
    ])
input = np.array([1,0,0])
print(np.vstack([keys, input]))
# print(hopfield_dict(input, keys, values))
