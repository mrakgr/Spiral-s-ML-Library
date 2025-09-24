import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return e_x / e_x.sum(axis=0)

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
temperature = 0.01
print((np.matmul(softmax(np.matmul(keys, input) / temperature), values)))
