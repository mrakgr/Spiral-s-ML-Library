import numpy as np

def random_argmax(arr):
    max_value = np.max(arr)
    max_positions = np.where(arr == max_value)[0]
    return np.random.choice(max_positions)
random_argmax(np.array([1,2,3,4,5,5,5,5,5,5]))


# 11111111111111
# 00011001101101

