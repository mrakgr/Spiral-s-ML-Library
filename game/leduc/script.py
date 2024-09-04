import cupy as cp

# Set the random seed
cp.random.seed(42)

# Generate random numbers
random_array = cp.random.rand(4, 3, 2)
print(type(random_array.get()))
