import cupy as cp

# Set the random seed
cp.random.seed(42)

# Generate random numbers
random_array = cp.random.rand(3, 2)
print(random_array)
