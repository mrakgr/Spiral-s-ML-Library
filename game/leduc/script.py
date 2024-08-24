# import cupy as cp

# # Set the random seed
# cp.random.seed(42)

# # Generate random numbers
# random_array = cp.random.rand(3, 2)
# print(random_array)

from pathlib import Path
file_path = Path("test_text_outputs/qwe/asd.txt")
file_path.parent.mkdir(parents=True, exist_ok=True)
file_path.write_text("Hello, world!")