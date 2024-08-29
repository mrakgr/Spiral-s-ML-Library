import cupy as cp

device = cp.cuda.Device()
print(device.attributes['MultiProcessorCount'])