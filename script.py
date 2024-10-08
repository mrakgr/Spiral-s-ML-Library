import cupy as cp
import cupyx
x = cp.array([1,2,3])
print(cupyx.scipy.special.softmax(x*0.01))