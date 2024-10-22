import cupy as cp
import cupyx

T = 10
def softmax(x : cp.ndarray) -> cp.ndarray : return cupyx.scipy.special.softmax(x * T)

m = 2 ** 20
n = 2
k = 1
keys = cp.zeros((m,n))
values = cp.zeros((m,k))

i = 0
def add(k,v):
    global i
    keys[i] = cp.array(k)
    values[i] = cp.array(v)
    i += 1

add([0,1],[[1]])
add([1,0],[[1]])
add([1,1],[[-1]])

# print(keys)
# print(values)

def index_keys(x):
    x = x.T
    return cp.matmul(keys,x)
def index_values(x):
    x = cp.array(x).T
    return cp.matmul(x,values)

key = cp.random.random((2 ** 10,n))
s = index_keys(key)
print(index_values(softmax(s)))

