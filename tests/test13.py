kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);

extern "C" __global__ void entry0() {
    int v0;
    v0 = threadIdx.x;
    cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v1 = console_lock;
    auto v2 = cooperative_groups::coalesced_threads();
    v1.acquire();
    printf("{%s = %s; %s = %d}\n","msg", "Hello", "tid", v0);
    v1.release();
    v2.sync() ;
    return ;
}
"""

import cupy as cp
from dataclasses import dataclass
from typing import NamedTuple, Union, Callable, Tuple
i8 = i16 = i32 = i64 = u8 = u16 = u32 = u64 = int; f32 = f64 = float; char = string = str

options = []
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39')
options.append('--restrict')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def main_body():
    v0 = 0
    v1 = raw_module.get_function(f"entry{v0}")
    del v0
    v1.max_dynamic_shared_size_bytes = 0 
    v1((1,),(32,),(),shared_mem=0)
    del v1
    return 

import sys

class Logger:
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w')

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()

def main():
    path = 'tests_io/output.txt'
    cp._sys.stdout = Logger(path)
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    print("Done.")
    cp._sys.stdout = cp._sys.__stdout__
    return r


if __name__ == '__main__': print(main())
