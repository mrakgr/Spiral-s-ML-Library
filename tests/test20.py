kernels_main = r"""
#include <cooperative_groups.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
extern "C" __global__ void __cluster_dims__(8,1,1) entry0() {
    int v0;
    v0 = threadIdx.x;
    int v1;
    v1 = blockIdx.x;
    int v2;
    v2 = v1 * 256;
    int v3;
    v3 = v0 + v2;
    bool v4;
    v4 = v3 == 0;
    if (v4){
        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v5 = console_lock;
        auto v6 = cooperative_groups::coalesced_threads();
        v5.acquire();
        printf("%s\n","Hello World!");
        v5.release();
        v6.sync() ;
    } else {
    }
    return ;
}
"""
from test20_auto import *
kernels = kernels_aux + kernels_main
import cupy as cp
from dataclasses import dataclass
from typing import NamedTuple, Union, Callable, Tuple
i8 = int; i16 = int; i32 = int; i64 = int; u8 = int; u16 = int; u32 = int; u64 = int; f32 = float; f64 = float; char = str; string = str

options = []
options.append('--define-macro=NDEBUG')
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39,177')
options.append('--restrict')
import os
home = os.getenv('HOME')
options.append(f'-I={home}/ThunderKittens/include')
options.append('--maxrregcount=255')
options.append('--std=c++20')
options.append('--expt-relaxed-constexpr')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernels, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def method0() -> None:
    kernel = "entry0"
    v0 = cp.cuda.Device().attributes['MultiProcessorCount']
    v1 = v0 >= 84
    del v0
    v2 = v1 == False
    if v2:
        v3 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v1, v3
        del v3
    else:
        pass
    del v1, v2
    v4 = raw_module.get_function(kernel)
    v4.max_dynamic_shared_size_bytes = 98304 
    print(f'Threads per block, blocks per grid: {256}, {84}')
    v4((84,),(256,),(),shared_mem=98304)
    del v4
    return 
def main():
    return method0()

if __name__ == '__main__': print(main())
