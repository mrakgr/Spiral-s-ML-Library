kernels_main = r"""
#include <cooperative_groups.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
extern "C" __global__ void entry0(int * v0) {
    int v1;
    v1 = v0[0];
    int v2;
    v2 = v0[1];
    int v3;
    v3 = v1 + v2;
    int v4;
    v4 = threadIdx.x;
    int v5;
    v5 = blockIdx.x;
    int v6;
    v6 = v5 * 256;
    int v7;
    v7 = v4 + v6;
    bool v8;
    v8 = v7 == 0;
    if (v8){
        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v9 = console_lock;
        auto v10 = cooperative_groups::coalesced_threads();
        v9.acquire();
        printf("{%s = %s; %s = %d}\n","message", "hello from cuda", "result", v3);
        v9.release();
        v10.sync() ;
    } else {
    }
    return ;
}
"""
from test1_auto import *
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
def method0(v0 : cp.ndarray) -> None:
    kernel = "entry0"
    v1 = cp.cuda.Device().attributes['MultiProcessorCount']
    v2 = v1 >= 84
    del v1
    v3 = v2 == False
    if v3:
        v4 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = raw_module.get_function(kernel)
    v5.max_dynamic_shared_size_bytes = 98304 
    print(f'Threads per block, blocks per grid: {256}, {84}')
    v5((84,),(256,),v0,shared_mem=98304)
    del v0, v5
    return 
def main():
    v0 = cp.array([1, 2, 3, 4],dtype=cp.int32)
    v1 = v0.size
    v2 = v0[2].item()
    v5 = "{{{} = {}; {} = {}; {} = {}}}\n"
    v6 = "index_2"
    v7 = "length_of_array"
    v8 = "message"
    v9 = "hello from host"
    print(v5.format(v6, v2, v7, v1, v8, v9),end="")
    del v1, v2, v5, v6, v7, v8, v9
    method0(v0)
    del v0
    cp.cuda.get_current_stream().synchronize()
    return 0

if __name__ == '__main__': print(main())
