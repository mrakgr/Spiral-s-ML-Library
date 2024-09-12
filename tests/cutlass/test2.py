kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include "cutlass/gemm/device/gemm_universal_adapter.h"
using namespace cute;

#include <cooperative_groups.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);

extern "C" __global__ void qwert_entry0() {
    int v0;
    v0 = threadIdx.x;
    int v1;
    v1 = blockIdx.x;
    int v2;
    v2 = v1 * 256l;
    int v3;
    v3 = v0 + v2;
    bool v4;
    v4 = v3 == 0l;
    if (v4){
        cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v5 = console_lock;
        auto v6 = cooperative_groups::coalesced_threads();
        v5.acquire();
        printf("%s\n","hello");
        v5.release();
        v6.sync() ;
        return ;
    } else {
        return ;
    }
}
"""

import cupy as cp
from dataclasses import dataclass
from typing import NamedTuple, Union, Callable, Tuple
i8 = int; i16 = int; i32 = int; i64 = int; u8 = int; u16 = int; u32 = int; u64 = int; f32 = float; f64 = float; char = str; string = str

options = []
options.append('--define-macro=NDEBUG')
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39,177')
options.append('--restrict')
options.append('--maxrregcount=256')
options.append('-I"G:/cutlass-3.5.1/include"')
options.append('-I"G:/cutlass-3.5.1/tools/util/include"')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def main_body():
    v0 = cp.cuda.Device().attributes['MultiProcessorCount']
    v1 = v0 == 24
    del v0
    v2 = v1 == False
    if v2:
        v3 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v1, v3
        del v3
    else:
        pass
    del v1, v2
    v5 = raw_module.get_function(f"qwert_entry0")
    v5.max_dynamic_shared_size_bytes = 81920 
    v5((24,),(256,),(),shared_mem=81920)
    del v5
    return 

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
