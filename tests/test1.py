kernels_main = r"""
#include <cooperative_groups.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
struct ClosureBase0 { int refc{0}; __device__ virtual int operator()(int, int) = 0; __device__ virtual ~ClosureBase0(){}; };
typedef csptr<ClosureBase0> Fun0;
struct Closure0 : public ClosureBase0 {
    __device__ int operator()(int tup0, int tup1) override {
        int v0 = tup0; int v1 = tup1;
        int v2;
        v2 = v0 + v1;
        return v2;
    }
    __device__ ~Closure0() override {  }
};
extern "C" __global__ void entry0() {
    int v0;
    v0 = threadIdx.x;
    int v1;
    v1 = blockIdx.x;
    int v2;
    v2 = v0 + v1;
    bool v3;
    v3 = v2 == 0;
    if (v3){
        Fun0 v4{new Closure0{}};
        int v5;
        v5 = v4(1, 2);
        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v6 = console_lock;
        auto v7 = cooperative_groups::coalesced_threads();
        v6.acquire();
        printf("%d\n",v5);
        v6.release();
        v7.sync() ;
        return ;
    } else {
        return ;
    }
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
options.append('--std=c++20')
options.append('--expt-relaxed-constexpr')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernels, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def main_body():
    v0 = cp.cuda.Device().attributes['MultiProcessorCount']
    v1 = v0 >= 1
    del v0
    v2 = v1 == False
    if v2:
        v3 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v1, v3
        del v3
    else:
        pass
    del v1, v2
    kernel = "entry0"
    v4 = raw_module.get_function(kernel)
    v4.max_dynamic_shared_size_bytes = 98304 
    print(f'Threads per block, blocks per grid: {1}, {1}')
    v4((1,),(1,),(),shared_mem=98304)
    del v4
    return 

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
