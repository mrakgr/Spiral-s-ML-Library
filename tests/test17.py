kernels_main = r"""
__device__ void noinline_qwe_1();
extern "C" __global__ void entry0();
__device__ __noinline__ void noinline_qwe_1(){
    __syncthreads();
    printf("hello\n");
    return ;
}
extern "C" __global__ void entry0() {
    int v0;
    v0 = threadIdx.x;
    bool v1;
    v1 = v0 < 15;
    if (v1){
        printf("true\n");
        return noinline_qwe_1();
    } else {
        printf("false\n");
        return noinline_qwe_1();
    }
}
"""
from test17_auto import *
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
    v1 = v0 >= 24
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
    print(f'Threads per block, blocks per grid: {256}, {24}')
    v4((24,),(256,),(),shared_mem=98304)
    del v4
    return 
def main():
    return method0()

if __name__ == '__main__': print(main())
