kernels_main = r"""
extern "C" __global__ void entry0(float * v0, float * v1);
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 67108864;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 4;
    return v1;
}
extern "C" __global__ void entry0(float * v0, float * v1) {
    int v2;
    v2 = threadIdx.x;
    int v3;
    v3 = blockIdx.x;
    int v4;
    v4 = v3 * 256;
    int v5;
    v5 = v2 + v4;
    int v6;
    v6 = v5;
    while (while_method_1(v6)){
        bool v8;
        v8 = 0 <= v6;
        bool v9;
        v9 = v8 == false;
        if (v9){
            assert("The index needs to be zero or positive." && v8);
        } else {
        }
        bool v11;
        v11 = v6 < 67108864;
        bool v12;
        v12 = v11 == false;
        if (v12){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v11);
        } else {
        }
        assert("Tensor range check" && 0 <= v6 && v6 < 67108864);
        int v14;
        v14 = 4 * v6;
        assert("Tensor range check" && 0 <= v6 && v6 < 67108864);
        float v15[4];
        float v16[4];
        int4* v17;
        v17 = reinterpret_cast<int4*>(v0 + v14);
        int4* v18;
        v18 = reinterpret_cast<int4*>(v15 + 0);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v17) % 16 == 0 && reinterpret_cast<unsigned long long>(v18) % 16 == 0);
        *v18 = *v17;
        // Pushing the loop unrolling to: 0
        int v19;
        v19 = 0;
        #pragma unroll
        while (while_method_2(v19)){
            assert("Tensor range check" && 0 <= v19 && v19 < 4);
            float v21;
            v21 = v15[v19];
            float v22;
            v22 = v21 + 10.0f;
            assert("Tensor range check" && 0 <= v19 && v19 < 4);
            v16[v19] = v22;
            v19 += 1 ;
        }
        // Poping the loop unrolling to: 0
        int4* v23;
        v23 = reinterpret_cast<int4*>(v16 + 0);
        int4* v24;
        v24 = reinterpret_cast<int4*>(v1 + v14);
        assert("Pointer alignment check" && reinterpret_cast<unsigned long long>(v23) % 16 == 0 && reinterpret_cast<unsigned long long>(v24) % 16 == 0);
        *v24 = *v23;
        v6 += 6144 ;
    }
    return ;
}
"""
from test0_auto import *
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
def method0(v0 : cp.ndarray, v1 : cp.ndarray) -> None:
    kernel = "entry0"
    v2 = cp.cuda.Device().attributes['MultiProcessorCount']
    v3 = v2 >= 24
    del v2
    v4 = v3 == False
    if v4:
        v5 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v3, v5
        del v5
    else:
        pass
    del v3, v4
    v6 = raw_module.get_function(kernel)
    v6.max_dynamic_shared_size_bytes = 98304 
    print(f'Threads per block, blocks per grid: {256}, {24}')
    v6((24,),(256,),(v0, v1),shared_mem=98304)
    del v0, v1, v6
    return 
def method1(v0 : i32) -> bool:
    v1 = v0 < 16
    del v0
    return v1
def main():
    v2 = "{}\n"
    v3 = "Running test 0. How long does the memory transfer take?"
    print(v2.format(v3),end="")
    del v2, v3
    v4 = cp.ones(268435456,dtype=cp.float32) # type: ignore
    v5 = cp.empty(268435456,dtype=cp.float32)
    method0(v4, v5)
    del v4
    v35 = 0
    v36 = "{}"
    print(v36.format('['),end="")
    v37 = 0
    while method1(v37):
        v39 = v35
        v40 = v39 >= 100
        del v39
        if v40:
            v41 = " ..."
            print(v36.format(v41),end="")
            del v41
            break
        else:
            pass
        del v40
        v42 = v37 == 0
        v43 = v42 != True
        del v42
        if v43:
            v44 = "; "
            print(v36.format(v44),end="")
            del v44
        else:
            pass
        del v43
        v45 = v35 + 1
        v35 = v45
        del v45
        v46 = v5[v37].item()
        v47 = "{:.6f}"
        print(v47.format(v46),end="")
        del v46, v47
        v37 += 1 
    del v5, v35, v37
    print(v36.format(']'),end="")
    del v36
    v48 = "\n"
    print(v48.format(),end="")
    del v48
    return 

if __name__ == '__main__': print(main())
