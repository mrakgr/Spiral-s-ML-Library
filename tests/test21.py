kernels_main = r"""
#include <cooperative_groups.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
struct Union0;
__device__ void method_6(Union0 v0);
extern "C" __global__ void entry0(static_array<Union0,5> v0);
struct Union0_0 { // None
};
struct Union0_1 { // Some
    unsigned long long v1;
    int v0;
    __device__ Union0_1(int t0, unsigned long long t1) : v0(t0), v1(t1) {}
    __device__ Union0_1() = delete;
};
struct Union0 {
    union {
        Union0_0 case0; // None
        Union0_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union0() {}
    __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // None
    __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // Some
    __device__ Union0(const Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // None
            case 1: new (&this->case1) Union0_1(x.case1); break; // Some
        }
    }
    __device__ Union0(const Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union0 & operator=(const Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union0();
            new (this) Union0{x};
        }
        return *this;
    }
    __device__ Union0 & operator=(const Union0 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // None
            case 1: this->case1.~Union0_1(); break; // Some
        }
        this->tag = 255;
    }
};
__device__ inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 5;
    return v1;
}
__device__ void method_6(Union0 v0){
    switch (v0.tag) {
        case 0: { // None
            printf("%s","None");
            return ;
            break;
        }
        case 1: { // Some
            int v1 = v0.case1.v0; unsigned long long v2 = v0.case1.v1;
            printf("%s(%d, %llu)","Some", v1, v2);
            return ;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            __trap();
        }
    }
}
extern "C" __global__ void entry0(static_array<Union0,5> v0) {
    int v1;
    v1 = threadIdx.x;
    int v2;
    v2 = blockIdx.x;
    int v3;
    v3 = v2 * 256;
    int v4;
    v4 = v1 + v3;
    bool v5;
    v5 = v4 == 0;
    if (v5){
        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v14 = console_lock;
        auto v15 = cooperative_groups::coalesced_threads();
        v14.acquire();
        printf("%s","[");
        int v16;
        v16 = 0;
        while (while_method_5(v16)){
            Union0 v19;
            v19 = v0[v16];
            printf("");
            method_6(v19);
            printf("");
            int v22;
            v22 = v16 + 1;
            bool v23;
            v23 = v22 < 5;
            if (v23){
                printf("%s","; ");
            } else {
            }
            v16 += 1 ;
        }
        printf("%s","]");
        printf("\n");
        v14.release();
        v15.sync() ;
        return ;
    } else {
        return ;
    }
}
"""
from test21_auto import *
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
class US0_0(NamedTuple): # None
    tag = 0
class US0_1(NamedTuple): # Some
    v0 : i32
    v1 : u64
    tag = 1
US0 = Union[US0_0, US0_1]
def method0(v0 : i32) -> bool:
    v1 = v0 < 5
    del v0
    return v1
def method1(v0 : static_array) -> None:
    kernel = "entry0"
    v1 = cp.cuda.Device().attributes['MultiProcessorCount']
    v2 = v1 >= 24
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
    print(f'Threads per block, blocks per grid: {256}, {24}')
    v5((24,),(256,),v0,shared_mem=98304)
    del v0, v5
    return 
def main():
    v3 = static_array(5)
    v4 = 0
    while method0(v4):
        v6 = u64(v4)
        v10 = US0_1(v4, v6)
        del v6
        v3[v4] = v10
        del v10
        v4 += 1 
    del v4
    method1(v3)
    del v3
    cp.cuda.get_current_stream().synchronize()
    return 0

if __name__ == '__main__': print(main())
