kernels_main = r"""
#include <cooperative_groups.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
struct Union1;
struct Union0;
__device__ void method_1(sptr<Union0> v0);
__device__ void method_0(sptr<Union1> v0);
struct Union1_0 { // A_Done
};
struct Union1_1 { // A_Rest
    sptr<Union0> v1;
    int v0;
    __device__ Union1_1(int t0, sptr<Union0> t1) : v0(t0), v1(t1) {}
    __device__ Union1_1() = delete;
};
struct Union1 {
    union {
        Union1_0 case0; // A_Done
        Union1_1 case1; // A_Rest
    };
    int refc{0};
    unsigned char tag{255};
    __device__ Union1() {}
    __device__ Union1(Union1_0 t) : tag(0), case0(t) {} // A_Done
    __device__ Union1(Union1_1 t) : tag(1), case1(t) {} // A_Rest
    __device__ Union1(Union1 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(x.case0); break; // A_Done
            case 1: new (&this->case1) Union1_1(x.case1); break; // A_Rest
        }
    }
    __device__ Union1(Union1 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(std::move(x.case0)); break; // A_Done
            case 1: new (&this->case1) Union1_1(std::move(x.case1)); break; // A_Rest
        }
    }
    __device__ Union1 & operator=(Union1 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // A_Done
                case 1: this->case1 = x.case1; break; // A_Rest
            }
        } else {
            this->~Union1();
            new (this) Union1{x};
        }
        return *this;
    }
    __device__ Union1 & operator=(Union1 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // A_Done
                case 1: this->case1 = std::move(x.case1); break; // A_Rest
            }
        } else {
            this->~Union1();
            new (this) Union1{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union1() {
        switch(this->tag){
            case 0: this->case0.~Union1_0(); break; // A_Done
            case 1: this->case1.~Union1_1(); break; // A_Rest
        }
        this->tag = 255;
    }
};
struct Union0_0 { // B_Done
};
struct Union0_1 { // B_Rest
    sptr<Union1> v1;
    int v0;
    __device__ Union0_1(int t0, sptr<Union1> t1) : v0(t0), v1(t1) {}
    __device__ Union0_1() = delete;
};
struct Union0 {
    union {
        Union0_0 case0; // B_Done
        Union0_1 case1; // B_Rest
    };
    int refc{0};
    unsigned char tag{255};
    __device__ Union0() {}
    __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // B_Done
    __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // B_Rest
    __device__ Union0(Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // B_Done
            case 1: new (&this->case1) Union0_1(x.case1); break; // B_Rest
        }
    }
    __device__ Union0(Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // B_Done
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // B_Rest
        }
    }
    __device__ Union0 & operator=(Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // B_Done
                case 1: this->case1 = x.case1; break; // B_Rest
            }
        } else {
            this->~Union0();
            new (this) Union0{x};
        }
        return *this;
    }
    __device__ Union0 & operator=(Union0 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // B_Done
                case 1: this->case1 = std::move(x.case1); break; // B_Rest
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // B_Done
            case 1: this->case1.~Union0_1(); break; // B_Rest
        }
        this->tag = 255;
    }
};
__device__ void method_1(sptr<Union0> v0){
    switch (v0.base->tag) {
        case 0: { // B_Done
            printf("%s","B_Done");
            return ;
            break;
        }
        case 1: { // B_Rest
            int v1 = v0.base->case1.v0; sptr<Union1> v2 = v0.base->case1.v1;
            printf("%s(%d, ","B_Rest", v1);
            method_0(v2);
            printf(")");
            return ;
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void method_0(sptr<Union1> v0){
    switch (v0.base->tag) {
        case 0: { // A_Done
            printf("%s","A_Done");
            return ;
            break;
        }
        case 1: { // A_Rest
            int v1 = v0.base->case1.v0; sptr<Union0> v2 = v0.base->case1.v1;
            printf("%s(%d, ","A_Rest", v1);
            method_1(v2);
            printf(")");
            return ;
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
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
        int v4;
        v4 = 1;
        int v5;
        v5 = 2;
        int v6;
        v6 = 3;
        sptr<Union0> v7;
        v7 = sptr<Union0>{new Union0{Union0_0{}}};
        sptr<Union1> v8;
        v8 = sptr<Union1>{new Union1{Union1_1{v6, v7}}};
        sptr<Union0> v9;
        v9 = sptr<Union0>{new Union0{Union0_1{v5, v8}}};
        sptr<Union1> v10;
        v10 = sptr<Union1>{new Union1{Union1_1{v4, v9}}};
        cuda::counting_semaphore<cuda::thread_scope_system, 1> & v11 = console_lock;
        auto v12 = cooperative_groups::coalesced_threads();
        v11.acquire();
        printf("");
        method_0(v10);
        printf("\n");
        v11.release();
        v12.sync() ;
        return ;
    } else {
        return ;
    }
}
"""
from test2_auto import *
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
