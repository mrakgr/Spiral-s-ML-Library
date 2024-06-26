kernel = r"""
#include "reference_counting.h"
#include <assert.h>
#include <stdio.h>
struct Union1;
struct Union0;
__device__ void write_2(const char * v0);
__device__ void write_1();
__device__ void write_3();
__device__ void write_5(long v0);
__device__ void write_7();
__device__ void write_8();
__device__ void write_9(long v0, sptr<Union1> v1);
__device__ void write_6(sptr<Union0> v0);
__device__ void write_4(long v0, sptr<Union0> v1);
__device__ void write_0(sptr<Union1> v0);
struct Union1_0 { // A_Done
};
struct Union1_1 { // A_Rest
    sptr<Union0> v1;
    long v0;
    __device__ Union1_1(long t0, sptr<Union0> t1) : v0(t0), v1(t1) {}
    __device__ Union1_1() = delete;
};
struct Union1 {
    union {
        Union1_0 case0; // A_Done
        Union1_1 case1; // A_Rest
    };
    __device__ Union1(Union1_0 t) : tag(0), case0(t) {} // A_Done
    __device__ Union1(Union1_1 t) : tag(1), case1(t) {} // A_Rest
    __device__ Union1() = delete;
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
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = x.case0; break; // A_Done
            case 1: this->case1 = x.case1; break; // A_Rest
        }
        return *this;
    }
    __device__ Union1 & operator=(Union1 && x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = std::move(x.case0); break; // A_Done
            case 1: this->case1 = std::move(x.case1); break; // A_Rest
        }
        return *this;
    }
    __device__ ~Union1() {
        switch(this->tag){
            case 0: this->case0.~Union1_0(); break; // A_Done
            case 1: this->case1.~Union1_1(); break; // A_Rest
        }
    }
    int refc = 0;
    unsigned char tag;
};
struct Union0_0 { // B_Done
};
struct Union0_1 { // B_Rest
    sptr<Union1> v1;
    long v0;
    __device__ Union0_1(long t0, sptr<Union1> t1) : v0(t0), v1(t1) {}
    __device__ Union0_1() = delete;
};
struct Union0 {
    union {
        Union0_0 case0; // B_Done
        Union0_1 case1; // B_Rest
    };
    __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // B_Done
    __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // B_Rest
    __device__ Union0() = delete;
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
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = x.case0; break; // B_Done
            case 1: this->case1 = x.case1; break; // B_Rest
        }
        return *this;
    }
    __device__ Union0 & operator=(Union0 && x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = std::move(x.case0); break; // B_Done
            case 1: this->case1 = std::move(x.case1); break; // B_Rest
        }
        return *this;
    }
    __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // B_Done
            case 1: this->case1.~Union0_1(); break; // B_Rest
        }
    }
    int refc = 0;
    unsigned char tag;
};
__device__ void write_2(const char * v0){
    const char * v1;
    v1 = "%s";
    printf(v1,v0);
    return ;
}
__device__ void write_1(){
    const char * v0;
    v0 = "A_Done";
    return write_2(v0);
}
__device__ void write_3(){
    const char * v0;
    v0 = "A_Rest";
    return write_2(v0);
}
__device__ void write_5(long v0){
    const char * v1;
    v1 = "%d";
    printf(v1,v0);
    return ;
}
__device__ void write_7(){
    const char * v0;
    v0 = "B_Done";
    return write_2(v0);
}
__device__ void write_8(){
    const char * v0;
    v0 = "B_Rest";
    return write_2(v0);
}
__device__ void write_9(long v0, sptr<Union1> v1){
    write_5(v0);
    const char * v2;
    v2 = ", ";
    write_2(v2);
    return write_0(v1);
}
__device__ void write_6(sptr<Union0> v0){
    switch (v0.base->tag) {
        case 0: { // B_Done
            return write_7();
            break;
        }
        case 1: { // B_Rest
            long v1 = v0.base->case1.v0; sptr<Union1> v2 = v0.base->case1.v1;
            write_8();
            const char * v3;
            v3 = "(";
            write_2(v3);
            write_9(v1, v2);
            const char * v4;
            v4 = ")";
            return write_2(v4);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void write_4(long v0, sptr<Union0> v1){
    write_5(v0);
    const char * v2;
    v2 = ", ";
    write_2(v2);
    return write_6(v1);
}
__device__ void write_0(sptr<Union1> v0){
    switch (v0.base->tag) {
        case 0: { // A_Done
            return write_1();
            break;
        }
        case 1: { // A_Rest
            long v1 = v0.base->case1.v0; sptr<Union0> v2 = v0.base->case1.v1;
            write_3();
            const char * v3;
            v3 = "(";
            write_2(v3);
            write_4(v1, v2);
            const char * v4;
            v4 = ")";
            return write_2(v4);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
extern "C" __global__ void entry0() {
    long v0;
    v0 = threadIdx.x;
    long v1;
    v1 = blockIdx.x;
    long v2;
    v2 = v1 * 32l;
    long v3;
    v3 = v0 + v2;
    bool v4;
    v4 = v3 == 0l;
    if (v4){
        long v5;
        v5 = 1l;
        long v6;
        v6 = 2l;
        long v7;
        v7 = 3l;
        sptr<Union0> v8;
        v8 = sptr<Union0>{new Union0{Union0_0{}}};
        sptr<Union1> v9;
        v9 = sptr<Union1>{new Union1{Union1_1{v7, v8}}};
        sptr<Union0> v10;
        v10 = sptr<Union0>{new Union0{Union0_1{v6, v9}}};
        sptr<Union1> v11;
        v11 = sptr<Union1>{new Union1{Union1_1{v5, v10}}};
        write_0(v11);
        printf("\n");
        return ;
    } else {
        return ;
    }
}
"""
class static_array(list):
    def __init__(self, length):
        for _ in range(length):
            self.append(None)

class static_array_list(static_array):
    def __init__(self, length):
        super().__init__(length)
        self.length = 0
import cupy as cp
from dataclasses import dataclass
from typing import NamedTuple, Union, Callable, Tuple
i8 = i16 = i32 = i64 = u8 = u16 = u32 = u64 = int; f32 = f64 = float; char = string = str

options = []
options.append('--diag-suppress=550,20012')
options.append('--dopt=on')
options.append('--restrict')
options.append('-I C:/Spiral_s_ML_Library/cpplib')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
def main():
    v0 = 0
    v1 = raw_module.get_function(f"entry{v0}")
    del v0
    v1.max_dynamic_shared_size_bytes = 0 
    v1((1,),(32,),(),shared_mem=0)
    del v1
    return 

if __name__ == '__main__': print(main())
