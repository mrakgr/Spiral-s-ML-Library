kernel = r"""
using default_int = long;
using default_uint = unsigned long;
#include "reference_counting.cuh"
#include <assert.h>
#include <stdio.h>
struct ClosureBase0 { int refc{0}; __device__ virtual void operator()() = 0; __device__ virtual ~ClosureBase0() = default; };
typedef csptr<ClosureBase0> Fun0;
struct Union1;
struct Union0;
struct Mut0;
__device__ void write_3(const char * v0);
__device__ void write_2();
__device__ void write_4();
__device__ void write_6(long v0);
__device__ void write_8();
__device__ void write_9();
__device__ void write_10(long v0, sptr<Union1> v1);
__device__ void write_7(sptr<Union0> v0);
__device__ void write_5(long v0, sptr<Union0> v1);
__device__ void write_1(sptr<Union1> v0);
__device__ Fun0 method_0();
typedef void (* Fun1)();
__device__ Fun1 method_11();
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
    long v0;
    __device__ Union0_1(long t0, sptr<Union1> t1) : v0(t0), v1(t1) {}
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
struct Mut0 {
    int refc{0};
    sptr<Union1> v0;
    __device__ Mut0() = default;
    __device__ Mut0(sptr<Union1> t0) : v0(t0) {}
};
struct Closure0 : public ClosureBase0 {
    __device__ void operator()() override {
        long v0;
        v0 = 1l;
        long v1;
        v1 = 2l;
        long v2;
        v2 = 3l;
        sptr<Union0> v3;
        v3 = sptr<Union0>{new Union0{Union0_0{}}};
        sptr<Union1> v4;
        v4 = sptr<Union1>{new Union1{Union1_1{v2, v3}}};
        sptr<Union0> v5;
        v5 = sptr<Union0>{new Union0{Union0_1{v1, v4}}};
        sptr<Union1> v6;
        v6 = sptr<Union1>{new Union1{Union1_1{v0, v5}}};
        long v7;
        v7 = 12l;
        long v8;
        v8 = 23l;
        long v9;
        v9 = 34l;
        sptr<Union0> v10;
        v10 = sptr<Union0>{new Union0{Union0_0{}}};
        sptr<Union1> v11;
        v11 = sptr<Union1>{new Union1{Union1_1{v9, v10}}};
        sptr<Union0> v12;
        v12 = sptr<Union0>{new Union0{Union0_1{v8, v11}}};
        sptr<Union1> v13;
        v13 = sptr<Union1>{new Union1{Union1_1{v7, v12}}};
        sptr<Mut0> v14;
        v14 = sptr<Mut0>{new Mut0{v6}};
        v14.base->v0 = v13;
        sptr<Union1> v15;
        v15 = v14.base->v0;
        write_1(v15);
        printf("\n");
        return ;
    }
    ~Closure0() override = default;
};
__device__ void write_3(const char * v0){
    const char * v1;
    v1 = "%s";
    printf(v1,v0);
    return ;
}
__device__ void write_2(){
    const char * v0;
    v0 = "A_Done";
    return write_3(v0);
}
__device__ void write_4(){
    const char * v0;
    v0 = "A_Rest";
    return write_3(v0);
}
__device__ void write_6(long v0){
    const char * v1;
    v1 = "%d";
    printf(v1,v0);
    return ;
}
__device__ void write_8(){
    const char * v0;
    v0 = "B_Done";
    return write_3(v0);
}
__device__ void write_9(){
    const char * v0;
    v0 = "B_Rest";
    return write_3(v0);
}
__device__ void write_10(long v0, sptr<Union1> v1){
    write_6(v0);
    const char * v2;
    v2 = ", ";
    write_3(v2);
    return write_1(v1);
}
__device__ void write_7(sptr<Union0> v0){
    switch (v0.base->tag) {
        case 0: { // B_Done
            return write_8();
            break;
        }
        case 1: { // B_Rest
            long v1 = v0.base->case1.v0; sptr<Union1> v2 = v0.base->case1.v1;
            write_9();
            const char * v3;
            v3 = "(";
            write_3(v3);
            write_10(v1, v2);
            const char * v4;
            v4 = ")";
            return write_3(v4);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void write_5(long v0, sptr<Union0> v1){
    write_6(v0);
    const char * v2;
    v2 = ", ";
    write_3(v2);
    return write_7(v1);
}
__device__ void write_1(sptr<Union1> v0){
    switch (v0.base->tag) {
        case 0: { // A_Done
            return write_2();
            break;
        }
        case 1: { // A_Rest
            long v1 = v0.base->case1.v0; sptr<Union0> v2 = v0.base->case1.v1;
            write_4();
            const char * v3;
            v3 = "(";
            write_3(v3);
            write_5(v1, v2);
            const char * v4;
            v4 = ")";
            return write_3(v4);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ Fun0 method_0(){
    return csptr<ClosureBase0>{new Closure0{}};
}
__device__ void FunPointerMethod1(){
    long v0;
    v0 = 1l;
    long v1;
    v1 = 2l;
    long v2;
    v2 = 3l;
    sptr<Union0> v3;
    v3 = sptr<Union0>{new Union0{Union0_0{}}};
    sptr<Union1> v4;
    v4 = sptr<Union1>{new Union1{Union1_1{v2, v3}}};
    sptr<Union0> v5;
    v5 = sptr<Union0>{new Union0{Union0_1{v1, v4}}};
    sptr<Union1> v6;
    v6 = sptr<Union1>{new Union1{Union1_1{v0, v5}}};
    long v7;
    v7 = 12l;
    long v8;
    v8 = 23l;
    long v9;
    v9 = 34l;
    sptr<Union0> v10;
    v10 = sptr<Union0>{new Union0{Union0_0{}}};
    sptr<Union1> v11;
    v11 = sptr<Union1>{new Union1{Union1_1{v9, v10}}};
    sptr<Union0> v12;
    v12 = sptr<Union0>{new Union0{Union0_1{v8, v11}}};
    sptr<Union1> v13;
    v13 = sptr<Union1>{new Union1{Union1_1{v7, v12}}};
    sptr<Mut0> v14;
    v14 = sptr<Mut0>{new Mut0{v6}};
    v14.base->v0 = v13;
    sptr<Union1> v15;
    v15 = v14.base->v0;
    write_1(v15);
    printf("\n");
    return ;
}
__device__ Fun1 method_11(){
    return FunPointerMethod1;
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
        Fun0 v5;
        v5 = method_0();
        v5();
        Fun1 v6;
        v6 = method_11();
        return v6();
    } else {
        return ;
    }
}
"""
class static_array():
    def __init__(self, length):
        self.ptr = []
        for _ in range(length):
            self.ptr.append(None)

    def __getitem__(self, index):
        assert 0 <= index < len(self.ptr), "The get index needs to be in range."
        return self.ptr[index]
    
    def __setitem__(self, index, value):
        assert 0 <= index < len(self.ptr), "The set index needs to be in range."
        self.ptr[index] = value

class static_array_list(static_array):
    def __init__(self, length):
        super().__init__(length)
        self.length = 0

    def __getitem__(self, index):
        assert 0 <= index < self.length, "The get index needs to be in range."
        return self.ptr[index]
    
    def __setitem__(self, index, value):
        assert 0 <= index < self.length, "The set index needs to be in range."
        self.ptr[index] = value

    def push(self,value):
        assert (self.length < len(self.ptr)), "The length before pushing has to be less than the maximum length of the array."
        self.ptr[self.length] = value
        self.length += 1

    def pop(self):
        assert (0 < self.length), "The length before popping has to be greater than 0."
        self.length -= 1
        return self.ptr[self.length]

    def unsafe_set_length(self,i):
        assert 0 <= i <= len(self.ptr), "The new length has to be in range."
        self.length = i
import cupy as cp
from dataclasses import dataclass
from typing import NamedTuple, Union, Callable, Tuple
i8 = i16 = i32 = i64 = u8 = u16 = u32 = u64 = int; f32 = f64 = float; char = string = str

options = []
options.append('--diag-suppress=550,20012,68')
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
