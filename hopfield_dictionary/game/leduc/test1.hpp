#pragma once
#include "test1.corelib.hpp"
// Cuda globals
#ifdef __CUDACC__
#include <cooperative_groups.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
#endif
// Cpp globals
// The Cuda type forward declarations
#ifdef __CUDACC__
struct Union5;
struct Union7;
struct StackMut9;
struct Tuple14;
struct Tuple16;
struct Union17;
struct StackMut18;
struct Union20;
struct StackMut21;
#endif
// The Cpp type forward declarations
#ifdef __CUDACC__
// The Cuda device methods forward declarations
__device__ void method_8(unsigned int * v0, static_array_list<Union5,5> v1, Union7 v2, bool v3, static_array<int,3> v4, int v5, unsigned int v6);
__device__ int method_15(unsigned int * v0, StackMut9 & v1, int v2);
__device__ void method_19(unsigned int * v0, StackMut9 & v1, int v2);
__device__ void method_22(unsigned int * v0, StackMut9 & v1, int v2);
__device__ Tuple14 method_13(unsigned int * v0);
__device__ void method_26(Union5 v0);
__device__ void method_27(Union7 v0);
// The Cuda host methods forward declarations
void run_cuda_device_from_cuda_host_3();
// The Cuda device main defs forward declarations
extern "C" __global__ void __cluster_dims__(12,1,1) cuda_device_entry0();
#else
// The Cpp host methods forward declarations
void run_cuda_host_from_cpp_host_1();
// The Cuda host main defs forward declarations
void cuda_host_entry0();
#endif
// The Cuda type definitions
#ifdef __CUDACC__
struct Union5_0 { // Call
};
struct Union5_1 { // Fold
};
struct Union5_2 { // Raise
    int v0;
    __host__ __device__ Union5_2(int t0) : v0(t0) {}
    __host__ __device__ Union5_2() = delete;
};
struct Union5 {
    union {
        Union5_0 case0; // Call
        Union5_1 case1; // Fold
        Union5_2 case2; // Raise
    };
    unsigned char tag{255};
    __host__ __device__ Union5() {}
    __host__ __device__ Union5(Union5_0 t) : tag(0), case0(t) {} // Call
    __host__ __device__ Union5(Union5_1 t) : tag(1), case1(t) {} // Fold
    __host__ __device__ Union5(Union5_2 t) : tag(2), case2(t) {} // Raise
    __host__ __device__ Union5(const Union5 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union5_0(x.case0); break; // Call
            case 1: new (&this->case1) Union5_1(x.case1); break; // Fold
            case 2: new (&this->case2) Union5_2(x.case2); break; // Raise
        }
    }
    __host__ __device__ Union5(const Union5 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union5_0(std::move(x.case0)); break; // Call
            case 1: new (&this->case1) Union5_1(std::move(x.case1)); break; // Fold
            case 2: new (&this->case2) Union5_2(std::move(x.case2)); break; // Raise
        }
    }
    __host__ __device__ Union5 & operator=(const Union5 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Call
                case 1: this->case1 = x.case1; break; // Fold
                case 2: this->case2 = x.case2; break; // Raise
            }
        } else {
            this->~Union5();
            new (this) Union5{x};
        }
        return *this;
    }
    __host__ __device__ Union5 & operator=(const Union5 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // Call
                case 1: this->case1 = std::move(x.case1); break; // Fold
                case 2: this->case2 = std::move(x.case2); break; // Raise
            }
        } else {
            this->~Union5();
            new (this) Union5{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union5() {
        switch(this->tag){
            case 0: this->case0.~Union5_0(); break; // Call
            case 1: this->case1.~Union5_1(); break; // Fold
            case 2: this->case2.~Union5_2(); break; // Raise
        }
        this->tag = 255;
    }
};
struct Union7_0 { // Jack
};
struct Union7_1 { // King
};
struct Union7_2 { // Queen
};
struct Union7 {
    union {
        Union7_0 case0; // Jack
        Union7_1 case1; // King
        Union7_2 case2; // Queen
    };
    unsigned char tag{255};
    __host__ __device__ Union7() {}
    __host__ __device__ Union7(Union7_0 t) : tag(0), case0(t) {} // Jack
    __host__ __device__ Union7(Union7_1 t) : tag(1), case1(t) {} // King
    __host__ __device__ Union7(Union7_2 t) : tag(2), case2(t) {} // Queen
    __host__ __device__ Union7(const Union7 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(x.case0); break; // Jack
            case 1: new (&this->case1) Union7_1(x.case1); break; // King
            case 2: new (&this->case2) Union7_2(x.case2); break; // Queen
        }
    }
    __host__ __device__ Union7(const Union7 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(std::move(x.case0)); break; // Jack
            case 1: new (&this->case1) Union7_1(std::move(x.case1)); break; // King
            case 2: new (&this->case2) Union7_2(std::move(x.case2)); break; // Queen
        }
    }
    __host__ __device__ Union7 & operator=(const Union7 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Jack
                case 1: this->case1 = x.case1; break; // King
                case 2: this->case2 = x.case2; break; // Queen
            }
        } else {
            this->~Union7();
            new (this) Union7{x};
        }
        return *this;
    }
    __host__ __device__ Union7 & operator=(const Union7 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // Jack
                case 1: this->case1 = std::move(x.case1); break; // King
                case 2: this->case2 = std::move(x.case2); break; // Queen
            }
        } else {
            this->~Union7();
            new (this) Union7{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union7() {
        switch(this->tag){
            case 0: this->case0.~Union7_0(); break; // Jack
            case 1: this->case1.~Union7_1(); break; // King
            case 2: this->case2.~Union7_2(); break; // Queen
        }
        this->tag = 255;
    }
};
struct StackMut9 {
    int v0;
    __host__ __device__ StackMut9() = default;
    __host__ __device__ StackMut9(int t0) : v0(t0) {}
};
struct Tuple14 {
    static_array_list<Union5,5> v0;
    Union7 v1;
    static_array<int,3> v3;
    int v4;
    unsigned int v5;
    bool v2;
    __host__ __device__ Tuple14() = default;
    __host__ __device__ Tuple14(static_array_list<Union5,5> t0, Union7 t1, bool t2, static_array<int,3> t3, int t4, unsigned int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
};
struct Tuple16 {
    int v0;
    int v1;
    int v2;
    __host__ __device__ Tuple16() = default;
    __host__ __device__ Tuple16(int t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Union17_0 { // None
};
struct Union17_1 { // Some
    Union7 v0;
    __host__ __device__ Union17_1(Union7 t0) : v0(t0) {}
    __host__ __device__ Union17_1() = delete;
};
struct Union17 {
    union {
        Union17_0 case0; // None
        Union17_1 case1; // Some
    };
    unsigned char tag{255};
    __host__ __device__ Union17() {}
    __host__ __device__ Union17(Union17_0 t) : tag(0), case0(t) {} // None
    __host__ __device__ Union17(Union17_1 t) : tag(1), case1(t) {} // Some
    __host__ __device__ Union17(const Union17 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union17_0(x.case0); break; // None
            case 1: new (&this->case1) Union17_1(x.case1); break; // Some
        }
    }
    __host__ __device__ Union17(const Union17 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union17_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union17_1(std::move(x.case1)); break; // Some
        }
    }
    __host__ __device__ Union17 & operator=(const Union17 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union17();
            new (this) Union17{x};
        }
        return *this;
    }
    __host__ __device__ Union17 & operator=(const Union17 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union17();
            new (this) Union17{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union17() {
        switch(this->tag){
            case 0: this->case0.~Union17_0(); break; // None
            case 1: this->case1.~Union17_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct StackMut18 {
    Union17 v0;
    __host__ __device__ StackMut18() = default;
    __host__ __device__ StackMut18(Union17 t0) : v0(t0) {}
};
struct Union20_0 { // None
};
struct Union20_1 { // Some
    Union5 v0;
    __host__ __device__ Union20_1(Union5 t0) : v0(t0) {}
    __host__ __device__ Union20_1() = delete;
};
struct Union20 {
    union {
        Union20_0 case0; // None
        Union20_1 case1; // Some
    };
    unsigned char tag{255};
    __host__ __device__ Union20() {}
    __host__ __device__ Union20(Union20_0 t) : tag(0), case0(t) {} // None
    __host__ __device__ Union20(Union20_1 t) : tag(1), case1(t) {} // Some
    __host__ __device__ Union20(const Union20 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union20_0(x.case0); break; // None
            case 1: new (&this->case1) Union20_1(x.case1); break; // Some
        }
    }
    __host__ __device__ Union20(const Union20 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union20_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union20_1(std::move(x.case1)); break; // Some
        }
    }
    __host__ __device__ Union20 & operator=(const Union20 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union20();
            new (this) Union20{x};
        }
        return *this;
    }
    __host__ __device__ Union20 & operator=(const Union20 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union20();
            new (this) Union20{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union20() {
        switch(this->tag){
            case 0: this->case0.~Union20_0(); break; // None
            case 1: this->case1.~Union20_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct StackMut21 {
    Union20 v0;
    __host__ __device__ StackMut21() = default;
    __host__ __device__ StackMut21(Union20 t0) : v0(t0) {}
};
#endif
// The Cpp type definitions
