#pragma once
#include "test1c.corelib.hpp"
#ifdef __CUDACC__
#ifdef __CUDA_ARCH__
// Cuda device backend
#else
// Cuda host backend
#endif
#else
// Cpp host backend
struct Union0;
struct Union1;
struct Tuple0;
struct Tuple1;
struct StackMut0;
struct Union2;
struct StackMut1;
struct Union3;
struct StackMut2;
struct Tuple2;
unsigned int f_2(unsigned int v0, unsigned int v1, unsigned int v2, unsigned int v3);
unsigned int sum_pow_1(unsigned int v0, unsigned int v1);
Tuple1 method_0(unsigned int v0);
unsigned int method_3(static_array_list<Tuple0,5> v0, static_array<int,2> v1);
int main();
struct Union0_0 { // Call
};
struct Union0_1 { // Fold
};
struct Union0_2 { // Raise
    int v0;
    __host__ __device__ Union0_2(int t0) : v0(t0) {}
    __host__ __device__ Union0_2() = delete;
};
struct Union0 {
    union {
        Union0_0 case0; // Call
        Union0_1 case1; // Fold
        Union0_2 case2; // Raise
    };
    unsigned char tag{255};
    __host__ __device__ Union0() {}
    __host__ __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // Call
    __host__ __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // Fold
    __host__ __device__ Union0(Union0_2 t) : tag(2), case2(t) {} // Raise
    __host__ __device__ Union0(const Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // Call
            case 1: new (&this->case1) Union0_1(x.case1); break; // Fold
            case 2: new (&this->case2) Union0_2(x.case2); break; // Raise
        }
    }
    __host__ __device__ Union0(const Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // Call
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // Fold
            case 2: new (&this->case2) Union0_2(std::move(x.case2)); break; // Raise
        }
    }
    __host__ __device__ Union0 & operator=(const Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Call
                case 1: this->case1 = x.case1; break; // Fold
                case 2: this->case2 = x.case2; break; // Raise
            }
        } else {
            this->~Union0();
            new (this) Union0{x};
        }
        return *this;
    }
    __host__ __device__ Union0 & operator=(const Union0 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // Call
                case 1: this->case1 = std::move(x.case1); break; // Fold
                case 2: this->case2 = std::move(x.case2); break; // Raise
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // Call
            case 1: this->case1.~Union0_1(); break; // Fold
            case 2: this->case2.~Union0_2(); break; // Raise
        }
        this->tag = 255;
    }
};
struct Union1_0 { // Jack
};
struct Union1_1 { // King
};
struct Union1_2 { // Queen
};
struct Union1 {
    union {
        Union1_0 case0; // Jack
        Union1_1 case1; // King
        Union1_2 case2; // Queen
    };
    unsigned char tag{255};
    __host__ __device__ Union1() {}
    __host__ __device__ Union1(Union1_0 t) : tag(0), case0(t) {} // Jack
    __host__ __device__ Union1(Union1_1 t) : tag(1), case1(t) {} // King
    __host__ __device__ Union1(Union1_2 t) : tag(2), case2(t) {} // Queen
    __host__ __device__ Union1(const Union1 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(x.case0); break; // Jack
            case 1: new (&this->case1) Union1_1(x.case1); break; // King
            case 2: new (&this->case2) Union1_2(x.case2); break; // Queen
        }
    }
    __host__ __device__ Union1(const Union1 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(std::move(x.case0)); break; // Jack
            case 1: new (&this->case1) Union1_1(std::move(x.case1)); break; // King
            case 2: new (&this->case2) Union1_2(std::move(x.case2)); break; // Queen
        }
    }
    __host__ __device__ Union1 & operator=(const Union1 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Jack
                case 1: this->case1 = x.case1; break; // King
                case 2: this->case2 = x.case2; break; // Queen
            }
        } else {
            this->~Union1();
            new (this) Union1{x};
        }
        return *this;
    }
    __host__ __device__ Union1 & operator=(const Union1 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // Jack
                case 1: this->case1 = std::move(x.case1); break; // King
                case 2: this->case2 = std::move(x.case2); break; // Queen
            }
        } else {
            this->~Union1();
            new (this) Union1{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union1() {
        switch(this->tag){
            case 0: this->case0.~Union1_0(); break; // Jack
            case 1: this->case1.~Union1_1(); break; // King
            case 2: this->case2.~Union1_2(); break; // Queen
        }
        this->tag = 255;
    }
};
struct Tuple0 {
    Union0 v0;
    Union1 v1;
    __host__ __device__ Tuple0() = default;
    __host__ __device__ Tuple0(Union0 t0, Union1 t1) : v0(t0), v1(t1) {}
};
struct Tuple1 {
    static_array_list<Tuple0,5> v0;
    static_array<int,2> v1;
    __host__ __device__ Tuple1() = default;
    __host__ __device__ Tuple1(static_array_list<Tuple0,5> t0, static_array<int,2> t1) : v0(t0), v1(t1) {}
};
struct StackMut0 {
    unsigned int v0;
    __host__ __device__ StackMut0() = default;
    __host__ __device__ StackMut0(unsigned int t0) : v0(t0) {}
};
struct Union2_0 { // None
};
struct Union2_1 { // Some
    Union1 v0;
    __host__ __device__ Union2_1(Union1 t0) : v0(t0) {}
    __host__ __device__ Union2_1() = delete;
};
struct Union2 {
    union {
        Union2_0 case0; // None
        Union2_1 case1; // Some
    };
    unsigned char tag{255};
    __host__ __device__ Union2() {}
    __host__ __device__ Union2(Union2_0 t) : tag(0), case0(t) {} // None
    __host__ __device__ Union2(Union2_1 t) : tag(1), case1(t) {} // Some
    __host__ __device__ Union2(const Union2 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(x.case0); break; // None
            case 1: new (&this->case1) Union2_1(x.case1); break; // Some
        }
    }
    __host__ __device__ Union2(const Union2 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union2_1(std::move(x.case1)); break; // Some
        }
    }
    __host__ __device__ Union2 & operator=(const Union2 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union2();
            new (this) Union2{x};
        }
        return *this;
    }
    __host__ __device__ Union2 & operator=(const Union2 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union2();
            new (this) Union2{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union2() {
        switch(this->tag){
            case 0: this->case0.~Union2_0(); break; // None
            case 1: this->case1.~Union2_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct StackMut1 {
    Union2 v0;
    __host__ __device__ StackMut1() = default;
    __host__ __device__ StackMut1(Union2 t0) : v0(t0) {}
};
struct Union3_0 { // None
};
struct Union3_1 { // Some
    Union0 v0;
    __host__ __device__ Union3_1(Union0 t0) : v0(t0) {}
    __host__ __device__ Union3_1() = delete;
};
struct Union3 {
    union {
        Union3_0 case0; // None
        Union3_1 case1; // Some
    };
    unsigned char tag{255};
    __host__ __device__ Union3() {}
    __host__ __device__ Union3(Union3_0 t) : tag(0), case0(t) {} // None
    __host__ __device__ Union3(Union3_1 t) : tag(1), case1(t) {} // Some
    __host__ __device__ Union3(const Union3 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(x.case0); break; // None
            case 1: new (&this->case1) Union3_1(x.case1); break; // Some
        }
    }
    __host__ __device__ Union3(const Union3 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union3_1(std::move(x.case1)); break; // Some
        }
    }
    __host__ __device__ Union3 & operator=(const Union3 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union3();
            new (this) Union3{x};
        }
        return *this;
    }
    __host__ __device__ Union3 & operator=(const Union3 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union3();
            new (this) Union3{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union3() {
        switch(this->tag){
            case 0: this->case0.~Union3_0(); break; // None
            case 1: this->case1.~Union3_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct StackMut2 {
    Union3 v0;
    __host__ __device__ StackMut2() = default;
    __host__ __device__ StackMut2(Union3 t0) : v0(t0) {}
};
struct Tuple2 {
    int v0;
    unsigned int v1;
    unsigned int v2;
    __host__ __device__ Tuple2() = default;
    __host__ __device__ Tuple2(int t0, unsigned int t1, unsigned int t2) : v0(t0), v1(t1), v2(t2) {}
};
#endif
