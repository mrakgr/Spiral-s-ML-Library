#pragma once
#include "test3.corelib.hpp"
// Cuda globals
#ifdef __CUDACC__
#endif
// Cpp globals
#include <Eigen/Dense>
// The Cuda type forward declarations
#ifdef __CUDACC__
#endif
// The Cpp type forward declarations
struct StackRefs2;
struct Union3;
struct Union5;
struct StackMut7;
struct Union12;
struct Tuple14;
struct Tuple16;
struct Union17;
struct StackMut18;
struct Union20;
struct StackMut21;
#ifdef __CUDACC__
// The Cuda device methods forward declarations
// The Cuda host methods forward declarations
// The Cuda device main defs forward declarations
#else
// The Cpp host methods forward declarations
void method_6(float * v0, static_array_list<Union3,5> v1, Union5 v2, bool v3, static_array<int,3> v4, int v5, unsigned int v6);
int method_15(float * v0, StackMut7 & v1, int v2);
void method_19(float * v0, StackMut7 & v1, int v2);
void method_22(float * v0, StackMut7 & v1, int v2);
Tuple14 method_13(float * v0);
void method_23(Union3 v0);
void method_24(Union5 v0);
// The Cuda host main defs forward declarations
#endif
// The Cuda type definitions
#ifdef __CUDACC__
#endif
// The Cpp type definitions
struct StackRefs2 {
    Eigen::Matrix<float,8,115> & v1;
    Eigen::Matrix<float,8,48> & v2;
    int & v0;
    __host__ __device__ StackRefs2() = default;
    __host__ __device__ StackRefs2(int & t0, Eigen::Matrix<float,8,115> & t1, Eigen::Matrix<float,8,48> & t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Union3_0 { // Call
};
struct Union3_1 { // Fold
};
struct Union3_2 { // Raise
    int v0;
    __host__ __device__ Union3_2(int t0) : v0(t0) {}
    __host__ __device__ Union3_2() = delete;
};
struct Union3 {
    union {
        Union3_0 case0; // Call
        Union3_1 case1; // Fold
        Union3_2 case2; // Raise
    };
    unsigned char tag{255};
    __host__ __device__ Union3() {}
    __host__ __device__ Union3(Union3_0 t) : tag(0), case0(t) {} // Call
    __host__ __device__ Union3(Union3_1 t) : tag(1), case1(t) {} // Fold
    __host__ __device__ Union3(Union3_2 t) : tag(2), case2(t) {} // Raise
    __host__ __device__ Union3(const Union3 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(x.case0); break; // Call
            case 1: new (&this->case1) Union3_1(x.case1); break; // Fold
            case 2: new (&this->case2) Union3_2(x.case2); break; // Raise
        }
    }
    __host__ __device__ Union3(const Union3 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(std::move(x.case0)); break; // Call
            case 1: new (&this->case1) Union3_1(std::move(x.case1)); break; // Fold
            case 2: new (&this->case2) Union3_2(std::move(x.case2)); break; // Raise
        }
    }
    __host__ __device__ Union3 & operator=(const Union3 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Call
                case 1: this->case1 = x.case1; break; // Fold
                case 2: this->case2 = x.case2; break; // Raise
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
                case 0: this->case0 = std::move(x.case0); break; // Call
                case 1: this->case1 = std::move(x.case1); break; // Fold
                case 2: this->case2 = std::move(x.case2); break; // Raise
            }
        } else {
            this->~Union3();
            new (this) Union3{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union3() {
        switch(this->tag){
            case 0: this->case0.~Union3_0(); break; // Call
            case 1: this->case1.~Union3_1(); break; // Fold
            case 2: this->case2.~Union3_2(); break; // Raise
        }
        this->tag = 255;
    }
};
struct Union5_0 { // Jack
};
struct Union5_1 { // King
};
struct Union5_2 { // Queen
};
struct Union5 {
    union {
        Union5_0 case0; // Jack
        Union5_1 case1; // King
        Union5_2 case2; // Queen
    };
    unsigned char tag{255};
    __host__ __device__ Union5() {}
    __host__ __device__ Union5(Union5_0 t) : tag(0), case0(t) {} // Jack
    __host__ __device__ Union5(Union5_1 t) : tag(1), case1(t) {} // King
    __host__ __device__ Union5(Union5_2 t) : tag(2), case2(t) {} // Queen
    __host__ __device__ Union5(const Union5 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union5_0(x.case0); break; // Jack
            case 1: new (&this->case1) Union5_1(x.case1); break; // King
            case 2: new (&this->case2) Union5_2(x.case2); break; // Queen
        }
    }
    __host__ __device__ Union5(const Union5 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union5_0(std::move(x.case0)); break; // Jack
            case 1: new (&this->case1) Union5_1(std::move(x.case1)); break; // King
            case 2: new (&this->case2) Union5_2(std::move(x.case2)); break; // Queen
        }
    }
    __host__ __device__ Union5 & operator=(const Union5 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Jack
                case 1: this->case1 = x.case1; break; // King
                case 2: this->case2 = x.case2; break; // Queen
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
                case 0: this->case0 = std::move(x.case0); break; // Jack
                case 1: this->case1 = std::move(x.case1); break; // King
                case 2: this->case2 = std::move(x.case2); break; // Queen
            }
        } else {
            this->~Union5();
            new (this) Union5{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union5() {
        switch(this->tag){
            case 0: this->case0.~Union5_0(); break; // Jack
            case 1: this->case1.~Union5_1(); break; // King
            case 2: this->case2.~Union5_2(); break; // Queen
        }
        this->tag = 255;
    }
};
struct StackMut7 {
    int v0;
    __host__ __device__ StackMut7() = default;
    __host__ __device__ StackMut7(int t0) : v0(t0) {}
};
struct Union12_0 { // None
};
struct Union12_1 { // Some
    static_array_list<Union3,5> v0;
    Union5 v1;
    static_array<int,3> v3;
    int v4;
    unsigned int v5;
    bool v2;
    __host__ __device__ Union12_1(static_array_list<Union3,5> t0, Union5 t1, bool t2, static_array<int,3> t3, int t4, unsigned int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __host__ __device__ Union12_1() = delete;
};
struct Union12 {
    union {
        Union12_0 case0; // None
        Union12_1 case1; // Some
    };
    unsigned char tag{255};
    __host__ __device__ Union12() {}
    __host__ __device__ Union12(Union12_0 t) : tag(0), case0(t) {} // None
    __host__ __device__ Union12(Union12_1 t) : tag(1), case1(t) {} // Some
    __host__ __device__ Union12(const Union12 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union12_0(x.case0); break; // None
            case 1: new (&this->case1) Union12_1(x.case1); break; // Some
        }
    }
    __host__ __device__ Union12(const Union12 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union12_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union12_1(std::move(x.case1)); break; // Some
        }
    }
    __host__ __device__ Union12 & operator=(const Union12 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union12();
            new (this) Union12{x};
        }
        return *this;
    }
    __host__ __device__ Union12 & operator=(const Union12 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union12();
            new (this) Union12{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union12() {
        switch(this->tag){
            case 0: this->case0.~Union12_0(); break; // None
            case 1: this->case1.~Union12_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Tuple14 {
    static_array_list<Union3,5> v0;
    Union5 v1;
    static_array<int,3> v3;
    int v4;
    unsigned int v5;
    bool v2;
    __host__ __device__ Tuple14() = default;
    __host__ __device__ Tuple14(static_array_list<Union3,5> t0, Union5 t1, bool t2, static_array<int,3> t3, int t4, unsigned int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
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
    Union5 v0;
    __host__ __device__ Union17_1(Union5 t0) : v0(t0) {}
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
    Union3 v0;
    __host__ __device__ Union20_1(Union3 t0) : v0(t0) {}
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
