#pragma once
#include "test2.corelib.hpp"
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
struct Union11;
struct Tuple13;
struct Tuple15;
struct Union16;
struct StackMut17;
struct Union19;
struct StackMut20;
#ifdef __CUDACC__
// The Cuda device methods forward declarations
// The Cuda host methods forward declarations
// The Cuda device main defs forward declarations
#else
// The Cpp host methods forward declarations
void method_6(float * v0, static_array_list<Union3,5> v1, Union5 v2, bool v3, static_array<int,3> v4, int v5, unsigned int v6);
int method_14(float * v0, StackMut7 & v1, int v2);
void method_18(float * v0, StackMut7 & v1, int v2);
void method_21(float * v0, StackMut7 & v1, int v2);
Tuple13 method_12(float * v0);
void method_22(Union3 v0);
void method_23(Union5 v0);
void method_24(Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v0, Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v1, Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v2, Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v3);
// The Cuda host main defs forward declarations
#endif
// The Cuda type definitions
#ifdef __CUDACC__
#endif
// The Cpp type definitions
struct StackRefs2 {
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v1;
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & v2;
    int & v0;
    __host__ __device__ StackRefs2() = default;
    __host__ __device__ StackRefs2(int & t0, Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & t1, Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> & t2) : v0(t0), v1(t1), v2(t2) {}
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
struct Union11_0 { // None
};
struct Union11_1 { // Some
    static_array_list<Union3,5> v0;
    Union5 v1;
    static_array<int,3> v3;
    int v4;
    unsigned int v5;
    bool v2;
    __host__ __device__ Union11_1(static_array_list<Union3,5> t0, Union5 t1, bool t2, static_array<int,3> t3, int t4, unsigned int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __host__ __device__ Union11_1() = delete;
};
struct Union11 {
    union {
        Union11_0 case0; // None
        Union11_1 case1; // Some
    };
    unsigned char tag{255};
    __host__ __device__ Union11() {}
    __host__ __device__ Union11(Union11_0 t) : tag(0), case0(t) {} // None
    __host__ __device__ Union11(Union11_1 t) : tag(1), case1(t) {} // Some
    __host__ __device__ Union11(const Union11 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union11_0(x.case0); break; // None
            case 1: new (&this->case1) Union11_1(x.case1); break; // Some
        }
    }
    __host__ __device__ Union11(const Union11 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union11_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union11_1(std::move(x.case1)); break; // Some
        }
    }
    __host__ __device__ Union11 & operator=(const Union11 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union11();
            new (this) Union11{x};
        }
        return *this;
    }
    __host__ __device__ Union11 & operator=(const Union11 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union11();
            new (this) Union11{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union11() {
        switch(this->tag){
            case 0: this->case0.~Union11_0(); break; // None
            case 1: this->case1.~Union11_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Tuple13 {
    static_array_list<Union3,5> v0;
    Union5 v1;
    static_array<int,3> v3;
    int v4;
    unsigned int v5;
    bool v2;
    __host__ __device__ Tuple13() = default;
    __host__ __device__ Tuple13(static_array_list<Union3,5> t0, Union5 t1, bool t2, static_array<int,3> t3, int t4, unsigned int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
};
struct Tuple15 {
    int v0;
    int v1;
    int v2;
    __host__ __device__ Tuple15() = default;
    __host__ __device__ Tuple15(int t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Union16_0 { // None
};
struct Union16_1 { // Some
    Union5 v0;
    __host__ __device__ Union16_1(Union5 t0) : v0(t0) {}
    __host__ __device__ Union16_1() = delete;
};
struct Union16 {
    union {
        Union16_0 case0; // None
        Union16_1 case1; // Some
    };
    unsigned char tag{255};
    __host__ __device__ Union16() {}
    __host__ __device__ Union16(Union16_0 t) : tag(0), case0(t) {} // None
    __host__ __device__ Union16(Union16_1 t) : tag(1), case1(t) {} // Some
    __host__ __device__ Union16(const Union16 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union16_0(x.case0); break; // None
            case 1: new (&this->case1) Union16_1(x.case1); break; // Some
        }
    }
    __host__ __device__ Union16(const Union16 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union16_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union16_1(std::move(x.case1)); break; // Some
        }
    }
    __host__ __device__ Union16 & operator=(const Union16 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union16();
            new (this) Union16{x};
        }
        return *this;
    }
    __host__ __device__ Union16 & operator=(const Union16 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union16();
            new (this) Union16{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union16() {
        switch(this->tag){
            case 0: this->case0.~Union16_0(); break; // None
            case 1: this->case1.~Union16_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct StackMut17 {
    Union16 v0;
    __host__ __device__ StackMut17() = default;
    __host__ __device__ StackMut17(Union16 t0) : v0(t0) {}
};
struct Union19_0 { // None
};
struct Union19_1 { // Some
    Union3 v0;
    __host__ __device__ Union19_1(Union3 t0) : v0(t0) {}
    __host__ __device__ Union19_1() = delete;
};
struct Union19 {
    union {
        Union19_0 case0; // None
        Union19_1 case1; // Some
    };
    unsigned char tag{255};
    __host__ __device__ Union19() {}
    __host__ __device__ Union19(Union19_0 t) : tag(0), case0(t) {} // None
    __host__ __device__ Union19(Union19_1 t) : tag(1), case1(t) {} // Some
    __host__ __device__ Union19(const Union19 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union19_0(x.case0); break; // None
            case 1: new (&this->case1) Union19_1(x.case1); break; // Some
        }
    }
    __host__ __device__ Union19(const Union19 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union19_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union19_1(std::move(x.case1)); break; // Some
        }
    }
    __host__ __device__ Union19 & operator=(const Union19 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union19();
            new (this) Union19{x};
        }
        return *this;
    }
    __host__ __device__ Union19 & operator=(const Union19 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union19();
            new (this) Union19{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union19() {
        switch(this->tag){
            case 0: this->case0.~Union19_0(); break; // None
            case 1: this->case1.~Union19_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct StackMut20 {
    Union19 v0;
    __host__ __device__ StackMut20() = default;
    __host__ __device__ StackMut20(Union19 t0) : v0(t0) {}
};
