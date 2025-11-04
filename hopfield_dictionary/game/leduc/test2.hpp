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
struct Union4;
struct StackMut6;
#ifdef __CUDACC__
// The Cuda device methods forward declarations
// The Cuda host methods forward declarations
// The Cuda device main defs forward declarations
#else
// The Cpp host methods forward declarations
void method_5(float * v0, static_array_list<Union3,5> v1, Union4 v2, bool v3, static_array<int,3> v4, int v5, unsigned int v6);
void method_10(Eigen::Matrix<float,1,115> & v0, Eigen::Matrix<float,1,48> & v1, Eigen::Matrix<float,1,115> & v2, Eigen::Matrix<float,1,48> & v3);
void method_12(Union3 v0);
// The Cuda host main defs forward declarations
#endif
// The Cuda type definitions
#ifdef __CUDACC__
#endif
// The Cpp type definitions
struct StackRefs2 {
    Eigen::Matrix<float,1,115> & v1;
    Eigen::Matrix<float,1,48> & v2;
    int & v0;
    __host__ __device__ StackRefs2() = default;
    __host__ __device__ StackRefs2(int & t0, Eigen::Matrix<float,1,115> & t1, Eigen::Matrix<float,1,48> & t2) : v0(t0), v1(t1), v2(t2) {}
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
struct Union4_0 { // Jack
};
struct Union4_1 { // King
};
struct Union4_2 { // Queen
};
struct Union4 {
    union {
        Union4_0 case0; // Jack
        Union4_1 case1; // King
        Union4_2 case2; // Queen
    };
    unsigned char tag{255};
    __host__ __device__ Union4() {}
    __host__ __device__ Union4(Union4_0 t) : tag(0), case0(t) {} // Jack
    __host__ __device__ Union4(Union4_1 t) : tag(1), case1(t) {} // King
    __host__ __device__ Union4(Union4_2 t) : tag(2), case2(t) {} // Queen
    __host__ __device__ Union4(const Union4 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(x.case0); break; // Jack
            case 1: new (&this->case1) Union4_1(x.case1); break; // King
            case 2: new (&this->case2) Union4_2(x.case2); break; // Queen
        }
    }
    __host__ __device__ Union4(const Union4 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(std::move(x.case0)); break; // Jack
            case 1: new (&this->case1) Union4_1(std::move(x.case1)); break; // King
            case 2: new (&this->case2) Union4_2(std::move(x.case2)); break; // Queen
        }
    }
    __host__ __device__ Union4 & operator=(const Union4 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Jack
                case 1: this->case1 = x.case1; break; // King
                case 2: this->case2 = x.case2; break; // Queen
            }
        } else {
            this->~Union4();
            new (this) Union4{x};
        }
        return *this;
    }
    __host__ __device__ Union4 & operator=(const Union4 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // Jack
                case 1: this->case1 = std::move(x.case1); break; // King
                case 2: this->case2 = std::move(x.case2); break; // Queen
            }
        } else {
            this->~Union4();
            new (this) Union4{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union4() {
        switch(this->tag){
            case 0: this->case0.~Union4_0(); break; // Jack
            case 1: this->case1.~Union4_1(); break; // King
            case 2: this->case2.~Union4_2(); break; // Queen
        }
        this->tag = 255;
    }
};
struct StackMut6 {
    int v0;
    __host__ __device__ StackMut6() = default;
    __host__ __device__ StackMut6(int t0) : v0(t0) {}
};
