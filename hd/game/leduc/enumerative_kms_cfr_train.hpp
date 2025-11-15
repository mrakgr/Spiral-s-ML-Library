#pragma once
#include "enumerative_kms_cfr_train.corelib.hpp"
#ifdef __CUDACC__
#ifdef __CUDA_ARCH__
// Cuda device backend
#else
// Cuda host backend
#endif
#else
// Cpp host backend
#include <unordered_map>
#include <arrayfire.h>
#include <xoshiro.h>
struct Union1;
struct Union2;
struct Union0;
struct Tuple0;
typedef unsigned long long (* Fun0)(Tuple0);
typedef bool (* Fun1)(Tuple0, Tuple0);
struct Tuple1;
struct StackRefs0;
struct Tuple2;
struct StackRefs1;
struct StackRefs2;
struct StackMut0;
struct StackRefs3;
struct StackRefs4;
struct Tuple3;
struct Union4;
struct Union3;
struct StackMut1;
struct Tuple4;
struct Union5;
struct Tuple5;
struct Union6;
struct Tuple6;
struct Union7;
struct StackMut2;
struct StackRefs5;
struct StackRefs6;
struct StackRefs7;
struct StackRefs8;
struct StackRefs9;
struct StackRefs10;
struct Union8;
float loop_1(StackRefs3 & v0, StackRefs0 & v1, StackRefs2 & v2, xso::rng & v3, StackMut0 & v4, StackRefs4 & v5, StackMut1 & v6, Union5 v7);
static_array<float,3> relu_4(static_array<float,3> v0);
static_array<float,3> masking_normalize_5(static_array<float,3> v0, static_array<bool,3> v1);
static_array<float,3> regret_match_3(static_array<float,3> v0, static_array<bool,3> v1);
float method_2(xso::rng & v0, StackRefs3 & v1, StackRefs4 & v2, Union4 v3, bool v4, static_array<Union1,2> v5, int v6, static_array<int,2> v7, int v8, StackRefs0 & v9, StackRefs2 & v10, StackMut0 & v11, StackMut1 & v12);
void method_7(float * v0, static_array_list<Union0,32> v1);
void method_8(StackRefs5 & v0, StackRefs6 & v1);
void method_9(StackRefs7 & v0, StackRefs8 & v1);
void method_10(StackRefs9 & v0, StackRefs10 & v1);
float method_6(xso::rng & v0, StackRefs3 & v1, StackRefs4 & v2, Union4 v3, bool v4, static_array<Union1,2> v5, int v6, static_array<int,2> v7, int v8, StackRefs0 & v9, StackRefs2 & v10, StackMut0 & v11, StackMut1 & v12);
int tag_12(Union1 v0);
bool is_pair_13(int v0, int v1);
Tuple5 order_14(int v0, int v1);
Union8 compare_hands_11(Union4 v0, bool v1, static_array<Union1,2> v2, int v3, static_array<int,2> v4, int v5);
float body_0(StackRefs3 & v0, StackRefs0 & v1, StackRefs2 & v2, xso::rng & v3, StackMut0 & v4, StackRefs4 & v5, Union3 v6);
float loop_16(StackRefs3 & v0, StackRefs0 & v1, StackRefs2 & v2, xso::rng & v3, StackMut0 & v4, StackRefs4 & v5, StackMut1 & v6, Union5 v7);
float method_17(xso::rng & v0, StackRefs3 & v1, StackRefs4 & v2, Union4 v3, bool v4, static_array<Union1,2> v5, int v6, static_array<int,2> v7, int v8, StackRefs0 & v9, StackRefs2 & v10, StackMut0 & v11, StackMut1 & v12);
float method_18(xso::rng & v0, StackRefs3 & v1, StackRefs4 & v2, Union4 v3, bool v4, static_array<Union1,2> v5, int v6, static_array<int,2> v7, int v8, StackRefs0 & v9, StackRefs2 & v10, StackMut0 & v11, StackMut1 & v12);
float body_15(StackRefs3 & v0, StackRefs0 & v1, StackRefs2 & v2, xso::rng & v3, StackMut0 & v4, StackRefs4 & v5, Union3 v6);
int main();
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
struct Union2_0 { // Call
};
struct Union2_1 { // Fold
};
struct Union2_2 { // Raise
};
struct Union2 {
    union {
        Union2_0 case0; // Call
        Union2_1 case1; // Fold
        Union2_2 case2; // Raise
    };
    unsigned char tag{255};
    __host__ __device__ Union2() {}
    __host__ __device__ Union2(Union2_0 t) : tag(0), case0(t) {} // Call
    __host__ __device__ Union2(Union2_1 t) : tag(1), case1(t) {} // Fold
    __host__ __device__ Union2(Union2_2 t) : tag(2), case2(t) {} // Raise
    __host__ __device__ Union2(const Union2 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(x.case0); break; // Call
            case 1: new (&this->case1) Union2_1(x.case1); break; // Fold
            case 2: new (&this->case2) Union2_2(x.case2); break; // Raise
        }
    }
    __host__ __device__ Union2(const Union2 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(std::move(x.case0)); break; // Call
            case 1: new (&this->case1) Union2_1(std::move(x.case1)); break; // Fold
            case 2: new (&this->case2) Union2_2(std::move(x.case2)); break; // Raise
        }
    }
    __host__ __device__ Union2 & operator=(const Union2 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Call
                case 1: this->case1 = x.case1; break; // Fold
                case 2: this->case2 = x.case2; break; // Raise
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
                case 0: this->case0 = std::move(x.case0); break; // Call
                case 1: this->case1 = std::move(x.case1); break; // Fold
                case 2: this->case2 = std::move(x.case2); break; // Raise
            }
        } else {
            this->~Union2();
            new (this) Union2{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union2() {
        switch(this->tag){
            case 0: this->case0.~Union2_0(); break; // Call
            case 1: this->case1.~Union2_1(); break; // Fold
            case 2: this->case2.~Union2_2(); break; // Raise
        }
        this->tag = 255;
    }
};
struct Union0_0 { // CommunityCardIs
    Union1 v0;
    __host__ __device__ Union0_0(Union1 t0) : v0(t0) {}
    __host__ __device__ Union0_0() = delete;
};
struct Union0_1 { // Hidden
};
struct Union0_2 { // PlayerAction
    Union2 v1;
    int v0;
    __host__ __device__ Union0_2(int t0, Union2 t1) : v0(t0), v1(t1) {}
    __host__ __device__ Union0_2() = delete;
};
struct Union0_3 { // PlayerGotCard
    Union1 v1;
    int v0;
    __host__ __device__ Union0_3(int t0, Union1 t1) : v0(t0), v1(t1) {}
    __host__ __device__ Union0_3() = delete;
};
struct Union0_4 { // Showdown
    static_array<Union1,2> v0;
    int v1;
    int v2;
    __host__ __device__ Union0_4(static_array<Union1,2> t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
    __host__ __device__ Union0_4() = delete;
};
struct Union0 {
    union {
        Union0_0 case0; // CommunityCardIs
        Union0_1 case1; // Hidden
        Union0_2 case2; // PlayerAction
        Union0_3 case3; // PlayerGotCard
        Union0_4 case4; // Showdown
    };
    unsigned char tag{255};
    __host__ __device__ Union0() {}
    __host__ __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // CommunityCardIs
    __host__ __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // Hidden
    __host__ __device__ Union0(Union0_2 t) : tag(2), case2(t) {} // PlayerAction
    __host__ __device__ Union0(Union0_3 t) : tag(3), case3(t) {} // PlayerGotCard
    __host__ __device__ Union0(Union0_4 t) : tag(4), case4(t) {} // Showdown
    __host__ __device__ Union0(const Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // CommunityCardIs
            case 1: new (&this->case1) Union0_1(x.case1); break; // Hidden
            case 2: new (&this->case2) Union0_2(x.case2); break; // PlayerAction
            case 3: new (&this->case3) Union0_3(x.case3); break; // PlayerGotCard
            case 4: new (&this->case4) Union0_4(x.case4); break; // Showdown
        }
    }
    __host__ __device__ Union0(const Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // CommunityCardIs
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // Hidden
            case 2: new (&this->case2) Union0_2(std::move(x.case2)); break; // PlayerAction
            case 3: new (&this->case3) Union0_3(std::move(x.case3)); break; // PlayerGotCard
            case 4: new (&this->case4) Union0_4(std::move(x.case4)); break; // Showdown
        }
    }
    __host__ __device__ Union0 & operator=(const Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // CommunityCardIs
                case 1: this->case1 = x.case1; break; // Hidden
                case 2: this->case2 = x.case2; break; // PlayerAction
                case 3: this->case3 = x.case3; break; // PlayerGotCard
                case 4: this->case4 = x.case4; break; // Showdown
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
                case 0: this->case0 = std::move(x.case0); break; // CommunityCardIs
                case 1: this->case1 = std::move(x.case1); break; // Hidden
                case 2: this->case2 = std::move(x.case2); break; // PlayerAction
                case 3: this->case3 = std::move(x.case3); break; // PlayerGotCard
                case 4: this->case4 = std::move(x.case4); break; // Showdown
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // CommunityCardIs
            case 1: this->case1.~Union0_1(); break; // Hidden
            case 2: this->case2.~Union0_2(); break; // PlayerAction
            case 3: this->case3.~Union0_3(); break; // PlayerGotCard
            case 4: this->case4.~Union0_4(); break; // Showdown
        }
        this->tag = 255;
    }
};
struct Tuple0 {
    unsigned long long v0;
    static_array_list<Union0,32> v1;
    __host__ __device__ Tuple0() = default;
    __host__ __device__ Tuple0(unsigned long long t0, static_array_list<Union0,32> t1) : v0(t0), v1(t1) {}
};
struct Tuple1 {
    static_array<float,3> v0;
    static_array<float,3> v1;
    __host__ __device__ Tuple1() = default;
    __host__ __device__ Tuple1(static_array<float,3> t0, static_array<float,3> t1) : v0(t0), v1(t1) {}
};
struct StackRefs0 {
    std::unordered_map<Tuple0, Tuple1, Fun0, Fun1> & v0;
    __host__ __device__ StackRefs0() = default;
    __host__ __device__ StackRefs0(std::unordered_map<Tuple0, Tuple1, Fun0, Fun1> & t0) : v0(t0) {}
};
struct Tuple2 {
    float v0;
    float v1;
    __host__ __device__ Tuple2() = default;
    __host__ __device__ Tuple2(float t0, float t1) : v0(t0), v1(t1) {}
};
struct StackRefs1 {
    std::unordered_map<Tuple0, static_array<Tuple2,3>, Fun0, Fun1> & v0;
    __host__ __device__ StackRefs1() = default;
    __host__ __device__ StackRefs1(std::unordered_map<Tuple0, static_array<Tuple2,3>, Fun0, Fun1> & t0) : v0(t0) {}
};
struct StackRefs2 {
    af::array & v0;
    af::array & v1;
    __host__ __device__ StackRefs2() = default;
    __host__ __device__ StackRefs2(af::array & t0, af::array & t1) : v0(t0), v1(t1) {}
};
struct StackMut0 {
    unsigned int v0;
    __host__ __device__ StackMut0() = default;
    __host__ __device__ StackMut0(unsigned int t0) : v0(t0) {}
};
struct StackRefs3 {
    static_array_list<Union0,32> & v0;
    __host__ __device__ StackRefs3() = default;
    __host__ __device__ StackRefs3(static_array_list<Union0,32> & t0) : v0(t0) {}
};
struct StackRefs4 {
    static_array<Tuple2,2> & v0;
    __host__ __device__ StackRefs4() = default;
    __host__ __device__ StackRefs4(static_array<Tuple2,2> & t0) : v0(t0) {}
};
struct Tuple3 {
    int v0;
    float v1;
    __host__ __device__ Tuple3() = default;
    __host__ __device__ Tuple3(int t0, float t1) : v0(t0), v1(t1) {}
};
struct Union4_0 { // None
};
struct Union4_1 { // Some
    Union1 v0;
    __host__ __device__ Union4_1(Union1 t0) : v0(t0) {}
    __host__ __device__ Union4_1() = delete;
};
struct Union4 {
    union {
        Union4_0 case0; // None
        Union4_1 case1; // Some
    };
    unsigned char tag{255};
    __host__ __device__ Union4() {}
    __host__ __device__ Union4(Union4_0 t) : tag(0), case0(t) {} // None
    __host__ __device__ Union4(Union4_1 t) : tag(1), case1(t) {} // Some
    __host__ __device__ Union4(const Union4 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(x.case0); break; // None
            case 1: new (&this->case1) Union4_1(x.case1); break; // Some
        }
    }
    __host__ __device__ Union4(const Union4 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union4_1(std::move(x.case1)); break; // Some
        }
    }
    __host__ __device__ Union4 & operator=(const Union4 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
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
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union4();
            new (this) Union4{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union4() {
        switch(this->tag){
            case 0: this->case0.~Union4_0(); break; // None
            case 1: this->case1.~Union4_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union3_0 { // ChanceCommunityCard
    Union4 v0;
    static_array<Union1,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union3_0(Union4 t0, bool t1, static_array<Union1,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __host__ __device__ Union3_0() = delete;
};
struct Union3_1 { // ChanceInit
};
struct Union3_2 { // Round
    Union4 v0;
    static_array<Union1,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union3_2(Union4 t0, bool t1, static_array<Union1,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __host__ __device__ Union3_2() = delete;
};
struct Union3_3 { // RoundWithAction
    Union4 v0;
    static_array<Union1,2> v2;
    static_array<int,2> v4;
    Union2 v6;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union3_3(Union4 t0, bool t1, static_array<Union1,2> t2, int t3, static_array<int,2> t4, int t5, Union2 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __host__ __device__ Union3_3() = delete;
};
struct Union3_4 { // TerminalCall
    Union4 v0;
    static_array<Union1,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union3_4(Union4 t0, bool t1, static_array<Union1,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __host__ __device__ Union3_4() = delete;
};
struct Union3_5 { // TerminalFold
    Union4 v0;
    static_array<Union1,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union3_5(Union4 t0, bool t1, static_array<Union1,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __host__ __device__ Union3_5() = delete;
};
struct Union3 {
    union {
        Union3_0 case0; // ChanceCommunityCard
        Union3_1 case1; // ChanceInit
        Union3_2 case2; // Round
        Union3_3 case3; // RoundWithAction
        Union3_4 case4; // TerminalCall
        Union3_5 case5; // TerminalFold
    };
    unsigned char tag{255};
    __host__ __device__ Union3() {}
    __host__ __device__ Union3(Union3_0 t) : tag(0), case0(t) {} // ChanceCommunityCard
    __host__ __device__ Union3(Union3_1 t) : tag(1), case1(t) {} // ChanceInit
    __host__ __device__ Union3(Union3_2 t) : tag(2), case2(t) {} // Round
    __host__ __device__ Union3(Union3_3 t) : tag(3), case3(t) {} // RoundWithAction
    __host__ __device__ Union3(Union3_4 t) : tag(4), case4(t) {} // TerminalCall
    __host__ __device__ Union3(Union3_5 t) : tag(5), case5(t) {} // TerminalFold
    __host__ __device__ Union3(const Union3 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(x.case0); break; // ChanceCommunityCard
            case 1: new (&this->case1) Union3_1(x.case1); break; // ChanceInit
            case 2: new (&this->case2) Union3_2(x.case2); break; // Round
            case 3: new (&this->case3) Union3_3(x.case3); break; // RoundWithAction
            case 4: new (&this->case4) Union3_4(x.case4); break; // TerminalCall
            case 5: new (&this->case5) Union3_5(x.case5); break; // TerminalFold
        }
    }
    __host__ __device__ Union3(const Union3 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(std::move(x.case0)); break; // ChanceCommunityCard
            case 1: new (&this->case1) Union3_1(std::move(x.case1)); break; // ChanceInit
            case 2: new (&this->case2) Union3_2(std::move(x.case2)); break; // Round
            case 3: new (&this->case3) Union3_3(std::move(x.case3)); break; // RoundWithAction
            case 4: new (&this->case4) Union3_4(std::move(x.case4)); break; // TerminalCall
            case 5: new (&this->case5) Union3_5(std::move(x.case5)); break; // TerminalFold
        }
    }
    __host__ __device__ Union3 & operator=(const Union3 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // ChanceCommunityCard
                case 1: this->case1 = x.case1; break; // ChanceInit
                case 2: this->case2 = x.case2; break; // Round
                case 3: this->case3 = x.case3; break; // RoundWithAction
                case 4: this->case4 = x.case4; break; // TerminalCall
                case 5: this->case5 = x.case5; break; // TerminalFold
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
                case 0: this->case0 = std::move(x.case0); break; // ChanceCommunityCard
                case 1: this->case1 = std::move(x.case1); break; // ChanceInit
                case 2: this->case2 = std::move(x.case2); break; // Round
                case 3: this->case3 = std::move(x.case3); break; // RoundWithAction
                case 4: this->case4 = std::move(x.case4); break; // TerminalCall
                case 5: this->case5 = std::move(x.case5); break; // TerminalFold
            }
        } else {
            this->~Union3();
            new (this) Union3{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union3() {
        switch(this->tag){
            case 0: this->case0.~Union3_0(); break; // ChanceCommunityCard
            case 1: this->case1.~Union3_1(); break; // ChanceInit
            case 2: this->case2.~Union3_2(); break; // Round
            case 3: this->case3.~Union3_3(); break; // RoundWithAction
            case 4: this->case4.~Union3_4(); break; // TerminalCall
            case 5: this->case5.~Union3_5(); break; // TerminalFold
        }
        this->tag = 255;
    }
};
struct StackMut1 {
    float v0;
    __host__ __device__ StackMut1() = default;
    __host__ __device__ StackMut1(float t0) : v0(t0) {}
};
struct Tuple4 {
    int v0;
    float v1;
    float v2;
    __host__ __device__ Tuple4() = default;
    __host__ __device__ Tuple4(int t0, float t1, float t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Union5_0 { // T_game_chance_community_card
    Union4 v0;
    static_array<Union1,2> v2;
    static_array<int,2> v4;
    Union1 v6;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union5_0(Union4 t0, bool t1, static_array<Union1,2> t2, int t3, static_array<int,2> t4, int t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __host__ __device__ Union5_0() = delete;
};
struct Union5_1 { // T_game_chance_init
    Union1 v0;
    Union1 v1;
    __host__ __device__ Union5_1(Union1 t0, Union1 t1) : v0(t0), v1(t1) {}
    __host__ __device__ Union5_1() = delete;
};
struct Union5_2 { // T_game_round
    Union4 v0;
    static_array<Union1,2> v2;
    static_array<int,2> v4;
    Union2 v6;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union5_2(Union4 t0, bool t1, static_array<Union1,2> t2, int t3, static_array<int,2> t4, int t5, Union2 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __host__ __device__ Union5_2() = delete;
};
struct Union5_3 { // T_none
};
struct Union5 {
    union {
        Union5_0 case0; // T_game_chance_community_card
        Union5_1 case1; // T_game_chance_init
        Union5_2 case2; // T_game_round
        Union5_3 case3; // T_none
    };
    unsigned char tag{255};
    __host__ __device__ Union5() {}
    __host__ __device__ Union5(Union5_0 t) : tag(0), case0(t) {} // T_game_chance_community_card
    __host__ __device__ Union5(Union5_1 t) : tag(1), case1(t) {} // T_game_chance_init
    __host__ __device__ Union5(Union5_2 t) : tag(2), case2(t) {} // T_game_round
    __host__ __device__ Union5(Union5_3 t) : tag(3), case3(t) {} // T_none
    __host__ __device__ Union5(const Union5 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union5_0(x.case0); break; // T_game_chance_community_card
            case 1: new (&this->case1) Union5_1(x.case1); break; // T_game_chance_init
            case 2: new (&this->case2) Union5_2(x.case2); break; // T_game_round
            case 3: new (&this->case3) Union5_3(x.case3); break; // T_none
        }
    }
    __host__ __device__ Union5(const Union5 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union5_0(std::move(x.case0)); break; // T_game_chance_community_card
            case 1: new (&this->case1) Union5_1(std::move(x.case1)); break; // T_game_chance_init
            case 2: new (&this->case2) Union5_2(std::move(x.case2)); break; // T_game_round
            case 3: new (&this->case3) Union5_3(std::move(x.case3)); break; // T_none
        }
    }
    __host__ __device__ Union5 & operator=(const Union5 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // T_game_chance_community_card
                case 1: this->case1 = x.case1; break; // T_game_chance_init
                case 2: this->case2 = x.case2; break; // T_game_round
                case 3: this->case3 = x.case3; break; // T_none
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
                case 0: this->case0 = std::move(x.case0); break; // T_game_chance_community_card
                case 1: this->case1 = std::move(x.case1); break; // T_game_chance_init
                case 2: this->case2 = std::move(x.case2); break; // T_game_round
                case 3: this->case3 = std::move(x.case3); break; // T_none
            }
        } else {
            this->~Union5();
            new (this) Union5{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union5() {
        switch(this->tag){
            case 0: this->case0.~Union5_0(); break; // T_game_chance_community_card
            case 1: this->case1.~Union5_1(); break; // T_game_chance_init
            case 2: this->case2.~Union5_2(); break; // T_game_round
            case 3: this->case3.~Union5_3(); break; // T_none
        }
        this->tag = 255;
    }
};
struct Tuple5 {
    int v0;
    int v1;
    __host__ __device__ Tuple5() = default;
    __host__ __device__ Tuple5(int t0, int t1) : v0(t0), v1(t1) {}
};
struct Union6_0 { // None
};
struct Union6_1 { // Some
    Union2 v0;
    __host__ __device__ Union6_1(Union2 t0) : v0(t0) {}
    __host__ __device__ Union6_1() = delete;
};
struct Union6 {
    union {
        Union6_0 case0; // None
        Union6_1 case1; // Some
    };
    unsigned char tag{255};
    __host__ __device__ Union6() {}
    __host__ __device__ Union6(Union6_0 t) : tag(0), case0(t) {} // None
    __host__ __device__ Union6(Union6_1 t) : tag(1), case1(t) {} // Some
    __host__ __device__ Union6(const Union6 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(x.case0); break; // None
            case 1: new (&this->case1) Union6_1(x.case1); break; // Some
        }
    }
    __host__ __device__ Union6(const Union6 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union6_1(std::move(x.case1)); break; // Some
        }
    }
    __host__ __device__ Union6 & operator=(const Union6 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union6();
            new (this) Union6{x};
        }
        return *this;
    }
    __host__ __device__ Union6 & operator=(const Union6 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union6();
            new (this) Union6{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union6() {
        switch(this->tag){
            case 0: this->case0.~Union6_0(); break; // None
            case 1: this->case1.~Union6_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Tuple6 {
    unsigned long long v1;
    unsigned long long v2;
    int v0;
    __host__ __device__ Tuple6() = default;
    __host__ __device__ Tuple6(int t0, unsigned long long t1, unsigned long long t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Union7_0 { // None
};
struct Union7_1 { // Some
    static_array<float,3> v0;
    static_array<float,3> v1;
    __host__ __device__ Union7_1(static_array<float,3> t0, static_array<float,3> t1) : v0(t0), v1(t1) {}
    __host__ __device__ Union7_1() = delete;
};
struct Union7 {
    union {
        Union7_0 case0; // None
        Union7_1 case1; // Some
    };
    unsigned char tag{255};
    __host__ __device__ Union7() {}
    __host__ __device__ Union7(Union7_0 t) : tag(0), case0(t) {} // None
    __host__ __device__ Union7(Union7_1 t) : tag(1), case1(t) {} // Some
    __host__ __device__ Union7(const Union7 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(x.case0); break; // None
            case 1: new (&this->case1) Union7_1(x.case1); break; // Some
        }
    }
    __host__ __device__ Union7(const Union7 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union7_1(std::move(x.case1)); break; // Some
        }
    }
    __host__ __device__ Union7 & operator=(const Union7 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
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
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union7();
            new (this) Union7{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union7() {
        switch(this->tag){
            case 0: this->case0.~Union7_0(); break; // None
            case 1: this->case1.~Union7_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct StackMut2 {
    int v0;
    __host__ __device__ StackMut2() = default;
    __host__ __device__ StackMut2(int t0) : v0(t0) {}
};
struct StackRefs5 {
    af::array & v0;
    af::array & v1;
    af::array & v2;
    __host__ __device__ StackRefs5() = default;
    __host__ __device__ StackRefs5(af::array & t0, af::array & t1, af::array & t2) : v0(t0), v1(t1), v2(t2) {}
};
struct StackRefs6 {
    af::array & v0;
    af::array & v1;
    __host__ __device__ StackRefs6() = default;
    __host__ __device__ StackRefs6(af::array & t0, af::array & t1) : v0(t0), v1(t1) {}
};
struct StackRefs7 {
    af::array & v0;
    af::array & v1;
    __host__ __device__ StackRefs7() = default;
    __host__ __device__ StackRefs7(af::array & t0, af::array & t1) : v0(t0), v1(t1) {}
};
struct StackRefs8 {
    af::array & v0;
    __host__ __device__ StackRefs8() = default;
    __host__ __device__ StackRefs8(af::array & t0) : v0(t0) {}
};
struct StackRefs9 {
    af::array & v0;
    af::array & v1;
    __host__ __device__ StackRefs9() = default;
    __host__ __device__ StackRefs9(af::array & t0, af::array & t1) : v0(t0), v1(t1) {}
};
struct StackRefs10 {
    af::array & v0;
    __host__ __device__ StackRefs10() = default;
    __host__ __device__ StackRefs10(af::array & t0) : v0(t0) {}
};
struct Union8_0 { // Eq
};
struct Union8_1 { // Gt
};
struct Union8_2 { // Lt
};
struct Union8 {
    union {
        Union8_0 case0; // Eq
        Union8_1 case1; // Gt
        Union8_2 case2; // Lt
    };
    unsigned char tag{255};
    __host__ __device__ Union8() {}
    __host__ __device__ Union8(Union8_0 t) : tag(0), case0(t) {} // Eq
    __host__ __device__ Union8(Union8_1 t) : tag(1), case1(t) {} // Gt
    __host__ __device__ Union8(Union8_2 t) : tag(2), case2(t) {} // Lt
    __host__ __device__ Union8(const Union8 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union8_0(x.case0); break; // Eq
            case 1: new (&this->case1) Union8_1(x.case1); break; // Gt
            case 2: new (&this->case2) Union8_2(x.case2); break; // Lt
        }
    }
    __host__ __device__ Union8(const Union8 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union8_0(std::move(x.case0)); break; // Eq
            case 1: new (&this->case1) Union8_1(std::move(x.case1)); break; // Gt
            case 2: new (&this->case2) Union8_2(std::move(x.case2)); break; // Lt
        }
    }
    __host__ __device__ Union8 & operator=(const Union8 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Eq
                case 1: this->case1 = x.case1; break; // Gt
                case 2: this->case2 = x.case2; break; // Lt
            }
        } else {
            this->~Union8();
            new (this) Union8{x};
        }
        return *this;
    }
    __host__ __device__ Union8 & operator=(const Union8 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // Eq
                case 1: this->case1 = std::move(x.case1); break; // Gt
                case 2: this->case2 = std::move(x.case2); break; // Lt
            }
        } else {
            this->~Union8();
            new (this) Union8{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union8() {
        switch(this->tag){
            case 0: this->case0.~Union8_0(); break; // Eq
            case 1: this->case1.~Union8_1(); break; // Gt
            case 2: this->case2.~Union8_2(); break; // Lt
        }
        this->tag = 255;
    }
};
#endif
