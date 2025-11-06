#pragma once
#include "hd_cfr_train.corelib.hpp"
// Cuda globals
#ifdef __CUDACC__
#endif
// Cpp globals
#include <unordered_map>
#include <xoshiro.h>
// The Cuda type forward declarations
#ifdef __CUDACC__
#endif
// The Cpp type forward declarations
struct Tuple4;
struct Union5;
struct Union6;
struct Union7;
typedef unsigned long long (* Fun8)(Tuple4);
typedef bool (* Fun12)(Tuple4, Tuple4);
struct Tuple13;
struct StackRefs14;
struct Tuple15;
struct StackRefs16;
struct StackMut18;
struct StackRefs19;
struct StackRefs20;
struct Tuple21;
struct Union23;
struct Union24;
struct StackMut26;
struct Tuple28;
struct StackMut31;
struct StackMut32;
struct Union34;
struct Tuple36;
struct Tuple38;
struct Union39;
struct Union41;
struct Union42;
struct Union51;
struct Tuple58;
#ifdef __CUDACC__
// The Cuda device methods forward declarations
// The Cuda host methods forward declarations
// The Cuda device main defs forward declarations
#else
// The Cpp host methods forward declarations
unsigned int loop_29(unsigned int v0, xso::rng & v1);
unsigned int find_nth_set_bit_30(int v0, unsigned int v1, unsigned int v2);
Tuple28 draw_card_27(xso::rng & v0, unsigned int v1);
float loop_35(StackRefs19 & v0, StackRefs16 & v1, StackRefs14 & v2, xso::rng & v3, StackMut18 & v4, StackRefs20 & v5, StackMut26 & v6, Union34 v7);
static_array<float,3> relu_44(static_array<float,3> v0);
static_array<float,3> masking_normalize_45(static_array<float,3> v0, static_array<bool,3> v1);
static_array<float,3> regret_match_43(static_array<float,3> v0, static_array<bool,3> v1);
int loop_50(static_array<float,3> v0, float v1, int v2);
int pick_discrete__47(static_array<float,3> v0, float v1);
int sample_discrete__46(static_array<float,3> v0, xso::rng & v1);
float method_37(xso::rng & v0, StackRefs19 & v1, StackRefs20 & v2, Union24 v3, bool v4, static_array<Union6,2> v5, int v6, static_array<int,2> v7, int v8, StackRefs16 & v9, StackRefs14 & v10, StackMut18 & v11, StackMut26 & v12);
int tag_53(Union6 v0);
bool is_pair_54(int v0, int v1);
Tuple36 order_55(int v0, int v1);
Union51 compare_hands_52(Union24 v0, bool v1, static_array<Union6,2> v2, int v3, static_array<int,2> v4, int v5);
float body_25(StackRefs19 & v0, StackRefs16 & v1, StackRefs14 & v2, xso::rng & v3, StackMut18 & v4, StackRefs20 & v5, Union23 v6);
float loop_60(StackRefs19 & v0, StackRefs14 & v1, xso::rng & v2, StackMut18 & v3, StackRefs20 & v4, StackMut26 & v5, Union34 v6);
float method_61(xso::rng & v0, StackRefs19 & v1, StackRefs20 & v2, Union24 v3, bool v4, static_array<Union6,2> v5, int v6, static_array<int,2> v7, int v8, StackRefs14 & v9, StackMut18 & v10, StackMut26 & v11);
float method_62(xso::rng & v0, StackRefs19 & v1, StackRefs20 & v2, Union24 v3, bool v4, static_array<Union6,2> v5, int v6, static_array<int,2> v7, int v8, StackRefs14 & v9, StackMut18 & v10, StackMut26 & v11);
float body_57(StackRefs19 & v0, StackRefs14 & v1, xso::rng & v2, StackMut18 & v3, StackRefs20 & v4, Union23 v5);
// The Cuda host main defs forward declarations
#endif
// The Cuda type definitions
#ifdef __CUDACC__
#endif
// The Cpp type definitions
struct Tuple4 {
    unsigned long long v0;
    static_array_list<Union5,32> v1;
    __host__ __device__ Tuple4() = default;
    __host__ __device__ Tuple4(unsigned long long t0, static_array_list<Union5,32> t1) : v0(t0), v1(t1) {}
};
struct Union5_0 { // CommunityCardIs
    Union6 v0;
    __host__ __device__ Union5_0(Union6 t0) : v0(t0) {}
    __host__ __device__ Union5_0() = delete;
};
struct Union5_1 { // PlayerAction
    Union7 v1;
    int v0;
    __host__ __device__ Union5_1(int t0, Union7 t1) : v0(t0), v1(t1) {}
    __host__ __device__ Union5_1() = delete;
};
struct Union5_2 { // PlayerGotCard
    Union6 v1;
    int v0;
    __host__ __device__ Union5_2(int t0, Union6 t1) : v0(t0), v1(t1) {}
    __host__ __device__ Union5_2() = delete;
};
struct Union5_3 { // Showdown
    static_array<Union6,2> v0;
    int v1;
    int v2;
    __host__ __device__ Union5_3(static_array<Union6,2> t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
    __host__ __device__ Union5_3() = delete;
};
struct Union5 {
    union {
        Union5_0 case0; // CommunityCardIs
        Union5_1 case1; // PlayerAction
        Union5_2 case2; // PlayerGotCard
        Union5_3 case3; // Showdown
    };
    unsigned char tag{255};
    __host__ __device__ Union5() {}
    __host__ __device__ Union5(Union5_0 t) : tag(0), case0(t) {} // CommunityCardIs
    __host__ __device__ Union5(Union5_1 t) : tag(1), case1(t) {} // PlayerAction
    __host__ __device__ Union5(Union5_2 t) : tag(2), case2(t) {} // PlayerGotCard
    __host__ __device__ Union5(Union5_3 t) : tag(3), case3(t) {} // Showdown
    __host__ __device__ Union5(const Union5 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union5_0(x.case0); break; // CommunityCardIs
            case 1: new (&this->case1) Union5_1(x.case1); break; // PlayerAction
            case 2: new (&this->case2) Union5_2(x.case2); break; // PlayerGotCard
            case 3: new (&this->case3) Union5_3(x.case3); break; // Showdown
        }
    }
    __host__ __device__ Union5(const Union5 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union5_0(std::move(x.case0)); break; // CommunityCardIs
            case 1: new (&this->case1) Union5_1(std::move(x.case1)); break; // PlayerAction
            case 2: new (&this->case2) Union5_2(std::move(x.case2)); break; // PlayerGotCard
            case 3: new (&this->case3) Union5_3(std::move(x.case3)); break; // Showdown
        }
    }
    __host__ __device__ Union5 & operator=(const Union5 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // CommunityCardIs
                case 1: this->case1 = x.case1; break; // PlayerAction
                case 2: this->case2 = x.case2; break; // PlayerGotCard
                case 3: this->case3 = x.case3; break; // Showdown
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
                case 0: this->case0 = std::move(x.case0); break; // CommunityCardIs
                case 1: this->case1 = std::move(x.case1); break; // PlayerAction
                case 2: this->case2 = std::move(x.case2); break; // PlayerGotCard
                case 3: this->case3 = std::move(x.case3); break; // Showdown
            }
        } else {
            this->~Union5();
            new (this) Union5{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union5() {
        switch(this->tag){
            case 0: this->case0.~Union5_0(); break; // CommunityCardIs
            case 1: this->case1.~Union5_1(); break; // PlayerAction
            case 2: this->case2.~Union5_2(); break; // PlayerGotCard
            case 3: this->case3.~Union5_3(); break; // Showdown
        }
        this->tag = 255;
    }
};
struct Union6_0 { // Jack
};
struct Union6_1 { // King
};
struct Union6_2 { // Queen
};
struct Union6 {
    union {
        Union6_0 case0; // Jack
        Union6_1 case1; // King
        Union6_2 case2; // Queen
    };
    unsigned char tag{255};
    __host__ __device__ Union6() {}
    __host__ __device__ Union6(Union6_0 t) : tag(0), case0(t) {} // Jack
    __host__ __device__ Union6(Union6_1 t) : tag(1), case1(t) {} // King
    __host__ __device__ Union6(Union6_2 t) : tag(2), case2(t) {} // Queen
    __host__ __device__ Union6(const Union6 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(x.case0); break; // Jack
            case 1: new (&this->case1) Union6_1(x.case1); break; // King
            case 2: new (&this->case2) Union6_2(x.case2); break; // Queen
        }
    }
    __host__ __device__ Union6(const Union6 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(std::move(x.case0)); break; // Jack
            case 1: new (&this->case1) Union6_1(std::move(x.case1)); break; // King
            case 2: new (&this->case2) Union6_2(std::move(x.case2)); break; // Queen
        }
    }
    __host__ __device__ Union6 & operator=(const Union6 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Jack
                case 1: this->case1 = x.case1; break; // King
                case 2: this->case2 = x.case2; break; // Queen
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
                case 0: this->case0 = std::move(x.case0); break; // Jack
                case 1: this->case1 = std::move(x.case1); break; // King
                case 2: this->case2 = std::move(x.case2); break; // Queen
            }
        } else {
            this->~Union6();
            new (this) Union6{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union6() {
        switch(this->tag){
            case 0: this->case0.~Union6_0(); break; // Jack
            case 1: this->case1.~Union6_1(); break; // King
            case 2: this->case2.~Union6_2(); break; // Queen
        }
        this->tag = 255;
    }
};
struct Union7_0 { // Call
};
struct Union7_1 { // Fold
};
struct Union7_2 { // Raise
};
struct Union7 {
    union {
        Union7_0 case0; // Call
        Union7_1 case1; // Fold
        Union7_2 case2; // Raise
    };
    unsigned char tag{255};
    __host__ __device__ Union7() {}
    __host__ __device__ Union7(Union7_0 t) : tag(0), case0(t) {} // Call
    __host__ __device__ Union7(Union7_1 t) : tag(1), case1(t) {} // Fold
    __host__ __device__ Union7(Union7_2 t) : tag(2), case2(t) {} // Raise
    __host__ __device__ Union7(const Union7 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(x.case0); break; // Call
            case 1: new (&this->case1) Union7_1(x.case1); break; // Fold
            case 2: new (&this->case2) Union7_2(x.case2); break; // Raise
        }
    }
    __host__ __device__ Union7(const Union7 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(std::move(x.case0)); break; // Call
            case 1: new (&this->case1) Union7_1(std::move(x.case1)); break; // Fold
            case 2: new (&this->case2) Union7_2(std::move(x.case2)); break; // Raise
        }
    }
    __host__ __device__ Union7 & operator=(const Union7 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Call
                case 1: this->case1 = x.case1; break; // Fold
                case 2: this->case2 = x.case2; break; // Raise
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
                case 0: this->case0 = std::move(x.case0); break; // Call
                case 1: this->case1 = std::move(x.case1); break; // Fold
                case 2: this->case2 = std::move(x.case2); break; // Raise
            }
        } else {
            this->~Union7();
            new (this) Union7{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union7() {
        switch(this->tag){
            case 0: this->case0.~Union7_0(); break; // Call
            case 1: this->case1.~Union7_1(); break; // Fold
            case 2: this->case2.~Union7_2(); break; // Raise
        }
        this->tag = 255;
    }
};
struct Tuple13 {
    static_array<float,3> v0;
    static_array<float,3> v1;
    __host__ __device__ Tuple13() = default;
    __host__ __device__ Tuple13(static_array<float,3> t0, static_array<float,3> t1) : v0(t0), v1(t1) {}
};
struct StackRefs14 {
    std::unordered_map<Tuple4, Tuple13, Fun8, Fun12> & v0;
    __host__ __device__ StackRefs14() = default;
    __host__ __device__ StackRefs14(std::unordered_map<Tuple4, Tuple13, Fun8, Fun12> & t0) : v0(t0) {}
};
struct Tuple15 {
    float v0;
    float v1;
    __host__ __device__ Tuple15() = default;
    __host__ __device__ Tuple15(float t0, float t1) : v0(t0), v1(t1) {}
};
struct StackRefs16 {
    std::unordered_map<Tuple4, static_array<Tuple15,3>, Fun8, Fun12> & v0;
    __host__ __device__ StackRefs16() = default;
    __host__ __device__ StackRefs16(std::unordered_map<Tuple4, static_array<Tuple15,3>, Fun8, Fun12> & t0) : v0(t0) {}
};
struct StackMut18 {
    unsigned int v0;
    __host__ __device__ StackMut18() = default;
    __host__ __device__ StackMut18(unsigned int t0) : v0(t0) {}
};
struct StackRefs19 {
    static_array_list<Union5,32> & v0;
    __host__ __device__ StackRefs19() = default;
    __host__ __device__ StackRefs19(static_array_list<Union5,32> & t0) : v0(t0) {}
};
struct StackRefs20 {
    static_array<Tuple15,2> & v0;
    __host__ __device__ StackRefs20() = default;
    __host__ __device__ StackRefs20(static_array<Tuple15,2> & t0) : v0(t0) {}
};
struct Tuple21 {
    int v0;
    float v1;
    __host__ __device__ Tuple21() = default;
    __host__ __device__ Tuple21(int t0, float t1) : v0(t0), v1(t1) {}
};
struct Union23_0 { // ChanceCommunityCard
    Union24 v0;
    static_array<Union6,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union23_0(Union24 t0, bool t1, static_array<Union6,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __host__ __device__ Union23_0() = delete;
};
struct Union23_1 { // ChanceInit
};
struct Union23_2 { // Round
    Union24 v0;
    static_array<Union6,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union23_2(Union24 t0, bool t1, static_array<Union6,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __host__ __device__ Union23_2() = delete;
};
struct Union23_3 { // RoundWithAction
    Union24 v0;
    static_array<Union6,2> v2;
    static_array<int,2> v4;
    Union7 v6;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union23_3(Union24 t0, bool t1, static_array<Union6,2> t2, int t3, static_array<int,2> t4, int t5, Union7 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __host__ __device__ Union23_3() = delete;
};
struct Union23_4 { // TerminalCall
    Union24 v0;
    static_array<Union6,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union23_4(Union24 t0, bool t1, static_array<Union6,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __host__ __device__ Union23_4() = delete;
};
struct Union23_5 { // TerminalFold
    Union24 v0;
    static_array<Union6,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union23_5(Union24 t0, bool t1, static_array<Union6,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __host__ __device__ Union23_5() = delete;
};
struct Union23 {
    union {
        Union23_0 case0; // ChanceCommunityCard
        Union23_1 case1; // ChanceInit
        Union23_2 case2; // Round
        Union23_3 case3; // RoundWithAction
        Union23_4 case4; // TerminalCall
        Union23_5 case5; // TerminalFold
    };
    unsigned char tag{255};
    __host__ __device__ Union23() {}
    __host__ __device__ Union23(Union23_0 t) : tag(0), case0(t) {} // ChanceCommunityCard
    __host__ __device__ Union23(Union23_1 t) : tag(1), case1(t) {} // ChanceInit
    __host__ __device__ Union23(Union23_2 t) : tag(2), case2(t) {} // Round
    __host__ __device__ Union23(Union23_3 t) : tag(3), case3(t) {} // RoundWithAction
    __host__ __device__ Union23(Union23_4 t) : tag(4), case4(t) {} // TerminalCall
    __host__ __device__ Union23(Union23_5 t) : tag(5), case5(t) {} // TerminalFold
    __host__ __device__ Union23(const Union23 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union23_0(x.case0); break; // ChanceCommunityCard
            case 1: new (&this->case1) Union23_1(x.case1); break; // ChanceInit
            case 2: new (&this->case2) Union23_2(x.case2); break; // Round
            case 3: new (&this->case3) Union23_3(x.case3); break; // RoundWithAction
            case 4: new (&this->case4) Union23_4(x.case4); break; // TerminalCall
            case 5: new (&this->case5) Union23_5(x.case5); break; // TerminalFold
        }
    }
    __host__ __device__ Union23(const Union23 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union23_0(std::move(x.case0)); break; // ChanceCommunityCard
            case 1: new (&this->case1) Union23_1(std::move(x.case1)); break; // ChanceInit
            case 2: new (&this->case2) Union23_2(std::move(x.case2)); break; // Round
            case 3: new (&this->case3) Union23_3(std::move(x.case3)); break; // RoundWithAction
            case 4: new (&this->case4) Union23_4(std::move(x.case4)); break; // TerminalCall
            case 5: new (&this->case5) Union23_5(std::move(x.case5)); break; // TerminalFold
        }
    }
    __host__ __device__ Union23 & operator=(const Union23 & x) {
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
            this->~Union23();
            new (this) Union23{x};
        }
        return *this;
    }
    __host__ __device__ Union23 & operator=(const Union23 && x) {
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
            this->~Union23();
            new (this) Union23{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union23() {
        switch(this->tag){
            case 0: this->case0.~Union23_0(); break; // ChanceCommunityCard
            case 1: this->case1.~Union23_1(); break; // ChanceInit
            case 2: this->case2.~Union23_2(); break; // Round
            case 3: this->case3.~Union23_3(); break; // RoundWithAction
            case 4: this->case4.~Union23_4(); break; // TerminalCall
            case 5: this->case5.~Union23_5(); break; // TerminalFold
        }
        this->tag = 255;
    }
};
struct Union24_0 { // None
};
struct Union24_1 { // Some
    Union6 v0;
    __host__ __device__ Union24_1(Union6 t0) : v0(t0) {}
    __host__ __device__ Union24_1() = delete;
};
struct Union24 {
    union {
        Union24_0 case0; // None
        Union24_1 case1; // Some
    };
    unsigned char tag{255};
    __host__ __device__ Union24() {}
    __host__ __device__ Union24(Union24_0 t) : tag(0), case0(t) {} // None
    __host__ __device__ Union24(Union24_1 t) : tag(1), case1(t) {} // Some
    __host__ __device__ Union24(const Union24 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union24_0(x.case0); break; // None
            case 1: new (&this->case1) Union24_1(x.case1); break; // Some
        }
    }
    __host__ __device__ Union24(const Union24 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union24_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union24_1(std::move(x.case1)); break; // Some
        }
    }
    __host__ __device__ Union24 & operator=(const Union24 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union24();
            new (this) Union24{x};
        }
        return *this;
    }
    __host__ __device__ Union24 & operator=(const Union24 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union24();
            new (this) Union24{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union24() {
        switch(this->tag){
            case 0: this->case0.~Union24_0(); break; // None
            case 1: this->case1.~Union24_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct StackMut26 {
    float v0;
    __host__ __device__ StackMut26() = default;
    __host__ __device__ StackMut26(float t0) : v0(t0) {}
};
struct Tuple28 {
    Union6 v0;
    unsigned int v1;
    __host__ __device__ Tuple28() = default;
    __host__ __device__ Tuple28(Union6 t0, unsigned int t1) : v0(t0), v1(t1) {}
};
struct StackMut31 {
    int v0;
    __host__ __device__ StackMut31() = default;
    __host__ __device__ StackMut31(int t0) : v0(t0) {}
};
struct StackMut32 {
    unsigned int v0;
    __host__ __device__ StackMut32() = default;
    __host__ __device__ StackMut32(unsigned int t0) : v0(t0) {}
};
struct Union34_0 { // T_game_chance_community_card
    Union24 v0;
    static_array<Union6,2> v2;
    static_array<int,2> v4;
    Union6 v6;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union34_0(Union24 t0, bool t1, static_array<Union6,2> t2, int t3, static_array<int,2> t4, int t5, Union6 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __host__ __device__ Union34_0() = delete;
};
struct Union34_1 { // T_game_chance_init
    Union6 v0;
    Union6 v1;
    __host__ __device__ Union34_1(Union6 t0, Union6 t1) : v0(t0), v1(t1) {}
    __host__ __device__ Union34_1() = delete;
};
struct Union34_2 { // T_game_round
    Union24 v0;
    static_array<Union6,2> v2;
    static_array<int,2> v4;
    Union7 v6;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union34_2(Union24 t0, bool t1, static_array<Union6,2> t2, int t3, static_array<int,2> t4, int t5, Union7 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __host__ __device__ Union34_2() = delete;
};
struct Union34_3 { // T_none
};
struct Union34 {
    union {
        Union34_0 case0; // T_game_chance_community_card
        Union34_1 case1; // T_game_chance_init
        Union34_2 case2; // T_game_round
        Union34_3 case3; // T_none
    };
    unsigned char tag{255};
    __host__ __device__ Union34() {}
    __host__ __device__ Union34(Union34_0 t) : tag(0), case0(t) {} // T_game_chance_community_card
    __host__ __device__ Union34(Union34_1 t) : tag(1), case1(t) {} // T_game_chance_init
    __host__ __device__ Union34(Union34_2 t) : tag(2), case2(t) {} // T_game_round
    __host__ __device__ Union34(Union34_3 t) : tag(3), case3(t) {} // T_none
    __host__ __device__ Union34(const Union34 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union34_0(x.case0); break; // T_game_chance_community_card
            case 1: new (&this->case1) Union34_1(x.case1); break; // T_game_chance_init
            case 2: new (&this->case2) Union34_2(x.case2); break; // T_game_round
            case 3: new (&this->case3) Union34_3(x.case3); break; // T_none
        }
    }
    __host__ __device__ Union34(const Union34 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union34_0(std::move(x.case0)); break; // T_game_chance_community_card
            case 1: new (&this->case1) Union34_1(std::move(x.case1)); break; // T_game_chance_init
            case 2: new (&this->case2) Union34_2(std::move(x.case2)); break; // T_game_round
            case 3: new (&this->case3) Union34_3(std::move(x.case3)); break; // T_none
        }
    }
    __host__ __device__ Union34 & operator=(const Union34 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // T_game_chance_community_card
                case 1: this->case1 = x.case1; break; // T_game_chance_init
                case 2: this->case2 = x.case2; break; // T_game_round
                case 3: this->case3 = x.case3; break; // T_none
            }
        } else {
            this->~Union34();
            new (this) Union34{x};
        }
        return *this;
    }
    __host__ __device__ Union34 & operator=(const Union34 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // T_game_chance_community_card
                case 1: this->case1 = std::move(x.case1); break; // T_game_chance_init
                case 2: this->case2 = std::move(x.case2); break; // T_game_round
                case 3: this->case3 = std::move(x.case3); break; // T_none
            }
        } else {
            this->~Union34();
            new (this) Union34{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union34() {
        switch(this->tag){
            case 0: this->case0.~Union34_0(); break; // T_game_chance_community_card
            case 1: this->case1.~Union34_1(); break; // T_game_chance_init
            case 2: this->case2.~Union34_2(); break; // T_game_round
            case 3: this->case3.~Union34_3(); break; // T_none
        }
        this->tag = 255;
    }
};
struct Tuple36 {
    int v0;
    int v1;
    __host__ __device__ Tuple36() = default;
    __host__ __device__ Tuple36(int t0, int t1) : v0(t0), v1(t1) {}
};
struct Tuple38 {
    unsigned long long v1;
    unsigned long long v2;
    int v0;
    __host__ __device__ Tuple38() = default;
    __host__ __device__ Tuple38(int t0, unsigned long long t1, unsigned long long t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Union39_0 { // None
};
struct Union39_1 { // Some
    static_array<float,3> v0;
    static_array<float,3> v1;
    __host__ __device__ Union39_1(static_array<float,3> t0, static_array<float,3> t1) : v0(t0), v1(t1) {}
    __host__ __device__ Union39_1() = delete;
};
struct Union39 {
    union {
        Union39_0 case0; // None
        Union39_1 case1; // Some
    };
    unsigned char tag{255};
    __host__ __device__ Union39() {}
    __host__ __device__ Union39(Union39_0 t) : tag(0), case0(t) {} // None
    __host__ __device__ Union39(Union39_1 t) : tag(1), case1(t) {} // Some
    __host__ __device__ Union39(const Union39 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union39_0(x.case0); break; // None
            case 1: new (&this->case1) Union39_1(x.case1); break; // Some
        }
    }
    __host__ __device__ Union39(const Union39 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union39_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union39_1(std::move(x.case1)); break; // Some
        }
    }
    __host__ __device__ Union39 & operator=(const Union39 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union39();
            new (this) Union39{x};
        }
        return *this;
    }
    __host__ __device__ Union39 & operator=(const Union39 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union39();
            new (this) Union39{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union39() {
        switch(this->tag){
            case 0: this->case0.~Union39_0(); break; // None
            case 1: this->case1.~Union39_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union41_0 { // None
};
struct Union41_1 { // Some
    Union7 v0;
    __host__ __device__ Union41_1(Union7 t0) : v0(t0) {}
    __host__ __device__ Union41_1() = delete;
};
struct Union41 {
    union {
        Union41_0 case0; // None
        Union41_1 case1; // Some
    };
    unsigned char tag{255};
    __host__ __device__ Union41() {}
    __host__ __device__ Union41(Union41_0 t) : tag(0), case0(t) {} // None
    __host__ __device__ Union41(Union41_1 t) : tag(1), case1(t) {} // Some
    __host__ __device__ Union41(const Union41 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union41_0(x.case0); break; // None
            case 1: new (&this->case1) Union41_1(x.case1); break; // Some
        }
    }
    __host__ __device__ Union41(const Union41 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union41_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union41_1(std::move(x.case1)); break; // Some
        }
    }
    __host__ __device__ Union41 & operator=(const Union41 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union41();
            new (this) Union41{x};
        }
        return *this;
    }
    __host__ __device__ Union41 & operator=(const Union41 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union41();
            new (this) Union41{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union41() {
        switch(this->tag){
            case 0: this->case0.~Union41_0(); break; // None
            case 1: this->case1.~Union41_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union42_0 { // None
};
struct Union42_1 { // Some
    static_array<Tuple15,3> v0;
    __host__ __device__ Union42_1(static_array<Tuple15,3> t0) : v0(t0) {}
    __host__ __device__ Union42_1() = delete;
};
struct Union42 {
    union {
        Union42_0 case0; // None
        Union42_1 case1; // Some
    };
    unsigned char tag{255};
    __host__ __device__ Union42() {}
    __host__ __device__ Union42(Union42_0 t) : tag(0), case0(t) {} // None
    __host__ __device__ Union42(Union42_1 t) : tag(1), case1(t) {} // Some
    __host__ __device__ Union42(const Union42 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union42_0(x.case0); break; // None
            case 1: new (&this->case1) Union42_1(x.case1); break; // Some
        }
    }
    __host__ __device__ Union42(const Union42 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union42_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union42_1(std::move(x.case1)); break; // Some
        }
    }
    __host__ __device__ Union42 & operator=(const Union42 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union42();
            new (this) Union42{x};
        }
        return *this;
    }
    __host__ __device__ Union42 & operator=(const Union42 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union42();
            new (this) Union42{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union42() {
        switch(this->tag){
            case 0: this->case0.~Union42_0(); break; // None
            case 1: this->case1.~Union42_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union51_0 { // Eq
};
struct Union51_1 { // Gt
};
struct Union51_2 { // Lt
};
struct Union51 {
    union {
        Union51_0 case0; // Eq
        Union51_1 case1; // Gt
        Union51_2 case2; // Lt
    };
    unsigned char tag{255};
    __host__ __device__ Union51() {}
    __host__ __device__ Union51(Union51_0 t) : tag(0), case0(t) {} // Eq
    __host__ __device__ Union51(Union51_1 t) : tag(1), case1(t) {} // Gt
    __host__ __device__ Union51(Union51_2 t) : tag(2), case2(t) {} // Lt
    __host__ __device__ Union51(const Union51 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union51_0(x.case0); break; // Eq
            case 1: new (&this->case1) Union51_1(x.case1); break; // Gt
            case 2: new (&this->case2) Union51_2(x.case2); break; // Lt
        }
    }
    __host__ __device__ Union51(const Union51 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union51_0(std::move(x.case0)); break; // Eq
            case 1: new (&this->case1) Union51_1(std::move(x.case1)); break; // Gt
            case 2: new (&this->case2) Union51_2(std::move(x.case2)); break; // Lt
        }
    }
    __host__ __device__ Union51 & operator=(const Union51 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Eq
                case 1: this->case1 = x.case1; break; // Gt
                case 2: this->case2 = x.case2; break; // Lt
            }
        } else {
            this->~Union51();
            new (this) Union51{x};
        }
        return *this;
    }
    __host__ __device__ Union51 & operator=(const Union51 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // Eq
                case 1: this->case1 = std::move(x.case1); break; // Gt
                case 2: this->case2 = std::move(x.case2); break; // Lt
            }
        } else {
            this->~Union51();
            new (this) Union51{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union51() {
        switch(this->tag){
            case 0: this->case0.~Union51_0(); break; // Eq
            case 1: this->case1.~Union51_1(); break; // Gt
            case 2: this->case2.~Union51_2(); break; // Lt
        }
        this->tag = 255;
    }
};
struct Tuple58 {
    int v0;
    float v1;
    float v2;
    __host__ __device__ Tuple58() = default;
    __host__ __device__ Tuple58(int t0, float t1, float t2) : v0(t0), v1(t1), v2(t2) {}
};
