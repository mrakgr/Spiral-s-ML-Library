kernel = r"""
using default_int = long;
using default_uint = unsigned long;
#include "reference_counting.cuh"
#include <assert.h>
#include <stdio.h>
#include <curand_kernel.h>
struct Union1;
struct Union2;
struct Union0;
__device__ long f_1(unsigned char * v0);
__device__ void f_3(unsigned char * v0);
__device__ Union1 f_2(unsigned char * v0);
__device__ Union2 f_5(unsigned char * v0);
__device__ static_array<Union2,2l> f_4(unsigned char * v0);
__device__ Union0 f_0(unsigned char * v0);
struct Union3;
struct Union4;
struct Union7;
struct Union6;
struct Union5;
struct Union8;
__device__ Union3 f_7(unsigned char * v0);
__device__ long f_8(unsigned char * v0);
__device__ long f_11(unsigned char * v0);
__device__ Tuple1 f_10(unsigned char * v0);
__device__ Tuple2 f_12(unsigned char * v0);
__device__ Tuple3 f_13(unsigned char * v0);
__device__ Union4 f_9(unsigned char * v0);
__device__ long f_14(unsigned char * v0);
__device__ Tuple4 f_16(unsigned char * v0);
__device__ long f_18(unsigned char * v0);
__device__ Tuple5 f_17(unsigned char * v0);
__device__ Union6 f_15(unsigned char * v0);
__device__ long f_19(unsigned char * v0);
__device__ Tuple0 f_6(unsigned char * v0);
__device__ unsigned long loop_22(unsigned long v0, curandStatePhilox4_32_10_t & v1);
struct Union9;
__device__ long tag_24(Union3 v0);
__device__ bool is_pair_25(long v0, long v1);
__device__ Tuple8 order_26(long v0, long v1);
__device__ Union9 compare_hands_23(Union7 v0, bool v1, static_array<Union3,2l> v2, long v3, static_array<long,2l> v4, long v5);
__device__ Union6 play_loop_inner_21(static_array_list<Union3,6l> & v0, static_array_list<Union4,32l> & v1, static_array<Union2,2l> v2, Union6 v3);
__device__ Tuple6 play_loop_20(Union5 v0, static_array<Union2,2l> v1, Union8 v2, static_array_list<Union3,6l> & v3, static_array_list<Union4,32l> & v4, Union6 v5);
__device__ void f_28(unsigned char * v0, long v1);
__device__ void f_30(unsigned char * v0);
__device__ void f_29(unsigned char * v0, Union3 v1);
__device__ void f_31(unsigned char * v0, long v1);
__device__ void f_34(unsigned char * v0, long v1);
__device__ void f_33(unsigned char * v0, long v1, Union1 v2);
__device__ void f_35(unsigned char * v0, long v1, Union3 v2);
__device__ void f_36(unsigned char * v0, static_array<Union3,2l> v1, long v2, long v3);
__device__ void f_32(unsigned char * v0, Union4 v1);
__device__ void f_37(unsigned char * v0, long v1);
__device__ void f_39(unsigned char * v0, Union7 v1, bool v2, static_array<Union3,2l> v3, long v4, static_array<long,2l> v5, long v6);
__device__ void f_41(unsigned char * v0, long v1);
__device__ void f_40(unsigned char * v0, Union7 v1, bool v2, static_array<Union3,2l> v3, long v4, static_array<long,2l> v5, long v6, Union1 v7);
__device__ void f_38(unsigned char * v0, Union6 v1);
__device__ void f_42(unsigned char * v0, Union2 v1);
__device__ void f_43(unsigned char * v0, long v1);
__device__ void f_27(unsigned char * v0, static_array_list<Union3,6l> v1, static_array_list<Union4,32l> v2, Union5 v3, static_array<Union2,2l> v4, Union8 v5);
__device__ void f_45(unsigned char * v0, long v1);
__device__ void f_44(unsigned char * v0, static_array_list<Union4,32l> v1, static_array<Union2,2l> v2, Union8 v3);
struct Union1_0 { // Call
};
struct Union1_1 { // Fold
};
struct Union1_2 { // Raise
};
struct Union1 {
    union {
        Union1_0 case0; // Call
        Union1_1 case1; // Fold
        Union1_2 case2; // Raise
    };
    __device__ Union1(Union1_0 t) : tag(0), case0(t) {} // Call
    __device__ Union1(Union1_1 t) : tag(1), case1(t) {} // Fold
    __device__ Union1(Union1_2 t) : tag(2), case2(t) {} // Raise
    __device__ Union1() = delete;
    __device__ Union1(Union1 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(x.case0); break; // Call
            case 1: new (&this->case1) Union1_1(x.case1); break; // Fold
            case 2: new (&this->case2) Union1_2(x.case2); break; // Raise
        }
    }
    __device__ Union1(Union1 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(std::move(x.case0)); break; // Call
            case 1: new (&this->case1) Union1_1(std::move(x.case1)); break; // Fold
            case 2: new (&this->case2) Union1_2(std::move(x.case2)); break; // Raise
        }
    }
    __device__ Union1 & operator=(Union1 & x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = x.case0; break; // Call
            case 1: this->case1 = x.case1; break; // Fold
            case 2: this->case2 = x.case2; break; // Raise
        }
        return *this;
    }
    __device__ Union1 & operator=(Union1 && x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = std::move(x.case0); break; // Call
            case 1: this->case1 = std::move(x.case1); break; // Fold
            case 2: this->case2 = std::move(x.case2); break; // Raise
        }
        return *this;
    }
    __device__ ~Union1() {
        switch(this->tag){
            case 0: this->case0.~Union1_0(); break; // Call
            case 1: this->case1.~Union1_1(); break; // Fold
            case 2: this->case2.~Union1_2(); break; // Raise
        }
    }
    unsigned char tag;
};
struct Union2_0 { // Computer
};
struct Union2_1 { // Human
};
struct Union2 {
    union {
        Union2_0 case0; // Computer
        Union2_1 case1; // Human
    };
    __device__ Union2(Union2_0 t) : tag(0), case0(t) {} // Computer
    __device__ Union2(Union2_1 t) : tag(1), case1(t) {} // Human
    __device__ Union2() = delete;
    __device__ Union2(Union2 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(x.case0); break; // Computer
            case 1: new (&this->case1) Union2_1(x.case1); break; // Human
        }
    }
    __device__ Union2(Union2 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(std::move(x.case0)); break; // Computer
            case 1: new (&this->case1) Union2_1(std::move(x.case1)); break; // Human
        }
    }
    __device__ Union2 & operator=(Union2 & x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = x.case0; break; // Computer
            case 1: this->case1 = x.case1; break; // Human
        }
        return *this;
    }
    __device__ Union2 & operator=(Union2 && x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = std::move(x.case0); break; // Computer
            case 1: this->case1 = std::move(x.case1); break; // Human
        }
        return *this;
    }
    __device__ ~Union2() {
        switch(this->tag){
            case 0: this->case0.~Union2_0(); break; // Computer
            case 1: this->case1.~Union2_1(); break; // Human
        }
    }
    unsigned char tag;
};
struct Union0_0 { // ActionSelected
    Union1 v0;
    __device__ Union0_0(Union1 t0) : v0(t0) {}
    __device__ Union0_0() = delete;
};
struct Union0_1 { // PlayerChanged
    static_array<Union2,2l> v0;
    __device__ Union0_1(static_array<Union2,2l> t0) : v0(t0) {}
    __device__ Union0_1() = delete;
};
struct Union0_2 { // StartGame
};
struct Union0 {
    union {
        Union0_0 case0; // ActionSelected
        Union0_1 case1; // PlayerChanged
        Union0_2 case2; // StartGame
    };
    __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // ActionSelected
    __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // PlayerChanged
    __device__ Union0(Union0_2 t) : tag(2), case2(t) {} // StartGame
    __device__ Union0() = delete;
    __device__ Union0(Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // ActionSelected
            case 1: new (&this->case1) Union0_1(x.case1); break; // PlayerChanged
            case 2: new (&this->case2) Union0_2(x.case2); break; // StartGame
        }
    }
    __device__ Union0(Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // ActionSelected
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // PlayerChanged
            case 2: new (&this->case2) Union0_2(std::move(x.case2)); break; // StartGame
        }
    }
    __device__ Union0 & operator=(Union0 & x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = x.case0; break; // ActionSelected
            case 1: this->case1 = x.case1; break; // PlayerChanged
            case 2: this->case2 = x.case2; break; // StartGame
        }
        return *this;
    }
    __device__ Union0 & operator=(Union0 && x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = std::move(x.case0); break; // ActionSelected
            case 1: this->case1 = std::move(x.case1); break; // PlayerChanged
            case 2: this->case2 = std::move(x.case2); break; // StartGame
        }
        return *this;
    }
    __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // ActionSelected
            case 1: this->case1.~Union0_1(); break; // PlayerChanged
            case 2: this->case2.~Union0_2(); break; // StartGame
        }
    }
    unsigned char tag;
};
struct Union3_0 { // Jack
};
struct Union3_1 { // King
};
struct Union3_2 { // Queen
};
struct Union3 {
    union {
        Union3_0 case0; // Jack
        Union3_1 case1; // King
        Union3_2 case2; // Queen
    };
    __device__ Union3(Union3_0 t) : tag(0), case0(t) {} // Jack
    __device__ Union3(Union3_1 t) : tag(1), case1(t) {} // King
    __device__ Union3(Union3_2 t) : tag(2), case2(t) {} // Queen
    __device__ Union3() = delete;
    __device__ Union3(Union3 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(x.case0); break; // Jack
            case 1: new (&this->case1) Union3_1(x.case1); break; // King
            case 2: new (&this->case2) Union3_2(x.case2); break; // Queen
        }
    }
    __device__ Union3(Union3 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(std::move(x.case0)); break; // Jack
            case 1: new (&this->case1) Union3_1(std::move(x.case1)); break; // King
            case 2: new (&this->case2) Union3_2(std::move(x.case2)); break; // Queen
        }
    }
    __device__ Union3 & operator=(Union3 & x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = x.case0; break; // Jack
            case 1: this->case1 = x.case1; break; // King
            case 2: this->case2 = x.case2; break; // Queen
        }
        return *this;
    }
    __device__ Union3 & operator=(Union3 && x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = std::move(x.case0); break; // Jack
            case 1: this->case1 = std::move(x.case1); break; // King
            case 2: this->case2 = std::move(x.case2); break; // Queen
        }
        return *this;
    }
    __device__ ~Union3() {
        switch(this->tag){
            case 0: this->case0.~Union3_0(); break; // Jack
            case 1: this->case1.~Union3_1(); break; // King
            case 2: this->case2.~Union3_2(); break; // Queen
        }
    }
    unsigned char tag;
};
struct Union4_0 { // CommunityCardIs
    Union3 v0;
    __device__ Union4_0(Union3 t0) : v0(t0) {}
    __device__ Union4_0() = delete;
};
struct Union4_1 { // PlayerAction
    Union1 v1;
    long v0;
    __device__ Union4_1(long t0, Union1 t1) : v0(t0), v1(t1) {}
    __device__ Union4_1() = delete;
};
struct Union4_2 { // PlayerGotCard
    Union3 v1;
    long v0;
    __device__ Union4_2(long t0, Union3 t1) : v0(t0), v1(t1) {}
    __device__ Union4_2() = delete;
};
struct Union4_3 { // Showdown
    static_array<Union3,2l> v0;
    long v1;
    long v2;
    __device__ Union4_3(static_array<Union3,2l> t0, long t1, long t2) : v0(t0), v1(t1), v2(t2) {}
    __device__ Union4_3() = delete;
};
struct Union4 {
    union {
        Union4_0 case0; // CommunityCardIs
        Union4_1 case1; // PlayerAction
        Union4_2 case2; // PlayerGotCard
        Union4_3 case3; // Showdown
    };
    __device__ Union4(Union4_0 t) : tag(0), case0(t) {} // CommunityCardIs
    __device__ Union4(Union4_1 t) : tag(1), case1(t) {} // PlayerAction
    __device__ Union4(Union4_2 t) : tag(2), case2(t) {} // PlayerGotCard
    __device__ Union4(Union4_3 t) : tag(3), case3(t) {} // Showdown
    __device__ Union4() = delete;
    __device__ Union4(Union4 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(x.case0); break; // CommunityCardIs
            case 1: new (&this->case1) Union4_1(x.case1); break; // PlayerAction
            case 2: new (&this->case2) Union4_2(x.case2); break; // PlayerGotCard
            case 3: new (&this->case3) Union4_3(x.case3); break; // Showdown
        }
    }
    __device__ Union4(Union4 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(std::move(x.case0)); break; // CommunityCardIs
            case 1: new (&this->case1) Union4_1(std::move(x.case1)); break; // PlayerAction
            case 2: new (&this->case2) Union4_2(std::move(x.case2)); break; // PlayerGotCard
            case 3: new (&this->case3) Union4_3(std::move(x.case3)); break; // Showdown
        }
    }
    __device__ Union4 & operator=(Union4 & x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = x.case0; break; // CommunityCardIs
            case 1: this->case1 = x.case1; break; // PlayerAction
            case 2: this->case2 = x.case2; break; // PlayerGotCard
            case 3: this->case3 = x.case3; break; // Showdown
        }
        return *this;
    }
    __device__ Union4 & operator=(Union4 && x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = std::move(x.case0); break; // CommunityCardIs
            case 1: this->case1 = std::move(x.case1); break; // PlayerAction
            case 2: this->case2 = std::move(x.case2); break; // PlayerGotCard
            case 3: this->case3 = std::move(x.case3); break; // Showdown
        }
        return *this;
    }
    __device__ ~Union4() {
        switch(this->tag){
            case 0: this->case0.~Union4_0(); break; // CommunityCardIs
            case 1: this->case1.~Union4_1(); break; // PlayerAction
            case 2: this->case2.~Union4_2(); break; // PlayerGotCard
            case 3: this->case3.~Union4_3(); break; // Showdown
        }
    }
    unsigned char tag;
};
struct Union7_0 { // None
};
struct Union7_1 { // Some
    Union3 v0;
    __device__ Union7_1(Union3 t0) : v0(t0) {}
    __device__ Union7_1() = delete;
};
struct Union7 {
    union {
        Union7_0 case0; // None
        Union7_1 case1; // Some
    };
    __device__ Union7(Union7_0 t) : tag(0), case0(t) {} // None
    __device__ Union7(Union7_1 t) : tag(1), case1(t) {} // Some
    __device__ Union7() = delete;
    __device__ Union7(Union7 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(x.case0); break; // None
            case 1: new (&this->case1) Union7_1(x.case1); break; // Some
        }
    }
    __device__ Union7(Union7 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union7_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union7 & operator=(Union7 & x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = x.case0; break; // None
            case 1: this->case1 = x.case1; break; // Some
        }
        return *this;
    }
    __device__ Union7 & operator=(Union7 && x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = std::move(x.case0); break; // None
            case 1: this->case1 = std::move(x.case1); break; // Some
        }
        return *this;
    }
    __device__ ~Union7() {
        switch(this->tag){
            case 0: this->case0.~Union7_0(); break; // None
            case 1: this->case1.~Union7_1(); break; // Some
        }
    }
    unsigned char tag;
};
struct Union6_0 { // ChanceCommunityCard
    Union7 v0;
    static_array<Union3,2l> v2;
    static_array<long,2l> v4;
    long v3;
    long v5;
    bool v1;
    __device__ Union6_0(Union7 t0, bool t1, static_array<Union3,2l> t2, long t3, static_array<long,2l> t4, long t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union6_0() = delete;
};
struct Union6_1 { // ChanceInit
};
struct Union6_2 { // Round
    Union7 v0;
    static_array<Union3,2l> v2;
    static_array<long,2l> v4;
    long v3;
    long v5;
    bool v1;
    __device__ Union6_2(Union7 t0, bool t1, static_array<Union3,2l> t2, long t3, static_array<long,2l> t4, long t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union6_2() = delete;
};
struct Union6_3 { // RoundWithAction
    Union7 v0;
    static_array<Union3,2l> v2;
    static_array<long,2l> v4;
    Union1 v6;
    long v3;
    long v5;
    bool v1;
    __device__ Union6_3(Union7 t0, bool t1, static_array<Union3,2l> t2, long t3, static_array<long,2l> t4, long t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __device__ Union6_3() = delete;
};
struct Union6_4 { // TerminalCall
    Union7 v0;
    static_array<Union3,2l> v2;
    static_array<long,2l> v4;
    long v3;
    long v5;
    bool v1;
    __device__ Union6_4(Union7 t0, bool t1, static_array<Union3,2l> t2, long t3, static_array<long,2l> t4, long t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union6_4() = delete;
};
struct Union6_5 { // TerminalFold
    Union7 v0;
    static_array<Union3,2l> v2;
    static_array<long,2l> v4;
    long v3;
    long v5;
    bool v1;
    __device__ Union6_5(Union7 t0, bool t1, static_array<Union3,2l> t2, long t3, static_array<long,2l> t4, long t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union6_5() = delete;
};
struct Union6 {
    union {
        Union6_0 case0; // ChanceCommunityCard
        Union6_1 case1; // ChanceInit
        Union6_2 case2; // Round
        Union6_3 case3; // RoundWithAction
        Union6_4 case4; // TerminalCall
        Union6_5 case5; // TerminalFold
    };
    __device__ Union6(Union6_0 t) : tag(0), case0(t) {} // ChanceCommunityCard
    __device__ Union6(Union6_1 t) : tag(1), case1(t) {} // ChanceInit
    __device__ Union6(Union6_2 t) : tag(2), case2(t) {} // Round
    __device__ Union6(Union6_3 t) : tag(3), case3(t) {} // RoundWithAction
    __device__ Union6(Union6_4 t) : tag(4), case4(t) {} // TerminalCall
    __device__ Union6(Union6_5 t) : tag(5), case5(t) {} // TerminalFold
    __device__ Union6() = delete;
    __device__ Union6(Union6 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(x.case0); break; // ChanceCommunityCard
            case 1: new (&this->case1) Union6_1(x.case1); break; // ChanceInit
            case 2: new (&this->case2) Union6_2(x.case2); break; // Round
            case 3: new (&this->case3) Union6_3(x.case3); break; // RoundWithAction
            case 4: new (&this->case4) Union6_4(x.case4); break; // TerminalCall
            case 5: new (&this->case5) Union6_5(x.case5); break; // TerminalFold
        }
    }
    __device__ Union6(Union6 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(std::move(x.case0)); break; // ChanceCommunityCard
            case 1: new (&this->case1) Union6_1(std::move(x.case1)); break; // ChanceInit
            case 2: new (&this->case2) Union6_2(std::move(x.case2)); break; // Round
            case 3: new (&this->case3) Union6_3(std::move(x.case3)); break; // RoundWithAction
            case 4: new (&this->case4) Union6_4(std::move(x.case4)); break; // TerminalCall
            case 5: new (&this->case5) Union6_5(std::move(x.case5)); break; // TerminalFold
        }
    }
    __device__ Union6 & operator=(Union6 & x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = x.case0; break; // ChanceCommunityCard
            case 1: this->case1 = x.case1; break; // ChanceInit
            case 2: this->case2 = x.case2; break; // Round
            case 3: this->case3 = x.case3; break; // RoundWithAction
            case 4: this->case4 = x.case4; break; // TerminalCall
            case 5: this->case5 = x.case5; break; // TerminalFold
        }
        return *this;
    }
    __device__ Union6 & operator=(Union6 && x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = std::move(x.case0); break; // ChanceCommunityCard
            case 1: this->case1 = std::move(x.case1); break; // ChanceInit
            case 2: this->case2 = std::move(x.case2); break; // Round
            case 3: this->case3 = std::move(x.case3); break; // RoundWithAction
            case 4: this->case4 = std::move(x.case4); break; // TerminalCall
            case 5: this->case5 = std::move(x.case5); break; // TerminalFold
        }
        return *this;
    }
    __device__ ~Union6() {
        switch(this->tag){
            case 0: this->case0.~Union6_0(); break; // ChanceCommunityCard
            case 1: this->case1.~Union6_1(); break; // ChanceInit
            case 2: this->case2.~Union6_2(); break; // Round
            case 3: this->case3.~Union6_3(); break; // RoundWithAction
            case 4: this->case4.~Union6_4(); break; // TerminalCall
            case 5: this->case5.~Union6_5(); break; // TerminalFold
        }
    }
    unsigned char tag;
};
struct Union5_0 { // None
};
struct Union5_1 { // Some
    Union6 v0;
    __device__ Union5_1(Union6 t0) : v0(t0) {}
    __device__ Union5_1() = delete;
};
struct Union5 {
    union {
        Union5_0 case0; // None
        Union5_1 case1; // Some
    };
    __device__ Union5(Union5_0 t) : tag(0), case0(t) {} // None
    __device__ Union5(Union5_1 t) : tag(1), case1(t) {} // Some
    __device__ Union5() = delete;
    __device__ Union5(Union5 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union5_0(x.case0); break; // None
            case 1: new (&this->case1) Union5_1(x.case1); break; // Some
        }
    }
    __device__ Union5(Union5 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union5_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union5_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union5 & operator=(Union5 & x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = x.case0; break; // None
            case 1: this->case1 = x.case1; break; // Some
        }
        return *this;
    }
    __device__ Union5 & operator=(Union5 && x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = std::move(x.case0); break; // None
            case 1: this->case1 = std::move(x.case1); break; // Some
        }
        return *this;
    }
    __device__ ~Union5() {
        switch(this->tag){
            case 0: this->case0.~Union5_0(); break; // None
            case 1: this->case1.~Union5_1(); break; // Some
        }
    }
    unsigned char tag;
};
struct Union8_0 { // GameNotStarted
};
struct Union8_1 { // GameOver
    Union7 v0;
    static_array<Union3,2l> v2;
    static_array<long,2l> v4;
    long v3;
    long v5;
    bool v1;
    __device__ Union8_1(Union7 t0, bool t1, static_array<Union3,2l> t2, long t3, static_array<long,2l> t4, long t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union8_1() = delete;
};
struct Union8_2 { // WaitingForActionFromPlayerId
    Union7 v0;
    static_array<Union3,2l> v2;
    static_array<long,2l> v4;
    long v3;
    long v5;
    bool v1;
    __device__ Union8_2(Union7 t0, bool t1, static_array<Union3,2l> t2, long t3, static_array<long,2l> t4, long t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union8_2() = delete;
};
struct Union8 {
    union {
        Union8_0 case0; // GameNotStarted
        Union8_1 case1; // GameOver
        Union8_2 case2; // WaitingForActionFromPlayerId
    };
    __device__ Union8(Union8_0 t) : tag(0), case0(t) {} // GameNotStarted
    __device__ Union8(Union8_1 t) : tag(1), case1(t) {} // GameOver
    __device__ Union8(Union8_2 t) : tag(2), case2(t) {} // WaitingForActionFromPlayerId
    __device__ Union8() = delete;
    __device__ Union8(Union8 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union8_0(x.case0); break; // GameNotStarted
            case 1: new (&this->case1) Union8_1(x.case1); break; // GameOver
            case 2: new (&this->case2) Union8_2(x.case2); break; // WaitingForActionFromPlayerId
        }
    }
    __device__ Union8(Union8 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union8_0(std::move(x.case0)); break; // GameNotStarted
            case 1: new (&this->case1) Union8_1(std::move(x.case1)); break; // GameOver
            case 2: new (&this->case2) Union8_2(std::move(x.case2)); break; // WaitingForActionFromPlayerId
        }
    }
    __device__ Union8 & operator=(Union8 & x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = x.case0; break; // GameNotStarted
            case 1: this->case1 = x.case1; break; // GameOver
            case 2: this->case2 = x.case2; break; // WaitingForActionFromPlayerId
        }
        return *this;
    }
    __device__ Union8 & operator=(Union8 && x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = std::move(x.case0); break; // GameNotStarted
            case 1: this->case1 = std::move(x.case1); break; // GameOver
            case 2: this->case2 = std::move(x.case2); break; // WaitingForActionFromPlayerId
        }
        return *this;
    }
    __device__ ~Union8() {
        switch(this->tag){
            case 0: this->case0.~Union8_0(); break; // GameNotStarted
            case 1: this->case1.~Union8_1(); break; // GameOver
            case 2: this->case2.~Union8_2(); break; // WaitingForActionFromPlayerId
        }
    }
    unsigned char tag;
};
struct Tuple0 {
    static_array_list<Union3,6l> v0;
    static_array_list<Union4,32l> v1;
    Union5 v2;
    static_array<Union2,2l> v3;
    Union8 v4;
    __device__ Tuple0(static_array_list<Union3,6l> t0, static_array_list<Union4,32l> t1, Union5 t2, static_array<Union2,2l> t3, Union8 t4) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4) {}
};
struct Tuple1 {
    Union1 v1;
    long v0;
    __device__ Tuple1(long t0, Union1 t1) : v0(t0), v1(t1) {}
};
struct Tuple2 {
    Union3 v1;
    long v0;
    __device__ Tuple2(long t0, Union3 t1) : v0(t0), v1(t1) {}
};
struct Tuple3 {
    static_array<Union3,2l> v0;
    long v1;
    long v2;
    __device__ Tuple3(static_array<Union3,2l> t0, long t1, long t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Tuple4 {
    Union7 v0;
    static_array<Union3,2l> v2;
    static_array<long,2l> v4;
    long v3;
    long v5;
    bool v1;
    __device__ Tuple4(Union7 t0, bool t1, static_array<Union3,2l> t2, long t3, static_array<long,2l> t4, long t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
};
struct Tuple5 {
    Union7 v0;
    static_array<Union3,2l> v2;
    static_array<long,2l> v4;
    Union1 v6;
    long v3;
    long v5;
    bool v1;
    __device__ Tuple5(Union7 t0, bool t1, static_array<Union3,2l> t2, long t3, static_array<long,2l> t4, long t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
};
struct Tuple6 {
    Union5 v0;
    static_array<Union2,2l> v1;
    Union8 v2;
    __device__ Tuple6(Union5 t0, static_array<Union2,2l> t1, Union8 t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Tuple7 {
    Union6 v1;
    bool v0;
    __device__ Tuple7(bool t0, Union6 t1) : v0(t0), v1(t1) {}
};
struct Tuple8 {
    long v0;
    long v1;
    __device__ Tuple8(long t0, long t1) : v0(t0), v1(t1) {}
};
struct Union9_0 { // Eq
};
struct Union9_1 { // Gt
};
struct Union9_2 { // Lt
};
struct Union9 {
    union {
        Union9_0 case0; // Eq
        Union9_1 case1; // Gt
        Union9_2 case2; // Lt
    };
    __device__ Union9(Union9_0 t) : tag(0), case0(t) {} // Eq
    __device__ Union9(Union9_1 t) : tag(1), case1(t) {} // Gt
    __device__ Union9(Union9_2 t) : tag(2), case2(t) {} // Lt
    __device__ Union9() = delete;
    __device__ Union9(Union9 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union9_0(x.case0); break; // Eq
            case 1: new (&this->case1) Union9_1(x.case1); break; // Gt
            case 2: new (&this->case2) Union9_2(x.case2); break; // Lt
        }
    }
    __device__ Union9(Union9 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union9_0(std::move(x.case0)); break; // Eq
            case 1: new (&this->case1) Union9_1(std::move(x.case1)); break; // Gt
            case 2: new (&this->case2) Union9_2(std::move(x.case2)); break; // Lt
        }
    }
    __device__ Union9 & operator=(Union9 & x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = x.case0; break; // Eq
            case 1: this->case1 = x.case1; break; // Gt
            case 2: this->case2 = x.case2; break; // Lt
        }
        return *this;
    }
    __device__ Union9 & operator=(Union9 && x) {
        this->tag = x.tag;
        switch(x.tag){
            case 0: this->case0 = std::move(x.case0); break; // Eq
            case 1: this->case1 = std::move(x.case1); break; // Gt
            case 2: this->case2 = std::move(x.case2); break; // Lt
        }
        return *this;
    }
    __device__ ~Union9() {
        switch(this->tag){
            case 0: this->case0.~Union9_0(); break; // Eq
            case 1: this->case1.~Union9_1(); break; // Gt
            case 2: this->case2.~Union9_2(); break; // Lt
        }
    }
    unsigned char tag;
};
__device__ long f_1(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+0ull);
    long v2;
    v2 = v1[0l];
    return v2;
}
__device__ void f_3(unsigned char * v0){
    return ;
}
__device__ Union1 f_2(unsigned char * v0){
    long v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    switch (v1) {
        case 0: {
            f_3(v2);
            return Union1{Union1_0{}};
            break;
        }
        case 1: {
            f_3(v2);
            return Union1{Union1_1{}};
            break;
        }
        case 2: {
            f_3(v2);
            return Union1{Union1_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ inline bool while_method_0(long v0){
    bool v1;
    v1 = v0 < 2l;
    return v1;
}
__device__ Union2 f_5(unsigned char * v0){
    long v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    switch (v1) {
        case 0: {
            f_3(v2);
            return Union2{Union2_0{}};
            break;
        }
        case 1: {
            f_3(v2);
            return Union2{Union2_1{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ static_array<Union2,2l> f_4(unsigned char * v0){
    static_array<Union2,2l> v1;
    long v2;
    v2 = 0l;
    while (while_method_0(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned long long v5;
        v5 = v4 * 4ull;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        Union2 v7;
        v7 = f_5(v6);
        v1[v2] = v7;
        v2 += 1l ;
    }
    return v1;
}
__device__ Union0 f_0(unsigned char * v0){
    long v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+8ull);
    switch (v1) {
        case 0: {
            Union1 v4;
            v4 = f_2(v2);
            return Union0{Union0_0{v4}};
            break;
        }
        case 1: {
            static_array<Union2,2l> v6;
            v6 = f_4(v2);
            return Union0{Union0_1{v6}};
            break;
        }
        case 2: {
            f_3(v2);
            return Union0{Union0_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ inline bool while_method_1(long v0, long v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ Union3 f_7(unsigned char * v0){
    long v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    switch (v1) {
        case 0: {
            f_3(v2);
            return Union3{Union3_0{}};
            break;
        }
        case 1: {
            f_3(v2);
            return Union3{Union3_1{}};
            break;
        }
        case 2: {
            f_3(v2);
            return Union3{Union3_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ long f_8(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+28ull);
    long v2;
    v2 = v1[0l];
    return v2;
}
__device__ long f_11(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+4ull);
    long v2;
    v2 = v1[0l];
    return v2;
}
__device__ Tuple1 f_10(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+0ull);
    long v2;
    v2 = v1[0l];
    long v3;
    v3 = f_11(v0);
    unsigned char * v4;
    v4 = (unsigned char *)(v0+8ull);
    Union1 v9;
    switch (v3) {
        case 0: {
            f_3(v4);
            v9 = Union1{Union1_0{}};
            break;
        }
        case 1: {
            f_3(v4);
            v9 = Union1{Union1_1{}};
            break;
        }
        case 2: {
            f_3(v4);
            v9 = Union1{Union1_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple1{{v2, v9}};
}
__device__ Tuple2 f_12(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+0ull);
    long v2;
    v2 = v1[0l];
    long v3;
    v3 = f_11(v0);
    unsigned char * v4;
    v4 = (unsigned char *)(v0+8ull);
    Union3 v9;
    switch (v3) {
        case 0: {
            f_3(v4);
            v9 = Union3{Union3_0{}};
            break;
        }
        case 1: {
            f_3(v4);
            v9 = Union3{Union3_1{}};
            break;
        }
        case 2: {
            f_3(v4);
            v9 = Union3{Union3_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple2{{v2, v9}};
}
__device__ Tuple3 f_13(unsigned char * v0){
    static_array<Union3,2l> v1;
    long v2;
    v2 = 0l;
    while (while_method_0(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned long long v5;
        v5 = v4 * 4ull;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        Union3 v7;
        v7 = f_7(v6);
        v1[v2] = v7;
        v2 += 1l ;
    }
    long * v8;
    v8 = (long *)(v0+8ull);
    long v9;
    v9 = v8[0l];
    long * v10;
    v10 = (long *)(v0+12ull);
    long v11;
    v11 = v10[0l];
    return Tuple3{{v1, v9, v11}};
}
__device__ Union4 f_9(unsigned char * v0){
    long v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+16ull);
    switch (v1) {
        case 0: {
            Union3 v4;
            v4 = f_7(v2);
            return Union4{Union4_0{v4}};
            break;
        }
        case 1: {
            long v6; Union1 v7;
            Tuple1 tmp0 = f_10(v2);
            v6 = tmp0.v0; v7 = tmp0.v1;
            return Union4{Union4_1{v6, v7}};
            break;
        }
        case 2: {
            long v9; Union3 v10;
            Tuple2 tmp1 = f_12(v2);
            v9 = tmp1.v0; v10 = tmp1.v1;
            return Union4{Union4_2{v9, v10}};
            break;
        }
        case 3: {
            static_array<Union3,2l> v12; long v13; long v14;
            Tuple3 tmp2 = f_13(v2);
            v12 = tmp2.v0; v13 = tmp2.v1; v14 = tmp2.v2;
            return Union4{Union4_3{v12, v13, v14}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ long f_14(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+1056ull);
    long v2;
    v2 = v1[0l];
    return v2;
}
__device__ Tuple4 f_16(unsigned char * v0){
    long v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    Union7 v7;
    switch (v1) {
        case 0: {
            f_3(v2);
            v7 = Union7{Union7_0{}};
            break;
        }
        case 1: {
            Union3 v5;
            v5 = f_7(v2);
            v7 = Union7{Union7_1{v5}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    bool * v8;
    v8 = (bool *)(v0+8ull);
    bool v9;
    v9 = v8[0l];
    static_array<Union3,2l> v10;
    long v11;
    v11 = 0l;
    while (while_method_0(v11)){
        unsigned long long v13;
        v13 = (unsigned long long)v11;
        unsigned long long v14;
        v14 = v13 * 4ull;
        unsigned long long v15;
        v15 = 12ull + v14;
        unsigned char * v16;
        v16 = (unsigned char *)(v0+v15);
        Union3 v17;
        v17 = f_7(v16);
        v10[v11] = v17;
        v11 += 1l ;
    }
    long * v18;
    v18 = (long *)(v0+20ull);
    long v19;
    v19 = v18[0l];
    static_array<long,2l> v20;
    long v21;
    v21 = 0l;
    while (while_method_0(v21)){
        unsigned long long v23;
        v23 = (unsigned long long)v21;
        unsigned long long v24;
        v24 = v23 * 4ull;
        unsigned long long v25;
        v25 = 24ull + v24;
        unsigned char * v26;
        v26 = (unsigned char *)(v0+v25);
        long v27;
        v27 = f_1(v26);
        v20[v21] = v27;
        v21 += 1l ;
    }
    long * v28;
    v28 = (long *)(v0+32ull);
    long v29;
    v29 = v28[0l];
    return Tuple4{{v7, v9, v10, v19, v20, v29}};
}
__device__ long f_18(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+36ull);
    long v2;
    v2 = v1[0l];
    return v2;
}
__device__ Tuple5 f_17(unsigned char * v0){
    long v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    Union7 v7;
    switch (v1) {
        case 0: {
            f_3(v2);
            v7 = Union7{Union7_0{}};
            break;
        }
        case 1: {
            Union3 v5;
            v5 = f_7(v2);
            v7 = Union7{Union7_1{v5}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    bool * v8;
    v8 = (bool *)(v0+8ull);
    bool v9;
    v9 = v8[0l];
    static_array<Union3,2l> v10;
    long v11;
    v11 = 0l;
    while (while_method_0(v11)){
        unsigned long long v13;
        v13 = (unsigned long long)v11;
        unsigned long long v14;
        v14 = v13 * 4ull;
        unsigned long long v15;
        v15 = 12ull + v14;
        unsigned char * v16;
        v16 = (unsigned char *)(v0+v15);
        Union3 v17;
        v17 = f_7(v16);
        v10[v11] = v17;
        v11 += 1l ;
    }
    long * v18;
    v18 = (long *)(v0+20ull);
    long v19;
    v19 = v18[0l];
    static_array<long,2l> v20;
    long v21;
    v21 = 0l;
    while (while_method_0(v21)){
        unsigned long long v23;
        v23 = (unsigned long long)v21;
        unsigned long long v24;
        v24 = v23 * 4ull;
        unsigned long long v25;
        v25 = 24ull + v24;
        unsigned char * v26;
        v26 = (unsigned char *)(v0+v25);
        long v27;
        v27 = f_1(v26);
        v20[v21] = v27;
        v21 += 1l ;
    }
    long * v28;
    v28 = (long *)(v0+32ull);
    long v29;
    v29 = v28[0l];
    long v30;
    v30 = f_18(v0);
    unsigned char * v31;
    v31 = (unsigned char *)(v0+40ull);
    Union1 v36;
    switch (v30) {
        case 0: {
            f_3(v31);
            v36 = Union1{Union1_0{}};
            break;
        }
        case 1: {
            f_3(v31);
            v36 = Union1{Union1_1{}};
            break;
        }
        case 2: {
            f_3(v31);
            v36 = Union1{Union1_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple5{{v7, v9, v10, v19, v20, v29, v36}};
}
__device__ Union6 f_15(unsigned char * v0){
    long v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+16ull);
    switch (v1) {
        case 0: {
            Union7 v4; bool v5; static_array<Union3,2l> v6; long v7; static_array<long,2l> v8; long v9;
            Tuple4 tmp3 = f_16(v2);
            v4 = tmp3.v0; v5 = tmp3.v1; v6 = tmp3.v2; v7 = tmp3.v3; v8 = tmp3.v4; v9 = tmp3.v5;
            return Union6{Union6_0{v4, v5, v6, v7, v8, v9}};
            break;
        }
        case 1: {
            f_3(v2);
            return Union6{Union6_1{}};
            break;
        }
        case 2: {
            Union7 v12; bool v13; static_array<Union3,2l> v14; long v15; static_array<long,2l> v16; long v17;
            Tuple4 tmp4 = f_16(v2);
            v12 = tmp4.v0; v13 = tmp4.v1; v14 = tmp4.v2; v15 = tmp4.v3; v16 = tmp4.v4; v17 = tmp4.v5;
            return Union6{Union6_2{v12, v13, v14, v15, v16, v17}};
            break;
        }
        case 3: {
            Union7 v19; bool v20; static_array<Union3,2l> v21; long v22; static_array<long,2l> v23; long v24; Union1 v25;
            Tuple5 tmp5 = f_17(v2);
            v19 = tmp5.v0; v20 = tmp5.v1; v21 = tmp5.v2; v22 = tmp5.v3; v23 = tmp5.v4; v24 = tmp5.v5; v25 = tmp5.v6;
            return Union6{Union6_3{v19, v20, v21, v22, v23, v24, v25}};
            break;
        }
        case 4: {
            Union7 v27; bool v28; static_array<Union3,2l> v29; long v30; static_array<long,2l> v31; long v32;
            Tuple4 tmp6 = f_16(v2);
            v27 = tmp6.v0; v28 = tmp6.v1; v29 = tmp6.v2; v30 = tmp6.v3; v31 = tmp6.v4; v32 = tmp6.v5;
            return Union6{Union6_4{v27, v28, v29, v30, v31, v32}};
            break;
        }
        case 5: {
            Union7 v34; bool v35; static_array<Union3,2l> v36; long v37; static_array<long,2l> v38; long v39;
            Tuple4 tmp7 = f_16(v2);
            v34 = tmp7.v0; v35 = tmp7.v1; v36 = tmp7.v2; v37 = tmp7.v3; v38 = tmp7.v4; v39 = tmp7.v5;
            return Union6{Union6_5{v34, v35, v36, v37, v38, v39}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
}
__device__ long f_19(unsigned char * v0){
    long * v1;
    v1 = (long *)(v0+1144ull);
    long v2;
    v2 = v1[0l];
    return v2;
}
__device__ Tuple0 f_6(unsigned char * v0){
    static_array_list<Union3,6l> v1;
    v1 = static_array_list<Union3,6l>{};
    long v2;
    v2 = f_1(v0);
    v1.unsafe_set_length(v2);
    long v3;
    v3 = v1.length;
    long v4;
    v4 = 0l;
    while (while_method_1(v3, v4)){
        unsigned long long v6;
        v6 = (unsigned long long)v4;
        unsigned long long v7;
        v7 = v6 * 4ull;
        unsigned long long v8;
        v8 = 4ull + v7;
        unsigned char * v9;
        v9 = (unsigned char *)(v0+v8);
        Union3 v10;
        v10 = f_7(v9);
        v1[v4] = v10;
        v4 += 1l ;
    }
    static_array_list<Union4,32l> v11;
    v11 = static_array_list<Union4,32l>{};
    long v12;
    v12 = f_8(v0);
    v11.unsafe_set_length(v12);
    long v13;
    v13 = v11.length;
    long v14;
    v14 = 0l;
    while (while_method_1(v13, v14)){
        unsigned long long v16;
        v16 = (unsigned long long)v14;
        unsigned long long v17;
        v17 = v16 * 32ull;
        unsigned long long v18;
        v18 = 32ull + v17;
        unsigned char * v19;
        v19 = (unsigned char *)(v0+v18);
        Union4 v20;
        v20 = f_9(v19);
        v11[v14] = v20;
        v14 += 1l ;
    }
    long v21;
    v21 = f_14(v0);
    unsigned char * v22;
    v22 = (unsigned char *)(v0+1072ull);
    Union5 v27;
    switch (v21) {
        case 0: {
            f_3(v22);
            v27 = Union5{Union5_0{}};
            break;
        }
        case 1: {
            Union6 v25;
            v25 = f_15(v22);
            v27 = Union5{Union5_1{v25}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    static_array<Union2,2l> v28;
    long v29;
    v29 = 0l;
    while (while_method_0(v29)){
        unsigned long long v31;
        v31 = (unsigned long long)v29;
        unsigned long long v32;
        v32 = v31 * 4ull;
        unsigned long long v33;
        v33 = 1136ull + v32;
        unsigned char * v34;
        v34 = (unsigned char *)(v0+v33);
        Union2 v35;
        v35 = f_5(v34);
        v28[v29] = v35;
        v29 += 1l ;
    }
    long v36;
    v36 = f_19(v0);
    unsigned char * v37;
    v37 = (unsigned char *)(v0+1152ull);
    Union8 v54;
    switch (v36) {
        case 0: {
            f_3(v37);
            v54 = Union8{Union8_0{}};
            break;
        }
        case 1: {
            Union7 v40; bool v41; static_array<Union3,2l> v42; long v43; static_array<long,2l> v44; long v45;
            Tuple4 tmp8 = f_16(v37);
            v40 = tmp8.v0; v41 = tmp8.v1; v42 = tmp8.v2; v43 = tmp8.v3; v44 = tmp8.v4; v45 = tmp8.v5;
            v54 = Union8{Union8_1{v40, v41, v42, v43, v44, v45}};
            break;
        }
        case 2: {
            Union7 v47; bool v48; static_array<Union3,2l> v49; long v50; static_array<long,2l> v51; long v52;
            Tuple4 tmp9 = f_16(v37);
            v47 = tmp9.v0; v48 = tmp9.v1; v49 = tmp9.v2; v50 = tmp9.v3; v51 = tmp9.v4; v52 = tmp9.v5;
            v54 = Union8{Union8_2{v47, v48, v49, v50, v51, v52}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            asm("exit;");
        }
    }
    return Tuple0{{v1, v11, v27, v28, v54}};
}
__device__ inline bool while_method_2(bool v0, Union6 v1){
    return v0;
}
__device__ unsigned long loop_22(unsigned long v0, curandStatePhilox4_32_10_t & v1){
    unsigned long v2;
    v2 = curand(&v1);
    unsigned long v3;
    v3 = v2 % v0;
    unsigned long v4;
    v4 = v2 - v3;
    unsigned long v5;
    v5 = 0ul - v0;
    bool v6;
    v6 = v4 <= v5;
    if (v6){
        return v3;
    } else {
        return loop_22(v0, v1);
    }
}
__device__ long tag_24(Union3 v0){
    switch (v0.tag) {
        case 0: { // Jack
            return 0l;
            break;
        }
        case 1: { // King
            return 2l;
            break;
        }
        case 2: { // Queen
            return 1l;
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ bool is_pair_25(long v0, long v1){
    bool v2;
    v2 = v1 == v0;
    return v2;
}
__device__ Tuple8 order_26(long v0, long v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple8{{v1, v0}};
    } else {
        return Tuple8{{v0, v1}};
    }
}
__device__ Union9 compare_hands_23(Union7 v0, bool v1, static_array<Union3,2l> v2, long v3, static_array<long,2l> v4, long v5){
    switch (v0.tag) {
        case 0: { // None
            printf("%s\n", "Expected the community card to be present in the table.");
            asm("exit;");
            break;
        }
        case 1: { // Some
            Union3 v7 = v0.case1.v0;
            long v8;
            v8 = tag_24(v7);
            Union3 v9;
            v9 = v2[0l];
            long v10;
            v10 = tag_24(v9);
            Union3 v11;
            v11 = v2[1l];
            long v12;
            v12 = tag_24(v11);
            bool v13;
            v13 = is_pair_25(v8, v10);
            bool v14;
            v14 = is_pair_25(v8, v12);
            if (v13){
                if (v14){
                    bool v15;
                    v15 = v10 < v12;
                    if (v15){
                        return Union9{Union9_2{}};
                    } else {
                        bool v17;
                        v17 = v10 > v12;
                        if (v17){
                            return Union9{Union9_1{}};
                        } else {
                            return Union9{Union9_0{}};
                        }
                    }
                } else {
                    return Union9{Union9_1{}};
                }
            } else {
                if (v14){
                    return Union9{Union9_2{}};
                } else {
                    long v25; long v26;
                    Tuple8 tmp19 = order_26(v8, v10);
                    v25 = tmp19.v0; v26 = tmp19.v1;
                    long v27; long v28;
                    Tuple8 tmp20 = order_26(v8, v12);
                    v27 = tmp20.v0; v28 = tmp20.v1;
                    bool v29;
                    v29 = v25 < v27;
                    Union9 v35;
                    if (v29){
                        v35 = Union9{Union9_2{}};
                    } else {
                        bool v31;
                        v31 = v25 > v27;
                        if (v31){
                            v35 = Union9{Union9_1{}};
                        } else {
                            v35 = Union9{Union9_0{}};
                        }
                    }
                    bool v36;
                    switch (v35.tag) {
                        case 0: { // Eq
                            v36 = true;
                            break;
                        }
                        default: {
                            v36 = false;
                        }
                    }
                    if (v36){
                        bool v37;
                        v37 = v26 < v28;
                        if (v37){
                            return Union9{Union9_2{}};
                        } else {
                            bool v39;
                            v39 = v26 > v28;
                            if (v39){
                                return Union9{Union9_1{}};
                            } else {
                                return Union9{Union9_0{}};
                            }
                        }
                    } else {
                        return v35;
                    }
                }
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ Union6 play_loop_inner_21(static_array_list<Union3,6l> & v0, static_array_list<Union4,32l> & v1, static_array<Union2,2l> v2, Union6 v3){
    static_array_list<Union4,32l> & v4 = v1;
    static_array_list<Union3,6l> & v5 = v0;
    bool v6; Union6 v7;
    Tuple7 tmp11 = Tuple7{{true, v3}};
    v6 = tmp11.v0; v7 = tmp11.v1;
    while (while_method_2(v6, v7)){
        bool v268; Union6 v269;
        switch (v7.tag) {
            case 0: { // ChanceCommunityCard
                Union7 v229 = v7.case0.v0; bool v230 = v7.case0.v1; static_array<Union3,2l> v231 = v7.case0.v2; long v232 = v7.case0.v3; static_array<long,2l> v233 = v7.case0.v4; long v234 = v7.case0.v5;
                Union3 v235;
                v235 = v5.pop();
                Union4 v236;
                v236 = Union4{Union4_0{v235}};
                v4.push(v236);
                long v237;
                v237 = 2l;
                long v238; long v239;
                Tuple8 tmp12 = Tuple8{{0l, 0l}};
                v238 = tmp12.v0; v239 = tmp12.v1;
                while (while_method_0(v238)){
                    long v241;
                    v241 = v233[v238];
                    bool v242;
                    v242 = v239 >= v241;
                    long v243;
                    if (v242){
                        v243 = v239;
                    } else {
                        v243 = v241;
                    }
                    v239 = v243;
                    v238 += 1l ;
                }
                static_array<long,2l> v244;
                long v245;
                v245 = 0l;
                while (while_method_0(v245)){
                    v244[v245] = v239;
                    v245 += 1l ;
                }
                Union7 v247;
                v247 = Union7{Union7_1{v235}};
                Union6 v248;
                v248 = Union6{Union6_2{v247, true, v231, 0l, v244, v237}};
                v268 = true; v269 = v248;
                break;
            }
            case 1: { // ChanceInit
                Union3 v249;
                v249 = v5.pop();
                Union3 v250;
                v250 = v5.pop();
                Union4 v251;
                v251 = Union4{Union4_2{0l, v249}};
                v4.push(v251);
                Union4 v252;
                v252 = Union4{Union4_2{1l, v250}};
                v4.push(v252);
                long v253;
                v253 = 2l;
                static_array<long,2l> v254;
                v254[0l] = 1l;
                v254[1l] = 1l;
                static_array<Union3,2l> v255;
                v255[0l] = v249;
                v255[1l] = v250;
                Union7 v256;
                v256 = Union7{Union7_0{}};
                Union6 v257;
                v257 = Union6{Union6_2{v256, true, v255, 0l, v254, v253}};
                v268 = true; v269 = v257;
                break;
            }
            case 2: { // Round
                Union7 v34 = v7.case2.v0; bool v35 = v7.case2.v1; static_array<Union3,2l> v36 = v7.case2.v2; long v37 = v7.case2.v3; static_array<long,2l> v38 = v7.case2.v4; long v39 = v7.case2.v5;
                Union2 v40;
                v40 = v2[v37];
                switch (v40.tag) {
                    case 0: { // Computer
                        static_array_list<Union1,3l> v41;
                        v41 = static_array_list<Union1,3l>{};
                        v41.unsafe_set_length(1l);
                        Union1 v42;
                        v42 = Union1{Union1_0{}};
                        v41[0l] = v42;
                        long v43;
                        v43 = v38[0l];
                        long v44;
                        v44 = v38[1l];
                        bool v45;
                        v45 = v43 == v44;
                        bool v46;
                        v46 = v45 != true;
                        if (v46){
                            Union1 v47;
                            v47 = Union1{Union1_1{}};
                            v41.push(v47);
                        } else {
                        }
                        bool v48;
                        v48 = v39 > 0l;
                        if (v48){
                            Union1 v49;
                            v49 = Union1{Union1_2{}};
                            v41.push(v49);
                        } else {
                        }
                        unsigned long long v50;
                        v50 = clock64();
                        curandStatePhilox4_32_10_t v51;
                        curand_init(v50,0ull,0ull,&v51);
                        long v52;
                        v52 = v41.length;
                        long v53;
                        v53 = v52 - 1l;
                        long v54;
                        v54 = 0l;
                        while (while_method_1(v53, v54)){
                            long v56;
                            v56 = v41.length;
                            long v57;
                            v57 = v56 - v54;
                            unsigned long v58;
                            v58 = (unsigned long)v57;
                            unsigned long v59;
                            v59 = loop_22(v58, v51);
                            unsigned long v60;
                            v60 = (unsigned long)v54;
                            unsigned long v61;
                            v61 = v59 + v60;
                            long v62;
                            v62 = (long)v61;
                            Union1 v63;
                            v63 = v41[v54];
                            Union1 v64;
                            v64 = v41[v62];
                            v41[v54] = v64;
                            v41[v62] = v63;
                            v54 += 1l ;
                        }
                        Union1 v65;
                        v65 = v41.pop();
                        Union4 v66;
                        v66 = Union4{Union4_1{v37, v65}};
                        v4.push(v66);
                        Union6 v140;
                        switch (v34.tag) {
                            case 0: { // None
                                switch (v65.tag) {
                                    case 0: { // Call
                                        if (v35){
                                            bool v109;
                                            v109 = v37 == 0l;
                                            long v110;
                                            if (v109){
                                                v110 = 1l;
                                            } else {
                                                v110 = 0l;
                                            }
                                            v140 = Union6{Union6_2{v34, false, v36, v110, v38, v39}};
                                        } else {
                                            v140 = Union6{Union6_0{v34, v35, v36, v37, v38, v39}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v140 = Union6{Union6_5{v34, v35, v36, v37, v38, v39}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        if (v48){
                                            bool v114;
                                            v114 = v37 == 0l;
                                            long v115;
                                            if (v114){
                                                v115 = 1l;
                                            } else {
                                                v115 = 0l;
                                            }
                                            long v116;
                                            v116 = -1l + v39;
                                            long v117; long v118;
                                            Tuple8 tmp13 = Tuple8{{0l, 0l}};
                                            v117 = tmp13.v0; v118 = tmp13.v1;
                                            while (while_method_0(v117)){
                                                long v120;
                                                v120 = v38[v117];
                                                bool v121;
                                                v121 = v118 >= v120;
                                                long v122;
                                                if (v121){
                                                    v122 = v118;
                                                } else {
                                                    v122 = v120;
                                                }
                                                v118 = v122;
                                                v117 += 1l ;
                                            }
                                            static_array<long,2l> v123;
                                            long v124;
                                            v124 = 0l;
                                            while (while_method_0(v124)){
                                                v123[v124] = v118;
                                                v124 += 1l ;
                                            }
                                            static_array<long,2l> v126;
                                            long v127;
                                            v127 = 0l;
                                            while (while_method_0(v127)){
                                                long v129;
                                                v129 = v123[v127];
                                                bool v130;
                                                v130 = v127 == v37;
                                                long v132;
                                                if (v130){
                                                    long v131;
                                                    v131 = v129 + 2l;
                                                    v132 = v131;
                                                } else {
                                                    v132 = v129;
                                                }
                                                v126[v127] = v132;
                                                v127 += 1l ;
                                            }
                                            v140 = Union6{Union6_2{v34, false, v36, v115, v126, v116}};
                                        } else {
                                            printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                            asm("exit;");
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false);
                                    }
                                }
                                break;
                            }
                            case 1: { // Some
                                Union3 v67 = v34.case1.v0;
                                switch (v65.tag) {
                                    case 0: { // Call
                                        if (v35){
                                            bool v69;
                                            v69 = v37 == 0l;
                                            long v70;
                                            if (v69){
                                                v70 = 1l;
                                            } else {
                                                v70 = 0l;
                                            }
                                            v140 = Union6{Union6_2{v34, false, v36, v70, v38, v39}};
                                        } else {
                                            long v72; long v73;
                                            Tuple8 tmp14 = Tuple8{{0l, 0l}};
                                            v72 = tmp14.v0; v73 = tmp14.v1;
                                            while (while_method_0(v72)){
                                                long v75;
                                                v75 = v38[v72];
                                                bool v76;
                                                v76 = v73 >= v75;
                                                long v77;
                                                if (v76){
                                                    v77 = v73;
                                                } else {
                                                    v77 = v75;
                                                }
                                                v73 = v77;
                                                v72 += 1l ;
                                            }
                                            static_array<long,2l> v78;
                                            long v79;
                                            v79 = 0l;
                                            while (while_method_0(v79)){
                                                v78[v79] = v73;
                                                v79 += 1l ;
                                            }
                                            v140 = Union6{Union6_4{v34, v35, v36, v37, v78, v39}};
                                        }
                                        break;
                                    }
                                    case 1: { // Fold
                                        v140 = Union6{Union6_5{v34, v35, v36, v37, v38, v39}};
                                        break;
                                    }
                                    case 2: { // Raise
                                        if (v48){
                                            bool v83;
                                            v83 = v37 == 0l;
                                            long v84;
                                            if (v83){
                                                v84 = 1l;
                                            } else {
                                                v84 = 0l;
                                            }
                                            long v85;
                                            v85 = -1l + v39;
                                            long v86; long v87;
                                            Tuple8 tmp15 = Tuple8{{0l, 0l}};
                                            v86 = tmp15.v0; v87 = tmp15.v1;
                                            while (while_method_0(v86)){
                                                long v89;
                                                v89 = v38[v86];
                                                bool v90;
                                                v90 = v87 >= v89;
                                                long v91;
                                                if (v90){
                                                    v91 = v87;
                                                } else {
                                                    v91 = v89;
                                                }
                                                v87 = v91;
                                                v86 += 1l ;
                                            }
                                            static_array<long,2l> v92;
                                            long v93;
                                            v93 = 0l;
                                            while (while_method_0(v93)){
                                                v92[v93] = v87;
                                                v93 += 1l ;
                                            }
                                            static_array<long,2l> v95;
                                            long v96;
                                            v96 = 0l;
                                            while (while_method_0(v96)){
                                                long v98;
                                                v98 = v92[v96];
                                                bool v99;
                                                v99 = v96 == v37;
                                                long v101;
                                                if (v99){
                                                    long v100;
                                                    v100 = v98 + 4l;
                                                    v101 = v100;
                                                } else {
                                                    v101 = v98;
                                                }
                                                v95[v96] = v101;
                                                v96 += 1l ;
                                            }
                                            v140 = Union6{Union6_2{v34, false, v36, v84, v95, v85}};
                                        } else {
                                            printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                            asm("exit;");
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false);
                                    }
                                }
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                            }
                        }
                        v268 = true; v269 = v140;
                        break;
                    }
                    case 1: { // Human
                        v268 = false; v269 = v7;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                    }
                }
                break;
            }
            case 3: { // RoundWithAction
                Union7 v145 = v7.case3.v0; bool v146 = v7.case3.v1; static_array<Union3,2l> v147 = v7.case3.v2; long v148 = v7.case3.v3; static_array<long,2l> v149 = v7.case3.v4; long v150 = v7.case3.v5; Union1 v151 = v7.case3.v6;
                Union4 v152;
                v152 = Union4{Union4_1{v148, v151}};
                v4.push(v152);
                Union6 v228;
                switch (v145.tag) {
                    case 0: { // None
                        switch (v151.tag) {
                            case 0: { // Call
                                if (v146){
                                    bool v196;
                                    v196 = v148 == 0l;
                                    long v197;
                                    if (v196){
                                        v197 = 1l;
                                    } else {
                                        v197 = 0l;
                                    }
                                    v228 = Union6{Union6_2{v145, false, v147, v197, v149, v150}};
                                } else {
                                    v228 = Union6{Union6_0{v145, v146, v147, v148, v149, v150}};
                                }
                                break;
                            }
                            case 1: { // Fold
                                v228 = Union6{Union6_5{v145, v146, v147, v148, v149, v150}};
                                break;
                            }
                            case 2: { // Raise
                                bool v201;
                                v201 = v150 > 0l;
                                if (v201){
                                    bool v202;
                                    v202 = v148 == 0l;
                                    long v203;
                                    if (v202){
                                        v203 = 1l;
                                    } else {
                                        v203 = 0l;
                                    }
                                    long v204;
                                    v204 = -1l + v150;
                                    long v205; long v206;
                                    Tuple8 tmp16 = Tuple8{{0l, 0l}};
                                    v205 = tmp16.v0; v206 = tmp16.v1;
                                    while (while_method_0(v205)){
                                        long v208;
                                        v208 = v149[v205];
                                        bool v209;
                                        v209 = v206 >= v208;
                                        long v210;
                                        if (v209){
                                            v210 = v206;
                                        } else {
                                            v210 = v208;
                                        }
                                        v206 = v210;
                                        v205 += 1l ;
                                    }
                                    static_array<long,2l> v211;
                                    long v212;
                                    v212 = 0l;
                                    while (while_method_0(v212)){
                                        v211[v212] = v206;
                                        v212 += 1l ;
                                    }
                                    static_array<long,2l> v214;
                                    long v215;
                                    v215 = 0l;
                                    while (while_method_0(v215)){
                                        long v217;
                                        v217 = v211[v215];
                                        bool v218;
                                        v218 = v215 == v148;
                                        long v220;
                                        if (v218){
                                            long v219;
                                            v219 = v217 + 2l;
                                            v220 = v219;
                                        } else {
                                            v220 = v217;
                                        }
                                        v214[v215] = v220;
                                        v215 += 1l ;
                                    }
                                    v228 = Union6{Union6_2{v145, false, v147, v203, v214, v204}};
                                } else {
                                    printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                    asm("exit;");
                                }
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                            }
                        }
                        break;
                    }
                    case 1: { // Some
                        Union3 v153 = v145.case1.v0;
                        switch (v151.tag) {
                            case 0: { // Call
                                if (v146){
                                    bool v155;
                                    v155 = v148 == 0l;
                                    long v156;
                                    if (v155){
                                        v156 = 1l;
                                    } else {
                                        v156 = 0l;
                                    }
                                    v228 = Union6{Union6_2{v145, false, v147, v156, v149, v150}};
                                } else {
                                    long v158; long v159;
                                    Tuple8 tmp17 = Tuple8{{0l, 0l}};
                                    v158 = tmp17.v0; v159 = tmp17.v1;
                                    while (while_method_0(v158)){
                                        long v161;
                                        v161 = v149[v158];
                                        bool v162;
                                        v162 = v159 >= v161;
                                        long v163;
                                        if (v162){
                                            v163 = v159;
                                        } else {
                                            v163 = v161;
                                        }
                                        v159 = v163;
                                        v158 += 1l ;
                                    }
                                    static_array<long,2l> v164;
                                    long v165;
                                    v165 = 0l;
                                    while (while_method_0(v165)){
                                        v164[v165] = v159;
                                        v165 += 1l ;
                                    }
                                    v228 = Union6{Union6_4{v145, v146, v147, v148, v164, v150}};
                                }
                                break;
                            }
                            case 1: { // Fold
                                v228 = Union6{Union6_5{v145, v146, v147, v148, v149, v150}};
                                break;
                            }
                            case 2: { // Raise
                                bool v169;
                                v169 = v150 > 0l;
                                if (v169){
                                    bool v170;
                                    v170 = v148 == 0l;
                                    long v171;
                                    if (v170){
                                        v171 = 1l;
                                    } else {
                                        v171 = 0l;
                                    }
                                    long v172;
                                    v172 = -1l + v150;
                                    long v173; long v174;
                                    Tuple8 tmp18 = Tuple8{{0l, 0l}};
                                    v173 = tmp18.v0; v174 = tmp18.v1;
                                    while (while_method_0(v173)){
                                        long v176;
                                        v176 = v149[v173];
                                        bool v177;
                                        v177 = v174 >= v176;
                                        long v178;
                                        if (v177){
                                            v178 = v174;
                                        } else {
                                            v178 = v176;
                                        }
                                        v174 = v178;
                                        v173 += 1l ;
                                    }
                                    static_array<long,2l> v179;
                                    long v180;
                                    v180 = 0l;
                                    while (while_method_0(v180)){
                                        v179[v180] = v174;
                                        v180 += 1l ;
                                    }
                                    static_array<long,2l> v182;
                                    long v183;
                                    v183 = 0l;
                                    while (while_method_0(v183)){
                                        long v185;
                                        v185 = v179[v183];
                                        bool v186;
                                        v186 = v183 == v148;
                                        long v188;
                                        if (v186){
                                            long v187;
                                            v187 = v185 + 4l;
                                            v188 = v187;
                                        } else {
                                            v188 = v185;
                                        }
                                        v182[v183] = v188;
                                        v183 += 1l ;
                                    }
                                    v228 = Union6{Union6_2{v145, false, v147, v171, v182, v172}};
                                } else {
                                    printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                    asm("exit;");
                                }
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                            }
                        }
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                    }
                }
                v268 = true; v269 = v228;
                break;
            }
            case 4: { // TerminalCall
                Union7 v19 = v7.case4.v0; bool v20 = v7.case4.v1; static_array<Union3,2l> v21 = v7.case4.v2; long v22 = v7.case4.v3; static_array<long,2l> v23 = v7.case4.v4; long v24 = v7.case4.v5;
                long v25;
                v25 = v23[v22];
                Union9 v26;
                v26 = compare_hands_23(v19, v20, v21, v22, v23, v24);
                long v31; long v32;
                switch (v26.tag) {
                    case 0: { // Eq
                        v31 = 0l; v32 = -1l;
                        break;
                    }
                    case 1: { // Gt
                        v31 = v25; v32 = 0l;
                        break;
                    }
                    case 2: { // Lt
                        v31 = v25; v32 = 1l;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                    }
                }
                Union4 v33;
                v33 = Union4{Union4_3{v21, v31, v32}};
                v4.push(v33);
                v268 = false; v269 = v7;
                break;
            }
            case 5: { // TerminalFold
                Union7 v9 = v7.case5.v0; bool v10 = v7.case5.v1; static_array<Union3,2l> v11 = v7.case5.v2; long v12 = v7.case5.v3; static_array<long,2l> v13 = v7.case5.v4; long v14 = v7.case5.v5;
                long v15;
                v15 = v13[v12];
                bool v16;
                v16 = v12 == 0l;
                long v17;
                if (v16){
                    v17 = 1l;
                } else {
                    v17 = 0l;
                }
                Union4 v18;
                v18 = Union4{Union4_3{v11, v15, v17}};
                v4.push(v18);
                v268 = false; v269 = v7;
                break;
            }
            default: {
                assert("Invalid tag." && false);
            }
        }
        v6 = v268;
        v7 = v269;
    }
    return v7;
}
__device__ Tuple6 play_loop_20(Union5 v0, static_array<Union2,2l> v1, Union8 v2, static_array_list<Union3,6l> & v3, static_array_list<Union4,32l> & v4, Union6 v5){
    Union6 v6;
    v6 = play_loop_inner_21(v3, v4, v1, v5);
    switch (v6.tag) {
        case 2: { // Round
            Union7 v7 = v6.case2.v0; bool v8 = v6.case2.v1; static_array<Union3,2l> v9 = v6.case2.v2; long v10 = v6.case2.v3; static_array<long,2l> v11 = v6.case2.v4; long v12 = v6.case2.v5;
            Union5 v13;
            v13 = Union5{Union5_1{v6}};
            Union8 v14;
            v14 = Union8{Union8_2{v7, v8, v9, v10, v11, v12}};
            return Tuple6{{v13, v1, v14}};
            break;
        }
        case 4: { // TerminalCall
            Union7 v15 = v6.case4.v0; bool v16 = v6.case4.v1; static_array<Union3,2l> v17 = v6.case4.v2; long v18 = v6.case4.v3; static_array<long,2l> v19 = v6.case4.v4; long v20 = v6.case4.v5;
            Union5 v21;
            v21 = Union5{Union5_0{}};
            Union8 v22;
            v22 = Union8{Union8_1{v15, v16, v17, v18, v19, v20}};
            return Tuple6{{v21, v1, v22}};
            break;
        }
        case 5: { // TerminalFold
            Union7 v23 = v6.case5.v0; bool v24 = v6.case5.v1; static_array<Union3,2l> v25 = v6.case5.v2; long v26 = v6.case5.v3; static_array<long,2l> v27 = v6.case5.v4; long v28 = v6.case5.v5;
            Union5 v29;
            v29 = Union5{Union5_0{}};
            Union8 v30;
            v30 = Union8{Union8_1{v23, v24, v25, v26, v27, v28}};
            return Tuple6{{v29, v1, v30}};
            break;
        }
        default: {
            printf("%s\n", "Unexpected node received in play_loop.");
            asm("exit;");
        }
    }
}
__device__ void f_28(unsigned char * v0, long v1){
    long * v2;
    v2 = (long *)(v0+0ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_30(unsigned char * v0){
    return ;
}
__device__ void f_29(unsigned char * v0, Union3 v1){
    long v2;
    v2 = v1.tag;
    f_28(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // Jack
            return f_30(v3);
            break;
        }
        case 1: { // King
            return f_30(v3);
            break;
        }
        case 2: { // Queen
            return f_30(v3);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_31(unsigned char * v0, long v1){
    long * v2;
    v2 = (long *)(v0+28ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_34(unsigned char * v0, long v1){
    long * v2;
    v2 = (long *)(v0+4ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_33(unsigned char * v0, long v1, Union1 v2){
    long * v3;
    v3 = (long *)(v0+0ull);
    v3[0l] = v1;
    long v4;
    v4 = v2.tag;
    f_34(v0, v4);
    unsigned char * v5;
    v5 = (unsigned char *)(v0+8ull);
    switch (v2.tag) {
        case 0: { // Call
            return f_30(v5);
            break;
        }
        case 1: { // Fold
            return f_30(v5);
            break;
        }
        case 2: { // Raise
            return f_30(v5);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_35(unsigned char * v0, long v1, Union3 v2){
    long * v3;
    v3 = (long *)(v0+0ull);
    v3[0l] = v1;
    long v4;
    v4 = v2.tag;
    f_34(v0, v4);
    unsigned char * v5;
    v5 = (unsigned char *)(v0+8ull);
    switch (v2.tag) {
        case 0: { // Jack
            return f_30(v5);
            break;
        }
        case 1: { // King
            return f_30(v5);
            break;
        }
        case 2: { // Queen
            return f_30(v5);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_36(unsigned char * v0, static_array<Union3,2l> v1, long v2, long v3){
    long v4;
    v4 = 0l;
    while (while_method_0(v4)){
        unsigned long long v6;
        v6 = (unsigned long long)v4;
        unsigned long long v7;
        v7 = v6 * 4ull;
        unsigned char * v8;
        v8 = (unsigned char *)(v0+v7);
        Union3 v9;
        v9 = v1[v4];
        f_29(v8, v9);
        v4 += 1l ;
    }
    long * v10;
    v10 = (long *)(v0+8ull);
    v10[0l] = v2;
    long * v11;
    v11 = (long *)(v0+12ull);
    v11[0l] = v3;
    return ;
}
__device__ void f_32(unsigned char * v0, Union4 v1){
    long v2;
    v2 = v1.tag;
    f_28(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // CommunityCardIs
            Union3 v4 = v1.case0.v0;
            return f_29(v3, v4);
            break;
        }
        case 1: { // PlayerAction
            long v5 = v1.case1.v0; Union1 v6 = v1.case1.v1;
            return f_33(v3, v5, v6);
            break;
        }
        case 2: { // PlayerGotCard
            long v7 = v1.case2.v0; Union3 v8 = v1.case2.v1;
            return f_35(v3, v7, v8);
            break;
        }
        case 3: { // Showdown
            static_array<Union3,2l> v9 = v1.case3.v0; long v10 = v1.case3.v1; long v11 = v1.case3.v2;
            return f_36(v3, v9, v10, v11);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_37(unsigned char * v0, long v1){
    long * v2;
    v2 = (long *)(v0+1056ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_39(unsigned char * v0, Union7 v1, bool v2, static_array<Union3,2l> v3, long v4, static_array<long,2l> v5, long v6){
    long v7;
    v7 = v1.tag;
    f_28(v0, v7);
    unsigned char * v8;
    v8 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // None
            f_30(v8);
            break;
        }
        case 1: { // Some
            Union3 v9 = v1.case1.v0;
            f_29(v8, v9);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    bool * v10;
    v10 = (bool *)(v0+8ull);
    v10[0l] = v2;
    long v11;
    v11 = 0l;
    while (while_method_0(v11)){
        unsigned long long v13;
        v13 = (unsigned long long)v11;
        unsigned long long v14;
        v14 = v13 * 4ull;
        unsigned long long v15;
        v15 = 12ull + v14;
        unsigned char * v16;
        v16 = (unsigned char *)(v0+v15);
        Union3 v17;
        v17 = v3[v11];
        f_29(v16, v17);
        v11 += 1l ;
    }
    long * v18;
    v18 = (long *)(v0+20ull);
    v18[0l] = v4;
    long v19;
    v19 = 0l;
    while (while_method_0(v19)){
        unsigned long long v21;
        v21 = (unsigned long long)v19;
        unsigned long long v22;
        v22 = v21 * 4ull;
        unsigned long long v23;
        v23 = 24ull + v22;
        unsigned char * v24;
        v24 = (unsigned char *)(v0+v23);
        long v25;
        v25 = v5[v19];
        f_28(v24, v25);
        v19 += 1l ;
    }
    long * v26;
    v26 = (long *)(v0+32ull);
    v26[0l] = v6;
    return ;
}
__device__ void f_41(unsigned char * v0, long v1){
    long * v2;
    v2 = (long *)(v0+36ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_40(unsigned char * v0, Union7 v1, bool v2, static_array<Union3,2l> v3, long v4, static_array<long,2l> v5, long v6, Union1 v7){
    long v8;
    v8 = v1.tag;
    f_28(v0, v8);
    unsigned char * v9;
    v9 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // None
            f_30(v9);
            break;
        }
        case 1: { // Some
            Union3 v10 = v1.case1.v0;
            f_29(v9, v10);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    bool * v11;
    v11 = (bool *)(v0+8ull);
    v11[0l] = v2;
    long v12;
    v12 = 0l;
    while (while_method_0(v12)){
        unsigned long long v14;
        v14 = (unsigned long long)v12;
        unsigned long long v15;
        v15 = v14 * 4ull;
        unsigned long long v16;
        v16 = 12ull + v15;
        unsigned char * v17;
        v17 = (unsigned char *)(v0+v16);
        Union3 v18;
        v18 = v3[v12];
        f_29(v17, v18);
        v12 += 1l ;
    }
    long * v19;
    v19 = (long *)(v0+20ull);
    v19[0l] = v4;
    long v20;
    v20 = 0l;
    while (while_method_0(v20)){
        unsigned long long v22;
        v22 = (unsigned long long)v20;
        unsigned long long v23;
        v23 = v22 * 4ull;
        unsigned long long v24;
        v24 = 24ull + v23;
        unsigned char * v25;
        v25 = (unsigned char *)(v0+v24);
        long v26;
        v26 = v5[v20];
        f_28(v25, v26);
        v20 += 1l ;
    }
    long * v27;
    v27 = (long *)(v0+32ull);
    v27[0l] = v6;
    long v28;
    v28 = v7.tag;
    f_41(v0, v28);
    unsigned char * v29;
    v29 = (unsigned char *)(v0+40ull);
    switch (v7.tag) {
        case 0: { // Call
            return f_30(v29);
            break;
        }
        case 1: { // Fold
            return f_30(v29);
            break;
        }
        case 2: { // Raise
            return f_30(v29);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_38(unsigned char * v0, Union6 v1){
    long v2;
    v2 = v1.tag;
    f_28(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // ChanceCommunityCard
            Union7 v4 = v1.case0.v0; bool v5 = v1.case0.v1; static_array<Union3,2l> v6 = v1.case0.v2; long v7 = v1.case0.v3; static_array<long,2l> v8 = v1.case0.v4; long v9 = v1.case0.v5;
            return f_39(v3, v4, v5, v6, v7, v8, v9);
            break;
        }
        case 1: { // ChanceInit
            return f_30(v3);
            break;
        }
        case 2: { // Round
            Union7 v10 = v1.case2.v0; bool v11 = v1.case2.v1; static_array<Union3,2l> v12 = v1.case2.v2; long v13 = v1.case2.v3; static_array<long,2l> v14 = v1.case2.v4; long v15 = v1.case2.v5;
            return f_39(v3, v10, v11, v12, v13, v14, v15);
            break;
        }
        case 3: { // RoundWithAction
            Union7 v16 = v1.case3.v0; bool v17 = v1.case3.v1; static_array<Union3,2l> v18 = v1.case3.v2; long v19 = v1.case3.v3; static_array<long,2l> v20 = v1.case3.v4; long v21 = v1.case3.v5; Union1 v22 = v1.case3.v6;
            return f_40(v3, v16, v17, v18, v19, v20, v21, v22);
            break;
        }
        case 4: { // TerminalCall
            Union7 v23 = v1.case4.v0; bool v24 = v1.case4.v1; static_array<Union3,2l> v25 = v1.case4.v2; long v26 = v1.case4.v3; static_array<long,2l> v27 = v1.case4.v4; long v28 = v1.case4.v5;
            return f_39(v3, v23, v24, v25, v26, v27, v28);
            break;
        }
        case 5: { // TerminalFold
            Union7 v29 = v1.case5.v0; bool v30 = v1.case5.v1; static_array<Union3,2l> v31 = v1.case5.v2; long v32 = v1.case5.v3; static_array<long,2l> v33 = v1.case5.v4; long v34 = v1.case5.v5;
            return f_39(v3, v29, v30, v31, v32, v33, v34);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_42(unsigned char * v0, Union2 v1){
    long v2;
    v2 = v1.tag;
    f_28(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // Computer
            return f_30(v3);
            break;
        }
        case 1: { // Human
            return f_30(v3);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_43(unsigned char * v0, long v1){
    long * v2;
    v2 = (long *)(v0+1144ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_27(unsigned char * v0, static_array_list<Union3,6l> v1, static_array_list<Union4,32l> v2, Union5 v3, static_array<Union2,2l> v4, Union8 v5){
    long v6;
    v6 = v1.length;
    f_28(v0, v6);
    long v7;
    v7 = v1.length;
    long v8;
    v8 = 0l;
    while (while_method_1(v7, v8)){
        unsigned long long v10;
        v10 = (unsigned long long)v8;
        unsigned long long v11;
        v11 = v10 * 4ull;
        unsigned long long v12;
        v12 = 4ull + v11;
        unsigned char * v13;
        v13 = (unsigned char *)(v0+v12);
        Union3 v14;
        v14 = v1[v8];
        f_29(v13, v14);
        v8 += 1l ;
    }
    long v15;
    v15 = v2.length;
    f_31(v0, v15);
    long v16;
    v16 = v2.length;
    long v17;
    v17 = 0l;
    while (while_method_1(v16, v17)){
        unsigned long long v19;
        v19 = (unsigned long long)v17;
        unsigned long long v20;
        v20 = v19 * 32ull;
        unsigned long long v21;
        v21 = 32ull + v20;
        unsigned char * v22;
        v22 = (unsigned char *)(v0+v21);
        Union4 v23;
        v23 = v2[v17];
        f_32(v22, v23);
        v17 += 1l ;
    }
    long v24;
    v24 = v3.tag;
    f_37(v0, v24);
    unsigned char * v25;
    v25 = (unsigned char *)(v0+1072ull);
    switch (v3.tag) {
        case 0: { // None
            f_30(v25);
            break;
        }
        case 1: { // Some
            Union6 v26 = v3.case1.v0;
            f_38(v25, v26);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    long v27;
    v27 = 0l;
    while (while_method_0(v27)){
        unsigned long long v29;
        v29 = (unsigned long long)v27;
        unsigned long long v30;
        v30 = v29 * 4ull;
        unsigned long long v31;
        v31 = 1136ull + v30;
        unsigned char * v32;
        v32 = (unsigned char *)(v0+v31);
        Union2 v33;
        v33 = v4[v27];
        f_42(v32, v33);
        v27 += 1l ;
    }
    long v34;
    v34 = v5.tag;
    f_43(v0, v34);
    unsigned char * v35;
    v35 = (unsigned char *)(v0+1152ull);
    switch (v5.tag) {
        case 0: { // GameNotStarted
            return f_30(v35);
            break;
        }
        case 1: { // GameOver
            Union7 v36 = v5.case1.v0; bool v37 = v5.case1.v1; static_array<Union3,2l> v38 = v5.case1.v2; long v39 = v5.case1.v3; static_array<long,2l> v40 = v5.case1.v4; long v41 = v5.case1.v5;
            return f_39(v35, v36, v37, v38, v39, v40, v41);
            break;
        }
        case 2: { // WaitingForActionFromPlayerId
            Union7 v42 = v5.case2.v0; bool v43 = v5.case2.v1; static_array<Union3,2l> v44 = v5.case2.v2; long v45 = v5.case2.v3; static_array<long,2l> v46 = v5.case2.v4; long v47 = v5.case2.v5;
            return f_39(v35, v42, v43, v44, v45, v46, v47);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
__device__ void f_45(unsigned char * v0, long v1){
    long * v2;
    v2 = (long *)(v0+1048ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_44(unsigned char * v0, static_array_list<Union4,32l> v1, static_array<Union2,2l> v2, Union8 v3){
    long v4;
    v4 = v1.length;
    f_28(v0, v4);
    long v5;
    v5 = v1.length;
    long v6;
    v6 = 0l;
    while (while_method_1(v5, v6)){
        unsigned long long v8;
        v8 = (unsigned long long)v6;
        unsigned long long v9;
        v9 = v8 * 32ull;
        unsigned long long v10;
        v10 = 16ull + v9;
        unsigned char * v11;
        v11 = (unsigned char *)(v0+v10);
        Union4 v12;
        v12 = v1[v6];
        f_32(v11, v12);
        v6 += 1l ;
    }
    long v13;
    v13 = 0l;
    while (while_method_0(v13)){
        unsigned long long v15;
        v15 = (unsigned long long)v13;
        unsigned long long v16;
        v16 = v15 * 4ull;
        unsigned long long v17;
        v17 = 1040ull + v16;
        unsigned char * v18;
        v18 = (unsigned char *)(v0+v17);
        Union2 v19;
        v19 = v2[v13];
        f_42(v18, v19);
        v13 += 1l ;
    }
    long v20;
    v20 = v3.tag;
    f_45(v0, v20);
    unsigned char * v21;
    v21 = (unsigned char *)(v0+1056ull);
    switch (v3.tag) {
        case 0: { // GameNotStarted
            return f_30(v21);
            break;
        }
        case 1: { // GameOver
            Union7 v22 = v3.case1.v0; bool v23 = v3.case1.v1; static_array<Union3,2l> v24 = v3.case1.v2; long v25 = v3.case1.v3; static_array<long,2l> v26 = v3.case1.v4; long v27 = v3.case1.v5;
            return f_39(v21, v22, v23, v24, v25, v26, v27);
            break;
        }
        case 2: { // WaitingForActionFromPlayerId
            Union7 v28 = v3.case2.v0; bool v29 = v3.case2.v1; static_array<Union3,2l> v30 = v3.case2.v2; long v31 = v3.case2.v3; static_array<long,2l> v32 = v3.case2.v4; long v33 = v3.case2.v5;
            return f_39(v21, v28, v29, v30, v31, v32, v33);
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1, unsigned char * v2) {
    long v3;
    v3 = threadIdx.x;
    long v4;
    v4 = blockIdx.x;
    long v5;
    v5 = v4 * 32l;
    long v6;
    v6 = v3 + v5;
    bool v7;
    v7 = v6 == 0l;
    if (v7){
        Union0 v8;
        v8 = f_0(v1);
        static_array_list<Union3,6l> v9; static_array_list<Union4,32l> v10; Union5 v11; static_array<Union2,2l> v12; Union8 v13;
        Tuple0 tmp10 = f_6(v0);
        v9 = tmp10.v0; v10 = tmp10.v1; v11 = tmp10.v2; v12 = tmp10.v3; v13 = tmp10.v4;
        static_array_list<Union3,6l> & v14 = v9;
        static_array_list<Union4,32l> & v15 = v10;
        Union5 v79; static_array<Union2,2l> v80; Union8 v81;
        switch (v8.tag) {
            case 0: { // ActionSelected
                Union1 v49 = v8.case0.v0;
                switch (v11.tag) {
                    case 0: { // None
                        v79 = v11; v80 = v12; v81 = v13;
                        break;
                    }
                    case 1: { // Some
                        Union6 v50 = v11.case1.v0;
                        switch (v50.tag) {
                            case 2: { // Round
                                Union7 v51 = v50.case2.v0; bool v52 = v50.case2.v1; static_array<Union3,2l> v53 = v50.case2.v2; long v54 = v50.case2.v3; static_array<long,2l> v55 = v50.case2.v4; long v56 = v50.case2.v5;
                                Union6 v57;
                                v57 = Union6{Union6_3{v51, v52, v53, v54, v55, v56, v49}};
                                Tuple6 tmp21 = play_loop_20(v11, v12, v13, v14, v15, v57);
                                v79 = tmp21.v0; v80 = tmp21.v1; v81 = tmp21.v2;
                                break;
                            }
                            default: {
                                printf("%s\n", "Unexpected game node in ActionSelected.");
                                asm("exit;");
                            }
                        }
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                    }
                }
                break;
            }
            case 1: { // PlayerChanged
                static_array<Union2,2l> v48 = v8.case1.v0;
                v79 = v11; v80 = v48; v81 = v13;
                break;
            }
            case 2: { // StartGame
                static_array<Union2,2l> v16;
                Union2 v17;
                v17 = Union2{Union2_0{}};
                v16[0l] = v17;
                Union2 v18;
                v18 = Union2{Union2_1{}};
                v16[1l] = v18;
                static_array_list<Union3,6l> v19;
                v19 = static_array_list<Union3,6l>{};
                v19.unsafe_set_length(6l);
                Union3 v20;
                v20 = Union3{Union3_1{}};
                v19[0l] = v20;
                Union3 v21;
                v21 = Union3{Union3_1{}};
                v19[1l] = v21;
                Union3 v22;
                v22 = Union3{Union3_2{}};
                v19[2l] = v22;
                Union3 v23;
                v23 = Union3{Union3_2{}};
                v19[3l] = v23;
                Union3 v24;
                v24 = Union3{Union3_0{}};
                v19[4l] = v24;
                Union3 v25;
                v25 = Union3{Union3_0{}};
                v19[5l] = v25;
                unsigned long long v26;
                v26 = clock64();
                curandStatePhilox4_32_10_t v27;
                curand_init(v26,0ull,0ull,&v27);
                long v28;
                v28 = v19.length;
                long v29;
                v29 = v28 - 1l;
                long v30;
                v30 = 0l;
                while (while_method_1(v29, v30)){
                    long v32;
                    v32 = v19.length;
                    long v33;
                    v33 = v32 - v30;
                    unsigned long v34;
                    v34 = (unsigned long)v33;
                    unsigned long v35;
                    v35 = loop_22(v34, v27);
                    unsigned long v36;
                    v36 = (unsigned long)v30;
                    unsigned long v37;
                    v37 = v35 + v36;
                    long v38;
                    v38 = (long)v37;
                    Union3 v39;
                    v39 = v19[v30];
                    Union3 v40;
                    v40 = v19[v38];
                    v19[v30] = v40;
                    v19[v38] = v39;
                    v30 += 1l ;
                }
                static_array_list<Union4,32l> v41;
                v41 = static_array_list<Union4,32l>{};
                v14 = v19;
                v15 = v41;
                Union5 v42;
                v42 = Union5{Union5_0{}};
                Union8 v43;
                v43 = Union8{Union8_0{}};
                Union6 v44;
                v44 = Union6{Union6_1{}};
                Tuple6 tmp22 = play_loop_20(v42, v16, v43, v14, v15, v44);
                v79 = tmp22.v0; v80 = tmp22.v1; v81 = tmp22.v2;
                break;
            }
            default: {
                assert("Invalid tag." && false);
            }
        }
        f_27(v0, v9, v10, v79, v80, v81);
        return f_44(v2, v10, v80, v81);
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
options.append('--diag-suppress=550,20012')
options.append('--dopt=on')
options.append('--restrict')
options.append('-I C:/Spiral_s_ML_Library/cpplib')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
import random
import collections
class US1_0(NamedTuple): # Call
    tag = 0
class US1_1(NamedTuple): # Fold
    tag = 1
class US1_2(NamedTuple): # Raise
    tag = 2
US1 = Union[US1_0, US1_1, US1_2]
class US0_0(NamedTuple): # ActionSelected
    v0 : US1
    tag = 0
class US0_1(NamedTuple): # PlayerChanged
    v0 : static_array
    tag = 1
class US0_2(NamedTuple): # StartGame
    tag = 2
US0 = Union[US0_0, US0_1, US0_2]
class US2_0(NamedTuple): # Computer
    tag = 0
class US2_1(NamedTuple): # Human
    tag = 1
US2 = Union[US2_0, US2_1]
class US6_0(NamedTuple): # Jack
    tag = 0
class US6_1(NamedTuple): # King
    tag = 1
class US6_2(NamedTuple): # Queen
    tag = 2
US6 = Union[US6_0, US6_1, US6_2]
class US5_0(NamedTuple): # None
    tag = 0
class US5_1(NamedTuple): # Some
    v0 : US6
    tag = 1
US5 = Union[US5_0, US5_1]
class US4_0(NamedTuple): # ChanceCommunityCard
    v0 : US5
    v1 : bool
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : i32
    tag = 0
class US4_1(NamedTuple): # ChanceInit
    tag = 1
class US4_2(NamedTuple): # Round
    v0 : US5
    v1 : bool
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : i32
    tag = 2
class US4_3(NamedTuple): # RoundWithAction
    v0 : US5
    v1 : bool
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : i32
    v6 : US1
    tag = 3
class US4_4(NamedTuple): # TerminalCall
    v0 : US5
    v1 : bool
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : i32
    tag = 4
class US4_5(NamedTuple): # TerminalFold
    v0 : US5
    v1 : bool
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : i32
    tag = 5
US4 = Union[US4_0, US4_1, US4_2, US4_3, US4_4, US4_5]
class US3_0(NamedTuple): # None
    tag = 0
class US3_1(NamedTuple): # Some
    v0 : US4
    tag = 1
US3 = Union[US3_0, US3_1]
class US7_0(NamedTuple): # GameNotStarted
    tag = 0
class US7_1(NamedTuple): # GameOver
    v0 : US5
    v1 : bool
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : i32
    tag = 1
class US7_2(NamedTuple): # WaitingForActionFromPlayerId
    v0 : US5
    v1 : bool
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : i32
    tag = 2
US7 = Union[US7_0, US7_1, US7_2]
class US8_0(NamedTuple): # CommunityCardIs
    v0 : US6
    tag = 0
class US8_1(NamedTuple): # PlayerAction
    v0 : i32
    v1 : US1
    tag = 1
class US8_2(NamedTuple): # PlayerGotCard
    v0 : i32
    v1 : US6
    tag = 2
class US8_3(NamedTuple): # Showdown
    v0 : static_array
    v1 : i32
    v2 : i32
    tag = 3
US8 = Union[US8_0, US8_1, US8_2, US8_3]
def Closure0():
    def inner(v0 : object, v1 : object) -> object:
        v2 = cp.empty(16,dtype=cp.uint8)
        v3 = cp.empty(1200,dtype=cp.uint8)
        v4 = cp.empty(1104,dtype=cp.uint8)
        v5 = method0(v0)
        v6, v7, v8, v9, v10 = method7(v1)
        method28(v2, v5)
        del v5
        method35(v3, v6, v7, v8, v9, v10)
        del v6, v7, v8, v9, v10
        v11 = 0
        v12 = raw_module.get_function(f"entry{v11}")
        del v11
        v12.max_dynamic_shared_size_bytes = 0 
        v12((1,),(32,),(v3, v2, v4),shared_mem=0)
        del v2, v12
        v13, v14, v15, v16, v17 = method49(v3)
        del v3
        v18, v19, v20 = method66(v4)
        del v4
        return method68(v13, v14, v15, v16, v17, v18, v19, v20)
    return inner
def Closure1():
    def inner() -> object:
        v0 = static_array(2)
        v1 = US2_0()
        v0[0] = v1
        del v1
        v2 = US2_1()
        v0[1] = v2
        del v2
        v3 = static_array_list(6)
        v3.unsafe_set_length(6)
        v4 = US6_1()
        v3[0] = v4
        del v4
        v5 = US6_1()
        v3[1] = v5
        del v5
        v6 = US6_2()
        v3[2] = v6
        del v6
        v7 = US6_2()
        v3[3] = v7
        del v7
        v8 = US6_0()
        v3[4] = v8
        del v8
        v9 = US6_0()
        v3[5] = v9
        del v9
        v10 = v3.length
        v11 = v10 - 1
        del v10
        v12 = 0
        while method5(v11, v12):
            v14 = v3.length
            v15 = random.randrange(v12, v14)
            del v14
            v16 = v3[v12]
            v17 = v3[v15]
            v3[v12] = v17
            del v17
            v3[v15] = v16
            del v15, v16
            v12 += 1 
        del v11, v12
        v18 = static_array_list(32)
        v19 = US3_0()
        v20 = US7_0()
        return method95(v3, v18, v19, v0, v20)
    return inner
def method3(v0 : object) -> None:
    assert v0 == [], f'Expected an unit type. Got: {v0}'
    del v0
    return 
def method2(v0 : object) -> US1:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "Call" == v1
    if v4:
        del v1, v4
        method3(v2)
        del v2
        return US1_0()
    else:
        del v4
        v7 = "Fold" == v1
        if v7:
            del v1, v7
            method3(v2)
            del v2
            return US1_1()
        else:
            del v7
            v10 = "Raise" == v1
            if v10:
                del v1, v10
                method3(v2)
                del v2
                return US1_2()
            else:
                del v2, v10
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method5(v0 : i32, v1 : i32) -> bool:
    v2 = v1 < v0
    del v0, v1
    return v2
def method6(v0 : object) -> US2:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "Computer" == v1
    if v4:
        del v1, v4
        method3(v2)
        del v2
        return US2_0()
    else:
        del v4
        v7 = "Human" == v1
        if v7:
            del v1, v7
            method3(v2)
            del v2
            return US2_1()
        else:
            del v2, v7
            raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
            del v1
            raise Exception("Error")
def method4(v0 : object) -> static_array:
    assert isinstance(v0,list), f'The object needs to be a Python list. Got: {v0}'
    v1 = len(v0) # type: ignore
    v2 = 2 == v1
    v3 = v2 == False
    if v3:
        v4 = "The type level dimension has to equal the value passed at runtime into create."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = static_array(2)
    v6 = 0
    while method5(v1, v6):
        v8 = v0[v6]
        v9 = method6(v8)
        del v8
        v5[v6] = v9
        del v9
        v6 += 1 
    del v0, v1, v6
    return v5
def method1(v0 : object) -> US0:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "ActionSelected" == v1
    if v4:
        del v1, v4
        v5 = method2(v2)
        del v2
        return US0_0(v5)
    else:
        del v4
        v8 = "PlayerChanged" == v1
        if v8:
            del v1, v8
            v9 = method4(v2)
            del v2
            return US0_1(v9)
        else:
            del v8
            v12 = "StartGame" == v1
            if v12:
                del v1, v12
                method3(v2)
                del v2
                return US0_2()
            else:
                del v2, v12
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method0(v0 : object) -> US0:
    return method1(v0)
def method11(v0 : object) -> US6:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "Jack" == v1
    if v4:
        del v1, v4
        method3(v2)
        del v2
        return US6_0()
    else:
        del v4
        v7 = "King" == v1
        if v7:
            del v1, v7
            method3(v2)
            del v2
            return US6_1()
        else:
            del v7
            v10 = "Queen" == v1
            if v10:
                del v1, v10
                method3(v2)
                del v2
                return US6_2()
            else:
                del v2, v10
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method10(v0 : object) -> static_array_list:
    v1 = len(v0) # type: ignore
    assert (6 >= v1), f'The length of the original object has to be greater than or equal to the static array dimension.\nExpected: 6\nGot: {v1} '
    del v1
    assert isinstance(v0,list), f'The object needs to be a Python list. Got: {v0}'
    v2 = len(v0) # type: ignore
    v3 = 6 >= v2
    v4 = v3 == False
    if v4:
        v5 = "The type level dimension has to equal the value passed at runtime into create."
        assert v3, v5
        del v5
    else:
        pass
    del v3, v4
    v6 = static_array_list(6)
    v6.unsafe_set_length(v2)
    v7 = 0
    while method5(v2, v7):
        v9 = v0[v7]
        v10 = method11(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v2, v7
    return v6
def method15(v0 : object) -> i32:
    assert isinstance(v0,i32), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method14(v0 : object) -> Tuple[i32, US1]:
    v1 = v0[0] # type: ignore
    v2 = method15(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method2(v3)
    del v3
    return v2, v4
def method16(v0 : object) -> Tuple[i32, US6]:
    v1 = v0[0] # type: ignore
    v2 = method15(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method11(v3)
    del v3
    return v2, v4
def method18(v0 : object) -> static_array:
    assert isinstance(v0,list), f'The object needs to be a Python list. Got: {v0}'
    v1 = len(v0) # type: ignore
    v2 = 2 == v1
    v3 = v2 == False
    if v3:
        v4 = "The type level dimension has to equal the value passed at runtime into create."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = static_array(2)
    v6 = 0
    while method5(v1, v6):
        v8 = v0[v6]
        v9 = method11(v8)
        del v8
        v5[v6] = v9
        del v9
        v6 += 1 
    del v0, v1, v6
    return v5
def method17(v0 : object) -> Tuple[static_array, i32, i32]:
    v1 = v0["cards_shown"] # type: ignore
    v2 = method18(v1)
    del v1
    v3 = v0["chips_won"] # type: ignore
    v4 = method15(v3)
    del v3
    v5 = v0["winner_id"] # type: ignore
    del v0
    v6 = method15(v5)
    del v5
    return v2, v4, v6
def method13(v0 : object) -> US8:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "CommunityCardIs" == v1
    if v4:
        del v1, v4
        v5 = method11(v2)
        del v2
        return US8_0(v5)
    else:
        del v4
        v8 = "PlayerAction" == v1
        if v8:
            del v1, v8
            v9, v10 = method14(v2)
            del v2
            return US8_1(v9, v10)
        else:
            del v8
            v13 = "PlayerGotCard" == v1
            if v13:
                del v1, v13
                v14, v15 = method16(v2)
                del v2
                return US8_2(v14, v15)
            else:
                del v13
                v18 = "Showdown" == v1
                if v18:
                    del v1, v18
                    v19, v20, v21 = method17(v2)
                    del v2
                    return US8_3(v19, v20, v21)
                else:
                    del v2, v18
                    raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                    del v1
                    raise Exception("Error")
def method12(v0 : object) -> static_array_list:
    v1 = len(v0) # type: ignore
    assert (32 >= v1), f'The length of the original object has to be greater than or equal to the static array dimension.\nExpected: 32\nGot: {v1} '
    del v1
    assert isinstance(v0,list), f'The object needs to be a Python list. Got: {v0}'
    v2 = len(v0) # type: ignore
    v3 = 32 >= v2
    v4 = v3 == False
    if v4:
        v5 = "The type level dimension has to equal the value passed at runtime into create."
        assert v3, v5
        del v5
    else:
        pass
    del v3, v4
    v6 = static_array_list(32)
    v6.unsafe_set_length(v2)
    v7 = 0
    while method5(v2, v7):
        v9 = v0[v7]
        v10 = method13(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v2, v7
    return v6
def method9(v0 : object) -> Tuple[static_array_list, static_array_list]:
    v1 = v0["deck"] # type: ignore
    v2 = method10(v1)
    del v1
    v3 = v0["messages"] # type: ignore
    del v0
    v4 = method12(v3)
    del v3
    return v2, v4
def method23(v0 : object) -> US5:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "None" == v1
    if v4:
        del v1, v4
        method3(v2)
        del v2
        return US5_0()
    else:
        del v4
        v7 = "Some" == v1
        if v7:
            del v1, v7
            v8 = method11(v2)
            del v2
            return US5_1(v8)
        else:
            del v2, v7
            raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
            del v1
            raise Exception("Error")
def method24(v0 : object) -> bool:
    assert isinstance(v0,bool), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method25(v0 : object) -> static_array:
    assert isinstance(v0,list), f'The object needs to be a Python list. Got: {v0}'
    v1 = len(v0) # type: ignore
    v2 = 2 == v1
    v3 = v2 == False
    if v3:
        v4 = "The type level dimension has to equal the value passed at runtime into create."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v5 = static_array(2)
    v6 = 0
    while method5(v1, v6):
        v8 = v0[v6]
        v9 = method15(v8)
        del v8
        v5[v6] = v9
        del v9
        v6 += 1 
    del v0, v1, v6
    return v5
def method22(v0 : object) -> Tuple[US5, bool, static_array, i32, static_array, i32]:
    v1 = v0["community_card"] # type: ignore
    v2 = method23(v1)
    del v1
    v3 = v0["is_button_s_first_move"] # type: ignore
    v4 = method24(v3)
    del v3
    v5 = v0["pl_card"] # type: ignore
    v6 = method18(v5)
    del v5
    v7 = v0["player_turn"] # type: ignore
    v8 = method15(v7)
    del v7
    v9 = v0["pot"] # type: ignore
    v10 = method25(v9)
    del v9
    v11 = v0["raises_left"] # type: ignore
    del v0
    v12 = method15(v11)
    del v11
    return v2, v4, v6, v8, v10, v12
def method26(v0 : object) -> Tuple[US5, bool, static_array, i32, static_array, i32, US1]:
    v1 = v0[0] # type: ignore
    v2, v3, v4, v5, v6, v7 = method22(v1)
    del v1
    v8 = v0[1] # type: ignore
    del v0
    v9 = method2(v8)
    del v8
    return v2, v3, v4, v5, v6, v7, v9
def method21(v0 : object) -> US4:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "ChanceCommunityCard" == v1
    if v4:
        del v1, v4
        v5, v6, v7, v8, v9, v10 = method22(v2)
        del v2
        return US4_0(v5, v6, v7, v8, v9, v10)
    else:
        del v4
        v13 = "ChanceInit" == v1
        if v13:
            del v1, v13
            method3(v2)
            del v2
            return US4_1()
        else:
            del v13
            v16 = "Round" == v1
            if v16:
                del v1, v16
                v17, v18, v19, v20, v21, v22 = method22(v2)
                del v2
                return US4_2(v17, v18, v19, v20, v21, v22)
            else:
                del v16
                v25 = "RoundWithAction" == v1
                if v25:
                    del v1, v25
                    v26, v27, v28, v29, v30, v31, v32 = method26(v2)
                    del v2
                    return US4_3(v26, v27, v28, v29, v30, v31, v32)
                else:
                    del v25
                    v35 = "TerminalCall" == v1
                    if v35:
                        del v1, v35
                        v36, v37, v38, v39, v40, v41 = method22(v2)
                        del v2
                        return US4_4(v36, v37, v38, v39, v40, v41)
                    else:
                        del v35
                        v44 = "TerminalFold" == v1
                        if v44:
                            del v1, v44
                            v45, v46, v47, v48, v49, v50 = method22(v2)
                            del v2
                            return US4_5(v45, v46, v47, v48, v49, v50)
                        else:
                            del v2, v44
                            raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                            del v1
                            raise Exception("Error")
def method20(v0 : object) -> US3:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "None" == v1
    if v4:
        del v1, v4
        method3(v2)
        del v2
        return US3_0()
    else:
        del v4
        v7 = "Some" == v1
        if v7:
            del v1, v7
            v8 = method21(v2)
            del v2
            return US3_1(v8)
        else:
            del v2, v7
            raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
            del v1
            raise Exception("Error")
def method27(v0 : object) -> US7:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v4 = "GameNotStarted" == v1
    if v4:
        del v1, v4
        method3(v2)
        del v2
        return US7_0()
    else:
        del v4
        v7 = "GameOver" == v1
        if v7:
            del v1, v7
            v8, v9, v10, v11, v12, v13 = method22(v2)
            del v2
            return US7_1(v8, v9, v10, v11, v12, v13)
        else:
            del v7
            v16 = "WaitingForActionFromPlayerId" == v1
            if v16:
                del v1, v16
                v17, v18, v19, v20, v21, v22 = method22(v2)
                del v2
                return US7_2(v17, v18, v19, v20, v21, v22)
            else:
                del v2, v16
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method19(v0 : object) -> Tuple[US3, static_array, US7]:
    v1 = v0["game"] # type: ignore
    v2 = method20(v1)
    del v1
    v3 = v0["pl_type"] # type: ignore
    v4 = method4(v3)
    del v3
    v5 = v0["ui_game_state"] # type: ignore
    del v0
    v6 = method27(v5)
    del v5
    return v2, v4, v6
def method8(v0 : object) -> Tuple[static_array_list, static_array_list, US3, static_array, US7]:
    v1 = v0["large"] # type: ignore
    v2, v3 = method9(v1)
    del v1
    v4 = v0["small"] # type: ignore
    del v0
    v5, v6, v7 = method19(v4)
    del v4
    return v2, v3, v5, v6, v7
def method7(v0 : object) -> Tuple[static_array_list, static_array_list, US3, static_array, US7]:
    return method8(v0)
def method29(v0 : cp.ndarray, v1 : i32) -> None:
    v2 = v0[0:].view(cp.int32)
    del v0
    v2[0] = v1
    del v1, v2
    return 
def method31(v0 : cp.ndarray) -> None:
    del v0
    return 
def method30(v0 : cp.ndarray, v1 : US1) -> None:
    v2 = v1.tag
    method29(v0, v2)
    del v2
    v3 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US1_0(): # Call
            del v1
            return method31(v3)
        case US1_1(): # Fold
            del v1
            return method31(v3)
        case US1_2(): # Raise
            del v1
            return method31(v3)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method33(v0 : i32) -> bool:
    v1 = v0 < 2
    del v0
    return v1
def method34(v0 : cp.ndarray, v1 : US2) -> None:
    v2 = v1.tag
    method29(v0, v2)
    del v2
    v3 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US2_0(): # Computer
            del v1
            return method31(v3)
        case US2_1(): # Human
            del v1
            return method31(v3)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method32(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method33(v2):
        v4 = u64(v2)
        v5 = v4 * 4
        del v4
        v6 = v0[v5:].view(cp.uint8)
        del v5
        v7 = v1[v2]
        method34(v6, v7)
        del v6, v7
        v2 += 1 
    del v0, v1, v2
    return 
def method28(v0 : cp.ndarray, v1 : US0) -> None:
    v2 = v1.tag
    method29(v0, v2)
    del v2
    v3 = v0[8:].view(cp.uint8)
    del v0
    match v1:
        case US0_0(v4): # ActionSelected
            del v1
            return method30(v3, v4)
        case US0_1(v5): # PlayerChanged
            del v1
            return method32(v3, v5)
        case US0_2(): # StartGame
            del v1
            return method31(v3)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method36(v0 : cp.ndarray, v1 : US6) -> None:
    v2 = v1.tag
    method29(v0, v2)
    del v2
    v3 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US6_0(): # Jack
            del v1
            return method31(v3)
        case US6_1(): # King
            del v1
            return method31(v3)
        case US6_2(): # Queen
            del v1
            return method31(v3)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method37(v0 : cp.ndarray, v1 : i32) -> None:
    v2 = v0[28:].view(cp.int32)
    del v0
    v2[0] = v1
    del v1, v2
    return 
def method40(v0 : cp.ndarray, v1 : i32) -> None:
    v2 = v0[4:].view(cp.int32)
    del v0
    v2[0] = v1
    del v1, v2
    return 
def method39(v0 : cp.ndarray, v1 : i32, v2 : US1) -> None:
    v3 = v0[0:].view(cp.int32)
    v3[0] = v1
    del v1, v3
    v4 = v2.tag
    method40(v0, v4)
    del v4
    v5 = v0[8:].view(cp.uint8)
    del v0
    match v2:
        case US1_0(): # Call
            del v2
            return method31(v5)
        case US1_1(): # Fold
            del v2
            return method31(v5)
        case US1_2(): # Raise
            del v2
            return method31(v5)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method41(v0 : cp.ndarray, v1 : i32, v2 : US6) -> None:
    v3 = v0[0:].view(cp.int32)
    v3[0] = v1
    del v1, v3
    v4 = v2.tag
    method40(v0, v4)
    del v4
    v5 = v0[8:].view(cp.uint8)
    del v0
    match v2:
        case US6_0(): # Jack
            del v2
            return method31(v5)
        case US6_1(): # King
            del v2
            return method31(v5)
        case US6_2(): # Queen
            del v2
            return method31(v5)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method42(v0 : cp.ndarray, v1 : static_array, v2 : i32, v3 : i32) -> None:
    v4 = 0
    while method33(v4):
        v6 = u64(v4)
        v7 = v6 * 4
        del v6
        v8 = v0[v7:].view(cp.uint8)
        del v7
        v9 = v1[v4]
        method36(v8, v9)
        del v8, v9
        v4 += 1 
    del v1, v4
    v10 = v0[8:].view(cp.int32)
    v10[0] = v2
    del v2, v10
    v11 = v0[12:].view(cp.int32)
    del v0
    v11[0] = v3
    del v3, v11
    return 
def method38(v0 : cp.ndarray, v1 : US8) -> None:
    v2 = v1.tag
    method29(v0, v2)
    del v2
    v3 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US8_0(v4): # CommunityCardIs
            del v1
            return method36(v3, v4)
        case US8_1(v5, v6): # PlayerAction
            del v1
            return method39(v3, v5, v6)
        case US8_2(v7, v8): # PlayerGotCard
            del v1
            return method41(v3, v7, v8)
        case US8_3(v9, v10, v11): # Showdown
            del v1
            return method42(v3, v9, v10, v11)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method43(v0 : cp.ndarray, v1 : i32) -> None:
    v2 = v0[1056:].view(cp.int32)
    del v0
    v2[0] = v1
    del v1, v2
    return 
def method45(v0 : cp.ndarray, v1 : US5, v2 : bool, v3 : static_array, v4 : i32, v5 : static_array, v6 : i32) -> None:
    v7 = v1.tag
    method29(v0, v7)
    del v7
    v8 = v0[4:].view(cp.uint8)
    match v1:
        case US5_0(): # None
            method31(v8)
        case US5_1(v9): # Some
            method36(v8, v9)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v1, v8
    v10 = v0[8:].view(cp.bool_)
    v10[0] = v2
    del v2, v10
    v11 = 0
    while method33(v11):
        v13 = u64(v11)
        v14 = v13 * 4
        del v13
        v15 = 12 + v14
        del v14
        v16 = v0[v15:].view(cp.uint8)
        del v15
        v17 = v3[v11]
        method36(v16, v17)
        del v16, v17
        v11 += 1 
    del v3, v11
    v18 = v0[20:].view(cp.int32)
    v18[0] = v4
    del v4, v18
    v19 = 0
    while method33(v19):
        v21 = u64(v19)
        v22 = v21 * 4
        del v21
        v23 = 24 + v22
        del v22
        v24 = v0[v23:].view(cp.uint8)
        del v23
        v25 = v5[v19]
        method29(v24, v25)
        del v24, v25
        v19 += 1 
    del v5, v19
    v26 = v0[32:].view(cp.int32)
    del v0
    v26[0] = v6
    del v6, v26
    return 
def method47(v0 : cp.ndarray, v1 : i32) -> None:
    v2 = v0[36:].view(cp.int32)
    del v0
    v2[0] = v1
    del v1, v2
    return 
def method46(v0 : cp.ndarray, v1 : US5, v2 : bool, v3 : static_array, v4 : i32, v5 : static_array, v6 : i32, v7 : US1) -> None:
    v8 = v1.tag
    method29(v0, v8)
    del v8
    v9 = v0[4:].view(cp.uint8)
    match v1:
        case US5_0(): # None
            method31(v9)
        case US5_1(v10): # Some
            method36(v9, v10)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v1, v9
    v11 = v0[8:].view(cp.bool_)
    v11[0] = v2
    del v2, v11
    v12 = 0
    while method33(v12):
        v14 = u64(v12)
        v15 = v14 * 4
        del v14
        v16 = 12 + v15
        del v15
        v17 = v0[v16:].view(cp.uint8)
        del v16
        v18 = v3[v12]
        method36(v17, v18)
        del v17, v18
        v12 += 1 
    del v3, v12
    v19 = v0[20:].view(cp.int32)
    v19[0] = v4
    del v4, v19
    v20 = 0
    while method33(v20):
        v22 = u64(v20)
        v23 = v22 * 4
        del v22
        v24 = 24 + v23
        del v23
        v25 = v0[v24:].view(cp.uint8)
        del v24
        v26 = v5[v20]
        method29(v25, v26)
        del v25, v26
        v20 += 1 
    del v5, v20
    v27 = v0[32:].view(cp.int32)
    v27[0] = v6
    del v6, v27
    v28 = v7.tag
    method47(v0, v28)
    del v28
    v29 = v0[40:].view(cp.uint8)
    del v0
    match v7:
        case US1_0(): # Call
            del v7
            return method31(v29)
        case US1_1(): # Fold
            del v7
            return method31(v29)
        case US1_2(): # Raise
            del v7
            return method31(v29)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method44(v0 : cp.ndarray, v1 : US4) -> None:
    v2 = v1.tag
    method29(v0, v2)
    del v2
    v3 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US4_0(v4, v5, v6, v7, v8, v9): # ChanceCommunityCard
            del v1
            return method45(v3, v4, v5, v6, v7, v8, v9)
        case US4_1(): # ChanceInit
            del v1
            return method31(v3)
        case US4_2(v10, v11, v12, v13, v14, v15): # Round
            del v1
            return method45(v3, v10, v11, v12, v13, v14, v15)
        case US4_3(v16, v17, v18, v19, v20, v21, v22): # RoundWithAction
            del v1
            return method46(v3, v16, v17, v18, v19, v20, v21, v22)
        case US4_4(v23, v24, v25, v26, v27, v28): # TerminalCall
            del v1
            return method45(v3, v23, v24, v25, v26, v27, v28)
        case US4_5(v29, v30, v31, v32, v33, v34): # TerminalFold
            del v1
            return method45(v3, v29, v30, v31, v32, v33, v34)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method48(v0 : cp.ndarray, v1 : i32) -> None:
    v2 = v0[1144:].view(cp.int32)
    del v0
    v2[0] = v1
    del v1, v2
    return 
def method35(v0 : cp.ndarray, v1 : static_array_list, v2 : static_array_list, v3 : US3, v4 : static_array, v5 : US7) -> None:
    v6 = v1.length
    method29(v0, v6)
    del v6
    v7 = v1.length
    v8 = 0
    while method5(v7, v8):
        v10 = u64(v8)
        v11 = v10 * 4
        del v10
        v12 = 4 + v11
        del v11
        v13 = v0[v12:].view(cp.uint8)
        del v12
        v14 = v1[v8]
        method36(v13, v14)
        del v13, v14
        v8 += 1 
    del v1, v7, v8
    v15 = v2.length
    method37(v0, v15)
    del v15
    v16 = v2.length
    v17 = 0
    while method5(v16, v17):
        v19 = u64(v17)
        v20 = v19 * 32
        del v19
        v21 = 32 + v20
        del v20
        v22 = v0[v21:].view(cp.uint8)
        del v21
        v23 = v2[v17]
        method38(v22, v23)
        del v22, v23
        v17 += 1 
    del v2, v16, v17
    v24 = v3.tag
    method43(v0, v24)
    del v24
    v25 = v0[1072:].view(cp.uint8)
    match v3:
        case US3_0(): # None
            method31(v25)
        case US3_1(v26): # Some
            method44(v25, v26)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v3, v25
    v27 = 0
    while method33(v27):
        v29 = u64(v27)
        v30 = v29 * 4
        del v29
        v31 = 1136 + v30
        del v30
        v32 = v0[v31:].view(cp.uint8)
        del v31
        v33 = v4[v27]
        method34(v32, v33)
        del v32, v33
        v27 += 1 
    del v4, v27
    v34 = v5.tag
    method48(v0, v34)
    del v34
    v35 = v0[1152:].view(cp.uint8)
    del v0
    match v5:
        case US7_0(): # GameNotStarted
            del v5
            return method31(v35)
        case US7_1(v36, v37, v38, v39, v40, v41): # GameOver
            del v5
            return method45(v35, v36, v37, v38, v39, v40, v41)
        case US7_2(v42, v43, v44, v45, v46, v47): # WaitingForActionFromPlayerId
            del v5
            return method45(v35, v42, v43, v44, v45, v46, v47)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method50(v0 : cp.ndarray) -> i32:
    v1 = v0[0:].view(cp.int32)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method52(v0 : cp.ndarray) -> None:
    del v0
    return 
def method51(v0 : cp.ndarray) -> US6:
    v1 = method50(v0)
    v2 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method52(v2)
        del v2
        return US6_0()
    elif v1 == 1:
        del v1
        method52(v2)
        del v2
        return US6_1()
    elif v1 == 2:
        del v1
        method52(v2)
        del v2
        return US6_2()
    else:
        del v1, v2
        raise Exception("Invalid tag.")
def method53(v0 : cp.ndarray) -> i32:
    v1 = v0[28:].view(cp.int32)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method56(v0 : cp.ndarray) -> i32:
    v1 = v0[4:].view(cp.int32)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method55(v0 : cp.ndarray) -> Tuple[i32, US1]:
    v1 = v0[0:].view(cp.int32)
    v2 = v1[0].item()
    del v1
    v3 = method56(v0)
    v4 = v0[8:].view(cp.uint8)
    del v0
    if v3 == 0:
        method52(v4)
        v9 = US1_0()
    elif v3 == 1:
        method52(v4)
        v9 = US1_1()
    elif v3 == 2:
        method52(v4)
        v9 = US1_2()
    else:
        raise Exception("Invalid tag.")
    del v3, v4
    return v2, v9
def method57(v0 : cp.ndarray) -> Tuple[i32, US6]:
    v1 = v0[0:].view(cp.int32)
    v2 = v1[0].item()
    del v1
    v3 = method56(v0)
    v4 = v0[8:].view(cp.uint8)
    del v0
    if v3 == 0:
        method52(v4)
        v9 = US6_0()
    elif v3 == 1:
        method52(v4)
        v9 = US6_1()
    elif v3 == 2:
        method52(v4)
        v9 = US6_2()
    else:
        raise Exception("Invalid tag.")
    del v3, v4
    return v2, v9
def method58(v0 : cp.ndarray) -> Tuple[static_array, i32, i32]:
    v1 = static_array(2)
    v2 = 0
    while method33(v2):
        v4 = u64(v2)
        v5 = v4 * 4
        del v4
        v6 = v0[v5:].view(cp.uint8)
        del v5
        v7 = method51(v6)
        del v6
        v1[v2] = v7
        del v7
        v2 += 1 
    del v2
    v8 = v0[8:].view(cp.int32)
    v9 = v8[0].item()
    del v8
    v10 = v0[12:].view(cp.int32)
    del v0
    v11 = v10[0].item()
    del v10
    return v1, v9, v11
def method54(v0 : cp.ndarray) -> US8:
    v1 = method50(v0)
    v2 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v4 = method51(v2)
        del v2
        return US8_0(v4)
    elif v1 == 1:
        del v1
        v6, v7 = method55(v2)
        del v2
        return US8_1(v6, v7)
    elif v1 == 2:
        del v1
        v9, v10 = method57(v2)
        del v2
        return US8_2(v9, v10)
    elif v1 == 3:
        del v1
        v12, v13, v14 = method58(v2)
        del v2
        return US8_3(v12, v13, v14)
    else:
        del v1, v2
        raise Exception("Invalid tag.")
def method59(v0 : cp.ndarray) -> i32:
    v1 = v0[1056:].view(cp.int32)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method61(v0 : cp.ndarray) -> Tuple[US5, bool, static_array, i32, static_array, i32]:
    v1 = method50(v0)
    v2 = v0[4:].view(cp.uint8)
    if v1 == 0:
        method52(v2)
        v7 = US5_0()
    elif v1 == 1:
        v5 = method51(v2)
        v7 = US5_1(v5)
    else:
        raise Exception("Invalid tag.")
    del v1, v2
    v8 = v0[8:].view(cp.bool_)
    v9 = v8[0].item()
    del v8
    v10 = static_array(2)
    v11 = 0
    while method33(v11):
        v13 = u64(v11)
        v14 = v13 * 4
        del v13
        v15 = 12 + v14
        del v14
        v16 = v0[v15:].view(cp.uint8)
        del v15
        v17 = method51(v16)
        del v16
        v10[v11] = v17
        del v17
        v11 += 1 
    del v11
    v18 = v0[20:].view(cp.int32)
    v19 = v18[0].item()
    del v18
    v20 = static_array(2)
    v21 = 0
    while method33(v21):
        v23 = u64(v21)
        v24 = v23 * 4
        del v23
        v25 = 24 + v24
        del v24
        v26 = v0[v25:].view(cp.uint8)
        del v25
        v27 = method50(v26)
        del v26
        v20[v21] = v27
        del v27
        v21 += 1 
    del v21
    v28 = v0[32:].view(cp.int32)
    del v0
    v29 = v28[0].item()
    del v28
    return v7, v9, v10, v19, v20, v29
def method63(v0 : cp.ndarray) -> i32:
    v1 = v0[36:].view(cp.int32)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method62(v0 : cp.ndarray) -> Tuple[US5, bool, static_array, i32, static_array, i32, US1]:
    v1 = method50(v0)
    v2 = v0[4:].view(cp.uint8)
    if v1 == 0:
        method52(v2)
        v7 = US5_0()
    elif v1 == 1:
        v5 = method51(v2)
        v7 = US5_1(v5)
    else:
        raise Exception("Invalid tag.")
    del v1, v2
    v8 = v0[8:].view(cp.bool_)
    v9 = v8[0].item()
    del v8
    v10 = static_array(2)
    v11 = 0
    while method33(v11):
        v13 = u64(v11)
        v14 = v13 * 4
        del v13
        v15 = 12 + v14
        del v14
        v16 = v0[v15:].view(cp.uint8)
        del v15
        v17 = method51(v16)
        del v16
        v10[v11] = v17
        del v17
        v11 += 1 
    del v11
    v18 = v0[20:].view(cp.int32)
    v19 = v18[0].item()
    del v18
    v20 = static_array(2)
    v21 = 0
    while method33(v21):
        v23 = u64(v21)
        v24 = v23 * 4
        del v23
        v25 = 24 + v24
        del v24
        v26 = v0[v25:].view(cp.uint8)
        del v25
        v27 = method50(v26)
        del v26
        v20[v21] = v27
        del v27
        v21 += 1 
    del v21
    v28 = v0[32:].view(cp.int32)
    v29 = v28[0].item()
    del v28
    v30 = method63(v0)
    v31 = v0[40:].view(cp.uint8)
    del v0
    if v30 == 0:
        method52(v31)
        v36 = US1_0()
    elif v30 == 1:
        method52(v31)
        v36 = US1_1()
    elif v30 == 2:
        method52(v31)
        v36 = US1_2()
    else:
        raise Exception("Invalid tag.")
    del v30, v31
    return v7, v9, v10, v19, v20, v29, v36
def method60(v0 : cp.ndarray) -> US4:
    v1 = method50(v0)
    v2 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v4, v5, v6, v7, v8, v9 = method61(v2)
        del v2
        return US4_0(v4, v5, v6, v7, v8, v9)
    elif v1 == 1:
        del v1
        method52(v2)
        del v2
        return US4_1()
    elif v1 == 2:
        del v1
        v12, v13, v14, v15, v16, v17 = method61(v2)
        del v2
        return US4_2(v12, v13, v14, v15, v16, v17)
    elif v1 == 3:
        del v1
        v19, v20, v21, v22, v23, v24, v25 = method62(v2)
        del v2
        return US4_3(v19, v20, v21, v22, v23, v24, v25)
    elif v1 == 4:
        del v1
        v27, v28, v29, v30, v31, v32 = method61(v2)
        del v2
        return US4_4(v27, v28, v29, v30, v31, v32)
    elif v1 == 5:
        del v1
        v34, v35, v36, v37, v38, v39 = method61(v2)
        del v2
        return US4_5(v34, v35, v36, v37, v38, v39)
    else:
        del v1, v2
        raise Exception("Invalid tag.")
def method64(v0 : cp.ndarray) -> US2:
    v1 = method50(v0)
    v2 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method52(v2)
        del v2
        return US2_0()
    elif v1 == 1:
        del v1
        method52(v2)
        del v2
        return US2_1()
    else:
        del v1, v2
        raise Exception("Invalid tag.")
def method65(v0 : cp.ndarray) -> i32:
    v1 = v0[1144:].view(cp.int32)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method49(v0 : cp.ndarray) -> Tuple[static_array_list, static_array_list, US3, static_array, US7]:
    v1 = static_array_list(6)
    v2 = method50(v0)
    v1.unsafe_set_length(v2)
    del v2
    v3 = v1.length
    v4 = 0
    while method5(v3, v4):
        v6 = u64(v4)
        v7 = v6 * 4
        del v6
        v8 = 4 + v7
        del v7
        v9 = v0[v8:].view(cp.uint8)
        del v8
        v10 = method51(v9)
        del v9
        v1[v4] = v10
        del v10
        v4 += 1 
    del v3, v4
    v11 = static_array_list(32)
    v12 = method53(v0)
    v11.unsafe_set_length(v12)
    del v12
    v13 = v11.length
    v14 = 0
    while method5(v13, v14):
        v16 = u64(v14)
        v17 = v16 * 32
        del v16
        v18 = 32 + v17
        del v17
        v19 = v0[v18:].view(cp.uint8)
        del v18
        v20 = method54(v19)
        del v19
        v11[v14] = v20
        del v20
        v14 += 1 
    del v13, v14
    v21 = method59(v0)
    v22 = v0[1072:].view(cp.uint8)
    if v21 == 0:
        method52(v22)
        v27 = US3_0()
    elif v21 == 1:
        v25 = method60(v22)
        v27 = US3_1(v25)
    else:
        raise Exception("Invalid tag.")
    del v21, v22
    v28 = static_array(2)
    v29 = 0
    while method33(v29):
        v31 = u64(v29)
        v32 = v31 * 4
        del v31
        v33 = 1136 + v32
        del v32
        v34 = v0[v33:].view(cp.uint8)
        del v33
        v35 = method64(v34)
        del v34
        v28[v29] = v35
        del v35
        v29 += 1 
    del v29
    v36 = method65(v0)
    v37 = v0[1152:].view(cp.uint8)
    del v0
    if v36 == 0:
        method52(v37)
        v54 = US7_0()
    elif v36 == 1:
        v40, v41, v42, v43, v44, v45 = method61(v37)
        v54 = US7_1(v40, v41, v42, v43, v44, v45)
    elif v36 == 2:
        v47, v48, v49, v50, v51, v52 = method61(v37)
        v54 = US7_2(v47, v48, v49, v50, v51, v52)
    else:
        raise Exception("Invalid tag.")
    del v36, v37
    return v1, v11, v27, v28, v54
def method67(v0 : cp.ndarray) -> i32:
    v1 = v0[1048:].view(cp.int32)
    del v0
    v2 = v1[0].item()
    del v1
    return v2
def method66(v0 : cp.ndarray) -> Tuple[static_array_list, static_array, US7]:
    v1 = static_array_list(32)
    v2 = method50(v0)
    v1.unsafe_set_length(v2)
    del v2
    v3 = v1.length
    v4 = 0
    while method5(v3, v4):
        v6 = u64(v4)
        v7 = v6 * 32
        del v6
        v8 = 16 + v7
        del v7
        v9 = v0[v8:].view(cp.uint8)
        del v8
        v10 = method54(v9)
        del v9
        v1[v4] = v10
        del v10
        v4 += 1 
    del v3, v4
    v11 = static_array(2)
    v12 = 0
    while method33(v12):
        v14 = u64(v12)
        v15 = v14 * 4
        del v14
        v16 = 1040 + v15
        del v15
        v17 = v0[v16:].view(cp.uint8)
        del v16
        v18 = method64(v17)
        del v17
        v11[v12] = v18
        del v18
        v12 += 1 
    del v12
    v19 = method67(v0)
    v20 = v0[1056:].view(cp.uint8)
    del v0
    if v19 == 0:
        method52(v20)
        v37 = US7_0()
    elif v19 == 1:
        v23, v24, v25, v26, v27, v28 = method61(v20)
        v37 = US7_1(v23, v24, v25, v26, v27, v28)
    elif v19 == 2:
        v30, v31, v32, v33, v34, v35 = method61(v20)
        v37 = US7_2(v30, v31, v32, v33, v34, v35)
    else:
        raise Exception("Invalid tag.")
    del v19, v20
    return v1, v11, v37
def method74() -> object:
    v0 = []
    return v0
def method73(v0 : US6) -> object:
    match v0:
        case US6_0(): # Jack
            del v0
            v1 = method74()
            v2 = "Jack"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US6_1(): # King
            del v0
            v4 = method74()
            v5 = "King"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US6_2(): # Queen
            del v0
            v7 = method74()
            v8 = "Queen"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method72(v0 : static_array_list) -> object:
    v1 = []
    v2 = v0.length
    v3 = 0
    while method5(v2, v3):
        v5 = v0[v3]
        v6 = method73(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method78(v0 : i32) -> object:
    v1 = v0
    del v0
    return v1
def method79(v0 : US1) -> object:
    match v0:
        case US1_0(): # Call
            del v0
            v1 = method74()
            v2 = "Call"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US1_1(): # Fold
            del v0
            v4 = method74()
            v5 = "Fold"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US1_2(): # Raise
            del v0
            v7 = method74()
            v8 = "Raise"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method77(v0 : i32, v1 : US1) -> object:
    v2 = []
    v3 = method78(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method79(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method80(v0 : i32, v1 : US6) -> object:
    v2 = []
    v3 = method78(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method73(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method82(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method33(v2):
        v4 = v0[v2]
        v5 = method73(v4)
        del v4
        v1.append(v5)
        del v5
        v2 += 1 
    del v0, v2
    return v1
def method81(v0 : static_array, v1 : i32, v2 : i32) -> object:
    v3 = method82(v0)
    del v0
    v4 = method78(v1)
    del v1
    v5 = method78(v2)
    del v2
    v6 = {'cards_shown': v3, 'chips_won': v4, 'winner_id': v5}
    del v3, v4, v5
    return v6
def method76(v0 : US8) -> object:
    match v0:
        case US8_0(v1): # CommunityCardIs
            del v0
            v2 = method73(v1)
            del v1
            v3 = "CommunityCardIs"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US8_1(v5, v6): # PlayerAction
            del v0
            v7 = method77(v5, v6)
            del v5, v6
            v8 = "PlayerAction"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US8_2(v10, v11): # PlayerGotCard
            del v0
            v12 = method80(v10, v11)
            del v10, v11
            v13 = "PlayerGotCard"
            v14 = [v13,v12]
            del v12, v13
            return v14
        case US8_3(v15, v16, v17): # Showdown
            del v0
            v18 = method81(v15, v16, v17)
            del v15, v16, v17
            v19 = "Showdown"
            v20 = [v19,v18]
            del v18, v19
            return v20
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method75(v0 : static_array_list) -> object:
    v1 = []
    v2 = v0.length
    v3 = 0
    while method5(v2, v3):
        v5 = v0[v3]
        v6 = method76(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method71(v0 : static_array_list, v1 : static_array_list) -> object:
    v2 = method72(v0)
    del v0
    v3 = method75(v1)
    del v1
    v4 = {'deck': v2, 'messages': v3}
    del v2, v3
    return v4
def method87(v0 : US5) -> object:
    match v0:
        case US5_0(): # None
            del v0
            v1 = method74()
            v2 = "None"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US5_1(v4): # Some
            del v0
            v5 = method73(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method88(v0 : bool) -> object:
    v1 = v0
    del v0
    return v1
def method89(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method33(v2):
        v4 = v0[v2]
        v5 = method78(v4)
        del v4
        v1.append(v5)
        del v5
        v2 += 1 
    del v0, v2
    return v1
def method86(v0 : US5, v1 : bool, v2 : static_array, v3 : i32, v4 : static_array, v5 : i32) -> object:
    v6 = method87(v0)
    del v0
    v7 = method88(v1)
    del v1
    v8 = method82(v2)
    del v2
    v9 = method78(v3)
    del v3
    v10 = method89(v4)
    del v4
    v11 = method78(v5)
    del v5
    v12 = {'community_card': v6, 'is_button_s_first_move': v7, 'pl_card': v8, 'player_turn': v9, 'pot': v10, 'raises_left': v11}
    del v6, v7, v8, v9, v10, v11
    return v12
def method90(v0 : US5, v1 : bool, v2 : static_array, v3 : i32, v4 : static_array, v5 : i32, v6 : US1) -> object:
    v7 = []
    v8 = method86(v0, v1, v2, v3, v4, v5)
    del v0, v1, v2, v3, v4, v5
    v7.append(v8)
    del v8
    v9 = method79(v6)
    del v6
    v7.append(v9)
    del v9
    v10 = v7
    del v7
    return v10
def method85(v0 : US4) -> object:
    match v0:
        case US4_0(v1, v2, v3, v4, v5, v6): # ChanceCommunityCard
            del v0
            v7 = method86(v1, v2, v3, v4, v5, v6)
            del v1, v2, v3, v4, v5, v6
            v8 = "ChanceCommunityCard"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US4_1(): # ChanceInit
            del v0
            v10 = method74()
            v11 = "ChanceInit"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case US4_2(v13, v14, v15, v16, v17, v18): # Round
            del v0
            v19 = method86(v13, v14, v15, v16, v17, v18)
            del v13, v14, v15, v16, v17, v18
            v20 = "Round"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case US4_3(v22, v23, v24, v25, v26, v27, v28): # RoundWithAction
            del v0
            v29 = method90(v22, v23, v24, v25, v26, v27, v28)
            del v22, v23, v24, v25, v26, v27, v28
            v30 = "RoundWithAction"
            v31 = [v30,v29]
            del v29, v30
            return v31
        case US4_4(v32, v33, v34, v35, v36, v37): # TerminalCall
            del v0
            v38 = method86(v32, v33, v34, v35, v36, v37)
            del v32, v33, v34, v35, v36, v37
            v39 = "TerminalCall"
            v40 = [v39,v38]
            del v38, v39
            return v40
        case US4_5(v41, v42, v43, v44, v45, v46): # TerminalFold
            del v0
            v47 = method86(v41, v42, v43, v44, v45, v46)
            del v41, v42, v43, v44, v45, v46
            v48 = "TerminalFold"
            v49 = [v48,v47]
            del v47, v48
            return v49
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method84(v0 : US3) -> object:
    match v0:
        case US3_0(): # None
            del v0
            v1 = method74()
            v2 = "None"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US3_1(v4): # Some
            del v0
            v5 = method85(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method92(v0 : US2) -> object:
    match v0:
        case US2_0(): # Computer
            del v0
            v1 = method74()
            v2 = "Computer"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US2_1(): # Human
            del v0
            v4 = method74()
            v5 = "Human"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method91(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method33(v2):
        v4 = v0[v2]
        v5 = method92(v4)
        del v4
        v1.append(v5)
        del v5
        v2 += 1 
    del v0, v2
    return v1
def method93(v0 : US7) -> object:
    match v0:
        case US7_0(): # GameNotStarted
            del v0
            v1 = method74()
            v2 = "GameNotStarted"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US7_1(v4, v5, v6, v7, v8, v9): # GameOver
            del v0
            v10 = method86(v4, v5, v6, v7, v8, v9)
            del v4, v5, v6, v7, v8, v9
            v11 = "GameOver"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case US7_2(v13, v14, v15, v16, v17, v18): # WaitingForActionFromPlayerId
            del v0
            v19 = method86(v13, v14, v15, v16, v17, v18)
            del v13, v14, v15, v16, v17, v18
            v20 = "WaitingForActionFromPlayerId"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method83(v0 : US3, v1 : static_array, v2 : US7) -> object:
    v3 = method84(v0)
    del v0
    v4 = method91(v1)
    del v1
    v5 = method93(v2)
    del v2
    v6 = {'game': v3, 'pl_type': v4, 'ui_game_state': v5}
    del v3, v4, v5
    return v6
def method70(v0 : static_array_list, v1 : static_array_list, v2 : US3, v3 : static_array, v4 : US7) -> object:
    v5 = method71(v0, v1)
    del v0, v1
    v6 = method83(v2, v3, v4)
    del v2, v3, v4
    v7 = {'large': v5, 'small': v6}
    del v5, v6
    return v7
def method94(v0 : static_array_list, v1 : static_array, v2 : US7) -> object:
    v3 = method75(v0)
    del v0
    v4 = method91(v1)
    del v1
    v5 = method93(v2)
    del v2
    v6 = {'messages': v3, 'pl_type': v4, 'ui_game_state': v5}
    del v3, v4, v5
    return v6
def method69(v0 : static_array_list, v1 : static_array_list, v2 : US3, v3 : static_array, v4 : US7, v5 : static_array_list, v6 : static_array, v7 : US7) -> object:
    v8 = method70(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    v9 = method94(v5, v6, v7)
    del v5, v6, v7
    v10 = {'game_state': v8, 'ui_state': v9}
    del v8, v9
    return v10
def method68(v0 : static_array_list, v1 : static_array_list, v2 : US3, v3 : static_array, v4 : US7, v5 : static_array_list, v6 : static_array, v7 : US7) -> object:
    v8 = method69(v0, v1, v2, v3, v4, v5, v6, v7)
    del v0, v1, v2, v3, v4, v5, v6, v7
    return v8
def method96(v0 : static_array_list, v1 : static_array_list, v2 : US3, v3 : static_array, v4 : US7) -> object:
    v5 = method70(v0, v1, v2, v3, v4)
    del v0, v2
    v6 = method94(v1, v3, v4)
    del v1, v3, v4
    v7 = {'game_state': v5, 'ui_state': v6}
    del v5, v6
    return v7
def method95(v0 : static_array_list, v1 : static_array_list, v2 : US3, v3 : static_array, v4 : US7) -> object:
    v5 = method96(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    return v5
def main():
    v0 = Closure0()
    v1 = Closure1()
    v2 = collections.namedtuple("Leduc_Game",['event_loop_gpu', 'init'])(v0, v1)
    del v0, v1
    return v2

if __name__ == '__main__': print(main())
