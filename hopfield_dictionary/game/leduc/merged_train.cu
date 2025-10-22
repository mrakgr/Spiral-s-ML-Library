#include "merged_train.auto.cu"
#include <thrust/device_vector.h>
#include <unordered_map>
#include <xoshiro.h>
struct Union1;
struct Union2;
struct Union0;
struct Tuple0;
typedef unsigned long long (* Fun0)(static_array_list<Union0,32>);
typedef bool (* Fun1)(static_array_list<Union0,32>, static_array_list<Union0,32>);
struct Tuple2;
struct Tuple1;
struct StackRefs0;
struct Union3;
struct StackMut0;
struct StackRefs1;
struct StackRefs2;
struct Tuple3;
struct Union5;
struct Union4;
struct StackMut1;
struct Tuple4;
unsigned int loop_2(unsigned int v0, xso::rng & v1);
struct StackMut2;
struct StackMut3;
unsigned int find_nth_set_bit_3(int v0, unsigned int v1, unsigned int v2);
Tuple4 draw_card_1(xso::rng & v0, unsigned int v1);
struct Union6;
struct Tuple5;
float loop_4(static_array<Union3,2> v0, StackRefs0 & v1, xso::rng & v2, StackRefs1 & v3, StackMut0 & v4, StackRefs2 & v5, StackMut1 & v6, Union6 v7);
struct Union7;
struct Union8;
static_array<float,3> masking_normalize_5(static_array<float,3> v0, static_array<bool,3> v1);
static_array<float,3> relu_7(static_array<float,3> v0);
static_array<float,3> regret_match_6(static_array<float,3> v0, static_array<bool,3> v1);
int loop_10(static_array<float,3> v0, float v1, int v2);
int pick_discrete__9(static_array<float,3> v0, float v1);
int sample_discrete__8(static_array<float,3> v0, xso::rng & v1);
struct Union9;
int tag_12(Union1 v0);
bool is_pair_13(int v0, int v1);
Tuple5 order_14(int v0, int v1);
Union9 compare_hands_11(Union5 v0, bool v1, static_array<Union1,2> v2, int v3, static_array<int,2> v4, int v5);
float body_0(static_array<Union3,2> v0, StackRefs0 & v1, xso::rng & v2, StackRefs1 & v3, StackMut0 & v4, StackRefs2 & v5, Union4 v6);
struct Tuple6;
float loop_16(static_array<Union3,2> v0, StackRefs0 & v1, xso::rng & v2, StackRefs1 & v3, StackMut0 & v4, StackRefs2 & v5, StackMut1 & v6, Union6 v7);
float body_15(static_array<Union3,2> v0, StackRefs0 & v1, xso::rng & v2, StackRefs1 & v3, StackMut0 & v4, StackRefs2 & v5, Union4 v6);
static_array<float,3> normalize_17(static_array<float,3> v0);
void method_19(Union1 v0);
void method_20(Union2 v0);
void method_18(Union0 v0);
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
struct Union0_1 { // PlayerAction
    Union2 v1;
    int v0;
    __host__ __device__ Union0_1(int t0, Union2 t1) : v0(t0), v1(t1) {}
    __host__ __device__ Union0_1() = delete;
};
struct Union0_2 { // PlayerGotCard
    Union1 v1;
    int v0;
    __host__ __device__ Union0_2(int t0, Union1 t1) : v0(t0), v1(t1) {}
    __host__ __device__ Union0_2() = delete;
};
struct Union0_3 { // Showdown
    static_array<Union1,2> v0;
    int v1;
    int v2;
    __host__ __device__ Union0_3(static_array<Union1,2> t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
    __host__ __device__ Union0_3() = delete;
};
struct Union0 {
    union {
        Union0_0 case0; // CommunityCardIs
        Union0_1 case1; // PlayerAction
        Union0_2 case2; // PlayerGotCard
        Union0_3 case3; // Showdown
    };
    unsigned char tag{255};
    __host__ __device__ Union0() {}
    __host__ __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // CommunityCardIs
    __host__ __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // PlayerAction
    __host__ __device__ Union0(Union0_2 t) : tag(2), case2(t) {} // PlayerGotCard
    __host__ __device__ Union0(Union0_3 t) : tag(3), case3(t) {} // Showdown
    __host__ __device__ Union0(const Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // CommunityCardIs
            case 1: new (&this->case1) Union0_1(x.case1); break; // PlayerAction
            case 2: new (&this->case2) Union0_2(x.case2); break; // PlayerGotCard
            case 3: new (&this->case3) Union0_3(x.case3); break; // Showdown
        }
    }
    __host__ __device__ Union0(const Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // CommunityCardIs
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // PlayerAction
            case 2: new (&this->case2) Union0_2(std::move(x.case2)); break; // PlayerGotCard
            case 3: new (&this->case3) Union0_3(std::move(x.case3)); break; // Showdown
        }
    }
    __host__ __device__ Union0 & operator=(const Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // CommunityCardIs
                case 1: this->case1 = x.case1; break; // PlayerAction
                case 2: this->case2 = x.case2; break; // PlayerGotCard
                case 3: this->case3 = x.case3; break; // Showdown
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
                case 1: this->case1 = std::move(x.case1); break; // PlayerAction
                case 2: this->case2 = std::move(x.case2); break; // PlayerGotCard
                case 3: this->case3 = std::move(x.case3); break; // Showdown
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
            case 1: this->case1.~Union0_1(); break; // PlayerAction
            case 2: this->case2.~Union0_2(); break; // PlayerGotCard
            case 3: this->case3.~Union0_3(); break; // Showdown
        }
        this->tag = 255;
    }
};
struct Tuple0 {
    unsigned long long v1;
    unsigned long long v2;
    int v0;
    __host__ __device__ Tuple0() = default;
    __host__ __device__ Tuple0(int t0, unsigned long long t1, unsigned long long t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Tuple2 {
    float v0;
    float v1;
    __host__ __device__ Tuple2() = default;
    __host__ __device__ Tuple2(float t0, float t1) : v0(t0), v1(t1) {}
};
struct Tuple1 {
    static_array<float,3> v0;
    static_array<float,3> v1;
    static_array<Tuple2,3> v2;
    __host__ __device__ Tuple1() = default;
    __host__ __device__ Tuple1(static_array<float,3> t0, static_array<float,3> t1, static_array<Tuple2,3> t2) : v0(t0), v1(t1), v2(t2) {}
};
struct StackRefs0 {
    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v0;
    __host__ __device__ StackRefs0() = default;
    __host__ __device__ StackRefs0(std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & t0) : v0(t0) {}
};
struct Union3_0 { // Frozen
};
struct Union3_1 { // TrainEnumerative
};
struct Union3_2 { // TrainSampling
};
struct Union3 {
    union {
        Union3_0 case0; // Frozen
        Union3_1 case1; // TrainEnumerative
        Union3_2 case2; // TrainSampling
    };
    unsigned char tag{255};
    __host__ __device__ Union3() {}
    __host__ __device__ Union3(Union3_0 t) : tag(0), case0(t) {} // Frozen
    __host__ __device__ Union3(Union3_1 t) : tag(1), case1(t) {} // TrainEnumerative
    __host__ __device__ Union3(Union3_2 t) : tag(2), case2(t) {} // TrainSampling
    __host__ __device__ Union3(const Union3 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(x.case0); break; // Frozen
            case 1: new (&this->case1) Union3_1(x.case1); break; // TrainEnumerative
            case 2: new (&this->case2) Union3_2(x.case2); break; // TrainSampling
        }
    }
    __host__ __device__ Union3(const Union3 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(std::move(x.case0)); break; // Frozen
            case 1: new (&this->case1) Union3_1(std::move(x.case1)); break; // TrainEnumerative
            case 2: new (&this->case2) Union3_2(std::move(x.case2)); break; // TrainSampling
        }
    }
    __host__ __device__ Union3 & operator=(const Union3 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Frozen
                case 1: this->case1 = x.case1; break; // TrainEnumerative
                case 2: this->case2 = x.case2; break; // TrainSampling
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
                case 0: this->case0 = std::move(x.case0); break; // Frozen
                case 1: this->case1 = std::move(x.case1); break; // TrainEnumerative
                case 2: this->case2 = std::move(x.case2); break; // TrainSampling
            }
        } else {
            this->~Union3();
            new (this) Union3{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union3() {
        switch(this->tag){
            case 0: this->case0.~Union3_0(); break; // Frozen
            case 1: this->case1.~Union3_1(); break; // TrainEnumerative
            case 2: this->case2.~Union3_2(); break; // TrainSampling
        }
        this->tag = 255;
    }
};
struct StackMut0 {
    unsigned int v0;
    __host__ __device__ StackMut0() = default;
    __host__ __device__ StackMut0(unsigned int t0) : v0(t0) {}
};
struct StackRefs1 {
    static_array_list<Union0,32> & v0;
    __host__ __device__ StackRefs1() = default;
    __host__ __device__ StackRefs1(static_array_list<Union0,32> & t0) : v0(t0) {}
};
struct StackRefs2 {
    static_array<Tuple2,2> & v0;
    __host__ __device__ StackRefs2() = default;
    __host__ __device__ StackRefs2(static_array<Tuple2,2> & t0) : v0(t0) {}
};
struct Tuple3 {
    int v0;
    float v1;
    __host__ __device__ Tuple3() = default;
    __host__ __device__ Tuple3(int t0, float t1) : v0(t0), v1(t1) {}
};
struct Union5_0 { // None
};
struct Union5_1 { // Some
    Union1 v0;
    __host__ __device__ Union5_1(Union1 t0) : v0(t0) {}
    __host__ __device__ Union5_1() = delete;
};
struct Union5 {
    union {
        Union5_0 case0; // None
        Union5_1 case1; // Some
    };
    unsigned char tag{255};
    __host__ __device__ Union5() {}
    __host__ __device__ Union5(Union5_0 t) : tag(0), case0(t) {} // None
    __host__ __device__ Union5(Union5_1 t) : tag(1), case1(t) {} // Some
    __host__ __device__ Union5(const Union5 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union5_0(x.case0); break; // None
            case 1: new (&this->case1) Union5_1(x.case1); break; // Some
        }
    }
    __host__ __device__ Union5(const Union5 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union5_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union5_1(std::move(x.case1)); break; // Some
        }
    }
    __host__ __device__ Union5 & operator=(const Union5 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
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
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union5();
            new (this) Union5{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union5() {
        switch(this->tag){
            case 0: this->case0.~Union5_0(); break; // None
            case 1: this->case1.~Union5_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union4_0 { // ChanceCommunityCard
    Union5 v0;
    static_array<Union1,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union4_0(Union5 t0, bool t1, static_array<Union1,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __host__ __device__ Union4_0() = delete;
};
struct Union4_1 { // ChanceInit
};
struct Union4_2 { // Round
    Union5 v0;
    static_array<Union1,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union4_2(Union5 t0, bool t1, static_array<Union1,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __host__ __device__ Union4_2() = delete;
};
struct Union4_3 { // RoundWithAction
    Union5 v0;
    static_array<Union1,2> v2;
    static_array<int,2> v4;
    Union2 v6;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union4_3(Union5 t0, bool t1, static_array<Union1,2> t2, int t3, static_array<int,2> t4, int t5, Union2 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __host__ __device__ Union4_3() = delete;
};
struct Union4_4 { // TerminalCall
    Union5 v0;
    static_array<Union1,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union4_4(Union5 t0, bool t1, static_array<Union1,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __host__ __device__ Union4_4() = delete;
};
struct Union4_5 { // TerminalFold
    Union5 v0;
    static_array<Union1,2> v2;
    static_array<int,2> v4;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union4_5(Union5 t0, bool t1, static_array<Union1,2> t2, int t3, static_array<int,2> t4, int t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __host__ __device__ Union4_5() = delete;
};
struct Union4 {
    union {
        Union4_0 case0; // ChanceCommunityCard
        Union4_1 case1; // ChanceInit
        Union4_2 case2; // Round
        Union4_3 case3; // RoundWithAction
        Union4_4 case4; // TerminalCall
        Union4_5 case5; // TerminalFold
    };
    unsigned char tag{255};
    __host__ __device__ Union4() {}
    __host__ __device__ Union4(Union4_0 t) : tag(0), case0(t) {} // ChanceCommunityCard
    __host__ __device__ Union4(Union4_1 t) : tag(1), case1(t) {} // ChanceInit
    __host__ __device__ Union4(Union4_2 t) : tag(2), case2(t) {} // Round
    __host__ __device__ Union4(Union4_3 t) : tag(3), case3(t) {} // RoundWithAction
    __host__ __device__ Union4(Union4_4 t) : tag(4), case4(t) {} // TerminalCall
    __host__ __device__ Union4(Union4_5 t) : tag(5), case5(t) {} // TerminalFold
    __host__ __device__ Union4(const Union4 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(x.case0); break; // ChanceCommunityCard
            case 1: new (&this->case1) Union4_1(x.case1); break; // ChanceInit
            case 2: new (&this->case2) Union4_2(x.case2); break; // Round
            case 3: new (&this->case3) Union4_3(x.case3); break; // RoundWithAction
            case 4: new (&this->case4) Union4_4(x.case4); break; // TerminalCall
            case 5: new (&this->case5) Union4_5(x.case5); break; // TerminalFold
        }
    }
    __host__ __device__ Union4(const Union4 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(std::move(x.case0)); break; // ChanceCommunityCard
            case 1: new (&this->case1) Union4_1(std::move(x.case1)); break; // ChanceInit
            case 2: new (&this->case2) Union4_2(std::move(x.case2)); break; // Round
            case 3: new (&this->case3) Union4_3(std::move(x.case3)); break; // RoundWithAction
            case 4: new (&this->case4) Union4_4(std::move(x.case4)); break; // TerminalCall
            case 5: new (&this->case5) Union4_5(std::move(x.case5)); break; // TerminalFold
        }
    }
    __host__ __device__ Union4 & operator=(const Union4 & x) {
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
            this->~Union4();
            new (this) Union4{x};
        }
        return *this;
    }
    __host__ __device__ Union4 & operator=(const Union4 && x) {
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
            this->~Union4();
            new (this) Union4{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union4() {
        switch(this->tag){
            case 0: this->case0.~Union4_0(); break; // ChanceCommunityCard
            case 1: this->case1.~Union4_1(); break; // ChanceInit
            case 2: this->case2.~Union4_2(); break; // Round
            case 3: this->case3.~Union4_3(); break; // RoundWithAction
            case 4: this->case4.~Union4_4(); break; // TerminalCall
            case 5: this->case5.~Union4_5(); break; // TerminalFold
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
    Union1 v0;
    unsigned int v1;
    __host__ __device__ Tuple4() = default;
    __host__ __device__ Tuple4(Union1 t0, unsigned int t1) : v0(t0), v1(t1) {}
};
struct StackMut2 {
    int v0;
    __host__ __device__ StackMut2() = default;
    __host__ __device__ StackMut2(int t0) : v0(t0) {}
};
struct StackMut3 {
    unsigned int v0;
    __host__ __device__ StackMut3() = default;
    __host__ __device__ StackMut3(unsigned int t0) : v0(t0) {}
};
struct Union6_0 { // T_game_chance_community_card
    Union5 v0;
    static_array<Union1,2> v2;
    static_array<int,2> v4;
    Union1 v6;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union6_0(Union5 t0, bool t1, static_array<Union1,2> t2, int t3, static_array<int,2> t4, int t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __host__ __device__ Union6_0() = delete;
};
struct Union6_1 { // T_game_chance_init
    Union1 v0;
    Union1 v1;
    __host__ __device__ Union6_1(Union1 t0, Union1 t1) : v0(t0), v1(t1) {}
    __host__ __device__ Union6_1() = delete;
};
struct Union6_2 { // T_game_round
    Union5 v0;
    static_array<Union1,2> v2;
    static_array<int,2> v4;
    Union2 v6;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union6_2(Union5 t0, bool t1, static_array<Union1,2> t2, int t3, static_array<int,2> t4, int t5, Union2 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __host__ __device__ Union6_2() = delete;
};
struct Union6_3 { // T_none
};
struct Union6 {
    union {
        Union6_0 case0; // T_game_chance_community_card
        Union6_1 case1; // T_game_chance_init
        Union6_2 case2; // T_game_round
        Union6_3 case3; // T_none
    };
    unsigned char tag{255};
    __host__ __device__ Union6() {}
    __host__ __device__ Union6(Union6_0 t) : tag(0), case0(t) {} // T_game_chance_community_card
    __host__ __device__ Union6(Union6_1 t) : tag(1), case1(t) {} // T_game_chance_init
    __host__ __device__ Union6(Union6_2 t) : tag(2), case2(t) {} // T_game_round
    __host__ __device__ Union6(Union6_3 t) : tag(3), case3(t) {} // T_none
    __host__ __device__ Union6(const Union6 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(x.case0); break; // T_game_chance_community_card
            case 1: new (&this->case1) Union6_1(x.case1); break; // T_game_chance_init
            case 2: new (&this->case2) Union6_2(x.case2); break; // T_game_round
            case 3: new (&this->case3) Union6_3(x.case3); break; // T_none
        }
    }
    __host__ __device__ Union6(const Union6 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(std::move(x.case0)); break; // T_game_chance_community_card
            case 1: new (&this->case1) Union6_1(std::move(x.case1)); break; // T_game_chance_init
            case 2: new (&this->case2) Union6_2(std::move(x.case2)); break; // T_game_round
            case 3: new (&this->case3) Union6_3(std::move(x.case3)); break; // T_none
        }
    }
    __host__ __device__ Union6 & operator=(const Union6 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // T_game_chance_community_card
                case 1: this->case1 = x.case1; break; // T_game_chance_init
                case 2: this->case2 = x.case2; break; // T_game_round
                case 3: this->case3 = x.case3; break; // T_none
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
                case 0: this->case0 = std::move(x.case0); break; // T_game_chance_community_card
                case 1: this->case1 = std::move(x.case1); break; // T_game_chance_init
                case 2: this->case2 = std::move(x.case2); break; // T_game_round
                case 3: this->case3 = std::move(x.case3); break; // T_none
            }
        } else {
            this->~Union6();
            new (this) Union6{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union6() {
        switch(this->tag){
            case 0: this->case0.~Union6_0(); break; // T_game_chance_community_card
            case 1: this->case1.~Union6_1(); break; // T_game_chance_init
            case 2: this->case2.~Union6_2(); break; // T_game_round
            case 3: this->case3.~Union6_3(); break; // T_none
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
struct Union7_0 { // None
};
struct Union7_1 { // Some
    static_array<float,3> v0;
    static_array<float,3> v1;
    static_array<Tuple2,3> v2;
    __host__ __device__ Union7_1(static_array<float,3> t0, static_array<float,3> t1, static_array<Tuple2,3> t2) : v0(t0), v1(t1), v2(t2) {}
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
struct Union8_0 { // None
};
struct Union8_1 { // Some
    Union2 v0;
    __host__ __device__ Union8_1(Union2 t0) : v0(t0) {}
    __host__ __device__ Union8_1() = delete;
};
struct Union8 {
    union {
        Union8_0 case0; // None
        Union8_1 case1; // Some
    };
    unsigned char tag{255};
    __host__ __device__ Union8() {}
    __host__ __device__ Union8(Union8_0 t) : tag(0), case0(t) {} // None
    __host__ __device__ Union8(Union8_1 t) : tag(1), case1(t) {} // Some
    __host__ __device__ Union8(const Union8 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union8_0(x.case0); break; // None
            case 1: new (&this->case1) Union8_1(x.case1); break; // Some
        }
    }
    __host__ __device__ Union8(const Union8 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union8_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union8_1(std::move(x.case1)); break; // Some
        }
    }
    __host__ __device__ Union8 & operator=(const Union8 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
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
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union8();
            new (this) Union8{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union8() {
        switch(this->tag){
            case 0: this->case0.~Union8_0(); break; // None
            case 1: this->case1.~Union8_1(); break; // Some
        }
        this->tag = 255;
    }
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
    unsigned char tag{255};
    __host__ __device__ Union9() {}
    __host__ __device__ Union9(Union9_0 t) : tag(0), case0(t) {} // Eq
    __host__ __device__ Union9(Union9_1 t) : tag(1), case1(t) {} // Gt
    __host__ __device__ Union9(Union9_2 t) : tag(2), case2(t) {} // Lt
    __host__ __device__ Union9(const Union9 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union9_0(x.case0); break; // Eq
            case 1: new (&this->case1) Union9_1(x.case1); break; // Gt
            case 2: new (&this->case2) Union9_2(x.case2); break; // Lt
        }
    }
    __host__ __device__ Union9(const Union9 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union9_0(std::move(x.case0)); break; // Eq
            case 1: new (&this->case1) Union9_1(std::move(x.case1)); break; // Gt
            case 2: new (&this->case2) Union9_2(std::move(x.case2)); break; // Lt
        }
    }
    __host__ __device__ Union9 & operator=(const Union9 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Eq
                case 1: this->case1 = x.case1; break; // Gt
                case 2: this->case2 = x.case2; break; // Lt
            }
        } else {
            this->~Union9();
            new (this) Union9{x};
        }
        return *this;
    }
    __host__ __device__ Union9 & operator=(const Union9 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // Eq
                case 1: this->case1 = std::move(x.case1); break; // Gt
                case 2: this->case2 = std::move(x.case2); break; // Lt
            }
        } else {
            this->~Union9();
            new (this) Union9{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union9() {
        switch(this->tag){
            case 0: this->case0.~Union9_0(); break; // Eq
            case 1: this->case1.~Union9_1(); break; // Gt
            case 2: this->case2.~Union9_2(); break; // Lt
        }
        this->tag = 255;
    }
};
struct Tuple6 {
    int v0;
    float v1;
    float v2;
    __host__ __device__ Tuple6() = default;
    __host__ __device__ Tuple6(int t0, float t1, float t2) : v0(t0), v1(t1), v2(t2) {}
};
inline bool while_method_0(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 2;
    return v1;
}
unsigned long long FunPointerMethod0(static_array_list<Union0,32> tup0){
    static_array_list<Union0,32> v0 = tup0;
    int v1;
    v1 = v0.length;
    int v2; unsigned long long v3; unsigned long long v4;
    Tuple0 tmp0 = Tuple0{0, 0ull, 1ull};
    v2 = tmp0.v0; v3 = tmp0.v1; v4 = tmp0.v2;
    while (while_method_0(v1, v2)){
        Union0 v7;
        v7 = v0[v2];
        unsigned long long v52;
        switch (v7.tag) {
            case 0: { // CommunityCardIs
                Union1 v9 = v7.case0.v0;
                unsigned long long v10;
                switch (v9.tag) {
                    case 0: { // Jack
                        v10 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v10 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v10 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v11;
                v11 = 9223372036854775807ull + v10;
                unsigned long long v12;
                v12 = v11 * 9973ull;
                v52 = v12;
                break;
            }
            case 1: { // PlayerAction
                int v13 = v7.case1.v0; Union2 v14 = v7.case1.v1;
                unsigned long long v15;
                v15 = std::hash<int>()(v13);
                unsigned long long v16;
                v16 = v15 * 9973ull;
                unsigned long long v17;
                switch (v14.tag) {
                    case 0: { // Call
                        v17 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // Fold
                        v17 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Raise
                        v17 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v18;
                v18 = v16 + v17;
                unsigned long long v19;
                v19 = 9223372036854775807ull + v18;
                unsigned long long v20;
                v20 = v19 * 9973ull;
                unsigned long long v21;
                v21 = v20 * 2ull;
                v52 = v21;
                break;
            }
            case 2: { // PlayerGotCard
                int v22 = v7.case2.v0; Union1 v23 = v7.case2.v1;
                unsigned long long v24;
                v24 = std::hash<int>()(v22);
                unsigned long long v25;
                v25 = v24 * 9973ull;
                unsigned long long v26;
                switch (v23.tag) {
                    case 0: { // Jack
                        v26 = 9223372036854765835ull;
                        break;
                    }
                    case 1: { // King
                        v26 = 18446744073709531670ull;
                        break;
                    }
                    case 2: { // Queen
                        v26 = 9223372036854745889ull;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v27;
                v27 = v25 + v26;
                unsigned long long v28;
                v28 = 9223372036854775807ull + v27;
                unsigned long long v29;
                v29 = v28 * 9973ull;
                unsigned long long v30;
                v30 = v29 * 3ull;
                v52 = v30;
                break;
            }
            case 3: { // Showdown
                static_array<Union1,2> v31 = v7.case3.v0; int v32 = v7.case3.v1; int v33 = v7.case3.v2;
                unsigned long long v34;
                v34 = std::hash<int>()(v33);
                unsigned long long v35;
                v35 = std::hash<int>()(v32);
                unsigned long long v36;
                v36 = v35 * 9973ull;
                unsigned long long v37;
                v37 = v34 + v36;
                int v38; unsigned long long v39; unsigned long long v40;
                Tuple0 tmp1 = Tuple0{0, 0ull, 1ull};
                v38 = tmp1.v0; v39 = tmp1.v1; v40 = tmp1.v2;
                while (while_method_1(v38)){
                    Union1 v43;
                    v43 = v31[v38];
                    unsigned long long v45;
                    switch (v43.tag) {
                        case 0: { // Jack
                            v45 = 9223372036854765835ull;
                            break;
                        }
                        case 1: { // King
                            v45 = 18446744073709531670ull;
                            break;
                        }
                        case 2: { // Queen
                            v45 = 9223372036854745889ull;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    unsigned long long v46;
                    v46 = v45 * v40;
                    unsigned long long v47;
                    v47 = v39 + v46;
                    unsigned long long v48;
                    v48 = v40 * 9973ull;
                    v39 = v47;
                    v40 = v48;
                    v38 += 1 ;
                }
                unsigned long long v49;
                v49 = 9223372036854775807ull + v37;
                unsigned long long v50;
                v50 = v49 * 9973ull;
                unsigned long long v51;
                v51 = v50 * 4ull;
                v52 = v51;
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        unsigned long long v53;
        v53 = v52 * v4;
        unsigned long long v54;
        v54 = v3 + v53;
        unsigned long long v55;
        v55 = v4 * 9973ull;
        v3 = v54;
        v4 = v55;
        v2 += 1 ;
    }
    return 0ull;
}
bool FunPointerMethod1(static_array_list<Union0,32> tup0, static_array_list<Union0,32> tup1){
    static_array_list<Union0,32> v0 = tup0; static_array_list<Union0,32> v1 = tup1;
    int v2;
    v2 = v0.length;
    int v3;
    v3 = v1.length;
    bool v4;
    v4 = v2 == v3;
    if (v4){
        bool v5;
        v5 = true;
        int v6;
        v6 = v1.length;
        int v7;
        v7 = 0;
        while (while_method_0(v6, v7)){
            Union0 v10;
            v10 = v0[v7];
            Union0 v13;
            v13 = v1[v7];
            bool v54;
            switch (v10.tag == v13.tag ? v10.tag : 255) {
                case 0: { // CommunityCardIs
                    Union1 v15 = v10.case0.v0;
                    Union1 v16 = v13.case0.v0;
                    switch (v15.tag == v16.tag ? v15.tag : 255) {
                        case 0: { // Jack
                            v54 = true;
                            break;
                        }
                        case 1: { // King
                            v54 = true;
                            break;
                        }
                        case 2: { // Queen
                            v54 = true;
                            break;
                        }
                        default: {
                            v54 = false;
                        }
                    }
                    break;
                }
                case 1: { // PlayerAction
                    int v18 = v10.case1.v0; Union2 v19 = v10.case1.v1;
                    int v20 = v13.case1.v0; Union2 v21 = v13.case1.v1;
                    bool v22;
                    v22 = v18 == v20;
                    if (v22){
                        switch (v19.tag == v21.tag ? v19.tag : 255) {
                            case 0: { // Call
                                v54 = true;
                                break;
                            }
                            case 1: { // Fold
                                v54 = true;
                                break;
                            }
                            case 2: { // Raise
                                v54 = true;
                                break;
                            }
                            default: {
                                v54 = false;
                            }
                        }
                    } else {
                        v54 = false;
                    }
                    break;
                }
                case 2: { // PlayerGotCard
                    int v25 = v10.case2.v0; Union1 v26 = v10.case2.v1;
                    int v27 = v13.case2.v0; Union1 v28 = v13.case2.v1;
                    bool v29;
                    v29 = v25 == v27;
                    if (v29){
                        switch (v26.tag == v28.tag ? v26.tag : 255) {
                            case 0: { // Jack
                                v54 = true;
                                break;
                            }
                            case 1: { // King
                                v54 = true;
                                break;
                            }
                            case 2: { // Queen
                                v54 = true;
                                break;
                            }
                            default: {
                                v54 = false;
                            }
                        }
                    } else {
                        v54 = false;
                    }
                    break;
                }
                case 3: { // Showdown
                    static_array<Union1,2> v32 = v10.case3.v0; int v33 = v10.case3.v1; int v34 = v10.case3.v2;
                    static_array<Union1,2> v35 = v13.case3.v0; int v36 = v13.case3.v1; int v37 = v13.case3.v2;
                    bool v38;
                    v38 = true;
                    int v39;
                    v39 = 0;
                    while (while_method_1(v39)){
                        Union1 v42;
                        v42 = v32[v39];
                        Union1 v45;
                        v45 = v35[v39];
                        bool v47;
                        switch (v42.tag == v45.tag ? v42.tag : 255) {
                            case 0: { // Jack
                                v47 = true;
                                break;
                            }
                            case 1: { // King
                                v47 = true;
                                break;
                            }
                            case 2: { // Queen
                                v47 = true;
                                break;
                            }
                            default: {
                                v47 = false;
                            }
                        }
                        bool v48;
                        v48 = v47 != true;
                        if (v48){
                            bool v49;
                            v49 = false;
                            v38 = v49;
                            break;
                        } else {
                        }
                        v39 += 1 ;
                    }
                    if (v38){
                        bool v50;
                        v50 = v33 == v36;
                        if (v50){
                            bool v51;
                            v51 = v34 == v37;
                            v54 = v51;
                        } else {
                            v54 = false;
                        }
                    } else {
                        v54 = false;
                    }
                    break;
                }
                default: {
                    v54 = false;
                }
            }
            bool v55;
            v55 = v54 != true;
            if (v55){
                bool v56;
                v56 = false;
                v5 = v56;
                break;
            } else {
            }
            v7 += 1 ;
        }
        return v5;
    } else {
        return false;
    }
}
inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 1000000;
    return v1;
}
unsigned int loop_2(unsigned int v0, xso::rng & v1){
    unsigned int v2;
    v2 = v1();
    unsigned int v3;
    v3 = v2 % v0;
    unsigned int v4;
    v4 = v2 - v3;
    unsigned int v5;
    v5 = 0u - v0;
    bool v6;
    v6 = v4 <= v5;
    if (v6){
        return v3;
    } else {
        return loop_2(v0, v1);
    }
}
inline bool while_method_3(unsigned int v0, unsigned int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
unsigned int find_nth_set_bit_3(int v0, unsigned int v1, unsigned int v2){
    int v4;
    v4 = (int)v1;
    unsigned int v5;
    v5 = v2 >> v4;
    StackMut2 v6{0};
    StackMut3 v7{4294967295u};
    unsigned int v8;
    v8 = 32u - v1;
    unsigned int v9;
    v9 = 0u;
    while (while_method_3(v8, v9)){
        int v11;
        v11 = (int)v9;
        unsigned int v12;
        v12 = v5 >> v11;
        unsigned int v13;
        v13 = v12 & 1u;
        bool v14;
        v14 = v13 == 0u;
        bool v15;
        v15 = v14 != true;
        if (v15){
            int v16 = v6.v0;
            int v17;
            v17 = v16 + 1;
            v6.v0 = v17;
            bool v18;
            v18 = v17 == v0;
            if (v18){
                unsigned int v19;
                v19 = v1 + v9;
                v7.v0 = v19;
                break;
            } else {
            }
        } else {
        }
        v9 += 1u ;
    }
    unsigned int v20 = v7.v0;
    return v20;
}
Tuple4 draw_card_1(xso::rng & v0, unsigned int v1){
    int v3;
    v3 = __builtin_popcount(v1);
    unsigned int v4;
    v4 = (unsigned int)v3;
    bool v5;
    v5 = 0u < v4;
    bool v6;
    v6 = v5 == false;
    if (v6){
        assert("The range has to be greater than 0." && v5);
    } else {
    }
    unsigned int v8;
    v8 = loop_2(v4, v0);
    int v9;
    v9 = (int)v8;
    int v11;
    v11 = __builtin_popcount(v1);
    bool v12;
    v12 = v9 < v11;
    unsigned int v18;
    if (v12){
        int v13;
        v13 = v9 + 1;
        unsigned int v14;
        v14 = 0u;
        v18 = find_nth_set_bit_3(v13, v14, v1);
    } else {
        int v16;
        v16 = v9 - v11;
        printf("%s\n", "Cannot find the n-th set bit.");
        exit(-1);
    }
    bool v19;
    v19 = 0u == v18;
    Union1 v37;
    if (v19){
        v37 = Union1{Union1_1{}};
    } else {
        bool v21;
        v21 = 1u == v18;
        if (v21){
            v37 = Union1{Union1_1{}};
        } else {
            bool v23;
            v23 = 2u == v18;
            if (v23){
                v37 = Union1{Union1_2{}};
            } else {
                bool v25;
                v25 = 3u == v18;
                if (v25){
                    v37 = Union1{Union1_2{}};
                } else {
                    bool v27;
                    v27 = 4u == v18;
                    if (v27){
                        v37 = Union1{Union1_0{}};
                    } else {
                        bool v29;
                        v29 = 5u == v18;
                        if (v29){
                            v37 = Union1{Union1_0{}};
                        } else {
                            printf("%s\n", "Invalid int in int_to_card.");
                            exit(-1);
                        }
                    }
                }
            }
        }
    }
    int v38;
    v38 = (int)v18;
    unsigned int v39;
    v39 = 1u << v38;
    unsigned int v40;
    v40 = v1 ^ v39;
    return Tuple4{v37, v40};
}
float loop_4(static_array<Union3,2> v0, StackRefs0 & v1, xso::rng & v2, StackRefs1 & v3, StackMut0 & v4, StackRefs2 & v5, StackMut1 & v6, Union6 v7){
    switch (v7.tag) {
        case 0: { // T_game_chance_community_card
            Union5 v9 = v7.case0.v0; bool v10 = v7.case0.v1; static_array<Union1,2> v11 = v7.case0.v2; int v12 = v7.case0.v3; static_array<int,2> v13 = v7.case0.v4; int v14 = v7.case0.v5; Union1 v15 = v7.case0.v6;
            int v16;
            v16 = 2;
            int v17; int v18;
            Tuple5 tmp4 = Tuple5{0, 0};
            v17 = tmp4.v0; v18 = tmp4.v1;
            while (while_method_1(v17)){
                int v21;
                v21 = v13[v17];
                bool v23;
                v23 = v18 >= v21;
                int v24;
                if (v23){
                    v24 = v18;
                } else {
                    v24 = v21;
                }
                v18 = v24;
                v17 += 1 ;
            }
            static_array<int,2> v26;
            int v28;
            v28 = 0;
            while (while_method_1(v28)){
                v26[v28] = v18;
                v28 += 1 ;
            }
            Union5 v30;
            v30 = Union5{Union5_1{v15}};
            bool v31;
            v31 = true;
            int v32;
            v32 = 0;
            Union4 v33;
            v33 = Union4{Union4_2{v30, v31, v11, v32, v26, v16}};
            return body_0(v0, v1, v2, v3, v4, v5, v33);
            break;
        }
        case 1: { // T_game_chance_init
            Union1 v35 = v7.case1.v0; Union1 v36 = v7.case1.v1;
            int v37;
            v37 = 2;
            static_array<int,2> v39;
            v39[0] = 1;
            v39[1] = 1;
            static_array<Union1,2> v42;
            v42[0] = v35;
            v42[1] = v36;
            Union5 v44;
            v44 = Union5{Union5_0{}};
            bool v45;
            v45 = true;
            int v46;
            v46 = 0;
            Union4 v47;
            v47 = Union4{Union4_2{v44, v45, v42, v46, v39, v37}};
            return body_0(v0, v1, v2, v3, v4, v5, v47);
            break;
        }
        case 2: { // T_game_round
            Union5 v49 = v7.case2.v0; bool v50 = v7.case2.v1; static_array<Union1,2> v51 = v7.case2.v2; int v52 = v7.case2.v3; static_array<int,2> v53 = v7.case2.v4; int v54 = v7.case2.v5; Union2 v55 = v7.case2.v6;
            Union4 v147;
            switch (v49.tag) {
                case 0: { // None
                    switch (v55.tag) {
                        case 0: { // Call
                            if (v50){
                                int v109;
                                v109 = v52 ^ 1;
                                v147 = Union4{Union4_2{v49, false, v51, v109, v53, v54}};
                            } else {
                                v147 = Union4{Union4_0{v49, v50, v51, v52, v53, v54}};
                            }
                            break;
                        }
                        case 1: { // Fold
                            v147 = Union4{Union4_5{v49, v50, v51, v52, v53, v54}};
                            break;
                        }
                        case 2: { // Raise
                            bool v113;
                            v113 = v54 > 0;
                            if (v113){
                                int v114;
                                v114 = v52 ^ 1;
                                int v115;
                                v115 = -1 + v54;
                                int v116; int v117;
                                Tuple5 tmp5 = Tuple5{0, 0};
                                v116 = tmp5.v0; v117 = tmp5.v1;
                                while (while_method_1(v116)){
                                    int v120;
                                    v120 = v53[v116];
                                    bool v122;
                                    v122 = v117 >= v120;
                                    int v123;
                                    if (v122){
                                        v123 = v117;
                                    } else {
                                        v123 = v120;
                                    }
                                    v117 = v123;
                                    v116 += 1 ;
                                }
                                static_array<int,2> v125;
                                int v127;
                                v127 = 0;
                                while (while_method_1(v127)){
                                    v125[v127] = v117;
                                    v127 += 1 ;
                                }
                                static_array<int,2> v130;
                                int v132;
                                v132 = 0;
                                while (while_method_1(v132)){
                                    int v135;
                                    v135 = v125[v132];
                                    bool v137;
                                    v137 = v132 == v52;
                                    int v139;
                                    if (v137){
                                        int v138;
                                        v138 = v135 + 2;
                                        v139 = v138;
                                    } else {
                                        v139 = v135;
                                    }
                                    v130[v132] = v139;
                                    v132 += 1 ;
                                }
                                v147 = Union4{Union4_2{v49, false, v51, v114, v130, v115}};
                            } else {
                                printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                exit(-1);
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    break;
                }
                case 1: { // Some
                    Union1 v56 = v49.case1.v0;
                    switch (v55.tag) {
                        case 0: { // Call
                            if (v50){
                                int v58;
                                v58 = v52 ^ 1;
                                v147 = Union4{Union4_2{v49, false, v51, v58, v53, v54}};
                            } else {
                                int v60; int v61;
                                Tuple5 tmp6 = Tuple5{0, 0};
                                v60 = tmp6.v0; v61 = tmp6.v1;
                                while (while_method_1(v60)){
                                    int v64;
                                    v64 = v53[v60];
                                    bool v66;
                                    v66 = v61 >= v64;
                                    int v67;
                                    if (v66){
                                        v67 = v61;
                                    } else {
                                        v67 = v64;
                                    }
                                    v61 = v67;
                                    v60 += 1 ;
                                }
                                static_array<int,2> v69;
                                int v71;
                                v71 = 0;
                                while (while_method_1(v71)){
                                    v69[v71] = v61;
                                    v71 += 1 ;
                                }
                                v147 = Union4{Union4_4{v49, v50, v51, v52, v69, v54}};
                            }
                            break;
                        }
                        case 1: { // Fold
                            v147 = Union4{Union4_5{v49, v50, v51, v52, v53, v54}};
                            break;
                        }
                        case 2: { // Raise
                            bool v75;
                            v75 = v54 > 0;
                            if (v75){
                                int v76;
                                v76 = v52 ^ 1;
                                int v77;
                                v77 = -1 + v54;
                                int v78; int v79;
                                Tuple5 tmp7 = Tuple5{0, 0};
                                v78 = tmp7.v0; v79 = tmp7.v1;
                                while (while_method_1(v78)){
                                    int v82;
                                    v82 = v53[v78];
                                    bool v84;
                                    v84 = v79 >= v82;
                                    int v85;
                                    if (v84){
                                        v85 = v79;
                                    } else {
                                        v85 = v82;
                                    }
                                    v79 = v85;
                                    v78 += 1 ;
                                }
                                static_array<int,2> v87;
                                int v89;
                                v89 = 0;
                                while (while_method_1(v89)){
                                    v87[v89] = v79;
                                    v89 += 1 ;
                                }
                                static_array<int,2> v92;
                                int v94;
                                v94 = 0;
                                while (while_method_1(v94)){
                                    int v97;
                                    v97 = v87[v94];
                                    bool v99;
                                    v99 = v94 == v52;
                                    int v101;
                                    if (v99){
                                        int v100;
                                        v100 = v97 + 4;
                                        v101 = v100;
                                    } else {
                                        v101 = v97;
                                    }
                                    v92[v94] = v101;
                                    v94 += 1 ;
                                }
                                v147 = Union4{Union4_2{v49, false, v51, v76, v92, v77}};
                            } else {
                                printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                exit(-1);
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            return body_0(v0, v1, v2, v3, v4, v5, v147);
            break;
        }
        case 3: { // T_none
            float v8 = v6.v0;
            return v8;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 3;
    return v1;
}
static_array<float,3> masking_normalize_5(static_array<float,3> v0, static_array<bool,3> v1){
    int v2; float v3;
    Tuple3 tmp12 = Tuple3{0, 0.0f};
    v2 = tmp12.v0; v3 = tmp12.v1;
    while (while_method_4(v2)){
        bool v6;
        v6 = v1[v2];
        float v8;
        if (v6){
            v8 = 1.0f;
        } else {
            v8 = 0.0f;
        }
        float v9;
        v9 = v3 + v8;
        v3 = v9;
        v2 += 1 ;
    }
    float v10;
    v10 = 1.0f / v3;
    static_array<float,3> v12;
    int v14;
    v14 = 0;
    while (while_method_4(v14)){
        float v17;
        v17 = v0[v14];
        bool v20;
        v20 = v1[v14];
        float v22;
        if (v20){
            v22 = v17;
        } else {
            v22 = 0.0f;
        }
        v12[v14] = v22;
        v14 += 1 ;
    }
    int v23; float v24;
    Tuple3 tmp13 = Tuple3{0, 0.0f};
    v23 = tmp13.v0; v24 = tmp13.v1;
    while (while_method_4(v23)){
        float v27;
        v27 = v12[v23];
        float v29;
        v29 = v24 + v27;
        v24 = v29;
        v23 += 1 ;
    }
    static_array<float,3> v31;
    int v33;
    v33 = 0;
    while (while_method_4(v33)){
        float v36;
        v36 = v12[v33];
        bool v39;
        v39 = v1[v33];
        bool v41;
        v41 = v39 == false;
        float v46;
        if (v41){
            v46 = 0.0f;
        } else {
            bool v42;
            v42 = v24 == 0.0f;
            bool v43;
            v43 = v42 != true;
            if (v43){
                float v44;
                v44 = v36 / v24;
                v46 = v44;
            } else {
                v46 = v10;
            }
        }
        v31[v33] = v46;
        v33 += 1 ;
    }
    return v31;
}
static_array<float,3> relu_7(static_array<float,3> v0){
    static_array<float,3> v2;
    int v4;
    v4 = 0;
    while (while_method_4(v4)){
        float v7;
        v7 = v0[v4];
        bool v9;
        v9 = 0.0f >= v7;
        float v10;
        if (v9){
            v10 = 0.0f;
        } else {
            v10 = v7;
        }
        v2[v4] = v10;
        v4 += 1 ;
    }
    return v2;
}
static_array<float,3> regret_match_6(static_array<float,3> v0, static_array<bool,3> v1){
    static_array<float,3> v2;
    v2 = relu_7(v0);
    return masking_normalize_5(v2, v1);
}
inline bool while_method_5(static_array<float,3> v0, int v1){
    bool v2;
    v2 = v1 < 3;
    return v2;
}
inline bool while_method_6(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    return v2;
}
int loop_10(static_array<float,3> v0, float v1, int v2){
    bool v3;
    v3 = v2 < 3;
    if (v3){
        float v5;
        v5 = v0[v2];
        bool v7;
        v7 = v1 < v5;
        if (v7){
            return v2;
        } else {
            int v8;
            v8 = v2 + 1;
            return loop_10(v0, v1, v8);
        }
    } else {
        return 2;
    }
}
int pick_discrete__9(static_array<float,3> v0, float v1){
    static_array<float,3> v3;
    int v5;
    v5 = 0;
    while (while_method_4(v5)){
        float v8;
        v8 = v0[v5];
        v3[v5] = v8;
        v5 += 1 ;
    }
    int v10;
    v10 = 1;
    while (while_method_5(v3, v10)){
        int v12;
        v12 = 3;
        while (while_method_6(v10, v12)){
            v12 -= 1 ;
            int v14;
            v14 = v12 - v10;
            float v16;
            v16 = v3[v14];
            float v19;
            v19 = v3[v12];
            float v21;
            v21 = v16 + v19;
            v3[v12] = v21;
        }
        int v22;
        v22 = v10 * 2;
        v10 = v22;
    }
    float v24;
    v24 = v3[2];
    float v26;
    v26 = v1 * v24;
    int v27;
    v27 = 0;
    return loop_10(v3, v26, v27);
}
int sample_discrete__8(static_array<float,3> v0, xso::rng & v1){
    std::uniform_real_distribution<float> v2(0.0, 1.0);
    float v3;
    v3 = v2(v1);
    return pick_discrete__9(v0, v3);
}
int tag_12(Union1 v0){
    switch (v0.tag) {
        case 0: { // Jack
            return 0;
            break;
        }
        case 1: { // King
            return 2;
            break;
        }
        case 2: { // Queen
            return 1;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
bool is_pair_13(int v0, int v1){
    bool v2;
    v2 = v1 == v0;
    return v2;
}
Tuple5 order_14(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple5{v1, v0};
    } else {
        return Tuple5{v0, v1};
    }
}
Union9 compare_hands_11(Union5 v0, bool v1, static_array<Union1,2> v2, int v3, static_array<int,2> v4, int v5){
    switch (v0.tag) {
        case 0: { // None
            printf("%s\n", "Expected the community card to be present in the table.");
            exit(-1);
            break;
        }
        case 1: { // Some
            Union1 v7 = v0.case1.v0;
            int v8;
            v8 = tag_12(v7);
            Union1 v10;
            v10 = v2[0];
            int v12;
            v12 = tag_12(v10);
            Union1 v14;
            v14 = v2[1];
            int v16;
            v16 = tag_12(v14);
            bool v17;
            v17 = is_pair_13(v8, v12);
            bool v18;
            v18 = is_pair_13(v8, v16);
            if (v17){
                if (v18){
                    bool v19;
                    v19 = v12 < v16;
                    if (v19){
                        return Union9{Union9_2{}};
                    } else {
                        bool v21;
                        v21 = v12 > v16;
                        if (v21){
                            return Union9{Union9_1{}};
                        } else {
                            return Union9{Union9_0{}};
                        }
                    }
                } else {
                    return Union9{Union9_1{}};
                }
            } else {
                if (v18){
                    return Union9{Union9_2{}};
                } else {
                    int v29; int v30;
                    Tuple5 tmp39 = order_14(v8, v12);
                    v29 = tmp39.v0; v30 = tmp39.v1;
                    int v31; int v32;
                    Tuple5 tmp40 = order_14(v8, v16);
                    v31 = tmp40.v0; v32 = tmp40.v1;
                    bool v33;
                    v33 = v29 < v31;
                    Union9 v39;
                    if (v33){
                        v39 = Union9{Union9_2{}};
                    } else {
                        bool v35;
                        v35 = v29 > v31;
                        if (v35){
                            v39 = Union9{Union9_1{}};
                        } else {
                            v39 = Union9{Union9_0{}};
                        }
                    }
                    bool v40;
                    switch (v39.tag) {
                        case 0: { // Eq
                            v40 = true;
                            break;
                        }
                        default: {
                            v40 = false;
                        }
                    }
                    if (v40){
                        bool v41;
                        v41 = v30 < v32;
                        if (v41){
                            return Union9{Union9_2{}};
                        } else {
                            bool v43;
                            v43 = v30 > v32;
                            if (v43){
                                return Union9{Union9_1{}};
                            } else {
                                return Union9{Union9_0{}};
                            }
                        }
                    } else {
                        return v39;
                    }
                }
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
float body_0(static_array<Union3,2> v0, StackRefs0 & v1, xso::rng & v2, StackRefs1 & v3, StackMut0 & v4, StackRefs2 & v5, Union4 v6){
    StackMut1 v7{0.0f};
    switch (v6.tag) {
        case 0: { // ChanceCommunityCard
            Union5 v719 = v6.case0.v0; bool v720 = v6.case0.v1; static_array<Union1,2> v721 = v6.case0.v2; int v722 = v6.case0.v3; static_array<int,2> v723 = v6.case0.v4; int v724 = v6.case0.v5;
            unsigned int v725 = v4.v0;
            Union1 v726; unsigned int v727;
            Tuple4 tmp3 = draw_card_1(v2, v725);
            v726 = tmp3.v0; v727 = tmp3.v1;
            v4.v0 = v727;
            static_array_list<Union0,32> & v728 = v3.v0;
            Union0 v729;
            v729 = Union0{Union0_0{v726}};
            v728.push(v729);
            Union6 v730;
            v730 = Union6{Union6_0{v719, v720, v721, v722, v723, v724, v726}};
            float v731;
            v731 = loop_4(v0, v1, v2, v3, v4, v5, v7, v730);
            static_array_list<Union0,32> & v732 = v3.v0;
            Union0 v733;
            v733 = v732.pop();
            v4.v0 = v725;
            return v731;
            break;
        }
        case 1: { // ChanceInit
            unsigned int v734 = v4.v0;
            Union1 v735; unsigned int v736;
            Tuple4 tmp8 = draw_card_1(v2, v734);
            v735 = tmp8.v0; v736 = tmp8.v1;
            v4.v0 = v736;
            unsigned int v737 = v4.v0;
            Union1 v738; unsigned int v739;
            Tuple4 tmp9 = draw_card_1(v2, v737);
            v738 = tmp9.v0; v739 = tmp9.v1;
            v4.v0 = v739;
            static_array_list<Union0,32> & v740 = v3.v0;
            Union0 v741;
            v741 = Union0{Union0_2{0, v735}};
            v740.push(v741);
            static_array_list<Union0,32> & v742 = v3.v0;
            Union0 v743;
            v743 = Union0{Union0_2{1, v738}};
            v742.push(v743);
            Union6 v744;
            v744 = Union6{Union6_1{v735, v738}};
            float v745;
            v745 = loop_4(v0, v1, v2, v3, v4, v5, v7, v744);
            static_array_list<Union0,32> & v746 = v3.v0;
            Union0 v747;
            v747 = v746.pop();
            static_array_list<Union0,32> & v748 = v3.v0;
            Union0 v749;
            v749 = v748.pop();
            v4.v0 = v737;
            v4.v0 = v734;
            return v745;
            break;
        }
        case 2: { // Round
            Union5 v58 = v6.case2.v0; bool v59 = v6.case2.v1; static_array<Union1,2> v60 = v6.case2.v2; int v61 = v6.case2.v3; static_array<int,2> v62 = v6.case2.v4; int v63 = v6.case2.v5;
            static_array_list<Union0,32> & v64 = v3.v0;
            int v65;
            v65 = v64.length;
            bool v66;
            v66 = 32 >= v65;
            bool v67;
            v67 = v66 == false;
            if (v67){
                assert("The type level dimension has to equal the value passed at runtime into create." && v66);
            } else {
            }
            static_array_list<Union0,32> v70;
            v70 = static_array_list<Union0,32>{};
            v70.unsafe_set_length(v65);
            int v72; int v73;
            Tuple5 tmp10 = Tuple5{0, 0};
            v72 = tmp10.v0; v73 = tmp10.v1;
            while (while_method_0(v65, v72)){
                Union0 v76;
                v76 = v64[v72];
                bool v81;
                switch (v76.tag) {
                    case 2: { // PlayerGotCard
                        int v78 = v76.case2.v0; Union1 v79 = v76.case2.v1;
                        bool v80;
                        v80 = v78 == v61;
                        v81 = v80;
                        break;
                    }
                    default: {
                        v81 = true;
                    }
                }
                int v83;
                if (v81){
                    v70[v73] = v76;
                    int v82;
                    v82 = v73 + 1;
                    v83 = v82;
                } else {
                    v83 = v73;
                }
                v73 = v83;
                v72 += 1 ;
            }
            bool v84;
            v84 = 32 >= v73;
            bool v85;
            v85 = v84 == false;
            if (v85){
                assert("The type level dimension has to equal the value passed at runtime into create." && v84);
            } else {
            }
            static_array_list<Union0,32> v88;
            v88 = static_array_list<Union0,32>{};
            v88.unsafe_set_length(v73);
            int v90;
            v90 = 0;
            while (while_method_0(v73, v90)){
                Union0 v93;
                v93 = v70[v90];
                v88[v90] = v93;
                v90 += 1 ;
            }
            std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v95 = v1.v0;
            auto v96 = v95.find(v88);
            bool v97;
            v97 = v96 != v95.end();
            Union7 v103;
            if (v97){
                static_array<float,3> v98; static_array<float,3> v99; static_array<Tuple2,3> v100;
                Tuple1 tmp11 = v96->second;
                v98 = tmp11.v0; v99 = tmp11.v1; v100 = tmp11.v2;
                v103 = Union7{Union7_1{v98, v99, v100}};
            } else {
                v103 = Union7{Union7_0{}};
            }
            static_array<float,3> v125; static_array<float,3> v126; static_array<Tuple2,3> v127;
            switch (v103.tag) {
                case 0: { // None
                    static_array<float,3> v108;
                    int v110;
                    v110 = 0;
                    while (while_method_4(v110)){
                        v108[v110] = 0.0f;
                        v110 += 1 ;
                    }
                    static_array<float,3> v113;
                    int v115;
                    v115 = 0;
                    while (while_method_4(v115)){
                        v113[v115] = 0.0f;
                        v115 += 1 ;
                    }
                    static_array<Tuple2,3> v118;
                    int v120;
                    v120 = 0;
                    while (while_method_4(v120)){
                        v118[v120] = Tuple2{0.0f, 0.0f};
                        v120 += 1 ;
                    }
                    v125 = v108; v126 = v113; v127 = v118;
                    break;
                }
                case 1: { // Some
                    static_array<float,3> v104 = v103.case1.v0; static_array<float,3> v105 = v103.case1.v1; static_array<Tuple2,3> v106 = v103.case1.v2;
                    v125 = v104; v126 = v105; v127 = v106;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            int v129;
            v129 = v62[0];
            int v132;
            v132 = v62[1];
            bool v134;
            v134 = v129 == v132;
            bool v135;
            v135 = v134 != true;
            Union8 v139;
            if (v135){
                Union2 v136;
                v136 = Union2{Union2_1{}};
                v139 = Union8{Union8_1{v136}};
            } else {
                v139 = Union8{Union8_0{}};
            }
            bool v140;
            v140 = v63 > 0;
            Union8 v144;
            if (v140){
                Union2 v141;
                v141 = Union2{Union2_2{}};
                v144 = Union8{Union8_1{v141}};
            } else {
                v144 = Union8{Union8_0{}};
            }
            bool v147;
            switch (v144.tag) {
                case 0: { // None
                    v147 = false;
                    break;
                }
                case 1: { // Some
                    Union2 v145 = v144.case1.v0;
                    v147 = true;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            bool v150;
            switch (v139.tag) {
                case 0: { // None
                    v150 = false;
                    break;
                }
                case 1: { // Some
                    Union2 v148 = v139.case1.v0;
                    v150 = true;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            static_array<bool,3> v152;
            v152[0] = true;
            v152[1] = v150;
            v152[2] = v147;
            Union3 v155;
            v155 = v0[v61];
            float v700;
            switch (v155.tag) {
                case 0: { // Frozen
                    static_array<float,3> v598;
                    v598 = masking_normalize_5(v125, v152);
                    float v624;
                    switch (v144.tag) {
                        case 0: { // None
                            v624 = 0.0f;
                            break;
                        }
                        case 1: { // Some
                            Union2 v599 = v144.case1.v0;
                            float v601;
                            v601 = v598[2];
                            static_array<Tuple2,2> & v603 = v5.v0;
                            float v606; float v607;
                            Tuple2 tmp14 = v603[v61];
                            v606 = tmp14.v0; v607 = tmp14.v1;
                            static_array<Tuple2,2> & v610 = v5.v0;
                            float v611;
                            v611 = log(v601);
                            float v612;
                            v612 = v611 + v606;
                            v610[v61] = Tuple2{v612, v607};
                            static_array_list<Union0,32> & v613 = v3.v0;
                            Union0 v614;
                            v614 = Union0{Union0_1{v61, v599}};
                            v613.push(v614);
                            Union6 v615;
                            v615 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v599}};
                            float v616;
                            v616 = loop_4(v0, v1, v2, v3, v4, v5, v7, v615);
                            static_array_list<Union0,32> & v617 = v3.v0;
                            Union0 v618;
                            v618 = v617.pop();
                            static_array<Tuple2,2> & v619 = v5.v0;
                            v619[v61] = Tuple2{v606, v607};
                            bool v620;
                            v620 = v61 == 0;
                            if (v620){
                                v624 = v616;
                            } else {
                                float v621;
                                v621 = -v616;
                                v624 = v621;
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    float v650;
                    switch (v139.tag) {
                        case 0: { // None
                            v650 = 0.0f;
                            break;
                        }
                        case 1: { // Some
                            Union2 v625 = v139.case1.v0;
                            float v627;
                            v627 = v598[1];
                            static_array<Tuple2,2> & v629 = v5.v0;
                            float v632; float v633;
                            Tuple2 tmp15 = v629[v61];
                            v632 = tmp15.v0; v633 = tmp15.v1;
                            static_array<Tuple2,2> & v636 = v5.v0;
                            float v637;
                            v637 = log(v627);
                            float v638;
                            v638 = v637 + v632;
                            v636[v61] = Tuple2{v638, v633};
                            static_array_list<Union0,32> & v639 = v3.v0;
                            Union0 v640;
                            v640 = Union0{Union0_1{v61, v625}};
                            v639.push(v640);
                            Union6 v641;
                            v641 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v625}};
                            float v642;
                            v642 = loop_4(v0, v1, v2, v3, v4, v5, v7, v641);
                            static_array_list<Union0,32> & v643 = v3.v0;
                            Union0 v644;
                            v644 = v643.pop();
                            static_array<Tuple2,2> & v645 = v5.v0;
                            v645[v61] = Tuple2{v632, v633};
                            bool v646;
                            v646 = v61 == 0;
                            if (v646){
                                v650 = v642;
                            } else {
                                float v647;
                                v647 = -v642;
                                v650 = v647;
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    float v652;
                    v652 = v598[0];
                    static_array<Tuple2,2> & v654 = v5.v0;
                    float v657; float v658;
                    Tuple2 tmp16 = v654[v61];
                    v657 = tmp16.v0; v658 = tmp16.v1;
                    static_array<Tuple2,2> & v661 = v5.v0;
                    float v662;
                    v662 = log(v652);
                    float v663;
                    v663 = v662 + v657;
                    v661[v61] = Tuple2{v663, v658};
                    static_array_list<Union0,32> & v664 = v3.v0;
                    Union2 v665;
                    v665 = Union2{Union2_0{}};
                    Union0 v666;
                    v666 = Union0{Union0_1{v61, v665}};
                    v664.push(v666);
                    Union2 v667;
                    v667 = Union2{Union2_0{}};
                    Union6 v668;
                    v668 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v667}};
                    float v669;
                    v669 = loop_4(v0, v1, v2, v3, v4, v5, v7, v668);
                    static_array_list<Union0,32> & v670 = v3.v0;
                    Union0 v671;
                    v671 = v670.pop();
                    static_array<Tuple2,2> & v672 = v5.v0;
                    v672[v61] = Tuple2{v657, v658};
                    bool v673;
                    v673 = v61 == 0;
                    float v675;
                    if (v673){
                        v675 = v669;
                    } else {
                        float v674;
                        v674 = -v669;
                        v675 = v674;
                    }
                    static_array<float,3> v677;
                    v677[0] = v675;
                    v677[1] = v650;
                    v677[2] = v624;
                    static_array<float,3> v680;
                    int v682;
                    v682 = 0;
                    while (while_method_4(v682)){
                        float v685;
                        v685 = v677[v682];
                        float v688;
                        v688 = v598[v682];
                        float v690;
                        v690 = v685 * v688;
                        v680[v682] = v690;
                        v682 += 1 ;
                    }
                    int v691; float v692;
                    Tuple3 tmp17 = Tuple3{0, 0.0f};
                    v691 = tmp17.v0; v692 = tmp17.v1;
                    while (while_method_4(v691)){
                        float v695;
                        v695 = v680[v691];
                        float v697;
                        v697 = v692 + v695;
                        v692 = v697;
                        v691 += 1 ;
                    }
                    v700 = v692;
                    break;
                }
                case 1: { // TrainEnumerative
                    static_array<float,3> v157;
                    v157 = regret_match_6(v126, v152);
                    float v183;
                    switch (v144.tag) {
                        case 0: { // None
                            v183 = 0.0f;
                            break;
                        }
                        case 1: { // Some
                            Union2 v158 = v144.case1.v0;
                            float v160;
                            v160 = v157[2];
                            static_array<Tuple2,2> & v162 = v5.v0;
                            float v165; float v166;
                            Tuple2 tmp18 = v162[v61];
                            v165 = tmp18.v0; v166 = tmp18.v1;
                            static_array<Tuple2,2> & v169 = v5.v0;
                            float v170;
                            v170 = log(v160);
                            float v171;
                            v171 = v170 + v165;
                            v169[v61] = Tuple2{v171, v166};
                            static_array_list<Union0,32> & v172 = v3.v0;
                            Union0 v173;
                            v173 = Union0{Union0_1{v61, v158}};
                            v172.push(v173);
                            Union6 v174;
                            v174 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v158}};
                            float v175;
                            v175 = loop_4(v0, v1, v2, v3, v4, v5, v7, v174);
                            static_array_list<Union0,32> & v176 = v3.v0;
                            Union0 v177;
                            v177 = v176.pop();
                            static_array<Tuple2,2> & v178 = v5.v0;
                            v178[v61] = Tuple2{v165, v166};
                            bool v179;
                            v179 = v61 == 0;
                            if (v179){
                                v183 = v175;
                            } else {
                                float v180;
                                v180 = -v175;
                                v183 = v180;
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    float v209;
                    switch (v139.tag) {
                        case 0: { // None
                            v209 = 0.0f;
                            break;
                        }
                        case 1: { // Some
                            Union2 v184 = v139.case1.v0;
                            float v186;
                            v186 = v157[1];
                            static_array<Tuple2,2> & v188 = v5.v0;
                            float v191; float v192;
                            Tuple2 tmp19 = v188[v61];
                            v191 = tmp19.v0; v192 = tmp19.v1;
                            static_array<Tuple2,2> & v195 = v5.v0;
                            float v196;
                            v196 = log(v186);
                            float v197;
                            v197 = v196 + v191;
                            v195[v61] = Tuple2{v197, v192};
                            static_array_list<Union0,32> & v198 = v3.v0;
                            Union0 v199;
                            v199 = Union0{Union0_1{v61, v184}};
                            v198.push(v199);
                            Union6 v200;
                            v200 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v184}};
                            float v201;
                            v201 = loop_4(v0, v1, v2, v3, v4, v5, v7, v200);
                            static_array_list<Union0,32> & v202 = v3.v0;
                            Union0 v203;
                            v203 = v202.pop();
                            static_array<Tuple2,2> & v204 = v5.v0;
                            v204[v61] = Tuple2{v191, v192};
                            bool v205;
                            v205 = v61 == 0;
                            if (v205){
                                v209 = v201;
                            } else {
                                float v206;
                                v206 = -v201;
                                v209 = v206;
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    float v211;
                    v211 = v157[0];
                    static_array<Tuple2,2> & v213 = v5.v0;
                    float v216; float v217;
                    Tuple2 tmp20 = v213[v61];
                    v216 = tmp20.v0; v217 = tmp20.v1;
                    static_array<Tuple2,2> & v220 = v5.v0;
                    float v221;
                    v221 = log(v211);
                    float v222;
                    v222 = v221 + v216;
                    v220[v61] = Tuple2{v222, v217};
                    static_array_list<Union0,32> & v223 = v3.v0;
                    Union2 v224;
                    v224 = Union2{Union2_0{}};
                    Union0 v225;
                    v225 = Union0{Union0_1{v61, v224}};
                    v223.push(v225);
                    Union2 v226;
                    v226 = Union2{Union2_0{}};
                    Union6 v227;
                    v227 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v226}};
                    float v228;
                    v228 = loop_4(v0, v1, v2, v3, v4, v5, v7, v227);
                    static_array_list<Union0,32> & v229 = v3.v0;
                    Union0 v230;
                    v230 = v229.pop();
                    static_array<Tuple2,2> & v231 = v5.v0;
                    v231[v61] = Tuple2{v216, v217};
                    bool v232;
                    v232 = v61 == 0;
                    float v234;
                    if (v232){
                        v234 = v228;
                    } else {
                        float v233;
                        v233 = -v228;
                        v234 = v233;
                    }
                    static_array<float,3> v236;
                    v236[0] = v234;
                    v236[1] = v209;
                    v236[2] = v183;
                    static_array<float,3> v239;
                    int v241;
                    v241 = 0;
                    while (while_method_4(v241)){
                        float v244;
                        v244 = v236[v241];
                        float v247;
                        v247 = v157[v241];
                        float v249;
                        v249 = v244 * v247;
                        v239[v241] = v249;
                        v241 += 1 ;
                    }
                    int v250; float v251;
                    Tuple3 tmp21 = Tuple3{0, 0.0f};
                    v250 = tmp21.v0; v251 = tmp21.v1;
                    while (while_method_4(v250)){
                        float v254;
                        v254 = v239[v250];
                        float v256;
                        v256 = v251 + v254;
                        v251 = v256;
                        v250 += 1 ;
                    }
                    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v257 = v1.v0;
                    static_array<float,3> v259;
                    int v261;
                    v261 = 0;
                    while (while_method_4(v261)){
                        float v264;
                        v264 = v125[v261];
                        float v267;
                        v267 = v157[v261];
                        float v269;
                        v269 = 0.99609375f * v264;
                        float v270;
                        v270 = v269 + v267;
                        v259[v261] = v270;
                        v261 += 1 ;
                    }
                    static_array<Tuple2,2> & v271 = v5.v0;
                    int v272; float v273;
                    Tuple3 tmp22 = Tuple3{0, 0.0f};
                    v272 = tmp22.v0; v273 = tmp22.v1;
                    while (while_method_1(v272)){
                        float v277; float v278;
                        Tuple2 tmp23 = v271[v272];
                        v277 = tmp23.v0; v278 = tmp23.v1;
                        bool v281;
                        v281 = v272 == v61;
                        float v282;
                        if (v281){
                            v282 = 0.0f;
                        } else {
                            v282 = v277;
                        }
                        float v283;
                        v283 = v273 + v282;
                        float v284;
                        v284 = v283 - v278;
                        v273 = v284;
                        v272 += 1 ;
                    }
                    float v285;
                    v285 = exp(v273);
                    static_array<float,3> v287;
                    int v289;
                    v289 = 0;
                    while (while_method_4(v289)){
                        float v292;
                        v292 = v126[v289];
                        float v295;
                        v295 = v236[v289];
                        float v297;
                        v297 = v295 - v251;
                        float v298;
                        v298 = v285 * v297;
                        float v299;
                        v299 = v292 + v298;
                        bool v300;
                        v300 = 0.0f >= v299;
                        float v301;
                        if (v300){
                            v301 = 0.0f;
                        } else {
                            v301 = v299;
                        }
                        v287[v289] = v301;
                        v289 += 1 ;
                    }
                    v257[v88] = Tuple1{v259, v287, v127};
                    v700 = v251;
                    break;
                }
                case 2: { // TrainSampling
                    static_array<float,3> v302;
                    v302 = regret_match_6(v126, v152);
                    static_array<float,3> v304;
                    int v306;
                    v306 = 0;
                    while (while_method_4(v306)){
                        v304[v306] = 0.0f;
                        v306 += 1 ;
                    }
                    static_array<float,3> v308;
                    v308 = masking_normalize_5(v304, v152);
                    static_array<float,3> v310;
                    int v312;
                    v312 = 0;
                    while (while_method_4(v312)){
                        float v315;
                        v315 = v302[v312];
                        float v318;
                        v318 = v308[v312];
                        float v320;
                        v320 = 0.0f * v315;
                        float v321;
                        v321 = v320 + v318;
                        v310[v312] = v321;
                        v312 += 1 ;
                    }
                    int v322;
                    v322 = sample_discrete__8(v310, v2);
                    StackMut1 v323{0.0f};
                    float v389;
                    switch (v144.tag) {
                        case 1: { // Some
                            Union2 v324 = v144.case1.v0;
                            bool v325;
                            v325 = 2 == v322;
                            if (v325){
                                float v327;
                                v327 = v302[2];
                                float v330;
                                v330 = v310[2];
                                static_array<Tuple2,2> & v332 = v5.v0;
                                float v335; float v336;
                                Tuple2 tmp24 = v332[v61];
                                v335 = tmp24.v0; v336 = tmp24.v1;
                                static_array<Tuple2,2> & v339 = v5.v0;
                                float v340;
                                v340 = log(v327);
                                float v341;
                                v341 = v340 + v335;
                                float v342;
                                v342 = log(v330);
                                float v343;
                                v343 = v342 + v336;
                                v339[v61] = Tuple2{v341, v343};
                                static_array_list<Union0,32> & v344 = v3.v0;
                                Union0 v345;
                                v345 = Union0{Union0_1{v61, v324}};
                                v344.push(v345);
                                Union6 v346;
                                v346 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v324}};
                                float v347;
                                v347 = loop_4(v0, v1, v2, v3, v4, v5, v7, v346);
                                static_array_list<Union0,32> & v348 = v3.v0;
                                Union0 v349;
                                v349 = v348.pop();
                                static_array<Tuple2,2> & v350 = v5.v0;
                                v350[v61] = Tuple2{v335, v336};
                                bool v351;
                                v351 = v61 == 0;
                                float v353;
                                if (v351){
                                    v353 = v347;
                                } else {
                                    float v352;
                                    v352 = -v347;
                                    v353 = v352;
                                }
                                v323.v0 = v353;
                                float v356; float v357;
                                Tuple2 tmp25 = v127[2];
                                v356 = tmp25.v0; v357 = tmp25.v1;
                                bool v360;
                                v360 = v357 == 0.0f;
                                bool v361;
                                v361 = v360 != true;
                                float v363;
                                if (v361){
                                    float v362;
                                    v362 = v356 / v357;
                                    v363 = v362;
                                } else {
                                    v363 = 0.0f;
                                }
                                float v364 = v323.v0;
                                float v365;
                                v365 = v364 - v363;
                                float v366;
                                v366 = v365 / v330;
                                float v367;
                                v367 = v366 + v363;
                                v389 = v367;
                            } else {
                                float v370; float v371;
                                Tuple2 tmp26 = v127[2];
                                v370 = tmp26.v0; v371 = tmp26.v1;
                                bool v374;
                                v374 = v371 == 0.0f;
                                bool v375;
                                v375 = v374 != true;
                                if (v375){
                                    float v376;
                                    v376 = v370 / v371;
                                    v389 = v376;
                                } else {
                                    v389 = 0.0f;
                                }
                            }
                            break;
                        }
                        default: {
                            float v381; float v382;
                            Tuple2 tmp27 = v127[2];
                            v381 = tmp27.v0; v382 = tmp27.v1;
                            bool v385;
                            v385 = v382 == 0.0f;
                            bool v386;
                            v386 = v385 != true;
                            if (v386){
                                float v387;
                                v387 = v381 / v382;
                                v389 = v387;
                            } else {
                                v389 = 0.0f;
                            }
                        }
                    }
                    float v455;
                    switch (v139.tag) {
                        case 1: { // Some
                            Union2 v390 = v139.case1.v0;
                            bool v391;
                            v391 = 1 == v322;
                            if (v391){
                                float v393;
                                v393 = v302[1];
                                float v396;
                                v396 = v310[1];
                                static_array<Tuple2,2> & v398 = v5.v0;
                                float v401; float v402;
                                Tuple2 tmp28 = v398[v61];
                                v401 = tmp28.v0; v402 = tmp28.v1;
                                static_array<Tuple2,2> & v405 = v5.v0;
                                float v406;
                                v406 = log(v393);
                                float v407;
                                v407 = v406 + v401;
                                float v408;
                                v408 = log(v396);
                                float v409;
                                v409 = v408 + v402;
                                v405[v61] = Tuple2{v407, v409};
                                static_array_list<Union0,32> & v410 = v3.v0;
                                Union0 v411;
                                v411 = Union0{Union0_1{v61, v390}};
                                v410.push(v411);
                                Union6 v412;
                                v412 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v390}};
                                float v413;
                                v413 = loop_4(v0, v1, v2, v3, v4, v5, v7, v412);
                                static_array_list<Union0,32> & v414 = v3.v0;
                                Union0 v415;
                                v415 = v414.pop();
                                static_array<Tuple2,2> & v416 = v5.v0;
                                v416[v61] = Tuple2{v401, v402};
                                bool v417;
                                v417 = v61 == 0;
                                float v419;
                                if (v417){
                                    v419 = v413;
                                } else {
                                    float v418;
                                    v418 = -v413;
                                    v419 = v418;
                                }
                                v323.v0 = v419;
                                float v422; float v423;
                                Tuple2 tmp29 = v127[1];
                                v422 = tmp29.v0; v423 = tmp29.v1;
                                bool v426;
                                v426 = v423 == 0.0f;
                                bool v427;
                                v427 = v426 != true;
                                float v429;
                                if (v427){
                                    float v428;
                                    v428 = v422 / v423;
                                    v429 = v428;
                                } else {
                                    v429 = 0.0f;
                                }
                                float v430 = v323.v0;
                                float v431;
                                v431 = v430 - v429;
                                float v432;
                                v432 = v431 / v396;
                                float v433;
                                v433 = v432 + v429;
                                v455 = v433;
                            } else {
                                float v436; float v437;
                                Tuple2 tmp30 = v127[1];
                                v436 = tmp30.v0; v437 = tmp30.v1;
                                bool v440;
                                v440 = v437 == 0.0f;
                                bool v441;
                                v441 = v440 != true;
                                if (v441){
                                    float v442;
                                    v442 = v436 / v437;
                                    v455 = v442;
                                } else {
                                    v455 = 0.0f;
                                }
                            }
                            break;
                        }
                        default: {
                            float v447; float v448;
                            Tuple2 tmp31 = v127[1];
                            v447 = tmp31.v0; v448 = tmp31.v1;
                            bool v451;
                            v451 = v448 == 0.0f;
                            bool v452;
                            v452 = v451 != true;
                            if (v452){
                                float v453;
                                v453 = v447 / v448;
                                v455 = v453;
                            } else {
                                v455 = 0.0f;
                            }
                        }
                    }
                    bool v456;
                    v456 = 0 == v322;
                    float v511;
                    if (v456){
                        float v458;
                        v458 = v302[0];
                        float v461;
                        v461 = v310[0];
                        static_array<Tuple2,2> & v463 = v5.v0;
                        float v466; float v467;
                        Tuple2 tmp32 = v463[v61];
                        v466 = tmp32.v0; v467 = tmp32.v1;
                        static_array<Tuple2,2> & v470 = v5.v0;
                        float v471;
                        v471 = log(v458);
                        float v472;
                        v472 = v471 + v466;
                        float v473;
                        v473 = log(v461);
                        float v474;
                        v474 = v473 + v467;
                        v470[v61] = Tuple2{v472, v474};
                        static_array_list<Union0,32> & v475 = v3.v0;
                        Union2 v476;
                        v476 = Union2{Union2_0{}};
                        Union0 v477;
                        v477 = Union0{Union0_1{v61, v476}};
                        v475.push(v477);
                        Union2 v478;
                        v478 = Union2{Union2_0{}};
                        Union6 v479;
                        v479 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v478}};
                        float v480;
                        v480 = loop_4(v0, v1, v2, v3, v4, v5, v7, v479);
                        static_array_list<Union0,32> & v481 = v3.v0;
                        Union0 v482;
                        v482 = v481.pop();
                        static_array<Tuple2,2> & v483 = v5.v0;
                        v483[v61] = Tuple2{v466, v467};
                        bool v484;
                        v484 = v61 == 0;
                        float v486;
                        if (v484){
                            v486 = v480;
                        } else {
                            float v485;
                            v485 = -v480;
                            v486 = v485;
                        }
                        v323.v0 = v486;
                        float v489; float v490;
                        Tuple2 tmp33 = v127[0];
                        v489 = tmp33.v0; v490 = tmp33.v1;
                        bool v493;
                        v493 = v490 == 0.0f;
                        bool v494;
                        v494 = v493 != true;
                        float v496;
                        if (v494){
                            float v495;
                            v495 = v489 / v490;
                            v496 = v495;
                        } else {
                            v496 = 0.0f;
                        }
                        float v497 = v323.v0;
                        float v498;
                        v498 = v497 - v496;
                        float v499;
                        v499 = v498 / v461;
                        float v500;
                        v500 = v499 + v496;
                        v511 = v500;
                    } else {
                        float v503; float v504;
                        Tuple2 tmp34 = v127[0];
                        v503 = tmp34.v0; v504 = tmp34.v1;
                        bool v507;
                        v507 = v504 == 0.0f;
                        bool v508;
                        v508 = v507 != true;
                        if (v508){
                            float v509;
                            v509 = v503 / v504;
                            v511 = v509;
                        } else {
                            v511 = 0.0f;
                        }
                    }
                    static_array<float,3> v513;
                    v513[0] = v511;
                    v513[1] = v455;
                    v513[2] = v389;
                    static_array<float,3> v516;
                    int v518;
                    v518 = 0;
                    while (while_method_4(v518)){
                        float v521;
                        v521 = v513[v518];
                        float v524;
                        v524 = v302[v518];
                        float v526;
                        v526 = v521 * v524;
                        v516[v518] = v526;
                        v518 += 1 ;
                    }
                    int v527; float v528;
                    Tuple3 tmp35 = Tuple3{0, 0.0f};
                    v527 = tmp35.v0; v528 = tmp35.v1;
                    while (while_method_4(v527)){
                        float v531;
                        v531 = v516[v527];
                        float v533;
                        v533 = v528 + v531;
                        v528 = v533;
                        v527 += 1 ;
                    }
                    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v534 = v1.v0;
                    static_array<float,3> v536;
                    int v538;
                    v538 = 0;
                    while (while_method_4(v538)){
                        float v541;
                        v541 = v125[v538];
                        float v544;
                        v544 = v302[v538];
                        float v546;
                        v546 = 0.99902344f * v541;
                        float v547;
                        v547 = v546 + v544;
                        v536[v538] = v547;
                        v538 += 1 ;
                    }
                    static_array<Tuple2,2> & v548 = v5.v0;
                    int v549; float v550;
                    Tuple3 tmp36 = Tuple3{0, 0.0f};
                    v549 = tmp36.v0; v550 = tmp36.v1;
                    while (while_method_1(v549)){
                        float v554; float v555;
                        Tuple2 tmp37 = v548[v549];
                        v554 = tmp37.v0; v555 = tmp37.v1;
                        bool v558;
                        v558 = v549 == v61;
                        float v559;
                        if (v558){
                            v559 = 0.0f;
                        } else {
                            v559 = v554;
                        }
                        float v560;
                        v560 = v550 + v559;
                        float v561;
                        v561 = v560 - v555;
                        v550 = v561;
                        v549 += 1 ;
                    }
                    float v562;
                    v562 = exp(v550);
                    static_array<float,3> v564;
                    int v566;
                    v566 = 0;
                    while (while_method_4(v566)){
                        float v569;
                        v569 = v126[v566];
                        float v572;
                        v572 = v513[v566];
                        float v574;
                        v574 = v572 - v528;
                        float v575;
                        v575 = v562 * v574;
                        float v576;
                        v576 = v569 + v575;
                        bool v577;
                        v577 = 0.0f >= v576;
                        float v578;
                        if (v577){
                            v578 = 0.0f;
                        } else {
                            v578 = v576;
                        }
                        v564[v566] = v578;
                        v566 += 1 ;
                    }
                    static_array<Tuple2,3> v580;
                    int v582;
                    v582 = 0;
                    while (while_method_4(v582)){
                        float v586; float v587;
                        Tuple2 tmp38 = v127[v582];
                        v586 = tmp38.v0; v587 = tmp38.v1;
                        bool v590;
                        v590 = v322 == v582;
                        float v596; float v597;
                        if (v590){
                            float v591;
                            v591 = v586 * 0.5f;
                            float v592 = v323.v0;
                            float v593;
                            v593 = v591 + v592;
                            float v594;
                            v594 = v587 * 0.5f;
                            float v595;
                            v595 = v594 + 1.0f;
                            v596 = v593; v597 = v595;
                        } else {
                            v596 = v586; v597 = v587;
                        }
                        v580[v582] = Tuple2{v596, v597};
                        v582 += 1 ;
                    }
                    v534[v88] = Tuple1{v536, v564, v580};
                    v700 = v528;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            bool v701;
            v701 = v61 == 0;
            float v703;
            if (v701){
                v703 = v700;
            } else {
                float v702;
                v702 = -v700;
                v703 = v702;
            }
            v7.v0 = v703;
            Union6 v704;
            v704 = Union6{Union6_3{}};
            return loop_4(v0, v1, v2, v3, v4, v5, v7, v704);
            break;
        }
        case 3: { // RoundWithAction
            Union5 v706 = v6.case3.v0; bool v707 = v6.case3.v1; static_array<Union1,2> v708 = v6.case3.v2; int v709 = v6.case3.v3; static_array<int,2> v710 = v6.case3.v4; int v711 = v6.case3.v5; Union2 v712 = v6.case3.v6;
            static_array_list<Union0,32> & v713 = v3.v0;
            Union0 v714;
            v714 = Union0{Union0_1{v709, v712}};
            v713.push(v714);
            Union6 v715;
            v715 = Union6{Union6_2{v706, v707, v708, v709, v710, v711, v712}};
            float v716;
            v716 = loop_4(v0, v1, v2, v3, v4, v5, v7, v715);
            static_array_list<Union0,32> & v717 = v3.v0;
            Union0 v718;
            v718 = v717.pop();
            return v716;
            break;
        }
        case 4: { // TerminalCall
            Union5 v29 = v6.case4.v0; bool v30 = v6.case4.v1; static_array<Union1,2> v31 = v6.case4.v2; int v32 = v6.case4.v3; static_array<int,2> v33 = v6.case4.v4; int v34 = v6.case4.v5;
            int v36;
            v36 = v33[v32];
            Union9 v38;
            v38 = compare_hands_11(v29, v30, v31, v32, v33, v34);
            int v43; int v44;
            switch (v38.tag) {
                case 0: { // Eq
                    v43 = 0; v44 = -1;
                    break;
                }
                case 1: { // Gt
                    v43 = v36; v44 = 0;
                    break;
                }
                case 2: { // Lt
                    v43 = v36; v44 = 1;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            int v45;
            v45 = -v44;
            bool v46;
            v46 = v44 >= v45;
            int v47;
            if (v46){
                v47 = v44;
            } else {
                v47 = v45;
            }
            float v48;
            v48 = (float)v43;
            bool v49;
            v49 = v47 == 0;
            float v51;
            if (v49){
                v51 = v48;
            } else {
                float v50;
                v50 = -v48;
                v51 = v50;
            }
            v7.v0 = v51;
            static_array_list<Union0,32> & v52 = v3.v0;
            Union0 v53;
            v53 = Union0{Union0_3{v31, v43, v44}};
            v52.push(v53);
            Union6 v54;
            v54 = Union6{Union6_3{}};
            float v55;
            v55 = loop_4(v0, v1, v2, v3, v4, v5, v7, v54);
            static_array_list<Union0,32> & v56 = v3.v0;
            Union0 v57;
            v57 = v56.pop();
            return v55;
            break;
        }
        case 5: { // TerminalFold
            Union5 v8 = v6.case5.v0; bool v9 = v6.case5.v1; static_array<Union1,2> v10 = v6.case5.v2; int v11 = v6.case5.v3; static_array<int,2> v12 = v6.case5.v4; int v13 = v6.case5.v5;
            int v15;
            v15 = v12[v11];
            int v17;
            v17 = -v15;
            float v18;
            v18 = (float)v17;
            bool v19;
            v19 = v11 == 0;
            float v21;
            if (v19){
                v21 = v18;
            } else {
                float v20;
                v20 = -v18;
                v21 = v20;
            }
            v7.v0 = v21;
            int v22;
            v22 = v11 ^ 1;
            static_array_list<Union0,32> & v23 = v3.v0;
            Union0 v24;
            v24 = Union0{Union0_3{v10, v15, v22}};
            v23.push(v24);
            Union6 v25;
            v25 = Union6{Union6_3{}};
            float v26;
            v26 = loop_4(v0, v1, v2, v3, v4, v5, v7, v25);
            static_array_list<Union0,32> & v27 = v3.v0;
            Union0 v28;
            v28 = v27.pop();
            return v26;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
inline bool while_method_7(int v0){
    bool v1;
    v1 = v0 < 100;
    return v1;
}
inline bool while_method_8(int v0){
    bool v1;
    v1 = v0 < 6;
    return v1;
}
float loop_16(static_array<Union3,2> v0, StackRefs0 & v1, xso::rng & v2, StackRefs1 & v3, StackMut0 & v4, StackRefs2 & v5, StackMut1 & v6, Union6 v7){
    switch (v7.tag) {
        case 0: { // T_game_chance_community_card
            Union5 v9 = v7.case0.v0; bool v10 = v7.case0.v1; static_array<Union1,2> v11 = v7.case0.v2; int v12 = v7.case0.v3; static_array<int,2> v13 = v7.case0.v4; int v14 = v7.case0.v5; Union1 v15 = v7.case0.v6;
            int v16;
            v16 = 2;
            int v17; int v18;
            Tuple5 tmp43 = Tuple5{0, 0};
            v17 = tmp43.v0; v18 = tmp43.v1;
            while (while_method_1(v17)){
                int v21;
                v21 = v13[v17];
                bool v23;
                v23 = v18 >= v21;
                int v24;
                if (v23){
                    v24 = v18;
                } else {
                    v24 = v21;
                }
                v18 = v24;
                v17 += 1 ;
            }
            static_array<int,2> v26;
            int v28;
            v28 = 0;
            while (while_method_1(v28)){
                v26[v28] = v18;
                v28 += 1 ;
            }
            Union5 v30;
            v30 = Union5{Union5_1{v15}};
            bool v31;
            v31 = true;
            int v32;
            v32 = 0;
            Union4 v33;
            v33 = Union4{Union4_2{v30, v31, v11, v32, v26, v16}};
            return body_15(v0, v1, v2, v3, v4, v5, v33);
            break;
        }
        case 1: { // T_game_chance_init
            Union1 v35 = v7.case1.v0; Union1 v36 = v7.case1.v1;
            int v37;
            v37 = 2;
            static_array<int,2> v39;
            v39[0] = 1;
            v39[1] = 1;
            static_array<Union1,2> v42;
            v42[0] = v35;
            v42[1] = v36;
            Union5 v44;
            v44 = Union5{Union5_0{}};
            bool v45;
            v45 = true;
            int v46;
            v46 = 0;
            Union4 v47;
            v47 = Union4{Union4_2{v44, v45, v42, v46, v39, v37}};
            return body_15(v0, v1, v2, v3, v4, v5, v47);
            break;
        }
        case 2: { // T_game_round
            Union5 v49 = v7.case2.v0; bool v50 = v7.case2.v1; static_array<Union1,2> v51 = v7.case2.v2; int v52 = v7.case2.v3; static_array<int,2> v53 = v7.case2.v4; int v54 = v7.case2.v5; Union2 v55 = v7.case2.v6;
            Union4 v147;
            switch (v49.tag) {
                case 0: { // None
                    switch (v55.tag) {
                        case 0: { // Call
                            if (v50){
                                int v109;
                                v109 = v52 ^ 1;
                                v147 = Union4{Union4_2{v49, false, v51, v109, v53, v54}};
                            } else {
                                v147 = Union4{Union4_0{v49, v50, v51, v52, v53, v54}};
                            }
                            break;
                        }
                        case 1: { // Fold
                            v147 = Union4{Union4_5{v49, v50, v51, v52, v53, v54}};
                            break;
                        }
                        case 2: { // Raise
                            bool v113;
                            v113 = v54 > 0;
                            if (v113){
                                int v114;
                                v114 = v52 ^ 1;
                                int v115;
                                v115 = -1 + v54;
                                int v116; int v117;
                                Tuple5 tmp44 = Tuple5{0, 0};
                                v116 = tmp44.v0; v117 = tmp44.v1;
                                while (while_method_1(v116)){
                                    int v120;
                                    v120 = v53[v116];
                                    bool v122;
                                    v122 = v117 >= v120;
                                    int v123;
                                    if (v122){
                                        v123 = v117;
                                    } else {
                                        v123 = v120;
                                    }
                                    v117 = v123;
                                    v116 += 1 ;
                                }
                                static_array<int,2> v125;
                                int v127;
                                v127 = 0;
                                while (while_method_1(v127)){
                                    v125[v127] = v117;
                                    v127 += 1 ;
                                }
                                static_array<int,2> v130;
                                int v132;
                                v132 = 0;
                                while (while_method_1(v132)){
                                    int v135;
                                    v135 = v125[v132];
                                    bool v137;
                                    v137 = v132 == v52;
                                    int v139;
                                    if (v137){
                                        int v138;
                                        v138 = v135 + 2;
                                        v139 = v138;
                                    } else {
                                        v139 = v135;
                                    }
                                    v130[v132] = v139;
                                    v132 += 1 ;
                                }
                                v147 = Union4{Union4_2{v49, false, v51, v114, v130, v115}};
                            } else {
                                printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                exit(-1);
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    break;
                }
                case 1: { // Some
                    Union1 v56 = v49.case1.v0;
                    switch (v55.tag) {
                        case 0: { // Call
                            if (v50){
                                int v58;
                                v58 = v52 ^ 1;
                                v147 = Union4{Union4_2{v49, false, v51, v58, v53, v54}};
                            } else {
                                int v60; int v61;
                                Tuple5 tmp45 = Tuple5{0, 0};
                                v60 = tmp45.v0; v61 = tmp45.v1;
                                while (while_method_1(v60)){
                                    int v64;
                                    v64 = v53[v60];
                                    bool v66;
                                    v66 = v61 >= v64;
                                    int v67;
                                    if (v66){
                                        v67 = v61;
                                    } else {
                                        v67 = v64;
                                    }
                                    v61 = v67;
                                    v60 += 1 ;
                                }
                                static_array<int,2> v69;
                                int v71;
                                v71 = 0;
                                while (while_method_1(v71)){
                                    v69[v71] = v61;
                                    v71 += 1 ;
                                }
                                v147 = Union4{Union4_4{v49, v50, v51, v52, v69, v54}};
                            }
                            break;
                        }
                        case 1: { // Fold
                            v147 = Union4{Union4_5{v49, v50, v51, v52, v53, v54}};
                            break;
                        }
                        case 2: { // Raise
                            bool v75;
                            v75 = v54 > 0;
                            if (v75){
                                int v76;
                                v76 = v52 ^ 1;
                                int v77;
                                v77 = -1 + v54;
                                int v78; int v79;
                                Tuple5 tmp46 = Tuple5{0, 0};
                                v78 = tmp46.v0; v79 = tmp46.v1;
                                while (while_method_1(v78)){
                                    int v82;
                                    v82 = v53[v78];
                                    bool v84;
                                    v84 = v79 >= v82;
                                    int v85;
                                    if (v84){
                                        v85 = v79;
                                    } else {
                                        v85 = v82;
                                    }
                                    v79 = v85;
                                    v78 += 1 ;
                                }
                                static_array<int,2> v87;
                                int v89;
                                v89 = 0;
                                while (while_method_1(v89)){
                                    v87[v89] = v79;
                                    v89 += 1 ;
                                }
                                static_array<int,2> v92;
                                int v94;
                                v94 = 0;
                                while (while_method_1(v94)){
                                    int v97;
                                    v97 = v87[v94];
                                    bool v99;
                                    v99 = v94 == v52;
                                    int v101;
                                    if (v99){
                                        int v100;
                                        v100 = v97 + 4;
                                        v101 = v100;
                                    } else {
                                        v101 = v97;
                                    }
                                    v92[v94] = v101;
                                    v94 += 1 ;
                                }
                                v147 = Union4{Union4_2{v49, false, v51, v76, v92, v77}};
                            } else {
                                printf("%s\n", "Invalid action. The number of raises left is not positive.");
                                exit(-1);
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            return body_15(v0, v1, v2, v3, v4, v5, v147);
            break;
        }
        case 3: { // T_none
            float v8 = v6.v0;
            return v8;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
float body_15(static_array<Union3,2> v0, StackRefs0 & v1, xso::rng & v2, StackRefs1 & v3, StackMut0 & v4, StackRefs2 & v5, Union4 v6){
    StackMut1 v7{0.0f};
    switch (v6.tag) {
        case 0: { // ChanceCommunityCard
            Union5 v719 = v6.case0.v0; bool v720 = v6.case0.v1; static_array<Union1,2> v721 = v6.case0.v2; int v722 = v6.case0.v3; static_array<int,2> v723 = v6.case0.v4; int v724 = v6.case0.v5;
            int v725; float v726; float v727;
            Tuple6 tmp42 = Tuple6{0, 0.0f, 0.0f};
            v725 = tmp42.v0; v726 = tmp42.v1; v727 = tmp42.v2;
            while (while_method_8(v725)){
                unsigned int v729 = v4.v0;
                unsigned int v730;
                v730 = 1u << v725;
                unsigned int v731;
                v731 = v729 & v730;
                bool v732;
                v732 = v731 == 0u;
                bool v733;
                v733 = v732 != true;
                float v765; float v766;
                if (v733){
                    unsigned int v734 = v4.v0;
                    unsigned int v735;
                    v735 = v734 ^ v730;
                    v4.v0 = v735;
                    bool v736;
                    v736 = 0 == v725;
                    Union1 v754;
                    if (v736){
                        v754 = Union1{Union1_1{}};
                    } else {
                        bool v738;
                        v738 = 1 == v725;
                        if (v738){
                            v754 = Union1{Union1_1{}};
                        } else {
                            bool v740;
                            v740 = 2 == v725;
                            if (v740){
                                v754 = Union1{Union1_2{}};
                            } else {
                                bool v742;
                                v742 = 3 == v725;
                                if (v742){
                                    v754 = Union1{Union1_2{}};
                                } else {
                                    bool v744;
                                    v744 = 4 == v725;
                                    if (v744){
                                        v754 = Union1{Union1_0{}};
                                    } else {
                                        bool v746;
                                        v746 = 5 == v725;
                                        if (v746){
                                            v754 = Union1{Union1_0{}};
                                        } else {
                                            printf("%s\n", "Invalid int in int_to_card.");
                                            exit(-1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    static_array_list<Union0,32> & v755 = v3.v0;
                    Union0 v756;
                    v756 = Union0{Union0_0{v754}};
                    v755.push(v756);
                    Union6 v757;
                    v757 = Union6{Union6_0{v719, v720, v721, v722, v723, v724, v754}};
                    float v758;
                    v758 = loop_16(v0, v1, v2, v3, v4, v5, v7, v757);
                    static_array_list<Union0,32> & v759 = v3.v0;
                    Union0 v760;
                    v760 = v759.pop();
                    unsigned int v761 = v4.v0;
                    unsigned int v762;
                    v762 = v761 ^ v730;
                    v4.v0 = v762;
                    float v763;
                    v763 = v726 + v758;
                    float v764;
                    v764 = v727 + 1.0f;
                    v765 = v763; v766 = v764;
                } else {
                    v765 = v726; v766 = v727;
                }
                v726 = v765;
                v727 = v766;
                v725 += 1 ;
            }
            bool v767;
            v767 = v727 == 0.0f;
            bool v768;
            v768 = v767 != true;
            if (v768){
                float v769;
                v769 = v726 / v727;
                return v769;
            } else {
                return 0.0f;
            }
            break;
        }
        case 1: { // ChanceInit
            int v771; float v772; float v773;
            Tuple6 tmp47 = Tuple6{0, 0.0f, 0.0f};
            v771 = tmp47.v0; v772 = tmp47.v1; v773 = tmp47.v2;
            while (while_method_8(v771)){
                unsigned int v775 = v4.v0;
                unsigned int v776;
                v776 = 1u << v771;
                unsigned int v777;
                v777 = v775 & v776;
                bool v778;
                v778 = v777 == 0u;
                bool v779;
                v779 = v778 != true;
                float v855; float v856;
                if (v779){
                    unsigned int v780 = v4.v0;
                    unsigned int v781;
                    v781 = v780 ^ v776;
                    v4.v0 = v781;
                    bool v782;
                    v782 = 0 == v771;
                    Union1 v800;
                    if (v782){
                        v800 = Union1{Union1_1{}};
                    } else {
                        bool v784;
                        v784 = 1 == v771;
                        if (v784){
                            v800 = Union1{Union1_1{}};
                        } else {
                            bool v786;
                            v786 = 2 == v771;
                            if (v786){
                                v800 = Union1{Union1_2{}};
                            } else {
                                bool v788;
                                v788 = 3 == v771;
                                if (v788){
                                    v800 = Union1{Union1_2{}};
                                } else {
                                    bool v790;
                                    v790 = 4 == v771;
                                    if (v790){
                                        v800 = Union1{Union1_0{}};
                                    } else {
                                        bool v792;
                                        v792 = 5 == v771;
                                        if (v792){
                                            v800 = Union1{Union1_0{}};
                                        } else {
                                            printf("%s\n", "Invalid int in int_to_card.");
                                            exit(-1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    int v801; float v802; float v803;
                    Tuple6 tmp48 = Tuple6{0, 0.0f, 0.0f};
                    v801 = tmp48.v0; v802 = tmp48.v1; v803 = tmp48.v2;
                    while (while_method_8(v801)){
                        unsigned int v805 = v4.v0;
                        unsigned int v806;
                        v806 = 1u << v801;
                        unsigned int v807;
                        v807 = v805 & v806;
                        bool v808;
                        v808 = v807 == 0u;
                        bool v809;
                        v809 = v808 != true;
                        float v845; float v846;
                        if (v809){
                            unsigned int v810 = v4.v0;
                            unsigned int v811;
                            v811 = v810 ^ v806;
                            v4.v0 = v811;
                            bool v812;
                            v812 = 0 == v801;
                            Union1 v830;
                            if (v812){
                                v830 = Union1{Union1_1{}};
                            } else {
                                bool v814;
                                v814 = 1 == v801;
                                if (v814){
                                    v830 = Union1{Union1_1{}};
                                } else {
                                    bool v816;
                                    v816 = 2 == v801;
                                    if (v816){
                                        v830 = Union1{Union1_2{}};
                                    } else {
                                        bool v818;
                                        v818 = 3 == v801;
                                        if (v818){
                                            v830 = Union1{Union1_2{}};
                                        } else {
                                            bool v820;
                                            v820 = 4 == v801;
                                            if (v820){
                                                v830 = Union1{Union1_0{}};
                                            } else {
                                                bool v822;
                                                v822 = 5 == v801;
                                                if (v822){
                                                    v830 = Union1{Union1_0{}};
                                                } else {
                                                    printf("%s\n", "Invalid int in int_to_card.");
                                                    exit(-1);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            static_array_list<Union0,32> & v831 = v3.v0;
                            Union0 v832;
                            v832 = Union0{Union0_2{0, v800}};
                            v831.push(v832);
                            static_array_list<Union0,32> & v833 = v3.v0;
                            Union0 v834;
                            v834 = Union0{Union0_2{1, v830}};
                            v833.push(v834);
                            Union6 v835;
                            v835 = Union6{Union6_1{v800, v830}};
                            float v836;
                            v836 = loop_16(v0, v1, v2, v3, v4, v5, v7, v835);
                            static_array_list<Union0,32> & v837 = v3.v0;
                            Union0 v838;
                            v838 = v837.pop();
                            static_array_list<Union0,32> & v839 = v3.v0;
                            Union0 v840;
                            v840 = v839.pop();
                            unsigned int v841 = v4.v0;
                            unsigned int v842;
                            v842 = v841 ^ v806;
                            v4.v0 = v842;
                            float v843;
                            v843 = v802 + v836;
                            float v844;
                            v844 = v803 + 1.0f;
                            v845 = v843; v846 = v844;
                        } else {
                            v845 = v802; v846 = v803;
                        }
                        v802 = v845;
                        v803 = v846;
                        v801 += 1 ;
                    }
                    bool v847;
                    v847 = v803 == 0.0f;
                    bool v848;
                    v848 = v847 != true;
                    float v850;
                    if (v848){
                        float v849;
                        v849 = v802 / v803;
                        v850 = v849;
                    } else {
                        v850 = 0.0f;
                    }
                    unsigned int v851 = v4.v0;
                    unsigned int v852;
                    v852 = v851 ^ v776;
                    v4.v0 = v852;
                    float v853;
                    v853 = v772 + v850;
                    float v854;
                    v854 = v773 + 1.0f;
                    v855 = v853; v856 = v854;
                } else {
                    v855 = v772; v856 = v773;
                }
                v772 = v855;
                v773 = v856;
                v771 += 1 ;
            }
            bool v857;
            v857 = v773 == 0.0f;
            bool v858;
            v858 = v857 != true;
            if (v858){
                float v859;
                v859 = v772 / v773;
                return v859;
            } else {
                return 0.0f;
            }
            break;
        }
        case 2: { // Round
            Union5 v58 = v6.case2.v0; bool v59 = v6.case2.v1; static_array<Union1,2> v60 = v6.case2.v2; int v61 = v6.case2.v3; static_array<int,2> v62 = v6.case2.v4; int v63 = v6.case2.v5;
            static_array_list<Union0,32> & v64 = v3.v0;
            int v65;
            v65 = v64.length;
            bool v66;
            v66 = 32 >= v65;
            bool v67;
            v67 = v66 == false;
            if (v67){
                assert("The type level dimension has to equal the value passed at runtime into create." && v66);
            } else {
            }
            static_array_list<Union0,32> v70;
            v70 = static_array_list<Union0,32>{};
            v70.unsafe_set_length(v65);
            int v72; int v73;
            Tuple5 tmp49 = Tuple5{0, 0};
            v72 = tmp49.v0; v73 = tmp49.v1;
            while (while_method_0(v65, v72)){
                Union0 v76;
                v76 = v64[v72];
                bool v81;
                switch (v76.tag) {
                    case 2: { // PlayerGotCard
                        int v78 = v76.case2.v0; Union1 v79 = v76.case2.v1;
                        bool v80;
                        v80 = v78 == v61;
                        v81 = v80;
                        break;
                    }
                    default: {
                        v81 = true;
                    }
                }
                int v83;
                if (v81){
                    v70[v73] = v76;
                    int v82;
                    v82 = v73 + 1;
                    v83 = v82;
                } else {
                    v83 = v73;
                }
                v73 = v83;
                v72 += 1 ;
            }
            bool v84;
            v84 = 32 >= v73;
            bool v85;
            v85 = v84 == false;
            if (v85){
                assert("The type level dimension has to equal the value passed at runtime into create." && v84);
            } else {
            }
            static_array_list<Union0,32> v88;
            v88 = static_array_list<Union0,32>{};
            v88.unsafe_set_length(v73);
            int v90;
            v90 = 0;
            while (while_method_0(v73, v90)){
                Union0 v93;
                v93 = v70[v90];
                v88[v90] = v93;
                v90 += 1 ;
            }
            std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v95 = v1.v0;
            auto v96 = v95.find(v88);
            bool v97;
            v97 = v96 != v95.end();
            Union7 v103;
            if (v97){
                static_array<float,3> v98; static_array<float,3> v99; static_array<Tuple2,3> v100;
                Tuple1 tmp50 = v96->second;
                v98 = tmp50.v0; v99 = tmp50.v1; v100 = tmp50.v2;
                v103 = Union7{Union7_1{v98, v99, v100}};
            } else {
                v103 = Union7{Union7_0{}};
            }
            static_array<float,3> v125; static_array<float,3> v126; static_array<Tuple2,3> v127;
            switch (v103.tag) {
                case 0: { // None
                    static_array<float,3> v108;
                    int v110;
                    v110 = 0;
                    while (while_method_4(v110)){
                        v108[v110] = 0.0f;
                        v110 += 1 ;
                    }
                    static_array<float,3> v113;
                    int v115;
                    v115 = 0;
                    while (while_method_4(v115)){
                        v113[v115] = 0.0f;
                        v115 += 1 ;
                    }
                    static_array<Tuple2,3> v118;
                    int v120;
                    v120 = 0;
                    while (while_method_4(v120)){
                        v118[v120] = Tuple2{0.0f, 0.0f};
                        v120 += 1 ;
                    }
                    v125 = v108; v126 = v113; v127 = v118;
                    break;
                }
                case 1: { // Some
                    static_array<float,3> v104 = v103.case1.v0; static_array<float,3> v105 = v103.case1.v1; static_array<Tuple2,3> v106 = v103.case1.v2;
                    v125 = v104; v126 = v105; v127 = v106;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            int v129;
            v129 = v62[0];
            int v132;
            v132 = v62[1];
            bool v134;
            v134 = v129 == v132;
            bool v135;
            v135 = v134 != true;
            Union8 v139;
            if (v135){
                Union2 v136;
                v136 = Union2{Union2_1{}};
                v139 = Union8{Union8_1{v136}};
            } else {
                v139 = Union8{Union8_0{}};
            }
            bool v140;
            v140 = v63 > 0;
            Union8 v144;
            if (v140){
                Union2 v141;
                v141 = Union2{Union2_2{}};
                v144 = Union8{Union8_1{v141}};
            } else {
                v144 = Union8{Union8_0{}};
            }
            bool v147;
            switch (v144.tag) {
                case 0: { // None
                    v147 = false;
                    break;
                }
                case 1: { // Some
                    Union2 v145 = v144.case1.v0;
                    v147 = true;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            bool v150;
            switch (v139.tag) {
                case 0: { // None
                    v150 = false;
                    break;
                }
                case 1: { // Some
                    Union2 v148 = v139.case1.v0;
                    v150 = true;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            static_array<bool,3> v152;
            v152[0] = true;
            v152[1] = v150;
            v152[2] = v147;
            Union3 v155;
            v155 = v0[v61];
            float v700;
            switch (v155.tag) {
                case 0: { // Frozen
                    static_array<float,3> v598;
                    v598 = masking_normalize_5(v125, v152);
                    float v624;
                    switch (v144.tag) {
                        case 0: { // None
                            v624 = 0.0f;
                            break;
                        }
                        case 1: { // Some
                            Union2 v599 = v144.case1.v0;
                            float v601;
                            v601 = v598[2];
                            static_array<Tuple2,2> & v603 = v5.v0;
                            float v606; float v607;
                            Tuple2 tmp51 = v603[v61];
                            v606 = tmp51.v0; v607 = tmp51.v1;
                            static_array<Tuple2,2> & v610 = v5.v0;
                            float v611;
                            v611 = log(v601);
                            float v612;
                            v612 = v611 + v606;
                            v610[v61] = Tuple2{v612, v607};
                            static_array_list<Union0,32> & v613 = v3.v0;
                            Union0 v614;
                            v614 = Union0{Union0_1{v61, v599}};
                            v613.push(v614);
                            Union6 v615;
                            v615 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v599}};
                            float v616;
                            v616 = loop_16(v0, v1, v2, v3, v4, v5, v7, v615);
                            static_array_list<Union0,32> & v617 = v3.v0;
                            Union0 v618;
                            v618 = v617.pop();
                            static_array<Tuple2,2> & v619 = v5.v0;
                            v619[v61] = Tuple2{v606, v607};
                            bool v620;
                            v620 = v61 == 0;
                            if (v620){
                                v624 = v616;
                            } else {
                                float v621;
                                v621 = -v616;
                                v624 = v621;
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    float v650;
                    switch (v139.tag) {
                        case 0: { // None
                            v650 = 0.0f;
                            break;
                        }
                        case 1: { // Some
                            Union2 v625 = v139.case1.v0;
                            float v627;
                            v627 = v598[1];
                            static_array<Tuple2,2> & v629 = v5.v0;
                            float v632; float v633;
                            Tuple2 tmp52 = v629[v61];
                            v632 = tmp52.v0; v633 = tmp52.v1;
                            static_array<Tuple2,2> & v636 = v5.v0;
                            float v637;
                            v637 = log(v627);
                            float v638;
                            v638 = v637 + v632;
                            v636[v61] = Tuple2{v638, v633};
                            static_array_list<Union0,32> & v639 = v3.v0;
                            Union0 v640;
                            v640 = Union0{Union0_1{v61, v625}};
                            v639.push(v640);
                            Union6 v641;
                            v641 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v625}};
                            float v642;
                            v642 = loop_16(v0, v1, v2, v3, v4, v5, v7, v641);
                            static_array_list<Union0,32> & v643 = v3.v0;
                            Union0 v644;
                            v644 = v643.pop();
                            static_array<Tuple2,2> & v645 = v5.v0;
                            v645[v61] = Tuple2{v632, v633};
                            bool v646;
                            v646 = v61 == 0;
                            if (v646){
                                v650 = v642;
                            } else {
                                float v647;
                                v647 = -v642;
                                v650 = v647;
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    float v652;
                    v652 = v598[0];
                    static_array<Tuple2,2> & v654 = v5.v0;
                    float v657; float v658;
                    Tuple2 tmp53 = v654[v61];
                    v657 = tmp53.v0; v658 = tmp53.v1;
                    static_array<Tuple2,2> & v661 = v5.v0;
                    float v662;
                    v662 = log(v652);
                    float v663;
                    v663 = v662 + v657;
                    v661[v61] = Tuple2{v663, v658};
                    static_array_list<Union0,32> & v664 = v3.v0;
                    Union2 v665;
                    v665 = Union2{Union2_0{}};
                    Union0 v666;
                    v666 = Union0{Union0_1{v61, v665}};
                    v664.push(v666);
                    Union2 v667;
                    v667 = Union2{Union2_0{}};
                    Union6 v668;
                    v668 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v667}};
                    float v669;
                    v669 = loop_16(v0, v1, v2, v3, v4, v5, v7, v668);
                    static_array_list<Union0,32> & v670 = v3.v0;
                    Union0 v671;
                    v671 = v670.pop();
                    static_array<Tuple2,2> & v672 = v5.v0;
                    v672[v61] = Tuple2{v657, v658};
                    bool v673;
                    v673 = v61 == 0;
                    float v675;
                    if (v673){
                        v675 = v669;
                    } else {
                        float v674;
                        v674 = -v669;
                        v675 = v674;
                    }
                    static_array<float,3> v677;
                    v677[0] = v675;
                    v677[1] = v650;
                    v677[2] = v624;
                    static_array<float,3> v680;
                    int v682;
                    v682 = 0;
                    while (while_method_4(v682)){
                        float v685;
                        v685 = v677[v682];
                        float v688;
                        v688 = v598[v682];
                        float v690;
                        v690 = v685 * v688;
                        v680[v682] = v690;
                        v682 += 1 ;
                    }
                    int v691; float v692;
                    Tuple3 tmp54 = Tuple3{0, 0.0f};
                    v691 = tmp54.v0; v692 = tmp54.v1;
                    while (while_method_4(v691)){
                        float v695;
                        v695 = v680[v691];
                        float v697;
                        v697 = v692 + v695;
                        v692 = v697;
                        v691 += 1 ;
                    }
                    v700 = v692;
                    break;
                }
                case 1: { // TrainEnumerative
                    static_array<float,3> v157;
                    v157 = regret_match_6(v126, v152);
                    float v183;
                    switch (v144.tag) {
                        case 0: { // None
                            v183 = 0.0f;
                            break;
                        }
                        case 1: { // Some
                            Union2 v158 = v144.case1.v0;
                            float v160;
                            v160 = v157[2];
                            static_array<Tuple2,2> & v162 = v5.v0;
                            float v165; float v166;
                            Tuple2 tmp55 = v162[v61];
                            v165 = tmp55.v0; v166 = tmp55.v1;
                            static_array<Tuple2,2> & v169 = v5.v0;
                            float v170;
                            v170 = log(v160);
                            float v171;
                            v171 = v170 + v165;
                            v169[v61] = Tuple2{v171, v166};
                            static_array_list<Union0,32> & v172 = v3.v0;
                            Union0 v173;
                            v173 = Union0{Union0_1{v61, v158}};
                            v172.push(v173);
                            Union6 v174;
                            v174 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v158}};
                            float v175;
                            v175 = loop_16(v0, v1, v2, v3, v4, v5, v7, v174);
                            static_array_list<Union0,32> & v176 = v3.v0;
                            Union0 v177;
                            v177 = v176.pop();
                            static_array<Tuple2,2> & v178 = v5.v0;
                            v178[v61] = Tuple2{v165, v166};
                            bool v179;
                            v179 = v61 == 0;
                            if (v179){
                                v183 = v175;
                            } else {
                                float v180;
                                v180 = -v175;
                                v183 = v180;
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    float v209;
                    switch (v139.tag) {
                        case 0: { // None
                            v209 = 0.0f;
                            break;
                        }
                        case 1: { // Some
                            Union2 v184 = v139.case1.v0;
                            float v186;
                            v186 = v157[1];
                            static_array<Tuple2,2> & v188 = v5.v0;
                            float v191; float v192;
                            Tuple2 tmp56 = v188[v61];
                            v191 = tmp56.v0; v192 = tmp56.v1;
                            static_array<Tuple2,2> & v195 = v5.v0;
                            float v196;
                            v196 = log(v186);
                            float v197;
                            v197 = v196 + v191;
                            v195[v61] = Tuple2{v197, v192};
                            static_array_list<Union0,32> & v198 = v3.v0;
                            Union0 v199;
                            v199 = Union0{Union0_1{v61, v184}};
                            v198.push(v199);
                            Union6 v200;
                            v200 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v184}};
                            float v201;
                            v201 = loop_16(v0, v1, v2, v3, v4, v5, v7, v200);
                            static_array_list<Union0,32> & v202 = v3.v0;
                            Union0 v203;
                            v203 = v202.pop();
                            static_array<Tuple2,2> & v204 = v5.v0;
                            v204[v61] = Tuple2{v191, v192};
                            bool v205;
                            v205 = v61 == 0;
                            if (v205){
                                v209 = v201;
                            } else {
                                float v206;
                                v206 = -v201;
                                v209 = v206;
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    float v211;
                    v211 = v157[0];
                    static_array<Tuple2,2> & v213 = v5.v0;
                    float v216; float v217;
                    Tuple2 tmp57 = v213[v61];
                    v216 = tmp57.v0; v217 = tmp57.v1;
                    static_array<Tuple2,2> & v220 = v5.v0;
                    float v221;
                    v221 = log(v211);
                    float v222;
                    v222 = v221 + v216;
                    v220[v61] = Tuple2{v222, v217};
                    static_array_list<Union0,32> & v223 = v3.v0;
                    Union2 v224;
                    v224 = Union2{Union2_0{}};
                    Union0 v225;
                    v225 = Union0{Union0_1{v61, v224}};
                    v223.push(v225);
                    Union2 v226;
                    v226 = Union2{Union2_0{}};
                    Union6 v227;
                    v227 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v226}};
                    float v228;
                    v228 = loop_16(v0, v1, v2, v3, v4, v5, v7, v227);
                    static_array_list<Union0,32> & v229 = v3.v0;
                    Union0 v230;
                    v230 = v229.pop();
                    static_array<Tuple2,2> & v231 = v5.v0;
                    v231[v61] = Tuple2{v216, v217};
                    bool v232;
                    v232 = v61 == 0;
                    float v234;
                    if (v232){
                        v234 = v228;
                    } else {
                        float v233;
                        v233 = -v228;
                        v234 = v233;
                    }
                    static_array<float,3> v236;
                    v236[0] = v234;
                    v236[1] = v209;
                    v236[2] = v183;
                    static_array<float,3> v239;
                    int v241;
                    v241 = 0;
                    while (while_method_4(v241)){
                        float v244;
                        v244 = v236[v241];
                        float v247;
                        v247 = v157[v241];
                        float v249;
                        v249 = v244 * v247;
                        v239[v241] = v249;
                        v241 += 1 ;
                    }
                    int v250; float v251;
                    Tuple3 tmp58 = Tuple3{0, 0.0f};
                    v250 = tmp58.v0; v251 = tmp58.v1;
                    while (while_method_4(v250)){
                        float v254;
                        v254 = v239[v250];
                        float v256;
                        v256 = v251 + v254;
                        v251 = v256;
                        v250 += 1 ;
                    }
                    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v257 = v1.v0;
                    static_array<float,3> v259;
                    int v261;
                    v261 = 0;
                    while (while_method_4(v261)){
                        float v264;
                        v264 = v125[v261];
                        float v267;
                        v267 = v157[v261];
                        float v269;
                        v269 = 0.99609375f * v264;
                        float v270;
                        v270 = v269 + v267;
                        v259[v261] = v270;
                        v261 += 1 ;
                    }
                    static_array<Tuple2,2> & v271 = v5.v0;
                    int v272; float v273;
                    Tuple3 tmp59 = Tuple3{0, 0.0f};
                    v272 = tmp59.v0; v273 = tmp59.v1;
                    while (while_method_1(v272)){
                        float v277; float v278;
                        Tuple2 tmp60 = v271[v272];
                        v277 = tmp60.v0; v278 = tmp60.v1;
                        bool v281;
                        v281 = v272 == v61;
                        float v282;
                        if (v281){
                            v282 = 0.0f;
                        } else {
                            v282 = v277;
                        }
                        float v283;
                        v283 = v273 + v282;
                        float v284;
                        v284 = v283 - v278;
                        v273 = v284;
                        v272 += 1 ;
                    }
                    float v285;
                    v285 = exp(v273);
                    static_array<float,3> v287;
                    int v289;
                    v289 = 0;
                    while (while_method_4(v289)){
                        float v292;
                        v292 = v126[v289];
                        float v295;
                        v295 = v236[v289];
                        float v297;
                        v297 = v295 - v251;
                        float v298;
                        v298 = v285 * v297;
                        float v299;
                        v299 = v292 + v298;
                        bool v300;
                        v300 = 0.0f >= v299;
                        float v301;
                        if (v300){
                            v301 = 0.0f;
                        } else {
                            v301 = v299;
                        }
                        v287[v289] = v301;
                        v289 += 1 ;
                    }
                    v257[v88] = Tuple1{v259, v287, v127};
                    v700 = v251;
                    break;
                }
                case 2: { // TrainSampling
                    static_array<float,3> v302;
                    v302 = regret_match_6(v126, v152);
                    static_array<float,3> v304;
                    int v306;
                    v306 = 0;
                    while (while_method_4(v306)){
                        v304[v306] = 0.0f;
                        v306 += 1 ;
                    }
                    static_array<float,3> v308;
                    v308 = masking_normalize_5(v304, v152);
                    static_array<float,3> v310;
                    int v312;
                    v312 = 0;
                    while (while_method_4(v312)){
                        float v315;
                        v315 = v302[v312];
                        float v318;
                        v318 = v308[v312];
                        float v320;
                        v320 = 0.0f * v315;
                        float v321;
                        v321 = v320 + v318;
                        v310[v312] = v321;
                        v312 += 1 ;
                    }
                    int v322;
                    v322 = sample_discrete__8(v310, v2);
                    StackMut1 v323{0.0f};
                    float v389;
                    switch (v144.tag) {
                        case 1: { // Some
                            Union2 v324 = v144.case1.v0;
                            bool v325;
                            v325 = 2 == v322;
                            if (v325){
                                float v327;
                                v327 = v302[2];
                                float v330;
                                v330 = v310[2];
                                static_array<Tuple2,2> & v332 = v5.v0;
                                float v335; float v336;
                                Tuple2 tmp61 = v332[v61];
                                v335 = tmp61.v0; v336 = tmp61.v1;
                                static_array<Tuple2,2> & v339 = v5.v0;
                                float v340;
                                v340 = log(v327);
                                float v341;
                                v341 = v340 + v335;
                                float v342;
                                v342 = log(v330);
                                float v343;
                                v343 = v342 + v336;
                                v339[v61] = Tuple2{v341, v343};
                                static_array_list<Union0,32> & v344 = v3.v0;
                                Union0 v345;
                                v345 = Union0{Union0_1{v61, v324}};
                                v344.push(v345);
                                Union6 v346;
                                v346 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v324}};
                                float v347;
                                v347 = loop_16(v0, v1, v2, v3, v4, v5, v7, v346);
                                static_array_list<Union0,32> & v348 = v3.v0;
                                Union0 v349;
                                v349 = v348.pop();
                                static_array<Tuple2,2> & v350 = v5.v0;
                                v350[v61] = Tuple2{v335, v336};
                                bool v351;
                                v351 = v61 == 0;
                                float v353;
                                if (v351){
                                    v353 = v347;
                                } else {
                                    float v352;
                                    v352 = -v347;
                                    v353 = v352;
                                }
                                v323.v0 = v353;
                                float v356; float v357;
                                Tuple2 tmp62 = v127[2];
                                v356 = tmp62.v0; v357 = tmp62.v1;
                                bool v360;
                                v360 = v357 == 0.0f;
                                bool v361;
                                v361 = v360 != true;
                                float v363;
                                if (v361){
                                    float v362;
                                    v362 = v356 / v357;
                                    v363 = v362;
                                } else {
                                    v363 = 0.0f;
                                }
                                float v364 = v323.v0;
                                float v365;
                                v365 = v364 - v363;
                                float v366;
                                v366 = v365 / v330;
                                float v367;
                                v367 = v366 + v363;
                                v389 = v367;
                            } else {
                                float v370; float v371;
                                Tuple2 tmp63 = v127[2];
                                v370 = tmp63.v0; v371 = tmp63.v1;
                                bool v374;
                                v374 = v371 == 0.0f;
                                bool v375;
                                v375 = v374 != true;
                                if (v375){
                                    float v376;
                                    v376 = v370 / v371;
                                    v389 = v376;
                                } else {
                                    v389 = 0.0f;
                                }
                            }
                            break;
                        }
                        default: {
                            float v381; float v382;
                            Tuple2 tmp64 = v127[2];
                            v381 = tmp64.v0; v382 = tmp64.v1;
                            bool v385;
                            v385 = v382 == 0.0f;
                            bool v386;
                            v386 = v385 != true;
                            if (v386){
                                float v387;
                                v387 = v381 / v382;
                                v389 = v387;
                            } else {
                                v389 = 0.0f;
                            }
                        }
                    }
                    float v455;
                    switch (v139.tag) {
                        case 1: { // Some
                            Union2 v390 = v139.case1.v0;
                            bool v391;
                            v391 = 1 == v322;
                            if (v391){
                                float v393;
                                v393 = v302[1];
                                float v396;
                                v396 = v310[1];
                                static_array<Tuple2,2> & v398 = v5.v0;
                                float v401; float v402;
                                Tuple2 tmp65 = v398[v61];
                                v401 = tmp65.v0; v402 = tmp65.v1;
                                static_array<Tuple2,2> & v405 = v5.v0;
                                float v406;
                                v406 = log(v393);
                                float v407;
                                v407 = v406 + v401;
                                float v408;
                                v408 = log(v396);
                                float v409;
                                v409 = v408 + v402;
                                v405[v61] = Tuple2{v407, v409};
                                static_array_list<Union0,32> & v410 = v3.v0;
                                Union0 v411;
                                v411 = Union0{Union0_1{v61, v390}};
                                v410.push(v411);
                                Union6 v412;
                                v412 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v390}};
                                float v413;
                                v413 = loop_16(v0, v1, v2, v3, v4, v5, v7, v412);
                                static_array_list<Union0,32> & v414 = v3.v0;
                                Union0 v415;
                                v415 = v414.pop();
                                static_array<Tuple2,2> & v416 = v5.v0;
                                v416[v61] = Tuple2{v401, v402};
                                bool v417;
                                v417 = v61 == 0;
                                float v419;
                                if (v417){
                                    v419 = v413;
                                } else {
                                    float v418;
                                    v418 = -v413;
                                    v419 = v418;
                                }
                                v323.v0 = v419;
                                float v422; float v423;
                                Tuple2 tmp66 = v127[1];
                                v422 = tmp66.v0; v423 = tmp66.v1;
                                bool v426;
                                v426 = v423 == 0.0f;
                                bool v427;
                                v427 = v426 != true;
                                float v429;
                                if (v427){
                                    float v428;
                                    v428 = v422 / v423;
                                    v429 = v428;
                                } else {
                                    v429 = 0.0f;
                                }
                                float v430 = v323.v0;
                                float v431;
                                v431 = v430 - v429;
                                float v432;
                                v432 = v431 / v396;
                                float v433;
                                v433 = v432 + v429;
                                v455 = v433;
                            } else {
                                float v436; float v437;
                                Tuple2 tmp67 = v127[1];
                                v436 = tmp67.v0; v437 = tmp67.v1;
                                bool v440;
                                v440 = v437 == 0.0f;
                                bool v441;
                                v441 = v440 != true;
                                if (v441){
                                    float v442;
                                    v442 = v436 / v437;
                                    v455 = v442;
                                } else {
                                    v455 = 0.0f;
                                }
                            }
                            break;
                        }
                        default: {
                            float v447; float v448;
                            Tuple2 tmp68 = v127[1];
                            v447 = tmp68.v0; v448 = tmp68.v1;
                            bool v451;
                            v451 = v448 == 0.0f;
                            bool v452;
                            v452 = v451 != true;
                            if (v452){
                                float v453;
                                v453 = v447 / v448;
                                v455 = v453;
                            } else {
                                v455 = 0.0f;
                            }
                        }
                    }
                    bool v456;
                    v456 = 0 == v322;
                    float v511;
                    if (v456){
                        float v458;
                        v458 = v302[0];
                        float v461;
                        v461 = v310[0];
                        static_array<Tuple2,2> & v463 = v5.v0;
                        float v466; float v467;
                        Tuple2 tmp69 = v463[v61];
                        v466 = tmp69.v0; v467 = tmp69.v1;
                        static_array<Tuple2,2> & v470 = v5.v0;
                        float v471;
                        v471 = log(v458);
                        float v472;
                        v472 = v471 + v466;
                        float v473;
                        v473 = log(v461);
                        float v474;
                        v474 = v473 + v467;
                        v470[v61] = Tuple2{v472, v474};
                        static_array_list<Union0,32> & v475 = v3.v0;
                        Union2 v476;
                        v476 = Union2{Union2_0{}};
                        Union0 v477;
                        v477 = Union0{Union0_1{v61, v476}};
                        v475.push(v477);
                        Union2 v478;
                        v478 = Union2{Union2_0{}};
                        Union6 v479;
                        v479 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v478}};
                        float v480;
                        v480 = loop_16(v0, v1, v2, v3, v4, v5, v7, v479);
                        static_array_list<Union0,32> & v481 = v3.v0;
                        Union0 v482;
                        v482 = v481.pop();
                        static_array<Tuple2,2> & v483 = v5.v0;
                        v483[v61] = Tuple2{v466, v467};
                        bool v484;
                        v484 = v61 == 0;
                        float v486;
                        if (v484){
                            v486 = v480;
                        } else {
                            float v485;
                            v485 = -v480;
                            v486 = v485;
                        }
                        v323.v0 = v486;
                        float v489; float v490;
                        Tuple2 tmp70 = v127[0];
                        v489 = tmp70.v0; v490 = tmp70.v1;
                        bool v493;
                        v493 = v490 == 0.0f;
                        bool v494;
                        v494 = v493 != true;
                        float v496;
                        if (v494){
                            float v495;
                            v495 = v489 / v490;
                            v496 = v495;
                        } else {
                            v496 = 0.0f;
                        }
                        float v497 = v323.v0;
                        float v498;
                        v498 = v497 - v496;
                        float v499;
                        v499 = v498 / v461;
                        float v500;
                        v500 = v499 + v496;
                        v511 = v500;
                    } else {
                        float v503; float v504;
                        Tuple2 tmp71 = v127[0];
                        v503 = tmp71.v0; v504 = tmp71.v1;
                        bool v507;
                        v507 = v504 == 0.0f;
                        bool v508;
                        v508 = v507 != true;
                        if (v508){
                            float v509;
                            v509 = v503 / v504;
                            v511 = v509;
                        } else {
                            v511 = 0.0f;
                        }
                    }
                    static_array<float,3> v513;
                    v513[0] = v511;
                    v513[1] = v455;
                    v513[2] = v389;
                    static_array<float,3> v516;
                    int v518;
                    v518 = 0;
                    while (while_method_4(v518)){
                        float v521;
                        v521 = v513[v518];
                        float v524;
                        v524 = v302[v518];
                        float v526;
                        v526 = v521 * v524;
                        v516[v518] = v526;
                        v518 += 1 ;
                    }
                    int v527; float v528;
                    Tuple3 tmp72 = Tuple3{0, 0.0f};
                    v527 = tmp72.v0; v528 = tmp72.v1;
                    while (while_method_4(v527)){
                        float v531;
                        v531 = v516[v527];
                        float v533;
                        v533 = v528 + v531;
                        v528 = v533;
                        v527 += 1 ;
                    }
                    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v534 = v1.v0;
                    static_array<float,3> v536;
                    int v538;
                    v538 = 0;
                    while (while_method_4(v538)){
                        float v541;
                        v541 = v125[v538];
                        float v544;
                        v544 = v302[v538];
                        float v546;
                        v546 = 0.99902344f * v541;
                        float v547;
                        v547 = v546 + v544;
                        v536[v538] = v547;
                        v538 += 1 ;
                    }
                    static_array<Tuple2,2> & v548 = v5.v0;
                    int v549; float v550;
                    Tuple3 tmp73 = Tuple3{0, 0.0f};
                    v549 = tmp73.v0; v550 = tmp73.v1;
                    while (while_method_1(v549)){
                        float v554; float v555;
                        Tuple2 tmp74 = v548[v549];
                        v554 = tmp74.v0; v555 = tmp74.v1;
                        bool v558;
                        v558 = v549 == v61;
                        float v559;
                        if (v558){
                            v559 = 0.0f;
                        } else {
                            v559 = v554;
                        }
                        float v560;
                        v560 = v550 + v559;
                        float v561;
                        v561 = v560 - v555;
                        v550 = v561;
                        v549 += 1 ;
                    }
                    float v562;
                    v562 = exp(v550);
                    static_array<float,3> v564;
                    int v566;
                    v566 = 0;
                    while (while_method_4(v566)){
                        float v569;
                        v569 = v126[v566];
                        float v572;
                        v572 = v513[v566];
                        float v574;
                        v574 = v572 - v528;
                        float v575;
                        v575 = v562 * v574;
                        float v576;
                        v576 = v569 + v575;
                        bool v577;
                        v577 = 0.0f >= v576;
                        float v578;
                        if (v577){
                            v578 = 0.0f;
                        } else {
                            v578 = v576;
                        }
                        v564[v566] = v578;
                        v566 += 1 ;
                    }
                    static_array<Tuple2,3> v580;
                    int v582;
                    v582 = 0;
                    while (while_method_4(v582)){
                        float v586; float v587;
                        Tuple2 tmp75 = v127[v582];
                        v586 = tmp75.v0; v587 = tmp75.v1;
                        bool v590;
                        v590 = v322 == v582;
                        float v596; float v597;
                        if (v590){
                            float v591;
                            v591 = v586 * 0.5f;
                            float v592 = v323.v0;
                            float v593;
                            v593 = v591 + v592;
                            float v594;
                            v594 = v587 * 0.5f;
                            float v595;
                            v595 = v594 + 1.0f;
                            v596 = v593; v597 = v595;
                        } else {
                            v596 = v586; v597 = v587;
                        }
                        v580[v582] = Tuple2{v596, v597};
                        v582 += 1 ;
                    }
                    v534[v88] = Tuple1{v536, v564, v580};
                    v700 = v528;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            bool v701;
            v701 = v61 == 0;
            float v703;
            if (v701){
                v703 = v700;
            } else {
                float v702;
                v702 = -v700;
                v703 = v702;
            }
            v7.v0 = v703;
            Union6 v704;
            v704 = Union6{Union6_3{}};
            return loop_16(v0, v1, v2, v3, v4, v5, v7, v704);
            break;
        }
        case 3: { // RoundWithAction
            Union5 v706 = v6.case3.v0; bool v707 = v6.case3.v1; static_array<Union1,2> v708 = v6.case3.v2; int v709 = v6.case3.v3; static_array<int,2> v710 = v6.case3.v4; int v711 = v6.case3.v5; Union2 v712 = v6.case3.v6;
            static_array_list<Union0,32> & v713 = v3.v0;
            Union0 v714;
            v714 = Union0{Union0_1{v709, v712}};
            v713.push(v714);
            Union6 v715;
            v715 = Union6{Union6_2{v706, v707, v708, v709, v710, v711, v712}};
            float v716;
            v716 = loop_16(v0, v1, v2, v3, v4, v5, v7, v715);
            static_array_list<Union0,32> & v717 = v3.v0;
            Union0 v718;
            v718 = v717.pop();
            return v716;
            break;
        }
        case 4: { // TerminalCall
            Union5 v29 = v6.case4.v0; bool v30 = v6.case4.v1; static_array<Union1,2> v31 = v6.case4.v2; int v32 = v6.case4.v3; static_array<int,2> v33 = v6.case4.v4; int v34 = v6.case4.v5;
            int v36;
            v36 = v33[v32];
            Union9 v38;
            v38 = compare_hands_11(v29, v30, v31, v32, v33, v34);
            int v43; int v44;
            switch (v38.tag) {
                case 0: { // Eq
                    v43 = 0; v44 = -1;
                    break;
                }
                case 1: { // Gt
                    v43 = v36; v44 = 0;
                    break;
                }
                case 2: { // Lt
                    v43 = v36; v44 = 1;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            int v45;
            v45 = -v44;
            bool v46;
            v46 = v44 >= v45;
            int v47;
            if (v46){
                v47 = v44;
            } else {
                v47 = v45;
            }
            float v48;
            v48 = (float)v43;
            bool v49;
            v49 = v47 == 0;
            float v51;
            if (v49){
                v51 = v48;
            } else {
                float v50;
                v50 = -v48;
                v51 = v50;
            }
            v7.v0 = v51;
            static_array_list<Union0,32> & v52 = v3.v0;
            Union0 v53;
            v53 = Union0{Union0_3{v31, v43, v44}};
            v52.push(v53);
            Union6 v54;
            v54 = Union6{Union6_3{}};
            float v55;
            v55 = loop_16(v0, v1, v2, v3, v4, v5, v7, v54);
            static_array_list<Union0,32> & v56 = v3.v0;
            Union0 v57;
            v57 = v56.pop();
            return v55;
            break;
        }
        case 5: { // TerminalFold
            Union5 v8 = v6.case5.v0; bool v9 = v6.case5.v1; static_array<Union1,2> v10 = v6.case5.v2; int v11 = v6.case5.v3; static_array<int,2> v12 = v6.case5.v4; int v13 = v6.case5.v5;
            int v15;
            v15 = v12[v11];
            int v17;
            v17 = -v15;
            float v18;
            v18 = (float)v17;
            bool v19;
            v19 = v11 == 0;
            float v21;
            if (v19){
                v21 = v18;
            } else {
                float v20;
                v20 = -v18;
                v21 = v20;
            }
            v7.v0 = v21;
            int v22;
            v22 = v11 ^ 1;
            static_array_list<Union0,32> & v23 = v3.v0;
            Union0 v24;
            v24 = Union0{Union0_3{v10, v15, v22}};
            v23.push(v24);
            Union6 v25;
            v25 = Union6{Union6_3{}};
            float v26;
            v26 = loop_16(v0, v1, v2, v3, v4, v5, v7, v25);
            static_array_list<Union0,32> & v27 = v3.v0;
            Union0 v28;
            v28 = v27.pop();
            return v26;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
inline bool while_method_9(std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v0, std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1>::iterator & v1){
    bool v2;
    v2 = v1 != v0.end();
    return v2;
}
static_array<float,3> normalize_17(static_array<float,3> v0){
    int v1; float v2;
    Tuple3 tmp77 = Tuple3{0, 0.0f};
    v1 = tmp77.v0; v2 = tmp77.v1;
    while (while_method_4(v1)){
        float v5;
        v5 = v0[v1];
        float v7;
        v7 = v2 + v5;
        v2 = v7;
        v1 += 1 ;
    }
    static_array<float,3> v9;
    int v11;
    v11 = 0;
    while (while_method_4(v11)){
        float v14;
        v14 = v0[v11];
        bool v16;
        v16 = v2 == 0.0f;
        bool v17;
        v17 = v16 != true;
        float v19;
        if (v17){
            float v18;
            v18 = v14 / v2;
            v19 = v18;
        } else {
            v19 = 0.33333334f;
        }
        v9[v11] = v19;
        v11 += 1 ;
    }
    return v9;
}
inline bool while_method_10(std::unordered_map<static_array_list<Union0,32>, static_array<float,3>, Fun0, Fun1> & v0, std::unordered_map<static_array_list<Union0,32>, static_array<float,3>, Fun0, Fun1>::iterator & v1){
    bool v2;
    v2 = v1 != v0.end();
    return v2;
}
void method_19(Union1 v0){
    switch (v0.tag) {
        case 0: { // Jack
            printf("%s","Jack");
            return ;
            break;
        }
        case 1: { // King
            printf("%s","King");
            return ;
            break;
        }
        case 2: { // Queen
            printf("%s","Queen");
            return ;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
void method_20(Union2 v0){
    switch (v0.tag) {
        case 0: { // Call
            printf("%s","Call");
            return ;
            break;
        }
        case 1: { // Fold
            printf("%s","Fold");
            return ;
            break;
        }
        case 2: { // Raise
            printf("%s","Raise");
            return ;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
void method_18(Union0 v0){
    switch (v0.tag) {
        case 0: { // CommunityCardIs
            Union1 v1 = v0.case0.v0;
            printf("%s(","CommunityCardIs");
            method_19(v1);
            printf(")");
            return ;
            break;
        }
        case 1: { // PlayerAction
            int v2 = v0.case1.v0; Union2 v3 = v0.case1.v1;
            printf("%s(%d, ","PlayerAction", v2);
            method_20(v3);
            printf(")");
            return ;
            break;
        }
        case 2: { // PlayerGotCard
            int v4 = v0.case2.v0; Union1 v5 = v0.case2.v1;
            printf("%s(%d, ","PlayerGotCard", v4);
            method_19(v5);
            printf(")");
            return ;
            break;
        }
        case 3: { // Showdown
            static_array<Union1,2> v6 = v0.case3.v0; int v7 = v0.case3.v1; int v8 = v0.case3.v2;
            printf("%s({%s = %s","Showdown", "cards_shown", "[");
            int v9;
            v9 = 0;
            while (while_method_1(v9)){
                Union1 v12;
                v12 = v6[v9];
                printf("");
                method_19(v12);
                printf("");
                int v14;
                v14 = v9 + 1;
                bool v15;
                v15 = v14 < 2;
                if (v15){
                    printf("%s","; ");
                } else {
                }
                v9 += 1 ;
            }
            printf("%s","]");
            printf("; %s = %d; %s = %d})","chips_won", v7, "winner_id", v8);
            return ;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
int main() {
    Fun0 v0 = FunPointerMethod0;
    Fun1 v1 = FunPointerMethod1;
    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> v2(512, v0, v1);
    StackRefs0 v3{v2};
    static_array<Union3,2> v5;
    Union3 v8;
    v8 = Union3{Union3_2{}};
    v5[0] = v8;
    Union3 v11;
    v11 = Union3{Union3_2{}};
    v5[1] = v11;
    xso::rng v13;
    StackMut0 v14{63u};
    static_array_list<Union0,32> v16;
    v16 = static_array_list<Union0,32>{};
    StackRefs1 v18{v16};
    static_array<Tuple2,2> v20;
    int v22;
    v22 = 0;
    while (while_method_1(v22)){
        v20[v22] = Tuple2{0.0f, 0.0f};
        v22 += 1 ;
    }
    StackRefs2 v24{v20};
    int v25; float v26;
    Tuple3 tmp2 = Tuple3{0, 0.0f};
    v25 = tmp2.v0; v26 = tmp2.v1;
    while (while_method_2(v25)){
        int v28;
        v28 = v25 % 40000;
        bool v29;
        v29 = v28 == 0;
        if (v29){
            printf("{%s = %d; %s = %d}\n","i", v25, "nearTo", 1000000);
            fflush(stdout);
        } else {
        }
        Union4 v35;
        v35 = Union4{Union4_1{}};
        float v36;
        v36 = body_0(v5, v3, v13, v18, v14, v24, v35);
        v26 = v36;
        v25 += 1 ;
    }
    static_array<Union3,2> v38;
    Union3 v41;
    v41 = Union3{Union3_1{}};
    v38[0] = v41;
    Union3 v44;
    v44 = Union3{Union3_0{}};
    v38[1] = v44;
    xso::rng v46;
    StackMut0 v47{63u};
    static_array_list<Union0,32> v49;
    v49 = static_array_list<Union0,32>{};
    StackRefs1 v51{v49};
    static_array<Tuple2,2> v53;
    int v55;
    v55 = 0;
    while (while_method_1(v55)){
        v53[v55] = Tuple2{0.0f, 0.0f};
        v55 += 1 ;
    }
    StackRefs2 v57{v53};
    int v58; float v59;
    Tuple3 tmp41 = Tuple3{0, 0.0f};
    v58 = tmp41.v0; v59 = tmp41.v1;
    while (while_method_7(v58)){
        int v61;
        v61 = v58 % 4;
        bool v62;
        v62 = v61 == 0;
        if (v62){
            printf("{%s = %d; %s = %d}\n","i", v58, "nearTo", 100);
            fflush(stdout);
        } else {
        }
        Union4 v68;
        v68 = Union4{Union4_1{}};
        float v69;
        v69 = body_15(v38, v3, v46, v51, v47, v57, v68);
        v59 = v69;
        v58 += 1 ;
    }
    printf("{%s = %f}\n","reward_for_pl0", v59);
    fflush(stdout);
    std::unordered_map<static_array_list<Union0,32>, static_array<float,3>, Fun0, Fun1> v74(512, v0, v1);
    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v75 = v3.v0;
    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v76 = v75;
    auto v77 = v76.begin();
    while (while_method_9(v76, v77)){
        static_array_list<Union0,32> v79;
        v79 = v77->first;
        static_array<float,3> v80; static_array<float,3> v81; static_array<Tuple2,3> v82;
        Tuple1 tmp76 = v77->second;
        v80 = tmp76.v0; v81 = tmp76.v1; v82 = tmp76.v2;
        static_array<float,3> v83;
        v83 = normalize_17(v80);
        v74[v79] = v83;
        ++v77;
    }
    printf("%s\n","{");
    std::unordered_map<static_array_list<Union0,32>, static_array<float,3>, Fun0, Fun1> & v111 = v74;
    auto v112 = v111.begin();
    while (while_method_10(v111, v112)){
        static_array_list<Union0,32> v114;
        v114 = v112->first;
        static_array<float,3> v115;
        v115 = v112->second;
        printf("%s","[");
        int v116;
        v116 = v114.length;
        bool v117;
        v117 = 100 < v116;
        int v118;
        if (v117){
            v118 = 100;
        } else {
            v118 = v116;
        }
        int v119;
        v119 = 0;
        while (while_method_0(v118, v119)){
            Union0 v122;
            v122 = v114[v119];
            printf("");
            method_18(v122);
            printf("");
            int v124;
            v124 = v119 + 1;
            int v125;
            v125 = v114.length;
            bool v126;
            v126 = v124 < v125;
            if (v126){
                printf("%s","; ");
            } else {
            }
            v119 += 1 ;
        }
        int v127;
        v127 = v114.length;
        bool v128;
        v128 = v127 > 100;
        if (v128){
            printf("%s","; ...");
        } else {
        }
        printf("%s","]");
        printf("");
        printf("%s"," => ");
        printf("%s","[");
        int v129;
        v129 = 0;
        while (while_method_4(v129)){
            float v132;
            v132 = v115[v129];
            printf("%f",v132);
            int v134;
            v134 = v129 + 1;
            bool v135;
            v135 = v134 < 3;
            if (v135){
                printf("%s","; ");
            } else {
            }
            v129 += 1 ;
        }
        printf("%s","]");
        printf("\n");
        ++v112;
    }
    printf("%s\n","}");
    printf("\n");
    fflush(stdout);
    return 0;
}
