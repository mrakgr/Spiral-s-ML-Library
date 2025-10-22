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
float loop_4(bool v0, static_array<Union3,2> v1, StackRefs0 & v2, xso::rng & v3, StackRefs1 & v4, StackMut0 & v5, StackRefs2 & v6, StackMut1 & v7, Union6 v8);
struct Tuple6;
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
float body_0(bool v0, static_array<Union3,2> v1, StackRefs0 & v2, xso::rng & v3, StackRefs1 & v4, StackMut0 & v5, StackRefs2 & v6, Union4 v7);
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
struct Tuple6 {
    int v0;
    float v1;
    float v2;
    __host__ __device__ Tuple6() = default;
    __host__ __device__ Tuple6(int t0, float t1, float t2) : v0(t0), v1(t1), v2(t2) {}
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
inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 30;
    return v1;
}
inline bool while_method_1(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
inline bool while_method_2(int v0){
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
    while (while_method_1(v1, v2)){
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
                while (while_method_2(v38)){
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
        while (while_method_1(v6, v7)){
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
                    while (while_method_2(v39)){
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
inline bool while_method_3(int v0){
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
inline bool while_method_4(unsigned int v0, unsigned int v1){
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
    while (while_method_4(v8, v9)){
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
float loop_4(bool v0, static_array<Union3,2> v1, StackRefs0 & v2, xso::rng & v3, StackRefs1 & v4, StackMut0 & v5, StackRefs2 & v6, StackMut1 & v7, Union6 v8){
    switch (v8.tag) {
        case 0: { // T_game_chance_community_card
            Union5 v10 = v8.case0.v0; bool v11 = v8.case0.v1; static_array<Union1,2> v12 = v8.case0.v2; int v13 = v8.case0.v3; static_array<int,2> v14 = v8.case0.v4; int v15 = v8.case0.v5; Union1 v16 = v8.case0.v6;
            int v17;
            v17 = 2;
            int v18; int v19;
            Tuple5 tmp4 = Tuple5{0, 0};
            v18 = tmp4.v0; v19 = tmp4.v1;
            while (while_method_2(v18)){
                int v22;
                v22 = v14[v18];
                bool v24;
                v24 = v19 >= v22;
                int v25;
                if (v24){
                    v25 = v19;
                } else {
                    v25 = v22;
                }
                v19 = v25;
                v18 += 1 ;
            }
            static_array<int,2> v27;
            int v29;
            v29 = 0;
            while (while_method_2(v29)){
                v27[v29] = v19;
                v29 += 1 ;
            }
            Union5 v31;
            v31 = Union5{Union5_1{v16}};
            bool v32;
            v32 = true;
            int v33;
            v33 = 0;
            Union4 v34;
            v34 = Union4{Union4_2{v31, v32, v12, v33, v27, v17}};
            return body_0(v0, v1, v2, v3, v4, v5, v6, v34);
            break;
        }
        case 1: { // T_game_chance_init
            Union1 v36 = v8.case1.v0; Union1 v37 = v8.case1.v1;
            int v38;
            v38 = 2;
            static_array<int,2> v40;
            v40[0] = 1;
            v40[1] = 1;
            static_array<Union1,2> v43;
            v43[0] = v36;
            v43[1] = v37;
            Union5 v45;
            v45 = Union5{Union5_0{}};
            bool v46;
            v46 = true;
            int v47;
            v47 = 0;
            Union4 v48;
            v48 = Union4{Union4_2{v45, v46, v43, v47, v40, v38}};
            return body_0(v0, v1, v2, v3, v4, v5, v6, v48);
            break;
        }
        case 2: { // T_game_round
            Union5 v50 = v8.case2.v0; bool v51 = v8.case2.v1; static_array<Union1,2> v52 = v8.case2.v2; int v53 = v8.case2.v3; static_array<int,2> v54 = v8.case2.v4; int v55 = v8.case2.v5; Union2 v56 = v8.case2.v6;
            Union4 v148;
            switch (v50.tag) {
                case 0: { // None
                    switch (v56.tag) {
                        case 0: { // Call
                            if (v51){
                                int v110;
                                v110 = v53 ^ 1;
                                v148 = Union4{Union4_2{v50, false, v52, v110, v54, v55}};
                            } else {
                                v148 = Union4{Union4_0{v50, v51, v52, v53, v54, v55}};
                            }
                            break;
                        }
                        case 1: { // Fold
                            v148 = Union4{Union4_5{v50, v51, v52, v53, v54, v55}};
                            break;
                        }
                        case 2: { // Raise
                            bool v114;
                            v114 = v55 > 0;
                            if (v114){
                                int v115;
                                v115 = v53 ^ 1;
                                int v116;
                                v116 = -1 + v55;
                                int v117; int v118;
                                Tuple5 tmp5 = Tuple5{0, 0};
                                v117 = tmp5.v0; v118 = tmp5.v1;
                                while (while_method_2(v117)){
                                    int v121;
                                    v121 = v54[v117];
                                    bool v123;
                                    v123 = v118 >= v121;
                                    int v124;
                                    if (v123){
                                        v124 = v118;
                                    } else {
                                        v124 = v121;
                                    }
                                    v118 = v124;
                                    v117 += 1 ;
                                }
                                static_array<int,2> v126;
                                int v128;
                                v128 = 0;
                                while (while_method_2(v128)){
                                    v126[v128] = v118;
                                    v128 += 1 ;
                                }
                                static_array<int,2> v131;
                                int v133;
                                v133 = 0;
                                while (while_method_2(v133)){
                                    int v136;
                                    v136 = v126[v133];
                                    bool v138;
                                    v138 = v133 == v53;
                                    int v140;
                                    if (v138){
                                        int v139;
                                        v139 = v136 + 2;
                                        v140 = v139;
                                    } else {
                                        v140 = v136;
                                    }
                                    v131[v133] = v140;
                                    v133 += 1 ;
                                }
                                v148 = Union4{Union4_2{v50, false, v52, v115, v131, v116}};
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
                    Union1 v57 = v50.case1.v0;
                    switch (v56.tag) {
                        case 0: { // Call
                            if (v51){
                                int v59;
                                v59 = v53 ^ 1;
                                v148 = Union4{Union4_2{v50, false, v52, v59, v54, v55}};
                            } else {
                                int v61; int v62;
                                Tuple5 tmp6 = Tuple5{0, 0};
                                v61 = tmp6.v0; v62 = tmp6.v1;
                                while (while_method_2(v61)){
                                    int v65;
                                    v65 = v54[v61];
                                    bool v67;
                                    v67 = v62 >= v65;
                                    int v68;
                                    if (v67){
                                        v68 = v62;
                                    } else {
                                        v68 = v65;
                                    }
                                    v62 = v68;
                                    v61 += 1 ;
                                }
                                static_array<int,2> v70;
                                int v72;
                                v72 = 0;
                                while (while_method_2(v72)){
                                    v70[v72] = v62;
                                    v72 += 1 ;
                                }
                                v148 = Union4{Union4_4{v50, v51, v52, v53, v70, v55}};
                            }
                            break;
                        }
                        case 1: { // Fold
                            v148 = Union4{Union4_5{v50, v51, v52, v53, v54, v55}};
                            break;
                        }
                        case 2: { // Raise
                            bool v76;
                            v76 = v55 > 0;
                            if (v76){
                                int v77;
                                v77 = v53 ^ 1;
                                int v78;
                                v78 = -1 + v55;
                                int v79; int v80;
                                Tuple5 tmp7 = Tuple5{0, 0};
                                v79 = tmp7.v0; v80 = tmp7.v1;
                                while (while_method_2(v79)){
                                    int v83;
                                    v83 = v54[v79];
                                    bool v85;
                                    v85 = v80 >= v83;
                                    int v86;
                                    if (v85){
                                        v86 = v80;
                                    } else {
                                        v86 = v83;
                                    }
                                    v80 = v86;
                                    v79 += 1 ;
                                }
                                static_array<int,2> v88;
                                int v90;
                                v90 = 0;
                                while (while_method_2(v90)){
                                    v88[v90] = v80;
                                    v90 += 1 ;
                                }
                                static_array<int,2> v93;
                                int v95;
                                v95 = 0;
                                while (while_method_2(v95)){
                                    int v98;
                                    v98 = v88[v95];
                                    bool v100;
                                    v100 = v95 == v53;
                                    int v102;
                                    if (v100){
                                        int v101;
                                        v101 = v98 + 4;
                                        v102 = v101;
                                    } else {
                                        v102 = v98;
                                    }
                                    v93[v95] = v102;
                                    v95 += 1 ;
                                }
                                v148 = Union4{Union4_2{v50, false, v52, v77, v93, v78}};
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
            return body_0(v0, v1, v2, v3, v4, v5, v6, v148);
            break;
        }
        case 3: { // T_none
            float v9 = v7.v0;
            return v9;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 6;
    return v1;
}
inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 3;
    return v1;
}
static_array<float,3> masking_normalize_5(static_array<float,3> v0, static_array<bool,3> v1){
    int v2; float v3;
    Tuple3 tmp15 = Tuple3{0, 0.0f};
    v2 = tmp15.v0; v3 = tmp15.v1;
    while (while_method_6(v2)){
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
    while (while_method_6(v14)){
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
    Tuple3 tmp16 = Tuple3{0, 0.0f};
    v23 = tmp16.v0; v24 = tmp16.v1;
    while (while_method_6(v23)){
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
    while (while_method_6(v33)){
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
    while (while_method_6(v4)){
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
inline bool while_method_7(static_array<float,3> v0, int v1){
    bool v2;
    v2 = v1 < 3;
    return v2;
}
inline bool while_method_8(int v0, int v1){
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
    while (while_method_6(v5)){
        float v8;
        v8 = v0[v5];
        v3[v5] = v8;
        v5 += 1 ;
    }
    int v10;
    v10 = 1;
    while (while_method_7(v3, v10)){
        int v12;
        v12 = 3;
        while (while_method_8(v10, v12)){
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
                    Tuple5 tmp37 = order_14(v8, v12);
                    v29 = tmp37.v0; v30 = tmp37.v1;
                    int v31; int v32;
                    Tuple5 tmp38 = order_14(v8, v16);
                    v31 = tmp38.v0; v32 = tmp38.v1;
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
float body_0(bool v0, static_array<Union3,2> v1, StackRefs0 & v2, xso::rng & v3, StackRefs1 & v4, StackMut0 & v5, StackRefs2 & v6, Union4 v7){
    StackMut1 v8{0.0f};
    switch (v7.tag) {
        case 0: { // ChanceCommunityCard
            Union5 v669 = v7.case0.v0; bool v670 = v7.case0.v1; static_array<Union1,2> v671 = v7.case0.v2; int v672 = v7.case0.v3; static_array<int,2> v673 = v7.case0.v4; int v674 = v7.case0.v5;
            if (v0){
                unsigned int v675 = v5.v0;
                Union1 v676; unsigned int v677;
                Tuple4 tmp3 = draw_card_1(v3, v675);
                v676 = tmp3.v0; v677 = tmp3.v1;
                v5.v0 = v677;
                static_array_list<Union0,32> & v678 = v4.v0;
                Union0 v679;
                v679 = Union0{Union0_0{v676}};
                v678.push(v679);
                Union6 v680;
                v680 = Union6{Union6_0{v669, v670, v671, v672, v673, v674, v676}};
                float v681;
                v681 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v680);
                static_array_list<Union0,32> & v682 = v4.v0;
                Union0 v683;
                v683 = v682.pop();
                v5.v0 = v675;
                return v681;
            } else {
                int v684; float v685; float v686;
                Tuple6 tmp8 = Tuple6{0, 0.0f, 0.0f};
                v684 = tmp8.v0; v685 = tmp8.v1; v686 = tmp8.v2;
                while (while_method_5(v684)){
                    unsigned int v688 = v5.v0;
                    unsigned int v689;
                    v689 = 1u << v684;
                    unsigned int v690;
                    v690 = v688 & v689;
                    bool v691;
                    v691 = v690 == 0u;
                    bool v692;
                    v692 = v691 != true;
                    float v724; float v725;
                    if (v692){
                        unsigned int v693 = v5.v0;
                        unsigned int v694;
                        v694 = v693 ^ v689;
                        v5.v0 = v694;
                        bool v695;
                        v695 = 0 == v684;
                        Union1 v713;
                        if (v695){
                            v713 = Union1{Union1_1{}};
                        } else {
                            bool v697;
                            v697 = 1 == v684;
                            if (v697){
                                v713 = Union1{Union1_1{}};
                            } else {
                                bool v699;
                                v699 = 2 == v684;
                                if (v699){
                                    v713 = Union1{Union1_2{}};
                                } else {
                                    bool v701;
                                    v701 = 3 == v684;
                                    if (v701){
                                        v713 = Union1{Union1_2{}};
                                    } else {
                                        bool v703;
                                        v703 = 4 == v684;
                                        if (v703){
                                            v713 = Union1{Union1_0{}};
                                        } else {
                                            bool v705;
                                            v705 = 5 == v684;
                                            if (v705){
                                                v713 = Union1{Union1_0{}};
                                            } else {
                                                printf("%s\n", "Invalid int in int_to_card.");
                                                exit(-1);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        static_array_list<Union0,32> & v714 = v4.v0;
                        Union0 v715;
                        v715 = Union0{Union0_0{v713}};
                        v714.push(v715);
                        Union6 v716;
                        v716 = Union6{Union6_0{v669, v670, v671, v672, v673, v674, v713}};
                        float v717;
                        v717 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v716);
                        static_array_list<Union0,32> & v718 = v4.v0;
                        Union0 v719;
                        v719 = v718.pop();
                        unsigned int v720 = v5.v0;
                        unsigned int v721;
                        v721 = v720 ^ v689;
                        v5.v0 = v721;
                        float v722;
                        v722 = v685 + v717;
                        float v723;
                        v723 = v686 + 1.0f;
                        v724 = v722; v725 = v723;
                    } else {
                        v724 = v685; v725 = v686;
                    }
                    v685 = v724;
                    v686 = v725;
                    v684 += 1 ;
                }
                bool v726;
                v726 = v686 == 0.0f;
                bool v727;
                v727 = v726 != true;
                if (v727){
                    float v728;
                    v728 = v685 / v686;
                    return v728;
                } else {
                    return 0.0f;
                }
            }
            break;
        }
        case 1: { // ChanceInit
            if (v0){
                unsigned int v731 = v5.v0;
                Union1 v732; unsigned int v733;
                Tuple4 tmp9 = draw_card_1(v3, v731);
                v732 = tmp9.v0; v733 = tmp9.v1;
                v5.v0 = v733;
                unsigned int v734 = v5.v0;
                Union1 v735; unsigned int v736;
                Tuple4 tmp10 = draw_card_1(v3, v734);
                v735 = tmp10.v0; v736 = tmp10.v1;
                v5.v0 = v736;
                static_array_list<Union0,32> & v737 = v4.v0;
                Union0 v738;
                v738 = Union0{Union0_2{0, v732}};
                v737.push(v738);
                static_array_list<Union0,32> & v739 = v4.v0;
                Union0 v740;
                v740 = Union0{Union0_2{1, v735}};
                v739.push(v740);
                Union6 v741;
                v741 = Union6{Union6_1{v732, v735}};
                float v742;
                v742 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v741);
                static_array_list<Union0,32> & v743 = v4.v0;
                Union0 v744;
                v744 = v743.pop();
                static_array_list<Union0,32> & v745 = v4.v0;
                Union0 v746;
                v746 = v745.pop();
                v5.v0 = v734;
                v5.v0 = v731;
                return v742;
            } else {
                int v747; float v748; float v749;
                Tuple6 tmp11 = Tuple6{0, 0.0f, 0.0f};
                v747 = tmp11.v0; v748 = tmp11.v1; v749 = tmp11.v2;
                while (while_method_5(v747)){
                    unsigned int v751 = v5.v0;
                    unsigned int v752;
                    v752 = 1u << v747;
                    unsigned int v753;
                    v753 = v751 & v752;
                    bool v754;
                    v754 = v753 == 0u;
                    bool v755;
                    v755 = v754 != true;
                    float v831; float v832;
                    if (v755){
                        unsigned int v756 = v5.v0;
                        unsigned int v757;
                        v757 = v756 ^ v752;
                        v5.v0 = v757;
                        bool v758;
                        v758 = 0 == v747;
                        Union1 v776;
                        if (v758){
                            v776 = Union1{Union1_1{}};
                        } else {
                            bool v760;
                            v760 = 1 == v747;
                            if (v760){
                                v776 = Union1{Union1_1{}};
                            } else {
                                bool v762;
                                v762 = 2 == v747;
                                if (v762){
                                    v776 = Union1{Union1_2{}};
                                } else {
                                    bool v764;
                                    v764 = 3 == v747;
                                    if (v764){
                                        v776 = Union1{Union1_2{}};
                                    } else {
                                        bool v766;
                                        v766 = 4 == v747;
                                        if (v766){
                                            v776 = Union1{Union1_0{}};
                                        } else {
                                            bool v768;
                                            v768 = 5 == v747;
                                            if (v768){
                                                v776 = Union1{Union1_0{}};
                                            } else {
                                                printf("%s\n", "Invalid int in int_to_card.");
                                                exit(-1);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        int v777; float v778; float v779;
                        Tuple6 tmp12 = Tuple6{0, 0.0f, 0.0f};
                        v777 = tmp12.v0; v778 = tmp12.v1; v779 = tmp12.v2;
                        while (while_method_5(v777)){
                            unsigned int v781 = v5.v0;
                            unsigned int v782;
                            v782 = 1u << v777;
                            unsigned int v783;
                            v783 = v781 & v782;
                            bool v784;
                            v784 = v783 == 0u;
                            bool v785;
                            v785 = v784 != true;
                            float v821; float v822;
                            if (v785){
                                unsigned int v786 = v5.v0;
                                unsigned int v787;
                                v787 = v786 ^ v782;
                                v5.v0 = v787;
                                bool v788;
                                v788 = 0 == v777;
                                Union1 v806;
                                if (v788){
                                    v806 = Union1{Union1_1{}};
                                } else {
                                    bool v790;
                                    v790 = 1 == v777;
                                    if (v790){
                                        v806 = Union1{Union1_1{}};
                                    } else {
                                        bool v792;
                                        v792 = 2 == v777;
                                        if (v792){
                                            v806 = Union1{Union1_2{}};
                                        } else {
                                            bool v794;
                                            v794 = 3 == v777;
                                            if (v794){
                                                v806 = Union1{Union1_2{}};
                                            } else {
                                                bool v796;
                                                v796 = 4 == v777;
                                                if (v796){
                                                    v806 = Union1{Union1_0{}};
                                                } else {
                                                    bool v798;
                                                    v798 = 5 == v777;
                                                    if (v798){
                                                        v806 = Union1{Union1_0{}};
                                                    } else {
                                                        printf("%s\n", "Invalid int in int_to_card.");
                                                        exit(-1);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                static_array_list<Union0,32> & v807 = v4.v0;
                                Union0 v808;
                                v808 = Union0{Union0_2{0, v776}};
                                v807.push(v808);
                                static_array_list<Union0,32> & v809 = v4.v0;
                                Union0 v810;
                                v810 = Union0{Union0_2{1, v806}};
                                v809.push(v810);
                                Union6 v811;
                                v811 = Union6{Union6_1{v776, v806}};
                                float v812;
                                v812 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v811);
                                static_array_list<Union0,32> & v813 = v4.v0;
                                Union0 v814;
                                v814 = v813.pop();
                                static_array_list<Union0,32> & v815 = v4.v0;
                                Union0 v816;
                                v816 = v815.pop();
                                unsigned int v817 = v5.v0;
                                unsigned int v818;
                                v818 = v817 ^ v782;
                                v5.v0 = v818;
                                float v819;
                                v819 = v778 + v812;
                                float v820;
                                v820 = v779 + 1.0f;
                                v821 = v819; v822 = v820;
                            } else {
                                v821 = v778; v822 = v779;
                            }
                            v778 = v821;
                            v779 = v822;
                            v777 += 1 ;
                        }
                        bool v823;
                        v823 = v779 == 0.0f;
                        bool v824;
                        v824 = v823 != true;
                        float v826;
                        if (v824){
                            float v825;
                            v825 = v778 / v779;
                            v826 = v825;
                        } else {
                            v826 = 0.0f;
                        }
                        unsigned int v827 = v5.v0;
                        unsigned int v828;
                        v828 = v827 ^ v752;
                        v5.v0 = v828;
                        float v829;
                        v829 = v748 + v826;
                        float v830;
                        v830 = v749 + 1.0f;
                        v831 = v829; v832 = v830;
                    } else {
                        v831 = v748; v832 = v749;
                    }
                    v748 = v831;
                    v749 = v832;
                    v747 += 1 ;
                }
                bool v833;
                v833 = v749 == 0.0f;
                bool v834;
                v834 = v833 != true;
                if (v834){
                    float v835;
                    v835 = v748 / v749;
                    return v835;
                } else {
                    return 0.0f;
                }
            }
            break;
        }
        case 2: { // Round
            Union5 v59 = v7.case2.v0; bool v60 = v7.case2.v1; static_array<Union1,2> v61 = v7.case2.v2; int v62 = v7.case2.v3; static_array<int,2> v63 = v7.case2.v4; int v64 = v7.case2.v5;
            static_array_list<Union0,32> & v65 = v4.v0;
            int v66;
            v66 = v65.length;
            bool v67;
            v67 = 32 >= v66;
            bool v68;
            v68 = v67 == false;
            if (v68){
                assert("The type level dimension has to equal the value passed at runtime into create." && v67);
            } else {
            }
            static_array_list<Union0,32> v71;
            v71 = static_array_list<Union0,32>{};
            v71.unsafe_set_length(v66);
            int v73; int v74;
            Tuple5 tmp13 = Tuple5{0, 0};
            v73 = tmp13.v0; v74 = tmp13.v1;
            while (while_method_1(v66, v73)){
                Union0 v77;
                v77 = v65[v73];
                bool v82;
                switch (v77.tag) {
                    case 2: { // PlayerGotCard
                        int v79 = v77.case2.v0; Union1 v80 = v77.case2.v1;
                        bool v81;
                        v81 = v79 == v62;
                        v82 = v81;
                        break;
                    }
                    default: {
                        v82 = true;
                    }
                }
                int v84;
                if (v82){
                    v71[v74] = v77;
                    int v83;
                    v83 = v74 + 1;
                    v84 = v83;
                } else {
                    v84 = v74;
                }
                v74 = v84;
                v73 += 1 ;
            }
            bool v85;
            v85 = 32 >= v74;
            bool v86;
            v86 = v85 == false;
            if (v86){
                assert("The type level dimension has to equal the value passed at runtime into create." && v85);
            } else {
            }
            static_array_list<Union0,32> v89;
            v89 = static_array_list<Union0,32>{};
            v89.unsafe_set_length(v74);
            int v91;
            v91 = 0;
            while (while_method_1(v74, v91)){
                Union0 v94;
                v94 = v71[v91];
                v89[v91] = v94;
                v91 += 1 ;
            }
            std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v96 = v2.v0;
            auto v97 = v96.find(v89);
            bool v98;
            v98 = v97 != v96.end();
            Union7 v104;
            if (v98){
                static_array<float,3> v99; static_array<float,3> v100; static_array<Tuple2,3> v101;
                Tuple1 tmp14 = v97->second;
                v99 = tmp14.v0; v100 = tmp14.v1; v101 = tmp14.v2;
                v104 = Union7{Union7_1{v99, v100, v101}};
            } else {
                v104 = Union7{Union7_0{}};
            }
            static_array<float,3> v126; static_array<float,3> v127; static_array<Tuple2,3> v128;
            switch (v104.tag) {
                case 0: { // None
                    static_array<float,3> v109;
                    int v111;
                    v111 = 0;
                    while (while_method_6(v111)){
                        v109[v111] = 0.0f;
                        v111 += 1 ;
                    }
                    static_array<float,3> v114;
                    int v116;
                    v116 = 0;
                    while (while_method_6(v116)){
                        v114[v116] = 0.0f;
                        v116 += 1 ;
                    }
                    static_array<Tuple2,3> v119;
                    int v121;
                    v121 = 0;
                    while (while_method_6(v121)){
                        v119[v121] = Tuple2{0.0f, 0.0f};
                        v121 += 1 ;
                    }
                    v126 = v109; v127 = v114; v128 = v119;
                    break;
                }
                case 1: { // Some
                    static_array<float,3> v105 = v104.case1.v0; static_array<float,3> v106 = v104.case1.v1; static_array<Tuple2,3> v107 = v104.case1.v2;
                    v126 = v105; v127 = v106; v128 = v107;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            int v130;
            v130 = v63[0];
            int v133;
            v133 = v63[1];
            bool v135;
            v135 = v130 == v133;
            bool v136;
            v136 = v135 != true;
            Union8 v140;
            if (v136){
                Union2 v137;
                v137 = Union2{Union2_1{}};
                v140 = Union8{Union8_1{v137}};
            } else {
                v140 = Union8{Union8_0{}};
            }
            bool v141;
            v141 = v64 > 0;
            Union8 v145;
            if (v141){
                Union2 v142;
                v142 = Union2{Union2_2{}};
                v145 = Union8{Union8_1{v142}};
            } else {
                v145 = Union8{Union8_0{}};
            }
            bool v148;
            switch (v145.tag) {
                case 0: { // None
                    v148 = false;
                    break;
                }
                case 1: { // Some
                    Union2 v146 = v145.case1.v0;
                    v148 = true;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            bool v151;
            switch (v140.tag) {
                case 0: { // None
                    v151 = false;
                    break;
                }
                case 1: { // Some
                    Union2 v149 = v140.case1.v0;
                    v151 = true;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            static_array<bool,3> v153;
            v153[0] = true;
            v153[1] = v151;
            v153[2] = v148;
            Union3 v156;
            v156 = v1[v62];
            float v650;
            switch (v156.tag) {
                case 0: { // Frozen
                    static_array<float,3> v548;
                    v548 = masking_normalize_5(v126, v153);
                    float v574;
                    switch (v145.tag) {
                        case 0: { // None
                            v574 = 0.0f;
                            break;
                        }
                        case 1: { // Some
                            Union2 v549 = v145.case1.v0;
                            float v551;
                            v551 = v548[2];
                            static_array<Tuple2,2> & v553 = v6.v0;
                            float v556; float v557;
                            Tuple2 tmp17 = v553[v62];
                            v556 = tmp17.v0; v557 = tmp17.v1;
                            static_array<Tuple2,2> & v560 = v6.v0;
                            float v561;
                            v561 = log(v551);
                            float v562;
                            v562 = v561 + v556;
                            v560[v62] = Tuple2{v562, v557};
                            static_array_list<Union0,32> & v563 = v4.v0;
                            Union0 v564;
                            v564 = Union0{Union0_1{v62, v549}};
                            v563.push(v564);
                            Union6 v565;
                            v565 = Union6{Union6_2{v59, v60, v61, v62, v63, v64, v549}};
                            float v566;
                            v566 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v565);
                            static_array_list<Union0,32> & v567 = v4.v0;
                            Union0 v568;
                            v568 = v567.pop();
                            static_array<Tuple2,2> & v569 = v6.v0;
                            v569[v62] = Tuple2{v556, v557};
                            bool v570;
                            v570 = v62 == 0;
                            if (v570){
                                v574 = v566;
                            } else {
                                float v571;
                                v571 = -v566;
                                v574 = v571;
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    float v600;
                    switch (v140.tag) {
                        case 0: { // None
                            v600 = 0.0f;
                            break;
                        }
                        case 1: { // Some
                            Union2 v575 = v140.case1.v0;
                            float v577;
                            v577 = v548[1];
                            static_array<Tuple2,2> & v579 = v6.v0;
                            float v582; float v583;
                            Tuple2 tmp18 = v579[v62];
                            v582 = tmp18.v0; v583 = tmp18.v1;
                            static_array<Tuple2,2> & v586 = v6.v0;
                            float v587;
                            v587 = log(v577);
                            float v588;
                            v588 = v587 + v582;
                            v586[v62] = Tuple2{v588, v583};
                            static_array_list<Union0,32> & v589 = v4.v0;
                            Union0 v590;
                            v590 = Union0{Union0_1{v62, v575}};
                            v589.push(v590);
                            Union6 v591;
                            v591 = Union6{Union6_2{v59, v60, v61, v62, v63, v64, v575}};
                            float v592;
                            v592 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v591);
                            static_array_list<Union0,32> & v593 = v4.v0;
                            Union0 v594;
                            v594 = v593.pop();
                            static_array<Tuple2,2> & v595 = v6.v0;
                            v595[v62] = Tuple2{v582, v583};
                            bool v596;
                            v596 = v62 == 0;
                            if (v596){
                                v600 = v592;
                            } else {
                                float v597;
                                v597 = -v592;
                                v600 = v597;
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    float v602;
                    v602 = v548[0];
                    static_array<Tuple2,2> & v604 = v6.v0;
                    float v607; float v608;
                    Tuple2 tmp19 = v604[v62];
                    v607 = tmp19.v0; v608 = tmp19.v1;
                    static_array<Tuple2,2> & v611 = v6.v0;
                    float v612;
                    v612 = log(v602);
                    float v613;
                    v613 = v612 + v607;
                    v611[v62] = Tuple2{v613, v608};
                    static_array_list<Union0,32> & v614 = v4.v0;
                    Union2 v615;
                    v615 = Union2{Union2_0{}};
                    Union0 v616;
                    v616 = Union0{Union0_1{v62, v615}};
                    v614.push(v616);
                    Union2 v617;
                    v617 = Union2{Union2_0{}};
                    Union6 v618;
                    v618 = Union6{Union6_2{v59, v60, v61, v62, v63, v64, v617}};
                    float v619;
                    v619 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v618);
                    static_array_list<Union0,32> & v620 = v4.v0;
                    Union0 v621;
                    v621 = v620.pop();
                    static_array<Tuple2,2> & v622 = v6.v0;
                    v622[v62] = Tuple2{v607, v608};
                    bool v623;
                    v623 = v62 == 0;
                    float v625;
                    if (v623){
                        v625 = v619;
                    } else {
                        float v624;
                        v624 = -v619;
                        v625 = v624;
                    }
                    static_array<float,3> v627;
                    v627[0] = v625;
                    v627[1] = v600;
                    v627[2] = v574;
                    static_array<float,3> v630;
                    int v632;
                    v632 = 0;
                    while (while_method_6(v632)){
                        float v635;
                        v635 = v627[v632];
                        float v638;
                        v638 = v548[v632];
                        float v640;
                        v640 = v635 * v638;
                        v630[v632] = v640;
                        v632 += 1 ;
                    }
                    int v641; float v642;
                    Tuple3 tmp20 = Tuple3{0, 0.0f};
                    v641 = tmp20.v0; v642 = tmp20.v1;
                    while (while_method_6(v641)){
                        float v645;
                        v645 = v630[v641];
                        float v647;
                        v647 = v642 + v645;
                        v642 = v647;
                        v641 += 1 ;
                    }
                    v650 = v642;
                    break;
                }
                case 1: { // TrainEnumerative
                    static_array<float,3> v158;
                    v158 = regret_match_6(v127, v153);
                    float v184;
                    switch (v145.tag) {
                        case 0: { // None
                            v184 = 0.0f;
                            break;
                        }
                        case 1: { // Some
                            Union2 v159 = v145.case1.v0;
                            float v161;
                            v161 = v158[2];
                            static_array<Tuple2,2> & v163 = v6.v0;
                            float v166; float v167;
                            Tuple2 tmp21 = v163[v62];
                            v166 = tmp21.v0; v167 = tmp21.v1;
                            static_array<Tuple2,2> & v170 = v6.v0;
                            float v171;
                            v171 = log(v161);
                            float v172;
                            v172 = v171 + v166;
                            v170[v62] = Tuple2{v172, v167};
                            static_array_list<Union0,32> & v173 = v4.v0;
                            Union0 v174;
                            v174 = Union0{Union0_1{v62, v159}};
                            v173.push(v174);
                            Union6 v175;
                            v175 = Union6{Union6_2{v59, v60, v61, v62, v63, v64, v159}};
                            float v176;
                            v176 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v175);
                            static_array_list<Union0,32> & v177 = v4.v0;
                            Union0 v178;
                            v178 = v177.pop();
                            static_array<Tuple2,2> & v179 = v6.v0;
                            v179[v62] = Tuple2{v166, v167};
                            bool v180;
                            v180 = v62 == 0;
                            if (v180){
                                v184 = v176;
                            } else {
                                float v181;
                                v181 = -v176;
                                v184 = v181;
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    float v210;
                    switch (v140.tag) {
                        case 0: { // None
                            v210 = 0.0f;
                            break;
                        }
                        case 1: { // Some
                            Union2 v185 = v140.case1.v0;
                            float v187;
                            v187 = v158[1];
                            static_array<Tuple2,2> & v189 = v6.v0;
                            float v192; float v193;
                            Tuple2 tmp22 = v189[v62];
                            v192 = tmp22.v0; v193 = tmp22.v1;
                            static_array<Tuple2,2> & v196 = v6.v0;
                            float v197;
                            v197 = log(v187);
                            float v198;
                            v198 = v197 + v192;
                            v196[v62] = Tuple2{v198, v193};
                            static_array_list<Union0,32> & v199 = v4.v0;
                            Union0 v200;
                            v200 = Union0{Union0_1{v62, v185}};
                            v199.push(v200);
                            Union6 v201;
                            v201 = Union6{Union6_2{v59, v60, v61, v62, v63, v64, v185}};
                            float v202;
                            v202 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v201);
                            static_array_list<Union0,32> & v203 = v4.v0;
                            Union0 v204;
                            v204 = v203.pop();
                            static_array<Tuple2,2> & v205 = v6.v0;
                            v205[v62] = Tuple2{v192, v193};
                            bool v206;
                            v206 = v62 == 0;
                            if (v206){
                                v210 = v202;
                            } else {
                                float v207;
                                v207 = -v202;
                                v210 = v207;
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    float v212;
                    v212 = v158[0];
                    static_array<Tuple2,2> & v214 = v6.v0;
                    float v217; float v218;
                    Tuple2 tmp23 = v214[v62];
                    v217 = tmp23.v0; v218 = tmp23.v1;
                    static_array<Tuple2,2> & v221 = v6.v0;
                    float v222;
                    v222 = log(v212);
                    float v223;
                    v223 = v222 + v217;
                    v221[v62] = Tuple2{v223, v218};
                    static_array_list<Union0,32> & v224 = v4.v0;
                    Union2 v225;
                    v225 = Union2{Union2_0{}};
                    Union0 v226;
                    v226 = Union0{Union0_1{v62, v225}};
                    v224.push(v226);
                    Union2 v227;
                    v227 = Union2{Union2_0{}};
                    Union6 v228;
                    v228 = Union6{Union6_2{v59, v60, v61, v62, v63, v64, v227}};
                    float v229;
                    v229 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v228);
                    static_array_list<Union0,32> & v230 = v4.v0;
                    Union0 v231;
                    v231 = v230.pop();
                    static_array<Tuple2,2> & v232 = v6.v0;
                    v232[v62] = Tuple2{v217, v218};
                    bool v233;
                    v233 = v62 == 0;
                    float v235;
                    if (v233){
                        v235 = v229;
                    } else {
                        float v234;
                        v234 = -v229;
                        v235 = v234;
                    }
                    static_array<float,3> v237;
                    v237[0] = v235;
                    v237[1] = v210;
                    v237[2] = v184;
                    static_array<float,3> v240;
                    int v242;
                    v242 = 0;
                    while (while_method_6(v242)){
                        float v245;
                        v245 = v237[v242];
                        float v248;
                        v248 = v158[v242];
                        float v250;
                        v250 = v245 * v248;
                        v240[v242] = v250;
                        v242 += 1 ;
                    }
                    int v251; float v252;
                    Tuple3 tmp24 = Tuple3{0, 0.0f};
                    v251 = tmp24.v0; v252 = tmp24.v1;
                    while (while_method_6(v251)){
                        float v255;
                        v255 = v240[v251];
                        float v257;
                        v257 = v252 + v255;
                        v252 = v257;
                        v251 += 1 ;
                    }
                    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v258 = v2.v0;
                    static_array<float,3> v260;
                    int v262;
                    v262 = 0;
                    while (while_method_6(v262)){
                        float v265;
                        v265 = v126[v262];
                        float v268;
                        v268 = v158[v262];
                        float v270;
                        v270 = 0.99609375f * v265;
                        float v271;
                        v271 = v270 + v268;
                        v260[v262] = v271;
                        v262 += 1 ;
                    }
                    static_array<Tuple2,2> & v272 = v6.v0;
                    int v273; float v274;
                    Tuple3 tmp25 = Tuple3{0, 0.0f};
                    v273 = tmp25.v0; v274 = tmp25.v1;
                    while (while_method_2(v273)){
                        float v278; float v279;
                        Tuple2 tmp26 = v272[v273];
                        v278 = tmp26.v0; v279 = tmp26.v1;
                        bool v282;
                        v282 = v273 == v62;
                        float v283;
                        if (v282){
                            v283 = 0.0f;
                        } else {
                            v283 = v278;
                        }
                        float v284;
                        v284 = v274 + v283;
                        float v285;
                        v285 = v284 - v279;
                        v274 = v285;
                        v273 += 1 ;
                    }
                    float v286;
                    v286 = exp(v274);
                    static_array<float,3> v288;
                    int v290;
                    v290 = 0;
                    while (while_method_6(v290)){
                        float v293;
                        v293 = v127[v290];
                        float v296;
                        v296 = v237[v290];
                        float v298;
                        v298 = v296 - v252;
                        float v299;
                        v299 = v286 * v298;
                        float v300;
                        v300 = v293 + v299;
                        bool v301;
                        v301 = 0.0f >= v300;
                        float v302;
                        if (v301){
                            v302 = 0.0f;
                        } else {
                            v302 = v300;
                        }
                        v288[v290] = v302;
                        v290 += 1 ;
                    }
                    v258[v89] = Tuple1{v260, v288, v128};
                    v650 = v252;
                    break;
                }
                case 2: { // TrainSampling
                    static_array<float,3> v303;
                    v303 = regret_match_6(v127, v153);
                    static_array<float,3> v304;
                    v304 = masking_normalize_5(v126, v153);
                    static_array<float,3> v306;
                    int v308;
                    v308 = 0;
                    while (while_method_6(v308)){
                        v306[v308] = 0.0f;
                        v308 += 1 ;
                    }
                    static_array<float,3> v310;
                    v310 = masking_normalize_5(v306, v153);
                    static_array<float,3> v312;
                    int v314;
                    v314 = 0;
                    while (while_method_6(v314)){
                        float v317;
                        v317 = v304[v314];
                        float v320;
                        v320 = v310[v314];
                        float v322;
                        v322 = 0.875f * v317;
                        float v323;
                        v323 = 0.125f * v320;
                        float v324;
                        v324 = v322 + v323;
                        v312[v314] = v324;
                        v314 += 1 ;
                    }
                    int v325;
                    v325 = sample_discrete__8(v312, v3);
                    StackMut1 v326{0.0f};
                    float v329; float v330;
                    Tuple2 tmp27 = v128[2];
                    v329 = tmp27.v0; v330 = tmp27.v1;
                    bool v333;
                    v333 = v330 == 0.0f;
                    bool v334;
                    v334 = v333 != true;
                    float v336;
                    if (v334){
                        float v335;
                        v335 = v329 / v330;
                        v336 = v335;
                    } else {
                        v336 = 0.0f;
                    }
                    float v372;
                    switch (v145.tag) {
                        case 1: { // Some
                            Union2 v337 = v145.case1.v0;
                            bool v338;
                            v338 = 2 == v325;
                            if (v338){
                                float v340;
                                v340 = v303[2];
                                float v343;
                                v343 = v312[2];
                                static_array<Tuple2,2> & v345 = v6.v0;
                                float v348; float v349;
                                Tuple2 tmp28 = v345[v62];
                                v348 = tmp28.v0; v349 = tmp28.v1;
                                static_array<Tuple2,2> & v352 = v6.v0;
                                float v353;
                                v353 = log(v340);
                                float v354;
                                v354 = v353 + v348;
                                float v355;
                                v355 = log(v343);
                                float v356;
                                v356 = v355 + v349;
                                v352[v62] = Tuple2{v354, v356};
                                static_array_list<Union0,32> & v357 = v4.v0;
                                Union0 v358;
                                v358 = Union0{Union0_1{v62, v337}};
                                v357.push(v358);
                                Union6 v359;
                                v359 = Union6{Union6_2{v59, v60, v61, v62, v63, v64, v337}};
                                float v360;
                                v360 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v359);
                                static_array_list<Union0,32> & v361 = v4.v0;
                                Union0 v362;
                                v362 = v361.pop();
                                static_array<Tuple2,2> & v363 = v6.v0;
                                v363[v62] = Tuple2{v348, v349};
                                bool v364;
                                v364 = v62 == 0;
                                float v366;
                                if (v364){
                                    v366 = v360;
                                } else {
                                    float v365;
                                    v365 = -v360;
                                    v366 = v365;
                                }
                                v326.v0 = v366;
                                float v367 = v326.v0;
                                float v368;
                                v368 = v367 - v336;
                                float v369;
                                v369 = v368 / v343;
                                float v370;
                                v370 = v369 + v336;
                                v372 = v370;
                            } else {
                                v372 = v336;
                            }
                            break;
                        }
                        default: {
                            v372 = v336;
                        }
                    }
                    float v375; float v376;
                    Tuple2 tmp29 = v128[1];
                    v375 = tmp29.v0; v376 = tmp29.v1;
                    bool v379;
                    v379 = v376 == 0.0f;
                    bool v380;
                    v380 = v379 != true;
                    float v382;
                    if (v380){
                        float v381;
                        v381 = v375 / v376;
                        v382 = v381;
                    } else {
                        v382 = 0.0f;
                    }
                    float v418;
                    switch (v140.tag) {
                        case 1: { // Some
                            Union2 v383 = v140.case1.v0;
                            bool v384;
                            v384 = 1 == v325;
                            if (v384){
                                float v386;
                                v386 = v303[1];
                                float v389;
                                v389 = v312[1];
                                static_array<Tuple2,2> & v391 = v6.v0;
                                float v394; float v395;
                                Tuple2 tmp30 = v391[v62];
                                v394 = tmp30.v0; v395 = tmp30.v1;
                                static_array<Tuple2,2> & v398 = v6.v0;
                                float v399;
                                v399 = log(v386);
                                float v400;
                                v400 = v399 + v394;
                                float v401;
                                v401 = log(v389);
                                float v402;
                                v402 = v401 + v395;
                                v398[v62] = Tuple2{v400, v402};
                                static_array_list<Union0,32> & v403 = v4.v0;
                                Union0 v404;
                                v404 = Union0{Union0_1{v62, v383}};
                                v403.push(v404);
                                Union6 v405;
                                v405 = Union6{Union6_2{v59, v60, v61, v62, v63, v64, v383}};
                                float v406;
                                v406 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v405);
                                static_array_list<Union0,32> & v407 = v4.v0;
                                Union0 v408;
                                v408 = v407.pop();
                                static_array<Tuple2,2> & v409 = v6.v0;
                                v409[v62] = Tuple2{v394, v395};
                                bool v410;
                                v410 = v62 == 0;
                                float v412;
                                if (v410){
                                    v412 = v406;
                                } else {
                                    float v411;
                                    v411 = -v406;
                                    v412 = v411;
                                }
                                v326.v0 = v412;
                                float v413 = v326.v0;
                                float v414;
                                v414 = v413 - v382;
                                float v415;
                                v415 = v414 / v389;
                                float v416;
                                v416 = v415 + v382;
                                v418 = v416;
                            } else {
                                v418 = v382;
                            }
                            break;
                        }
                        default: {
                            v418 = v382;
                        }
                    }
                    float v421; float v422;
                    Tuple2 tmp31 = v128[0];
                    v421 = tmp31.v0; v422 = tmp31.v1;
                    bool v425;
                    v425 = v422 == 0.0f;
                    bool v426;
                    v426 = v425 != true;
                    float v428;
                    if (v426){
                        float v427;
                        v427 = v421 / v422;
                        v428 = v427;
                    } else {
                        v428 = 0.0f;
                    }
                    bool v429;
                    v429 = 0 == v325;
                    float v464;
                    if (v429){
                        float v431;
                        v431 = v303[0];
                        float v434;
                        v434 = v312[0];
                        static_array<Tuple2,2> & v436 = v6.v0;
                        float v439; float v440;
                        Tuple2 tmp32 = v436[v62];
                        v439 = tmp32.v0; v440 = tmp32.v1;
                        static_array<Tuple2,2> & v443 = v6.v0;
                        float v444;
                        v444 = log(v431);
                        float v445;
                        v445 = v444 + v439;
                        float v446;
                        v446 = log(v434);
                        float v447;
                        v447 = v446 + v440;
                        v443[v62] = Tuple2{v445, v447};
                        static_array_list<Union0,32> & v448 = v4.v0;
                        Union2 v449;
                        v449 = Union2{Union2_0{}};
                        Union0 v450;
                        v450 = Union0{Union0_1{v62, v449}};
                        v448.push(v450);
                        Union2 v451;
                        v451 = Union2{Union2_0{}};
                        Union6 v452;
                        v452 = Union6{Union6_2{v59, v60, v61, v62, v63, v64, v451}};
                        float v453;
                        v453 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v452);
                        static_array_list<Union0,32> & v454 = v4.v0;
                        Union0 v455;
                        v455 = v454.pop();
                        static_array<Tuple2,2> & v456 = v6.v0;
                        v456[v62] = Tuple2{v439, v440};
                        bool v457;
                        v457 = v62 == 0;
                        float v459;
                        if (v457){
                            v459 = v453;
                        } else {
                            float v458;
                            v458 = -v453;
                            v459 = v458;
                        }
                        v326.v0 = v459;
                        float v460 = v326.v0;
                        float v461;
                        v461 = v460 - v428;
                        float v462;
                        v462 = v461 / v434;
                        float v463;
                        v463 = v462 + v428;
                        v464 = v463;
                    } else {
                        v464 = v428;
                    }
                    static_array<float,3> v466;
                    v466[0] = v464;
                    v466[1] = v418;
                    v466[2] = v372;
                    static_array<float,3> v469;
                    int v471;
                    v471 = 0;
                    while (while_method_6(v471)){
                        float v474;
                        v474 = v466[v471];
                        float v477;
                        v477 = v303[v471];
                        float v479;
                        v479 = v474 * v477;
                        v469[v471] = v479;
                        v471 += 1 ;
                    }
                    int v480; float v481;
                    Tuple3 tmp33 = Tuple3{0, 0.0f};
                    v480 = tmp33.v0; v481 = tmp33.v1;
                    while (while_method_6(v480)){
                        float v484;
                        v484 = v469[v480];
                        float v486;
                        v486 = v481 + v484;
                        v481 = v486;
                        v480 += 1 ;
                    }
                    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v487 = v2.v0;
                    static_array<float,3> v489;
                    int v491;
                    v491 = 0;
                    while (while_method_6(v491)){
                        float v494;
                        v494 = v126[v491];
                        float v497;
                        v497 = v303[v491];
                        float v499;
                        v499 = v494 + v497;
                        v489[v491] = v499;
                        v491 += 1 ;
                    }
                    static_array<Tuple2,2> & v500 = v6.v0;
                    int v501; float v502;
                    Tuple3 tmp34 = Tuple3{0, 0.0f};
                    v501 = tmp34.v0; v502 = tmp34.v1;
                    while (while_method_2(v501)){
                        float v506; float v507;
                        Tuple2 tmp35 = v500[v501];
                        v506 = tmp35.v0; v507 = tmp35.v1;
                        bool v510;
                        v510 = v501 == v62;
                        float v511;
                        if (v510){
                            v511 = 0.0f;
                        } else {
                            v511 = v506;
                        }
                        float v512;
                        v512 = v502 + v511;
                        float v513;
                        v513 = v512 - v507;
                        v502 = v513;
                        v501 += 1 ;
                    }
                    float v514;
                    v514 = exp(v502);
                    static_array<float,3> v516;
                    int v518;
                    v518 = 0;
                    while (while_method_6(v518)){
                        float v521;
                        v521 = v127[v518];
                        float v524;
                        v524 = v466[v518];
                        float v526;
                        v526 = v524 - v481;
                        float v527;
                        v527 = v514 * v526;
                        float v528;
                        v528 = v521 + v527;
                        bool v529;
                        v529 = 0.0f >= v528;
                        float v530;
                        if (v529){
                            v530 = 0.0f;
                        } else {
                            v530 = v528;
                        }
                        v516[v518] = v530;
                        v518 += 1 ;
                    }
                    static_array<Tuple2,3> v532;
                    int v534;
                    v534 = 0;
                    while (while_method_6(v534)){
                        float v538; float v539;
                        Tuple2 tmp36 = v128[v534];
                        v538 = tmp36.v0; v539 = tmp36.v1;
                        bool v542;
                        v542 = v325 == v534;
                        float v546; float v547;
                        if (v542){
                            float v543 = v326.v0;
                            float v544;
                            v544 = v538 + v543;
                            float v545;
                            v545 = v539 + 1.0f;
                            v546 = v544; v547 = v545;
                        } else {
                            v546 = v538; v547 = v539;
                        }
                        v532[v534] = Tuple2{v546, v547};
                        v534 += 1 ;
                    }
                    v487[v89] = Tuple1{v489, v516, v532};
                    v650 = v481;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            bool v651;
            v651 = v62 == 0;
            float v653;
            if (v651){
                v653 = v650;
            } else {
                float v652;
                v652 = -v650;
                v653 = v652;
            }
            v8.v0 = v653;
            Union6 v654;
            v654 = Union6{Union6_3{}};
            return loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v654);
            break;
        }
        case 3: { // RoundWithAction
            Union5 v656 = v7.case3.v0; bool v657 = v7.case3.v1; static_array<Union1,2> v658 = v7.case3.v2; int v659 = v7.case3.v3; static_array<int,2> v660 = v7.case3.v4; int v661 = v7.case3.v5; Union2 v662 = v7.case3.v6;
            static_array_list<Union0,32> & v663 = v4.v0;
            Union0 v664;
            v664 = Union0{Union0_1{v659, v662}};
            v663.push(v664);
            Union6 v665;
            v665 = Union6{Union6_2{v656, v657, v658, v659, v660, v661, v662}};
            float v666;
            v666 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v665);
            static_array_list<Union0,32> & v667 = v4.v0;
            Union0 v668;
            v668 = v667.pop();
            return v666;
            break;
        }
        case 4: { // TerminalCall
            Union5 v30 = v7.case4.v0; bool v31 = v7.case4.v1; static_array<Union1,2> v32 = v7.case4.v2; int v33 = v7.case4.v3; static_array<int,2> v34 = v7.case4.v4; int v35 = v7.case4.v5;
            int v37;
            v37 = v34[v33];
            Union9 v39;
            v39 = compare_hands_11(v30, v31, v32, v33, v34, v35);
            int v44; int v45;
            switch (v39.tag) {
                case 0: { // Eq
                    v44 = 0; v45 = -1;
                    break;
                }
                case 1: { // Gt
                    v44 = v37; v45 = 0;
                    break;
                }
                case 2: { // Lt
                    v44 = v37; v45 = 1;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            int v46;
            v46 = -v45;
            bool v47;
            v47 = v45 >= v46;
            int v48;
            if (v47){
                v48 = v45;
            } else {
                v48 = v46;
            }
            float v49;
            v49 = (float)v44;
            bool v50;
            v50 = v48 == 0;
            float v52;
            if (v50){
                v52 = v49;
            } else {
                float v51;
                v51 = -v49;
                v52 = v51;
            }
            v8.v0 = v52;
            static_array_list<Union0,32> & v53 = v4.v0;
            Union0 v54;
            v54 = Union0{Union0_3{v32, v44, v45}};
            v53.push(v54);
            Union6 v55;
            v55 = Union6{Union6_3{}};
            float v56;
            v56 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v55);
            static_array_list<Union0,32> & v57 = v4.v0;
            Union0 v58;
            v58 = v57.pop();
            return v56;
            break;
        }
        case 5: { // TerminalFold
            Union5 v9 = v7.case5.v0; bool v10 = v7.case5.v1; static_array<Union1,2> v11 = v7.case5.v2; int v12 = v7.case5.v3; static_array<int,2> v13 = v7.case5.v4; int v14 = v7.case5.v5;
            int v16;
            v16 = v13[v12];
            int v18;
            v18 = -v16;
            float v19;
            v19 = (float)v18;
            bool v20;
            v20 = v12 == 0;
            float v22;
            if (v20){
                v22 = v19;
            } else {
                float v21;
                v21 = -v19;
                v22 = v21;
            }
            v8.v0 = v22;
            int v23;
            v23 = v12 ^ 1;
            static_array_list<Union0,32> & v24 = v4.v0;
            Union0 v25;
            v25 = Union0{Union0_3{v11, v16, v23}};
            v24.push(v25);
            Union6 v26;
            v26 = Union6{Union6_3{}};
            float v27;
            v27 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v26);
            static_array_list<Union0,32> & v28 = v4.v0;
            Union0 v29;
            v29 = v28.pop();
            return v27;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
inline bool while_method_9(int v0){
    bool v1;
    v1 = v0 < 100;
    return v1;
}
int main() {
    int v0;
    v0 = 0;
    while (while_method_0(v0)){
        Fun0 v2 = FunPointerMethod0;
        Fun1 v3 = FunPointerMethod1;
        std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> v4(512, v2, v3);
        StackRefs0 v5{v4};
        static_array<Union3,2> v7;
        Union3 v10;
        v10 = Union3{Union3_2{}};
        v7[0] = v10;
        Union3 v13;
        v13 = Union3{Union3_2{}};
        v7[1] = v13;
        bool v15;
        v15 = true;
        xso::rng v16;
        StackMut0 v17{63u};
        static_array_list<Union0,32> v19;
        v19 = static_array_list<Union0,32>{};
        StackRefs1 v21{v19};
        static_array<Tuple2,2> v23;
        int v25;
        v25 = 0;
        while (while_method_2(v25)){
            v23[v25] = Tuple2{0.0f, 0.0f};
            v25 += 1 ;
        }
        StackRefs2 v27{v23};
        int v28; float v29;
        Tuple3 tmp2 = Tuple3{0, 0.0f};
        v28 = tmp2.v0; v29 = tmp2.v1;
        while (while_method_3(v28)){
            int v31;
            v31 = v28 % 40000;
            bool v32;
            v32 = v31 == 0;
            if (v32){
                printf("{%s = %d; %s = %d}\n","i", v28, "nearTo", 1000000);
                fflush(stdout);
            } else {
            }
            Union4 v38;
            v38 = Union4{Union4_1{}};
            float v39;
            v39 = body_0(v15, v7, v5, v16, v21, v17, v27, v38);
            v29 = v39;
            v28 += 1 ;
        }
        static_array<Union3,2> v41;
        Union3 v44;
        v44 = Union3{Union3_1{}};
        v41[0] = v44;
        Union3 v47;
        v47 = Union3{Union3_0{}};
        v41[1] = v47;
        bool v49;
        v49 = false;
        xso::rng v50;
        StackMut0 v51{63u};
        static_array_list<Union0,32> v53;
        v53 = static_array_list<Union0,32>{};
        StackRefs1 v55{v53};
        static_array<Tuple2,2> v57;
        int v59;
        v59 = 0;
        while (while_method_2(v59)){
            v57[v59] = Tuple2{0.0f, 0.0f};
            v59 += 1 ;
        }
        StackRefs2 v61{v57};
        int v62; float v63;
        Tuple3 tmp39 = Tuple3{0, 0.0f};
        v62 = tmp39.v0; v63 = tmp39.v1;
        while (while_method_9(v62)){
            int v65;
            v65 = v62 % 4;
            bool v66;
            v66 = v65 == 0;
            if (v66){
                printf("{%s = %d; %s = %d}\n","i", v62, "nearTo", 100);
                fflush(stdout);
            } else {
            }
            Union4 v72;
            v72 = Union4{Union4_1{}};
            float v73;
            v73 = body_0(v49, v41, v5, v50, v55, v51, v61, v72);
            v63 = v73;
            v62 += 1 ;
        }
        printf("{%s = %f}\n","reward_for_pl0", v63);
        fflush(stdout);
        v0 += 1 ;
    }
    return 0;
}
