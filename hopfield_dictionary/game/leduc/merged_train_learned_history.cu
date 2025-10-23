#include "merged_train_learned_history.auto.cu"
#include <thrust/device_vector.h>
#include <unordered_map>
#include <xoshiro.h>
struct Union1;
struct Union2;
struct Union0;
struct Tuple0;
typedef unsigned long long (* Fun0)(Tuple0);
typedef bool (* Fun1)(Tuple0, Tuple0);
struct Tuple1;
struct Tuple2;
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
struct Tuple7;
struct Union7;
struct Union8;
static_array<float,3> masking_normalize_5(static_array<float,3> v0, static_array<bool,3> v1);
static_array<float,3> relu_7(static_array<float,3> v0);
static_array<float,3> regret_match_6(static_array<float,3> v0, static_array<bool,3> v1);
struct Union9;
int loop_10(static_array<float,3> v0, float v1, int v2);
int pick_discrete__9(static_array<float,3> v0, float v1);
int sample_discrete__8(static_array<float,3> v0, xso::rng & v1);
struct Union10;
int tag_12(Union1 v0);
bool is_pair_13(int v0, int v1);
Tuple5 order_14(int v0, int v1);
Union10 compare_hands_11(Union5 v0, bool v1, static_array<Union1,2> v2, int v3, static_array<int,2> v4, int v5);
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
struct Tuple2 {
    float v0;
    float v1;
    __host__ __device__ Tuple2() = default;
    __host__ __device__ Tuple2(float t0, float t1) : v0(t0), v1(t1) {}
};
struct StackRefs0 {
    std::unordered_map<Tuple0, static_array<Tuple2,3>, Fun0, Fun1> & v0;
    std::unordered_map<Tuple0, Tuple1, Fun0, Fun1> & v1;
    __host__ __device__ StackRefs0() = default;
    __host__ __device__ StackRefs0(std::unordered_map<Tuple0, static_array<Tuple2,3>, Fun0, Fun1> & t0, std::unordered_map<Tuple0, Tuple1, Fun0, Fun1> & t1) : v0(t0), v1(t1) {}
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
struct Tuple7 {
    unsigned long long v1;
    unsigned long long v2;
    int v0;
    __host__ __device__ Tuple7() = default;
    __host__ __device__ Tuple7(int t0, unsigned long long t1, unsigned long long t2) : v0(t0), v1(t1), v2(t2) {}
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
struct Union9_0 { // None
};
struct Union9_1 { // Some
    static_array<Tuple2,3> v0;
    __host__ __device__ Union9_1(static_array<Tuple2,3> t0) : v0(t0) {}
    __host__ __device__ Union9_1() = delete;
};
struct Union9 {
    union {
        Union9_0 case0; // None
        Union9_1 case1; // Some
    };
    unsigned char tag{255};
    __host__ __device__ Union9() {}
    __host__ __device__ Union9(Union9_0 t) : tag(0), case0(t) {} // None
    __host__ __device__ Union9(Union9_1 t) : tag(1), case1(t) {} // Some
    __host__ __device__ Union9(const Union9 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union9_0(x.case0); break; // None
            case 1: new (&this->case1) Union9_1(x.case1); break; // Some
        }
    }
    __host__ __device__ Union9(const Union9 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union9_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union9_1(std::move(x.case1)); break; // Some
        }
    }
    __host__ __device__ Union9 & operator=(const Union9 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
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
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union9();
            new (this) Union9{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union9() {
        switch(this->tag){
            case 0: this->case0.~Union9_0(); break; // None
            case 1: this->case1.~Union9_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union10_0 { // Eq
};
struct Union10_1 { // Gt
};
struct Union10_2 { // Lt
};
struct Union10 {
    union {
        Union10_0 case0; // Eq
        Union10_1 case1; // Gt
        Union10_2 case2; // Lt
    };
    unsigned char tag{255};
    __host__ __device__ Union10() {}
    __host__ __device__ Union10(Union10_0 t) : tag(0), case0(t) {} // Eq
    __host__ __device__ Union10(Union10_1 t) : tag(1), case1(t) {} // Gt
    __host__ __device__ Union10(Union10_2 t) : tag(2), case2(t) {} // Lt
    __host__ __device__ Union10(const Union10 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union10_0(x.case0); break; // Eq
            case 1: new (&this->case1) Union10_1(x.case1); break; // Gt
            case 2: new (&this->case2) Union10_2(x.case2); break; // Lt
        }
    }
    __host__ __device__ Union10(const Union10 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union10_0(std::move(x.case0)); break; // Eq
            case 1: new (&this->case1) Union10_1(std::move(x.case1)); break; // Gt
            case 2: new (&this->case2) Union10_2(std::move(x.case2)); break; // Lt
        }
    }
    __host__ __device__ Union10 & operator=(const Union10 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Eq
                case 1: this->case1 = x.case1; break; // Gt
                case 2: this->case2 = x.case2; break; // Lt
            }
        } else {
            this->~Union10();
            new (this) Union10{x};
        }
        return *this;
    }
    __host__ __device__ Union10 & operator=(const Union10 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // Eq
                case 1: this->case1 = std::move(x.case1); break; // Gt
                case 2: this->case2 = std::move(x.case2); break; // Lt
            }
        } else {
            this->~Union10();
            new (this) Union10{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union10() {
        switch(this->tag){
            case 0: this->case0.~Union10_0(); break; // Eq
            case 1: this->case1.~Union10_1(); break; // Gt
            case 2: this->case2.~Union10_2(); break; // Lt
        }
        this->tag = 255;
    }
};
inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 30;
    return v1;
}
unsigned long long FunPointerMethod0(Tuple0 tup0){
    unsigned long long v0 = tup0.v0; static_array_list<Union0,32> v1 = tup0.v1;
    return v0;
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
bool FunPointerMethod1(Tuple0 tup0, Tuple0 tup1){
    unsigned long long v0 = tup0.v0; static_array_list<Union0,32> v1 = tup0.v1; unsigned long long v2 = tup1.v0; static_array_list<Union0,32> v3 = tup1.v1;
    bool v4;
    v4 = v0 == v2;
    if (v4){
        int v5;
        v5 = v1.length;
        int v6;
        v6 = v3.length;
        bool v7;
        v7 = v5 == v6;
        if (v7){
            bool v8;
            v8 = true;
            int v9;
            v9 = v3.length;
            int v10;
            v10 = 0;
            while (while_method_1(v9, v10)){
                Union0 v13;
                v13 = v1[v10];
                Union0 v16;
                v16 = v3[v10];
                bool v57;
                switch (v13.tag == v16.tag ? v13.tag : 255) {
                    case 0: { // CommunityCardIs
                        Union1 v18 = v13.case0.v0;
                        Union1 v19 = v16.case0.v0;
                        switch (v18.tag == v19.tag ? v18.tag : 255) {
                            case 0: { // Jack
                                v57 = true;
                                break;
                            }
                            case 1: { // King
                                v57 = true;
                                break;
                            }
                            case 2: { // Queen
                                v57 = true;
                                break;
                            }
                            default: {
                                v57 = false;
                            }
                        }
                        break;
                    }
                    case 1: { // PlayerAction
                        int v21 = v13.case1.v0; Union2 v22 = v13.case1.v1;
                        int v23 = v16.case1.v0; Union2 v24 = v16.case1.v1;
                        bool v25;
                        v25 = v21 == v23;
                        if (v25){
                            switch (v22.tag == v24.tag ? v22.tag : 255) {
                                case 0: { // Call
                                    v57 = true;
                                    break;
                                }
                                case 1: { // Fold
                                    v57 = true;
                                    break;
                                }
                                case 2: { // Raise
                                    v57 = true;
                                    break;
                                }
                                default: {
                                    v57 = false;
                                }
                            }
                        } else {
                            v57 = false;
                        }
                        break;
                    }
                    case 2: { // PlayerGotCard
                        int v28 = v13.case2.v0; Union1 v29 = v13.case2.v1;
                        int v30 = v16.case2.v0; Union1 v31 = v16.case2.v1;
                        bool v32;
                        v32 = v28 == v30;
                        if (v32){
                            switch (v29.tag == v31.tag ? v29.tag : 255) {
                                case 0: { // Jack
                                    v57 = true;
                                    break;
                                }
                                case 1: { // King
                                    v57 = true;
                                    break;
                                }
                                case 2: { // Queen
                                    v57 = true;
                                    break;
                                }
                                default: {
                                    v57 = false;
                                }
                            }
                        } else {
                            v57 = false;
                        }
                        break;
                    }
                    case 3: { // Showdown
                        static_array<Union1,2> v35 = v13.case3.v0; int v36 = v13.case3.v1; int v37 = v13.case3.v2;
                        static_array<Union1,2> v38 = v16.case3.v0; int v39 = v16.case3.v1; int v40 = v16.case3.v2;
                        bool v41;
                        v41 = true;
                        int v42;
                        v42 = 0;
                        while (while_method_2(v42)){
                            Union1 v45;
                            v45 = v35[v42];
                            Union1 v48;
                            v48 = v38[v42];
                            bool v50;
                            switch (v45.tag == v48.tag ? v45.tag : 255) {
                                case 0: { // Jack
                                    v50 = true;
                                    break;
                                }
                                case 1: { // King
                                    v50 = true;
                                    break;
                                }
                                case 2: { // Queen
                                    v50 = true;
                                    break;
                                }
                                default: {
                                    v50 = false;
                                }
                            }
                            bool v51;
                            v51 = v50 != true;
                            if (v51){
                                bool v52;
                                v52 = false;
                                v41 = v52;
                                break;
                            } else {
                            }
                            v42 += 1 ;
                        }
                        if (v41){
                            bool v53;
                            v53 = v36 == v39;
                            if (v53){
                                bool v54;
                                v54 = v37 == v40;
                                v57 = v54;
                            } else {
                                v57 = false;
                            }
                        } else {
                            v57 = false;
                        }
                        break;
                    }
                    default: {
                        v57 = false;
                    }
                }
                bool v58;
                v58 = v57 != true;
                if (v58){
                    bool v59;
                    v59 = false;
                    v8 = v59;
                    break;
                } else {
                }
                v10 += 1 ;
            }
            return v8;
        } else {
            return false;
        }
    } else {
        return false;
    }
}
inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 2000000;
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
            Tuple5 tmp2 = Tuple5{0, 0};
            v18 = tmp2.v0; v19 = tmp2.v1;
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
                                Tuple5 tmp3 = Tuple5{0, 0};
                                v117 = tmp3.v0; v118 = tmp3.v1;
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
                                Tuple5 tmp4 = Tuple5{0, 0};
                                v61 = tmp4.v0; v62 = tmp4.v1;
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
                                Tuple5 tmp5 = Tuple5{0, 0};
                                v79 = tmp5.v0; v80 = tmp5.v1;
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
Union10 compare_hands_11(Union5 v0, bool v1, static_array<Union1,2> v2, int v3, static_array<int,2> v4, int v5){
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
                        return Union10{Union10_2{}};
                    } else {
                        bool v21;
                        v21 = v12 > v16;
                        if (v21){
                            return Union10{Union10_1{}};
                        } else {
                            return Union10{Union10_0{}};
                        }
                    }
                } else {
                    return Union10{Union10_1{}};
                }
            } else {
                if (v18){
                    return Union10{Union10_2{}};
                } else {
                    int v29; int v30;
                    Tuple5 tmp45 = order_14(v8, v12);
                    v29 = tmp45.v0; v30 = tmp45.v1;
                    int v31; int v32;
                    Tuple5 tmp46 = order_14(v8, v16);
                    v31 = tmp46.v0; v32 = tmp46.v1;
                    bool v33;
                    v33 = v29 < v31;
                    Union10 v39;
                    if (v33){
                        v39 = Union10{Union10_2{}};
                    } else {
                        bool v35;
                        v35 = v29 > v31;
                        if (v35){
                            v39 = Union10{Union10_1{}};
                        } else {
                            v39 = Union10{Union10_0{}};
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
                            return Union10{Union10_2{}};
                        } else {
                            bool v43;
                            v43 = v30 > v32;
                            if (v43){
                                return Union10{Union10_1{}};
                            } else {
                                return Union10{Union10_0{}};
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
            Union5 v928 = v7.case0.v0; bool v929 = v7.case0.v1; static_array<Union1,2> v930 = v7.case0.v2; int v931 = v7.case0.v3; static_array<int,2> v932 = v7.case0.v4; int v933 = v7.case0.v5;
            if (v0){
                unsigned int v934 = v5.v0;
                Union1 v935; unsigned int v936;
                Tuple4 tmp1 = draw_card_1(v3, v934);
                v935 = tmp1.v0; v936 = tmp1.v1;
                v5.v0 = v936;
                static_array_list<Union0,32> & v937 = v4.v0;
                Union0 v938;
                v938 = Union0{Union0_0{v935}};
                v937.push(v938);
                Union6 v939;
                v939 = Union6{Union6_0{v928, v929, v930, v931, v932, v933, v935}};
                float v940;
                v940 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v939);
                static_array_list<Union0,32> & v941 = v4.v0;
                Union0 v942;
                v942 = v941.pop();
                v5.v0 = v934;
                return v940;
            } else {
                int v943; float v944; float v945;
                Tuple6 tmp6 = Tuple6{0, 0.0f, 0.0f};
                v943 = tmp6.v0; v944 = tmp6.v1; v945 = tmp6.v2;
                while (while_method_5(v943)){
                    unsigned int v947 = v5.v0;
                    unsigned int v948;
                    v948 = 1u << v943;
                    unsigned int v949;
                    v949 = v947 & v948;
                    bool v950;
                    v950 = v949 == 0u;
                    bool v951;
                    v951 = v950 != true;
                    float v983; float v984;
                    if (v951){
                        unsigned int v952 = v5.v0;
                        unsigned int v953;
                        v953 = v952 ^ v948;
                        v5.v0 = v953;
                        bool v954;
                        v954 = 0 == v943;
                        Union1 v972;
                        if (v954){
                            v972 = Union1{Union1_1{}};
                        } else {
                            bool v956;
                            v956 = 1 == v943;
                            if (v956){
                                v972 = Union1{Union1_1{}};
                            } else {
                                bool v958;
                                v958 = 2 == v943;
                                if (v958){
                                    v972 = Union1{Union1_2{}};
                                } else {
                                    bool v960;
                                    v960 = 3 == v943;
                                    if (v960){
                                        v972 = Union1{Union1_2{}};
                                    } else {
                                        bool v962;
                                        v962 = 4 == v943;
                                        if (v962){
                                            v972 = Union1{Union1_0{}};
                                        } else {
                                            bool v964;
                                            v964 = 5 == v943;
                                            if (v964){
                                                v972 = Union1{Union1_0{}};
                                            } else {
                                                printf("%s\n", "Invalid int in int_to_card.");
                                                exit(-1);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        static_array_list<Union0,32> & v973 = v4.v0;
                        Union0 v974;
                        v974 = Union0{Union0_0{v972}};
                        v973.push(v974);
                        Union6 v975;
                        v975 = Union6{Union6_0{v928, v929, v930, v931, v932, v933, v972}};
                        float v976;
                        v976 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v975);
                        static_array_list<Union0,32> & v977 = v4.v0;
                        Union0 v978;
                        v978 = v977.pop();
                        unsigned int v979 = v5.v0;
                        unsigned int v980;
                        v980 = v979 ^ v948;
                        v5.v0 = v980;
                        float v981;
                        v981 = v944 + v976;
                        float v982;
                        v982 = v945 + 1.0f;
                        v983 = v981; v984 = v982;
                    } else {
                        v983 = v944; v984 = v945;
                    }
                    v944 = v983;
                    v945 = v984;
                    v943 += 1 ;
                }
                bool v985;
                v985 = v945 == 0.0f;
                bool v986;
                v986 = v985 != true;
                if (v986){
                    float v987;
                    v987 = v944 / v945;
                    return v987;
                } else {
                    return 0.0f;
                }
            }
            break;
        }
        case 1: { // ChanceInit
            if (v0){
                unsigned int v990 = v5.v0;
                Union1 v991; unsigned int v992;
                Tuple4 tmp7 = draw_card_1(v3, v990);
                v991 = tmp7.v0; v992 = tmp7.v1;
                v5.v0 = v992;
                unsigned int v993 = v5.v0;
                Union1 v994; unsigned int v995;
                Tuple4 tmp8 = draw_card_1(v3, v993);
                v994 = tmp8.v0; v995 = tmp8.v1;
                v5.v0 = v995;
                static_array_list<Union0,32> & v996 = v4.v0;
                Union0 v997;
                v997 = Union0{Union0_2{0, v991}};
                v996.push(v997);
                static_array_list<Union0,32> & v998 = v4.v0;
                Union0 v999;
                v999 = Union0{Union0_2{1, v994}};
                v998.push(v999);
                Union6 v1000;
                v1000 = Union6{Union6_1{v991, v994}};
                float v1001;
                v1001 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v1000);
                static_array_list<Union0,32> & v1002 = v4.v0;
                Union0 v1003;
                v1003 = v1002.pop();
                static_array_list<Union0,32> & v1004 = v4.v0;
                Union0 v1005;
                v1005 = v1004.pop();
                v5.v0 = v993;
                v5.v0 = v990;
                return v1001;
            } else {
                int v1006; float v1007; float v1008;
                Tuple6 tmp9 = Tuple6{0, 0.0f, 0.0f};
                v1006 = tmp9.v0; v1007 = tmp9.v1; v1008 = tmp9.v2;
                while (while_method_5(v1006)){
                    unsigned int v1010 = v5.v0;
                    unsigned int v1011;
                    v1011 = 1u << v1006;
                    unsigned int v1012;
                    v1012 = v1010 & v1011;
                    bool v1013;
                    v1013 = v1012 == 0u;
                    bool v1014;
                    v1014 = v1013 != true;
                    float v1090; float v1091;
                    if (v1014){
                        unsigned int v1015 = v5.v0;
                        unsigned int v1016;
                        v1016 = v1015 ^ v1011;
                        v5.v0 = v1016;
                        bool v1017;
                        v1017 = 0 == v1006;
                        Union1 v1035;
                        if (v1017){
                            v1035 = Union1{Union1_1{}};
                        } else {
                            bool v1019;
                            v1019 = 1 == v1006;
                            if (v1019){
                                v1035 = Union1{Union1_1{}};
                            } else {
                                bool v1021;
                                v1021 = 2 == v1006;
                                if (v1021){
                                    v1035 = Union1{Union1_2{}};
                                } else {
                                    bool v1023;
                                    v1023 = 3 == v1006;
                                    if (v1023){
                                        v1035 = Union1{Union1_2{}};
                                    } else {
                                        bool v1025;
                                        v1025 = 4 == v1006;
                                        if (v1025){
                                            v1035 = Union1{Union1_0{}};
                                        } else {
                                            bool v1027;
                                            v1027 = 5 == v1006;
                                            if (v1027){
                                                v1035 = Union1{Union1_0{}};
                                            } else {
                                                printf("%s\n", "Invalid int in int_to_card.");
                                                exit(-1);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        int v1036; float v1037; float v1038;
                        Tuple6 tmp10 = Tuple6{0, 0.0f, 0.0f};
                        v1036 = tmp10.v0; v1037 = tmp10.v1; v1038 = tmp10.v2;
                        while (while_method_5(v1036)){
                            unsigned int v1040 = v5.v0;
                            unsigned int v1041;
                            v1041 = 1u << v1036;
                            unsigned int v1042;
                            v1042 = v1040 & v1041;
                            bool v1043;
                            v1043 = v1042 == 0u;
                            bool v1044;
                            v1044 = v1043 != true;
                            float v1080; float v1081;
                            if (v1044){
                                unsigned int v1045 = v5.v0;
                                unsigned int v1046;
                                v1046 = v1045 ^ v1041;
                                v5.v0 = v1046;
                                bool v1047;
                                v1047 = 0 == v1036;
                                Union1 v1065;
                                if (v1047){
                                    v1065 = Union1{Union1_1{}};
                                } else {
                                    bool v1049;
                                    v1049 = 1 == v1036;
                                    if (v1049){
                                        v1065 = Union1{Union1_1{}};
                                    } else {
                                        bool v1051;
                                        v1051 = 2 == v1036;
                                        if (v1051){
                                            v1065 = Union1{Union1_2{}};
                                        } else {
                                            bool v1053;
                                            v1053 = 3 == v1036;
                                            if (v1053){
                                                v1065 = Union1{Union1_2{}};
                                            } else {
                                                bool v1055;
                                                v1055 = 4 == v1036;
                                                if (v1055){
                                                    v1065 = Union1{Union1_0{}};
                                                } else {
                                                    bool v1057;
                                                    v1057 = 5 == v1036;
                                                    if (v1057){
                                                        v1065 = Union1{Union1_0{}};
                                                    } else {
                                                        printf("%s\n", "Invalid int in int_to_card.");
                                                        exit(-1);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                static_array_list<Union0,32> & v1066 = v4.v0;
                                Union0 v1067;
                                v1067 = Union0{Union0_2{0, v1035}};
                                v1066.push(v1067);
                                static_array_list<Union0,32> & v1068 = v4.v0;
                                Union0 v1069;
                                v1069 = Union0{Union0_2{1, v1065}};
                                v1068.push(v1069);
                                Union6 v1070;
                                v1070 = Union6{Union6_1{v1035, v1065}};
                                float v1071;
                                v1071 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v1070);
                                static_array_list<Union0,32> & v1072 = v4.v0;
                                Union0 v1073;
                                v1073 = v1072.pop();
                                static_array_list<Union0,32> & v1074 = v4.v0;
                                Union0 v1075;
                                v1075 = v1074.pop();
                                unsigned int v1076 = v5.v0;
                                unsigned int v1077;
                                v1077 = v1076 ^ v1041;
                                v5.v0 = v1077;
                                float v1078;
                                v1078 = v1037 + v1071;
                                float v1079;
                                v1079 = v1038 + 1.0f;
                                v1080 = v1078; v1081 = v1079;
                            } else {
                                v1080 = v1037; v1081 = v1038;
                            }
                            v1037 = v1080;
                            v1038 = v1081;
                            v1036 += 1 ;
                        }
                        bool v1082;
                        v1082 = v1038 == 0.0f;
                        bool v1083;
                        v1083 = v1082 != true;
                        float v1085;
                        if (v1083){
                            float v1084;
                            v1084 = v1037 / v1038;
                            v1085 = v1084;
                        } else {
                            v1085 = 0.0f;
                        }
                        unsigned int v1086 = v5.v0;
                        unsigned int v1087;
                        v1087 = v1086 ^ v1011;
                        v5.v0 = v1087;
                        float v1088;
                        v1088 = v1007 + v1085;
                        float v1089;
                        v1089 = v1008 + 1.0f;
                        v1090 = v1088; v1091 = v1089;
                    } else {
                        v1090 = v1007; v1091 = v1008;
                    }
                    v1007 = v1090;
                    v1008 = v1091;
                    v1006 += 1 ;
                }
                bool v1092;
                v1092 = v1008 == 0.0f;
                bool v1093;
                v1093 = v1092 != true;
                if (v1093){
                    float v1094;
                    v1094 = v1007 / v1008;
                    return v1094;
                } else {
                    return 0.0f;
                }
            }
            break;
        }
        case 2: { // Round
            Union5 v59 = v7.case2.v0; bool v60 = v7.case2.v1; static_array<Union1,2> v61 = v7.case2.v2; int v62 = v7.case2.v3; static_array<int,2> v63 = v7.case2.v4; int v64 = v7.case2.v5;
            static_array_list<Union0,32> & v65 = v4.v0;
            static_array_list<Union0,32> & v66 = v4.v0;
            int v67;
            v67 = v66.length;
            bool v68;
            v68 = 32 >= v67;
            bool v69;
            v69 = v68 == false;
            if (v69){
                assert("The type level dimension has to equal the value passed at runtime into create." && v68);
            } else {
            }
            static_array_list<Union0,32> v72;
            v72 = static_array_list<Union0,32>{};
            v72.unsafe_set_length(v67);
            int v74; int v75;
            Tuple5 tmp11 = Tuple5{0, 0};
            v74 = tmp11.v0; v75 = tmp11.v1;
            while (while_method_1(v67, v74)){
                Union0 v78;
                v78 = v66[v74];
                bool v83;
                switch (v78.tag) {
                    case 2: { // PlayerGotCard
                        int v80 = v78.case2.v0; Union1 v81 = v78.case2.v1;
                        bool v82;
                        v82 = v80 == v62;
                        v83 = v82;
                        break;
                    }
                    default: {
                        v83 = true;
                    }
                }
                int v85;
                if (v83){
                    v72[v75] = v78;
                    int v84;
                    v84 = v75 + 1;
                    v85 = v84;
                } else {
                    v85 = v75;
                }
                v75 = v85;
                v74 += 1 ;
            }
            bool v86;
            v86 = 32 >= v75;
            bool v87;
            v87 = v86 == false;
            if (v87){
                assert("The type level dimension has to equal the value passed at runtime into create." && v86);
            } else {
            }
            static_array_list<Union0,32> v90;
            v90 = static_array_list<Union0,32>{};
            v90.unsafe_set_length(v75);
            int v92;
            v92 = 0;
            while (while_method_1(v75, v92)){
                Union0 v95;
                v95 = v72[v92];
                v90[v92] = v95;
                v92 += 1 ;
            }
            std::unordered_map<Tuple0, Tuple1, Fun0, Fun1> & v97 = v2.v1;
            int v98;
            v98 = v90.length;
            int v99; unsigned long long v100; unsigned long long v101;
            Tuple7 tmp12 = Tuple7{0, 0ull, 1ull};
            v99 = tmp12.v0; v100 = tmp12.v1; v101 = tmp12.v2;
            while (while_method_1(v98, v99)){
                Union0 v104;
                v104 = v90[v99];
                unsigned long long v149;
                switch (v104.tag) {
                    case 0: { // CommunityCardIs
                        Union1 v106 = v104.case0.v0;
                        unsigned long long v107;
                        switch (v106.tag) {
                            case 0: { // Jack
                                v107 = 9223372036854765835ull;
                                break;
                            }
                            case 1: { // King
                                v107 = 18446744073709531670ull;
                                break;
                            }
                            case 2: { // Queen
                                v107 = 9223372036854745889ull;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                                exit(-1);
                            }
                        }
                        unsigned long long v108;
                        v108 = 9223372036854775807ull + v107;
                        unsigned long long v109;
                        v109 = v108 * 9973ull;
                        v149 = v109;
                        break;
                    }
                    case 1: { // PlayerAction
                        int v110 = v104.case1.v0; Union2 v111 = v104.case1.v1;
                        unsigned long long v112;
                        v112 = std::hash<int>()(v110);
                        unsigned long long v113;
                        v113 = v112 * 9973ull;
                        unsigned long long v114;
                        switch (v111.tag) {
                            case 0: { // Call
                                v114 = 9223372036854765835ull;
                                break;
                            }
                            case 1: { // Fold
                                v114 = 18446744073709531670ull;
                                break;
                            }
                            case 2: { // Raise
                                v114 = 9223372036854745889ull;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                                exit(-1);
                            }
                        }
                        unsigned long long v115;
                        v115 = v113 + v114;
                        unsigned long long v116;
                        v116 = 9223372036854775807ull + v115;
                        unsigned long long v117;
                        v117 = v116 * 9973ull;
                        unsigned long long v118;
                        v118 = v117 * 2ull;
                        v149 = v118;
                        break;
                    }
                    case 2: { // PlayerGotCard
                        int v119 = v104.case2.v0; Union1 v120 = v104.case2.v1;
                        unsigned long long v121;
                        v121 = std::hash<int>()(v119);
                        unsigned long long v122;
                        v122 = v121 * 9973ull;
                        unsigned long long v123;
                        switch (v120.tag) {
                            case 0: { // Jack
                                v123 = 9223372036854765835ull;
                                break;
                            }
                            case 1: { // King
                                v123 = 18446744073709531670ull;
                                break;
                            }
                            case 2: { // Queen
                                v123 = 9223372036854745889ull;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                                exit(-1);
                            }
                        }
                        unsigned long long v124;
                        v124 = v122 + v123;
                        unsigned long long v125;
                        v125 = 9223372036854775807ull + v124;
                        unsigned long long v126;
                        v126 = v125 * 9973ull;
                        unsigned long long v127;
                        v127 = v126 * 3ull;
                        v149 = v127;
                        break;
                    }
                    case 3: { // Showdown
                        static_array<Union1,2> v128 = v104.case3.v0; int v129 = v104.case3.v1; int v130 = v104.case3.v2;
                        unsigned long long v131;
                        v131 = std::hash<int>()(v130);
                        unsigned long long v132;
                        v132 = std::hash<int>()(v129);
                        unsigned long long v133;
                        v133 = v132 * 9973ull;
                        unsigned long long v134;
                        v134 = v131 + v133;
                        int v135; unsigned long long v136; unsigned long long v137;
                        Tuple7 tmp13 = Tuple7{0, 0ull, 1ull};
                        v135 = tmp13.v0; v136 = tmp13.v1; v137 = tmp13.v2;
                        while (while_method_2(v135)){
                            Union1 v140;
                            v140 = v128[v135];
                            unsigned long long v142;
                            switch (v140.tag) {
                                case 0: { // Jack
                                    v142 = 9223372036854765835ull;
                                    break;
                                }
                                case 1: { // King
                                    v142 = 18446744073709531670ull;
                                    break;
                                }
                                case 2: { // Queen
                                    v142 = 9223372036854745889ull;
                                    break;
                                }
                                default: {
                                    assert("Invalid tag." && false);
                                    exit(-1);
                                }
                            }
                            unsigned long long v143;
                            v143 = v142 * v137;
                            unsigned long long v144;
                            v144 = v136 + v143;
                            unsigned long long v145;
                            v145 = v137 * 9973ull;
                            v136 = v144;
                            v137 = v145;
                            v135 += 1 ;
                        }
                        unsigned long long v146;
                        v146 = 9223372036854775807ull + v134;
                        unsigned long long v147;
                        v147 = v146 * 9973ull;
                        unsigned long long v148;
                        v148 = v147 * 4ull;
                        v149 = v148;
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                        exit(-1);
                    }
                }
                unsigned long long v150;
                v150 = v149 * v101;
                unsigned long long v151;
                v151 = v100 + v150;
                unsigned long long v152;
                v152 = v101 * 9973ull;
                v100 = v151;
                v101 = v152;
                v99 += 1 ;
            }
            auto v153 = v97.find(Tuple0{0ull, v90});
            bool v154;
            v154 = v153 != v97.end();
            Union7 v159;
            if (v154){
                static_array<float,3> v155; static_array<float,3> v156;
                Tuple1 tmp14 = v153->second;
                v155 = tmp14.v0; v156 = tmp14.v1;
                v159 = Union7{Union7_1{v155, v156}};
            } else {
                v159 = Union7{Union7_0{}};
            }
            static_array<float,3> v174; static_array<float,3> v175;
            switch (v159.tag) {
                case 0: { // None
                    static_array<float,3> v163;
                    int v165;
                    v165 = 0;
                    while (while_method_6(v165)){
                        v163[v165] = 0.0f;
                        v165 += 1 ;
                    }
                    static_array<float,3> v168;
                    int v170;
                    v170 = 0;
                    while (while_method_6(v170)){
                        v168[v170] = 0.0f;
                        v170 += 1 ;
                    }
                    v174 = v163; v175 = v168;
                    break;
                }
                case 1: { // Some
                    static_array<float,3> v160 = v159.case1.v0; static_array<float,3> v161 = v159.case1.v1;
                    v174 = v160; v175 = v161;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            int v177;
            v177 = v63[0];
            int v180;
            v180 = v63[1];
            bool v182;
            v182 = v177 == v180;
            bool v183;
            v183 = v182 != true;
            Union8 v187;
            if (v183){
                Union2 v184;
                v184 = Union2{Union2_1{}};
                v187 = Union8{Union8_1{v184}};
            } else {
                v187 = Union8{Union8_0{}};
            }
            bool v188;
            v188 = v64 > 0;
            Union8 v192;
            if (v188){
                Union2 v189;
                v189 = Union2{Union2_2{}};
                v192 = Union8{Union8_1{v189}};
            } else {
                v192 = Union8{Union8_0{}};
            }
            bool v195;
            switch (v192.tag) {
                case 0: { // None
                    v195 = false;
                    break;
                }
                case 1: { // Some
                    Union2 v193 = v192.case1.v0;
                    v195 = true;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            bool v198;
            switch (v187.tag) {
                case 0: { // None
                    v198 = false;
                    break;
                }
                case 1: { // Some
                    Union2 v196 = v187.case1.v0;
                    v198 = true;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            static_array<bool,3> v200;
            v200[0] = true;
            v200[1] = v198;
            v200[2] = v195;
            Union3 v203;
            v203 = v1[v62];
            float v909;
            switch (v203.tag) {
                case 0: { // Frozen
                    static_array<float,3> v815;
                    v815 = masking_normalize_5(v174, v200);
                    float v841;
                    switch (v192.tag) {
                        case 0: { // None
                            v841 = 0.0f;
                            break;
                        }
                        case 1: { // Some
                            Union2 v816 = v192.case1.v0;
                            float v818;
                            v818 = v815[2];
                            static_array<Tuple2,2> & v820 = v6.v0;
                            float v823; float v824;
                            Tuple2 tmp17 = v820[v62];
                            v823 = tmp17.v0; v824 = tmp17.v1;
                            static_array<Tuple2,2> & v827 = v6.v0;
                            float v828;
                            v828 = log(v818);
                            float v829;
                            v829 = v828 + v823;
                            v827[v62] = Tuple2{v829, v824};
                            static_array_list<Union0,32> & v830 = v4.v0;
                            Union0 v831;
                            v831 = Union0{Union0_1{v62, v816}};
                            v830.push(v831);
                            Union6 v832;
                            v832 = Union6{Union6_2{v59, v60, v61, v62, v63, v64, v816}};
                            float v833;
                            v833 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v832);
                            static_array_list<Union0,32> & v834 = v4.v0;
                            Union0 v835;
                            v835 = v834.pop();
                            static_array<Tuple2,2> & v836 = v6.v0;
                            v836[v62] = Tuple2{v823, v824};
                            bool v837;
                            v837 = v62 == 0;
                            if (v837){
                                v841 = v833;
                            } else {
                                float v838;
                                v838 = -v833;
                                v841 = v838;
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    float v867;
                    switch (v187.tag) {
                        case 0: { // None
                            v867 = 0.0f;
                            break;
                        }
                        case 1: { // Some
                            Union2 v842 = v187.case1.v0;
                            float v844;
                            v844 = v815[1];
                            static_array<Tuple2,2> & v846 = v6.v0;
                            float v849; float v850;
                            Tuple2 tmp18 = v846[v62];
                            v849 = tmp18.v0; v850 = tmp18.v1;
                            static_array<Tuple2,2> & v853 = v6.v0;
                            float v854;
                            v854 = log(v844);
                            float v855;
                            v855 = v854 + v849;
                            v853[v62] = Tuple2{v855, v850};
                            static_array_list<Union0,32> & v856 = v4.v0;
                            Union0 v857;
                            v857 = Union0{Union0_1{v62, v842}};
                            v856.push(v857);
                            Union6 v858;
                            v858 = Union6{Union6_2{v59, v60, v61, v62, v63, v64, v842}};
                            float v859;
                            v859 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v858);
                            static_array_list<Union0,32> & v860 = v4.v0;
                            Union0 v861;
                            v861 = v860.pop();
                            static_array<Tuple2,2> & v862 = v6.v0;
                            v862[v62] = Tuple2{v849, v850};
                            bool v863;
                            v863 = v62 == 0;
                            if (v863){
                                v867 = v859;
                            } else {
                                float v864;
                                v864 = -v859;
                                v867 = v864;
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    float v869;
                    v869 = v815[0];
                    static_array<Tuple2,2> & v871 = v6.v0;
                    float v874; float v875;
                    Tuple2 tmp19 = v871[v62];
                    v874 = tmp19.v0; v875 = tmp19.v1;
                    static_array<Tuple2,2> & v878 = v6.v0;
                    float v879;
                    v879 = log(v869);
                    float v880;
                    v880 = v879 + v874;
                    v878[v62] = Tuple2{v880, v875};
                    static_array_list<Union0,32> & v881 = v4.v0;
                    Union2 v882;
                    v882 = Union2{Union2_0{}};
                    Union0 v883;
                    v883 = Union0{Union0_1{v62, v882}};
                    v881.push(v883);
                    Union2 v884;
                    v884 = Union2{Union2_0{}};
                    Union6 v885;
                    v885 = Union6{Union6_2{v59, v60, v61, v62, v63, v64, v884}};
                    float v886;
                    v886 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v885);
                    static_array_list<Union0,32> & v887 = v4.v0;
                    Union0 v888;
                    v888 = v887.pop();
                    static_array<Tuple2,2> & v889 = v6.v0;
                    v889[v62] = Tuple2{v874, v875};
                    bool v890;
                    v890 = v62 == 0;
                    float v892;
                    if (v890){
                        v892 = v886;
                    } else {
                        float v891;
                        v891 = -v886;
                        v892 = v891;
                    }
                    static_array<float,3> v894;
                    v894[0] = v892;
                    v894[1] = v867;
                    v894[2] = v841;
                    int v896; float v897;
                    Tuple3 tmp20 = Tuple3{0, 0.0f};
                    v896 = tmp20.v0; v897 = tmp20.v1;
                    while (while_method_6(v896)){
                        float v900;
                        v900 = v894[v896];
                        float v903;
                        v903 = v815[v896];
                        float v905;
                        v905 = v900 * v903;
                        float v906;
                        v906 = v897 + v905;
                        v897 = v906;
                        v896 += 1 ;
                    }
                    v909 = v897;
                    break;
                }
                case 1: { // TrainEnumerative
                    static_array<float,3> v205;
                    v205 = regret_match_6(v175, v200);
                    float v231;
                    switch (v192.tag) {
                        case 0: { // None
                            v231 = 0.0f;
                            break;
                        }
                        case 1: { // Some
                            Union2 v206 = v192.case1.v0;
                            float v208;
                            v208 = v205[2];
                            static_array<Tuple2,2> & v210 = v6.v0;
                            float v213; float v214;
                            Tuple2 tmp21 = v210[v62];
                            v213 = tmp21.v0; v214 = tmp21.v1;
                            static_array<Tuple2,2> & v217 = v6.v0;
                            float v218;
                            v218 = log(v208);
                            float v219;
                            v219 = v218 + v213;
                            v217[v62] = Tuple2{v219, v214};
                            static_array_list<Union0,32> & v220 = v4.v0;
                            Union0 v221;
                            v221 = Union0{Union0_1{v62, v206}};
                            v220.push(v221);
                            Union6 v222;
                            v222 = Union6{Union6_2{v59, v60, v61, v62, v63, v64, v206}};
                            float v223;
                            v223 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v222);
                            static_array_list<Union0,32> & v224 = v4.v0;
                            Union0 v225;
                            v225 = v224.pop();
                            static_array<Tuple2,2> & v226 = v6.v0;
                            v226[v62] = Tuple2{v213, v214};
                            bool v227;
                            v227 = v62 == 0;
                            if (v227){
                                v231 = v223;
                            } else {
                                float v228;
                                v228 = -v223;
                                v231 = v228;
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    float v257;
                    switch (v187.tag) {
                        case 0: { // None
                            v257 = 0.0f;
                            break;
                        }
                        case 1: { // Some
                            Union2 v232 = v187.case1.v0;
                            float v234;
                            v234 = v205[1];
                            static_array<Tuple2,2> & v236 = v6.v0;
                            float v239; float v240;
                            Tuple2 tmp22 = v236[v62];
                            v239 = tmp22.v0; v240 = tmp22.v1;
                            static_array<Tuple2,2> & v243 = v6.v0;
                            float v244;
                            v244 = log(v234);
                            float v245;
                            v245 = v244 + v239;
                            v243[v62] = Tuple2{v245, v240};
                            static_array_list<Union0,32> & v246 = v4.v0;
                            Union0 v247;
                            v247 = Union0{Union0_1{v62, v232}};
                            v246.push(v247);
                            Union6 v248;
                            v248 = Union6{Union6_2{v59, v60, v61, v62, v63, v64, v232}};
                            float v249;
                            v249 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v248);
                            static_array_list<Union0,32> & v250 = v4.v0;
                            Union0 v251;
                            v251 = v250.pop();
                            static_array<Tuple2,2> & v252 = v6.v0;
                            v252[v62] = Tuple2{v239, v240};
                            bool v253;
                            v253 = v62 == 0;
                            if (v253){
                                v257 = v249;
                            } else {
                                float v254;
                                v254 = -v249;
                                v257 = v254;
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    float v259;
                    v259 = v205[0];
                    static_array<Tuple2,2> & v261 = v6.v0;
                    float v264; float v265;
                    Tuple2 tmp23 = v261[v62];
                    v264 = tmp23.v0; v265 = tmp23.v1;
                    static_array<Tuple2,2> & v268 = v6.v0;
                    float v269;
                    v269 = log(v259);
                    float v270;
                    v270 = v269 + v264;
                    v268[v62] = Tuple2{v270, v265};
                    static_array_list<Union0,32> & v271 = v4.v0;
                    Union2 v272;
                    v272 = Union2{Union2_0{}};
                    Union0 v273;
                    v273 = Union0{Union0_1{v62, v272}};
                    v271.push(v273);
                    Union2 v274;
                    v274 = Union2{Union2_0{}};
                    Union6 v275;
                    v275 = Union6{Union6_2{v59, v60, v61, v62, v63, v64, v274}};
                    float v276;
                    v276 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v275);
                    static_array_list<Union0,32> & v277 = v4.v0;
                    Union0 v278;
                    v278 = v277.pop();
                    static_array<Tuple2,2> & v279 = v6.v0;
                    v279[v62] = Tuple2{v264, v265};
                    bool v280;
                    v280 = v62 == 0;
                    float v282;
                    if (v280){
                        v282 = v276;
                    } else {
                        float v281;
                        v281 = -v276;
                        v282 = v281;
                    }
                    static_array<float,3> v284;
                    v284[0] = v282;
                    v284[1] = v257;
                    v284[2] = v231;
                    int v286; float v287;
                    Tuple3 tmp24 = Tuple3{0, 0.0f};
                    v286 = tmp24.v0; v287 = tmp24.v1;
                    while (while_method_6(v286)){
                        float v290;
                        v290 = v284[v286];
                        float v293;
                        v293 = v205[v286];
                        float v295;
                        v295 = v290 * v293;
                        float v296;
                        v296 = v287 + v295;
                        v287 = v296;
                        v286 += 1 ;
                    }
                    std::unordered_map<Tuple0, Tuple1, Fun0, Fun1> & v297 = v2.v1;
                    static_array<float,3> v299;
                    int v301;
                    v301 = 0;
                    while (while_method_6(v301)){
                        float v304;
                        v304 = v174[v301];
                        float v307;
                        v307 = v205[v301];
                        float v309;
                        v309 = 0.99609375f * v304;
                        float v310;
                        v310 = v309 + v307;
                        v299[v301] = v310;
                        v301 += 1 ;
                    }
                    static_array<Tuple2,2> & v311 = v6.v0;
                    int v312; float v313;
                    Tuple3 tmp25 = Tuple3{0, 0.0f};
                    v312 = tmp25.v0; v313 = tmp25.v1;
                    while (while_method_2(v312)){
                        float v317; float v318;
                        Tuple2 tmp26 = v311[v312];
                        v317 = tmp26.v0; v318 = tmp26.v1;
                        bool v321;
                        v321 = v312 == v62;
                        float v322;
                        if (v321){
                            v322 = 0.0f;
                        } else {
                            v322 = v317;
                        }
                        float v323;
                        v323 = v313 + v322;
                        float v324;
                        v324 = v323 - v318;
                        v313 = v324;
                        v312 += 1 ;
                    }
                    float v325;
                    v325 = exp(v313);
                    static_array<float,3> v327;
                    int v329;
                    v329 = 0;
                    while (while_method_6(v329)){
                        float v332;
                        v332 = v175[v329];
                        float v335;
                        v335 = v284[v329];
                        float v337;
                        v337 = v335 - v287;
                        float v338;
                        v338 = v325 * v337;
                        float v339;
                        v339 = v332 + v338;
                        bool v340;
                        v340 = 0.0f >= v339;
                        float v341;
                        if (v340){
                            v341 = 0.0f;
                        } else {
                            v341 = v339;
                        }
                        v327[v329] = v341;
                        v329 += 1 ;
                    }
                    int v342;
                    v342 = v90.length;
                    int v343; unsigned long long v344; unsigned long long v345;
                    Tuple7 tmp27 = Tuple7{0, 0ull, 1ull};
                    v343 = tmp27.v0; v344 = tmp27.v1; v345 = tmp27.v2;
                    while (while_method_1(v342, v343)){
                        Union0 v348;
                        v348 = v90[v343];
                        unsigned long long v393;
                        switch (v348.tag) {
                            case 0: { // CommunityCardIs
                                Union1 v350 = v348.case0.v0;
                                unsigned long long v351;
                                switch (v350.tag) {
                                    case 0: { // Jack
                                        v351 = 9223372036854765835ull;
                                        break;
                                    }
                                    case 1: { // King
                                        v351 = 18446744073709531670ull;
                                        break;
                                    }
                                    case 2: { // Queen
                                        v351 = 9223372036854745889ull;
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false);
                                        exit(-1);
                                    }
                                }
                                unsigned long long v352;
                                v352 = 9223372036854775807ull + v351;
                                unsigned long long v353;
                                v353 = v352 * 9973ull;
                                v393 = v353;
                                break;
                            }
                            case 1: { // PlayerAction
                                int v354 = v348.case1.v0; Union2 v355 = v348.case1.v1;
                                unsigned long long v356;
                                v356 = std::hash<int>()(v354);
                                unsigned long long v357;
                                v357 = v356 * 9973ull;
                                unsigned long long v358;
                                switch (v355.tag) {
                                    case 0: { // Call
                                        v358 = 9223372036854765835ull;
                                        break;
                                    }
                                    case 1: { // Fold
                                        v358 = 18446744073709531670ull;
                                        break;
                                    }
                                    case 2: { // Raise
                                        v358 = 9223372036854745889ull;
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false);
                                        exit(-1);
                                    }
                                }
                                unsigned long long v359;
                                v359 = v357 + v358;
                                unsigned long long v360;
                                v360 = 9223372036854775807ull + v359;
                                unsigned long long v361;
                                v361 = v360 * 9973ull;
                                unsigned long long v362;
                                v362 = v361 * 2ull;
                                v393 = v362;
                                break;
                            }
                            case 2: { // PlayerGotCard
                                int v363 = v348.case2.v0; Union1 v364 = v348.case2.v1;
                                unsigned long long v365;
                                v365 = std::hash<int>()(v363);
                                unsigned long long v366;
                                v366 = v365 * 9973ull;
                                unsigned long long v367;
                                switch (v364.tag) {
                                    case 0: { // Jack
                                        v367 = 9223372036854765835ull;
                                        break;
                                    }
                                    case 1: { // King
                                        v367 = 18446744073709531670ull;
                                        break;
                                    }
                                    case 2: { // Queen
                                        v367 = 9223372036854745889ull;
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false);
                                        exit(-1);
                                    }
                                }
                                unsigned long long v368;
                                v368 = v366 + v367;
                                unsigned long long v369;
                                v369 = 9223372036854775807ull + v368;
                                unsigned long long v370;
                                v370 = v369 * 9973ull;
                                unsigned long long v371;
                                v371 = v370 * 3ull;
                                v393 = v371;
                                break;
                            }
                            case 3: { // Showdown
                                static_array<Union1,2> v372 = v348.case3.v0; int v373 = v348.case3.v1; int v374 = v348.case3.v2;
                                unsigned long long v375;
                                v375 = std::hash<int>()(v374);
                                unsigned long long v376;
                                v376 = std::hash<int>()(v373);
                                unsigned long long v377;
                                v377 = v376 * 9973ull;
                                unsigned long long v378;
                                v378 = v375 + v377;
                                int v379; unsigned long long v380; unsigned long long v381;
                                Tuple7 tmp28 = Tuple7{0, 0ull, 1ull};
                                v379 = tmp28.v0; v380 = tmp28.v1; v381 = tmp28.v2;
                                while (while_method_2(v379)){
                                    Union1 v384;
                                    v384 = v372[v379];
                                    unsigned long long v386;
                                    switch (v384.tag) {
                                        case 0: { // Jack
                                            v386 = 9223372036854765835ull;
                                            break;
                                        }
                                        case 1: { // King
                                            v386 = 18446744073709531670ull;
                                            break;
                                        }
                                        case 2: { // Queen
                                            v386 = 9223372036854745889ull;
                                            break;
                                        }
                                        default: {
                                            assert("Invalid tag." && false);
                                            exit(-1);
                                        }
                                    }
                                    unsigned long long v387;
                                    v387 = v386 * v381;
                                    unsigned long long v388;
                                    v388 = v380 + v387;
                                    unsigned long long v389;
                                    v389 = v381 * 9973ull;
                                    v380 = v388;
                                    v381 = v389;
                                    v379 += 1 ;
                                }
                                unsigned long long v390;
                                v390 = 9223372036854775807ull + v378;
                                unsigned long long v391;
                                v391 = v390 * 9973ull;
                                unsigned long long v392;
                                v392 = v391 * 4ull;
                                v393 = v392;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                                exit(-1);
                            }
                        }
                        unsigned long long v394;
                        v394 = v393 * v345;
                        unsigned long long v395;
                        v395 = v344 + v394;
                        unsigned long long v396;
                        v396 = v345 * 9973ull;
                        v344 = v395;
                        v345 = v396;
                        v343 += 1 ;
                    }
                    v297[Tuple0{0ull, v90}] = Tuple1{v299, v327};
                    v909 = v287;
                    break;
                }
                case 2: { // TrainSampling
                    std::unordered_map<Tuple0, static_array<Tuple2,3>, Fun0, Fun1> & v397 = v2.v0;
                    int v398;
                    v398 = v65.length;
                    int v399; unsigned long long v400; unsigned long long v401;
                    Tuple7 tmp29 = Tuple7{0, 0ull, 1ull};
                    v399 = tmp29.v0; v400 = tmp29.v1; v401 = tmp29.v2;
                    while (while_method_1(v398, v399)){
                        Union0 v404;
                        v404 = v65[v399];
                        unsigned long long v449;
                        switch (v404.tag) {
                            case 0: { // CommunityCardIs
                                Union1 v406 = v404.case0.v0;
                                unsigned long long v407;
                                switch (v406.tag) {
                                    case 0: { // Jack
                                        v407 = 9223372036854765835ull;
                                        break;
                                    }
                                    case 1: { // King
                                        v407 = 18446744073709531670ull;
                                        break;
                                    }
                                    case 2: { // Queen
                                        v407 = 9223372036854745889ull;
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false);
                                        exit(-1);
                                    }
                                }
                                unsigned long long v408;
                                v408 = 9223372036854775807ull + v407;
                                unsigned long long v409;
                                v409 = v408 * 9973ull;
                                v449 = v409;
                                break;
                            }
                            case 1: { // PlayerAction
                                int v410 = v404.case1.v0; Union2 v411 = v404.case1.v1;
                                unsigned long long v412;
                                v412 = std::hash<int>()(v410);
                                unsigned long long v413;
                                v413 = v412 * 9973ull;
                                unsigned long long v414;
                                switch (v411.tag) {
                                    case 0: { // Call
                                        v414 = 9223372036854765835ull;
                                        break;
                                    }
                                    case 1: { // Fold
                                        v414 = 18446744073709531670ull;
                                        break;
                                    }
                                    case 2: { // Raise
                                        v414 = 9223372036854745889ull;
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false);
                                        exit(-1);
                                    }
                                }
                                unsigned long long v415;
                                v415 = v413 + v414;
                                unsigned long long v416;
                                v416 = 9223372036854775807ull + v415;
                                unsigned long long v417;
                                v417 = v416 * 9973ull;
                                unsigned long long v418;
                                v418 = v417 * 2ull;
                                v449 = v418;
                                break;
                            }
                            case 2: { // PlayerGotCard
                                int v419 = v404.case2.v0; Union1 v420 = v404.case2.v1;
                                unsigned long long v421;
                                v421 = std::hash<int>()(v419);
                                unsigned long long v422;
                                v422 = v421 * 9973ull;
                                unsigned long long v423;
                                switch (v420.tag) {
                                    case 0: { // Jack
                                        v423 = 9223372036854765835ull;
                                        break;
                                    }
                                    case 1: { // King
                                        v423 = 18446744073709531670ull;
                                        break;
                                    }
                                    case 2: { // Queen
                                        v423 = 9223372036854745889ull;
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false);
                                        exit(-1);
                                    }
                                }
                                unsigned long long v424;
                                v424 = v422 + v423;
                                unsigned long long v425;
                                v425 = 9223372036854775807ull + v424;
                                unsigned long long v426;
                                v426 = v425 * 9973ull;
                                unsigned long long v427;
                                v427 = v426 * 3ull;
                                v449 = v427;
                                break;
                            }
                            case 3: { // Showdown
                                static_array<Union1,2> v428 = v404.case3.v0; int v429 = v404.case3.v1; int v430 = v404.case3.v2;
                                unsigned long long v431;
                                v431 = std::hash<int>()(v430);
                                unsigned long long v432;
                                v432 = std::hash<int>()(v429);
                                unsigned long long v433;
                                v433 = v432 * 9973ull;
                                unsigned long long v434;
                                v434 = v431 + v433;
                                int v435; unsigned long long v436; unsigned long long v437;
                                Tuple7 tmp30 = Tuple7{0, 0ull, 1ull};
                                v435 = tmp30.v0; v436 = tmp30.v1; v437 = tmp30.v2;
                                while (while_method_2(v435)){
                                    Union1 v440;
                                    v440 = v428[v435];
                                    unsigned long long v442;
                                    switch (v440.tag) {
                                        case 0: { // Jack
                                            v442 = 9223372036854765835ull;
                                            break;
                                        }
                                        case 1: { // King
                                            v442 = 18446744073709531670ull;
                                            break;
                                        }
                                        case 2: { // Queen
                                            v442 = 9223372036854745889ull;
                                            break;
                                        }
                                        default: {
                                            assert("Invalid tag." && false);
                                            exit(-1);
                                        }
                                    }
                                    unsigned long long v443;
                                    v443 = v442 * v437;
                                    unsigned long long v444;
                                    v444 = v436 + v443;
                                    unsigned long long v445;
                                    v445 = v437 * 9973ull;
                                    v436 = v444;
                                    v437 = v445;
                                    v435 += 1 ;
                                }
                                unsigned long long v446;
                                v446 = 9223372036854775807ull + v434;
                                unsigned long long v447;
                                v447 = v446 * 9973ull;
                                unsigned long long v448;
                                v448 = v447 * 4ull;
                                v449 = v448;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                                exit(-1);
                            }
                        }
                        unsigned long long v450;
                        v450 = v449 * v401;
                        unsigned long long v451;
                        v451 = v400 + v450;
                        unsigned long long v452;
                        v452 = v401 * 9973ull;
                        v400 = v451;
                        v401 = v452;
                        v399 += 1 ;
                    }
                    auto v453 = v397.find(Tuple0{0ull, v65});
                    bool v454;
                    v454 = v453 != v397.end();
                    Union9 v458;
                    if (v454){
                        static_array<Tuple2,3> v455;
                        v455 = v453->second;
                        v458 = Union9{Union9_1{v455}};
                    } else {
                        v458 = Union9{Union9_0{}};
                    }
                    static_array<Tuple2,3> v466;
                    switch (v458.tag) {
                        case 0: { // None
                            static_array<Tuple2,3> v461;
                            int v463;
                            v463 = 0;
                            while (while_method_6(v463)){
                                v461[v463] = Tuple2{0.0f, 0.0f};
                                v463 += 1 ;
                            }
                            v466 = v461;
                            break;
                        }
                        case 1: { // Some
                            static_array<Tuple2,3> v459 = v458.case1.v0;
                            v466 = v459;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    static_array<float,3> v467;
                    v467 = regret_match_6(v175, v200);
                    static_array<float,3> v468;
                    v468 = masking_normalize_5(v174, v200);
                    static_array<float,3> v470;
                    int v472;
                    v472 = 0;
                    while (while_method_6(v472)){
                        v470[v472] = 0.0f;
                        v472 += 1 ;
                    }
                    static_array<float,3> v474;
                    v474 = masking_normalize_5(v470, v200);
                    static_array<float,3> v476;
                    int v478;
                    v478 = 0;
                    while (while_method_6(v478)){
                        float v481;
                        v481 = v468[v478];
                        float v484;
                        v484 = v474[v478];
                        float v486;
                        v486 = 0.875f * v481;
                        float v487;
                        v487 = 0.125f * v484;
                        float v488;
                        v488 = v486 + v487;
                        v476[v478] = v488;
                        v478 += 1 ;
                    }
                    int v489;
                    v489 = sample_discrete__8(v476, v3);
                    StackMut1 v490{0.0f};
                    float v493; float v494;
                    Tuple2 tmp31 = v466[2];
                    v493 = tmp31.v0; v494 = tmp31.v1;
                    bool v497;
                    v497 = v494 == 0.0f;
                    bool v498;
                    v498 = v497 != true;
                    float v500;
                    if (v498){
                        float v499;
                        v499 = v493 / v494;
                        v500 = v499;
                    } else {
                        v500 = 0.0f;
                    }
                    float v536;
                    switch (v192.tag) {
                        case 1: { // Some
                            Union2 v501 = v192.case1.v0;
                            bool v502;
                            v502 = 2 == v489;
                            if (v502){
                                float v504;
                                v504 = v467[2];
                                float v507;
                                v507 = v476[2];
                                static_array<Tuple2,2> & v509 = v6.v0;
                                float v512; float v513;
                                Tuple2 tmp32 = v509[v62];
                                v512 = tmp32.v0; v513 = tmp32.v1;
                                static_array<Tuple2,2> & v516 = v6.v0;
                                float v517;
                                v517 = log(v504);
                                float v518;
                                v518 = v517 + v512;
                                float v519;
                                v519 = log(v507);
                                float v520;
                                v520 = v519 + v513;
                                v516[v62] = Tuple2{v518, v520};
                                static_array_list<Union0,32> & v521 = v4.v0;
                                Union0 v522;
                                v522 = Union0{Union0_1{v62, v501}};
                                v521.push(v522);
                                Union6 v523;
                                v523 = Union6{Union6_2{v59, v60, v61, v62, v63, v64, v501}};
                                float v524;
                                v524 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v523);
                                static_array_list<Union0,32> & v525 = v4.v0;
                                Union0 v526;
                                v526 = v525.pop();
                                static_array<Tuple2,2> & v527 = v6.v0;
                                v527[v62] = Tuple2{v512, v513};
                                bool v528;
                                v528 = v62 == 0;
                                float v530;
                                if (v528){
                                    v530 = v524;
                                } else {
                                    float v529;
                                    v529 = -v524;
                                    v530 = v529;
                                }
                                v490.v0 = v530;
                                float v531 = v490.v0;
                                float v532;
                                v532 = v531 - v500;
                                float v533;
                                v533 = v532 / v507;
                                float v534;
                                v534 = v533 + v500;
                                v536 = v534;
                            } else {
                                v536 = v500;
                            }
                            break;
                        }
                        default: {
                            v536 = v500;
                        }
                    }
                    float v539; float v540;
                    Tuple2 tmp33 = v466[1];
                    v539 = tmp33.v0; v540 = tmp33.v1;
                    bool v543;
                    v543 = v540 == 0.0f;
                    bool v544;
                    v544 = v543 != true;
                    float v546;
                    if (v544){
                        float v545;
                        v545 = v539 / v540;
                        v546 = v545;
                    } else {
                        v546 = 0.0f;
                    }
                    float v582;
                    switch (v187.tag) {
                        case 1: { // Some
                            Union2 v547 = v187.case1.v0;
                            bool v548;
                            v548 = 1 == v489;
                            if (v548){
                                float v550;
                                v550 = v467[1];
                                float v553;
                                v553 = v476[1];
                                static_array<Tuple2,2> & v555 = v6.v0;
                                float v558; float v559;
                                Tuple2 tmp34 = v555[v62];
                                v558 = tmp34.v0; v559 = tmp34.v1;
                                static_array<Tuple2,2> & v562 = v6.v0;
                                float v563;
                                v563 = log(v550);
                                float v564;
                                v564 = v563 + v558;
                                float v565;
                                v565 = log(v553);
                                float v566;
                                v566 = v565 + v559;
                                v562[v62] = Tuple2{v564, v566};
                                static_array_list<Union0,32> & v567 = v4.v0;
                                Union0 v568;
                                v568 = Union0{Union0_1{v62, v547}};
                                v567.push(v568);
                                Union6 v569;
                                v569 = Union6{Union6_2{v59, v60, v61, v62, v63, v64, v547}};
                                float v570;
                                v570 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v569);
                                static_array_list<Union0,32> & v571 = v4.v0;
                                Union0 v572;
                                v572 = v571.pop();
                                static_array<Tuple2,2> & v573 = v6.v0;
                                v573[v62] = Tuple2{v558, v559};
                                bool v574;
                                v574 = v62 == 0;
                                float v576;
                                if (v574){
                                    v576 = v570;
                                } else {
                                    float v575;
                                    v575 = -v570;
                                    v576 = v575;
                                }
                                v490.v0 = v576;
                                float v577 = v490.v0;
                                float v578;
                                v578 = v577 - v546;
                                float v579;
                                v579 = v578 / v553;
                                float v580;
                                v580 = v579 + v546;
                                v582 = v580;
                            } else {
                                v582 = v546;
                            }
                            break;
                        }
                        default: {
                            v582 = v546;
                        }
                    }
                    float v585; float v586;
                    Tuple2 tmp35 = v466[0];
                    v585 = tmp35.v0; v586 = tmp35.v1;
                    bool v589;
                    v589 = v586 == 0.0f;
                    bool v590;
                    v590 = v589 != true;
                    float v592;
                    if (v590){
                        float v591;
                        v591 = v585 / v586;
                        v592 = v591;
                    } else {
                        v592 = 0.0f;
                    }
                    bool v593;
                    v593 = 0 == v489;
                    float v628;
                    if (v593){
                        float v595;
                        v595 = v467[0];
                        float v598;
                        v598 = v476[0];
                        static_array<Tuple2,2> & v600 = v6.v0;
                        float v603; float v604;
                        Tuple2 tmp36 = v600[v62];
                        v603 = tmp36.v0; v604 = tmp36.v1;
                        static_array<Tuple2,2> & v607 = v6.v0;
                        float v608;
                        v608 = log(v595);
                        float v609;
                        v609 = v608 + v603;
                        float v610;
                        v610 = log(v598);
                        float v611;
                        v611 = v610 + v604;
                        v607[v62] = Tuple2{v609, v611};
                        static_array_list<Union0,32> & v612 = v4.v0;
                        Union2 v613;
                        v613 = Union2{Union2_0{}};
                        Union0 v614;
                        v614 = Union0{Union0_1{v62, v613}};
                        v612.push(v614);
                        Union2 v615;
                        v615 = Union2{Union2_0{}};
                        Union6 v616;
                        v616 = Union6{Union6_2{v59, v60, v61, v62, v63, v64, v615}};
                        float v617;
                        v617 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v616);
                        static_array_list<Union0,32> & v618 = v4.v0;
                        Union0 v619;
                        v619 = v618.pop();
                        static_array<Tuple2,2> & v620 = v6.v0;
                        v620[v62] = Tuple2{v603, v604};
                        bool v621;
                        v621 = v62 == 0;
                        float v623;
                        if (v621){
                            v623 = v617;
                        } else {
                            float v622;
                            v622 = -v617;
                            v623 = v622;
                        }
                        v490.v0 = v623;
                        float v624 = v490.v0;
                        float v625;
                        v625 = v624 - v592;
                        float v626;
                        v626 = v625 / v598;
                        float v627;
                        v627 = v626 + v592;
                        v628 = v627;
                    } else {
                        v628 = v592;
                    }
                    static_array<float,3> v630;
                    v630[0] = v628;
                    v630[1] = v582;
                    v630[2] = v536;
                    int v632; float v633;
                    Tuple3 tmp37 = Tuple3{0, 0.0f};
                    v632 = tmp37.v0; v633 = tmp37.v1;
                    while (while_method_6(v632)){
                        float v636;
                        v636 = v630[v632];
                        float v639;
                        v639 = v467[v632];
                        float v641;
                        v641 = v636 * v639;
                        float v642;
                        v642 = v633 + v641;
                        v633 = v642;
                        v632 += 1 ;
                    }
                    std::unordered_map<Tuple0, Tuple1, Fun0, Fun1> & v643 = v2.v1;
                    static_array<float,3> v645;
                    int v647;
                    v647 = 0;
                    while (while_method_6(v647)){
                        float v650;
                        v650 = v174[v647];
                        float v653;
                        v653 = v467[v647];
                        float v655;
                        v655 = v650 + v653;
                        v645[v647] = v655;
                        v647 += 1 ;
                    }
                    static_array<Tuple2,2> & v656 = v6.v0;
                    int v657; float v658;
                    Tuple3 tmp38 = Tuple3{0, 0.0f};
                    v657 = tmp38.v0; v658 = tmp38.v1;
                    while (while_method_2(v657)){
                        float v662; float v663;
                        Tuple2 tmp39 = v656[v657];
                        v662 = tmp39.v0; v663 = tmp39.v1;
                        bool v666;
                        v666 = v657 == v62;
                        float v667;
                        if (v666){
                            v667 = 0.0f;
                        } else {
                            v667 = v662;
                        }
                        float v668;
                        v668 = v658 + v667;
                        float v669;
                        v669 = v668 - v663;
                        v658 = v669;
                        v657 += 1 ;
                    }
                    float v670;
                    v670 = exp(v658);
                    static_array<float,3> v672;
                    int v674;
                    v674 = 0;
                    while (while_method_6(v674)){
                        float v677;
                        v677 = v175[v674];
                        float v680;
                        v680 = v630[v674];
                        float v682;
                        v682 = v680 - v633;
                        float v683;
                        v683 = v670 * v682;
                        float v684;
                        v684 = v677 + v683;
                        bool v685;
                        v685 = 0.0f >= v684;
                        float v686;
                        if (v685){
                            v686 = 0.0f;
                        } else {
                            v686 = v684;
                        }
                        v672[v674] = v686;
                        v674 += 1 ;
                    }
                    int v687;
                    v687 = v90.length;
                    int v688; unsigned long long v689; unsigned long long v690;
                    Tuple7 tmp40 = Tuple7{0, 0ull, 1ull};
                    v688 = tmp40.v0; v689 = tmp40.v1; v690 = tmp40.v2;
                    while (while_method_1(v687, v688)){
                        Union0 v693;
                        v693 = v90[v688];
                        unsigned long long v738;
                        switch (v693.tag) {
                            case 0: { // CommunityCardIs
                                Union1 v695 = v693.case0.v0;
                                unsigned long long v696;
                                switch (v695.tag) {
                                    case 0: { // Jack
                                        v696 = 9223372036854765835ull;
                                        break;
                                    }
                                    case 1: { // King
                                        v696 = 18446744073709531670ull;
                                        break;
                                    }
                                    case 2: { // Queen
                                        v696 = 9223372036854745889ull;
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false);
                                        exit(-1);
                                    }
                                }
                                unsigned long long v697;
                                v697 = 9223372036854775807ull + v696;
                                unsigned long long v698;
                                v698 = v697 * 9973ull;
                                v738 = v698;
                                break;
                            }
                            case 1: { // PlayerAction
                                int v699 = v693.case1.v0; Union2 v700 = v693.case1.v1;
                                unsigned long long v701;
                                v701 = std::hash<int>()(v699);
                                unsigned long long v702;
                                v702 = v701 * 9973ull;
                                unsigned long long v703;
                                switch (v700.tag) {
                                    case 0: { // Call
                                        v703 = 9223372036854765835ull;
                                        break;
                                    }
                                    case 1: { // Fold
                                        v703 = 18446744073709531670ull;
                                        break;
                                    }
                                    case 2: { // Raise
                                        v703 = 9223372036854745889ull;
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false);
                                        exit(-1);
                                    }
                                }
                                unsigned long long v704;
                                v704 = v702 + v703;
                                unsigned long long v705;
                                v705 = 9223372036854775807ull + v704;
                                unsigned long long v706;
                                v706 = v705 * 9973ull;
                                unsigned long long v707;
                                v707 = v706 * 2ull;
                                v738 = v707;
                                break;
                            }
                            case 2: { // PlayerGotCard
                                int v708 = v693.case2.v0; Union1 v709 = v693.case2.v1;
                                unsigned long long v710;
                                v710 = std::hash<int>()(v708);
                                unsigned long long v711;
                                v711 = v710 * 9973ull;
                                unsigned long long v712;
                                switch (v709.tag) {
                                    case 0: { // Jack
                                        v712 = 9223372036854765835ull;
                                        break;
                                    }
                                    case 1: { // King
                                        v712 = 18446744073709531670ull;
                                        break;
                                    }
                                    case 2: { // Queen
                                        v712 = 9223372036854745889ull;
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false);
                                        exit(-1);
                                    }
                                }
                                unsigned long long v713;
                                v713 = v711 + v712;
                                unsigned long long v714;
                                v714 = 9223372036854775807ull + v713;
                                unsigned long long v715;
                                v715 = v714 * 9973ull;
                                unsigned long long v716;
                                v716 = v715 * 3ull;
                                v738 = v716;
                                break;
                            }
                            case 3: { // Showdown
                                static_array<Union1,2> v717 = v693.case3.v0; int v718 = v693.case3.v1; int v719 = v693.case3.v2;
                                unsigned long long v720;
                                v720 = std::hash<int>()(v719);
                                unsigned long long v721;
                                v721 = std::hash<int>()(v718);
                                unsigned long long v722;
                                v722 = v721 * 9973ull;
                                unsigned long long v723;
                                v723 = v720 + v722;
                                int v724; unsigned long long v725; unsigned long long v726;
                                Tuple7 tmp41 = Tuple7{0, 0ull, 1ull};
                                v724 = tmp41.v0; v725 = tmp41.v1; v726 = tmp41.v2;
                                while (while_method_2(v724)){
                                    Union1 v729;
                                    v729 = v717[v724];
                                    unsigned long long v731;
                                    switch (v729.tag) {
                                        case 0: { // Jack
                                            v731 = 9223372036854765835ull;
                                            break;
                                        }
                                        case 1: { // King
                                            v731 = 18446744073709531670ull;
                                            break;
                                        }
                                        case 2: { // Queen
                                            v731 = 9223372036854745889ull;
                                            break;
                                        }
                                        default: {
                                            assert("Invalid tag." && false);
                                            exit(-1);
                                        }
                                    }
                                    unsigned long long v732;
                                    v732 = v731 * v726;
                                    unsigned long long v733;
                                    v733 = v725 + v732;
                                    unsigned long long v734;
                                    v734 = v726 * 9973ull;
                                    v725 = v733;
                                    v726 = v734;
                                    v724 += 1 ;
                                }
                                unsigned long long v735;
                                v735 = 9223372036854775807ull + v723;
                                unsigned long long v736;
                                v736 = v735 * 9973ull;
                                unsigned long long v737;
                                v737 = v736 * 4ull;
                                v738 = v737;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                                exit(-1);
                            }
                        }
                        unsigned long long v739;
                        v739 = v738 * v690;
                        unsigned long long v740;
                        v740 = v689 + v739;
                        unsigned long long v741;
                        v741 = v690 * 9973ull;
                        v689 = v740;
                        v690 = v741;
                        v688 += 1 ;
                    }
                    v643[Tuple0{0ull, v90}] = Tuple1{v645, v672};
                    std::unordered_map<Tuple0, static_array<Tuple2,3>, Fun0, Fun1> & v742 = v2.v0;
                    static_array<Tuple2,3> v744;
                    int v746;
                    v746 = 0;
                    while (while_method_6(v746)){
                        float v750; float v751;
                        Tuple2 tmp42 = v466[v746];
                        v750 = tmp42.v0; v751 = tmp42.v1;
                        bool v754;
                        v754 = v489 == v746;
                        float v758; float v759;
                        if (v754){
                            float v755 = v490.v0;
                            float v756;
                            v756 = v750 + v755;
                            float v757;
                            v757 = v751 + 1.0f;
                            v758 = v756; v759 = v757;
                        } else {
                            v758 = v750; v759 = v751;
                        }
                        v744[v746] = Tuple2{v758, v759};
                        v746 += 1 ;
                    }
                    int v760;
                    v760 = v65.length;
                    int v761; unsigned long long v762; unsigned long long v763;
                    Tuple7 tmp43 = Tuple7{0, 0ull, 1ull};
                    v761 = tmp43.v0; v762 = tmp43.v1; v763 = tmp43.v2;
                    while (while_method_1(v760, v761)){
                        Union0 v766;
                        v766 = v65[v761];
                        unsigned long long v811;
                        switch (v766.tag) {
                            case 0: { // CommunityCardIs
                                Union1 v768 = v766.case0.v0;
                                unsigned long long v769;
                                switch (v768.tag) {
                                    case 0: { // Jack
                                        v769 = 9223372036854765835ull;
                                        break;
                                    }
                                    case 1: { // King
                                        v769 = 18446744073709531670ull;
                                        break;
                                    }
                                    case 2: { // Queen
                                        v769 = 9223372036854745889ull;
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false);
                                        exit(-1);
                                    }
                                }
                                unsigned long long v770;
                                v770 = 9223372036854775807ull + v769;
                                unsigned long long v771;
                                v771 = v770 * 9973ull;
                                v811 = v771;
                                break;
                            }
                            case 1: { // PlayerAction
                                int v772 = v766.case1.v0; Union2 v773 = v766.case1.v1;
                                unsigned long long v774;
                                v774 = std::hash<int>()(v772);
                                unsigned long long v775;
                                v775 = v774 * 9973ull;
                                unsigned long long v776;
                                switch (v773.tag) {
                                    case 0: { // Call
                                        v776 = 9223372036854765835ull;
                                        break;
                                    }
                                    case 1: { // Fold
                                        v776 = 18446744073709531670ull;
                                        break;
                                    }
                                    case 2: { // Raise
                                        v776 = 9223372036854745889ull;
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false);
                                        exit(-1);
                                    }
                                }
                                unsigned long long v777;
                                v777 = v775 + v776;
                                unsigned long long v778;
                                v778 = 9223372036854775807ull + v777;
                                unsigned long long v779;
                                v779 = v778 * 9973ull;
                                unsigned long long v780;
                                v780 = v779 * 2ull;
                                v811 = v780;
                                break;
                            }
                            case 2: { // PlayerGotCard
                                int v781 = v766.case2.v0; Union1 v782 = v766.case2.v1;
                                unsigned long long v783;
                                v783 = std::hash<int>()(v781);
                                unsigned long long v784;
                                v784 = v783 * 9973ull;
                                unsigned long long v785;
                                switch (v782.tag) {
                                    case 0: { // Jack
                                        v785 = 9223372036854765835ull;
                                        break;
                                    }
                                    case 1: { // King
                                        v785 = 18446744073709531670ull;
                                        break;
                                    }
                                    case 2: { // Queen
                                        v785 = 9223372036854745889ull;
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false);
                                        exit(-1);
                                    }
                                }
                                unsigned long long v786;
                                v786 = v784 + v785;
                                unsigned long long v787;
                                v787 = 9223372036854775807ull + v786;
                                unsigned long long v788;
                                v788 = v787 * 9973ull;
                                unsigned long long v789;
                                v789 = v788 * 3ull;
                                v811 = v789;
                                break;
                            }
                            case 3: { // Showdown
                                static_array<Union1,2> v790 = v766.case3.v0; int v791 = v766.case3.v1; int v792 = v766.case3.v2;
                                unsigned long long v793;
                                v793 = std::hash<int>()(v792);
                                unsigned long long v794;
                                v794 = std::hash<int>()(v791);
                                unsigned long long v795;
                                v795 = v794 * 9973ull;
                                unsigned long long v796;
                                v796 = v793 + v795;
                                int v797; unsigned long long v798; unsigned long long v799;
                                Tuple7 tmp44 = Tuple7{0, 0ull, 1ull};
                                v797 = tmp44.v0; v798 = tmp44.v1; v799 = tmp44.v2;
                                while (while_method_2(v797)){
                                    Union1 v802;
                                    v802 = v790[v797];
                                    unsigned long long v804;
                                    switch (v802.tag) {
                                        case 0: { // Jack
                                            v804 = 9223372036854765835ull;
                                            break;
                                        }
                                        case 1: { // King
                                            v804 = 18446744073709531670ull;
                                            break;
                                        }
                                        case 2: { // Queen
                                            v804 = 9223372036854745889ull;
                                            break;
                                        }
                                        default: {
                                            assert("Invalid tag." && false);
                                            exit(-1);
                                        }
                                    }
                                    unsigned long long v805;
                                    v805 = v804 * v799;
                                    unsigned long long v806;
                                    v806 = v798 + v805;
                                    unsigned long long v807;
                                    v807 = v799 * 9973ull;
                                    v798 = v806;
                                    v799 = v807;
                                    v797 += 1 ;
                                }
                                unsigned long long v808;
                                v808 = 9223372036854775807ull + v796;
                                unsigned long long v809;
                                v809 = v808 * 9973ull;
                                unsigned long long v810;
                                v810 = v809 * 4ull;
                                v811 = v810;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                                exit(-1);
                            }
                        }
                        unsigned long long v812;
                        v812 = v811 * v763;
                        unsigned long long v813;
                        v813 = v762 + v812;
                        unsigned long long v814;
                        v814 = v763 * 9973ull;
                        v762 = v813;
                        v763 = v814;
                        v761 += 1 ;
                    }
                    v742[Tuple0{0ull, v65}] = v744;
                    v909 = v633;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            bool v910;
            v910 = v62 == 0;
            float v912;
            if (v910){
                v912 = v909;
            } else {
                float v911;
                v911 = -v909;
                v912 = v911;
            }
            v8.v0 = v912;
            Union6 v913;
            v913 = Union6{Union6_3{}};
            return loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v913);
            break;
        }
        case 3: { // RoundWithAction
            Union5 v915 = v7.case3.v0; bool v916 = v7.case3.v1; static_array<Union1,2> v917 = v7.case3.v2; int v918 = v7.case3.v3; static_array<int,2> v919 = v7.case3.v4; int v920 = v7.case3.v5; Union2 v921 = v7.case3.v6;
            static_array_list<Union0,32> & v922 = v4.v0;
            Union0 v923;
            v923 = Union0{Union0_1{v918, v921}};
            v922.push(v923);
            Union6 v924;
            v924 = Union6{Union6_2{v915, v916, v917, v918, v919, v920, v921}};
            float v925;
            v925 = loop_4(v0, v1, v2, v3, v4, v5, v6, v8, v924);
            static_array_list<Union0,32> & v926 = v4.v0;
            Union0 v927;
            v927 = v926.pop();
            return v925;
            break;
        }
        case 4: { // TerminalCall
            Union5 v30 = v7.case4.v0; bool v31 = v7.case4.v1; static_array<Union1,2> v32 = v7.case4.v2; int v33 = v7.case4.v3; static_array<int,2> v34 = v7.case4.v4; int v35 = v7.case4.v5;
            int v37;
            v37 = v34[v33];
            Union10 v39;
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
        std::unordered_map<Tuple0, Tuple1, Fun0, Fun1> v4(512, v2, v3);
        std::unordered_map<Tuple0, static_array<Tuple2,3>, Fun0, Fun1> v5(512, v2, v3);
        StackRefs0 v6{v5, v4};
        static_array<Union3,2> v8;
        Union3 v11;
        v11 = Union3{Union3_2{}};
        v8[0] = v11;
        Union3 v14;
        v14 = Union3{Union3_2{}};
        v8[1] = v14;
        bool v16;
        v16 = true;
        xso::rng v17;
        StackMut0 v18{63u};
        static_array_list<Union0,32> v20;
        v20 = static_array_list<Union0,32>{};
        StackRefs1 v22{v20};
        static_array<Tuple2,2> v24;
        int v26;
        v26 = 0;
        while (while_method_2(v26)){
            v24[v26] = Tuple2{0.0f, 0.0f};
            v26 += 1 ;
        }
        StackRefs2 v28{v24};
        int v29; float v30;
        Tuple3 tmp0 = Tuple3{0, 0.0f};
        v29 = tmp0.v0; v30 = tmp0.v1;
        while (while_method_3(v29)){
            Union4 v32;
            v32 = Union4{Union4_1{}};
            float v33;
            v33 = body_0(v16, v8, v6, v17, v22, v18, v28, v32);
            v30 = v33;
            v29 += 1 ;
        }
        static_array<Union3,2> v35;
        Union3 v38;
        v38 = Union3{Union3_1{}};
        v35[0] = v38;
        Union3 v41;
        v41 = Union3{Union3_0{}};
        v35[1] = v41;
        bool v43;
        v43 = false;
        xso::rng v44;
        StackMut0 v45{63u};
        static_array_list<Union0,32> v47;
        v47 = static_array_list<Union0,32>{};
        StackRefs1 v49{v47};
        static_array<Tuple2,2> v51;
        int v53;
        v53 = 0;
        while (while_method_2(v53)){
            v51[v53] = Tuple2{0.0f, 0.0f};
            v53 += 1 ;
        }
        StackRefs2 v55{v51};
        int v56; float v57;
        Tuple3 tmp47 = Tuple3{0, 0.0f};
        v56 = tmp47.v0; v57 = tmp47.v1;
        while (while_method_9(v56)){
            Union4 v59;
            v59 = Union4{Union4_1{}};
            float v60;
            v60 = body_0(v43, v35, v6, v44, v49, v45, v55, v59);
            v57 = v60;
            v56 += 1 ;
        }
        printf("{%s = %f}\n","reward_for_pl0", v57);
        fflush(stdout);
        v0 += 1 ;
    }
    return 0;
}
