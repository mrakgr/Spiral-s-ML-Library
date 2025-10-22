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
struct Union6;
struct Tuple5;
float loop_1(static_array<Union3,2> v0, StackRefs0 & v1, xso::rng & v2, StackRefs1 & v3, StackMut0 & v4, StackRefs2 & v5, StackMut1 & v6, Union6 v7);
struct Union7;
struct Union8;
static_array<float,3> masking_normalize_2(static_array<float,3> v0, static_array<bool,3> v1);
static_array<float,3> relu_4(static_array<float,3> v0);
static_array<float,3> regret_match_3(static_array<float,3> v0, static_array<bool,3> v1);
int loop_7(static_array<float,3> v0, float v1, int v2);
int pick_discrete__6(static_array<float,3> v0, float v1);
int sample_discrete__5(static_array<float,3> v0, xso::rng & v1);
struct Union9;
int tag_9(Union1 v0);
bool is_pair_10(int v0, int v1);
Tuple5 order_11(int v0, int v1);
Union9 compare_hands_8(Union5 v0, bool v1, static_array<Union1,2> v2, int v3, static_array<int,2> v4, int v5);
float body_0(static_array<Union3,2> v0, StackRefs0 & v1, xso::rng & v2, StackRefs1 & v3, StackMut0 & v4, StackRefs2 & v5, Union4 v6);
static_array<float,3> normalize_12(static_array<float,3> v0);
void method_14(Union1 v0);
void method_15(Union2 v0);
void method_13(Union0 v0);
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
    int v0;
    float v1;
    float v2;
    __host__ __device__ Tuple4() = default;
    __host__ __device__ Tuple4(int t0, float t1, float t2) : v0(t0), v1(t1), v2(t2) {}
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
    v1 = v0 < 100000;
    return v1;
}
inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 6;
    return v1;
}
float loop_1(static_array<Union3,2> v0, StackRefs0 & v1, xso::rng & v2, StackRefs1 & v3, StackMut0 & v4, StackRefs2 & v5, StackMut1 & v6, Union6 v7){
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
static_array<float,3> masking_normalize_2(static_array<float,3> v0, static_array<bool,3> v1){
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
static_array<float,3> relu_4(static_array<float,3> v0){
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
static_array<float,3> regret_match_3(static_array<float,3> v0, static_array<bool,3> v1){
    static_array<float,3> v2;
    v2 = relu_4(v0);
    return masking_normalize_2(v2, v1);
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
int loop_7(static_array<float,3> v0, float v1, int v2){
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
            return loop_7(v0, v1, v8);
        }
    } else {
        return 2;
    }
}
int pick_discrete__6(static_array<float,3> v0, float v1){
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
    return loop_7(v3, v26, v27);
}
int sample_discrete__5(static_array<float,3> v0, xso::rng & v1){
    std::uniform_real_distribution<float> v2(0.0, 1.0);
    float v3;
    v3 = v2(v1);
    return pick_discrete__6(v0, v3);
}
int tag_9(Union1 v0){
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
bool is_pair_10(int v0, int v1){
    bool v2;
    v2 = v1 == v0;
    return v2;
}
Tuple5 order_11(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple5{v1, v0};
    } else {
        return Tuple5{v0, v1};
    }
}
Union9 compare_hands_8(Union5 v0, bool v1, static_array<Union1,2> v2, int v3, static_array<int,2> v4, int v5){
    switch (v0.tag) {
        case 0: { // None
            printf("%s\n", "Expected the community card to be present in the table.");
            exit(-1);
            break;
        }
        case 1: { // Some
            Union1 v7 = v0.case1.v0;
            int v8;
            v8 = tag_9(v7);
            Union1 v10;
            v10 = v2[0];
            int v12;
            v12 = tag_9(v10);
            Union1 v14;
            v14 = v2[1];
            int v16;
            v16 = tag_9(v14);
            bool v17;
            v17 = is_pair_10(v8, v12);
            bool v18;
            v18 = is_pair_10(v8, v16);
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
                    Tuple5 tmp39 = order_11(v8, v12);
                    v29 = tmp39.v0; v30 = tmp39.v1;
                    int v31; int v32;
                    Tuple5 tmp40 = order_11(v8, v16);
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
            Union5 v718 = v6.case0.v0; bool v719 = v6.case0.v1; static_array<Union1,2> v720 = v6.case0.v2; int v721 = v6.case0.v3; static_array<int,2> v722 = v6.case0.v4; int v723 = v6.case0.v5;
            int v724; float v725; float v726;
            Tuple4 tmp3 = Tuple4{0, 0.0f, 0.0f};
            v724 = tmp3.v0; v725 = tmp3.v1; v726 = tmp3.v2;
            while (while_method_3(v724)){
                unsigned int v728 = v4.v0;
                unsigned int v729;
                v729 = 1u << v724;
                unsigned int v730;
                v730 = v728 & v729;
                bool v731;
                v731 = v730 == 0u;
                bool v732;
                v732 = v731 != true;
                float v764; float v765;
                if (v732){
                    unsigned int v733 = v4.v0;
                    unsigned int v734;
                    v734 = v733 ^ v729;
                    v4.v0 = v734;
                    bool v735;
                    v735 = 0 == v724;
                    Union1 v753;
                    if (v735){
                        v753 = Union1{Union1_1{}};
                    } else {
                        bool v737;
                        v737 = 1 == v724;
                        if (v737){
                            v753 = Union1{Union1_1{}};
                        } else {
                            bool v739;
                            v739 = 2 == v724;
                            if (v739){
                                v753 = Union1{Union1_2{}};
                            } else {
                                bool v741;
                                v741 = 3 == v724;
                                if (v741){
                                    v753 = Union1{Union1_2{}};
                                } else {
                                    bool v743;
                                    v743 = 4 == v724;
                                    if (v743){
                                        v753 = Union1{Union1_0{}};
                                    } else {
                                        bool v745;
                                        v745 = 5 == v724;
                                        if (v745){
                                            v753 = Union1{Union1_0{}};
                                        } else {
                                            printf("%s\n", "Invalid int in int_to_card.");
                                            exit(-1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    static_array_list<Union0,32> & v754 = v3.v0;
                    Union0 v755;
                    v755 = Union0{Union0_0{v753}};
                    v754.push(v755);
                    Union6 v756;
                    v756 = Union6{Union6_0{v718, v719, v720, v721, v722, v723, v753}};
                    float v757;
                    v757 = loop_1(v0, v1, v2, v3, v4, v5, v7, v756);
                    static_array_list<Union0,32> & v758 = v3.v0;
                    Union0 v759;
                    v759 = v758.pop();
                    unsigned int v760 = v4.v0;
                    unsigned int v761;
                    v761 = v760 ^ v729;
                    v4.v0 = v761;
                    float v762;
                    v762 = v725 + v757;
                    float v763;
                    v763 = v726 + 1.0f;
                    v764 = v762; v765 = v763;
                } else {
                    v764 = v725; v765 = v726;
                }
                v725 = v764;
                v726 = v765;
                v724 += 1 ;
            }
            bool v766;
            v766 = v726 == 0.0f;
            bool v767;
            v767 = v766 != true;
            if (v767){
                float v768;
                v768 = v725 / v726;
                return v768;
            } else {
                return 0.0f;
            }
            break;
        }
        case 1: { // ChanceInit
            int v770; float v771; float v772;
            Tuple4 tmp8 = Tuple4{0, 0.0f, 0.0f};
            v770 = tmp8.v0; v771 = tmp8.v1; v772 = tmp8.v2;
            while (while_method_3(v770)){
                unsigned int v774 = v4.v0;
                unsigned int v775;
                v775 = 1u << v770;
                unsigned int v776;
                v776 = v774 & v775;
                bool v777;
                v777 = v776 == 0u;
                bool v778;
                v778 = v777 != true;
                float v854; float v855;
                if (v778){
                    unsigned int v779 = v4.v0;
                    unsigned int v780;
                    v780 = v779 ^ v775;
                    v4.v0 = v780;
                    bool v781;
                    v781 = 0 == v770;
                    Union1 v799;
                    if (v781){
                        v799 = Union1{Union1_1{}};
                    } else {
                        bool v783;
                        v783 = 1 == v770;
                        if (v783){
                            v799 = Union1{Union1_1{}};
                        } else {
                            bool v785;
                            v785 = 2 == v770;
                            if (v785){
                                v799 = Union1{Union1_2{}};
                            } else {
                                bool v787;
                                v787 = 3 == v770;
                                if (v787){
                                    v799 = Union1{Union1_2{}};
                                } else {
                                    bool v789;
                                    v789 = 4 == v770;
                                    if (v789){
                                        v799 = Union1{Union1_0{}};
                                    } else {
                                        bool v791;
                                        v791 = 5 == v770;
                                        if (v791){
                                            v799 = Union1{Union1_0{}};
                                        } else {
                                            printf("%s\n", "Invalid int in int_to_card.");
                                            exit(-1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    int v800; float v801; float v802;
                    Tuple4 tmp9 = Tuple4{0, 0.0f, 0.0f};
                    v800 = tmp9.v0; v801 = tmp9.v1; v802 = tmp9.v2;
                    while (while_method_3(v800)){
                        unsigned int v804 = v4.v0;
                        unsigned int v805;
                        v805 = 1u << v800;
                        unsigned int v806;
                        v806 = v804 & v805;
                        bool v807;
                        v807 = v806 == 0u;
                        bool v808;
                        v808 = v807 != true;
                        float v844; float v845;
                        if (v808){
                            unsigned int v809 = v4.v0;
                            unsigned int v810;
                            v810 = v809 ^ v805;
                            v4.v0 = v810;
                            bool v811;
                            v811 = 0 == v800;
                            Union1 v829;
                            if (v811){
                                v829 = Union1{Union1_1{}};
                            } else {
                                bool v813;
                                v813 = 1 == v800;
                                if (v813){
                                    v829 = Union1{Union1_1{}};
                                } else {
                                    bool v815;
                                    v815 = 2 == v800;
                                    if (v815){
                                        v829 = Union1{Union1_2{}};
                                    } else {
                                        bool v817;
                                        v817 = 3 == v800;
                                        if (v817){
                                            v829 = Union1{Union1_2{}};
                                        } else {
                                            bool v819;
                                            v819 = 4 == v800;
                                            if (v819){
                                                v829 = Union1{Union1_0{}};
                                            } else {
                                                bool v821;
                                                v821 = 5 == v800;
                                                if (v821){
                                                    v829 = Union1{Union1_0{}};
                                                } else {
                                                    printf("%s\n", "Invalid int in int_to_card.");
                                                    exit(-1);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            static_array_list<Union0,32> & v830 = v3.v0;
                            Union0 v831;
                            v831 = Union0{Union0_2{0, v799}};
                            v830.push(v831);
                            static_array_list<Union0,32> & v832 = v3.v0;
                            Union0 v833;
                            v833 = Union0{Union0_2{1, v829}};
                            v832.push(v833);
                            Union6 v834;
                            v834 = Union6{Union6_1{v799, v829}};
                            float v835;
                            v835 = loop_1(v0, v1, v2, v3, v4, v5, v7, v834);
                            static_array_list<Union0,32> & v836 = v3.v0;
                            Union0 v837;
                            v837 = v836.pop();
                            static_array_list<Union0,32> & v838 = v3.v0;
                            Union0 v839;
                            v839 = v838.pop();
                            unsigned int v840 = v4.v0;
                            unsigned int v841;
                            v841 = v840 ^ v805;
                            v4.v0 = v841;
                            float v842;
                            v842 = v801 + v835;
                            float v843;
                            v843 = v802 + 1.0f;
                            v844 = v842; v845 = v843;
                        } else {
                            v844 = v801; v845 = v802;
                        }
                        v801 = v844;
                        v802 = v845;
                        v800 += 1 ;
                    }
                    bool v846;
                    v846 = v802 == 0.0f;
                    bool v847;
                    v847 = v846 != true;
                    float v849;
                    if (v847){
                        float v848;
                        v848 = v801 / v802;
                        v849 = v848;
                    } else {
                        v849 = 0.0f;
                    }
                    unsigned int v850 = v4.v0;
                    unsigned int v851;
                    v851 = v850 ^ v775;
                    v4.v0 = v851;
                    float v852;
                    v852 = v771 + v849;
                    float v853;
                    v853 = v772 + 1.0f;
                    v854 = v852; v855 = v853;
                } else {
                    v854 = v771; v855 = v772;
                }
                v771 = v854;
                v772 = v855;
                v770 += 1 ;
            }
            bool v856;
            v856 = v772 == 0.0f;
            bool v857;
            v857 = v856 != true;
            if (v857){
                float v858;
                v858 = v771 / v772;
                return v858;
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
            float v699;
            switch (v155.tag) {
                case 0: { // Frozen
                    static_array<float,3> v597;
                    v597 = masking_normalize_2(v125, v152);
                    float v623;
                    switch (v144.tag) {
                        case 0: { // None
                            v623 = 0.0f;
                            break;
                        }
                        case 1: { // Some
                            Union2 v598 = v144.case1.v0;
                            float v600;
                            v600 = v597[2];
                            static_array<Tuple2,2> & v602 = v5.v0;
                            float v605; float v606;
                            Tuple2 tmp14 = v602[v61];
                            v605 = tmp14.v0; v606 = tmp14.v1;
                            static_array<Tuple2,2> & v609 = v5.v0;
                            float v610;
                            v610 = log(v600);
                            float v611;
                            v611 = v610 + v605;
                            v609[v61] = Tuple2{v611, v606};
                            static_array_list<Union0,32> & v612 = v3.v0;
                            Union0 v613;
                            v613 = Union0{Union0_1{v61, v598}};
                            v612.push(v613);
                            Union6 v614;
                            v614 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v598}};
                            float v615;
                            v615 = loop_1(v0, v1, v2, v3, v4, v5, v7, v614);
                            static_array_list<Union0,32> & v616 = v3.v0;
                            Union0 v617;
                            v617 = v616.pop();
                            static_array<Tuple2,2> & v618 = v5.v0;
                            v618[v61] = Tuple2{v605, v606};
                            bool v619;
                            v619 = v61 == 0;
                            if (v619){
                                v623 = v615;
                            } else {
                                float v620;
                                v620 = -v615;
                                v623 = v620;
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    float v649;
                    switch (v139.tag) {
                        case 0: { // None
                            v649 = 0.0f;
                            break;
                        }
                        case 1: { // Some
                            Union2 v624 = v139.case1.v0;
                            float v626;
                            v626 = v597[1];
                            static_array<Tuple2,2> & v628 = v5.v0;
                            float v631; float v632;
                            Tuple2 tmp15 = v628[v61];
                            v631 = tmp15.v0; v632 = tmp15.v1;
                            static_array<Tuple2,2> & v635 = v5.v0;
                            float v636;
                            v636 = log(v626);
                            float v637;
                            v637 = v636 + v631;
                            v635[v61] = Tuple2{v637, v632};
                            static_array_list<Union0,32> & v638 = v3.v0;
                            Union0 v639;
                            v639 = Union0{Union0_1{v61, v624}};
                            v638.push(v639);
                            Union6 v640;
                            v640 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v624}};
                            float v641;
                            v641 = loop_1(v0, v1, v2, v3, v4, v5, v7, v640);
                            static_array_list<Union0,32> & v642 = v3.v0;
                            Union0 v643;
                            v643 = v642.pop();
                            static_array<Tuple2,2> & v644 = v5.v0;
                            v644[v61] = Tuple2{v631, v632};
                            bool v645;
                            v645 = v61 == 0;
                            if (v645){
                                v649 = v641;
                            } else {
                                float v646;
                                v646 = -v641;
                                v649 = v646;
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                            exit(-1);
                        }
                    }
                    float v651;
                    v651 = v597[0];
                    static_array<Tuple2,2> & v653 = v5.v0;
                    float v656; float v657;
                    Tuple2 tmp16 = v653[v61];
                    v656 = tmp16.v0; v657 = tmp16.v1;
                    static_array<Tuple2,2> & v660 = v5.v0;
                    float v661;
                    v661 = log(v651);
                    float v662;
                    v662 = v661 + v656;
                    v660[v61] = Tuple2{v662, v657};
                    static_array_list<Union0,32> & v663 = v3.v0;
                    Union2 v664;
                    v664 = Union2{Union2_0{}};
                    Union0 v665;
                    v665 = Union0{Union0_1{v61, v664}};
                    v663.push(v665);
                    Union2 v666;
                    v666 = Union2{Union2_0{}};
                    Union6 v667;
                    v667 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v666}};
                    float v668;
                    v668 = loop_1(v0, v1, v2, v3, v4, v5, v7, v667);
                    static_array_list<Union0,32> & v669 = v3.v0;
                    Union0 v670;
                    v670 = v669.pop();
                    static_array<Tuple2,2> & v671 = v5.v0;
                    v671[v61] = Tuple2{v656, v657};
                    bool v672;
                    v672 = v61 == 0;
                    float v674;
                    if (v672){
                        v674 = v668;
                    } else {
                        float v673;
                        v673 = -v668;
                        v674 = v673;
                    }
                    static_array<float,3> v676;
                    v676[0] = v674;
                    v676[1] = v649;
                    v676[2] = v623;
                    static_array<float,3> v679;
                    int v681;
                    v681 = 0;
                    while (while_method_4(v681)){
                        float v684;
                        v684 = v676[v681];
                        float v687;
                        v687 = v597[v681];
                        float v689;
                        v689 = v684 * v687;
                        v679[v681] = v689;
                        v681 += 1 ;
                    }
                    int v690; float v691;
                    Tuple3 tmp17 = Tuple3{0, 0.0f};
                    v690 = tmp17.v0; v691 = tmp17.v1;
                    while (while_method_4(v690)){
                        float v694;
                        v694 = v679[v690];
                        float v696;
                        v696 = v691 + v694;
                        v691 = v696;
                        v690 += 1 ;
                    }
                    v699 = v691;
                    break;
                }
                case 1: { // TrainEnumerative
                    static_array<float,3> v157;
                    v157 = regret_match_3(v126, v152);
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
                            v175 = loop_1(v0, v1, v2, v3, v4, v5, v7, v174);
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
                            v201 = loop_1(v0, v1, v2, v3, v4, v5, v7, v200);
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
                    v228 = loop_1(v0, v1, v2, v3, v4, v5, v7, v227);
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
                    v699 = v251;
                    break;
                }
                case 2: { // TrainSampling
                    static_array<float,3> v302;
                    v302 = regret_match_3(v126, v152);
                    static_array<float,3> v303;
                    v303 = masking_normalize_2(v125, v152);
                    static_array<float,3> v305;
                    int v307;
                    v307 = 0;
                    while (while_method_4(v307)){
                        v305[v307] = 0.0f;
                        v307 += 1 ;
                    }
                    static_array<float,3> v309;
                    v309 = masking_normalize_2(v305, v152);
                    static_array<float,3> v311;
                    int v313;
                    v313 = 0;
                    while (while_method_4(v313)){
                        float v316;
                        v316 = v303[v313];
                        float v319;
                        v319 = v309[v313];
                        float v321;
                        v321 = 0.875f * v316;
                        float v322;
                        v322 = 0.125f * v319;
                        float v323;
                        v323 = v321 + v322;
                        v311[v313] = v323;
                        v313 += 1 ;
                    }
                    int v324;
                    v324 = sample_discrete__5(v311, v2);
                    StackMut1 v325{0.0f};
                    float v391;
                    switch (v144.tag) {
                        case 1: { // Some
                            Union2 v326 = v144.case1.v0;
                            bool v327;
                            v327 = 2 == v324;
                            if (v327){
                                float v329;
                                v329 = v302[2];
                                float v332;
                                v332 = v311[2];
                                static_array<Tuple2,2> & v334 = v5.v0;
                                float v337; float v338;
                                Tuple2 tmp24 = v334[v61];
                                v337 = tmp24.v0; v338 = tmp24.v1;
                                static_array<Tuple2,2> & v341 = v5.v0;
                                float v342;
                                v342 = log(v329);
                                float v343;
                                v343 = v342 + v337;
                                float v344;
                                v344 = log(v332);
                                float v345;
                                v345 = v344 + v338;
                                v341[v61] = Tuple2{v343, v345};
                                static_array_list<Union0,32> & v346 = v3.v0;
                                Union0 v347;
                                v347 = Union0{Union0_1{v61, v326}};
                                v346.push(v347);
                                Union6 v348;
                                v348 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v326}};
                                float v349;
                                v349 = loop_1(v0, v1, v2, v3, v4, v5, v7, v348);
                                static_array_list<Union0,32> & v350 = v3.v0;
                                Union0 v351;
                                v351 = v350.pop();
                                static_array<Tuple2,2> & v352 = v5.v0;
                                v352[v61] = Tuple2{v337, v338};
                                bool v353;
                                v353 = v61 == 0;
                                float v355;
                                if (v353){
                                    v355 = v349;
                                } else {
                                    float v354;
                                    v354 = -v349;
                                    v355 = v354;
                                }
                                v325.v0 = v355;
                                float v358; float v359;
                                Tuple2 tmp25 = v127[2];
                                v358 = tmp25.v0; v359 = tmp25.v1;
                                bool v362;
                                v362 = v359 == 0.0f;
                                bool v363;
                                v363 = v362 != true;
                                float v365;
                                if (v363){
                                    float v364;
                                    v364 = v358 / v359;
                                    v365 = v364;
                                } else {
                                    v365 = 0.0f;
                                }
                                float v366 = v325.v0;
                                float v367;
                                v367 = v366 - v365;
                                float v368;
                                v368 = v367 / v332;
                                float v369;
                                v369 = v368 + v365;
                                v391 = v369;
                            } else {
                                float v372; float v373;
                                Tuple2 tmp26 = v127[2];
                                v372 = tmp26.v0; v373 = tmp26.v1;
                                bool v376;
                                v376 = v373 == 0.0f;
                                bool v377;
                                v377 = v376 != true;
                                if (v377){
                                    float v378;
                                    v378 = v372 / v373;
                                    v391 = v378;
                                } else {
                                    v391 = 0.0f;
                                }
                            }
                            break;
                        }
                        default: {
                            float v383; float v384;
                            Tuple2 tmp27 = v127[2];
                            v383 = tmp27.v0; v384 = tmp27.v1;
                            bool v387;
                            v387 = v384 == 0.0f;
                            bool v388;
                            v388 = v387 != true;
                            if (v388){
                                float v389;
                                v389 = v383 / v384;
                                v391 = v389;
                            } else {
                                v391 = 0.0f;
                            }
                        }
                    }
                    float v457;
                    switch (v139.tag) {
                        case 1: { // Some
                            Union2 v392 = v139.case1.v0;
                            bool v393;
                            v393 = 1 == v324;
                            if (v393){
                                float v395;
                                v395 = v302[1];
                                float v398;
                                v398 = v311[1];
                                static_array<Tuple2,2> & v400 = v5.v0;
                                float v403; float v404;
                                Tuple2 tmp28 = v400[v61];
                                v403 = tmp28.v0; v404 = tmp28.v1;
                                static_array<Tuple2,2> & v407 = v5.v0;
                                float v408;
                                v408 = log(v395);
                                float v409;
                                v409 = v408 + v403;
                                float v410;
                                v410 = log(v398);
                                float v411;
                                v411 = v410 + v404;
                                v407[v61] = Tuple2{v409, v411};
                                static_array_list<Union0,32> & v412 = v3.v0;
                                Union0 v413;
                                v413 = Union0{Union0_1{v61, v392}};
                                v412.push(v413);
                                Union6 v414;
                                v414 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v392}};
                                float v415;
                                v415 = loop_1(v0, v1, v2, v3, v4, v5, v7, v414);
                                static_array_list<Union0,32> & v416 = v3.v0;
                                Union0 v417;
                                v417 = v416.pop();
                                static_array<Tuple2,2> & v418 = v5.v0;
                                v418[v61] = Tuple2{v403, v404};
                                bool v419;
                                v419 = v61 == 0;
                                float v421;
                                if (v419){
                                    v421 = v415;
                                } else {
                                    float v420;
                                    v420 = -v415;
                                    v421 = v420;
                                }
                                v325.v0 = v421;
                                float v424; float v425;
                                Tuple2 tmp29 = v127[1];
                                v424 = tmp29.v0; v425 = tmp29.v1;
                                bool v428;
                                v428 = v425 == 0.0f;
                                bool v429;
                                v429 = v428 != true;
                                float v431;
                                if (v429){
                                    float v430;
                                    v430 = v424 / v425;
                                    v431 = v430;
                                } else {
                                    v431 = 0.0f;
                                }
                                float v432 = v325.v0;
                                float v433;
                                v433 = v432 - v431;
                                float v434;
                                v434 = v433 / v398;
                                float v435;
                                v435 = v434 + v431;
                                v457 = v435;
                            } else {
                                float v438; float v439;
                                Tuple2 tmp30 = v127[1];
                                v438 = tmp30.v0; v439 = tmp30.v1;
                                bool v442;
                                v442 = v439 == 0.0f;
                                bool v443;
                                v443 = v442 != true;
                                if (v443){
                                    float v444;
                                    v444 = v438 / v439;
                                    v457 = v444;
                                } else {
                                    v457 = 0.0f;
                                }
                            }
                            break;
                        }
                        default: {
                            float v449; float v450;
                            Tuple2 tmp31 = v127[1];
                            v449 = tmp31.v0; v450 = tmp31.v1;
                            bool v453;
                            v453 = v450 == 0.0f;
                            bool v454;
                            v454 = v453 != true;
                            if (v454){
                                float v455;
                                v455 = v449 / v450;
                                v457 = v455;
                            } else {
                                v457 = 0.0f;
                            }
                        }
                    }
                    bool v458;
                    v458 = 0 == v324;
                    float v513;
                    if (v458){
                        float v460;
                        v460 = v302[0];
                        float v463;
                        v463 = v311[0];
                        static_array<Tuple2,2> & v465 = v5.v0;
                        float v468; float v469;
                        Tuple2 tmp32 = v465[v61];
                        v468 = tmp32.v0; v469 = tmp32.v1;
                        static_array<Tuple2,2> & v472 = v5.v0;
                        float v473;
                        v473 = log(v460);
                        float v474;
                        v474 = v473 + v468;
                        float v475;
                        v475 = log(v463);
                        float v476;
                        v476 = v475 + v469;
                        v472[v61] = Tuple2{v474, v476};
                        static_array_list<Union0,32> & v477 = v3.v0;
                        Union2 v478;
                        v478 = Union2{Union2_0{}};
                        Union0 v479;
                        v479 = Union0{Union0_1{v61, v478}};
                        v477.push(v479);
                        Union2 v480;
                        v480 = Union2{Union2_0{}};
                        Union6 v481;
                        v481 = Union6{Union6_2{v58, v59, v60, v61, v62, v63, v480}};
                        float v482;
                        v482 = loop_1(v0, v1, v2, v3, v4, v5, v7, v481);
                        static_array_list<Union0,32> & v483 = v3.v0;
                        Union0 v484;
                        v484 = v483.pop();
                        static_array<Tuple2,2> & v485 = v5.v0;
                        v485[v61] = Tuple2{v468, v469};
                        bool v486;
                        v486 = v61 == 0;
                        float v488;
                        if (v486){
                            v488 = v482;
                        } else {
                            float v487;
                            v487 = -v482;
                            v488 = v487;
                        }
                        v325.v0 = v488;
                        float v491; float v492;
                        Tuple2 tmp33 = v127[0];
                        v491 = tmp33.v0; v492 = tmp33.v1;
                        bool v495;
                        v495 = v492 == 0.0f;
                        bool v496;
                        v496 = v495 != true;
                        float v498;
                        if (v496){
                            float v497;
                            v497 = v491 / v492;
                            v498 = v497;
                        } else {
                            v498 = 0.0f;
                        }
                        float v499 = v325.v0;
                        float v500;
                        v500 = v499 - v498;
                        float v501;
                        v501 = v500 / v463;
                        float v502;
                        v502 = v501 + v498;
                        v513 = v502;
                    } else {
                        float v505; float v506;
                        Tuple2 tmp34 = v127[0];
                        v505 = tmp34.v0; v506 = tmp34.v1;
                        bool v509;
                        v509 = v506 == 0.0f;
                        bool v510;
                        v510 = v509 != true;
                        if (v510){
                            float v511;
                            v511 = v505 / v506;
                            v513 = v511;
                        } else {
                            v513 = 0.0f;
                        }
                    }
                    static_array<float,3> v515;
                    v515[0] = v513;
                    v515[1] = v457;
                    v515[2] = v391;
                    static_array<float,3> v518;
                    int v520;
                    v520 = 0;
                    while (while_method_4(v520)){
                        float v523;
                        v523 = v515[v520];
                        float v526;
                        v526 = v302[v520];
                        float v528;
                        v528 = v523 * v526;
                        v518[v520] = v528;
                        v520 += 1 ;
                    }
                    int v529; float v530;
                    Tuple3 tmp35 = Tuple3{0, 0.0f};
                    v529 = tmp35.v0; v530 = tmp35.v1;
                    while (while_method_4(v529)){
                        float v533;
                        v533 = v518[v529];
                        float v535;
                        v535 = v530 + v533;
                        v530 = v535;
                        v529 += 1 ;
                    }
                    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v536 = v1.v0;
                    static_array<float,3> v538;
                    int v540;
                    v540 = 0;
                    while (while_method_4(v540)){
                        float v543;
                        v543 = v125[v540];
                        float v546;
                        v546 = v302[v540];
                        float v548;
                        v548 = v543 + v546;
                        v538[v540] = v548;
                        v540 += 1 ;
                    }
                    static_array<Tuple2,2> & v549 = v5.v0;
                    int v550; float v551;
                    Tuple3 tmp36 = Tuple3{0, 0.0f};
                    v550 = tmp36.v0; v551 = tmp36.v1;
                    while (while_method_1(v550)){
                        float v555; float v556;
                        Tuple2 tmp37 = v549[v550];
                        v555 = tmp37.v0; v556 = tmp37.v1;
                        bool v559;
                        v559 = v550 == v61;
                        float v560;
                        if (v559){
                            v560 = 0.0f;
                        } else {
                            v560 = v555;
                        }
                        float v561;
                        v561 = v551 + v560;
                        float v562;
                        v562 = v561 - v556;
                        v551 = v562;
                        v550 += 1 ;
                    }
                    float v563;
                    v563 = exp(v551);
                    static_array<float,3> v565;
                    int v567;
                    v567 = 0;
                    while (while_method_4(v567)){
                        float v570;
                        v570 = v126[v567];
                        float v573;
                        v573 = v515[v567];
                        float v575;
                        v575 = v573 - v530;
                        float v576;
                        v576 = v563 * v575;
                        float v577;
                        v577 = v570 + v576;
                        bool v578;
                        v578 = 0.0f >= v577;
                        float v579;
                        if (v578){
                            v579 = 0.0f;
                        } else {
                            v579 = v577;
                        }
                        v565[v567] = v579;
                        v567 += 1 ;
                    }
                    static_array<Tuple2,3> v581;
                    int v583;
                    v583 = 0;
                    while (while_method_4(v583)){
                        float v587; float v588;
                        Tuple2 tmp38 = v127[v583];
                        v587 = tmp38.v0; v588 = tmp38.v1;
                        bool v591;
                        v591 = v324 == v583;
                        float v595; float v596;
                        if (v591){
                            float v592 = v325.v0;
                            float v593;
                            v593 = v587 + v592;
                            float v594;
                            v594 = v588 + 1.0f;
                            v595 = v593; v596 = v594;
                        } else {
                            v595 = v587; v596 = v588;
                        }
                        v581[v583] = Tuple2{v595, v596};
                        v583 += 1 ;
                    }
                    v536[v88] = Tuple1{v538, v565, v581};
                    v699 = v530;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            bool v700;
            v700 = v61 == 0;
            float v702;
            if (v700){
                v702 = v699;
            } else {
                float v701;
                v701 = -v699;
                v702 = v701;
            }
            v7.v0 = v702;
            Union6 v703;
            v703 = Union6{Union6_3{}};
            return loop_1(v0, v1, v2, v3, v4, v5, v7, v703);
            break;
        }
        case 3: { // RoundWithAction
            Union5 v705 = v6.case3.v0; bool v706 = v6.case3.v1; static_array<Union1,2> v707 = v6.case3.v2; int v708 = v6.case3.v3; static_array<int,2> v709 = v6.case3.v4; int v710 = v6.case3.v5; Union2 v711 = v6.case3.v6;
            static_array_list<Union0,32> & v712 = v3.v0;
            Union0 v713;
            v713 = Union0{Union0_1{v708, v711}};
            v712.push(v713);
            Union6 v714;
            v714 = Union6{Union6_2{v705, v706, v707, v708, v709, v710, v711}};
            float v715;
            v715 = loop_1(v0, v1, v2, v3, v4, v5, v7, v714);
            static_array_list<Union0,32> & v716 = v3.v0;
            Union0 v717;
            v717 = v716.pop();
            return v715;
            break;
        }
        case 4: { // TerminalCall
            Union5 v29 = v6.case4.v0; bool v30 = v6.case4.v1; static_array<Union1,2> v31 = v6.case4.v2; int v32 = v6.case4.v3; static_array<int,2> v33 = v6.case4.v4; int v34 = v6.case4.v5;
            int v36;
            v36 = v33[v32];
            Union9 v38;
            v38 = compare_hands_8(v29, v30, v31, v32, v33, v34);
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
            v55 = loop_1(v0, v1, v2, v3, v4, v5, v7, v54);
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
            v26 = loop_1(v0, v1, v2, v3, v4, v5, v7, v25);
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
inline bool while_method_8(std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v0, std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1>::iterator & v1){
    bool v2;
    v2 = v1 != v0.end();
    return v2;
}
static_array<float,3> normalize_12(static_array<float,3> v0){
    int v1; float v2;
    Tuple3 tmp43 = Tuple3{0, 0.0f};
    v1 = tmp43.v0; v2 = tmp43.v1;
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
inline bool while_method_9(std::unordered_map<static_array_list<Union0,32>, static_array<float,3>, Fun0, Fun1> & v0, std::unordered_map<static_array_list<Union0,32>, static_array<float,3>, Fun0, Fun1>::iterator & v1){
    bool v2;
    v2 = v1 != v0.end();
    return v2;
}
void method_14(Union1 v0){
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
void method_15(Union2 v0){
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
void method_13(Union0 v0){
    switch (v0.tag) {
        case 0: { // CommunityCardIs
            Union1 v1 = v0.case0.v0;
            printf("%s(","CommunityCardIs");
            method_14(v1);
            printf(")");
            return ;
            break;
        }
        case 1: { // PlayerAction
            int v2 = v0.case1.v0; Union2 v3 = v0.case1.v1;
            printf("%s(%d, ","PlayerAction", v2);
            method_15(v3);
            printf(")");
            return ;
            break;
        }
        case 2: { // PlayerGotCard
            int v4 = v0.case2.v0; Union1 v5 = v0.case2.v1;
            printf("%s(%d, ","PlayerGotCard", v4);
            method_14(v5);
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
                method_14(v12);
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
        v28 = v25 % 4000;
        bool v29;
        v29 = v28 == 0;
        if (v29){
            printf("{%s = %d; %s = %d}\n","i", v25, "nearTo", 100000);
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
        v69 = body_0(v38, v3, v46, v51, v47, v57, v68);
        v59 = v69;
        v58 += 1 ;
    }
    printf("{%s = %f}\n","reward_for_pl0", v59);
    fflush(stdout);
    printf("%s\n","Average policies:");
    fflush(stdout);
    std::unordered_map<static_array_list<Union0,32>, static_array<float,3>, Fun0, Fun1> v78(512, v0, v1);
    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v79 = v3.v0;
    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v80 = v79;
    auto v81 = v80.begin();
    while (while_method_8(v80, v81)){
        static_array_list<Union0,32> v83;
        v83 = v81->first;
        static_array<float,3> v84; static_array<float,3> v85; static_array<Tuple2,3> v86;
        Tuple1 tmp42 = v81->second;
        v84 = tmp42.v0; v85 = tmp42.v1; v86 = tmp42.v2;
        static_array<float,3> v87;
        v87 = normalize_12(v84);
        v78[v83] = v87;
        ++v81;
    }
    printf("%s\n","{");
    std::unordered_map<static_array_list<Union0,32>, static_array<float,3>, Fun0, Fun1> & v115 = v78;
    auto v116 = v115.begin();
    while (while_method_9(v115, v116)){
        static_array_list<Union0,32> v118;
        v118 = v116->first;
        static_array<float,3> v119;
        v119 = v116->second;
        printf("%s","[");
        int v120;
        v120 = v118.length;
        bool v121;
        v121 = 100 < v120;
        int v122;
        if (v121){
            v122 = 100;
        } else {
            v122 = v120;
        }
        int v123;
        v123 = 0;
        while (while_method_0(v122, v123)){
            Union0 v126;
            v126 = v118[v123];
            printf("");
            method_13(v126);
            printf("");
            int v128;
            v128 = v123 + 1;
            int v129;
            v129 = v118.length;
            bool v130;
            v130 = v128 < v129;
            if (v130){
                printf("%s","; ");
            } else {
            }
            v123 += 1 ;
        }
        int v131;
        v131 = v118.length;
        bool v132;
        v132 = v131 > 100;
        if (v132){
            printf("%s","; ...");
        } else {
        }
        printf("%s","]");
        printf("");
        printf("%s"," => ");
        printf("%s","[");
        int v133;
        v133 = 0;
        while (while_method_4(v133)){
            float v136;
            v136 = v119[v133];
            printf("%f",v136);
            int v138;
            v138 = v133 + 1;
            bool v139;
            v139 = v138 < 3;
            if (v139){
                printf("%s","; ");
            } else {
            }
            v133 += 1 ;
        }
        printf("%s","]");
        printf("\n");
        ++v116;
    }
    printf("%s\n","}");
    printf("\n");
    fflush(stdout);
    printf("%s\n","Expected values:");
    fflush(stdout);
    std::unordered_map<static_array_list<Union0,32>, static_array<float,3>, Fun0, Fun1> v184(512, v0, v1);
    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v185 = v3.v0;
    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v186 = v185;
    auto v187 = v186.begin();
    while (while_method_8(v186, v187)){
        static_array_list<Union0,32> v189;
        v189 = v187->first;
        static_array<float,3> v190; static_array<float,3> v191; static_array<Tuple2,3> v192;
        Tuple1 tmp44 = v187->second;
        v190 = tmp44.v0; v191 = tmp44.v1; v192 = tmp44.v2;
        static_array<float,3> v194;
        int v196;
        v196 = 0;
        while (while_method_4(v196)){
            float v200; float v201;
            Tuple2 tmp45 = v192[v196];
            v200 = tmp45.v0; v201 = tmp45.v1;
            bool v204;
            v204 = v201 == 0.0f;
            bool v205;
            v205 = v204 != true;
            float v207;
            if (v205){
                float v206;
                v206 = v200 / v201;
                v207 = v206;
            } else {
                v207 = 0.0f;
            }
            v194[v196] = v207;
            v196 += 1 ;
        }
        v184[v189] = v194;
        ++v187;
    }
    printf("%s\n","{");
    std::unordered_map<static_array_list<Union0,32>, static_array<float,3>, Fun0, Fun1> & v235 = v184;
    auto v236 = v235.begin();
    while (while_method_9(v235, v236)){
        static_array_list<Union0,32> v238;
        v238 = v236->first;
        static_array<float,3> v239;
        v239 = v236->second;
        printf("%s","[");
        int v240;
        v240 = v238.length;
        bool v241;
        v241 = 100 < v240;
        int v242;
        if (v241){
            v242 = 100;
        } else {
            v242 = v240;
        }
        int v243;
        v243 = 0;
        while (while_method_0(v242, v243)){
            Union0 v246;
            v246 = v238[v243];
            printf("");
            method_13(v246);
            printf("");
            int v248;
            v248 = v243 + 1;
            int v249;
            v249 = v238.length;
            bool v250;
            v250 = v248 < v249;
            if (v250){
                printf("%s","; ");
            } else {
            }
            v243 += 1 ;
        }
        int v251;
        v251 = v238.length;
        bool v252;
        v252 = v251 > 100;
        if (v252){
            printf("%s","; ...");
        } else {
        }
        printf("%s","]");
        printf("");
        printf("%s"," => ");
        printf("%s","[");
        int v253;
        v253 = 0;
        while (while_method_4(v253)){
            float v256;
            v256 = v239[v253];
            printf("%f",v256);
            int v258;
            v258 = v253 + 1;
            bool v259;
            v259 = v258 < 3;
            if (v259){
                printf("%s","; ");
            } else {
            }
            v253 += 1 ;
        }
        printf("%s","]");
        printf("\n");
        ++v236;
    }
    printf("%s\n","}");
    printf("\n");
    fflush(stdout);
    return 0;
}
