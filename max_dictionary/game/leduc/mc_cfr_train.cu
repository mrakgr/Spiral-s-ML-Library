#include "mc_cfr_train.auto.cu"
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
struct StackRefs1;
struct Union4;
struct Union3;
struct StackMut0;
struct Tuple3;
unsigned int loop_2(unsigned int v0, xso::rng & v1);
struct StackMut1;
struct StackMut2;
unsigned int find_nth_set_bit_3(int v0, unsigned int v1, unsigned int v2);
Tuple3 draw_card_1(xso::rng & v0, unsigned int v1);
struct Union5;
struct Tuple4;
float loop_4(StackRefs0 & v0, xso::rng & v1, StackRefs1 & v2, StackMut0 & v3, Union5 v4);
struct Union6;
struct Union7;
static_array<float,3> relu_6(static_array<float,3> v0);
struct Tuple5;
static_array<float,3> masking_normalize_7(static_array<float,3> v0, static_array<bool,3> v1);
static_array<float,3> regret_match_5(static_array<float,3> v0, static_array<bool,3> v1);
int loop_10(static_array<float,3> v0, float v1, int v2);
int pick_discrete__9(static_array<float,3> v0, float v1);
int sample_discrete__8(static_array<float,3> v0, xso::rng & v1);
struct Union8;
int tag_12(Union1 v0);
bool is_pair_13(int v0, int v1);
Tuple4 order_14(int v0, int v1);
Union8 compare_hands_11(Union4 v0, bool v1, static_array<Union1,2> v2, int v3, static_array<int,2> v4, int v5);
float body_0(StackRefs0 & v0, xso::rng & v1, StackRefs1 & v2, Union3 v3);
static_array<float,3> normalize_15(static_array<float,3> v0);
void method_17(Union1 v0);
void method_18(Union2 v0);
void method_16(Union0 v0);
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
struct StackRefs1 {
    static_array<Tuple2,2> & v1;
    static_array_list<Union0,32> & v2;
    unsigned int & v0;
    __host__ __device__ StackRefs1() = default;
    __host__ __device__ StackRefs1(unsigned int & t0, static_array<Tuple2,2> & t1, static_array_list<Union0,32> & t2) : v0(t0), v1(t1), v2(t2) {}
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
struct StackMut0 {
    float v0;
    __host__ __device__ StackMut0() = default;
    __host__ __device__ StackMut0(float t0) : v0(t0) {}
};
struct Tuple3 {
    Union1 v0;
    unsigned int v1;
    __host__ __device__ Tuple3() = default;
    __host__ __device__ Tuple3(Union1 t0, unsigned int t1) : v0(t0), v1(t1) {}
};
struct StackMut1 {
    int v0;
    __host__ __device__ StackMut1() = default;
    __host__ __device__ StackMut1(int t0) : v0(t0) {}
};
struct StackMut2 {
    unsigned int v0;
    __host__ __device__ StackMut2() = default;
    __host__ __device__ StackMut2(unsigned int t0) : v0(t0) {}
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
struct Tuple4 {
    int v0;
    int v1;
    __host__ __device__ Tuple4() = default;
    __host__ __device__ Tuple4(int t0, int t1) : v0(t0), v1(t1) {}
};
struct Union6_0 { // None
};
struct Union6_1 { // Some
    static_array<float,3> v0;
    static_array<float,3> v1;
    static_array<Tuple2,3> v2;
    __host__ __device__ Union6_1(static_array<float,3> t0, static_array<float,3> t1, static_array<Tuple2,3> t2) : v0(t0), v1(t1), v2(t2) {}
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
struct Union7_0 { // None
};
struct Union7_1 { // Some
    Union2 v0;
    __host__ __device__ Union7_1(Union2 t0) : v0(t0) {}
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
struct Tuple5 {
    int v0;
    float v1;
    __host__ __device__ Tuple5() = default;
    __host__ __device__ Tuple5(int t0, float t1) : v0(t0), v1(t1) {}
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
    StackMut1 v6{0};
    StackMut2 v7{4294967295u};
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
Tuple3 draw_card_1(xso::rng & v0, unsigned int v1){
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
    return Tuple3{v37, v40};
}
float loop_4(StackRefs0 & v0, xso::rng & v1, StackRefs1 & v2, StackMut0 & v3, Union5 v4){
    switch (v4.tag) {
        case 0: { // T_game_chance_community_card
            Union4 v6 = v4.case0.v0; bool v7 = v4.case0.v1; static_array<Union1,2> v8 = v4.case0.v2; int v9 = v4.case0.v3; static_array<int,2> v10 = v4.case0.v4; int v11 = v4.case0.v5; Union1 v12 = v4.case0.v6;
            int v13;
            v13 = 2;
            int v14; int v15;
            Tuple4 tmp3 = Tuple4{0, 0};
            v14 = tmp3.v0; v15 = tmp3.v1;
            while (while_method_1(v14)){
                int v18;
                v18 = v10[v14];
                bool v20;
                v20 = v15 >= v18;
                int v21;
                if (v20){
                    v21 = v15;
                } else {
                    v21 = v18;
                }
                v15 = v21;
                v14 += 1 ;
            }
            static_array<int,2> v23;
            int v25;
            v25 = 0;
            while (while_method_1(v25)){
                v23[v25] = v15;
                v25 += 1 ;
            }
            Union4 v27;
            v27 = Union4{Union4_1{v12}};
            bool v28;
            v28 = true;
            int v29;
            v29 = 0;
            Union3 v30;
            v30 = Union3{Union3_2{v27, v28, v8, v29, v23, v13}};
            return body_0(v0, v1, v2, v30);
            break;
        }
        case 1: { // T_game_chance_init
            Union1 v32 = v4.case1.v0; Union1 v33 = v4.case1.v1;
            int v34;
            v34 = 2;
            static_array<int,2> v36;
            v36[0] = 1;
            v36[1] = 1;
            static_array<Union1,2> v39;
            v39[0] = v32;
            v39[1] = v33;
            Union4 v41;
            v41 = Union4{Union4_0{}};
            bool v42;
            v42 = true;
            int v43;
            v43 = 0;
            Union3 v44;
            v44 = Union3{Union3_2{v41, v42, v39, v43, v36, v34}};
            return body_0(v0, v1, v2, v44);
            break;
        }
        case 2: { // T_game_round
            Union4 v46 = v4.case2.v0; bool v47 = v4.case2.v1; static_array<Union1,2> v48 = v4.case2.v2; int v49 = v4.case2.v3; static_array<int,2> v50 = v4.case2.v4; int v51 = v4.case2.v5; Union2 v52 = v4.case2.v6;
            Union3 v144;
            switch (v46.tag) {
                case 0: { // None
                    switch (v52.tag) {
                        case 0: { // Call
                            if (v47){
                                int v106;
                                v106 = v49 ^ 1;
                                v144 = Union3{Union3_2{v46, false, v48, v106, v50, v51}};
                            } else {
                                v144 = Union3{Union3_0{v46, v47, v48, v49, v50, v51}};
                            }
                            break;
                        }
                        case 1: { // Fold
                            v144 = Union3{Union3_5{v46, v47, v48, v49, v50, v51}};
                            break;
                        }
                        case 2: { // Raise
                            bool v110;
                            v110 = v51 > 0;
                            if (v110){
                                int v111;
                                v111 = v49 ^ 1;
                                int v112;
                                v112 = -1 + v51;
                                int v113; int v114;
                                Tuple4 tmp4 = Tuple4{0, 0};
                                v113 = tmp4.v0; v114 = tmp4.v1;
                                while (while_method_1(v113)){
                                    int v117;
                                    v117 = v50[v113];
                                    bool v119;
                                    v119 = v114 >= v117;
                                    int v120;
                                    if (v119){
                                        v120 = v114;
                                    } else {
                                        v120 = v117;
                                    }
                                    v114 = v120;
                                    v113 += 1 ;
                                }
                                static_array<int,2> v122;
                                int v124;
                                v124 = 0;
                                while (while_method_1(v124)){
                                    v122[v124] = v114;
                                    v124 += 1 ;
                                }
                                static_array<int,2> v127;
                                int v129;
                                v129 = 0;
                                while (while_method_1(v129)){
                                    int v132;
                                    v132 = v122[v129];
                                    bool v134;
                                    v134 = v129 == v49;
                                    int v136;
                                    if (v134){
                                        int v135;
                                        v135 = v132 + 2;
                                        v136 = v135;
                                    } else {
                                        v136 = v132;
                                    }
                                    v127[v129] = v136;
                                    v129 += 1 ;
                                }
                                v144 = Union3{Union3_2{v46, false, v48, v111, v127, v112}};
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
                    Union1 v53 = v46.case1.v0;
                    switch (v52.tag) {
                        case 0: { // Call
                            if (v47){
                                int v55;
                                v55 = v49 ^ 1;
                                v144 = Union3{Union3_2{v46, false, v48, v55, v50, v51}};
                            } else {
                                int v57; int v58;
                                Tuple4 tmp5 = Tuple4{0, 0};
                                v57 = tmp5.v0; v58 = tmp5.v1;
                                while (while_method_1(v57)){
                                    int v61;
                                    v61 = v50[v57];
                                    bool v63;
                                    v63 = v58 >= v61;
                                    int v64;
                                    if (v63){
                                        v64 = v58;
                                    } else {
                                        v64 = v61;
                                    }
                                    v58 = v64;
                                    v57 += 1 ;
                                }
                                static_array<int,2> v66;
                                int v68;
                                v68 = 0;
                                while (while_method_1(v68)){
                                    v66[v68] = v58;
                                    v68 += 1 ;
                                }
                                v144 = Union3{Union3_4{v46, v47, v48, v49, v66, v51}};
                            }
                            break;
                        }
                        case 1: { // Fold
                            v144 = Union3{Union3_5{v46, v47, v48, v49, v50, v51}};
                            break;
                        }
                        case 2: { // Raise
                            bool v72;
                            v72 = v51 > 0;
                            if (v72){
                                int v73;
                                v73 = v49 ^ 1;
                                int v74;
                                v74 = -1 + v51;
                                int v75; int v76;
                                Tuple4 tmp6 = Tuple4{0, 0};
                                v75 = tmp6.v0; v76 = tmp6.v1;
                                while (while_method_1(v75)){
                                    int v79;
                                    v79 = v50[v75];
                                    bool v81;
                                    v81 = v76 >= v79;
                                    int v82;
                                    if (v81){
                                        v82 = v76;
                                    } else {
                                        v82 = v79;
                                    }
                                    v76 = v82;
                                    v75 += 1 ;
                                }
                                static_array<int,2> v84;
                                int v86;
                                v86 = 0;
                                while (while_method_1(v86)){
                                    v84[v86] = v76;
                                    v86 += 1 ;
                                }
                                static_array<int,2> v89;
                                int v91;
                                v91 = 0;
                                while (while_method_1(v91)){
                                    int v94;
                                    v94 = v84[v91];
                                    bool v96;
                                    v96 = v91 == v49;
                                    int v98;
                                    if (v96){
                                        int v97;
                                        v97 = v94 + 4;
                                        v98 = v97;
                                    } else {
                                        v98 = v94;
                                    }
                                    v89[v91] = v98;
                                    v91 += 1 ;
                                }
                                v144 = Union3{Union3_2{v46, false, v48, v73, v89, v74}};
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
            return body_0(v0, v1, v2, v144);
            break;
        }
        case 3: { // T_none
            float v5 = v3.v0;
            return v5;
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
static_array<float,3> relu_6(static_array<float,3> v0){
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
static_array<float,3> masking_normalize_7(static_array<float,3> v0, static_array<bool,3> v1){
    int v2; float v3;
    Tuple5 tmp11 = Tuple5{0, 0.0f};
    v2 = tmp11.v0; v3 = tmp11.v1;
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
    Tuple5 tmp12 = Tuple5{0, 0.0f};
    v23 = tmp12.v0; v24 = tmp12.v1;
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
static_array<float,3> regret_match_5(static_array<float,3> v0, static_array<bool,3> v1){
    static_array<float,3> v2;
    v2 = relu_6(v0);
    return masking_normalize_7(v2, v1);
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
Tuple4 order_14(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple4{v1, v0};
    } else {
        return Tuple4{v0, v1};
    }
}
Union8 compare_hands_11(Union4 v0, bool v1, static_array<Union1,2> v2, int v3, static_array<int,2> v4, int v5){
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
                        return Union8{Union8_2{}};
                    } else {
                        bool v21;
                        v21 = v12 > v16;
                        if (v21){
                            return Union8{Union8_1{}};
                        } else {
                            return Union8{Union8_0{}};
                        }
                    }
                } else {
                    return Union8{Union8_1{}};
                }
            } else {
                if (v18){
                    return Union8{Union8_2{}};
                } else {
                    int v29; int v30;
                    Tuple4 tmp28 = order_14(v8, v12);
                    v29 = tmp28.v0; v30 = tmp28.v1;
                    int v31; int v32;
                    Tuple4 tmp29 = order_14(v8, v16);
                    v31 = tmp29.v0; v32 = tmp29.v1;
                    bool v33;
                    v33 = v29 < v31;
                    Union8 v39;
                    if (v33){
                        v39 = Union8{Union8_2{}};
                    } else {
                        bool v35;
                        v35 = v29 > v31;
                        if (v35){
                            v39 = Union8{Union8_1{}};
                        } else {
                            v39 = Union8{Union8_0{}};
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
                            return Union8{Union8_2{}};
                        } else {
                            bool v43;
                            v43 = v30 > v32;
                            if (v43){
                                return Union8{Union8_1{}};
                            } else {
                                return Union8{Union8_0{}};
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
float body_0(StackRefs0 & v0, xso::rng & v1, StackRefs1 & v2, Union3 v3){
    StackMut0 v4{0.0f};
    switch (v3.tag) {
        case 0: { // ChanceCommunityCard
            Union4 v465 = v3.case0.v0; bool v466 = v3.case0.v1; static_array<Union1,2> v467 = v3.case0.v2; int v468 = v3.case0.v3; static_array<int,2> v469 = v3.case0.v4; int v470 = v3.case0.v5;
            unsigned int & v471 = v2.v0;
            unsigned int v472;
            v472 = v471;
            Union1 v473; unsigned int v474;
            Tuple3 tmp2 = draw_card_1(v1, v472);
            v473 = tmp2.v0; v474 = tmp2.v1;
            v2.v0 = v474;
            static_array_list<Union0,32> & v475 = v2.v2;
            Union0 v476;
            v476 = Union0{Union0_0{v473}};
            v475.push(v476);
            Union5 v477;
            v477 = Union5{Union5_0{v465, v466, v467, v468, v469, v470, v473}};
            float v478;
            v478 = loop_4(v0, v1, v2, v4, v477);
            static_array_list<Union0,32> & v479 = v2.v2;
            Union0 v480;
            v480 = v479.pop();
            v2.v0 = v472;
            return v478;
            break;
        }
        case 1: { // ChanceInit
            unsigned int & v481 = v2.v0;
            unsigned int v482;
            v482 = v481;
            Union1 v483; unsigned int v484;
            Tuple3 tmp7 = draw_card_1(v1, v482);
            v483 = tmp7.v0; v484 = tmp7.v1;
            v2.v0 = v484;
            unsigned int & v485 = v2.v0;
            unsigned int v486;
            v486 = v485;
            Union1 v487; unsigned int v488;
            Tuple3 tmp8 = draw_card_1(v1, v486);
            v487 = tmp8.v0; v488 = tmp8.v1;
            v2.v0 = v488;
            static_array_list<Union0,32> & v489 = v2.v2;
            Union0 v490;
            v490 = Union0{Union0_2{0, v483}};
            v489.push(v490);
            static_array_list<Union0,32> & v491 = v2.v2;
            Union0 v492;
            v492 = Union0{Union0_2{1, v487}};
            v491.push(v492);
            Union5 v493;
            v493 = Union5{Union5_1{v483, v487}};
            float v494;
            v494 = loop_4(v0, v1, v2, v4, v493);
            static_array_list<Union0,32> & v495 = v2.v2;
            Union0 v496;
            v496 = v495.pop();
            static_array_list<Union0,32> & v497 = v2.v2;
            Union0 v498;
            v498 = v497.pop();
            v2.v0 = v486;
            v2.v0 = v482;
            return v494;
            break;
        }
        case 2: { // Round
            Union4 v55 = v3.case2.v0; bool v56 = v3.case2.v1; static_array<Union1,2> v57 = v3.case2.v2; int v58 = v3.case2.v3; static_array<int,2> v59 = v3.case2.v4; int v60 = v3.case2.v5;
            static_array_list<Union0,32> & v61 = v2.v2;
            int v62;
            v62 = v61.length;
            bool v63;
            v63 = 32 >= v62;
            bool v64;
            v64 = v63 == false;
            if (v64){
                assert("The type level dimension has to equal the value passed at runtime into create." && v63);
            } else {
            }
            static_array_list<Union0,32> v67;
            v67 = static_array_list<Union0,32>{};
            v67.unsafe_set_length(v62);
            int v69; int v70;
            Tuple4 tmp9 = Tuple4{0, 0};
            v69 = tmp9.v0; v70 = tmp9.v1;
            while (while_method_0(v62, v69)){
                Union0 v73;
                v73 = v61[v69];
                bool v78;
                switch (v73.tag) {
                    case 2: { // PlayerGotCard
                        int v75 = v73.case2.v0; Union1 v76 = v73.case2.v1;
                        bool v77;
                        v77 = v75 == v58;
                        v78 = v77;
                        break;
                    }
                    default: {
                        v78 = true;
                    }
                }
                int v80;
                if (v78){
                    v67[v70] = v73;
                    int v79;
                    v79 = v70 + 1;
                    v80 = v79;
                } else {
                    v80 = v70;
                }
                v70 = v80;
                v69 += 1 ;
            }
            bool v81;
            v81 = 32 >= v70;
            bool v82;
            v82 = v81 == false;
            if (v82){
                assert("The type level dimension has to equal the value passed at runtime into create." && v81);
            } else {
            }
            static_array_list<Union0,32> v85;
            v85 = static_array_list<Union0,32>{};
            v85.unsafe_set_length(v70);
            int v87;
            v87 = 0;
            while (while_method_0(v70, v87)){
                Union0 v90;
                v90 = v67[v87];
                v85[v87] = v90;
                v87 += 1 ;
            }
            std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v92 = v0.v0;
            auto v93 = v92.find(v85);
            bool v94;
            v94 = v93 != v92.end();
            Union6 v100;
            if (v94){
                static_array<float,3> v95; static_array<float,3> v96; static_array<Tuple2,3> v97;
                Tuple1 tmp10 = v93->second;
                v95 = tmp10.v0; v96 = tmp10.v1; v97 = tmp10.v2;
                v100 = Union6{Union6_1{v95, v96, v97}};
            } else {
                v100 = Union6{Union6_0{}};
            }
            static_array<float,3> v122; static_array<float,3> v123; static_array<Tuple2,3> v124;
            switch (v100.tag) {
                case 0: { // None
                    static_array<float,3> v105;
                    int v107;
                    v107 = 0;
                    while (while_method_4(v107)){
                        v105[v107] = 0.0f;
                        v107 += 1 ;
                    }
                    static_array<float,3> v110;
                    int v112;
                    v112 = 0;
                    while (while_method_4(v112)){
                        v110[v112] = 0.0f;
                        v112 += 1 ;
                    }
                    static_array<Tuple2,3> v115;
                    int v117;
                    v117 = 0;
                    while (while_method_4(v117)){
                        v115[v117] = Tuple2{0.0f, 0.0f};
                        v117 += 1 ;
                    }
                    v122 = v105; v123 = v110; v124 = v115;
                    break;
                }
                case 1: { // Some
                    static_array<float,3> v101 = v100.case1.v0; static_array<float,3> v102 = v100.case1.v1; static_array<Tuple2,3> v103 = v100.case1.v2;
                    v122 = v101; v123 = v102; v124 = v103;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            int v126;
            v126 = v59[0];
            int v129;
            v129 = v59[1];
            bool v131;
            v131 = v126 == v129;
            bool v132;
            v132 = v131 != true;
            Union7 v136;
            if (v132){
                Union2 v133;
                v133 = Union2{Union2_1{}};
                v136 = Union7{Union7_1{v133}};
            } else {
                v136 = Union7{Union7_0{}};
            }
            bool v137;
            v137 = v60 > 0;
            Union7 v141;
            if (v137){
                Union2 v138;
                v138 = Union2{Union2_2{}};
                v141 = Union7{Union7_1{v138}};
            } else {
                v141 = Union7{Union7_0{}};
            }
            bool v144;
            switch (v141.tag) {
                case 0: { // None
                    v144 = false;
                    break;
                }
                case 1: { // Some
                    Union2 v142 = v141.case1.v0;
                    v144 = true;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            bool v147;
            switch (v136.tag) {
                case 0: { // None
                    v147 = false;
                    break;
                }
                case 1: { // Some
                    Union2 v145 = v136.case1.v0;
                    v147 = true;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            static_array<bool,3> v149;
            v149[0] = true;
            v149[1] = v147;
            v149[2] = v144;
            static_array<float,3> v151;
            v151 = regret_match_5(v123, v149);
            static_array<float,3> v153;
            int v155;
            v155 = 0;
            while (while_method_4(v155)){
                v153[v155] = 0.0f;
                v155 += 1 ;
            }
            static_array<float,3> v157;
            v157 = masking_normalize_7(v153, v149);
            static_array<float,3> v159;
            int v161;
            v161 = 0;
            while (while_method_4(v161)){
                float v164;
                v164 = v151[v161];
                float v167;
                v167 = v157[v161];
                float v169;
                v169 = 0.0f * v164;
                float v170;
                v170 = v169 + v167;
                v159[v161] = v170;
                v161 += 1 ;
            }
            int v171;
            v171 = sample_discrete__8(v159, v1);
            StackMut0 v172{0.0f};
            float v238;
            switch (v141.tag) {
                case 1: { // Some
                    Union2 v173 = v141.case1.v0;
                    bool v174;
                    v174 = 2 == v171;
                    if (v174){
                        float v176;
                        v176 = v151[2];
                        float v179;
                        v179 = v159[2];
                        static_array<Tuple2,2> & v181 = v2.v1;
                        float v184; float v185;
                        Tuple2 tmp13 = v181[v58];
                        v184 = tmp13.v0; v185 = tmp13.v1;
                        static_array<Tuple2,2> & v188 = v2.v1;
                        float v189;
                        v189 = log(v176);
                        float v190;
                        v190 = v189 + v184;
                        float v191;
                        v191 = log(v179);
                        float v192;
                        v192 = v191 + v185;
                        v188[v58] = Tuple2{v190, v192};
                        static_array_list<Union0,32> & v193 = v2.v2;
                        Union0 v194;
                        v194 = Union0{Union0_1{v58, v173}};
                        v193.push(v194);
                        Union5 v195;
                        v195 = Union5{Union5_2{v55, v56, v57, v58, v59, v60, v173}};
                        float v196;
                        v196 = loop_4(v0, v1, v2, v4, v195);
                        static_array_list<Union0,32> & v197 = v2.v2;
                        Union0 v198;
                        v198 = v197.pop();
                        static_array<Tuple2,2> & v199 = v2.v1;
                        v199[v58] = Tuple2{v184, v185};
                        bool v200;
                        v200 = v58 == 0;
                        float v202;
                        if (v200){
                            v202 = v196;
                        } else {
                            float v201;
                            v201 = -v196;
                            v202 = v201;
                        }
                        v172.v0 = v202;
                        float v205; float v206;
                        Tuple2 tmp14 = v124[2];
                        v205 = tmp14.v0; v206 = tmp14.v1;
                        bool v209;
                        v209 = v206 == 0.0f;
                        bool v210;
                        v210 = v209 != true;
                        float v212;
                        if (v210){
                            float v211;
                            v211 = v205 / v206;
                            v212 = v211;
                        } else {
                            v212 = 0.0f;
                        }
                        float v213 = v172.v0;
                        float v214;
                        v214 = v213 - v212;
                        float v215;
                        v215 = v214 / v179;
                        float v216;
                        v216 = v215 + v212;
                        v238 = v216;
                    } else {
                        float v219; float v220;
                        Tuple2 tmp15 = v124[2];
                        v219 = tmp15.v0; v220 = tmp15.v1;
                        bool v223;
                        v223 = v220 == 0.0f;
                        bool v224;
                        v224 = v223 != true;
                        if (v224){
                            float v225;
                            v225 = v219 / v220;
                            v238 = v225;
                        } else {
                            v238 = 0.0f;
                        }
                    }
                    break;
                }
                default: {
                    float v230; float v231;
                    Tuple2 tmp16 = v124[2];
                    v230 = tmp16.v0; v231 = tmp16.v1;
                    bool v234;
                    v234 = v231 == 0.0f;
                    bool v235;
                    v235 = v234 != true;
                    if (v235){
                        float v236;
                        v236 = v230 / v231;
                        v238 = v236;
                    } else {
                        v238 = 0.0f;
                    }
                }
            }
            float v304;
            switch (v136.tag) {
                case 1: { // Some
                    Union2 v239 = v136.case1.v0;
                    bool v240;
                    v240 = 1 == v171;
                    if (v240){
                        float v242;
                        v242 = v151[1];
                        float v245;
                        v245 = v159[1];
                        static_array<Tuple2,2> & v247 = v2.v1;
                        float v250; float v251;
                        Tuple2 tmp17 = v247[v58];
                        v250 = tmp17.v0; v251 = tmp17.v1;
                        static_array<Tuple2,2> & v254 = v2.v1;
                        float v255;
                        v255 = log(v242);
                        float v256;
                        v256 = v255 + v250;
                        float v257;
                        v257 = log(v245);
                        float v258;
                        v258 = v257 + v251;
                        v254[v58] = Tuple2{v256, v258};
                        static_array_list<Union0,32> & v259 = v2.v2;
                        Union0 v260;
                        v260 = Union0{Union0_1{v58, v239}};
                        v259.push(v260);
                        Union5 v261;
                        v261 = Union5{Union5_2{v55, v56, v57, v58, v59, v60, v239}};
                        float v262;
                        v262 = loop_4(v0, v1, v2, v4, v261);
                        static_array_list<Union0,32> & v263 = v2.v2;
                        Union0 v264;
                        v264 = v263.pop();
                        static_array<Tuple2,2> & v265 = v2.v1;
                        v265[v58] = Tuple2{v250, v251};
                        bool v266;
                        v266 = v58 == 0;
                        float v268;
                        if (v266){
                            v268 = v262;
                        } else {
                            float v267;
                            v267 = -v262;
                            v268 = v267;
                        }
                        v172.v0 = v268;
                        float v271; float v272;
                        Tuple2 tmp18 = v124[1];
                        v271 = tmp18.v0; v272 = tmp18.v1;
                        bool v275;
                        v275 = v272 == 0.0f;
                        bool v276;
                        v276 = v275 != true;
                        float v278;
                        if (v276){
                            float v277;
                            v277 = v271 / v272;
                            v278 = v277;
                        } else {
                            v278 = 0.0f;
                        }
                        float v279 = v172.v0;
                        float v280;
                        v280 = v279 - v278;
                        float v281;
                        v281 = v280 / v245;
                        float v282;
                        v282 = v281 + v278;
                        v304 = v282;
                    } else {
                        float v285; float v286;
                        Tuple2 tmp19 = v124[1];
                        v285 = tmp19.v0; v286 = tmp19.v1;
                        bool v289;
                        v289 = v286 == 0.0f;
                        bool v290;
                        v290 = v289 != true;
                        if (v290){
                            float v291;
                            v291 = v285 / v286;
                            v304 = v291;
                        } else {
                            v304 = 0.0f;
                        }
                    }
                    break;
                }
                default: {
                    float v296; float v297;
                    Tuple2 tmp20 = v124[1];
                    v296 = tmp20.v0; v297 = tmp20.v1;
                    bool v300;
                    v300 = v297 == 0.0f;
                    bool v301;
                    v301 = v300 != true;
                    if (v301){
                        float v302;
                        v302 = v296 / v297;
                        v304 = v302;
                    } else {
                        v304 = 0.0f;
                    }
                }
            }
            bool v305;
            v305 = 0 == v171;
            float v360;
            if (v305){
                float v307;
                v307 = v151[0];
                float v310;
                v310 = v159[0];
                static_array<Tuple2,2> & v312 = v2.v1;
                float v315; float v316;
                Tuple2 tmp21 = v312[v58];
                v315 = tmp21.v0; v316 = tmp21.v1;
                static_array<Tuple2,2> & v319 = v2.v1;
                float v320;
                v320 = log(v307);
                float v321;
                v321 = v320 + v315;
                float v322;
                v322 = log(v310);
                float v323;
                v323 = v322 + v316;
                v319[v58] = Tuple2{v321, v323};
                static_array_list<Union0,32> & v324 = v2.v2;
                Union2 v325;
                v325 = Union2{Union2_0{}};
                Union0 v326;
                v326 = Union0{Union0_1{v58, v325}};
                v324.push(v326);
                Union2 v327;
                v327 = Union2{Union2_0{}};
                Union5 v328;
                v328 = Union5{Union5_2{v55, v56, v57, v58, v59, v60, v327}};
                float v329;
                v329 = loop_4(v0, v1, v2, v4, v328);
                static_array_list<Union0,32> & v330 = v2.v2;
                Union0 v331;
                v331 = v330.pop();
                static_array<Tuple2,2> & v332 = v2.v1;
                v332[v58] = Tuple2{v315, v316};
                bool v333;
                v333 = v58 == 0;
                float v335;
                if (v333){
                    v335 = v329;
                } else {
                    float v334;
                    v334 = -v329;
                    v335 = v334;
                }
                v172.v0 = v335;
                float v338; float v339;
                Tuple2 tmp22 = v124[0];
                v338 = tmp22.v0; v339 = tmp22.v1;
                bool v342;
                v342 = v339 == 0.0f;
                bool v343;
                v343 = v342 != true;
                float v345;
                if (v343){
                    float v344;
                    v344 = v338 / v339;
                    v345 = v344;
                } else {
                    v345 = 0.0f;
                }
                float v346 = v172.v0;
                float v347;
                v347 = v346 - v345;
                float v348;
                v348 = v347 / v310;
                float v349;
                v349 = v348 + v345;
                v360 = v349;
            } else {
                float v352; float v353;
                Tuple2 tmp23 = v124[0];
                v352 = tmp23.v0; v353 = tmp23.v1;
                bool v356;
                v356 = v353 == 0.0f;
                bool v357;
                v357 = v356 != true;
                if (v357){
                    float v358;
                    v358 = v352 / v353;
                    v360 = v358;
                } else {
                    v360 = 0.0f;
                }
            }
            static_array<float,3> v362;
            v362[0] = v360;
            v362[1] = v304;
            v362[2] = v238;
            static_array<float,3> v365;
            int v367;
            v367 = 0;
            while (while_method_4(v367)){
                float v370;
                v370 = v362[v367];
                float v373;
                v373 = v151[v367];
                float v375;
                v375 = v370 * v373;
                v365[v367] = v375;
                v367 += 1 ;
            }
            int v376; float v377;
            Tuple5 tmp24 = Tuple5{0, 0.0f};
            v376 = tmp24.v0; v377 = tmp24.v1;
            while (while_method_4(v376)){
                float v380;
                v380 = v365[v376];
                float v382;
                v382 = v377 + v380;
                v377 = v382;
                v376 += 1 ;
            }
            std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v383 = v0.v0;
            static_array<float,3> v385;
            int v387;
            v387 = 0;
            while (while_method_4(v387)){
                float v390;
                v390 = v122[v387];
                float v393;
                v393 = v151[v387];
                float v395;
                v395 = 0.99902344f * v390;
                float v396;
                v396 = v395 + v393;
                v385[v387] = v396;
                v387 += 1 ;
            }
            static_array<Tuple2,2> & v397 = v2.v1;
            int v398; float v399;
            Tuple5 tmp25 = Tuple5{0, 0.0f};
            v398 = tmp25.v0; v399 = tmp25.v1;
            while (while_method_1(v398)){
                float v403; float v404;
                Tuple2 tmp26 = v397[v398];
                v403 = tmp26.v0; v404 = tmp26.v1;
                bool v407;
                v407 = v398 == v58;
                float v408;
                if (v407){
                    v408 = 0.0f;
                } else {
                    v408 = v403;
                }
                float v409;
                v409 = v399 + v408;
                float v410;
                v410 = v409 - v404;
                v399 = v410;
                v398 += 1 ;
            }
            float v411;
            v411 = exp(v399);
            static_array<float,3> v413;
            int v415;
            v415 = 0;
            while (while_method_4(v415)){
                float v418;
                v418 = v123[v415];
                float v421;
                v421 = v362[v415];
                float v423;
                v423 = v421 - v377;
                float v424;
                v424 = v411 * v423;
                float v425;
                v425 = v418 + v424;
                bool v426;
                v426 = 0.0f >= v425;
                float v427;
                if (v426){
                    v427 = 0.0f;
                } else {
                    v427 = v425;
                }
                v413[v415] = v427;
                v415 += 1 ;
            }
            static_array<Tuple2,3> v429;
            int v431;
            v431 = 0;
            while (while_method_4(v431)){
                float v435; float v436;
                Tuple2 tmp27 = v124[v431];
                v435 = tmp27.v0; v436 = tmp27.v1;
                bool v439;
                v439 = v171 == v431;
                float v445; float v446;
                if (v439){
                    float v440;
                    v440 = v435 * 0.5f;
                    float v441 = v172.v0;
                    float v442;
                    v442 = v440 + v441;
                    float v443;
                    v443 = v436 * 0.5f;
                    float v444;
                    v444 = v443 + 1.0f;
                    v445 = v442; v446 = v444;
                } else {
                    v445 = v435; v446 = v436;
                }
                v429[v431] = Tuple2{v445, v446};
                v431 += 1 ;
            }
            v383[v85] = Tuple1{v385, v413, v429};
            bool v447;
            v447 = v58 == 0;
            float v449;
            if (v447){
                v449 = v377;
            } else {
                float v448;
                v448 = -v377;
                v449 = v448;
            }
            v4.v0 = v449;
            Union5 v450;
            v450 = Union5{Union5_3{}};
            return loop_4(v0, v1, v2, v4, v450);
            break;
        }
        case 3: { // RoundWithAction
            Union4 v452 = v3.case3.v0; bool v453 = v3.case3.v1; static_array<Union1,2> v454 = v3.case3.v2; int v455 = v3.case3.v3; static_array<int,2> v456 = v3.case3.v4; int v457 = v3.case3.v5; Union2 v458 = v3.case3.v6;
            static_array_list<Union0,32> & v459 = v2.v2;
            Union0 v460;
            v460 = Union0{Union0_1{v455, v458}};
            v459.push(v460);
            Union5 v461;
            v461 = Union5{Union5_2{v452, v453, v454, v455, v456, v457, v458}};
            float v462;
            v462 = loop_4(v0, v1, v2, v4, v461);
            static_array_list<Union0,32> & v463 = v2.v2;
            Union0 v464;
            v464 = v463.pop();
            return v462;
            break;
        }
        case 4: { // TerminalCall
            Union4 v26 = v3.case4.v0; bool v27 = v3.case4.v1; static_array<Union1,2> v28 = v3.case4.v2; int v29 = v3.case4.v3; static_array<int,2> v30 = v3.case4.v4; int v31 = v3.case4.v5;
            int v33;
            v33 = v30[v29];
            Union8 v35;
            v35 = compare_hands_11(v26, v27, v28, v29, v30, v31);
            int v40; int v41;
            switch (v35.tag) {
                case 0: { // Eq
                    v40 = 0; v41 = -1;
                    break;
                }
                case 1: { // Gt
                    v40 = v33; v41 = 0;
                    break;
                }
                case 2: { // Lt
                    v40 = v33; v41 = 1;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            int v42;
            v42 = -v41;
            bool v43;
            v43 = v41 >= v42;
            int v44;
            if (v43){
                v44 = v41;
            } else {
                v44 = v42;
            }
            float v45;
            v45 = (float)v40;
            bool v46;
            v46 = v44 == 0;
            float v48;
            if (v46){
                v48 = v45;
            } else {
                float v47;
                v47 = -v45;
                v48 = v47;
            }
            v4.v0 = v48;
            static_array_list<Union0,32> & v49 = v2.v2;
            Union0 v50;
            v50 = Union0{Union0_3{v28, v40, v41}};
            v49.push(v50);
            Union5 v51;
            v51 = Union5{Union5_3{}};
            float v52;
            v52 = loop_4(v0, v1, v2, v4, v51);
            static_array_list<Union0,32> & v53 = v2.v2;
            Union0 v54;
            v54 = v53.pop();
            return v52;
            break;
        }
        case 5: { // TerminalFold
            Union4 v5 = v3.case5.v0; bool v6 = v3.case5.v1; static_array<Union1,2> v7 = v3.case5.v2; int v8 = v3.case5.v3; static_array<int,2> v9 = v3.case5.v4; int v10 = v3.case5.v5;
            int v12;
            v12 = v9[v8];
            int v14;
            v14 = -v12;
            float v15;
            v15 = (float)v14;
            bool v16;
            v16 = v8 == 0;
            float v18;
            if (v16){
                v18 = v15;
            } else {
                float v17;
                v17 = -v15;
                v18 = v17;
            }
            v4.v0 = v18;
            int v19;
            v19 = v8 ^ 1;
            static_array_list<Union0,32> & v20 = v2.v2;
            Union0 v21;
            v21 = Union0{Union0_3{v7, v12, v19}};
            v20.push(v21);
            Union5 v22;
            v22 = Union5{Union5_3{}};
            float v23;
            v23 = loop_4(v0, v1, v2, v4, v22);
            static_array_list<Union0,32> & v24 = v2.v2;
            Union0 v25;
            v25 = v24.pop();
            return v23;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
inline bool while_method_7(std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v0, std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1>::iterator & v1){
    bool v2;
    v2 = v1 != v0.end();
    return v2;
}
static_array<float,3> normalize_15(static_array<float,3> v0){
    int v1; float v2;
    Tuple5 tmp31 = Tuple5{0, 0.0f};
    v1 = tmp31.v0; v2 = tmp31.v1;
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
inline bool while_method_8(std::unordered_map<static_array_list<Union0,32>, static_array<float,3>, Fun0, Fun1> & v0, std::unordered_map<static_array_list<Union0,32>, static_array<float,3>, Fun0, Fun1>::iterator & v1){
    bool v2;
    v2 = v1 != v0.end();
    return v2;
}
void method_17(Union1 v0){
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
void method_18(Union2 v0){
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
void method_16(Union0 v0){
    switch (v0.tag) {
        case 0: { // CommunityCardIs
            Union1 v1 = v0.case0.v0;
            printf("%s(","CommunityCardIs");
            method_17(v1);
            printf(")");
            return ;
            break;
        }
        case 1: { // PlayerAction
            int v2 = v0.case1.v0; Union2 v3 = v0.case1.v1;
            printf("%s(%d, ","PlayerAction", v2);
            method_18(v3);
            printf(")");
            return ;
            break;
        }
        case 2: { // PlayerGotCard
            int v4 = v0.case2.v0; Union1 v5 = v0.case2.v1;
            printf("%s(%d, ","PlayerGotCard", v4);
            method_17(v5);
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
                method_17(v12);
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
    xso::rng v4;
    static_array_list<Union0,32> v6;
    v6 = static_array_list<Union0,32>{};
    static_array<Tuple2,2> v9;
    int v11;
    v11 = 0;
    while (while_method_1(v11)){
        v9[v11] = Tuple2{0.0f, 0.0f};
        v11 += 1 ;
    }
    unsigned int v13;
    v13 = 63u;
    StackRefs1 v14{v13, v9, v6};
    int v15;
    v15 = 0;
    while (while_method_2(v15)){
        int v17;
        v17 = v15 % 1000;
        bool v18;
        v18 = v17 == 0;
        if (v18){
            printf("{%s = %d; %s = %d}\n","i", v15, "nearTo", 1000000);
            fflush(stdout);
        } else {
        }
        Union3 v24;
        v24 = Union3{Union3_1{}};
        float v25;
        v25 = body_0(v3, v4, v14, v24);
        v15 += 1 ;
    }
    std::unordered_map<static_array_list<Union0,32>, static_array<float,3>, Fun0, Fun1> v26(512, v0, v1);
    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v27 = v3.v0;
    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v28 = v27;
    auto v29 = v28.begin();
    while (while_method_7(v28, v29)){
        static_array_list<Union0,32> v31;
        v31 = v29->first;
        static_array<float,3> v32; static_array<float,3> v33; static_array<Tuple2,3> v34;
        Tuple1 tmp30 = v29->second;
        v32 = tmp30.v0; v33 = tmp30.v1; v34 = tmp30.v2;
        static_array<float,3> v35;
        v35 = normalize_15(v32);
        v26[v31] = v35;
        ++v29;
    }
    printf("%s\n","{");
    std::unordered_map<static_array_list<Union0,32>, static_array<float,3>, Fun0, Fun1> & v63 = v26;
    auto v64 = v63.begin();
    while (while_method_8(v63, v64)){
        static_array_list<Union0,32> v66;
        v66 = v64->first;
        static_array<float,3> v67;
        v67 = v64->second;
        printf("%s","[");
        int v68;
        v68 = v66.length;
        bool v69;
        v69 = 100 < v68;
        int v70;
        if (v69){
            v70 = 100;
        } else {
            v70 = v68;
        }
        int v71;
        v71 = 0;
        while (while_method_0(v70, v71)){
            Union0 v74;
            v74 = v66[v71];
            printf("");
            method_16(v74);
            printf("");
            int v76;
            v76 = v71 + 1;
            int v77;
            v77 = v66.length;
            bool v78;
            v78 = v76 < v77;
            if (v78){
                printf("%s","; ");
            } else {
            }
            v71 += 1 ;
        }
        int v79;
        v79 = v66.length;
        bool v80;
        v80 = v79 > 100;
        if (v80){
            printf("%s","; ...");
        } else {
        }
        printf("%s","]");
        printf("");
        printf("%s"," => ");
        printf("%s","[");
        int v81;
        v81 = 0;
        while (while_method_4(v81)){
            float v84;
            v84 = v67[v81];
            printf("%f",v84);
            int v86;
            v86 = v81 + 1;
            bool v87;
            v87 = v86 < 3;
            if (v87){
                printf("%s","; ");
            } else {
            }
            v81 += 1 ;
        }
        printf("%s","]");
        printf("\n");
        ++v64;
    }
    printf("%s\n","}");
    printf("\n");
    fflush(stdout);
    return 0;
}
