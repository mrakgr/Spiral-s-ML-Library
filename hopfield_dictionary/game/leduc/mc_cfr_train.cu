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
unsigned int loop_3(unsigned int v0, xso::rng & v1);
struct StackMut1;
struct StackMut2;
unsigned int find_nth_set_bit_4(int v0, unsigned int v1, unsigned int v2);
Tuple3 draw_card_2(xso::rng & v0, unsigned int v1);
struct Union5;
struct Tuple4;
float loop_5(StackRefs0 & v0, xso::rng & v1, StackRefs1 & v2, StackMut0 & v3, Union5 v4);
struct Union6;
struct Union7;
static_array<float,3> relu_7(static_array<float,3> v0);
struct Tuple5;
static_array<float,3> masking_normalize_8(static_array<float,3> v0, static_array<bool,3> v1);
static_array<float,3> regret_match_6(static_array<float,3> v0, static_array<bool,3> v1);
int loop_11(static_array<float,3> v0, float v1, int v2);
int pick_discrete__10(static_array<float,3> v0, float v1);
int sample_discrete__9(static_array<float,3> v0, xso::rng & v1);
struct Union8;
int tag_13(Union1 v0);
bool is_pair_14(int v0, int v1);
Tuple4 order_15(int v0, int v1);
Union8 compare_hands_12(Union4 v0, bool v1, static_array<Union1,2> v2, int v3, static_array<int,2> v4, int v5);
float body_1(StackRefs0 & v0, xso::rng & v1, StackRefs1 & v2, Union3 v3);
void train_loop_0(StackRefs0 & v0);
static_array<float,3> normalize_16(static_array<float,3> v0);
void method_18(Union1 v0);
void method_19(Union2 v0);
void method_17(Union0 v0);
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
unsigned int loop_3(unsigned int v0, xso::rng & v1){
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
        return loop_3(v0, v1);
    }
}
inline bool while_method_3(unsigned int v0, unsigned int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
unsigned int find_nth_set_bit_4(int v0, unsigned int v1, unsigned int v2){
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
Tuple3 draw_card_2(xso::rng & v0, unsigned int v1){
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
    v8 = loop_3(v4, v0);
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
        v18 = find_nth_set_bit_4(v13, v14, v1);
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
float loop_5(StackRefs0 & v0, xso::rng & v1, StackRefs1 & v2, StackMut0 & v3, Union5 v4){
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
            return body_1(v0, v1, v2, v30);
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
            return body_1(v0, v1, v2, v44);
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
            return body_1(v0, v1, v2, v144);
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
static_array<float,3> masking_normalize_8(static_array<float,3> v0, static_array<bool,3> v1){
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
static_array<float,3> regret_match_6(static_array<float,3> v0, static_array<bool,3> v1){
    static_array<float,3> v2;
    v2 = relu_7(v0);
    return masking_normalize_8(v2, v1);
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
int loop_11(static_array<float,3> v0, float v1, int v2){
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
            return loop_11(v0, v1, v8);
        }
    } else {
        return 2;
    }
}
int pick_discrete__10(static_array<float,3> v0, float v1){
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
    return loop_11(v3, v26, v27);
}
int sample_discrete__9(static_array<float,3> v0, xso::rng & v1){
    std::uniform_real_distribution<float> v2(0.0, 1.0);
    float v3;
    v3 = v2(v1);
    return pick_discrete__10(v0, v3);
}
int tag_13(Union1 v0){
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
bool is_pair_14(int v0, int v1){
    bool v2;
    v2 = v1 == v0;
    return v2;
}
Tuple4 order_15(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple4{v1, v0};
    } else {
        return Tuple4{v0, v1};
    }
}
Union8 compare_hands_12(Union4 v0, bool v1, static_array<Union1,2> v2, int v3, static_array<int,2> v4, int v5){
    switch (v0.tag) {
        case 0: { // None
            printf("%s\n", "Expected the community card to be present in the table.");
            exit(-1);
            break;
        }
        case 1: { // Some
            Union1 v7 = v0.case1.v0;
            int v8;
            v8 = tag_13(v7);
            Union1 v10;
            v10 = v2[0];
            int v12;
            v12 = tag_13(v10);
            Union1 v14;
            v14 = v2[1];
            int v16;
            v16 = tag_13(v14);
            bool v17;
            v17 = is_pair_14(v8, v12);
            bool v18;
            v18 = is_pair_14(v8, v16);
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
                    Tuple4 tmp28 = order_15(v8, v12);
                    v29 = tmp28.v0; v30 = tmp28.v1;
                    int v31; int v32;
                    Tuple4 tmp29 = order_15(v8, v16);
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
float body_1(StackRefs0 & v0, xso::rng & v1, StackRefs1 & v2, Union3 v3){
    StackMut0 v4{0.0f};
    switch (v3.tag) {
        case 0: { // ChanceCommunityCard
            Union4 v464 = v3.case0.v0; bool v465 = v3.case0.v1; static_array<Union1,2> v466 = v3.case0.v2; int v467 = v3.case0.v3; static_array<int,2> v468 = v3.case0.v4; int v469 = v3.case0.v5;
            unsigned int & v470 = v2.v0;
            unsigned int v471;
            v471 = v470;
            Union1 v472; unsigned int v473;
            Tuple3 tmp2 = draw_card_2(v1, v471);
            v472 = tmp2.v0; v473 = tmp2.v1;
            v2.v0 = v473;
            static_array_list<Union0,32> & v474 = v2.v2;
            Union0 v475;
            v475 = Union0{Union0_0{v472}};
            v474.push(v475);
            Union5 v476;
            v476 = Union5{Union5_0{v464, v465, v466, v467, v468, v469, v472}};
            float v477;
            v477 = loop_5(v0, v1, v2, v4, v476);
            static_array_list<Union0,32> & v478 = v2.v2;
            Union0 v479;
            v479 = v478.pop();
            v2.v0 = v471;
            return v477;
            break;
        }
        case 1: { // ChanceInit
            unsigned int & v480 = v2.v0;
            unsigned int v481;
            v481 = v480;
            Union1 v482; unsigned int v483;
            Tuple3 tmp7 = draw_card_2(v1, v481);
            v482 = tmp7.v0; v483 = tmp7.v1;
            v2.v0 = v483;
            unsigned int & v484 = v2.v0;
            unsigned int v485;
            v485 = v484;
            Union1 v486; unsigned int v487;
            Tuple3 tmp8 = draw_card_2(v1, v485);
            v486 = tmp8.v0; v487 = tmp8.v1;
            v2.v0 = v487;
            static_array_list<Union0,32> & v488 = v2.v2;
            Union0 v489;
            v489 = Union0{Union0_2{0, v482}};
            v488.push(v489);
            static_array_list<Union0,32> & v490 = v2.v2;
            Union0 v491;
            v491 = Union0{Union0_2{1, v486}};
            v490.push(v491);
            Union5 v492;
            v492 = Union5{Union5_1{v482, v486}};
            float v493;
            v493 = loop_5(v0, v1, v2, v4, v492);
            static_array_list<Union0,32> & v494 = v2.v2;
            Union0 v495;
            v495 = v494.pop();
            static_array_list<Union0,32> & v496 = v2.v2;
            Union0 v497;
            v497 = v496.pop();
            v2.v0 = v485;
            v2.v0 = v481;
            return v493;
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
            v151 = regret_match_6(v123, v149);
            static_array<float,3> v153;
            int v155;
            v155 = 0;
            while (while_method_4(v155)){
                v153[v155] = 0.0f;
                v155 += 1 ;
            }
            static_array<float,3> v157;
            v157 = masking_normalize_8(v153, v149);
            static_array<float,3> v159;
            int v161;
            v161 = 0;
            while (while_method_4(v161)){
                float v164;
                v164 = v151[v161];
                float v167;
                v167 = v157[v161];
                float v169;
                v169 = 0.75f * v164;
                float v170;
                v170 = 0.25f * v167;
                float v171;
                v171 = v169 + v170;
                v159[v161] = v171;
                v161 += 1 ;
            }
            int v172;
            v172 = sample_discrete__9(v159, v1);
            StackMut0 v173{0.0f};
            float v239;
            switch (v141.tag) {
                case 1: { // Some
                    Union2 v174 = v141.case1.v0;
                    bool v175;
                    v175 = 2 == v172;
                    if (v175){
                        float v177;
                        v177 = v151[2];
                        float v180;
                        v180 = v159[2];
                        static_array<Tuple2,2> & v182 = v2.v1;
                        float v185; float v186;
                        Tuple2 tmp13 = v182[v58];
                        v185 = tmp13.v0; v186 = tmp13.v1;
                        static_array<Tuple2,2> & v189 = v2.v1;
                        float v190;
                        v190 = log(v177);
                        float v191;
                        v191 = v190 + v185;
                        float v192;
                        v192 = log(v180);
                        float v193;
                        v193 = v192 + v186;
                        v189[v58] = Tuple2{v191, v193};
                        static_array_list<Union0,32> & v194 = v2.v2;
                        Union0 v195;
                        v195 = Union0{Union0_1{v58, v174}};
                        v194.push(v195);
                        Union5 v196;
                        v196 = Union5{Union5_2{v55, v56, v57, v58, v59, v60, v174}};
                        float v197;
                        v197 = loop_5(v0, v1, v2, v4, v196);
                        static_array_list<Union0,32> & v198 = v2.v2;
                        Union0 v199;
                        v199 = v198.pop();
                        static_array<Tuple2,2> & v200 = v2.v1;
                        v200[v58] = Tuple2{v185, v186};
                        bool v201;
                        v201 = v58 == 0;
                        float v203;
                        if (v201){
                            v203 = v197;
                        } else {
                            float v202;
                            v202 = -v197;
                            v203 = v202;
                        }
                        v173.v0 = v203;
                        float v206; float v207;
                        Tuple2 tmp14 = v124[2];
                        v206 = tmp14.v0; v207 = tmp14.v1;
                        bool v210;
                        v210 = v207 == 0.0f;
                        bool v211;
                        v211 = v210 != true;
                        float v213;
                        if (v211){
                            float v212;
                            v212 = v206 / v207;
                            v213 = v212;
                        } else {
                            v213 = 0.0f;
                        }
                        float v214 = v173.v0;
                        float v215;
                        v215 = v214 - v213;
                        float v216;
                        v216 = v215 / v180;
                        float v217;
                        v217 = v216 + v213;
                        v239 = v217;
                    } else {
                        float v220; float v221;
                        Tuple2 tmp15 = v124[2];
                        v220 = tmp15.v0; v221 = tmp15.v1;
                        bool v224;
                        v224 = v221 == 0.0f;
                        bool v225;
                        v225 = v224 != true;
                        if (v225){
                            float v226;
                            v226 = v220 / v221;
                            v239 = v226;
                        } else {
                            v239 = 0.0f;
                        }
                    }
                    break;
                }
                default: {
                    float v231; float v232;
                    Tuple2 tmp16 = v124[2];
                    v231 = tmp16.v0; v232 = tmp16.v1;
                    bool v235;
                    v235 = v232 == 0.0f;
                    bool v236;
                    v236 = v235 != true;
                    if (v236){
                        float v237;
                        v237 = v231 / v232;
                        v239 = v237;
                    } else {
                        v239 = 0.0f;
                    }
                }
            }
            float v305;
            switch (v136.tag) {
                case 1: { // Some
                    Union2 v240 = v136.case1.v0;
                    bool v241;
                    v241 = 1 == v172;
                    if (v241){
                        float v243;
                        v243 = v151[1];
                        float v246;
                        v246 = v159[1];
                        static_array<Tuple2,2> & v248 = v2.v1;
                        float v251; float v252;
                        Tuple2 tmp17 = v248[v58];
                        v251 = tmp17.v0; v252 = tmp17.v1;
                        static_array<Tuple2,2> & v255 = v2.v1;
                        float v256;
                        v256 = log(v243);
                        float v257;
                        v257 = v256 + v251;
                        float v258;
                        v258 = log(v246);
                        float v259;
                        v259 = v258 + v252;
                        v255[v58] = Tuple2{v257, v259};
                        static_array_list<Union0,32> & v260 = v2.v2;
                        Union0 v261;
                        v261 = Union0{Union0_1{v58, v240}};
                        v260.push(v261);
                        Union5 v262;
                        v262 = Union5{Union5_2{v55, v56, v57, v58, v59, v60, v240}};
                        float v263;
                        v263 = loop_5(v0, v1, v2, v4, v262);
                        static_array_list<Union0,32> & v264 = v2.v2;
                        Union0 v265;
                        v265 = v264.pop();
                        static_array<Tuple2,2> & v266 = v2.v1;
                        v266[v58] = Tuple2{v251, v252};
                        bool v267;
                        v267 = v58 == 0;
                        float v269;
                        if (v267){
                            v269 = v263;
                        } else {
                            float v268;
                            v268 = -v263;
                            v269 = v268;
                        }
                        v173.v0 = v269;
                        float v272; float v273;
                        Tuple2 tmp18 = v124[1];
                        v272 = tmp18.v0; v273 = tmp18.v1;
                        bool v276;
                        v276 = v273 == 0.0f;
                        bool v277;
                        v277 = v276 != true;
                        float v279;
                        if (v277){
                            float v278;
                            v278 = v272 / v273;
                            v279 = v278;
                        } else {
                            v279 = 0.0f;
                        }
                        float v280 = v173.v0;
                        float v281;
                        v281 = v280 - v279;
                        float v282;
                        v282 = v281 / v246;
                        float v283;
                        v283 = v282 + v279;
                        v305 = v283;
                    } else {
                        float v286; float v287;
                        Tuple2 tmp19 = v124[1];
                        v286 = tmp19.v0; v287 = tmp19.v1;
                        bool v290;
                        v290 = v287 == 0.0f;
                        bool v291;
                        v291 = v290 != true;
                        if (v291){
                            float v292;
                            v292 = v286 / v287;
                            v305 = v292;
                        } else {
                            v305 = 0.0f;
                        }
                    }
                    break;
                }
                default: {
                    float v297; float v298;
                    Tuple2 tmp20 = v124[1];
                    v297 = tmp20.v0; v298 = tmp20.v1;
                    bool v301;
                    v301 = v298 == 0.0f;
                    bool v302;
                    v302 = v301 != true;
                    if (v302){
                        float v303;
                        v303 = v297 / v298;
                        v305 = v303;
                    } else {
                        v305 = 0.0f;
                    }
                }
            }
            bool v306;
            v306 = 0 == v172;
            float v361;
            if (v306){
                float v308;
                v308 = v151[0];
                float v311;
                v311 = v159[0];
                static_array<Tuple2,2> & v313 = v2.v1;
                float v316; float v317;
                Tuple2 tmp21 = v313[v58];
                v316 = tmp21.v0; v317 = tmp21.v1;
                static_array<Tuple2,2> & v320 = v2.v1;
                float v321;
                v321 = log(v308);
                float v322;
                v322 = v321 + v316;
                float v323;
                v323 = log(v311);
                float v324;
                v324 = v323 + v317;
                v320[v58] = Tuple2{v322, v324};
                static_array_list<Union0,32> & v325 = v2.v2;
                Union2 v326;
                v326 = Union2{Union2_0{}};
                Union0 v327;
                v327 = Union0{Union0_1{v58, v326}};
                v325.push(v327);
                Union2 v328;
                v328 = Union2{Union2_0{}};
                Union5 v329;
                v329 = Union5{Union5_2{v55, v56, v57, v58, v59, v60, v328}};
                float v330;
                v330 = loop_5(v0, v1, v2, v4, v329);
                static_array_list<Union0,32> & v331 = v2.v2;
                Union0 v332;
                v332 = v331.pop();
                static_array<Tuple2,2> & v333 = v2.v1;
                v333[v58] = Tuple2{v316, v317};
                bool v334;
                v334 = v58 == 0;
                float v336;
                if (v334){
                    v336 = v330;
                } else {
                    float v335;
                    v335 = -v330;
                    v336 = v335;
                }
                v173.v0 = v336;
                float v339; float v340;
                Tuple2 tmp22 = v124[0];
                v339 = tmp22.v0; v340 = tmp22.v1;
                bool v343;
                v343 = v340 == 0.0f;
                bool v344;
                v344 = v343 != true;
                float v346;
                if (v344){
                    float v345;
                    v345 = v339 / v340;
                    v346 = v345;
                } else {
                    v346 = 0.0f;
                }
                float v347 = v173.v0;
                float v348;
                v348 = v347 - v346;
                float v349;
                v349 = v348 / v311;
                float v350;
                v350 = v349 + v346;
                v361 = v350;
            } else {
                float v353; float v354;
                Tuple2 tmp23 = v124[0];
                v353 = tmp23.v0; v354 = tmp23.v1;
                bool v357;
                v357 = v354 == 0.0f;
                bool v358;
                v358 = v357 != true;
                if (v358){
                    float v359;
                    v359 = v353 / v354;
                    v361 = v359;
                } else {
                    v361 = 0.0f;
                }
            }
            static_array<float,3> v363;
            v363[0] = v361;
            v363[1] = v305;
            v363[2] = v239;
            static_array<float,3> v366;
            int v368;
            v368 = 0;
            while (while_method_4(v368)){
                float v371;
                v371 = v363[v368];
                float v374;
                v374 = v151[v368];
                float v376;
                v376 = v371 * v374;
                v366[v368] = v376;
                v368 += 1 ;
            }
            int v377; float v378;
            Tuple5 tmp24 = Tuple5{0, 0.0f};
            v377 = tmp24.v0; v378 = tmp24.v1;
            while (while_method_4(v377)){
                float v381;
                v381 = v366[v377];
                float v383;
                v383 = v378 + v381;
                v378 = v383;
                v377 += 1 ;
            }
            std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v384 = v0.v0;
            static_array<float,3> v386;
            int v388;
            v388 = 0;
            while (while_method_4(v388)){
                float v391;
                v391 = v122[v388];
                float v394;
                v394 = v151[v388];
                float v396;
                v396 = 0.99902344f * v391;
                float v397;
                v397 = v396 + v394;
                v386[v388] = v397;
                v388 += 1 ;
            }
            static_array<Tuple2,2> & v398 = v2.v1;
            int v399; float v400;
            Tuple5 tmp25 = Tuple5{0, 0.0f};
            v399 = tmp25.v0; v400 = tmp25.v1;
            while (while_method_1(v399)){
                float v404; float v405;
                Tuple2 tmp26 = v398[v399];
                v404 = tmp26.v0; v405 = tmp26.v1;
                bool v408;
                v408 = v399 == v58;
                float v409;
                if (v408){
                    v409 = 0.0f;
                } else {
                    v409 = v404;
                }
                float v410;
                v410 = v400 + v409;
                float v411;
                v411 = v410 - v405;
                v400 = v411;
                v399 += 1 ;
            }
            float v412;
            v412 = exp(v400);
            static_array<float,3> v414;
            int v416;
            v416 = 0;
            while (while_method_4(v416)){
                float v419;
                v419 = v123[v416];
                float v422;
                v422 = v363[v416];
                float v424;
                v424 = v422 - v378;
                float v425;
                v425 = v412 * v424;
                float v426;
                v426 = v419 + v425;
                bool v427;
                v427 = 0.0f >= v426;
                float v428;
                if (v427){
                    v428 = 0.0f;
                } else {
                    v428 = v426;
                }
                v414[v416] = v428;
                v416 += 1 ;
            }
            static_array<Tuple2,3> v430;
            int v432;
            v432 = 0;
            while (while_method_4(v432)){
                float v436; float v437;
                Tuple2 tmp27 = v124[v432];
                v436 = tmp27.v0; v437 = tmp27.v1;
                bool v440;
                v440 = v172 == v432;
                float v444; float v445;
                if (v440){
                    float v441 = v173.v0;
                    float v442;
                    v442 = v441 + v436;
                    float v443;
                    v443 = 1.0f + v437;
                    v444 = v442; v445 = v443;
                } else {
                    v444 = v436; v445 = v437;
                }
                v430[v432] = Tuple2{v444, v445};
                v432 += 1 ;
            }
            v384[v85] = Tuple1{v386, v414, v430};
            bool v446;
            v446 = v58 == 0;
            float v448;
            if (v446){
                v448 = v378;
            } else {
                float v447;
                v447 = -v378;
                v448 = v447;
            }
            v4.v0 = v448;
            Union5 v449;
            v449 = Union5{Union5_3{}};
            return loop_5(v0, v1, v2, v4, v449);
            break;
        }
        case 3: { // RoundWithAction
            Union4 v451 = v3.case3.v0; bool v452 = v3.case3.v1; static_array<Union1,2> v453 = v3.case3.v2; int v454 = v3.case3.v3; static_array<int,2> v455 = v3.case3.v4; int v456 = v3.case3.v5; Union2 v457 = v3.case3.v6;
            static_array_list<Union0,32> & v458 = v2.v2;
            Union0 v459;
            v459 = Union0{Union0_1{v454, v457}};
            v458.push(v459);
            Union5 v460;
            v460 = Union5{Union5_2{v451, v452, v453, v454, v455, v456, v457}};
            float v461;
            v461 = loop_5(v0, v1, v2, v4, v460);
            static_array_list<Union0,32> & v462 = v2.v2;
            Union0 v463;
            v463 = v462.pop();
            return v461;
            break;
        }
        case 4: { // TerminalCall
            Union4 v26 = v3.case4.v0; bool v27 = v3.case4.v1; static_array<Union1,2> v28 = v3.case4.v2; int v29 = v3.case4.v3; static_array<int,2> v30 = v3.case4.v4; int v31 = v3.case4.v5;
            int v33;
            v33 = v30[v29];
            Union8 v35;
            v35 = compare_hands_12(v26, v27, v28, v29, v30, v31);
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
            v52 = loop_5(v0, v1, v2, v4, v51);
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
            v23 = loop_5(v0, v1, v2, v4, v22);
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
void train_loop_0(StackRefs0 & v0){
    xso::rng v1;
    static_array_list<Union0,32> v3;
    v3 = static_array_list<Union0,32>{};
    static_array<Tuple2,2> v6;
    int v8;
    v8 = 0;
    while (while_method_1(v8)){
        v6[v8] = Tuple2{0.0f, 0.0f};
        v8 += 1 ;
    }
    unsigned int v10;
    v10 = 63u;
    StackRefs1 v11{v10, v6, v3};
    Union3 v12;
    v12 = Union3{Union3_1{}};
    float v13;
    v13 = body_1(v0, v1, v11, v12);
    return ;
}
inline bool while_method_7(std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v0, std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1>::iterator & v1){
    bool v2;
    v2 = v1 != v0.end();
    return v2;
}
static_array<float,3> normalize_16(static_array<float,3> v0){
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
void method_18(Union1 v0){
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
void method_19(Union2 v0){
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
void method_17(Union0 v0){
    switch (v0.tag) {
        case 0: { // CommunityCardIs
            Union1 v1 = v0.case0.v0;
            printf("%s(","CommunityCardIs");
            method_18(v1);
            printf(")");
            return ;
            break;
        }
        case 1: { // PlayerAction
            int v2 = v0.case1.v0; Union2 v3 = v0.case1.v1;
            printf("%s(%d, ","PlayerAction", v2);
            method_19(v3);
            printf(")");
            return ;
            break;
        }
        case 2: { // PlayerGotCard
            int v4 = v0.case2.v0; Union1 v5 = v0.case2.v1;
            printf("%s(%d, ","PlayerGotCard", v4);
            method_18(v5);
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
                method_18(v12);
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
    int v4;
    v4 = 0;
    while (while_method_2(v4)){
        int v6;
        v6 = v4 % 1000;
        bool v7;
        v7 = v6 == 0;
        if (v7){
            printf("{%s = %d; %s = %d}\n","i", v4, "nearTo", 1000000);
            fflush(stdout);
        } else {
        }
        train_loop_0(v3);
        v4 += 1 ;
    }
    std::unordered_map<static_array_list<Union0,32>, static_array<float,3>, Fun0, Fun1> v13(512, v0, v1);
    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v14 = v3.v0;
    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v15 = v14;
    auto v16 = v15.begin();
    while (while_method_7(v15, v16)){
        static_array_list<Union0,32> v18;
        v18 = v16->first;
        static_array<float,3> v19; static_array<float,3> v20; static_array<Tuple2,3> v21;
        Tuple1 tmp30 = v16->second;
        v19 = tmp30.v0; v20 = tmp30.v1; v21 = tmp30.v2;
        static_array<float,3> v22;
        v22 = normalize_16(v19);
        v13[v18] = v22;
        ++v16;
    }
    printf("%s\n","{");
    std::unordered_map<static_array_list<Union0,32>, static_array<float,3>, Fun0, Fun1> & v50 = v13;
    auto v51 = v50.begin();
    while (while_method_8(v50, v51)){
        static_array_list<Union0,32> v53;
        v53 = v51->first;
        static_array<float,3> v54;
        v54 = v51->second;
        printf("%s","[");
        int v55;
        v55 = v53.length;
        bool v56;
        v56 = 100 < v55;
        int v57;
        if (v56){
            v57 = 100;
        } else {
            v57 = v55;
        }
        int v58;
        v58 = 0;
        while (while_method_0(v57, v58)){
            Union0 v61;
            v61 = v53[v58];
            printf("");
            method_17(v61);
            printf("");
            int v63;
            v63 = v58 + 1;
            int v64;
            v64 = v53.length;
            bool v65;
            v65 = v63 < v64;
            if (v65){
                printf("%s","; ");
            } else {
            }
            v58 += 1 ;
        }
        int v66;
        v66 = v53.length;
        bool v67;
        v67 = v66 > 100;
        if (v67){
            printf("%s","; ...");
        } else {
        }
        printf("%s","]");
        printf("");
        printf("%s"," => ");
        printf("%s","[");
        int v68;
        v68 = 0;
        while (while_method_4(v68)){
            float v71;
            v71 = v54[v68];
            printf("%f",v71);
            int v73;
            v73 = v68 + 1;
            bool v74;
            v74 = v73 < 3;
            if (v74){
                printf("%s","; ");
            } else {
            }
            v68 += 1 ;
        }
        printf("%s","]");
        printf("\n");
        ++v51;
    }
    printf("%s\n","}");
    printf("\n");
    fflush(stdout);
    return 0;
}
