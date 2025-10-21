#include "enumerative_cfr_train.auto.cu"
#include <thrust/device_vector.h>
#include <unordered_map>
struct Union1;
struct Union2;
struct Union0;
struct Tuple0;
typedef unsigned long long (* Fun0)(static_array_list<Union0,32>);
typedef bool (* Fun1)(static_array_list<Union0,32>, static_array_list<Union0,32>);
struct Tuple1;
struct StackRefs0;
struct StackRefs1;
struct Union4;
struct Union3;
struct StackMut0;
struct Tuple2;
struct Union5;
struct Tuple3;
float loop_2(StackRefs0 & v0, StackRefs1 & v1, StackMut0 & v2, Union5 v3);
struct Union6;
struct Union7;
static_array<float,3> relu_4(static_array<float,3> v0);
struct Tuple4;
static_array<float,3> masking_normalize_5(static_array<float,3> v0, static_array<bool,3> v1);
static_array<float,3> regret_match_3(static_array<float,3> v0, static_array<bool,3> v1);
struct Union8;
int tag_7(Union1 v0);
bool is_pair_8(int v0, int v1);
Tuple3 order_9(int v0, int v1);
Union8 compare_hands_6(Union4 v0, bool v1, static_array<Union1,2> v2, int v3, static_array<int,2> v4, int v5);
float body_1(StackRefs0 & v0, StackRefs1 & v1, Union3 v2);
void train_loop_0(StackRefs0 & v0);
static_array<float,3> normalize_10(static_array<float,3> v0);
void method_12(Union1 v0);
void method_13(Union2 v0);
void method_11(Union0 v0);
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
struct Tuple1 {
    static_array<float,3> v0;
    static_array<float,3> v1;
    __host__ __device__ Tuple1() = default;
    __host__ __device__ Tuple1(static_array<float,3> t0, static_array<float,3> t1) : v0(t0), v1(t1) {}
};
struct StackRefs0 {
    std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v0;
    __host__ __device__ StackRefs0() = default;
    __host__ __device__ StackRefs0(std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & t0) : v0(t0) {}
};
struct StackRefs1 {
    static_array<float,2> & v1;
    static_array_list<Union0,32> & v2;
    unsigned int & v0;
    __host__ __device__ StackRefs1() = default;
    __host__ __device__ StackRefs1(unsigned int & t0, static_array<float,2> & t1, static_array_list<Union0,32> & t2) : v0(t0), v1(t1), v2(t2) {}
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
struct Tuple2 {
    int v0;
    float v1;
    float v2;
    __host__ __device__ Tuple2() = default;
    __host__ __device__ Tuple2(int t0, float t1, float t2) : v0(t0), v1(t1), v2(t2) {}
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
struct Tuple3 {
    int v0;
    int v1;
    __host__ __device__ Tuple3() = default;
    __host__ __device__ Tuple3(int t0, int t1) : v0(t0), v1(t1) {}
};
struct Union6_0 { // None
};
struct Union6_1 { // Some
    static_array<float,3> v0;
    static_array<float,3> v1;
    __host__ __device__ Union6_1(static_array<float,3> t0, static_array<float,3> t1) : v0(t0), v1(t1) {}
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
struct Tuple4 {
    int v0;
    float v1;
    __host__ __device__ Tuple4() = default;
    __host__ __device__ Tuple4(int t0, float t1) : v0(t0), v1(t1) {}
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
    v1 = v0 < 1000;
    return v1;
}
inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 6;
    return v1;
}
float loop_2(StackRefs0 & v0, StackRefs1 & v1, StackMut0 & v2, Union5 v3){
    switch (v3.tag) {
        case 0: { // T_game_chance_community_card
            Union4 v5 = v3.case0.v0; bool v6 = v3.case0.v1; static_array<Union1,2> v7 = v3.case0.v2; int v8 = v3.case0.v3; static_array<int,2> v9 = v3.case0.v4; int v10 = v3.case0.v5; Union1 v11 = v3.case0.v6;
            int v12;
            v12 = 2;
            int v13; int v14;
            Tuple3 tmp3 = Tuple3{0, 0};
            v13 = tmp3.v0; v14 = tmp3.v1;
            while (while_method_1(v13)){
                int v17;
                v17 = v9[v13];
                bool v19;
                v19 = v14 >= v17;
                int v20;
                if (v19){
                    v20 = v14;
                } else {
                    v20 = v17;
                }
                v14 = v20;
                v13 += 1 ;
            }
            static_array<int,2> v22;
            int v24;
            v24 = 0;
            while (while_method_1(v24)){
                v22[v24] = v14;
                v24 += 1 ;
            }
            Union4 v26;
            v26 = Union4{Union4_1{v11}};
            bool v27;
            v27 = true;
            int v28;
            v28 = 0;
            Union3 v29;
            v29 = Union3{Union3_2{v26, v27, v7, v28, v22, v12}};
            return body_1(v0, v1, v29);
            break;
        }
        case 1: { // T_game_chance_init
            Union1 v31 = v3.case1.v0; Union1 v32 = v3.case1.v1;
            int v33;
            v33 = 2;
            static_array<int,2> v35;
            v35[0] = 1;
            v35[1] = 1;
            static_array<Union1,2> v38;
            v38[0] = v31;
            v38[1] = v32;
            Union4 v40;
            v40 = Union4{Union4_0{}};
            bool v41;
            v41 = true;
            int v42;
            v42 = 0;
            Union3 v43;
            v43 = Union3{Union3_2{v40, v41, v38, v42, v35, v33}};
            return body_1(v0, v1, v43);
            break;
        }
        case 2: { // T_game_round
            Union4 v45 = v3.case2.v0; bool v46 = v3.case2.v1; static_array<Union1,2> v47 = v3.case2.v2; int v48 = v3.case2.v3; static_array<int,2> v49 = v3.case2.v4; int v50 = v3.case2.v5; Union2 v51 = v3.case2.v6;
            Union3 v143;
            switch (v45.tag) {
                case 0: { // None
                    switch (v51.tag) {
                        case 0: { // Call
                            if (v46){
                                int v105;
                                v105 = v48 ^ 1;
                                v143 = Union3{Union3_2{v45, false, v47, v105, v49, v50}};
                            } else {
                                v143 = Union3{Union3_0{v45, v46, v47, v48, v49, v50}};
                            }
                            break;
                        }
                        case 1: { // Fold
                            v143 = Union3{Union3_5{v45, v46, v47, v48, v49, v50}};
                            break;
                        }
                        case 2: { // Raise
                            bool v109;
                            v109 = v50 > 0;
                            if (v109){
                                int v110;
                                v110 = v48 ^ 1;
                                int v111;
                                v111 = -1 + v50;
                                int v112; int v113;
                                Tuple3 tmp4 = Tuple3{0, 0};
                                v112 = tmp4.v0; v113 = tmp4.v1;
                                while (while_method_1(v112)){
                                    int v116;
                                    v116 = v49[v112];
                                    bool v118;
                                    v118 = v113 >= v116;
                                    int v119;
                                    if (v118){
                                        v119 = v113;
                                    } else {
                                        v119 = v116;
                                    }
                                    v113 = v119;
                                    v112 += 1 ;
                                }
                                static_array<int,2> v121;
                                int v123;
                                v123 = 0;
                                while (while_method_1(v123)){
                                    v121[v123] = v113;
                                    v123 += 1 ;
                                }
                                static_array<int,2> v126;
                                int v128;
                                v128 = 0;
                                while (while_method_1(v128)){
                                    int v131;
                                    v131 = v121[v128];
                                    bool v133;
                                    v133 = v128 == v48;
                                    int v135;
                                    if (v133){
                                        int v134;
                                        v134 = v131 + 2;
                                        v135 = v134;
                                    } else {
                                        v135 = v131;
                                    }
                                    v126[v128] = v135;
                                    v128 += 1 ;
                                }
                                v143 = Union3{Union3_2{v45, false, v47, v110, v126, v111}};
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
                    Union1 v52 = v45.case1.v0;
                    switch (v51.tag) {
                        case 0: { // Call
                            if (v46){
                                int v54;
                                v54 = v48 ^ 1;
                                v143 = Union3{Union3_2{v45, false, v47, v54, v49, v50}};
                            } else {
                                int v56; int v57;
                                Tuple3 tmp5 = Tuple3{0, 0};
                                v56 = tmp5.v0; v57 = tmp5.v1;
                                while (while_method_1(v56)){
                                    int v60;
                                    v60 = v49[v56];
                                    bool v62;
                                    v62 = v57 >= v60;
                                    int v63;
                                    if (v62){
                                        v63 = v57;
                                    } else {
                                        v63 = v60;
                                    }
                                    v57 = v63;
                                    v56 += 1 ;
                                }
                                static_array<int,2> v65;
                                int v67;
                                v67 = 0;
                                while (while_method_1(v67)){
                                    v65[v67] = v57;
                                    v67 += 1 ;
                                }
                                v143 = Union3{Union3_4{v45, v46, v47, v48, v65, v50}};
                            }
                            break;
                        }
                        case 1: { // Fold
                            v143 = Union3{Union3_5{v45, v46, v47, v48, v49, v50}};
                            break;
                        }
                        case 2: { // Raise
                            bool v71;
                            v71 = v50 > 0;
                            if (v71){
                                int v72;
                                v72 = v48 ^ 1;
                                int v73;
                                v73 = -1 + v50;
                                int v74; int v75;
                                Tuple3 tmp6 = Tuple3{0, 0};
                                v74 = tmp6.v0; v75 = tmp6.v1;
                                while (while_method_1(v74)){
                                    int v78;
                                    v78 = v49[v74];
                                    bool v80;
                                    v80 = v75 >= v78;
                                    int v81;
                                    if (v80){
                                        v81 = v75;
                                    } else {
                                        v81 = v78;
                                    }
                                    v75 = v81;
                                    v74 += 1 ;
                                }
                                static_array<int,2> v83;
                                int v85;
                                v85 = 0;
                                while (while_method_1(v85)){
                                    v83[v85] = v75;
                                    v85 += 1 ;
                                }
                                static_array<int,2> v88;
                                int v90;
                                v90 = 0;
                                while (while_method_1(v90)){
                                    int v93;
                                    v93 = v83[v90];
                                    bool v95;
                                    v95 = v90 == v48;
                                    int v97;
                                    if (v95){
                                        int v96;
                                        v96 = v93 + 4;
                                        v97 = v96;
                                    } else {
                                        v97 = v93;
                                    }
                                    v88[v90] = v97;
                                    v90 += 1 ;
                                }
                                v143 = Union3{Union3_2{v45, false, v47, v72, v88, v73}};
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
            return body_1(v0, v1, v143);
            break;
        }
        case 3: { // T_none
            float v4 = v2.v0;
            return v4;
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
static_array<float,3> masking_normalize_5(static_array<float,3> v0, static_array<bool,3> v1){
    int v2; float v3;
    Tuple4 tmp11 = Tuple4{0, 0.0f};
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
    Tuple4 tmp12 = Tuple4{0, 0.0f};
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
static_array<float,3> regret_match_3(static_array<float,3> v0, static_array<bool,3> v1){
    static_array<float,3> v2;
    v2 = relu_4(v0);
    return masking_normalize_5(v2, v1);
}
int tag_7(Union1 v0){
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
bool is_pair_8(int v0, int v1){
    bool v2;
    v2 = v1 == v0;
    return v2;
}
Tuple3 order_9(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple3{v1, v0};
    } else {
        return Tuple3{v0, v1};
    }
}
Union8 compare_hands_6(Union4 v0, bool v1, static_array<Union1,2> v2, int v3, static_array<int,2> v4, int v5){
    switch (v0.tag) {
        case 0: { // None
            printf("%s\n", "Expected the community card to be present in the table.");
            exit(-1);
            break;
        }
        case 1: { // Some
            Union1 v7 = v0.case1.v0;
            int v8;
            v8 = tag_7(v7);
            Union1 v10;
            v10 = v2[0];
            int v12;
            v12 = tag_7(v10);
            Union1 v14;
            v14 = v2[1];
            int v16;
            v16 = tag_7(v14);
            bool v17;
            v17 = is_pair_8(v8, v12);
            bool v18;
            v18 = is_pair_8(v8, v16);
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
                    Tuple3 tmp14 = order_9(v8, v12);
                    v29 = tmp14.v0; v30 = tmp14.v1;
                    int v31; int v32;
                    Tuple3 tmp15 = order_9(v8, v16);
                    v31 = tmp15.v0; v32 = tmp15.v1;
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
float body_1(StackRefs0 & v0, StackRefs1 & v1, Union3 v2){
    StackMut0 v3{0.0f};
    switch (v2.tag) {
        case 0: { // ChanceCommunityCard
            Union4 v291 = v2.case0.v0; bool v292 = v2.case0.v1; static_array<Union1,2> v293 = v2.case0.v2; int v294 = v2.case0.v3; static_array<int,2> v295 = v2.case0.v4; int v296 = v2.case0.v5;
            int v297; float v298; float v299;
            Tuple2 tmp2 = Tuple2{0, 0.0f, 0.0f};
            v297 = tmp2.v0; v298 = tmp2.v1; v299 = tmp2.v2;
            while (while_method_3(v297)){
                unsigned int & v301 = v1.v0;
                unsigned int v302;
                v302 = 1u << v297;
                unsigned int v303;
                v303 = v301 & v302;
                bool v304;
                v304 = v303 == 0u;
                bool v305;
                v305 = v304 != true;
                float v337; float v338;
                if (v305){
                    unsigned int & v306 = v1.v0;
                    unsigned int v307;
                    v307 = v306 ^ v302;
                    v1.v0 = v307;
                    bool v308;
                    v308 = 0 == v297;
                    Union1 v326;
                    if (v308){
                        v326 = Union1{Union1_1{}};
                    } else {
                        bool v310;
                        v310 = 1 == v297;
                        if (v310){
                            v326 = Union1{Union1_1{}};
                        } else {
                            bool v312;
                            v312 = 2 == v297;
                            if (v312){
                                v326 = Union1{Union1_2{}};
                            } else {
                                bool v314;
                                v314 = 3 == v297;
                                if (v314){
                                    v326 = Union1{Union1_2{}};
                                } else {
                                    bool v316;
                                    v316 = 4 == v297;
                                    if (v316){
                                        v326 = Union1{Union1_0{}};
                                    } else {
                                        bool v318;
                                        v318 = 5 == v297;
                                        if (v318){
                                            v326 = Union1{Union1_0{}};
                                        } else {
                                            printf("%s\n", "Invalid int in int_to_card.");
                                            exit(-1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    static_array_list<Union0,32> & v327 = v1.v2;
                    Union0 v328;
                    v328 = Union0{Union0_0{v326}};
                    v327.push(v328);
                    Union5 v329;
                    v329 = Union5{Union5_0{v291, v292, v293, v294, v295, v296, v326}};
                    float v330;
                    v330 = loop_2(v0, v1, v3, v329);
                    static_array_list<Union0,32> & v331 = v1.v2;
                    Union0 v332;
                    v332 = v331.pop();
                    unsigned int & v333 = v1.v0;
                    unsigned int v334;
                    v334 = v333 ^ v302;
                    v1.v0 = v334;
                    float v335;
                    v335 = v298 + v330;
                    float v336;
                    v336 = v299 + 1.0f;
                    v337 = v335; v338 = v336;
                } else {
                    v337 = v298; v338 = v299;
                }
                v298 = v337;
                v299 = v338;
                v297 += 1 ;
            }
            bool v339;
            v339 = v299 == 0.0f;
            bool v340;
            v340 = v339 != true;
            if (v340){
                float v341;
                v341 = v298 / v299;
                return v341;
            } else {
                return 0.0f;
            }
            break;
        }
        case 1: { // ChanceInit
            int v343; float v344; float v345;
            Tuple2 tmp7 = Tuple2{0, 0.0f, 0.0f};
            v343 = tmp7.v0; v344 = tmp7.v1; v345 = tmp7.v2;
            while (while_method_3(v343)){
                unsigned int & v347 = v1.v0;
                unsigned int v348;
                v348 = 1u << v343;
                unsigned int v349;
                v349 = v347 & v348;
                bool v350;
                v350 = v349 == 0u;
                bool v351;
                v351 = v350 != true;
                float v427; float v428;
                if (v351){
                    unsigned int & v352 = v1.v0;
                    unsigned int v353;
                    v353 = v352 ^ v348;
                    v1.v0 = v353;
                    bool v354;
                    v354 = 0 == v343;
                    Union1 v372;
                    if (v354){
                        v372 = Union1{Union1_1{}};
                    } else {
                        bool v356;
                        v356 = 1 == v343;
                        if (v356){
                            v372 = Union1{Union1_1{}};
                        } else {
                            bool v358;
                            v358 = 2 == v343;
                            if (v358){
                                v372 = Union1{Union1_2{}};
                            } else {
                                bool v360;
                                v360 = 3 == v343;
                                if (v360){
                                    v372 = Union1{Union1_2{}};
                                } else {
                                    bool v362;
                                    v362 = 4 == v343;
                                    if (v362){
                                        v372 = Union1{Union1_0{}};
                                    } else {
                                        bool v364;
                                        v364 = 5 == v343;
                                        if (v364){
                                            v372 = Union1{Union1_0{}};
                                        } else {
                                            printf("%s\n", "Invalid int in int_to_card.");
                                            exit(-1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    int v373; float v374; float v375;
                    Tuple2 tmp8 = Tuple2{0, 0.0f, 0.0f};
                    v373 = tmp8.v0; v374 = tmp8.v1; v375 = tmp8.v2;
                    while (while_method_3(v373)){
                        unsigned int & v377 = v1.v0;
                        unsigned int v378;
                        v378 = 1u << v373;
                        unsigned int v379;
                        v379 = v377 & v378;
                        bool v380;
                        v380 = v379 == 0u;
                        bool v381;
                        v381 = v380 != true;
                        float v417; float v418;
                        if (v381){
                            unsigned int & v382 = v1.v0;
                            unsigned int v383;
                            v383 = v382 ^ v378;
                            v1.v0 = v383;
                            bool v384;
                            v384 = 0 == v373;
                            Union1 v402;
                            if (v384){
                                v402 = Union1{Union1_1{}};
                            } else {
                                bool v386;
                                v386 = 1 == v373;
                                if (v386){
                                    v402 = Union1{Union1_1{}};
                                } else {
                                    bool v388;
                                    v388 = 2 == v373;
                                    if (v388){
                                        v402 = Union1{Union1_2{}};
                                    } else {
                                        bool v390;
                                        v390 = 3 == v373;
                                        if (v390){
                                            v402 = Union1{Union1_2{}};
                                        } else {
                                            bool v392;
                                            v392 = 4 == v373;
                                            if (v392){
                                                v402 = Union1{Union1_0{}};
                                            } else {
                                                bool v394;
                                                v394 = 5 == v373;
                                                if (v394){
                                                    v402 = Union1{Union1_0{}};
                                                } else {
                                                    printf("%s\n", "Invalid int in int_to_card.");
                                                    exit(-1);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            static_array_list<Union0,32> & v403 = v1.v2;
                            Union0 v404;
                            v404 = Union0{Union0_2{0, v372}};
                            v403.push(v404);
                            static_array_list<Union0,32> & v405 = v1.v2;
                            Union0 v406;
                            v406 = Union0{Union0_2{1, v402}};
                            v405.push(v406);
                            Union5 v407;
                            v407 = Union5{Union5_1{v372, v402}};
                            float v408;
                            v408 = loop_2(v0, v1, v3, v407);
                            static_array_list<Union0,32> & v409 = v1.v2;
                            Union0 v410;
                            v410 = v409.pop();
                            static_array_list<Union0,32> & v411 = v1.v2;
                            Union0 v412;
                            v412 = v411.pop();
                            unsigned int & v413 = v1.v0;
                            unsigned int v414;
                            v414 = v413 ^ v378;
                            v1.v0 = v414;
                            float v415;
                            v415 = v374 + v408;
                            float v416;
                            v416 = v375 + 1.0f;
                            v417 = v415; v418 = v416;
                        } else {
                            v417 = v374; v418 = v375;
                        }
                        v374 = v417;
                        v375 = v418;
                        v373 += 1 ;
                    }
                    bool v419;
                    v419 = v375 == 0.0f;
                    bool v420;
                    v420 = v419 != true;
                    float v422;
                    if (v420){
                        float v421;
                        v421 = v374 / v375;
                        v422 = v421;
                    } else {
                        v422 = 0.0f;
                    }
                    unsigned int & v423 = v1.v0;
                    unsigned int v424;
                    v424 = v423 ^ v348;
                    v1.v0 = v424;
                    float v425;
                    v425 = v344 + v422;
                    float v426;
                    v426 = v345 + 1.0f;
                    v427 = v425; v428 = v426;
                } else {
                    v427 = v344; v428 = v345;
                }
                v344 = v427;
                v345 = v428;
                v343 += 1 ;
            }
            bool v429;
            v429 = v345 == 0.0f;
            bool v430;
            v430 = v429 != true;
            if (v430){
                float v431;
                v431 = v344 / v345;
                return v431;
            } else {
                return 0.0f;
            }
            break;
        }
        case 2: { // Round
            Union4 v54 = v2.case2.v0; bool v55 = v2.case2.v1; static_array<Union1,2> v56 = v2.case2.v2; int v57 = v2.case2.v3; static_array<int,2> v58 = v2.case2.v4; int v59 = v2.case2.v5;
            static_array_list<Union0,32> & v60 = v1.v2;
            int v61;
            v61 = v60.length;
            bool v62;
            v62 = 32 >= v61;
            bool v63;
            v63 = v62 == false;
            if (v63){
                assert("The type level dimension has to equal the value passed at runtime into create." && v62);
            } else {
            }
            static_array_list<Union0,32> v66;
            v66 = static_array_list<Union0,32>{};
            v66.unsafe_set_length(v61);
            int v68; int v69;
            Tuple3 tmp9 = Tuple3{0, 0};
            v68 = tmp9.v0; v69 = tmp9.v1;
            while (while_method_0(v61, v68)){
                Union0 v72;
                v72 = v60[v68];
                bool v77;
                switch (v72.tag) {
                    case 2: { // PlayerGotCard
                        int v74 = v72.case2.v0; Union1 v75 = v72.case2.v1;
                        bool v76;
                        v76 = v74 == v57;
                        v77 = v76;
                        break;
                    }
                    default: {
                        v77 = true;
                    }
                }
                int v79;
                if (v77){
                    v66[v69] = v72;
                    int v78;
                    v78 = v69 + 1;
                    v79 = v78;
                } else {
                    v79 = v69;
                }
                v69 = v79;
                v68 += 1 ;
            }
            bool v80;
            v80 = 32 >= v69;
            bool v81;
            v81 = v80 == false;
            if (v81){
                assert("The type level dimension has to equal the value passed at runtime into create." && v80);
            } else {
            }
            static_array_list<Union0,32> v84;
            v84 = static_array_list<Union0,32>{};
            v84.unsafe_set_length(v69);
            int v86;
            v86 = 0;
            while (while_method_0(v69, v86)){
                Union0 v89;
                v89 = v66[v86];
                v84[v86] = v89;
                v86 += 1 ;
            }
            std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v91 = v0.v0;
            auto v92 = v91.find(v84);
            bool v93;
            v93 = v92 != v91.end();
            Union6 v98;
            if (v93){
                static_array<float,3> v94; static_array<float,3> v95;
                Tuple1 tmp10 = v92->second;
                v94 = tmp10.v0; v95 = tmp10.v1;
                v98 = Union6{Union6_1{v94, v95}};
            } else {
                v98 = Union6{Union6_0{}};
            }
            static_array<float,3> v113; static_array<float,3> v114;
            switch (v98.tag) {
                case 0: { // None
                    static_array<float,3> v102;
                    int v104;
                    v104 = 0;
                    while (while_method_4(v104)){
                        v102[v104] = 0.0f;
                        v104 += 1 ;
                    }
                    static_array<float,3> v107;
                    int v109;
                    v109 = 0;
                    while (while_method_4(v109)){
                        v107[v109] = 0.0f;
                        v109 += 1 ;
                    }
                    v113 = v102; v114 = v107;
                    break;
                }
                case 1: { // Some
                    static_array<float,3> v99 = v98.case1.v0; static_array<float,3> v100 = v98.case1.v1;
                    v113 = v99; v114 = v100;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            int v116;
            v116 = v58[0];
            int v119;
            v119 = v58[1];
            bool v121;
            v121 = v116 == v119;
            bool v122;
            v122 = v121 != true;
            Union7 v126;
            if (v122){
                Union2 v123;
                v123 = Union2{Union2_1{}};
                v126 = Union7{Union7_1{v123}};
            } else {
                v126 = Union7{Union7_0{}};
            }
            bool v127;
            v127 = v59 > 0;
            Union7 v131;
            if (v127){
                Union2 v128;
                v128 = Union2{Union2_2{}};
                v131 = Union7{Union7_1{v128}};
            } else {
                v131 = Union7{Union7_0{}};
            }
            bool v134;
            switch (v131.tag) {
                case 0: { // None
                    v134 = false;
                    break;
                }
                case 1: { // Some
                    Union2 v132 = v131.case1.v0;
                    v134 = true;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            bool v137;
            switch (v126.tag) {
                case 0: { // None
                    v137 = false;
                    break;
                }
                case 1: { // Some
                    Union2 v135 = v126.case1.v0;
                    v137 = true;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            static_array<bool,3> v139;
            v139[0] = true;
            v139[1] = v137;
            v139[2] = v134;
            static_array<float,3> v141;
            v141 = regret_match_3(v114, v139);
            float v164;
            switch (v131.tag) {
                case 0: { // None
                    v164 = 0.0f;
                    break;
                }
                case 1: { // Some
                    Union2 v142 = v131.case1.v0;
                    float v144;
                    v144 = v141[2];
                    static_array<float,2> & v146 = v1.v1;
                    float v148;
                    v148 = v146[v57];
                    static_array<float,2> & v150 = v1.v1;
                    float v151;
                    v151 = log(v144);
                    float v152;
                    v152 = v148 + v151;
                    v150[v57] = v152;
                    static_array_list<Union0,32> & v153 = v1.v2;
                    Union0 v154;
                    v154 = Union0{Union0_1{v57, v142}};
                    v153.push(v154);
                    Union5 v155;
                    v155 = Union5{Union5_2{v54, v55, v56, v57, v58, v59, v142}};
                    float v156;
                    v156 = loop_2(v0, v1, v3, v155);
                    static_array_list<Union0,32> & v157 = v1.v2;
                    Union0 v158;
                    v158 = v157.pop();
                    static_array<float,2> & v159 = v1.v1;
                    v159[v57] = v148;
                    bool v160;
                    v160 = v57 == 0;
                    if (v160){
                        v164 = v156;
                    } else {
                        float v161;
                        v161 = -v156;
                        v164 = v161;
                    }
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            float v187;
            switch (v126.tag) {
                case 0: { // None
                    v187 = 0.0f;
                    break;
                }
                case 1: { // Some
                    Union2 v165 = v126.case1.v0;
                    float v167;
                    v167 = v141[1];
                    static_array<float,2> & v169 = v1.v1;
                    float v171;
                    v171 = v169[v57];
                    static_array<float,2> & v173 = v1.v1;
                    float v174;
                    v174 = log(v167);
                    float v175;
                    v175 = v171 + v174;
                    v173[v57] = v175;
                    static_array_list<Union0,32> & v176 = v1.v2;
                    Union0 v177;
                    v177 = Union0{Union0_1{v57, v165}};
                    v176.push(v177);
                    Union5 v178;
                    v178 = Union5{Union5_2{v54, v55, v56, v57, v58, v59, v165}};
                    float v179;
                    v179 = loop_2(v0, v1, v3, v178);
                    static_array_list<Union0,32> & v180 = v1.v2;
                    Union0 v181;
                    v181 = v180.pop();
                    static_array<float,2> & v182 = v1.v1;
                    v182[v57] = v171;
                    bool v183;
                    v183 = v57 == 0;
                    if (v183){
                        v187 = v179;
                    } else {
                        float v184;
                        v184 = -v179;
                        v187 = v184;
                    }
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            float v189;
            v189 = v141[0];
            static_array<float,2> & v191 = v1.v1;
            float v193;
            v193 = v191[v57];
            static_array<float,2> & v195 = v1.v1;
            float v196;
            v196 = log(v189);
            float v197;
            v197 = v193 + v196;
            v195[v57] = v197;
            static_array_list<Union0,32> & v198 = v1.v2;
            Union2 v199;
            v199 = Union2{Union2_0{}};
            Union0 v200;
            v200 = Union0{Union0_1{v57, v199}};
            v198.push(v200);
            Union2 v201;
            v201 = Union2{Union2_0{}};
            Union5 v202;
            v202 = Union5{Union5_2{v54, v55, v56, v57, v58, v59, v201}};
            float v203;
            v203 = loop_2(v0, v1, v3, v202);
            static_array_list<Union0,32> & v204 = v1.v2;
            Union0 v205;
            v205 = v204.pop();
            static_array<float,2> & v206 = v1.v1;
            v206[v57] = v193;
            bool v207;
            v207 = v57 == 0;
            float v209;
            if (v207){
                v209 = v203;
            } else {
                float v208;
                v208 = -v203;
                v209 = v208;
            }
            static_array<float,3> v211;
            v211[0] = v209;
            v211[1] = v187;
            v211[2] = v164;
            static_array<float,3> v214;
            int v216;
            v216 = 0;
            while (while_method_4(v216)){
                float v219;
                v219 = v211[v216];
                float v222;
                v222 = v141[v216];
                float v224;
                v224 = v219 * v222;
                v214[v216] = v224;
                v216 += 1 ;
            }
            int v225; float v226;
            Tuple4 tmp13 = Tuple4{0, 0.0f};
            v225 = tmp13.v0; v226 = tmp13.v1;
            while (while_method_4(v225)){
                float v229;
                v229 = v214[v225];
                float v231;
                v231 = v226 + v229;
                v226 = v231;
                v225 += 1 ;
            }
            std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v232 = v0.v0;
            static_array<float,2> & v233 = v1.v1;
            float v235;
            v235 = v233[v57];
            float v237;
            v237 = exp(v235);
            static_array<float,3> v239;
            int v241;
            v241 = 0;
            while (while_method_4(v241)){
                float v244;
                v244 = v113[v241];
                float v247;
                v247 = v141[v241];
                float v249;
                v249 = 0.99609375f * v244;
                float v250;
                v250 = v237 * v247;
                float v251;
                v251 = v249 + v250;
                v239[v241] = v251;
                v241 += 1 ;
            }
            int v252;
            v252 = v57 ^ 1;
            static_array<float,2> & v253 = v1.v1;
            float v255;
            v255 = v253[v252];
            float v257;
            v257 = exp(v255);
            static_array<float,3> v259;
            int v261;
            v261 = 0;
            while (while_method_4(v261)){
                float v264;
                v264 = v114[v261];
                float v267;
                v267 = v211[v261];
                float v269;
                v269 = v267 - v226;
                float v270;
                v270 = v257 * v269;
                float v271;
                v271 = v264 + v270;
                bool v272;
                v272 = 0.0f >= v271;
                float v273;
                if (v272){
                    v273 = 0.0f;
                } else {
                    v273 = v271;
                }
                v259[v261] = v273;
                v261 += 1 ;
            }
            v232[v84] = Tuple1{v239, v259};
            float v275;
            if (v207){
                v275 = v226;
            } else {
                float v274;
                v274 = -v226;
                v275 = v274;
            }
            v3.v0 = v275;
            Union5 v276;
            v276 = Union5{Union5_3{}};
            return loop_2(v0, v1, v3, v276);
            break;
        }
        case 3: { // RoundWithAction
            Union4 v278 = v2.case3.v0; bool v279 = v2.case3.v1; static_array<Union1,2> v280 = v2.case3.v2; int v281 = v2.case3.v3; static_array<int,2> v282 = v2.case3.v4; int v283 = v2.case3.v5; Union2 v284 = v2.case3.v6;
            static_array_list<Union0,32> & v285 = v1.v2;
            Union0 v286;
            v286 = Union0{Union0_1{v281, v284}};
            v285.push(v286);
            Union5 v287;
            v287 = Union5{Union5_2{v278, v279, v280, v281, v282, v283, v284}};
            float v288;
            v288 = loop_2(v0, v1, v3, v287);
            static_array_list<Union0,32> & v289 = v1.v2;
            Union0 v290;
            v290 = v289.pop();
            return v288;
            break;
        }
        case 4: { // TerminalCall
            Union4 v25 = v2.case4.v0; bool v26 = v2.case4.v1; static_array<Union1,2> v27 = v2.case4.v2; int v28 = v2.case4.v3; static_array<int,2> v29 = v2.case4.v4; int v30 = v2.case4.v5;
            int v32;
            v32 = v29[v28];
            Union8 v34;
            v34 = compare_hands_6(v25, v26, v27, v28, v29, v30);
            int v39; int v40;
            switch (v34.tag) {
                case 0: { // Eq
                    v39 = 0; v40 = -1;
                    break;
                }
                case 1: { // Gt
                    v39 = v32; v40 = 0;
                    break;
                }
                case 2: { // Lt
                    v39 = v32; v40 = 1;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            int v41;
            v41 = -v40;
            bool v42;
            v42 = v40 >= v41;
            int v43;
            if (v42){
                v43 = v40;
            } else {
                v43 = v41;
            }
            float v44;
            v44 = (float)v39;
            bool v45;
            v45 = v43 == 0;
            float v47;
            if (v45){
                v47 = v44;
            } else {
                float v46;
                v46 = -v44;
                v47 = v46;
            }
            v3.v0 = v47;
            static_array_list<Union0,32> & v48 = v1.v2;
            Union0 v49;
            v49 = Union0{Union0_3{v27, v39, v40}};
            v48.push(v49);
            Union5 v50;
            v50 = Union5{Union5_3{}};
            float v51;
            v51 = loop_2(v0, v1, v3, v50);
            static_array_list<Union0,32> & v52 = v1.v2;
            Union0 v53;
            v53 = v52.pop();
            return v51;
            break;
        }
        case 5: { // TerminalFold
            Union4 v4 = v2.case5.v0; bool v5 = v2.case5.v1; static_array<Union1,2> v6 = v2.case5.v2; int v7 = v2.case5.v3; static_array<int,2> v8 = v2.case5.v4; int v9 = v2.case5.v5;
            int v11;
            v11 = v8[v7];
            int v13;
            v13 = -v11;
            float v14;
            v14 = (float)v13;
            bool v15;
            v15 = v7 == 0;
            float v17;
            if (v15){
                v17 = v14;
            } else {
                float v16;
                v16 = -v14;
                v17 = v16;
            }
            v3.v0 = v17;
            int v18;
            v18 = v7 ^ 1;
            static_array_list<Union0,32> & v19 = v1.v2;
            Union0 v20;
            v20 = Union0{Union0_3{v6, v11, v18}};
            v19.push(v20);
            Union5 v21;
            v21 = Union5{Union5_3{}};
            float v22;
            v22 = loop_2(v0, v1, v3, v21);
            static_array_list<Union0,32> & v23 = v1.v2;
            Union0 v24;
            v24 = v23.pop();
            return v22;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
void train_loop_0(StackRefs0 & v0){
    static_array_list<Union0,32> v2;
    v2 = static_array_list<Union0,32>{};
    static_array<float,2> v5;
    int v7;
    v7 = 0;
    while (while_method_1(v7)){
        v5[v7] = 0.0f;
        v7 += 1 ;
    }
    unsigned int v9;
    v9 = 63u;
    StackRefs1 v10{v9, v5, v2};
    Union3 v11;
    v11 = Union3{Union3_1{}};
    float v12;
    v12 = body_1(v0, v10, v11);
    return ;
}
inline bool while_method_5(std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1> & v0, std::unordered_map<static_array_list<Union0,32>, Tuple1, Fun0, Fun1>::iterator & v1){
    bool v2;
    v2 = v1 != v0.end();
    return v2;
}
static_array<float,3> normalize_10(static_array<float,3> v0){
    int v1; float v2;
    Tuple4 tmp17 = Tuple4{0, 0.0f};
    v1 = tmp17.v0; v2 = tmp17.v1;
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
inline bool while_method_6(std::unordered_map<static_array_list<Union0,32>, static_array<float,3>, Fun0, Fun1> & v0, std::unordered_map<static_array_list<Union0,32>, static_array<float,3>, Fun0, Fun1>::iterator & v1){
    bool v2;
    v2 = v1 != v0.end();
    return v2;
}
void method_12(Union1 v0){
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
void method_13(Union2 v0){
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
void method_11(Union0 v0){
    switch (v0.tag) {
        case 0: { // CommunityCardIs
            Union1 v1 = v0.case0.v0;
            printf("%s(","CommunityCardIs");
            method_12(v1);
            printf(")");
            return ;
            break;
        }
        case 1: { // PlayerAction
            int v2 = v0.case1.v0; Union2 v3 = v0.case1.v1;
            printf("%s(%d, ","PlayerAction", v2);
            method_13(v3);
            printf(")");
            return ;
            break;
        }
        case 2: { // PlayerGotCard
            int v4 = v0.case2.v0; Union1 v5 = v0.case2.v1;
            printf("%s(%d, ","PlayerGotCard", v4);
            method_12(v5);
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
                method_12(v12);
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
        v6 = v4 % 10;
        bool v7;
        v7 = v6 == 0;
        if (v7){
            printf("{%s = %d; %s = %d}\n","i", v4, "nearTo", 1000);
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
    while (while_method_5(v15, v16)){
        static_array_list<Union0,32> v18;
        v18 = v16->first;
        static_array<float,3> v19; static_array<float,3> v20;
        Tuple1 tmp16 = v16->second;
        v19 = tmp16.v0; v20 = tmp16.v1;
        static_array<float,3> v21;
        v21 = normalize_10(v19);
        v13[v18] = v21;
        ++v16;
    }
    printf("%s\n","{");
    std::unordered_map<static_array_list<Union0,32>, static_array<float,3>, Fun0, Fun1> & v49 = v13;
    auto v50 = v49.begin();
    while (while_method_6(v49, v50)){
        static_array_list<Union0,32> v52;
        v52 = v50->first;
        static_array<float,3> v53;
        v53 = v50->second;
        printf("%s","[");
        int v54;
        v54 = v52.length;
        bool v55;
        v55 = 100 < v54;
        int v56;
        if (v55){
            v56 = 100;
        } else {
            v56 = v54;
        }
        int v57;
        v57 = 0;
        while (while_method_0(v56, v57)){
            Union0 v60;
            v60 = v52[v57];
            printf("");
            method_11(v60);
            printf("");
            int v62;
            v62 = v57 + 1;
            int v63;
            v63 = v52.length;
            bool v64;
            v64 = v62 < v63;
            if (v64){
                printf("%s","; ");
            } else {
            }
            v57 += 1 ;
        }
        int v65;
        v65 = v52.length;
        bool v66;
        v66 = v65 > 100;
        if (v66){
            printf("%s","; ...");
        } else {
        }
        printf("%s","]");
        printf("");
        printf("%s"," => ");
        printf("%s","[");
        int v67;
        v67 = 0;
        while (while_method_4(v67)){
            float v70;
            v70 = v53[v67];
            printf("%f",v70);
            int v72;
            v72 = v67 + 1;
            bool v73;
            v73 = v72 < 3;
            if (v73){
                printf("%s","; ");
            } else {
            }
            v67 += 1 ;
        }
        printf("%s","]");
        printf("\n");
        ++v50;
    }
    printf("%s\n","}");
    printf("\n");
    fflush(stdout);
    return 0;
}
