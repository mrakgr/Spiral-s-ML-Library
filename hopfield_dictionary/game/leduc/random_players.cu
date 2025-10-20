#include "random_players.auto.cu"
#include <thrust/device_vector.h>
#include <xoshiro.h>
struct Union1;
struct Union2;
struct Union0;
struct StackRefs0;
struct StackRefs1;
struct StackMut0;
struct Union4;
struct Union3;
struct Union5;
void method_4(Union1 v0);
void method_3(Union4 v0);
void method_5(Union2 v0);
void method_2(Union3 v0);
struct Union6;
struct Tuple0;
unsigned int loop_7(unsigned int v0, xso::rng & v1);
struct StackMut1;
struct StackMut2;
unsigned int find_nth_set_bit_8(int v0, unsigned int v1, unsigned int v2);
Tuple0 draw_card_6(xso::rng & v0, unsigned int v1);
int int_range_9(int v0, int v1, xso::rng & v2);
struct Union7;
int tag_11(Union1 v0);
bool is_pair_12(int v0, int v1);
struct Tuple1;
Tuple1 order_13(int v0, int v1);
Union7 compare_hands_10(Union4 v0, bool v1, static_array<Union1,2> v2, int v3, static_array<int,2> v4, int v5);
Union5 body_1(xso::rng & v0, StackRefs0 & v1, StackRefs1 & v2, StackMut0 & v3, Union3 v4);
static_array_list<Union0,32> train_loop_0();
void method_14(Union0 v0);
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
struct StackRefs0 {
    static_array_list<Union0,32> & v0;
    __host__ __device__ StackRefs0() = default;
    __host__ __device__ StackRefs0(static_array_list<Union0,32> & t0) : v0(t0) {}
};
struct StackRefs1 {
    unsigned int & v0;
    __host__ __device__ StackRefs1() = default;
    __host__ __device__ StackRefs1(unsigned int & t0) : v0(t0) {}
};
struct StackMut0 {
    float v0;
    __host__ __device__ StackMut0() = default;
    __host__ __device__ StackMut0(float t0) : v0(t0) {}
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
struct Union5_0 { // None
};
struct Union5_1 { // Some
    Union3 v0;
    __host__ __device__ Union5_1(Union3 t0) : v0(t0) {}
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
struct Union6_0 { // T_game_chance_community_card
    Union4 v0;
    static_array<Union1,2> v2;
    static_array<int,2> v4;
    Union1 v6;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union6_0(Union4 t0, bool t1, static_array<Union1,2> t2, int t3, static_array<int,2> t4, int t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __host__ __device__ Union6_0() = delete;
};
struct Union6_1 { // T_game_chance_init
    Union1 v0;
    Union1 v1;
    __host__ __device__ Union6_1(Union1 t0, Union1 t1) : v0(t0), v1(t1) {}
    __host__ __device__ Union6_1() = delete;
};
struct Union6_2 { // T_game_round
    Union4 v0;
    static_array<Union1,2> v2;
    static_array<int,2> v4;
    Union2 v6;
    int v3;
    int v5;
    bool v1;
    __host__ __device__ Union6_2(Union4 t0, bool t1, static_array<Union1,2> t2, int t3, static_array<int,2> t4, int t5, Union2 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
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
struct Tuple0 {
    Union1 v0;
    unsigned int v1;
    __host__ __device__ Tuple0() = default;
    __host__ __device__ Tuple0(Union1 t0, unsigned int t1) : v0(t0), v1(t1) {}
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
struct Union7_0 { // Eq
};
struct Union7_1 { // Gt
};
struct Union7_2 { // Lt
};
struct Union7 {
    union {
        Union7_0 case0; // Eq
        Union7_1 case1; // Gt
        Union7_2 case2; // Lt
    };
    unsigned char tag{255};
    __host__ __device__ Union7() {}
    __host__ __device__ Union7(Union7_0 t) : tag(0), case0(t) {} // Eq
    __host__ __device__ Union7(Union7_1 t) : tag(1), case1(t) {} // Gt
    __host__ __device__ Union7(Union7_2 t) : tag(2), case2(t) {} // Lt
    __host__ __device__ Union7(const Union7 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(x.case0); break; // Eq
            case 1: new (&this->case1) Union7_1(x.case1); break; // Gt
            case 2: new (&this->case2) Union7_2(x.case2); break; // Lt
        }
    }
    __host__ __device__ Union7(const Union7 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(std::move(x.case0)); break; // Eq
            case 1: new (&this->case1) Union7_1(std::move(x.case1)); break; // Gt
            case 2: new (&this->case2) Union7_2(std::move(x.case2)); break; // Lt
        }
    }
    __host__ __device__ Union7 & operator=(const Union7 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Eq
                case 1: this->case1 = x.case1; break; // Gt
                case 2: this->case2 = x.case2; break; // Lt
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
                case 0: this->case0 = std::move(x.case0); break; // Eq
                case 1: this->case1 = std::move(x.case1); break; // Gt
                case 2: this->case2 = std::move(x.case2); break; // Lt
            }
        } else {
            this->~Union7();
            new (this) Union7{std::move(x)};
        }
        return *this;
    }
    __host__ __device__ ~Union7() {
        switch(this->tag){
            case 0: this->case0.~Union7_0(); break; // Eq
            case 1: this->case1.~Union7_1(); break; // Gt
            case 2: this->case2.~Union7_2(); break; // Lt
        }
        this->tag = 255;
    }
};
struct Tuple1 {
    int v0;
    int v1;
    __host__ __device__ Tuple1() = default;
    __host__ __device__ Tuple1(int t0, int t1) : v0(t0), v1(t1) {}
};
inline bool while_method_0(Union5 v0){
    switch (v0.tag) {
        case 0: { // None
            return false;
            break;
        }
        case 1: { // Some
            Union3 v1 = v0.case1.v0;
            return true;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
void method_4(Union1 v0){
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
void method_3(Union4 v0){
    switch (v0.tag) {
        case 0: { // None
            printf("%s","None");
            return ;
            break;
        }
        case 1: { // Some
            Union1 v1 = v0.case1.v0;
            printf("%s(","Some");
            method_4(v1);
            printf(")");
            return ;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 2;
    return v1;
}
void method_5(Union2 v0){
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
void method_2(Union3 v0){
    switch (v0.tag) {
        case 0: { // ChanceCommunityCard
            Union4 v1 = v0.case0.v0; bool v2 = v0.case0.v1; static_array<Union1,2> v3 = v0.case0.v2; int v4 = v0.case0.v3; static_array<int,2> v5 = v0.case0.v4; int v6 = v0.case0.v5;
            printf("%s({%s = ","ChanceCommunityCard", "community_card");
            method_3(v1);
            const char * v9;
            if (v2){
                const char * v7;
                v7 = "true";
                v9 = v7;
            } else {
                const char * v8;
                v8 = "false";
                v9 = v8;
            }
            printf("; %s = %s; %s = %s","is_button_s_first_move", v9, "pl_card", "[");
            int v10;
            v10 = 0;
            while (while_method_1(v10)){
                Union1 v13;
                v13 = v3[v10];
                printf("");
                method_4(v13);
                printf("");
                int v15;
                v15 = v10 + 1;
                bool v16;
                v16 = v15 < 2;
                if (v16){
                    printf("%s","; ");
                } else {
                }
                v10 += 1 ;
            }
            printf("%s","]");
            printf("; %s = %d; %s = %s","player_turn", v4, "pot", "[");
            int v17;
            v17 = 0;
            while (while_method_1(v17)){
                int v20;
                v20 = v5[v17];
                printf("%d",v20);
                int v22;
                v22 = v17 + 1;
                bool v23;
                v23 = v22 < 2;
                if (v23){
                    printf("%s","; ");
                } else {
                }
                v17 += 1 ;
            }
            printf("%s","]");
            printf("; %s = %d})","raises_left", v6);
            return ;
            break;
        }
        case 1: { // ChanceInit
            printf("%s","ChanceInit");
            return ;
            break;
        }
        case 2: { // Round
            Union4 v24 = v0.case2.v0; bool v25 = v0.case2.v1; static_array<Union1,2> v26 = v0.case2.v2; int v27 = v0.case2.v3; static_array<int,2> v28 = v0.case2.v4; int v29 = v0.case2.v5;
            printf("%s({%s = ","Round", "community_card");
            method_3(v24);
            const char * v32;
            if (v25){
                const char * v30;
                v30 = "true";
                v32 = v30;
            } else {
                const char * v31;
                v31 = "false";
                v32 = v31;
            }
            printf("; %s = %s; %s = %s","is_button_s_first_move", v32, "pl_card", "[");
            int v33;
            v33 = 0;
            while (while_method_1(v33)){
                Union1 v36;
                v36 = v26[v33];
                printf("");
                method_4(v36);
                printf("");
                int v38;
                v38 = v33 + 1;
                bool v39;
                v39 = v38 < 2;
                if (v39){
                    printf("%s","; ");
                } else {
                }
                v33 += 1 ;
            }
            printf("%s","]");
            printf("; %s = %d; %s = %s","player_turn", v27, "pot", "[");
            int v40;
            v40 = 0;
            while (while_method_1(v40)){
                int v43;
                v43 = v28[v40];
                printf("%d",v43);
                int v45;
                v45 = v40 + 1;
                bool v46;
                v46 = v45 < 2;
                if (v46){
                    printf("%s","; ");
                } else {
                }
                v40 += 1 ;
            }
            printf("%s","]");
            printf("; %s = %d})","raises_left", v29);
            return ;
            break;
        }
        case 3: { // RoundWithAction
            Union4 v47 = v0.case3.v0; bool v48 = v0.case3.v1; static_array<Union1,2> v49 = v0.case3.v2; int v50 = v0.case3.v3; static_array<int,2> v51 = v0.case3.v4; int v52 = v0.case3.v5; Union2 v53 = v0.case3.v6;
            printf("%s({%s = ","RoundWithAction", "community_card");
            method_3(v47);
            const char * v56;
            if (v48){
                const char * v54;
                v54 = "true";
                v56 = v54;
            } else {
                const char * v55;
                v55 = "false";
                v56 = v55;
            }
            printf("; %s = %s; %s = %s","is_button_s_first_move", v56, "pl_card", "[");
            int v57;
            v57 = 0;
            while (while_method_1(v57)){
                Union1 v60;
                v60 = v49[v57];
                printf("");
                method_4(v60);
                printf("");
                int v62;
                v62 = v57 + 1;
                bool v63;
                v63 = v62 < 2;
                if (v63){
                    printf("%s","; ");
                } else {
                }
                v57 += 1 ;
            }
            printf("%s","]");
            printf("; %s = %d; %s = %s","player_turn", v50, "pot", "[");
            int v64;
            v64 = 0;
            while (while_method_1(v64)){
                int v67;
                v67 = v51[v64];
                printf("%d",v67);
                int v69;
                v69 = v64 + 1;
                bool v70;
                v70 = v69 < 2;
                if (v70){
                    printf("%s","; ");
                } else {
                }
                v64 += 1 ;
            }
            printf("%s","]");
            printf("; %s = %d}, ","raises_left", v52);
            method_5(v53);
            printf(")");
            return ;
            break;
        }
        case 4: { // TerminalCall
            Union4 v71 = v0.case4.v0; bool v72 = v0.case4.v1; static_array<Union1,2> v73 = v0.case4.v2; int v74 = v0.case4.v3; static_array<int,2> v75 = v0.case4.v4; int v76 = v0.case4.v5;
            printf("%s({%s = ","TerminalCall", "community_card");
            method_3(v71);
            const char * v79;
            if (v72){
                const char * v77;
                v77 = "true";
                v79 = v77;
            } else {
                const char * v78;
                v78 = "false";
                v79 = v78;
            }
            printf("; %s = %s; %s = %s","is_button_s_first_move", v79, "pl_card", "[");
            int v80;
            v80 = 0;
            while (while_method_1(v80)){
                Union1 v83;
                v83 = v73[v80];
                printf("");
                method_4(v83);
                printf("");
                int v85;
                v85 = v80 + 1;
                bool v86;
                v86 = v85 < 2;
                if (v86){
                    printf("%s","; ");
                } else {
                }
                v80 += 1 ;
            }
            printf("%s","]");
            printf("; %s = %d; %s = %s","player_turn", v74, "pot", "[");
            int v87;
            v87 = 0;
            while (while_method_1(v87)){
                int v90;
                v90 = v75[v87];
                printf("%d",v90);
                int v92;
                v92 = v87 + 1;
                bool v93;
                v93 = v92 < 2;
                if (v93){
                    printf("%s","; ");
                } else {
                }
                v87 += 1 ;
            }
            printf("%s","]");
            printf("; %s = %d})","raises_left", v76);
            return ;
            break;
        }
        case 5: { // TerminalFold
            Union4 v94 = v0.case5.v0; bool v95 = v0.case5.v1; static_array<Union1,2> v96 = v0.case5.v2; int v97 = v0.case5.v3; static_array<int,2> v98 = v0.case5.v4; int v99 = v0.case5.v5;
            printf("%s({%s = ","TerminalFold", "community_card");
            method_3(v94);
            const char * v102;
            if (v95){
                const char * v100;
                v100 = "true";
                v102 = v100;
            } else {
                const char * v101;
                v101 = "false";
                v102 = v101;
            }
            printf("; %s = %s; %s = %s","is_button_s_first_move", v102, "pl_card", "[");
            int v103;
            v103 = 0;
            while (while_method_1(v103)){
                Union1 v106;
                v106 = v96[v103];
                printf("");
                method_4(v106);
                printf("");
                int v108;
                v108 = v103 + 1;
                bool v109;
                v109 = v108 < 2;
                if (v109){
                    printf("%s","; ");
                } else {
                }
                v103 += 1 ;
            }
            printf("%s","]");
            printf("; %s = %d; %s = %s","player_turn", v97, "pot", "[");
            int v110;
            v110 = 0;
            while (while_method_1(v110)){
                int v113;
                v113 = v98[v110];
                printf("%d",v113);
                int v115;
                v115 = v110 + 1;
                bool v116;
                v116 = v115 < 2;
                if (v116){
                    printf("%s","; ");
                } else {
                }
                v110 += 1 ;
            }
            printf("%s","]");
            printf("; %s = %d})","raises_left", v99);
            return ;
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
unsigned int loop_7(unsigned int v0, xso::rng & v1){
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
        return loop_7(v0, v1);
    }
}
inline bool while_method_2(unsigned int v0, unsigned int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
unsigned int find_nth_set_bit_8(int v0, unsigned int v1, unsigned int v2){
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
    while (while_method_2(v8, v9)){
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
Tuple0 draw_card_6(xso::rng & v0, unsigned int v1){
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
    v8 = loop_7(v4, v0);
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
        v18 = find_nth_set_bit_8(v13, v14, v1);
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
    return Tuple0{v37, v40};
}
inline bool while_method_3(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
int int_range_9(int v0, int v1, xso::rng & v2){
    int v3;
    v3 = v0 - v1;
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
    v8 = loop_7(v4, v2);
    unsigned int v9;
    v9 = (unsigned int)v1;
    unsigned int v10;
    v10 = v8 + v9;
    int v11;
    v11 = (int)v10;
    return v11;
}
int tag_11(Union1 v0){
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
bool is_pair_12(int v0, int v1){
    bool v2;
    v2 = v1 == v0;
    return v2;
}
Tuple1 order_13(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    if (v2){
        return Tuple1{v1, v0};
    } else {
        return Tuple1{v0, v1};
    }
}
Union7 compare_hands_10(Union4 v0, bool v1, static_array<Union1,2> v2, int v3, static_array<int,2> v4, int v5){
    switch (v0.tag) {
        case 0: { // None
            printf("%s\n", "Expected the community card to be present in the table.");
            exit(-1);
            break;
        }
        case 1: { // Some
            Union1 v7 = v0.case1.v0;
            int v8;
            v8 = tag_11(v7);
            Union1 v10;
            v10 = v2[0];
            int v12;
            v12 = tag_11(v10);
            Union1 v14;
            v14 = v2[1];
            int v16;
            v16 = tag_11(v14);
            bool v17;
            v17 = is_pair_12(v8, v12);
            bool v18;
            v18 = is_pair_12(v8, v16);
            if (v17){
                if (v18){
                    bool v19;
                    v19 = v12 < v16;
                    if (v19){
                        return Union7{Union7_2{}};
                    } else {
                        bool v21;
                        v21 = v12 > v16;
                        if (v21){
                            return Union7{Union7_1{}};
                        } else {
                            return Union7{Union7_0{}};
                        }
                    }
                } else {
                    return Union7{Union7_1{}};
                }
            } else {
                if (v18){
                    return Union7{Union7_2{}};
                } else {
                    int v29; int v30;
                    Tuple1 tmp3 = order_13(v8, v12);
                    v29 = tmp3.v0; v30 = tmp3.v1;
                    int v31; int v32;
                    Tuple1 tmp4 = order_13(v8, v16);
                    v31 = tmp4.v0; v32 = tmp4.v1;
                    bool v33;
                    v33 = v29 < v31;
                    Union7 v39;
                    if (v33){
                        v39 = Union7{Union7_2{}};
                    } else {
                        bool v35;
                        v35 = v29 > v31;
                        if (v35){
                            v39 = Union7{Union7_1{}};
                        } else {
                            v39 = Union7{Union7_0{}};
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
                            return Union7{Union7_2{}};
                        } else {
                            bool v43;
                            v43 = v30 > v32;
                            if (v43){
                                return Union7{Union7_1{}};
                            } else {
                                return Union7{Union7_0{}};
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
Union5 body_1(xso::rng & v0, StackRefs0 & v1, StackRefs1 & v2, StackMut0 & v3, Union3 v4){
    printf("{%s = ","node");
    method_2(v4);
    printf("}\n");
    Union6 v131;
    switch (v4.tag) {
        case 0: { // ChanceCommunityCard
            Union4 v103 = v4.case0.v0; bool v104 = v4.case0.v1; static_array<Union1,2> v105 = v4.case0.v2; int v106 = v4.case0.v3; static_array<int,2> v107 = v4.case0.v4; int v108 = v4.case0.v5;
            unsigned int & v109 = v2.v0;
            Union1 v110; unsigned int v111;
            Tuple0 tmp0 = draw_card_6(v0, v109);
            v110 = tmp0.v0; v111 = tmp0.v1;
            v2.v0 = v111;
            static_array_list<Union0,32> & v112 = v1.v0;
            Union0 v113;
            v113 = Union0{Union0_0{v110}};
            v112.push(v113);
            v131 = Union6{Union6_0{v103, v104, v105, v106, v107, v108, v110}};
            break;
        }
        case 1: { // ChanceInit
            unsigned int & v115 = v2.v0;
            Union1 v116; unsigned int v117;
            Tuple0 tmp1 = draw_card_6(v0, v115);
            v116 = tmp1.v0; v117 = tmp1.v1;
            v2.v0 = v117;
            unsigned int & v118 = v2.v0;
            Union1 v119; unsigned int v120;
            Tuple0 tmp2 = draw_card_6(v0, v118);
            v119 = tmp2.v0; v120 = tmp2.v1;
            v2.v0 = v120;
            static_array_list<Union0,32> & v121 = v1.v0;
            Union0 v122;
            v122 = Union0{Union0_2{0, v116}};
            v121.push(v122);
            static_array_list<Union0,32> & v123 = v1.v0;
            Union0 v124;
            v124 = Union0{Union0_2{1, v119}};
            v123.push(v124);
            v131 = Union6{Union6_1{v116, v119}};
            break;
        }
        case 2: { // Round
            Union4 v54 = v4.case2.v0; bool v55 = v4.case2.v1; static_array<Union1,2> v56 = v4.case2.v2; int v57 = v4.case2.v3; static_array<int,2> v58 = v4.case2.v4; int v59 = v4.case2.v5;
            static_array_list<Union2,3> v61;
            v61 = static_array_list<Union2,3>{};
            v61.unsafe_set_length(1);
            Union2 v64;
            v64 = Union2{Union2_0{}};
            v61[0] = v64;
            int v67;
            v67 = v58[0];
            int v70;
            v70 = v58[1];
            bool v72;
            v72 = v67 == v70;
            bool v73;
            v73 = v72 != true;
            if (v73){
                Union2 v74;
                v74 = Union2{Union2_1{}};
                v61.push(v74);
            } else {
            }
            bool v75;
            v75 = v59 > 0;
            if (v75){
                Union2 v76;
                v76 = Union2{Union2_2{}};
                v61.push(v76);
            } else {
            }
            int v77;
            v77 = v61.length;
            int v78;
            v78 = v77 - 1;
            int v79;
            v79 = 0;
            while (while_method_3(v78, v79)){
                int v81;
                v81 = v61.length;
                int v82;
                v82 = int_range_9(v81, v79, v0);
                Union2 v84;
                v84 = v61[v79];
                Union2 v87;
                v87 = v61[v82];
                v61[v79] = v87;
                v61[v82] = v84;
                v79 += 1 ;
            }
            Union2 v89;
            v89 = v61.pop();
            static_array_list<Union0,32> & v90 = v1.v0;
            Union0 v91;
            v91 = Union0{Union0_1{v57, v89}};
            v90.push(v91);
            v131 = Union6{Union6_2{v54, v55, v56, v57, v58, v59, v89}};
            break;
        }
        case 3: { // RoundWithAction
            Union4 v93 = v4.case3.v0; bool v94 = v4.case3.v1; static_array<Union1,2> v95 = v4.case3.v2; int v96 = v4.case3.v3; static_array<int,2> v97 = v4.case3.v4; int v98 = v4.case3.v5; Union2 v99 = v4.case3.v6;
            static_array_list<Union0,32> & v100 = v1.v0;
            Union0 v101;
            v101 = Union0{Union0_1{v96, v99}};
            v100.push(v101);
            v131 = Union6{Union6_2{v93, v94, v95, v96, v97, v98, v99}};
            break;
        }
        case 4: { // TerminalCall
            Union4 v28 = v4.case4.v0; bool v29 = v4.case4.v1; static_array<Union1,2> v30 = v4.case4.v2; int v31 = v4.case4.v3; static_array<int,2> v32 = v4.case4.v4; int v33 = v4.case4.v5;
            int v35;
            v35 = v32[v31];
            Union7 v37;
            v37 = compare_hands_10(v28, v29, v30, v31, v32, v33);
            int v42; int v43;
            switch (v37.tag) {
                case 0: { // Eq
                    v42 = 0; v43 = -1;
                    break;
                }
                case 1: { // Gt
                    v42 = v35; v43 = 0;
                    break;
                }
                case 2: { // Lt
                    v42 = v35; v43 = 1;
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                    exit(-1);
                }
            }
            int v44;
            v44 = -v43;
            bool v45;
            v45 = v43 >= v44;
            int v46;
            if (v45){
                v46 = v43;
            } else {
                v46 = v44;
            }
            float v47;
            v47 = (float)v42;
            bool v48;
            v48 = v46 == 0;
            float v50;
            if (v48){
                v50 = v47;
            } else {
                float v49;
                v49 = -v47;
                v50 = v49;
            }
            v3.v0 = v50;
            static_array_list<Union0,32> & v51 = v1.v0;
            Union0 v52;
            v52 = Union0{Union0_3{v30, v42, v43}};
            v51.push(v52);
            v131 = Union6{Union6_3{}};
            break;
        }
        case 5: { // TerminalFold
            Union4 v10 = v4.case5.v0; bool v11 = v4.case5.v1; static_array<Union1,2> v12 = v4.case5.v2; int v13 = v4.case5.v3; static_array<int,2> v14 = v4.case5.v4; int v15 = v4.case5.v5;
            int v17;
            v17 = v14[v13];
            int v19;
            v19 = -v17;
            float v20;
            v20 = (float)v19;
            bool v21;
            v21 = v13 == 0;
            float v23;
            if (v21){
                v23 = v20;
            } else {
                float v22;
                v22 = -v20;
                v23 = v22;
            }
            v3.v0 = v23;
            int v24;
            v24 = v13 ^ 1;
            static_array_list<Union0,32> & v25 = v1.v0;
            Union0 v26;
            v26 = Union0{Union0_3{v12, v17, v24}};
            v25.push(v26);
            v131 = Union6{Union6_3{}};
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
    switch (v131.tag) {
        case 0: { // T_game_chance_community_card
            Union4 v133 = v131.case0.v0; bool v134 = v131.case0.v1; static_array<Union1,2> v135 = v131.case0.v2; int v136 = v131.case0.v3; static_array<int,2> v137 = v131.case0.v4; int v138 = v131.case0.v5; Union1 v139 = v131.case0.v6;
            int v140;
            v140 = 2;
            int v141; int v142;
            Tuple1 tmp5 = Tuple1{0, 0};
            v141 = tmp5.v0; v142 = tmp5.v1;
            while (while_method_1(v141)){
                int v145;
                v145 = v137[v141];
                bool v147;
                v147 = v142 >= v145;
                int v148;
                if (v147){
                    v148 = v142;
                } else {
                    v148 = v145;
                }
                v142 = v148;
                v141 += 1 ;
            }
            static_array<int,2> v150;
            int v152;
            v152 = 0;
            while (while_method_1(v152)){
                v150[v152] = v142;
                v152 += 1 ;
            }
            Union4 v154;
            v154 = Union4{Union4_1{v139}};
            Union3 v155;
            v155 = Union3{Union3_2{v154, true, v135, 0, v150, v140}};
            return Union5{Union5_1{v155}};
            break;
        }
        case 1: { // T_game_chance_init
            Union1 v157 = v131.case1.v0; Union1 v158 = v131.case1.v1;
            int v159;
            v159 = 2;
            static_array<int,2> v161;
            v161[0] = 1;
            v161[1] = 1;
            static_array<Union1,2> v164;
            v164[0] = v157;
            v164[1] = v158;
            Union4 v166;
            v166 = Union4{Union4_0{}};
            Union3 v167;
            v167 = Union3{Union3_2{v166, true, v164, 0, v161, v159}};
            return Union5{Union5_1{v167}};
            break;
        }
        case 2: { // T_game_round
            Union4 v169 = v131.case2.v0; bool v170 = v131.case2.v1; static_array<Union1,2> v171 = v131.case2.v2; int v172 = v131.case2.v3; static_array<int,2> v173 = v131.case2.v4; int v174 = v131.case2.v5; Union2 v175 = v131.case2.v6;
            Union3 v267;
            switch (v169.tag) {
                case 0: { // None
                    switch (v175.tag) {
                        case 0: { // Call
                            if (v170){
                                int v229;
                                v229 = v172 ^ 1;
                                v267 = Union3{Union3_2{v169, false, v171, v229, v173, v174}};
                            } else {
                                v267 = Union3{Union3_0{v169, v170, v171, v172, v173, v174}};
                            }
                            break;
                        }
                        case 1: { // Fold
                            v267 = Union3{Union3_5{v169, v170, v171, v172, v173, v174}};
                            break;
                        }
                        case 2: { // Raise
                            bool v233;
                            v233 = v174 > 0;
                            if (v233){
                                int v234;
                                v234 = v172 ^ 1;
                                int v235;
                                v235 = -1 + v174;
                                int v236; int v237;
                                Tuple1 tmp6 = Tuple1{0, 0};
                                v236 = tmp6.v0; v237 = tmp6.v1;
                                while (while_method_1(v236)){
                                    int v240;
                                    v240 = v173[v236];
                                    bool v242;
                                    v242 = v237 >= v240;
                                    int v243;
                                    if (v242){
                                        v243 = v237;
                                    } else {
                                        v243 = v240;
                                    }
                                    v237 = v243;
                                    v236 += 1 ;
                                }
                                static_array<int,2> v245;
                                int v247;
                                v247 = 0;
                                while (while_method_1(v247)){
                                    v245[v247] = v237;
                                    v247 += 1 ;
                                }
                                static_array<int,2> v250;
                                int v252;
                                v252 = 0;
                                while (while_method_1(v252)){
                                    int v255;
                                    v255 = v245[v252];
                                    bool v257;
                                    v257 = v252 == v172;
                                    int v259;
                                    if (v257){
                                        int v258;
                                        v258 = v255 + 2;
                                        v259 = v258;
                                    } else {
                                        v259 = v255;
                                    }
                                    v250[v252] = v259;
                                    v252 += 1 ;
                                }
                                v267 = Union3{Union3_2{v169, false, v171, v234, v250, v235}};
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
                    Union1 v176 = v169.case1.v0;
                    switch (v175.tag) {
                        case 0: { // Call
                            if (v170){
                                int v178;
                                v178 = v172 ^ 1;
                                v267 = Union3{Union3_2{v169, false, v171, v178, v173, v174}};
                            } else {
                                int v180; int v181;
                                Tuple1 tmp7 = Tuple1{0, 0};
                                v180 = tmp7.v0; v181 = tmp7.v1;
                                while (while_method_1(v180)){
                                    int v184;
                                    v184 = v173[v180];
                                    bool v186;
                                    v186 = v181 >= v184;
                                    int v187;
                                    if (v186){
                                        v187 = v181;
                                    } else {
                                        v187 = v184;
                                    }
                                    v181 = v187;
                                    v180 += 1 ;
                                }
                                static_array<int,2> v189;
                                int v191;
                                v191 = 0;
                                while (while_method_1(v191)){
                                    v189[v191] = v181;
                                    v191 += 1 ;
                                }
                                v267 = Union3{Union3_4{v169, v170, v171, v172, v189, v174}};
                            }
                            break;
                        }
                        case 1: { // Fold
                            v267 = Union3{Union3_5{v169, v170, v171, v172, v173, v174}};
                            break;
                        }
                        case 2: { // Raise
                            bool v195;
                            v195 = v174 > 0;
                            if (v195){
                                int v196;
                                v196 = v172 ^ 1;
                                int v197;
                                v197 = -1 + v174;
                                int v198; int v199;
                                Tuple1 tmp8 = Tuple1{0, 0};
                                v198 = tmp8.v0; v199 = tmp8.v1;
                                while (while_method_1(v198)){
                                    int v202;
                                    v202 = v173[v198];
                                    bool v204;
                                    v204 = v199 >= v202;
                                    int v205;
                                    if (v204){
                                        v205 = v199;
                                    } else {
                                        v205 = v202;
                                    }
                                    v199 = v205;
                                    v198 += 1 ;
                                }
                                static_array<int,2> v207;
                                int v209;
                                v209 = 0;
                                while (while_method_1(v209)){
                                    v207[v209] = v199;
                                    v209 += 1 ;
                                }
                                static_array<int,2> v212;
                                int v214;
                                v214 = 0;
                                while (while_method_1(v214)){
                                    int v217;
                                    v217 = v207[v214];
                                    bool v219;
                                    v219 = v214 == v172;
                                    int v221;
                                    if (v219){
                                        int v220;
                                        v220 = v217 + 4;
                                        v221 = v220;
                                    } else {
                                        v221 = v217;
                                    }
                                    v212[v214] = v221;
                                    v214 += 1 ;
                                }
                                v267 = Union3{Union3_2{v169, false, v171, v196, v212, v197}};
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
            return Union5{Union5_1{v267}};
            break;
        }
        case 3: { // T_none
            return Union5{Union5_0{}};
            break;
        }
        default: {
            assert("Invalid tag." && false);
            exit(-1);
        }
    }
}
static_array_list<Union0,32> train_loop_0(){
    xso::rng v0;
    static_array_list<Union0,32> v2;
    v2 = static_array_list<Union0,32>{};
    StackRefs0 v4{v2};
    unsigned int v5;
    v5 = 63u;
    StackRefs1 v6{v5};
    StackMut0 v7{0.0f};
    Union3 v8;
    v8 = Union3{Union3_1{}};
    Union5 v9;
    v9 = Union5{Union5_1{v8}};
    Union5 v10;
    v10 = v9;
    while (while_method_0(v10)){
        Union5 v16;
        switch (v10.tag) {
            case 0: { // None
                v16 = Union5{Union5_0{}};
                break;
            }
            case 1: { // Some
                Union3 v12 = v10.case1.v0;
                v16 = body_1(v0, v4, v6, v7, v12);
                break;
            }
            default: {
                assert("Invalid tag." && false);
                exit(-1);
            }
        }
        v10 = v16;
    }
    static_array_list<Union0,32> & v17 = v4.v0;
    return v17;
}
void method_14(Union0 v0){
    switch (v0.tag) {
        case 0: { // CommunityCardIs
            Union1 v1 = v0.case0.v0;
            printf("%s(","CommunityCardIs");
            method_4(v1);
            printf(")");
            return ;
            break;
        }
        case 1: { // PlayerAction
            int v2 = v0.case1.v0; Union2 v3 = v0.case1.v1;
            printf("%s(%d, ","PlayerAction", v2);
            method_5(v3);
            printf(")");
            return ;
            break;
        }
        case 2: { // PlayerGotCard
            int v4 = v0.case2.v0; Union1 v5 = v0.case2.v1;
            printf("%s(%d, ","PlayerGotCard", v4);
            method_4(v5);
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
                method_4(v12);
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
    static_array_list<Union0,32> v0;
    v0 = train_loop_0();
    printf("%s","[");
    int v16;
    v16 = v0.length;
    bool v17;
    v17 = 100 < v16;
    int v18;
    if (v17){
        v18 = 100;
    } else {
        v18 = v16;
    }
    int v19;
    v19 = 0;
    while (while_method_3(v18, v19)){
        Union0 v22;
        v22 = v0[v19];
        printf("");
        method_14(v22);
        printf("");
        int v24;
        v24 = v19 + 1;
        int v25;
        v25 = v0.length;
        bool v26;
        v26 = v24 < v25;
        if (v26){
            printf("%s","; ");
        } else {
        }
        v19 += 1 ;
    }
    int v27;
    v27 = v0.length;
    bool v28;
    v28 = v27 > 100;
    if (v28){
        printf("%s","; ...");
    } else {
    }
    printf("%s","]");
    printf("\n");
    return 0;
}
