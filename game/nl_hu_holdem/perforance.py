kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <cooperative_groups/reduce.h>
using default_int = int;
using default_uint = unsigned int;
template <typename el>
struct sptr // Shared pointer for the Spiral datatypes. They have to have the refc field inside them to work.
{
    el* base;

    __device__ sptr() : base(nullptr) {}
    __device__ sptr(el* ptr) : base(ptr) { this->base->refc++; }

    __device__ ~sptr()
    {
        if (this->base != nullptr && --this->base->refc == 0)
        {
            delete this->base;
            this->base = nullptr;
        }
    }

    __device__ sptr(sptr& x)
    {
        this->base = x.base;
        this->base->refc++;
    }

    __device__ sptr(sptr&& x)
    {
        this->base = x.base;
        x.base = nullptr;
    }

    __device__ sptr& operator=(sptr& x)
    {
        if (this->base != x.base)
        {
            delete this->base;
            this->base = x.base;
            this->base->refc++;
        }
        return *this;
    }

    __device__ sptr& operator=(sptr&& x)
    {
        if (this->base != x.base)
        {
            delete this->base;
            this->base = x.base;
            x.base = nullptr;
        }
        return *this;
    }
};

template <typename el>
struct csptr : public sptr<el>
{ // Shared pointer for closures specifically.
    using sptr<el>::sptr;
    template <typename... Args>
    __device__ auto operator()(Args... args) -> decltype(this->base->operator()(args...))
    {
        return this->base->operator()(args...);
    }
};

template <typename el, default_int max_length>
struct static_array
{
    el ptr[max_length];
    __device__ el& operator[](default_int i) {
        assert("The index has to be in range." && 0 <= i && i < max_length);
        return this->ptr[i];
    }
};

template <typename el, default_int max_length>
struct static_array_list
{
    default_int length{ 0 };
    el ptr[max_length];

    __device__ el& operator[](default_int i) {
        assert("The index has to be in range." && 0 <= i && i < this->length);
        return this->ptr[i];
    }
    __device__ void push(el& x) {
        ptr[this->length++] = x;
        assert("The array after pushing should not be greater than max length." && this->length <= max_length);
    }
    __device__ void push(el&& x) {
        ptr[this->length++] = std::move(x);
        assert("The array after pushing should not be greater than max length." && this->length <= max_length);
    }
    __device__ el pop() {
        assert("The array before popping should be greater than 0." && 0 < this->length);
        auto x = ptr[--this->length];
        ptr[this->length].~el();
        new (&ptr[this->length]) el();
        return x;
    }
    // Should be used only during initialization.
    __device__ void unsafe_set_length(default_int i) {
        assert("The new length should be in range." && 0 <= i && i <= max_length);
        this->length = i;
    }
};

template <typename el, default_int max_length>
struct dynamic_array_base
{
    int refc{ 0 };
    el* ptr;

    __device__ dynamic_array_base() : ptr(new el[max_length]) {}
    __device__ ~dynamic_array_base() { delete[] this->ptr; }

    __device__ el& operator[](default_int i) {
        assert("The index has to be in range." && 0 <= i && i < this->length);
        return this->ptr[i];
    }
};

template <typename el, default_int max_length>
struct dynamic_array
{
    sptr<dynamic_array_base<el, max_length>> ptr;

    __device__ dynamic_array() = default;
    __device__ dynamic_array(bool t) : ptr(new dynamic_array_base<el, max_length>()) {}
    __device__ el& operator[](default_int i) {
        return this->ptr.base->operator[](i);
    }
};

template <typename el, default_int max_length>
struct dynamic_array_list_base
{
    int refc{ 0 };
    default_int length{ 0 };
    el* ptr;

    __device__ dynamic_array_list_base() : ptr(new el[max_length]) {}
    __device__ dynamic_array_list_base(default_int l) : ptr(new el[max_length]) { this->unsafe_set_length(l); }
    __device__ ~dynamic_array_list_base() { delete[] this->ptr; }

    __device__ el& operator[](default_int i) {
        assert("The index has to be in range." && 0 <= i && i < this->length);
        return this->ptr[i];
    }
    __device__ void push(el& x) {
        ptr[this->length++] = x;
        assert("The array after pushing should not be greater than max length." && this->length <= max_length);
    }
    __device__ void push(el&& x) {
        ptr[this->length++] = std::move(x);
        assert("The array after pushing should not be greater than max length." && this->length <= max_length);
    }
    __device__ el pop() {
        assert("The array before popping should be greater than 0." && 0 < this->length);
        auto x = ptr[--this->length];
        ptr[this->length].~el();
        new (&ptr[this->length]) el();
        return x;
    }
    // Should be used only during initialization.
    __device__ void unsafe_set_length(default_int i) {
        assert("The new length should be in range." && 0 <= i && i <= max_length);
        this->length = i;
    }
};

template <typename el, default_int max_length>
struct dynamic_array_list
{
    sptr<dynamic_array_list_base<el, max_length>> ptr;

    __device__ dynamic_array_list() = default;
    __device__ dynamic_array_list(default_int l) : ptr(new dynamic_array_list_base<el, max_length>(l)) {}

    __device__ el& operator[](default_int i) {
        return this->ptr.base->operator[](i);
    }
    __device__ void push(el& x) {
        this->ptr.base->push(x);
    }
    __device__ void push(el&& x) {
        this->ptr.base->push(std::move(x));
    }
    __device__ el pop() {
        return this->ptr.base->pop();
    }
    // Should be used only during initialization.
    __device__ void unsafe_set_length(default_int i) {
        this->ptr.base->unsafe_set_length(i);
    }
    __device__ default_int length_() {
        return this->ptr.base->length;
    }
};

struct Union0;
struct Union2;
struct Tuple0;
struct Union1;
struct StackMut0;
struct Union4;
struct Union3;
struct Union5;
struct Union6;
struct Tuple1;
struct Tuple2;
struct Tuple3;
__device__ unsigned int loop_3(unsigned int v0, curandStatePhilox4_32_10_t & v1);
__device__ Tuple3 draw_card_2(curandStatePhilox4_32_10_t & v0, unsigned long long v1);
__device__ Tuple1 draw_cards_1(curandStatePhilox4_32_10_t & v0, unsigned long long v1);
__device__ static_array_list<unsigned char,5> get_community_cards_4(Union4 v0, static_array<unsigned char,3> v1);
struct Tuple4;
__device__ bool player_can_act_6(int v0, static_array<static_array<unsigned char,2>,2> v1, static_array<int,2> v2, int v3, static_array<int,2> v4, Union4 v5);
__device__ Union3 go_next_street_7(int v0, static_array<static_array<unsigned char,2>,2> v1, static_array<int,2> v2, int v3, static_array<int,2> v4, Union4 v5);
__device__ Union3 try_round_5(int v0, static_array<static_array<unsigned char,2>,2> v1, static_array<int,2> v2, int v3, static_array<int,2> v4, Union4 v5);
struct Tuple5;
__device__ Tuple5 draw_cards_8(curandStatePhilox4_32_10_t & v0, unsigned long long v1);
struct Tuple6;
__device__ Tuple6 draw_cards_9(curandStatePhilox4_32_10_t & v0, unsigned long long v1);
__device__ static_array_list<unsigned char,5> get_community_cards_10(Union4 v0, static_array<unsigned char,1> v1);
struct Tuple7;
__device__ int loop_14(static_array<float,6> v0, float v1, int v2);
__device__ int pick_discrete__13(static_array<float,6> v0, float v1);
__device__ int sample_discrete__12(static_array<float,6> v0, curandStatePhilox4_32_10_t & v1);
__device__ Union2 sample_discrete_11(static_array<Tuple7,6> v0, curandStatePhilox4_32_10_t & v1);
struct Tuple8;
struct Tuple9;
struct Union7;
struct Tuple10;
struct Union8;
struct Tuple11;
struct Tuple12;
struct Union9;
struct Union10;
struct Union11;
struct Union12;
struct Union13;
__device__ Tuple0 score_15(static_array<unsigned char,7> v0);
__device__ void method_0(StackMut0 & v0, int v1, Union3 v2);
__device__ int int_range_16(int v0, int v1, curandStatePhilox4_32_10_t & v2);
struct Union0_0 { // Computer
};
struct Union0_1 { // Human
};
struct Union0_2 { // Random
};
struct Union0 {
    union {
        Union0_0 case0; // Computer
        Union0_1 case1; // Human
        Union0_2 case2; // Random
    };
    unsigned char tag{255};
    __device__ Union0() {}
    __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // Computer
    __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // Human
    __device__ Union0(Union0_2 t) : tag(2), case2(t) {} // Random
    __device__ Union0(Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // Computer
            case 1: new (&this->case1) Union0_1(x.case1); break; // Human
            case 2: new (&this->case2) Union0_2(x.case2); break; // Random
        }
    }
    __device__ Union0(Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // Computer
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // Human
            case 2: new (&this->case2) Union0_2(std::move(x.case2)); break; // Random
        }
    }
    __device__ Union0 & operator=(Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Computer
                case 1: this->case1 = x.case1; break; // Human
                case 2: this->case2 = x.case2; break; // Random
            }
        } else {
            this->~Union0();
            new (this) Union0{x};
        }
        return *this;
    }
    __device__ Union0 & operator=(Union0 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // Computer
                case 1: this->case1 = std::move(x.case1); break; // Human
                case 2: this->case2 = std::move(x.case2); break; // Random
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // Computer
            case 1: this->case1.~Union0_1(); break; // Human
            case 2: this->case2.~Union0_2(); break; // Random
        }
        this->tag = 255;
    }
};
struct Union2_0 { // A_All_In
};
struct Union2_1 { // A_Call
};
struct Union2_2 { // A_Fold
};
struct Union2_3 { // A_Raise
    int v0;
    __device__ Union2_3(int t0) : v0(t0) {}
    __device__ Union2_3() = delete;
};
struct Union2 {
    union {
        Union2_0 case0; // A_All_In
        Union2_1 case1; // A_Call
        Union2_2 case2; // A_Fold
        Union2_3 case3; // A_Raise
    };
    unsigned char tag{255};
    __device__ Union2() {}
    __device__ Union2(Union2_0 t) : tag(0), case0(t) {} // A_All_In
    __device__ Union2(Union2_1 t) : tag(1), case1(t) {} // A_Call
    __device__ Union2(Union2_2 t) : tag(2), case2(t) {} // A_Fold
    __device__ Union2(Union2_3 t) : tag(3), case3(t) {} // A_Raise
    __device__ Union2(Union2 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(x.case0); break; // A_All_In
            case 1: new (&this->case1) Union2_1(x.case1); break; // A_Call
            case 2: new (&this->case2) Union2_2(x.case2); break; // A_Fold
            case 3: new (&this->case3) Union2_3(x.case3); break; // A_Raise
        }
    }
    __device__ Union2(Union2 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(std::move(x.case0)); break; // A_All_In
            case 1: new (&this->case1) Union2_1(std::move(x.case1)); break; // A_Call
            case 2: new (&this->case2) Union2_2(std::move(x.case2)); break; // A_Fold
            case 3: new (&this->case3) Union2_3(std::move(x.case3)); break; // A_Raise
        }
    }
    __device__ Union2 & operator=(Union2 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // A_All_In
                case 1: this->case1 = x.case1; break; // A_Call
                case 2: this->case2 = x.case2; break; // A_Fold
                case 3: this->case3 = x.case3; break; // A_Raise
            }
        } else {
            this->~Union2();
            new (this) Union2{x};
        }
        return *this;
    }
    __device__ Union2 & operator=(Union2 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // A_All_In
                case 1: this->case1 = std::move(x.case1); break; // A_Call
                case 2: this->case2 = std::move(x.case2); break; // A_Fold
                case 3: this->case3 = std::move(x.case3); break; // A_Raise
            }
        } else {
            this->~Union2();
            new (this) Union2{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union2() {
        switch(this->tag){
            case 0: this->case0.~Union2_0(); break; // A_All_In
            case 1: this->case1.~Union2_1(); break; // A_Call
            case 2: this->case2.~Union2_2(); break; // A_Fold
            case 3: this->case3.~Union2_3(); break; // A_Raise
        }
        this->tag = 255;
    }
};
struct Tuple0 {
    static_array<unsigned char,5> v0;
    char v1;
    __device__ Tuple0() = default;
    __device__ Tuple0(static_array<unsigned char,5> t0, char t1) : v0(t0), v1(t1) {}
};
struct Union1_0 { // CommunityCardsAre
    static_array_list<unsigned char,5> v0;
    __device__ Union1_0(static_array_list<unsigned char,5> t0) : v0(t0) {}
    __device__ Union1_0() = delete;
};
struct Union1_1 { // Fold
    int v0;
    int v1;
    __device__ Union1_1(int t0, int t1) : v0(t0), v1(t1) {}
    __device__ Union1_1() = delete;
};
struct Union1_2 { // PlayerAction
    Union2 v1;
    int v0;
    __device__ Union1_2(int t0, Union2 t1) : v0(t0), v1(t1) {}
    __device__ Union1_2() = delete;
};
struct Union1_3 { // PlayerGotCards
    static_array<unsigned char,2> v1;
    int v0;
    __device__ Union1_3(int t0, static_array<unsigned char,2> t1) : v0(t0), v1(t1) {}
    __device__ Union1_3() = delete;
};
struct Union1_4 { // Showdown
    static_array<Tuple0,2> v1;
    int v0;
    int v2;
    __device__ Union1_4(int t0, static_array<Tuple0,2> t1, int t2) : v0(t0), v1(t1), v2(t2) {}
    __device__ Union1_4() = delete;
};
struct Union1 {
    union {
        Union1_0 case0; // CommunityCardsAre
        Union1_1 case1; // Fold
        Union1_2 case2; // PlayerAction
        Union1_3 case3; // PlayerGotCards
        Union1_4 case4; // Showdown
    };
    unsigned char tag{255};
    __device__ Union1() {}
    __device__ Union1(Union1_0 t) : tag(0), case0(t) {} // CommunityCardsAre
    __device__ Union1(Union1_1 t) : tag(1), case1(t) {} // Fold
    __device__ Union1(Union1_2 t) : tag(2), case2(t) {} // PlayerAction
    __device__ Union1(Union1_3 t) : tag(3), case3(t) {} // PlayerGotCards
    __device__ Union1(Union1_4 t) : tag(4), case4(t) {} // Showdown
    __device__ Union1(Union1 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(x.case0); break; // CommunityCardsAre
            case 1: new (&this->case1) Union1_1(x.case1); break; // Fold
            case 2: new (&this->case2) Union1_2(x.case2); break; // PlayerAction
            case 3: new (&this->case3) Union1_3(x.case3); break; // PlayerGotCards
            case 4: new (&this->case4) Union1_4(x.case4); break; // Showdown
        }
    }
    __device__ Union1(Union1 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(std::move(x.case0)); break; // CommunityCardsAre
            case 1: new (&this->case1) Union1_1(std::move(x.case1)); break; // Fold
            case 2: new (&this->case2) Union1_2(std::move(x.case2)); break; // PlayerAction
            case 3: new (&this->case3) Union1_3(std::move(x.case3)); break; // PlayerGotCards
            case 4: new (&this->case4) Union1_4(std::move(x.case4)); break; // Showdown
        }
    }
    __device__ Union1 & operator=(Union1 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // CommunityCardsAre
                case 1: this->case1 = x.case1; break; // Fold
                case 2: this->case2 = x.case2; break; // PlayerAction
                case 3: this->case3 = x.case3; break; // PlayerGotCards
                case 4: this->case4 = x.case4; break; // Showdown
            }
        } else {
            this->~Union1();
            new (this) Union1{x};
        }
        return *this;
    }
    __device__ Union1 & operator=(Union1 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // CommunityCardsAre
                case 1: this->case1 = std::move(x.case1); break; // Fold
                case 2: this->case2 = std::move(x.case2); break; // PlayerAction
                case 3: this->case3 = std::move(x.case3); break; // PlayerGotCards
                case 4: this->case4 = std::move(x.case4); break; // Showdown
            }
        } else {
            this->~Union1();
            new (this) Union1{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union1() {
        switch(this->tag){
            case 0: this->case0.~Union1_0(); break; // CommunityCardsAre
            case 1: this->case1.~Union1_1(); break; // Fold
            case 2: this->case2.~Union1_2(); break; // PlayerAction
            case 3: this->case3.~Union1_3(); break; // PlayerGotCards
            case 4: this->case4.~Union1_4(); break; // Showdown
        }
        this->tag = 255;
    }
};
struct StackMut0 {
    unsigned long long v0;
    cooperative_groups::grid_group v1;
    static_array_list<Union1,128> v2;
    static_array<Union0,2> v3;
    static_array<float,2> v4;
    curandStatePhilox4_32_10_t v5;
    __device__ StackMut0() = default;
    __device__ StackMut0(unsigned long long t0, cooperative_groups::grid_group t1, static_array_list<Union1,128> t2, static_array<Union0,2> t3, static_array<float,2> t4, curandStatePhilox4_32_10_t t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
};
struct Union4_0 { // Flop
    static_array<unsigned char,3> v0;
    __device__ Union4_0(static_array<unsigned char,3> t0) : v0(t0) {}
    __device__ Union4_0() = delete;
};
struct Union4_1 { // Preflop
};
struct Union4_2 { // River
    static_array<unsigned char,5> v0;
    __device__ Union4_2(static_array<unsigned char,5> t0) : v0(t0) {}
    __device__ Union4_2() = delete;
};
struct Union4_3 { // Turn
    static_array<unsigned char,4> v0;
    __device__ Union4_3(static_array<unsigned char,4> t0) : v0(t0) {}
    __device__ Union4_3() = delete;
};
struct Union4 {
    union {
        Union4_0 case0; // Flop
        Union4_1 case1; // Preflop
        Union4_2 case2; // River
        Union4_3 case3; // Turn
    };
    unsigned char tag{255};
    __device__ Union4() {}
    __device__ Union4(Union4_0 t) : tag(0), case0(t) {} // Flop
    __device__ Union4(Union4_1 t) : tag(1), case1(t) {} // Preflop
    __device__ Union4(Union4_2 t) : tag(2), case2(t) {} // River
    __device__ Union4(Union4_3 t) : tag(3), case3(t) {} // Turn
    __device__ Union4(Union4 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(x.case0); break; // Flop
            case 1: new (&this->case1) Union4_1(x.case1); break; // Preflop
            case 2: new (&this->case2) Union4_2(x.case2); break; // River
            case 3: new (&this->case3) Union4_3(x.case3); break; // Turn
        }
    }
    __device__ Union4(Union4 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(std::move(x.case0)); break; // Flop
            case 1: new (&this->case1) Union4_1(std::move(x.case1)); break; // Preflop
            case 2: new (&this->case2) Union4_2(std::move(x.case2)); break; // River
            case 3: new (&this->case3) Union4_3(std::move(x.case3)); break; // Turn
        }
    }
    __device__ Union4 & operator=(Union4 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Flop
                case 1: this->case1 = x.case1; break; // Preflop
                case 2: this->case2 = x.case2; break; // River
                case 3: this->case3 = x.case3; break; // Turn
            }
        } else {
            this->~Union4();
            new (this) Union4{x};
        }
        return *this;
    }
    __device__ Union4 & operator=(Union4 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // Flop
                case 1: this->case1 = std::move(x.case1); break; // Preflop
                case 2: this->case2 = std::move(x.case2); break; // River
                case 3: this->case3 = std::move(x.case3); break; // Turn
            }
        } else {
            this->~Union4();
            new (this) Union4{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union4() {
        switch(this->tag){
            case 0: this->case0.~Union4_0(); break; // Flop
            case 1: this->case1.~Union4_1(); break; // Preflop
            case 2: this->case2.~Union4_2(); break; // River
            case 3: this->case3.~Union4_3(); break; // Turn
        }
        this->tag = 255;
    }
};
struct Union3_0 { // G_Flop
    static_array<static_array<unsigned char,2>,2> v1;
    static_array<int,2> v2;
    static_array<int,2> v4;
    Union4 v5;
    int v0;
    int v3;
    __device__ Union3_0(int t0, static_array<static_array<unsigned char,2>,2> t1, static_array<int,2> t2, int t3, static_array<int,2> t4, Union4 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union3_0() = delete;
};
struct Union3_1 { // G_Fold
    static_array<static_array<unsigned char,2>,2> v1;
    static_array<int,2> v2;
    static_array<int,2> v4;
    Union4 v5;
    int v0;
    int v3;
    __device__ Union3_1(int t0, static_array<static_array<unsigned char,2>,2> t1, static_array<int,2> t2, int t3, static_array<int,2> t4, Union4 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union3_1() = delete;
};
struct Union3_2 { // G_Preflop
};
struct Union3_3 { // G_River
    static_array<static_array<unsigned char,2>,2> v1;
    static_array<int,2> v2;
    static_array<int,2> v4;
    Union4 v5;
    int v0;
    int v3;
    __device__ Union3_3(int t0, static_array<static_array<unsigned char,2>,2> t1, static_array<int,2> t2, int t3, static_array<int,2> t4, Union4 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union3_3() = delete;
};
struct Union3_4 { // G_Round
    static_array<static_array<unsigned char,2>,2> v1;
    static_array<int,2> v2;
    static_array<int,2> v4;
    Union4 v5;
    int v0;
    int v3;
    __device__ Union3_4(int t0, static_array<static_array<unsigned char,2>,2> t1, static_array<int,2> t2, int t3, static_array<int,2> t4, Union4 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union3_4() = delete;
};
struct Union3_5 { // G_Round'
    static_array<static_array<unsigned char,2>,2> v1;
    static_array<int,2> v2;
    static_array<int,2> v4;
    Union4 v5;
    Union2 v6;
    int v0;
    int v3;
    __device__ Union3_5(int t0, static_array<static_array<unsigned char,2>,2> t1, static_array<int,2> t2, int t3, static_array<int,2> t4, Union4 t5, Union2 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __device__ Union3_5() = delete;
};
struct Union3_6 { // G_Showdown
    static_array<static_array<unsigned char,2>,2> v1;
    static_array<int,2> v2;
    static_array<int,2> v4;
    Union4 v5;
    int v0;
    int v3;
    __device__ Union3_6(int t0, static_array<static_array<unsigned char,2>,2> t1, static_array<int,2> t2, int t3, static_array<int,2> t4, Union4 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union3_6() = delete;
};
struct Union3_7 { // G_Turn
    static_array<static_array<unsigned char,2>,2> v1;
    static_array<int,2> v2;
    static_array<int,2> v4;
    Union4 v5;
    int v0;
    int v3;
    __device__ Union3_7(int t0, static_array<static_array<unsigned char,2>,2> t1, static_array<int,2> t2, int t3, static_array<int,2> t4, Union4 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union3_7() = delete;
};
struct Union3 {
    union {
        Union3_0 case0; // G_Flop
        Union3_1 case1; // G_Fold
        Union3_2 case2; // G_Preflop
        Union3_3 case3; // G_River
        Union3_4 case4; // G_Round
        Union3_5 case5; // G_Round'
        Union3_6 case6; // G_Showdown
        Union3_7 case7; // G_Turn
    };
    unsigned char tag{255};
    __device__ Union3() {}
    __device__ Union3(Union3_0 t) : tag(0), case0(t) {} // G_Flop
    __device__ Union3(Union3_1 t) : tag(1), case1(t) {} // G_Fold
    __device__ Union3(Union3_2 t) : tag(2), case2(t) {} // G_Preflop
    __device__ Union3(Union3_3 t) : tag(3), case3(t) {} // G_River
    __device__ Union3(Union3_4 t) : tag(4), case4(t) {} // G_Round
    __device__ Union3(Union3_5 t) : tag(5), case5(t) {} // G_Round'
    __device__ Union3(Union3_6 t) : tag(6), case6(t) {} // G_Showdown
    __device__ Union3(Union3_7 t) : tag(7), case7(t) {} // G_Turn
    __device__ Union3(Union3 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(x.case0); break; // G_Flop
            case 1: new (&this->case1) Union3_1(x.case1); break; // G_Fold
            case 2: new (&this->case2) Union3_2(x.case2); break; // G_Preflop
            case 3: new (&this->case3) Union3_3(x.case3); break; // G_River
            case 4: new (&this->case4) Union3_4(x.case4); break; // G_Round
            case 5: new (&this->case5) Union3_5(x.case5); break; // G_Round'
            case 6: new (&this->case6) Union3_6(x.case6); break; // G_Showdown
            case 7: new (&this->case7) Union3_7(x.case7); break; // G_Turn
        }
    }
    __device__ Union3(Union3 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(std::move(x.case0)); break; // G_Flop
            case 1: new (&this->case1) Union3_1(std::move(x.case1)); break; // G_Fold
            case 2: new (&this->case2) Union3_2(std::move(x.case2)); break; // G_Preflop
            case 3: new (&this->case3) Union3_3(std::move(x.case3)); break; // G_River
            case 4: new (&this->case4) Union3_4(std::move(x.case4)); break; // G_Round
            case 5: new (&this->case5) Union3_5(std::move(x.case5)); break; // G_Round'
            case 6: new (&this->case6) Union3_6(std::move(x.case6)); break; // G_Showdown
            case 7: new (&this->case7) Union3_7(std::move(x.case7)); break; // G_Turn
        }
    }
    __device__ Union3 & operator=(Union3 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // G_Flop
                case 1: this->case1 = x.case1; break; // G_Fold
                case 2: this->case2 = x.case2; break; // G_Preflop
                case 3: this->case3 = x.case3; break; // G_River
                case 4: this->case4 = x.case4; break; // G_Round
                case 5: this->case5 = x.case5; break; // G_Round'
                case 6: this->case6 = x.case6; break; // G_Showdown
                case 7: this->case7 = x.case7; break; // G_Turn
            }
        } else {
            this->~Union3();
            new (this) Union3{x};
        }
        return *this;
    }
    __device__ Union3 & operator=(Union3 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // G_Flop
                case 1: this->case1 = std::move(x.case1); break; // G_Fold
                case 2: this->case2 = std::move(x.case2); break; // G_Preflop
                case 3: this->case3 = std::move(x.case3); break; // G_River
                case 4: this->case4 = std::move(x.case4); break; // G_Round
                case 5: this->case5 = std::move(x.case5); break; // G_Round'
                case 6: this->case6 = std::move(x.case6); break; // G_Showdown
                case 7: this->case7 = std::move(x.case7); break; // G_Turn
            }
        } else {
            this->~Union3();
            new (this) Union3{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union3() {
        switch(this->tag){
            case 0: this->case0.~Union3_0(); break; // G_Flop
            case 1: this->case1.~Union3_1(); break; // G_Fold
            case 2: this->case2.~Union3_2(); break; // G_Preflop
            case 3: this->case3.~Union3_3(); break; // G_River
            case 4: this->case4.~Union3_4(); break; // G_Round
            case 5: this->case5.~Union3_5(); break; // G_Round'
            case 6: this->case6.~Union3_6(); break; // G_Showdown
            case 7: this->case7.~Union3_7(); break; // G_Turn
        }
        this->tag = 255;
    }
};
struct Union5_0 { // None
};
struct Union5_1 { // Some
    Union3 v0;
    __device__ Union5_1(Union3 t0) : v0(t0) {}
    __device__ Union5_1() = delete;
};
struct Union5 {
    union {
        Union5_0 case0; // None
        Union5_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union5() {}
    __device__ Union5(Union5_0 t) : tag(0), case0(t) {} // None
    __device__ Union5(Union5_1 t) : tag(1), case1(t) {} // Some
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
    __device__ Union5 & operator=(Union5 && x) {
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
    __device__ ~Union5() {
        switch(this->tag){
            case 0: this->case0.~Union5_0(); break; // None
            case 1: this->case1.~Union5_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union6_0 { // T_none
};
struct Union6_1 { // T_round
    static_array<static_array<unsigned char,2>,2> v1;
    static_array<int,2> v2;
    static_array<int,2> v4;
    Union4 v5;
    Union2 v6;
    int v0;
    int v3;
    __device__ Union6_1(int t0, static_array<static_array<unsigned char,2>,2> t1, static_array<int,2> t2, int t3, static_array<int,2> t4, Union4 t5, Union2 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __device__ Union6_1() = delete;
};
struct Union6_2 { // T_some
    Union3 v0;
    __device__ Union6_2(Union3 t0) : v0(t0) {}
    __device__ Union6_2() = delete;
};
struct Union6 {
    union {
        Union6_0 case0; // T_none
        Union6_1 case1; // T_round
        Union6_2 case2; // T_some
    };
    unsigned char tag{255};
    __device__ Union6() {}
    __device__ Union6(Union6_0 t) : tag(0), case0(t) {} // T_none
    __device__ Union6(Union6_1 t) : tag(1), case1(t) {} // T_round
    __device__ Union6(Union6_2 t) : tag(2), case2(t) {} // T_some
    __device__ Union6(Union6 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(x.case0); break; // T_none
            case 1: new (&this->case1) Union6_1(x.case1); break; // T_round
            case 2: new (&this->case2) Union6_2(x.case2); break; // T_some
        }
    }
    __device__ Union6(Union6 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(std::move(x.case0)); break; // T_none
            case 1: new (&this->case1) Union6_1(std::move(x.case1)); break; // T_round
            case 2: new (&this->case2) Union6_2(std::move(x.case2)); break; // T_some
        }
    }
    __device__ Union6 & operator=(Union6 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // T_none
                case 1: this->case1 = x.case1; break; // T_round
                case 2: this->case2 = x.case2; break; // T_some
            }
        } else {
            this->~Union6();
            new (this) Union6{x};
        }
        return *this;
    }
    __device__ Union6 & operator=(Union6 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // T_none
                case 1: this->case1 = std::move(x.case1); break; // T_round
                case 2: this->case2 = std::move(x.case2); break; // T_some
            }
        } else {
            this->~Union6();
            new (this) Union6{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union6() {
        switch(this->tag){
            case 0: this->case0.~Union6_0(); break; // T_none
            case 1: this->case1.~Union6_1(); break; // T_round
            case 2: this->case2.~Union6_2(); break; // T_some
        }
        this->tag = 255;
    }
};
struct Tuple1 {
    static_array<unsigned char,3> v0;
    unsigned long long v1;
    __device__ Tuple1() = default;
    __device__ Tuple1(static_array<unsigned char,3> t0, unsigned long long t1) : v0(t0), v1(t1) {}
};
struct Tuple2 {
    unsigned long long v1;
    int v0;
    __device__ Tuple2() = default;
    __device__ Tuple2(int t0, unsigned long long t1) : v0(t0), v1(t1) {}
};
struct Tuple3 {
    unsigned long long v1;
    unsigned char v0;
    __device__ Tuple3() = default;
    __device__ Tuple3(unsigned char t0, unsigned long long t1) : v0(t0), v1(t1) {}
};
struct Tuple4 {
    int v0;
    int v1;
    __device__ Tuple4() = default;
    __device__ Tuple4(int t0, int t1) : v0(t0), v1(t1) {}
};
struct Tuple5 {
    static_array<unsigned char,2> v0;
    unsigned long long v1;
    __device__ Tuple5() = default;
    __device__ Tuple5(static_array<unsigned char,2> t0, unsigned long long t1) : v0(t0), v1(t1) {}
};
struct Tuple6 {
    static_array<unsigned char,1> v0;
    unsigned long long v1;
    __device__ Tuple6() = default;
    __device__ Tuple6(static_array<unsigned char,1> t0, unsigned long long t1) : v0(t0), v1(t1) {}
};
struct Tuple7 {
    Union2 v0;
    float v1;
    __device__ Tuple7() = default;
    __device__ Tuple7(Union2 t0, float t1) : v0(t0), v1(t1) {}
};
struct Tuple8 {
    int v1;
    bool v0;
    __device__ Tuple8() = default;
    __device__ Tuple8(bool t0, int t1) : v0(t0), v1(t1) {}
};
struct Tuple9 {
    int v0;
    int v1;
    int v2;
    __device__ Tuple9() = default;
    __device__ Tuple9(int t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
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
    __device__ Union7() {}
    __device__ Union7(Union7_0 t) : tag(0), case0(t) {} // Eq
    __device__ Union7(Union7_1 t) : tag(1), case1(t) {} // Gt
    __device__ Union7(Union7_2 t) : tag(2), case2(t) {} // Lt
    __device__ Union7(Union7 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(x.case0); break; // Eq
            case 1: new (&this->case1) Union7_1(x.case1); break; // Gt
            case 2: new (&this->case2) Union7_2(x.case2); break; // Lt
        }
    }
    __device__ Union7(Union7 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(std::move(x.case0)); break; // Eq
            case 1: new (&this->case1) Union7_1(std::move(x.case1)); break; // Gt
            case 2: new (&this->case2) Union7_2(std::move(x.case2)); break; // Lt
        }
    }
    __device__ Union7 & operator=(Union7 & x) {
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
    __device__ Union7 & operator=(Union7 && x) {
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
    __device__ ~Union7() {
        switch(this->tag){
            case 0: this->case0.~Union7_0(); break; // Eq
            case 1: this->case1.~Union7_1(); break; // Gt
            case 2: this->case2.~Union7_2(); break; // Lt
        }
        this->tag = 255;
    }
};
struct Tuple10 {
    int v0;
    int v1;
    unsigned char v2;
    __device__ Tuple10() = default;
    __device__ Tuple10(int t0, int t1, unsigned char t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Union8_0 { // None
};
struct Union8_1 { // Some
    static_array<unsigned char,5> v0;
    __device__ Union8_1(static_array<unsigned char,5> t0) : v0(t0) {}
    __device__ Union8_1() = delete;
};
struct Union8 {
    union {
        Union8_0 case0; // None
        Union8_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union8() {}
    __device__ Union8(Union8_0 t) : tag(0), case0(t) {} // None
    __device__ Union8(Union8_1 t) : tag(1), case1(t) {} // Some
    __device__ Union8(Union8 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union8_0(x.case0); break; // None
            case 1: new (&this->case1) Union8_1(x.case1); break; // Some
        }
    }
    __device__ Union8(Union8 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union8_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union8_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union8 & operator=(Union8 & x) {
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
    __device__ Union8 & operator=(Union8 && x) {
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
    __device__ ~Union8() {
        switch(this->tag){
            case 0: this->case0.~Union8_0(); break; // None
            case 1: this->case1.~Union8_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Tuple11 {
    Union7 v1;
    int v0;
    __device__ Tuple11() = default;
    __device__ Tuple11(int t0, Union7 t1) : v0(t0), v1(t1) {}
};
struct Tuple12 {
    int v0;
    int v1;
    int v2;
    unsigned char v3;
    __device__ Tuple12() = default;
    __device__ Tuple12(int t0, int t1, int t2, unsigned char t3) : v0(t0), v1(t1), v2(t2), v3(t3) {}
};
struct Union9_0 { // None
};
struct Union9_1 { // Some
    static_array<unsigned char,4> v0;
    static_array<unsigned char,3> v1;
    __device__ Union9_1(static_array<unsigned char,4> t0, static_array<unsigned char,3> t1) : v0(t0), v1(t1) {}
    __device__ Union9_1() = delete;
};
struct Union9 {
    union {
        Union9_0 case0; // None
        Union9_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union9() {}
    __device__ Union9(Union9_0 t) : tag(0), case0(t) {} // None
    __device__ Union9(Union9_1 t) : tag(1), case1(t) {} // Some
    __device__ Union9(Union9 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union9_0(x.case0); break; // None
            case 1: new (&this->case1) Union9_1(x.case1); break; // Some
        }
    }
    __device__ Union9(Union9 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union9_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union9_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union9 & operator=(Union9 & x) {
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
    __device__ Union9 & operator=(Union9 && x) {
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
    __device__ ~Union9() {
        switch(this->tag){
            case 0: this->case0.~Union9_0(); break; // None
            case 1: this->case1.~Union9_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union10_0 { // None
};
struct Union10_1 { // Some
    static_array<unsigned char,3> v0;
    static_array<unsigned char,4> v1;
    __device__ Union10_1(static_array<unsigned char,3> t0, static_array<unsigned char,4> t1) : v0(t0), v1(t1) {}
    __device__ Union10_1() = delete;
};
struct Union10 {
    union {
        Union10_0 case0; // None
        Union10_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union10() {}
    __device__ Union10(Union10_0 t) : tag(0), case0(t) {} // None
    __device__ Union10(Union10_1 t) : tag(1), case1(t) {} // Some
    __device__ Union10(Union10 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union10_0(x.case0); break; // None
            case 1: new (&this->case1) Union10_1(x.case1); break; // Some
        }
    }
    __device__ Union10(Union10 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union10_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union10_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union10 & operator=(Union10 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union10();
            new (this) Union10{x};
        }
        return *this;
    }
    __device__ Union10 & operator=(Union10 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union10();
            new (this) Union10{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union10() {
        switch(this->tag){
            case 0: this->case0.~Union10_0(); break; // None
            case 1: this->case1.~Union10_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union11_0 { // None
};
struct Union11_1 { // Some
    static_array<unsigned char,2> v0;
    static_array<unsigned char,2> v1;
    __device__ Union11_1(static_array<unsigned char,2> t0, static_array<unsigned char,2> t1) : v0(t0), v1(t1) {}
    __device__ Union11_1() = delete;
};
struct Union11 {
    union {
        Union11_0 case0; // None
        Union11_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union11() {}
    __device__ Union11(Union11_0 t) : tag(0), case0(t) {} // None
    __device__ Union11(Union11_1 t) : tag(1), case1(t) {} // Some
    __device__ Union11(Union11 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union11_0(x.case0); break; // None
            case 1: new (&this->case1) Union11_1(x.case1); break; // Some
        }
    }
    __device__ Union11(Union11 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union11_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union11_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union11 & operator=(Union11 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union11();
            new (this) Union11{x};
        }
        return *this;
    }
    __device__ Union11 & operator=(Union11 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union11();
            new (this) Union11{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union11() {
        switch(this->tag){
            case 0: this->case0.~Union11_0(); break; // None
            case 1: this->case1.~Union11_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union12_0 { // None
};
struct Union12_1 { // Some
    static_array<unsigned char,2> v0;
    static_array<unsigned char,5> v1;
    __device__ Union12_1(static_array<unsigned char,2> t0, static_array<unsigned char,5> t1) : v0(t0), v1(t1) {}
    __device__ Union12_1() = delete;
};
struct Union12 {
    union {
        Union12_0 case0; // None
        Union12_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union12() {}
    __device__ Union12(Union12_0 t) : tag(0), case0(t) {} // None
    __device__ Union12(Union12_1 t) : tag(1), case1(t) {} // Some
    __device__ Union12(Union12 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union12_0(x.case0); break; // None
            case 1: new (&this->case1) Union12_1(x.case1); break; // Some
        }
    }
    __device__ Union12(Union12 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union12_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union12_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union12 & operator=(Union12 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union12();
            new (this) Union12{x};
        }
        return *this;
    }
    __device__ Union12 & operator=(Union12 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union12();
            new (this) Union12{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union12() {
        switch(this->tag){
            case 0: this->case0.~Union12_0(); break; // None
            case 1: this->case1.~Union12_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union13_0 { // None
};
struct Union13_1 { // Some
    static_array<unsigned char,2> v0;
    static_array<unsigned char,3> v1;
    __device__ Union13_1(static_array<unsigned char,2> t0, static_array<unsigned char,3> t1) : v0(t0), v1(t1) {}
    __device__ Union13_1() = delete;
};
struct Union13 {
    union {
        Union13_0 case0; // None
        Union13_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union13() {}
    __device__ Union13(Union13_0 t) : tag(0), case0(t) {} // None
    __device__ Union13(Union13_1 t) : tag(1), case1(t) {} // Some
    __device__ Union13(Union13 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union13_0(x.case0); break; // None
            case 1: new (&this->case1) Union13_1(x.case1); break; // Some
        }
    }
    __device__ Union13(Union13 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union13_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union13_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union13 & operator=(Union13 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union13();
            new (this) Union13{x};
        }
        return *this;
    }
    __device__ Union13 & operator=(Union13 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union13();
            new (this) Union13{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union13() {
        switch(this->tag){
            case 0: this->case0.~Union13_0(); break; // None
            case 1: this->case1.~Union13_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Closure0 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Closure1 {
    __device__ int operator()(int tup0, int tup1){
        int v0 = tup0; int v1 = tup1;
        int v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Closure2 {
    __device__ bool operator()(bool tup0, bool tup1){
        bool v0 = tup0; bool v1 = tup1;
        bool v2;
        v2 = v0 || v1;
        return v2;
    }
};
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 4;
    return v1;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 8;
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 2;
    return v1;
}
__device__ inline bool while_method_3(Union5 v0){
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
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ inline bool while_method_4(int v0){
    bool v1;
    v1 = v0 < 3;
    return v1;
}
__device__ unsigned int loop_3(unsigned int v0, curandStatePhilox4_32_10_t & v1){
    unsigned int v2;
    v2 = curand(&v1);
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
__device__ Tuple3 draw_card_2(curandStatePhilox4_32_10_t & v0, unsigned long long v1){
    int v2;
    v2 = __popcll(v1);
    unsigned int v3;
    v3 = (unsigned int)v2;
    unsigned int v4;
    v4 = loop_3(v3, v0);
    int v5;
    v5 = (int)v4;
    unsigned int v6;
    v6 = (unsigned int)v1;
    unsigned long long v7;
    v7 = v1 >> 32;
    unsigned int v8;
    v8 = (unsigned int)v7;
    int v9;
    v9 = __popc(v6);
    bool v10;
    v10 = v5 < v9;
    unsigned int v22;
    if (v10){
        int v11;
        v11 = v5 + 1;
        unsigned int v12;
        v12 = __fns(v6,0u,v11);
        v22 = v12;
    } else {
        int v13;
        v13 = v5 - v9;
        int v14;
        v14 = __popc(v8);
        bool v15;
        v15 = v13 < v14;
        if (v15){
            int v16;
            v16 = v13 + 1;
            unsigned int v17;
            v17 = __fns(v8,0u,v16);
            unsigned int v18;
            v18 = v17 + 32u;
            v22 = v18;
        } else {
            int v19;
            v19 = v13 - v14;
            printf("%s\n", "Cannot find the n-th set bit.");
            __trap();
        }
    }
    unsigned char v23;
    v23 = (unsigned char)v22;
    int v24;
    v24 = (int)v22;
    unsigned long long v25;
    v25 = 1ull << v24;
    unsigned long long v26;
    v26 = v1 ^ v25;
    return Tuple3{v23, v26};
}
__device__ Tuple1 draw_cards_1(curandStatePhilox4_32_10_t & v0, unsigned long long v1){
    static_array<unsigned char,3> v2;
    int v4; unsigned long long v5;
    Tuple2 tmp0 = Tuple2{0, v1};
    v4 = tmp0.v0; v5 = tmp0.v1;
    while (while_method_4(v4)){
        unsigned char v7; unsigned long long v8;
        Tuple3 tmp1 = draw_card_2(v0, v5);
        v7 = tmp1.v0; v8 = tmp1.v1;
        v2[v4] = v7;
        v5 = v8;
        v4 += 1 ;
    }
    return Tuple1{v2, v5};
}
__device__ inline bool while_method_5(int v0){
    bool v1;
    v1 = v0 < 5;
    return v1;
}
__device__ static_array_list<unsigned char,5> get_community_cards_4(Union4 v0, static_array<unsigned char,3> v1){
    static_array_list<unsigned char,5> v2;
    v2 = static_array_list<unsigned char,5>{};
    switch (v0.tag) {
        case 0: { // Flop
            static_array<unsigned char,3> v4 = v0.case0.v0;
            int v5;
            v5 = 0;
            while (while_method_4(v5)){
                bool v7;
                v7 = 0 <= v5;
                bool v9;
                if (v7){
                    bool v8;
                    v8 = v5 < 3;
                    v9 = v8;
                } else {
                    v9 = false;
                }
                bool v10;
                v10 = v9 == false;
                if (v10){
                    assert("Index must be in range." && v9);
                } else {
                }
                unsigned char v12;
                v12 = v4[v5];
                v2.push(v12);
                v5 += 1 ;
            }
            break;
        }
        case 1: { // Preflop
            break;
        }
        case 2: { // River
            static_array<unsigned char,5> v24 = v0.case2.v0;
            int v25;
            v25 = 0;
            while (while_method_5(v25)){
                bool v27;
                v27 = 0 <= v25;
                bool v29;
                if (v27){
                    bool v28;
                    v28 = v25 < 5;
                    v29 = v28;
                } else {
                    v29 = false;
                }
                bool v30;
                v30 = v29 == false;
                if (v30){
                    assert("Index must be in range." && v29);
                } else {
                }
                unsigned char v32;
                v32 = v24[v25];
                v2.push(v32);
                v25 += 1 ;
            }
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4> v14 = v0.case3.v0;
            int v15;
            v15 = 0;
            while (while_method_0(v15)){
                bool v17;
                v17 = 0 <= v15;
                bool v19;
                if (v17){
                    bool v18;
                    v18 = v15 < 4;
                    v19 = v18;
                } else {
                    v19 = false;
                }
                bool v20;
                v20 = v19 == false;
                if (v20){
                    assert("Index must be in range." && v19);
                } else {
                }
                unsigned char v22;
                v22 = v14[v15];
                v2.push(v22);
                v15 += 1 ;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v34;
    v34 = 0;
    while (while_method_4(v34)){
        bool v36;
        v36 = 0 <= v34;
        bool v38;
        if (v36){
            bool v37;
            v37 = v34 < 3;
            v38 = v37;
        } else {
            v38 = false;
        }
        bool v39;
        v39 = v38 == false;
        if (v39){
            assert("Index must be in range." && v38);
        } else {
        }
        unsigned char v41;
        v41 = v1[v34];
        v2.push(v41);
        v34 += 1 ;
    }
    return v2;
}
__device__ bool player_can_act_6(int v0, static_array<static_array<unsigned char,2>,2> v1, static_array<int,2> v2, int v3, static_array<int,2> v4, Union4 v5){
    int v6;
    v6 = v3 % 2;
    bool v7;
    v7 = 0 <= v6;
    bool v9;
    if (v7){
        bool v8;
        v8 = v6 < 2;
        v9 = v8;
    } else {
        v9 = false;
    }
    bool v10;
    v10 = v9 == false;
    if (v10){
        assert("Index must be in range." && v9);
    } else {
    }
    int v12;
    v12 = v4[v6];
    bool v14;
    v14 = v12 > 0;
    bool v16;
    if (v7){
        bool v15;
        v15 = v6 < 2;
        v16 = v15;
    } else {
        v16 = false;
    }
    bool v17;
    v17 = v16 == false;
    if (v17){
        assert("Index must be in range." && v16);
    } else {
    }
    int v19;
    v19 = v2[v6];
    int v21;
    v21 = v2[0];
    int v23; int v24;
    Tuple4 tmp3 = Tuple4{1, v21};
    v23 = tmp3.v0; v24 = tmp3.v1;
    while (while_method_2(v23)){
        bool v26;
        v26 = 0 <= v23;
        bool v28;
        if (v26){
            bool v27;
            v27 = v23 < 2;
            v28 = v27;
        } else {
            v28 = false;
        }
        bool v29;
        v29 = v28 == false;
        if (v29){
            assert("Index must be in range." && v28);
        } else {
        }
        int v31;
        v31 = v2[v23];
        bool v33;
        v33 = v24 >= v31;
        int v34;
        if (v33){
            v34 = v24;
        } else {
            v34 = v31;
        }
        v24 = v34;
        v23 += 1 ;
    }
    bool v35;
    v35 = v19 < v24;
    int v36; int v37;
    Tuple4 tmp4 = Tuple4{0, 0};
    v36 = tmp4.v0; v37 = tmp4.v1;
    while (while_method_2(v36)){
        bool v39;
        v39 = 0 <= v36;
        bool v41;
        if (v39){
            bool v40;
            v40 = v36 < 2;
            v41 = v40;
        } else {
            v41 = false;
        }
        bool v42;
        v42 = v41 == false;
        if (v42){
            assert("Index must be in range." && v41);
        } else {
        }
        int v44;
        v44 = v4[v36];
        bool v46;
        v46 = 0 < v44;
        int v47;
        if (v46){
            v47 = 1;
        } else {
            v47 = 0;
        }
        int v48;
        v48 = v37 + v47;
        v37 = v48;
        v36 += 1 ;
    }
    if (v14){
        if (v35){
            return true;
        } else {
            bool v49;
            v49 = v3 < 2;
            if (v49){
                bool v50;
                v50 = 0 < v37;
                return v50;
            } else {
                return false;
            }
        }
    } else {
        return false;
    }
}
__device__ Union3 go_next_street_7(int v0, static_array<static_array<unsigned char,2>,2> v1, static_array<int,2> v2, int v3, static_array<int,2> v4, Union4 v5){
    switch (v5.tag) {
        case 0: { // Flop
            static_array<unsigned char,3> v7 = v5.case0.v0;
            return Union3{Union3_7{v0, v1, v2, v3, v4, v5}};
            break;
        }
        case 1: { // Preflop
            return Union3{Union3_0{v0, v1, v2, v3, v4, v5}};
            break;
        }
        case 2: { // River
            static_array<unsigned char,5> v11 = v5.case2.v0;
            return Union3{Union3_6{v0, v1, v2, v3, v4, v5}};
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4> v9 = v5.case3.v0;
            return Union3{Union3_3{v0, v1, v2, v3, v4, v5}};
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ Union3 try_round_5(int v0, static_array<static_array<unsigned char,2>,2> v1, static_array<int,2> v2, int v3, static_array<int,2> v4, Union4 v5){
    int v6;
    v6 = v3 + 1;
    bool v7;
    v7 = player_can_act_6(v0, v1, v2, v3, v4, v5);
    if (v7){
        return Union3{Union3_4{v0, v1, v2, v3, v4, v5}};
    } else {
        bool v9;
        v9 = player_can_act_6(v0, v1, v2, v6, v4, v5);
        if (v9){
            return Union3{Union3_4{v0, v1, v2, v6, v4, v5}};
        } else {
            return go_next_street_7(v0, v1, v2, v3, v4, v5);
        }
    }
}
__device__ Tuple5 draw_cards_8(curandStatePhilox4_32_10_t & v0, unsigned long long v1){
    static_array<unsigned char,2> v2;
    int v4; unsigned long long v5;
    Tuple2 tmp5 = Tuple2{0, v1};
    v4 = tmp5.v0; v5 = tmp5.v1;
    while (while_method_2(v4)){
        unsigned char v7; unsigned long long v8;
        Tuple3 tmp6 = draw_card_2(v0, v5);
        v7 = tmp6.v0; v8 = tmp6.v1;
        v2[v4] = v7;
        v5 = v8;
        v4 += 1 ;
    }
    return Tuple5{v2, v5};
}
__device__ inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 1;
    return v1;
}
__device__ Tuple6 draw_cards_9(curandStatePhilox4_32_10_t & v0, unsigned long long v1){
    static_array<unsigned char,1> v2;
    int v4; unsigned long long v5;
    Tuple2 tmp9 = Tuple2{0, v1};
    v4 = tmp9.v0; v5 = tmp9.v1;
    while (while_method_6(v4)){
        unsigned char v7; unsigned long long v8;
        Tuple3 tmp10 = draw_card_2(v0, v5);
        v7 = tmp10.v0; v8 = tmp10.v1;
        v2[v4] = v7;
        v5 = v8;
        v4 += 1 ;
    }
    return Tuple6{v2, v5};
}
__device__ static_array_list<unsigned char,5> get_community_cards_10(Union4 v0, static_array<unsigned char,1> v1){
    static_array_list<unsigned char,5> v2;
    v2 = static_array_list<unsigned char,5>{};
    switch (v0.tag) {
        case 0: { // Flop
            static_array<unsigned char,3> v4 = v0.case0.v0;
            int v5;
            v5 = 0;
            while (while_method_4(v5)){
                bool v7;
                v7 = 0 <= v5;
                bool v9;
                if (v7){
                    bool v8;
                    v8 = v5 < 3;
                    v9 = v8;
                } else {
                    v9 = false;
                }
                bool v10;
                v10 = v9 == false;
                if (v10){
                    assert("Index must be in range." && v9);
                } else {
                }
                unsigned char v12;
                v12 = v4[v5];
                v2.push(v12);
                v5 += 1 ;
            }
            break;
        }
        case 1: { // Preflop
            break;
        }
        case 2: { // River
            static_array<unsigned char,5> v24 = v0.case2.v0;
            int v25;
            v25 = 0;
            while (while_method_5(v25)){
                bool v27;
                v27 = 0 <= v25;
                bool v29;
                if (v27){
                    bool v28;
                    v28 = v25 < 5;
                    v29 = v28;
                } else {
                    v29 = false;
                }
                bool v30;
                v30 = v29 == false;
                if (v30){
                    assert("Index must be in range." && v29);
                } else {
                }
                unsigned char v32;
                v32 = v24[v25];
                v2.push(v32);
                v25 += 1 ;
            }
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4> v14 = v0.case3.v0;
            int v15;
            v15 = 0;
            while (while_method_0(v15)){
                bool v17;
                v17 = 0 <= v15;
                bool v19;
                if (v17){
                    bool v18;
                    v18 = v15 < 4;
                    v19 = v18;
                } else {
                    v19 = false;
                }
                bool v20;
                v20 = v19 == false;
                if (v20){
                    assert("Index must be in range." && v19);
                } else {
                }
                unsigned char v22;
                v22 = v14[v15];
                v2.push(v22);
                v15 += 1 ;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v34;
    v34 = 0;
    while (while_method_6(v34)){
        bool v36;
        v36 = 0 <= v34;
        bool v38;
        if (v36){
            bool v37;
            v37 = v34 < 1;
            v38 = v37;
        } else {
            v38 = false;
        }
        bool v39;
        v39 = v38 == false;
        if (v39){
            assert("Index must be in range." && v38);
        } else {
        }
        unsigned char v41;
        v41 = v1[v34];
        v2.push(v41);
        v34 += 1 ;
    }
    return v2;
}
__device__ inline bool while_method_7(int v0){
    bool v1;
    v1 = v0 < 6;
    return v1;
}
__device__ inline bool while_method_8(static_array<float,6> v0, int v1){
    bool v2;
    v2 = v1 < 6;
    return v2;
}
__device__ inline bool while_method_9(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    return v2;
}
__device__ int loop_14(static_array<float,6> v0, float v1, int v2){
    bool v3;
    v3 = v2 < 6;
    if (v3){
        bool v4;
        v4 = 0 <= v2;
        bool v5;
        v5 = v4 && v3;
        bool v6;
        v6 = v5 == false;
        if (v6){
            assert("Index must be in range." && v5);
        } else {
        }
        float v8;
        v8 = v0[v2];
        bool v10;
        v10 = v1 <= v8;
        if (v10){
            return v2;
        } else {
            int v11;
            v11 = v2 + 1;
            return loop_14(v0, v1, v11);
        }
    } else {
        return 5;
    }
}
__device__ int pick_discrete__13(static_array<float,6> v0, float v1){
    static_array<float,6> v2;
    int v4;
    v4 = 0;
    while (while_method_7(v4)){
        bool v6;
        v6 = 0 <= v4;
        bool v8;
        if (v6){
            bool v7;
            v7 = v4 < 6;
            v8 = v7;
        } else {
            v8 = false;
        }
        bool v9;
        v9 = v8 == false;
        if (v9){
            assert("Index must be in range." && v8);
        } else {
        }
        float v11;
        v11 = v0[v4];
        v2[v4] = v11;
        v4 += 1 ;
    }
    int v13;
    v13 = 1;
    while (while_method_8(v2, v13)){
        int v15;
        v15 = 6;
        while (while_method_9(v13, v15)){
            v15 -= 1 ;
            int v17;
            v17 = v15 - v13;
            bool v18;
            v18 = 0 <= v17;
            bool v20;
            if (v18){
                bool v19;
                v19 = v17 < 6;
                v20 = v19;
            } else {
                v20 = false;
            }
            bool v21;
            v21 = v20 == false;
            if (v21){
                assert("Index must be in range." && v20);
            } else {
            }
            float v23;
            v23 = v2[v17];
            bool v25;
            v25 = 0 <= v15;
            bool v27;
            if (v25){
                bool v26;
                v26 = v15 < 6;
                v27 = v26;
            } else {
                v27 = false;
            }
            bool v28;
            v28 = v27 == false;
            if (v28){
                assert("Index must be in range." && v27);
            } else {
            }
            float v30;
            v30 = v2[v15];
            float v32;
            v32 = v23 + v30;
            v2[v15] = v32;
        }
        int v33;
        v33 = v13 * 2;
        v13 = v33;
    }
    float v34;
    v34 = v2[5];
    float v36;
    v36 = v1 * v34;
    int v37;
    v37 = 0;
    return loop_14(v2, v36, v37);
}
__device__ int sample_discrete__12(static_array<float,6> v0, curandStatePhilox4_32_10_t & v1){
    float v2;
    v2 = curand_uniform(&v1);
    return pick_discrete__13(v0, v2);
}
__device__ Union2 sample_discrete_11(static_array<Tuple7,6> v0, curandStatePhilox4_32_10_t & v1){
    static_array<float,6> v2;
    int v4;
    v4 = 0;
    while (while_method_7(v4)){
        bool v6;
        v6 = 0 <= v4;
        bool v8;
        if (v6){
            bool v7;
            v7 = v4 < 6;
            v8 = v7;
        } else {
            v8 = false;
        }
        bool v9;
        v9 = v8 == false;
        if (v9){
            assert("Index must be in range." && v8);
        } else {
        }
        Union2 v11; float v12;
        Tuple7 tmp14 = v0[v4];
        v11 = tmp14.v0; v12 = tmp14.v1;
        v2[v4] = v12;
        v4 += 1 ;
    }
    int v15;
    v15 = sample_discrete__12(v2, v1);
    bool v16;
    v16 = 0 <= v15;
    bool v18;
    if (v16){
        bool v17;
        v17 = v15 < 6;
        v18 = v17;
    } else {
        v18 = false;
    }
    bool v19;
    v19 = v18 == false;
    if (v19){
        assert("Index must be in range." && v18);
    } else {
    }
    Union2 v21; float v22;
    Tuple7 tmp15 = v0[v15];
    v21 = tmp15.v0; v22 = tmp15.v1;
    return v21;
}
__device__ inline bool while_method_10(int v0){
    bool v1;
    v1 = v0 < 7;
    return v1;
}
__device__ inline bool while_method_11(static_array<unsigned char,7> v0, bool v1, int v2){
    bool v3;
    v3 = v2 < 7;
    return v3;
}
__device__ inline bool while_method_12(static_array<unsigned char,7> v0, int v1){
    bool v2;
    v2 = v1 < 7;
    return v2;
}
__device__ inline bool while_method_13(int v0, int v1, int v2, int v3){
    bool v4;
    v4 = v3 < v0;
    return v4;
}
__device__ Tuple0 score_15(static_array<unsigned char,7> v0){
    static_array<unsigned char,7> v1;
    int v3;
    v3 = 0;
    while (while_method_10(v3)){
        bool v5;
        v5 = 0 <= v3;
        bool v7;
        if (v5){
            bool v6;
            v6 = v3 < 7;
            v7 = v6;
        } else {
            v7 = false;
        }
        bool v8;
        v8 = v7 == false;
        if (v8){
            assert("Index must be in range." && v7);
        } else {
        }
        unsigned char v10;
        v10 = v0[v3];
        v1[v3] = v10;
        v3 += 1 ;
    }
    static_array<unsigned char,7> v12;
    bool v14; int v15;
    Tuple8 tmp18 = Tuple8{true, 1};
    v14 = tmp18.v0; v15 = tmp18.v1;
    while (while_method_11(v1, v14, v15)){
        int v17;
        v17 = 0;
        while (while_method_12(v1, v17)){
            int v19;
            v19 = v17 + v15;
            bool v20;
            v20 = v19 < 7;
            int v21;
            if (v20){
                v21 = v19;
            } else {
                v21 = 7;
            }
            int v22;
            v22 = v15 * 2;
            int v23;
            v23 = v17 + v22;
            bool v24;
            v24 = v23 < 7;
            int v25;
            if (v24){
                v25 = v23;
            } else {
                v25 = 7;
            }
            int v26; int v27; int v28;
            Tuple9 tmp19 = Tuple9{v17, v21, v17};
            v26 = tmp19.v0; v27 = tmp19.v1; v28 = tmp19.v2;
            while (while_method_13(v25, v26, v27, v28)){
                bool v30;
                v30 = v26 < v21;
                bool v32;
                if (v30){
                    bool v31;
                    v31 = v27 < v25;
                    v32 = v31;
                } else {
                    v32 = false;
                }
                unsigned char v122; int v123; int v124;
                if (v32){
                    unsigned char v47;
                    if (v14){
                        bool v33;
                        v33 = 0 <= v26;
                        bool v35;
                        if (v33){
                            bool v34;
                            v34 = v26 < 7;
                            v35 = v34;
                        } else {
                            v35 = false;
                        }
                        bool v36;
                        v36 = v35 == false;
                        if (v36){
                            assert("Index must be in range." && v35);
                        } else {
                        }
                        unsigned char v38;
                        v38 = v1[v26];
                        v47 = v38;
                    } else {
                        bool v40;
                        v40 = 0 <= v26;
                        bool v42;
                        if (v40){
                            bool v41;
                            v41 = v26 < 7;
                            v42 = v41;
                        } else {
                            v42 = false;
                        }
                        bool v43;
                        v43 = v42 == false;
                        if (v43){
                            assert("Index must be in range." && v42);
                        } else {
                        }
                        unsigned char v45;
                        v45 = v12[v26];
                        v47 = v45;
                    }
                    unsigned char v62;
                    if (v14){
                        bool v48;
                        v48 = 0 <= v27;
                        bool v50;
                        if (v48){
                            bool v49;
                            v49 = v27 < 7;
                            v50 = v49;
                        } else {
                            v50 = false;
                        }
                        bool v51;
                        v51 = v50 == false;
                        if (v51){
                            assert("Index must be in range." && v50);
                        } else {
                        }
                        unsigned char v53;
                        v53 = v1[v27];
                        v62 = v53;
                    } else {
                        bool v55;
                        v55 = 0 <= v27;
                        bool v57;
                        if (v55){
                            bool v56;
                            v56 = v27 < 7;
                            v57 = v56;
                        } else {
                            v57 = false;
                        }
                        bool v58;
                        v58 = v57 == false;
                        if (v58){
                            assert("Index must be in range." && v57);
                        } else {
                        }
                        unsigned char v60;
                        v60 = v12[v27];
                        v62 = v60;
                    }
                    unsigned char v63;
                    v63 = v62 / 4u;
                    unsigned char v64;
                    v64 = v47 / 4u;
                    bool v65;
                    v65 = v63 < v64;
                    Union7 v71;
                    if (v65){
                        v71 = Union7{Union7_2{}};
                    } else {
                        bool v67;
                        v67 = v63 > v64;
                        if (v67){
                            v71 = Union7{Union7_1{}};
                        } else {
                            v71 = Union7{Union7_0{}};
                        }
                    }
                    Union7 v81;
                    switch (v71.tag) {
                        case 0: { // Eq
                            unsigned char v72;
                            v72 = v47 % 4u;
                            unsigned char v73;
                            v73 = v62 % 4u;
                            bool v74;
                            v74 = v72 < v73;
                            if (v74){
                                v81 = Union7{Union7_2{}};
                            } else {
                                bool v76;
                                v76 = v72 > v73;
                                if (v76){
                                    v81 = Union7{Union7_1{}};
                                } else {
                                    v81 = Union7{Union7_0{}};
                                }
                            }
                            break;
                        }
                        default: {
                            v81 = v71;
                        }
                    }
                    switch (v81.tag) {
                        case 1: { // Gt
                            int v82;
                            v82 = v27 + 1;
                            v122 = v62; v123 = v26; v124 = v82;
                            break;
                        }
                        default: {
                            int v83;
                            v83 = v26 + 1;
                            v122 = v47; v123 = v83; v124 = v27;
                        }
                    }
                } else {
                    if (v30){
                        unsigned char v101;
                        if (v14){
                            bool v87;
                            v87 = 0 <= v26;
                            bool v89;
                            if (v87){
                                bool v88;
                                v88 = v26 < 7;
                                v89 = v88;
                            } else {
                                v89 = false;
                            }
                            bool v90;
                            v90 = v89 == false;
                            if (v90){
                                assert("Index must be in range." && v89);
                            } else {
                            }
                            unsigned char v92;
                            v92 = v1[v26];
                            v101 = v92;
                        } else {
                            bool v94;
                            v94 = 0 <= v26;
                            bool v96;
                            if (v94){
                                bool v95;
                                v95 = v26 < 7;
                                v96 = v95;
                            } else {
                                v96 = false;
                            }
                            bool v97;
                            v97 = v96 == false;
                            if (v97){
                                assert("Index must be in range." && v96);
                            } else {
                            }
                            unsigned char v99;
                            v99 = v12[v26];
                            v101 = v99;
                        }
                        int v102;
                        v102 = v26 + 1;
                        v122 = v101; v123 = v102; v124 = v27;
                    } else {
                        unsigned char v117;
                        if (v14){
                            bool v103;
                            v103 = 0 <= v27;
                            bool v105;
                            if (v103){
                                bool v104;
                                v104 = v27 < 7;
                                v105 = v104;
                            } else {
                                v105 = false;
                            }
                            bool v106;
                            v106 = v105 == false;
                            if (v106){
                                assert("Index must be in range." && v105);
                            } else {
                            }
                            unsigned char v108;
                            v108 = v1[v27];
                            v117 = v108;
                        } else {
                            bool v110;
                            v110 = 0 <= v27;
                            bool v112;
                            if (v110){
                                bool v111;
                                v111 = v27 < 7;
                                v112 = v111;
                            } else {
                                v112 = false;
                            }
                            bool v113;
                            v113 = v112 == false;
                            if (v113){
                                assert("Index must be in range." && v112);
                            } else {
                            }
                            unsigned char v115;
                            v115 = v12[v27];
                            v117 = v115;
                        }
                        int v118;
                        v118 = v27 + 1;
                        v122 = v117; v123 = v26; v124 = v118;
                    }
                }
                if (v14){
                    v12[v28] = v122;
                } else {
                    v1[v28] = v122;
                }
                int v125;
                v125 = v28 + 1;
                v26 = v123;
                v27 = v124;
                v28 = v125;
            }
            v17 = v23;
        }
        bool v126;
        v126 = v14 == false;
        int v127;
        v127 = v15 * 2;
        v14 = v126;
        v15 = v127;
    }
    bool v128;
    v128 = v14 == false;
    static_array<unsigned char,7> v129;
    if (v128){
        v129 = v12;
    } else {
        v129 = v1;
    }
    static_array<unsigned char,5> v130;
    int v132; int v133; unsigned char v134;
    Tuple10 tmp20 = Tuple10{0, 0, 12u};
    v132 = tmp20.v0; v133 = tmp20.v1; v134 = tmp20.v2;
    while (while_method_10(v132)){
        bool v136;
        v136 = 0 <= v132;
        bool v138;
        if (v136){
            bool v137;
            v137 = v132 < 7;
            v138 = v137;
        } else {
            v138 = false;
        }
        bool v139;
        v139 = v138 == false;
        if (v139){
            assert("Index must be in range." && v138);
        } else {
        }
        unsigned char v141;
        v141 = v129[v132];
        bool v143;
        v143 = v133 < 5;
        int v155; unsigned char v156;
        if (v143){
            unsigned char v144;
            v144 = v141 % 4u;
            bool v145;
            v145 = 0u == v144;
            if (v145){
                unsigned char v146;
                v146 = v141 / 4u;
                bool v147;
                v147 = v134 == v146;
                int v148;
                if (v147){
                    v148 = v133;
                } else {
                    v148 = 0;
                }
                v130[v148] = v141;
                int v149;
                v149 = v148 + 1;
                unsigned char v150;
                v150 = v146 - 1u;
                v155 = v149; v156 = v150;
            } else {
                v155 = v133; v156 = v134;
            }
        } else {
            break;
        }
        v133 = v155;
        v134 = v156;
        v132 += 1 ;
    }
    bool v157;
    v157 = v133 == 4;
    bool v196;
    if (v157){
        unsigned char v158;
        v158 = v134 + 1u;
        bool v159;
        v159 = v158 == 0u;
        if (v159){
            unsigned char v160;
            v160 = v129[0];
            unsigned char v162;
            v162 = v160 % 4u;
            bool v163;
            v163 = 0u == v162;
            bool v167;
            if (v163){
                unsigned char v164;
                v164 = v160 / 4u;
                bool v165;
                v165 = v164 == 12u;
                if (v165){
                    v130[4] = v160;
                    v167 = true;
                } else {
                    v167 = false;
                }
            } else {
                v167 = false;
            }
            if (v167){
                v196 = true;
            } else {
                unsigned char v168;
                v168 = v129[1];
                unsigned char v170;
                v170 = v168 % 4u;
                bool v171;
                v171 = 0u == v170;
                bool v175;
                if (v171){
                    unsigned char v172;
                    v172 = v168 / 4u;
                    bool v173;
                    v173 = v172 == 12u;
                    if (v173){
                        v130[4] = v168;
                        v175 = true;
                    } else {
                        v175 = false;
                    }
                } else {
                    v175 = false;
                }
                if (v175){
                    v196 = true;
                } else {
                    unsigned char v176;
                    v176 = v129[2];
                    unsigned char v178;
                    v178 = v176 % 4u;
                    bool v179;
                    v179 = 0u == v178;
                    bool v183;
                    if (v179){
                        unsigned char v180;
                        v180 = v176 / 4u;
                        bool v181;
                        v181 = v180 == 12u;
                        if (v181){
                            v130[4] = v176;
                            v183 = true;
                        } else {
                            v183 = false;
                        }
                    } else {
                        v183 = false;
                    }
                    if (v183){
                        v196 = true;
                    } else {
                        unsigned char v184;
                        v184 = v129[3];
                        unsigned char v186;
                        v186 = v184 % 4u;
                        bool v187;
                        v187 = 0u == v186;
                        if (v187){
                            unsigned char v188;
                            v188 = v184 / 4u;
                            bool v189;
                            v189 = v188 == 12u;
                            if (v189){
                                v130[4] = v184;
                                v196 = true;
                            } else {
                                v196 = false;
                            }
                        } else {
                            v196 = false;
                        }
                    }
                }
            }
        } else {
            v196 = false;
        }
    } else {
        v196 = false;
    }
    Union8 v202;
    if (v196){
        v202 = Union8{Union8_1{v130}};
    } else {
        bool v198;
        v198 = v133 == 5;
        if (v198){
            v202 = Union8{Union8_1{v130}};
        } else {
            v202 = Union8{Union8_0{}};
        }
    }
    static_array<unsigned char,5> v203;
    int v205; int v206; unsigned char v207;
    Tuple10 tmp21 = Tuple10{0, 0, 12u};
    v205 = tmp21.v0; v206 = tmp21.v1; v207 = tmp21.v2;
    while (while_method_10(v205)){
        bool v209;
        v209 = 0 <= v205;
        bool v211;
        if (v209){
            bool v210;
            v210 = v205 < 7;
            v211 = v210;
        } else {
            v211 = false;
        }
        bool v212;
        v212 = v211 == false;
        if (v212){
            assert("Index must be in range." && v211);
        } else {
        }
        unsigned char v214;
        v214 = v129[v205];
        bool v216;
        v216 = v206 < 5;
        int v228; unsigned char v229;
        if (v216){
            unsigned char v217;
            v217 = v214 % 4u;
            bool v218;
            v218 = 1u == v217;
            if (v218){
                unsigned char v219;
                v219 = v214 / 4u;
                bool v220;
                v220 = v207 == v219;
                int v221;
                if (v220){
                    v221 = v206;
                } else {
                    v221 = 0;
                }
                v203[v221] = v214;
                int v222;
                v222 = v221 + 1;
                unsigned char v223;
                v223 = v219 - 1u;
                v228 = v222; v229 = v223;
            } else {
                v228 = v206; v229 = v207;
            }
        } else {
            break;
        }
        v206 = v228;
        v207 = v229;
        v205 += 1 ;
    }
    bool v230;
    v230 = v206 == 4;
    bool v269;
    if (v230){
        unsigned char v231;
        v231 = v207 + 1u;
        bool v232;
        v232 = v231 == 0u;
        if (v232){
            unsigned char v233;
            v233 = v129[0];
            unsigned char v235;
            v235 = v233 % 4u;
            bool v236;
            v236 = 1u == v235;
            bool v240;
            if (v236){
                unsigned char v237;
                v237 = v233 / 4u;
                bool v238;
                v238 = v237 == 12u;
                if (v238){
                    v203[4] = v233;
                    v240 = true;
                } else {
                    v240 = false;
                }
            } else {
                v240 = false;
            }
            if (v240){
                v269 = true;
            } else {
                unsigned char v241;
                v241 = v129[1];
                unsigned char v243;
                v243 = v241 % 4u;
                bool v244;
                v244 = 1u == v243;
                bool v248;
                if (v244){
                    unsigned char v245;
                    v245 = v241 / 4u;
                    bool v246;
                    v246 = v245 == 12u;
                    if (v246){
                        v203[4] = v241;
                        v248 = true;
                    } else {
                        v248 = false;
                    }
                } else {
                    v248 = false;
                }
                if (v248){
                    v269 = true;
                } else {
                    unsigned char v249;
                    v249 = v129[2];
                    unsigned char v251;
                    v251 = v249 % 4u;
                    bool v252;
                    v252 = 1u == v251;
                    bool v256;
                    if (v252){
                        unsigned char v253;
                        v253 = v249 / 4u;
                        bool v254;
                        v254 = v253 == 12u;
                        if (v254){
                            v203[4] = v249;
                            v256 = true;
                        } else {
                            v256 = false;
                        }
                    } else {
                        v256 = false;
                    }
                    if (v256){
                        v269 = true;
                    } else {
                        unsigned char v257;
                        v257 = v129[3];
                        unsigned char v259;
                        v259 = v257 % 4u;
                        bool v260;
                        v260 = 1u == v259;
                        if (v260){
                            unsigned char v261;
                            v261 = v257 / 4u;
                            bool v262;
                            v262 = v261 == 12u;
                            if (v262){
                                v203[4] = v257;
                                v269 = true;
                            } else {
                                v269 = false;
                            }
                        } else {
                            v269 = false;
                        }
                    }
                }
            }
        } else {
            v269 = false;
        }
    } else {
        v269 = false;
    }
    Union8 v275;
    if (v269){
        v275 = Union8{Union8_1{v203}};
    } else {
        bool v271;
        v271 = v206 == 5;
        if (v271){
            v275 = Union8{Union8_1{v203}};
        } else {
            v275 = Union8{Union8_0{}};
        }
    }
    Union8 v312;
    switch (v202.tag) {
        case 0: { // None
            v312 = v275;
            break;
        }
        case 1: { // Some
            static_array<unsigned char,5> v276 = v202.case1.v0;
            switch (v275.tag) {
                case 0: { // None
                    v312 = v202;
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,5> v277 = v275.case1.v0;
                    Union7 v278;
                    v278 = Union7{Union7_0{}};
                    int v279; Union7 v280;
                    Tuple11 tmp22 = Tuple11{0, v278};
                    v279 = tmp22.v0; v280 = tmp22.v1;
                    while (while_method_5(v279)){
                        bool v282;
                        v282 = 0 <= v279;
                        bool v284;
                        if (v282){
                            bool v283;
                            v283 = v279 < 5;
                            v284 = v283;
                        } else {
                            v284 = false;
                        }
                        bool v285;
                        v285 = v284 == false;
                        if (v285){
                            assert("Index must be in range." && v284);
                        } else {
                        }
                        unsigned char v287;
                        v287 = v276[v279];
                        bool v290;
                        if (v282){
                            bool v289;
                            v289 = v279 < 5;
                            v290 = v289;
                        } else {
                            v290 = false;
                        }
                        bool v291;
                        v291 = v290 == false;
                        if (v291){
                            assert("Index must be in range." && v290);
                        } else {
                        }
                        unsigned char v293;
                        v293 = v277[v279];
                        Union7 v305;
                        switch (v280.tag) {
                            case 0: { // Eq
                                unsigned char v295;
                                v295 = v287 / 4u;
                                unsigned char v296;
                                v296 = v293 / 4u;
                                bool v297;
                                v297 = v295 < v296;
                                if (v297){
                                    v305 = Union7{Union7_2{}};
                                } else {
                                    bool v299;
                                    v299 = v295 > v296;
                                    if (v299){
                                        v305 = Union7{Union7_1{}};
                                    } else {
                                        v305 = Union7{Union7_0{}};
                                    }
                                }
                                break;
                            }
                            default: {
                                break;
                            }
                        }
                        v280 = v305;
                        v279 += 1 ;
                    }
                    bool v306;
                    switch (v280.tag) {
                        case 1: { // Gt
                            v306 = true;
                            break;
                        }
                        default: {
                            v306 = false;
                        }
                    }
                    static_array<unsigned char,5> v307;
                    if (v306){
                        v307 = v276;
                    } else {
                        v307 = v277;
                    }
                    v312 = Union8{Union8_1{v307}};
                    break;
                }
                default: {
                    assert("Invalid tag." && false); __trap();
                }
            }
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    static_array<unsigned char,5> v313;
    int v315; int v316; unsigned char v317;
    Tuple10 tmp23 = Tuple10{0, 0, 12u};
    v315 = tmp23.v0; v316 = tmp23.v1; v317 = tmp23.v2;
    while (while_method_10(v315)){
        bool v319;
        v319 = 0 <= v315;
        bool v321;
        if (v319){
            bool v320;
            v320 = v315 < 7;
            v321 = v320;
        } else {
            v321 = false;
        }
        bool v322;
        v322 = v321 == false;
        if (v322){
            assert("Index must be in range." && v321);
        } else {
        }
        unsigned char v324;
        v324 = v129[v315];
        bool v326;
        v326 = v316 < 5;
        int v338; unsigned char v339;
        if (v326){
            unsigned char v327;
            v327 = v324 % 4u;
            bool v328;
            v328 = 2u == v327;
            if (v328){
                unsigned char v329;
                v329 = v324 / 4u;
                bool v330;
                v330 = v317 == v329;
                int v331;
                if (v330){
                    v331 = v316;
                } else {
                    v331 = 0;
                }
                v313[v331] = v324;
                int v332;
                v332 = v331 + 1;
                unsigned char v333;
                v333 = v329 - 1u;
                v338 = v332; v339 = v333;
            } else {
                v338 = v316; v339 = v317;
            }
        } else {
            break;
        }
        v316 = v338;
        v317 = v339;
        v315 += 1 ;
    }
    bool v340;
    v340 = v316 == 4;
    bool v379;
    if (v340){
        unsigned char v341;
        v341 = v317 + 1u;
        bool v342;
        v342 = v341 == 0u;
        if (v342){
            unsigned char v343;
            v343 = v129[0];
            unsigned char v345;
            v345 = v343 % 4u;
            bool v346;
            v346 = 2u == v345;
            bool v350;
            if (v346){
                unsigned char v347;
                v347 = v343 / 4u;
                bool v348;
                v348 = v347 == 12u;
                if (v348){
                    v313[4] = v343;
                    v350 = true;
                } else {
                    v350 = false;
                }
            } else {
                v350 = false;
            }
            if (v350){
                v379 = true;
            } else {
                unsigned char v351;
                v351 = v129[1];
                unsigned char v353;
                v353 = v351 % 4u;
                bool v354;
                v354 = 2u == v353;
                bool v358;
                if (v354){
                    unsigned char v355;
                    v355 = v351 / 4u;
                    bool v356;
                    v356 = v355 == 12u;
                    if (v356){
                        v313[4] = v351;
                        v358 = true;
                    } else {
                        v358 = false;
                    }
                } else {
                    v358 = false;
                }
                if (v358){
                    v379 = true;
                } else {
                    unsigned char v359;
                    v359 = v129[2];
                    unsigned char v361;
                    v361 = v359 % 4u;
                    bool v362;
                    v362 = 2u == v361;
                    bool v366;
                    if (v362){
                        unsigned char v363;
                        v363 = v359 / 4u;
                        bool v364;
                        v364 = v363 == 12u;
                        if (v364){
                            v313[4] = v359;
                            v366 = true;
                        } else {
                            v366 = false;
                        }
                    } else {
                        v366 = false;
                    }
                    if (v366){
                        v379 = true;
                    } else {
                        unsigned char v367;
                        v367 = v129[3];
                        unsigned char v369;
                        v369 = v367 % 4u;
                        bool v370;
                        v370 = 2u == v369;
                        if (v370){
                            unsigned char v371;
                            v371 = v367 / 4u;
                            bool v372;
                            v372 = v371 == 12u;
                            if (v372){
                                v313[4] = v367;
                                v379 = true;
                            } else {
                                v379 = false;
                            }
                        } else {
                            v379 = false;
                        }
                    }
                }
            }
        } else {
            v379 = false;
        }
    } else {
        v379 = false;
    }
    Union8 v385;
    if (v379){
        v385 = Union8{Union8_1{v313}};
    } else {
        bool v381;
        v381 = v316 == 5;
        if (v381){
            v385 = Union8{Union8_1{v313}};
        } else {
            v385 = Union8{Union8_0{}};
        }
    }
    Union8 v422;
    switch (v312.tag) {
        case 0: { // None
            v422 = v385;
            break;
        }
        case 1: { // Some
            static_array<unsigned char,5> v386 = v312.case1.v0;
            switch (v385.tag) {
                case 0: { // None
                    v422 = v312;
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,5> v387 = v385.case1.v0;
                    Union7 v388;
                    v388 = Union7{Union7_0{}};
                    int v389; Union7 v390;
                    Tuple11 tmp24 = Tuple11{0, v388};
                    v389 = tmp24.v0; v390 = tmp24.v1;
                    while (while_method_5(v389)){
                        bool v392;
                        v392 = 0 <= v389;
                        bool v394;
                        if (v392){
                            bool v393;
                            v393 = v389 < 5;
                            v394 = v393;
                        } else {
                            v394 = false;
                        }
                        bool v395;
                        v395 = v394 == false;
                        if (v395){
                            assert("Index must be in range." && v394);
                        } else {
                        }
                        unsigned char v397;
                        v397 = v386[v389];
                        bool v400;
                        if (v392){
                            bool v399;
                            v399 = v389 < 5;
                            v400 = v399;
                        } else {
                            v400 = false;
                        }
                        bool v401;
                        v401 = v400 == false;
                        if (v401){
                            assert("Index must be in range." && v400);
                        } else {
                        }
                        unsigned char v403;
                        v403 = v387[v389];
                        Union7 v415;
                        switch (v390.tag) {
                            case 0: { // Eq
                                unsigned char v405;
                                v405 = v397 / 4u;
                                unsigned char v406;
                                v406 = v403 / 4u;
                                bool v407;
                                v407 = v405 < v406;
                                if (v407){
                                    v415 = Union7{Union7_2{}};
                                } else {
                                    bool v409;
                                    v409 = v405 > v406;
                                    if (v409){
                                        v415 = Union7{Union7_1{}};
                                    } else {
                                        v415 = Union7{Union7_0{}};
                                    }
                                }
                                break;
                            }
                            default: {
                                break;
                            }
                        }
                        v390 = v415;
                        v389 += 1 ;
                    }
                    bool v416;
                    switch (v390.tag) {
                        case 1: { // Gt
                            v416 = true;
                            break;
                        }
                        default: {
                            v416 = false;
                        }
                    }
                    static_array<unsigned char,5> v417;
                    if (v416){
                        v417 = v386;
                    } else {
                        v417 = v387;
                    }
                    v422 = Union8{Union8_1{v417}};
                    break;
                }
                default: {
                    assert("Invalid tag." && false); __trap();
                }
            }
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    static_array<unsigned char,5> v423;
    int v425; int v426; unsigned char v427;
    Tuple10 tmp25 = Tuple10{0, 0, 12u};
    v425 = tmp25.v0; v426 = tmp25.v1; v427 = tmp25.v2;
    while (while_method_10(v425)){
        bool v429;
        v429 = 0 <= v425;
        bool v431;
        if (v429){
            bool v430;
            v430 = v425 < 7;
            v431 = v430;
        } else {
            v431 = false;
        }
        bool v432;
        v432 = v431 == false;
        if (v432){
            assert("Index must be in range." && v431);
        } else {
        }
        unsigned char v434;
        v434 = v129[v425];
        bool v436;
        v436 = v426 < 5;
        int v448; unsigned char v449;
        if (v436){
            unsigned char v437;
            v437 = v434 % 4u;
            bool v438;
            v438 = 3u == v437;
            if (v438){
                unsigned char v439;
                v439 = v434 / 4u;
                bool v440;
                v440 = v427 == v439;
                int v441;
                if (v440){
                    v441 = v426;
                } else {
                    v441 = 0;
                }
                v423[v441] = v434;
                int v442;
                v442 = v441 + 1;
                unsigned char v443;
                v443 = v439 - 1u;
                v448 = v442; v449 = v443;
            } else {
                v448 = v426; v449 = v427;
            }
        } else {
            break;
        }
        v426 = v448;
        v427 = v449;
        v425 += 1 ;
    }
    bool v450;
    v450 = v426 == 4;
    bool v489;
    if (v450){
        unsigned char v451;
        v451 = v427 + 1u;
        bool v452;
        v452 = v451 == 0u;
        if (v452){
            unsigned char v453;
            v453 = v129[0];
            unsigned char v455;
            v455 = v453 % 4u;
            bool v456;
            v456 = 3u == v455;
            bool v460;
            if (v456){
                unsigned char v457;
                v457 = v453 / 4u;
                bool v458;
                v458 = v457 == 12u;
                if (v458){
                    v423[4] = v453;
                    v460 = true;
                } else {
                    v460 = false;
                }
            } else {
                v460 = false;
            }
            if (v460){
                v489 = true;
            } else {
                unsigned char v461;
                v461 = v129[1];
                unsigned char v463;
                v463 = v461 % 4u;
                bool v464;
                v464 = 3u == v463;
                bool v468;
                if (v464){
                    unsigned char v465;
                    v465 = v461 / 4u;
                    bool v466;
                    v466 = v465 == 12u;
                    if (v466){
                        v423[4] = v461;
                        v468 = true;
                    } else {
                        v468 = false;
                    }
                } else {
                    v468 = false;
                }
                if (v468){
                    v489 = true;
                } else {
                    unsigned char v469;
                    v469 = v129[2];
                    unsigned char v471;
                    v471 = v469 % 4u;
                    bool v472;
                    v472 = 3u == v471;
                    bool v476;
                    if (v472){
                        unsigned char v473;
                        v473 = v469 / 4u;
                        bool v474;
                        v474 = v473 == 12u;
                        if (v474){
                            v423[4] = v469;
                            v476 = true;
                        } else {
                            v476 = false;
                        }
                    } else {
                        v476 = false;
                    }
                    if (v476){
                        v489 = true;
                    } else {
                        unsigned char v477;
                        v477 = v129[3];
                        unsigned char v479;
                        v479 = v477 % 4u;
                        bool v480;
                        v480 = 3u == v479;
                        if (v480){
                            unsigned char v481;
                            v481 = v477 / 4u;
                            bool v482;
                            v482 = v481 == 12u;
                            if (v482){
                                v423[4] = v477;
                                v489 = true;
                            } else {
                                v489 = false;
                            }
                        } else {
                            v489 = false;
                        }
                    }
                }
            }
        } else {
            v489 = false;
        }
    } else {
        v489 = false;
    }
    Union8 v495;
    if (v489){
        v495 = Union8{Union8_1{v423}};
    } else {
        bool v491;
        v491 = v426 == 5;
        if (v491){
            v495 = Union8{Union8_1{v423}};
        } else {
            v495 = Union8{Union8_0{}};
        }
    }
    Union8 v532;
    switch (v422.tag) {
        case 0: { // None
            v532 = v495;
            break;
        }
        case 1: { // Some
            static_array<unsigned char,5> v496 = v422.case1.v0;
            switch (v495.tag) {
                case 0: { // None
                    v532 = v422;
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,5> v497 = v495.case1.v0;
                    Union7 v498;
                    v498 = Union7{Union7_0{}};
                    int v499; Union7 v500;
                    Tuple11 tmp26 = Tuple11{0, v498};
                    v499 = tmp26.v0; v500 = tmp26.v1;
                    while (while_method_5(v499)){
                        bool v502;
                        v502 = 0 <= v499;
                        bool v504;
                        if (v502){
                            bool v503;
                            v503 = v499 < 5;
                            v504 = v503;
                        } else {
                            v504 = false;
                        }
                        bool v505;
                        v505 = v504 == false;
                        if (v505){
                            assert("Index must be in range." && v504);
                        } else {
                        }
                        unsigned char v507;
                        v507 = v496[v499];
                        bool v510;
                        if (v502){
                            bool v509;
                            v509 = v499 < 5;
                            v510 = v509;
                        } else {
                            v510 = false;
                        }
                        bool v511;
                        v511 = v510 == false;
                        if (v511){
                            assert("Index must be in range." && v510);
                        } else {
                        }
                        unsigned char v513;
                        v513 = v497[v499];
                        Union7 v525;
                        switch (v500.tag) {
                            case 0: { // Eq
                                unsigned char v515;
                                v515 = v507 / 4u;
                                unsigned char v516;
                                v516 = v513 / 4u;
                                bool v517;
                                v517 = v515 < v516;
                                if (v517){
                                    v525 = Union7{Union7_2{}};
                                } else {
                                    bool v519;
                                    v519 = v515 > v516;
                                    if (v519){
                                        v525 = Union7{Union7_1{}};
                                    } else {
                                        v525 = Union7{Union7_0{}};
                                    }
                                }
                                break;
                            }
                            default: {
                                break;
                            }
                        }
                        v500 = v525;
                        v499 += 1 ;
                    }
                    bool v526;
                    switch (v500.tag) {
                        case 1: { // Gt
                            v526 = true;
                            break;
                        }
                        default: {
                            v526 = false;
                        }
                    }
                    static_array<unsigned char,5> v527;
                    if (v526){
                        v527 = v496;
                    } else {
                        v527 = v497;
                    }
                    v532 = Union8{Union8_1{v527}};
                    break;
                }
                default: {
                    assert("Invalid tag." && false); __trap();
                }
            }
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    static_array<unsigned char,5> v1331; char v1332;
    switch (v532.tag) {
        case 0: { // None
            static_array<unsigned char,4> v534;
            static_array<unsigned char,3> v536;
            int v538; int v539; int v540; unsigned char v541;
            Tuple12 tmp27 = Tuple12{0, 0, 0, 12u};
            v538 = tmp27.v0; v539 = tmp27.v1; v540 = tmp27.v2; v541 = tmp27.v3;
            while (while_method_10(v538)){
                bool v543;
                v543 = 0 <= v538;
                bool v545;
                if (v543){
                    bool v544;
                    v544 = v538 < 7;
                    v545 = v544;
                } else {
                    v545 = false;
                }
                bool v546;
                v546 = v545 == false;
                if (v546){
                    assert("Index must be in range." && v545);
                } else {
                }
                unsigned char v548;
                v548 = v129[v538];
                bool v550;
                v550 = v540 < 4;
                int v558; int v559; unsigned char v560;
                if (v550){
                    unsigned char v551;
                    v551 = v548 / 4u;
                    bool v552;
                    v552 = v541 == v551;
                    int v553;
                    if (v552){
                        v553 = v540;
                    } else {
                        v553 = 0;
                    }
                    v534[v553] = v548;
                    int v554;
                    v554 = v553 + 1;
                    v558 = v538; v559 = v554; v560 = v551;
                } else {
                    break;
                }
                v539 = v558;
                v540 = v559;
                v541 = v560;
                v538 += 1 ;
            }
            bool v561;
            v561 = v540 == 4;
            Union9 v577;
            if (v561){
                int v562;
                v562 = 0;
                while (while_method_4(v562)){
                    int v564;
                    v564 = v539 + -3;
                    bool v565;
                    v565 = v562 < v564;
                    int v566;
                    if (v565){
                        v566 = 0;
                    } else {
                        v566 = 4;
                    }
                    int v567;
                    v567 = v566 + v562;
                    bool v568;
                    v568 = 0 <= v567;
                    bool v570;
                    if (v568){
                        bool v569;
                        v569 = v567 < 7;
                        v570 = v569;
                    } else {
                        v570 = false;
                    }
                    bool v571;
                    v571 = v570 == false;
                    if (v571){
                        assert("Index must be in range." && v570);
                    } else {
                    }
                    unsigned char v573;
                    v573 = v129[v567];
                    v536[v562] = v573;
                    v562 += 1 ;
                }
                v577 = Union9{Union9_1{v534, v536}};
            } else {
                v577 = Union9{Union9_0{}};
            }
            Union8 v615;
            switch (v577.tag) {
                case 0: { // None
                    v615 = Union8{Union8_0{}};
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,4> v578 = v577.case1.v0; static_array<unsigned char,3> v579 = v577.case1.v1;
                    static_array<unsigned char,1> v580;
                    int v582;
                    v582 = 0;
                    while (while_method_6(v582)){
                        bool v584;
                        v584 = 0 <= v582;
                        bool v586;
                        if (v584){
                            bool v585;
                            v585 = v582 < 3;
                            v586 = v585;
                        } else {
                            v586 = false;
                        }
                        bool v587;
                        v587 = v586 == false;
                        if (v587){
                            assert("Index must be in range." && v586);
                        } else {
                        }
                        unsigned char v589;
                        v589 = v579[v582];
                        v580[v582] = v589;
                        v582 += 1 ;
                    }
                    static_array<unsigned char,5> v591;
                    int v593;
                    v593 = 0;
                    while (while_method_0(v593)){
                        bool v595;
                        v595 = 0 <= v593;
                        bool v597;
                        if (v595){
                            bool v596;
                            v596 = v593 < 4;
                            v597 = v596;
                        } else {
                            v597 = false;
                        }
                        bool v598;
                        v598 = v597 == false;
                        if (v598){
                            assert("Index must be in range." && v597);
                        } else {
                        }
                        unsigned char v600;
                        v600 = v578[v593];
                        v591[v593] = v600;
                        v593 += 1 ;
                    }
                    int v602;
                    v602 = 0;
                    while (while_method_6(v602)){
                        bool v604;
                        v604 = 0 <= v602;
                        bool v606;
                        if (v604){
                            bool v605;
                            v605 = v602 < 1;
                            v606 = v605;
                        } else {
                            v606 = false;
                        }
                        bool v607;
                        v607 = v606 == false;
                        if (v607){
                            assert("Index must be in range." && v606);
                        } else {
                        }
                        unsigned char v609;
                        v609 = v580[v602];
                        int v611;
                        v611 = 4 + v602;
                        v591[v611] = v609;
                        v602 += 1 ;
                    }
                    v615 = Union8{Union8_1{v591}};
                    break;
                }
                default: {
                    assert("Invalid tag." && false); __trap();
                }
            }
            switch (v615.tag) {
                case 0: { // None
                    static_array<unsigned char,3> v617;
                    static_array<unsigned char,4> v619;
                    int v621; int v622; int v623; unsigned char v624;
                    Tuple12 tmp28 = Tuple12{0, 0, 0, 12u};
                    v621 = tmp28.v0; v622 = tmp28.v1; v623 = tmp28.v2; v624 = tmp28.v3;
                    while (while_method_10(v621)){
                        bool v626;
                        v626 = 0 <= v621;
                        bool v628;
                        if (v626){
                            bool v627;
                            v627 = v621 < 7;
                            v628 = v627;
                        } else {
                            v628 = false;
                        }
                        bool v629;
                        v629 = v628 == false;
                        if (v629){
                            assert("Index must be in range." && v628);
                        } else {
                        }
                        unsigned char v631;
                        v631 = v129[v621];
                        bool v633;
                        v633 = v623 < 3;
                        int v641; int v642; unsigned char v643;
                        if (v633){
                            unsigned char v634;
                            v634 = v631 / 4u;
                            bool v635;
                            v635 = v624 == v634;
                            int v636;
                            if (v635){
                                v636 = v623;
                            } else {
                                v636 = 0;
                            }
                            v617[v636] = v631;
                            int v637;
                            v637 = v636 + 1;
                            v641 = v621; v642 = v637; v643 = v634;
                        } else {
                            break;
                        }
                        v622 = v641;
                        v623 = v642;
                        v624 = v643;
                        v621 += 1 ;
                    }
                    bool v644;
                    v644 = v623 == 3;
                    Union10 v660;
                    if (v644){
                        int v645;
                        v645 = 0;
                        while (while_method_0(v645)){
                            int v647;
                            v647 = v622 + -2;
                            bool v648;
                            v648 = v645 < v647;
                            int v649;
                            if (v648){
                                v649 = 0;
                            } else {
                                v649 = 3;
                            }
                            int v650;
                            v650 = v649 + v645;
                            bool v651;
                            v651 = 0 <= v650;
                            bool v653;
                            if (v651){
                                bool v652;
                                v652 = v650 < 7;
                                v653 = v652;
                            } else {
                                v653 = false;
                            }
                            bool v654;
                            v654 = v653 == false;
                            if (v654){
                                assert("Index must be in range." && v653);
                            } else {
                            }
                            unsigned char v656;
                            v656 = v129[v650];
                            v619[v645] = v656;
                            v645 += 1 ;
                        }
                        v660 = Union10{Union10_1{v617, v619}};
                    } else {
                        v660 = Union10{Union10_0{}};
                    }
                    Union8 v736;
                    switch (v660.tag) {
                        case 0: { // None
                            v736 = Union8{Union8_0{}};
                            break;
                        }
                        case 1: { // Some
                            static_array<unsigned char,3> v661 = v660.case1.v0; static_array<unsigned char,4> v662 = v660.case1.v1;
                            static_array<unsigned char,2> v663;
                            static_array<unsigned char,2> v665;
                            int v667; int v668; int v669; unsigned char v670;
                            Tuple12 tmp29 = Tuple12{0, 0, 0, 12u};
                            v667 = tmp29.v0; v668 = tmp29.v1; v669 = tmp29.v2; v670 = tmp29.v3;
                            while (while_method_0(v667)){
                                bool v672;
                                v672 = 0 <= v667;
                                bool v674;
                                if (v672){
                                    bool v673;
                                    v673 = v667 < 4;
                                    v674 = v673;
                                } else {
                                    v674 = false;
                                }
                                bool v675;
                                v675 = v674 == false;
                                if (v675){
                                    assert("Index must be in range." && v674);
                                } else {
                                }
                                unsigned char v677;
                                v677 = v662[v667];
                                bool v679;
                                v679 = v669 < 2;
                                int v687; int v688; unsigned char v689;
                                if (v679){
                                    unsigned char v680;
                                    v680 = v677 / 4u;
                                    bool v681;
                                    v681 = v670 == v680;
                                    int v682;
                                    if (v681){
                                        v682 = v669;
                                    } else {
                                        v682 = 0;
                                    }
                                    v663[v682] = v677;
                                    int v683;
                                    v683 = v682 + 1;
                                    v687 = v667; v688 = v683; v689 = v680;
                                } else {
                                    break;
                                }
                                v668 = v687;
                                v669 = v688;
                                v670 = v689;
                                v667 += 1 ;
                            }
                            bool v690;
                            v690 = v669 == 2;
                            Union11 v706;
                            if (v690){
                                int v691;
                                v691 = 0;
                                while (while_method_2(v691)){
                                    int v693;
                                    v693 = v668 + -1;
                                    bool v694;
                                    v694 = v691 < v693;
                                    int v695;
                                    if (v694){
                                        v695 = 0;
                                    } else {
                                        v695 = 2;
                                    }
                                    int v696;
                                    v696 = v695 + v691;
                                    bool v697;
                                    v697 = 0 <= v696;
                                    bool v699;
                                    if (v697){
                                        bool v698;
                                        v698 = v696 < 4;
                                        v699 = v698;
                                    } else {
                                        v699 = false;
                                    }
                                    bool v700;
                                    v700 = v699 == false;
                                    if (v700){
                                        assert("Index must be in range." && v699);
                                    } else {
                                    }
                                    unsigned char v702;
                                    v702 = v662[v696];
                                    v665[v691] = v702;
                                    v691 += 1 ;
                                }
                                v706 = Union11{Union11_1{v663, v665}};
                            } else {
                                v706 = Union11{Union11_0{}};
                            }
                            switch (v706.tag) {
                                case 0: { // None
                                    v736 = Union8{Union8_0{}};
                                    break;
                                }
                                case 1: { // Some
                                    static_array<unsigned char,2> v707 = v706.case1.v0; static_array<unsigned char,2> v708 = v706.case1.v1;
                                    static_array<unsigned char,5> v709;
                                    int v711;
                                    v711 = 0;
                                    while (while_method_4(v711)){
                                        bool v713;
                                        v713 = 0 <= v711;
                                        bool v715;
                                        if (v713){
                                            bool v714;
                                            v714 = v711 < 3;
                                            v715 = v714;
                                        } else {
                                            v715 = false;
                                        }
                                        bool v716;
                                        v716 = v715 == false;
                                        if (v716){
                                            assert("Index must be in range." && v715);
                                        } else {
                                        }
                                        unsigned char v718;
                                        v718 = v661[v711];
                                        v709[v711] = v718;
                                        v711 += 1 ;
                                    }
                                    int v720;
                                    v720 = 0;
                                    while (while_method_2(v720)){
                                        bool v722;
                                        v722 = 0 <= v720;
                                        bool v724;
                                        if (v722){
                                            bool v723;
                                            v723 = v720 < 2;
                                            v724 = v723;
                                        } else {
                                            v724 = false;
                                        }
                                        bool v725;
                                        v725 = v724 == false;
                                        if (v725){
                                            assert("Index must be in range." && v724);
                                        } else {
                                        }
                                        unsigned char v727;
                                        v727 = v707[v720];
                                        int v729;
                                        v729 = 3 + v720;
                                        v709[v729] = v727;
                                        v720 += 1 ;
                                    }
                                    v736 = Union8{Union8_1{v709}};
                                    break;
                                }
                                default: {
                                    assert("Invalid tag." && false); __trap();
                                }
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false); __trap();
                        }
                    }
                    switch (v736.tag) {
                        case 0: { // None
                            static_array<unsigned char,5> v738;
                            int v740; int v741;
                            Tuple4 tmp30 = Tuple4{0, 0};
                            v740 = tmp30.v0; v741 = tmp30.v1;
                            while (while_method_10(v740)){
                                bool v743;
                                v743 = 0 <= v740;
                                bool v745;
                                if (v743){
                                    bool v744;
                                    v744 = v740 < 7;
                                    v745 = v744;
                                } else {
                                    v745 = false;
                                }
                                bool v746;
                                v746 = v745 == false;
                                if (v746){
                                    assert("Index must be in range." && v745);
                                } else {
                                }
                                unsigned char v748;
                                v748 = v129[v740];
                                unsigned char v750;
                                v750 = v748 % 4u;
                                bool v751;
                                v751 = v750 == 0u;
                                bool v753;
                                if (v751){
                                    bool v752;
                                    v752 = v741 < 5;
                                    v753 = v752;
                                } else {
                                    v753 = false;
                                }
                                int v755;
                                if (v753){
                                    v738[v741] = v748;
                                    int v754;
                                    v754 = v741 + 1;
                                    v755 = v754;
                                } else {
                                    v755 = v741;
                                }
                                v741 = v755;
                                v740 += 1 ;
                            }
                            bool v756;
                            v756 = v741 == 5;
                            Union8 v759;
                            if (v756){
                                v759 = Union8{Union8_1{v738}};
                            } else {
                                v759 = Union8{Union8_0{}};
                            }
                            static_array<unsigned char,5> v760;
                            int v762; int v763;
                            Tuple4 tmp31 = Tuple4{0, 0};
                            v762 = tmp31.v0; v763 = tmp31.v1;
                            while (while_method_10(v762)){
                                bool v765;
                                v765 = 0 <= v762;
                                bool v767;
                                if (v765){
                                    bool v766;
                                    v766 = v762 < 7;
                                    v767 = v766;
                                } else {
                                    v767 = false;
                                }
                                bool v768;
                                v768 = v767 == false;
                                if (v768){
                                    assert("Index must be in range." && v767);
                                } else {
                                }
                                unsigned char v770;
                                v770 = v129[v762];
                                unsigned char v772;
                                v772 = v770 % 4u;
                                bool v773;
                                v773 = v772 == 1u;
                                bool v775;
                                if (v773){
                                    bool v774;
                                    v774 = v763 < 5;
                                    v775 = v774;
                                } else {
                                    v775 = false;
                                }
                                int v777;
                                if (v775){
                                    v760[v763] = v770;
                                    int v776;
                                    v776 = v763 + 1;
                                    v777 = v776;
                                } else {
                                    v777 = v763;
                                }
                                v763 = v777;
                                v762 += 1 ;
                            }
                            bool v778;
                            v778 = v763 == 5;
                            Union8 v781;
                            if (v778){
                                v781 = Union8{Union8_1{v760}};
                            } else {
                                v781 = Union8{Union8_0{}};
                            }
                            Union8 v818;
                            switch (v759.tag) {
                                case 0: { // None
                                    v818 = v781;
                                    break;
                                }
                                case 1: { // Some
                                    static_array<unsigned char,5> v782 = v759.case1.v0;
                                    switch (v781.tag) {
                                        case 0: { // None
                                            v818 = v759;
                                            break;
                                        }
                                        case 1: { // Some
                                            static_array<unsigned char,5> v783 = v781.case1.v0;
                                            Union7 v784;
                                            v784 = Union7{Union7_0{}};
                                            int v785; Union7 v786;
                                            Tuple11 tmp32 = Tuple11{0, v784};
                                            v785 = tmp32.v0; v786 = tmp32.v1;
                                            while (while_method_5(v785)){
                                                bool v788;
                                                v788 = 0 <= v785;
                                                bool v790;
                                                if (v788){
                                                    bool v789;
                                                    v789 = v785 < 5;
                                                    v790 = v789;
                                                } else {
                                                    v790 = false;
                                                }
                                                bool v791;
                                                v791 = v790 == false;
                                                if (v791){
                                                    assert("Index must be in range." && v790);
                                                } else {
                                                }
                                                unsigned char v793;
                                                v793 = v782[v785];
                                                bool v796;
                                                if (v788){
                                                    bool v795;
                                                    v795 = v785 < 5;
                                                    v796 = v795;
                                                } else {
                                                    v796 = false;
                                                }
                                                bool v797;
                                                v797 = v796 == false;
                                                if (v797){
                                                    assert("Index must be in range." && v796);
                                                } else {
                                                }
                                                unsigned char v799;
                                                v799 = v783[v785];
                                                Union7 v811;
                                                switch (v786.tag) {
                                                    case 0: { // Eq
                                                        unsigned char v801;
                                                        v801 = v793 / 4u;
                                                        unsigned char v802;
                                                        v802 = v799 / 4u;
                                                        bool v803;
                                                        v803 = v801 < v802;
                                                        if (v803){
                                                            v811 = Union7{Union7_2{}};
                                                        } else {
                                                            bool v805;
                                                            v805 = v801 > v802;
                                                            if (v805){
                                                                v811 = Union7{Union7_1{}};
                                                            } else {
                                                                v811 = Union7{Union7_0{}};
                                                            }
                                                        }
                                                        break;
                                                    }
                                                    default: {
                                                        break;
                                                    }
                                                }
                                                v786 = v811;
                                                v785 += 1 ;
                                            }
                                            bool v812;
                                            switch (v786.tag) {
                                                case 1: { // Gt
                                                    v812 = true;
                                                    break;
                                                }
                                                default: {
                                                    v812 = false;
                                                }
                                            }
                                            static_array<unsigned char,5> v813;
                                            if (v812){
                                                v813 = v782;
                                            } else {
                                                v813 = v783;
                                            }
                                            v818 = Union8{Union8_1{v813}};
                                            break;
                                        }
                                        default: {
                                            assert("Invalid tag." && false); __trap();
                                        }
                                    }
                                    break;
                                }
                                default: {
                                    assert("Invalid tag." && false); __trap();
                                }
                            }
                            static_array<unsigned char,5> v819;
                            int v821; int v822;
                            Tuple4 tmp33 = Tuple4{0, 0};
                            v821 = tmp33.v0; v822 = tmp33.v1;
                            while (while_method_10(v821)){
                                bool v824;
                                v824 = 0 <= v821;
                                bool v826;
                                if (v824){
                                    bool v825;
                                    v825 = v821 < 7;
                                    v826 = v825;
                                } else {
                                    v826 = false;
                                }
                                bool v827;
                                v827 = v826 == false;
                                if (v827){
                                    assert("Index must be in range." && v826);
                                } else {
                                }
                                unsigned char v829;
                                v829 = v129[v821];
                                unsigned char v831;
                                v831 = v829 % 4u;
                                bool v832;
                                v832 = v831 == 2u;
                                bool v834;
                                if (v832){
                                    bool v833;
                                    v833 = v822 < 5;
                                    v834 = v833;
                                } else {
                                    v834 = false;
                                }
                                int v836;
                                if (v834){
                                    v819[v822] = v829;
                                    int v835;
                                    v835 = v822 + 1;
                                    v836 = v835;
                                } else {
                                    v836 = v822;
                                }
                                v822 = v836;
                                v821 += 1 ;
                            }
                            bool v837;
                            v837 = v822 == 5;
                            Union8 v840;
                            if (v837){
                                v840 = Union8{Union8_1{v819}};
                            } else {
                                v840 = Union8{Union8_0{}};
                            }
                            Union8 v877;
                            switch (v818.tag) {
                                case 0: { // None
                                    v877 = v840;
                                    break;
                                }
                                case 1: { // Some
                                    static_array<unsigned char,5> v841 = v818.case1.v0;
                                    switch (v840.tag) {
                                        case 0: { // None
                                            v877 = v818;
                                            break;
                                        }
                                        case 1: { // Some
                                            static_array<unsigned char,5> v842 = v840.case1.v0;
                                            Union7 v843;
                                            v843 = Union7{Union7_0{}};
                                            int v844; Union7 v845;
                                            Tuple11 tmp34 = Tuple11{0, v843};
                                            v844 = tmp34.v0; v845 = tmp34.v1;
                                            while (while_method_5(v844)){
                                                bool v847;
                                                v847 = 0 <= v844;
                                                bool v849;
                                                if (v847){
                                                    bool v848;
                                                    v848 = v844 < 5;
                                                    v849 = v848;
                                                } else {
                                                    v849 = false;
                                                }
                                                bool v850;
                                                v850 = v849 == false;
                                                if (v850){
                                                    assert("Index must be in range." && v849);
                                                } else {
                                                }
                                                unsigned char v852;
                                                v852 = v841[v844];
                                                bool v855;
                                                if (v847){
                                                    bool v854;
                                                    v854 = v844 < 5;
                                                    v855 = v854;
                                                } else {
                                                    v855 = false;
                                                }
                                                bool v856;
                                                v856 = v855 == false;
                                                if (v856){
                                                    assert("Index must be in range." && v855);
                                                } else {
                                                }
                                                unsigned char v858;
                                                v858 = v842[v844];
                                                Union7 v870;
                                                switch (v845.tag) {
                                                    case 0: { // Eq
                                                        unsigned char v860;
                                                        v860 = v852 / 4u;
                                                        unsigned char v861;
                                                        v861 = v858 / 4u;
                                                        bool v862;
                                                        v862 = v860 < v861;
                                                        if (v862){
                                                            v870 = Union7{Union7_2{}};
                                                        } else {
                                                            bool v864;
                                                            v864 = v860 > v861;
                                                            if (v864){
                                                                v870 = Union7{Union7_1{}};
                                                            } else {
                                                                v870 = Union7{Union7_0{}};
                                                            }
                                                        }
                                                        break;
                                                    }
                                                    default: {
                                                        break;
                                                    }
                                                }
                                                v845 = v870;
                                                v844 += 1 ;
                                            }
                                            bool v871;
                                            switch (v845.tag) {
                                                case 1: { // Gt
                                                    v871 = true;
                                                    break;
                                                }
                                                default: {
                                                    v871 = false;
                                                }
                                            }
                                            static_array<unsigned char,5> v872;
                                            if (v871){
                                                v872 = v841;
                                            } else {
                                                v872 = v842;
                                            }
                                            v877 = Union8{Union8_1{v872}};
                                            break;
                                        }
                                        default: {
                                            assert("Invalid tag." && false); __trap();
                                        }
                                    }
                                    break;
                                }
                                default: {
                                    assert("Invalid tag." && false); __trap();
                                }
                            }
                            static_array<unsigned char,5> v878;
                            int v880; int v881;
                            Tuple4 tmp35 = Tuple4{0, 0};
                            v880 = tmp35.v0; v881 = tmp35.v1;
                            while (while_method_10(v880)){
                                bool v883;
                                v883 = 0 <= v880;
                                bool v885;
                                if (v883){
                                    bool v884;
                                    v884 = v880 < 7;
                                    v885 = v884;
                                } else {
                                    v885 = false;
                                }
                                bool v886;
                                v886 = v885 == false;
                                if (v886){
                                    assert("Index must be in range." && v885);
                                } else {
                                }
                                unsigned char v888;
                                v888 = v129[v880];
                                unsigned char v890;
                                v890 = v888 % 4u;
                                bool v891;
                                v891 = v890 == 3u;
                                bool v893;
                                if (v891){
                                    bool v892;
                                    v892 = v881 < 5;
                                    v893 = v892;
                                } else {
                                    v893 = false;
                                }
                                int v895;
                                if (v893){
                                    v878[v881] = v888;
                                    int v894;
                                    v894 = v881 + 1;
                                    v895 = v894;
                                } else {
                                    v895 = v881;
                                }
                                v881 = v895;
                                v880 += 1 ;
                            }
                            bool v896;
                            v896 = v881 == 5;
                            Union8 v899;
                            if (v896){
                                v899 = Union8{Union8_1{v878}};
                            } else {
                                v899 = Union8{Union8_0{}};
                            }
                            Union8 v936;
                            switch (v877.tag) {
                                case 0: { // None
                                    v936 = v899;
                                    break;
                                }
                                case 1: { // Some
                                    static_array<unsigned char,5> v900 = v877.case1.v0;
                                    switch (v899.tag) {
                                        case 0: { // None
                                            v936 = v877;
                                            break;
                                        }
                                        case 1: { // Some
                                            static_array<unsigned char,5> v901 = v899.case1.v0;
                                            Union7 v902;
                                            v902 = Union7{Union7_0{}};
                                            int v903; Union7 v904;
                                            Tuple11 tmp36 = Tuple11{0, v902};
                                            v903 = tmp36.v0; v904 = tmp36.v1;
                                            while (while_method_5(v903)){
                                                bool v906;
                                                v906 = 0 <= v903;
                                                bool v908;
                                                if (v906){
                                                    bool v907;
                                                    v907 = v903 < 5;
                                                    v908 = v907;
                                                } else {
                                                    v908 = false;
                                                }
                                                bool v909;
                                                v909 = v908 == false;
                                                if (v909){
                                                    assert("Index must be in range." && v908);
                                                } else {
                                                }
                                                unsigned char v911;
                                                v911 = v900[v903];
                                                bool v914;
                                                if (v906){
                                                    bool v913;
                                                    v913 = v903 < 5;
                                                    v914 = v913;
                                                } else {
                                                    v914 = false;
                                                }
                                                bool v915;
                                                v915 = v914 == false;
                                                if (v915){
                                                    assert("Index must be in range." && v914);
                                                } else {
                                                }
                                                unsigned char v917;
                                                v917 = v901[v903];
                                                Union7 v929;
                                                switch (v904.tag) {
                                                    case 0: { // Eq
                                                        unsigned char v919;
                                                        v919 = v911 / 4u;
                                                        unsigned char v920;
                                                        v920 = v917 / 4u;
                                                        bool v921;
                                                        v921 = v919 < v920;
                                                        if (v921){
                                                            v929 = Union7{Union7_2{}};
                                                        } else {
                                                            bool v923;
                                                            v923 = v919 > v920;
                                                            if (v923){
                                                                v929 = Union7{Union7_1{}};
                                                            } else {
                                                                v929 = Union7{Union7_0{}};
                                                            }
                                                        }
                                                        break;
                                                    }
                                                    default: {
                                                        break;
                                                    }
                                                }
                                                v904 = v929;
                                                v903 += 1 ;
                                            }
                                            bool v930;
                                            switch (v904.tag) {
                                                case 1: { // Gt
                                                    v930 = true;
                                                    break;
                                                }
                                                default: {
                                                    v930 = false;
                                                }
                                            }
                                            static_array<unsigned char,5> v931;
                                            if (v930){
                                                v931 = v900;
                                            } else {
                                                v931 = v901;
                                            }
                                            v936 = Union8{Union8_1{v931}};
                                            break;
                                        }
                                        default: {
                                            assert("Invalid tag." && false); __trap();
                                        }
                                    }
                                    break;
                                }
                                default: {
                                    assert("Invalid tag." && false); __trap();
                                }
                            }
                            switch (v936.tag) {
                                case 0: { // None
                                    static_array<unsigned char,5> v938;
                                    int v940; int v941; unsigned char v942;
                                    Tuple10 tmp37 = Tuple10{0, 0, 12u};
                                    v940 = tmp37.v0; v941 = tmp37.v1; v942 = tmp37.v2;
                                    while (while_method_10(v940)){
                                        bool v944;
                                        v944 = 0 <= v940;
                                        bool v946;
                                        if (v944){
                                            bool v945;
                                            v945 = v940 < 7;
                                            v946 = v945;
                                        } else {
                                            v946 = false;
                                        }
                                        bool v947;
                                        v947 = v946 == false;
                                        if (v947){
                                            assert("Index must be in range." && v946);
                                        } else {
                                        }
                                        unsigned char v949;
                                        v949 = v129[v940];
                                        bool v951;
                                        v951 = v941 < 5;
                                        int v963; unsigned char v964;
                                        if (v951){
                                            unsigned char v952;
                                            v952 = v949 / 4u;
                                            unsigned char v953;
                                            v953 = v952 - 1u;
                                            bool v954;
                                            v954 = v942 == v953;
                                            bool v955;
                                            v955 = v954 != true;
                                            if (v955){
                                                bool v956;
                                                v956 = v942 == v952;
                                                int v957;
                                                if (v956){
                                                    v957 = v941;
                                                } else {
                                                    v957 = 0;
                                                }
                                                v938[v957] = v949;
                                                int v958;
                                                v958 = v957 + 1;
                                                v963 = v958; v964 = v953;
                                            } else {
                                                v963 = v941; v964 = v942;
                                            }
                                        } else {
                                            break;
                                        }
                                        v941 = v963;
                                        v942 = v964;
                                        v940 += 1 ;
                                    }
                                    bool v965;
                                    v965 = v941 == 4;
                                    bool v974;
                                    if (v965){
                                        unsigned char v966;
                                        v966 = v942 + 1u;
                                        bool v967;
                                        v967 = v966 == 0u;
                                        if (v967){
                                            unsigned char v968;
                                            v968 = v129[0];
                                            unsigned char v970;
                                            v970 = v968 / 4u;
                                            bool v971;
                                            v971 = v970 == 12u;
                                            if (v971){
                                                v938[4] = v968;
                                                v974 = true;
                                            } else {
                                                v974 = false;
                                            }
                                        } else {
                                            v974 = false;
                                        }
                                    } else {
                                        v974 = false;
                                    }
                                    Union8 v980;
                                    if (v974){
                                        v980 = Union8{Union8_1{v938}};
                                    } else {
                                        bool v976;
                                        v976 = v941 == 5;
                                        if (v976){
                                            v980 = Union8{Union8_1{v938}};
                                        } else {
                                            v980 = Union8{Union8_0{}};
                                        }
                                    }
                                    switch (v980.tag) {
                                        case 0: { // None
                                            static_array<unsigned char,3> v982;
                                            static_array<unsigned char,4> v984;
                                            int v986; int v987; int v988; unsigned char v989;
                                            Tuple12 tmp38 = Tuple12{0, 0, 0, 12u};
                                            v986 = tmp38.v0; v987 = tmp38.v1; v988 = tmp38.v2; v989 = tmp38.v3;
                                            while (while_method_10(v986)){
                                                bool v991;
                                                v991 = 0 <= v986;
                                                bool v993;
                                                if (v991){
                                                    bool v992;
                                                    v992 = v986 < 7;
                                                    v993 = v992;
                                                } else {
                                                    v993 = false;
                                                }
                                                bool v994;
                                                v994 = v993 == false;
                                                if (v994){
                                                    assert("Index must be in range." && v993);
                                                } else {
                                                }
                                                unsigned char v996;
                                                v996 = v129[v986];
                                                bool v998;
                                                v998 = v988 < 3;
                                                int v1006; int v1007; unsigned char v1008;
                                                if (v998){
                                                    unsigned char v999;
                                                    v999 = v996 / 4u;
                                                    bool v1000;
                                                    v1000 = v989 == v999;
                                                    int v1001;
                                                    if (v1000){
                                                        v1001 = v988;
                                                    } else {
                                                        v1001 = 0;
                                                    }
                                                    v982[v1001] = v996;
                                                    int v1002;
                                                    v1002 = v1001 + 1;
                                                    v1006 = v986; v1007 = v1002; v1008 = v999;
                                                } else {
                                                    break;
                                                }
                                                v987 = v1006;
                                                v988 = v1007;
                                                v989 = v1008;
                                                v986 += 1 ;
                                            }
                                            bool v1009;
                                            v1009 = v988 == 3;
                                            Union10 v1025;
                                            if (v1009){
                                                int v1010;
                                                v1010 = 0;
                                                while (while_method_0(v1010)){
                                                    int v1012;
                                                    v1012 = v987 + -2;
                                                    bool v1013;
                                                    v1013 = v1010 < v1012;
                                                    int v1014;
                                                    if (v1013){
                                                        v1014 = 0;
                                                    } else {
                                                        v1014 = 3;
                                                    }
                                                    int v1015;
                                                    v1015 = v1014 + v1010;
                                                    bool v1016;
                                                    v1016 = 0 <= v1015;
                                                    bool v1018;
                                                    if (v1016){
                                                        bool v1017;
                                                        v1017 = v1015 < 7;
                                                        v1018 = v1017;
                                                    } else {
                                                        v1018 = false;
                                                    }
                                                    bool v1019;
                                                    v1019 = v1018 == false;
                                                    if (v1019){
                                                        assert("Index must be in range." && v1018);
                                                    } else {
                                                    }
                                                    unsigned char v1021;
                                                    v1021 = v129[v1015];
                                                    v984[v1010] = v1021;
                                                    v1010 += 1 ;
                                                }
                                                v1025 = Union10{Union10_1{v982, v984}};
                                            } else {
                                                v1025 = Union10{Union10_0{}};
                                            }
                                            Union8 v1063;
                                            switch (v1025.tag) {
                                                case 0: { // None
                                                    v1063 = Union8{Union8_0{}};
                                                    break;
                                                }
                                                case 1: { // Some
                                                    static_array<unsigned char,3> v1026 = v1025.case1.v0; static_array<unsigned char,4> v1027 = v1025.case1.v1;
                                                    static_array<unsigned char,2> v1028;
                                                    int v1030;
                                                    v1030 = 0;
                                                    while (while_method_2(v1030)){
                                                        bool v1032;
                                                        v1032 = 0 <= v1030;
                                                        bool v1034;
                                                        if (v1032){
                                                            bool v1033;
                                                            v1033 = v1030 < 4;
                                                            v1034 = v1033;
                                                        } else {
                                                            v1034 = false;
                                                        }
                                                        bool v1035;
                                                        v1035 = v1034 == false;
                                                        if (v1035){
                                                            assert("Index must be in range." && v1034);
                                                        } else {
                                                        }
                                                        unsigned char v1037;
                                                        v1037 = v1027[v1030];
                                                        v1028[v1030] = v1037;
                                                        v1030 += 1 ;
                                                    }
                                                    static_array<unsigned char,5> v1039;
                                                    int v1041;
                                                    v1041 = 0;
                                                    while (while_method_4(v1041)){
                                                        bool v1043;
                                                        v1043 = 0 <= v1041;
                                                        bool v1045;
                                                        if (v1043){
                                                            bool v1044;
                                                            v1044 = v1041 < 3;
                                                            v1045 = v1044;
                                                        } else {
                                                            v1045 = false;
                                                        }
                                                        bool v1046;
                                                        v1046 = v1045 == false;
                                                        if (v1046){
                                                            assert("Index must be in range." && v1045);
                                                        } else {
                                                        }
                                                        unsigned char v1048;
                                                        v1048 = v1026[v1041];
                                                        v1039[v1041] = v1048;
                                                        v1041 += 1 ;
                                                    }
                                                    int v1050;
                                                    v1050 = 0;
                                                    while (while_method_2(v1050)){
                                                        bool v1052;
                                                        v1052 = 0 <= v1050;
                                                        bool v1054;
                                                        if (v1052){
                                                            bool v1053;
                                                            v1053 = v1050 < 2;
                                                            v1054 = v1053;
                                                        } else {
                                                            v1054 = false;
                                                        }
                                                        bool v1055;
                                                        v1055 = v1054 == false;
                                                        if (v1055){
                                                            assert("Index must be in range." && v1054);
                                                        } else {
                                                        }
                                                        unsigned char v1057;
                                                        v1057 = v1028[v1050];
                                                        int v1059;
                                                        v1059 = 3 + v1050;
                                                        v1039[v1059] = v1057;
                                                        v1050 += 1 ;
                                                    }
                                                    v1063 = Union8{Union8_1{v1039}};
                                                    break;
                                                }
                                                default: {
                                                    assert("Invalid tag." && false); __trap();
                                                }
                                            }
                                            switch (v1063.tag) {
                                                case 0: { // None
                                                    static_array<unsigned char,2> v1065;
                                                    static_array<unsigned char,5> v1067;
                                                    int v1069; int v1070; int v1071; unsigned char v1072;
                                                    Tuple12 tmp39 = Tuple12{0, 0, 0, 12u};
                                                    v1069 = tmp39.v0; v1070 = tmp39.v1; v1071 = tmp39.v2; v1072 = tmp39.v3;
                                                    while (while_method_10(v1069)){
                                                        bool v1074;
                                                        v1074 = 0 <= v1069;
                                                        bool v1076;
                                                        if (v1074){
                                                            bool v1075;
                                                            v1075 = v1069 < 7;
                                                            v1076 = v1075;
                                                        } else {
                                                            v1076 = false;
                                                        }
                                                        bool v1077;
                                                        v1077 = v1076 == false;
                                                        if (v1077){
                                                            assert("Index must be in range." && v1076);
                                                        } else {
                                                        }
                                                        unsigned char v1079;
                                                        v1079 = v129[v1069];
                                                        bool v1081;
                                                        v1081 = v1071 < 2;
                                                        int v1089; int v1090; unsigned char v1091;
                                                        if (v1081){
                                                            unsigned char v1082;
                                                            v1082 = v1079 / 4u;
                                                            bool v1083;
                                                            v1083 = v1072 == v1082;
                                                            int v1084;
                                                            if (v1083){
                                                                v1084 = v1071;
                                                            } else {
                                                                v1084 = 0;
                                                            }
                                                            v1065[v1084] = v1079;
                                                            int v1085;
                                                            v1085 = v1084 + 1;
                                                            v1089 = v1069; v1090 = v1085; v1091 = v1082;
                                                        } else {
                                                            break;
                                                        }
                                                        v1070 = v1089;
                                                        v1071 = v1090;
                                                        v1072 = v1091;
                                                        v1069 += 1 ;
                                                    }
                                                    bool v1092;
                                                    v1092 = v1071 == 2;
                                                    Union12 v1108;
                                                    if (v1092){
                                                        int v1093;
                                                        v1093 = 0;
                                                        while (while_method_5(v1093)){
                                                            int v1095;
                                                            v1095 = v1070 + -1;
                                                            bool v1096;
                                                            v1096 = v1093 < v1095;
                                                            int v1097;
                                                            if (v1096){
                                                                v1097 = 0;
                                                            } else {
                                                                v1097 = 2;
                                                            }
                                                            int v1098;
                                                            v1098 = v1097 + v1093;
                                                            bool v1099;
                                                            v1099 = 0 <= v1098;
                                                            bool v1101;
                                                            if (v1099){
                                                                bool v1100;
                                                                v1100 = v1098 < 7;
                                                                v1101 = v1100;
                                                            } else {
                                                                v1101 = false;
                                                            }
                                                            bool v1102;
                                                            v1102 = v1101 == false;
                                                            if (v1102){
                                                                assert("Index must be in range." && v1101);
                                                            } else {
                                                            }
                                                            unsigned char v1104;
                                                            v1104 = v129[v1098];
                                                            v1067[v1093] = v1104;
                                                            v1093 += 1 ;
                                                        }
                                                        v1108 = Union12{Union12_1{v1065, v1067}};
                                                    } else {
                                                        v1108 = Union12{Union12_0{}};
                                                    }
                                                    Union8 v1205;
                                                    switch (v1108.tag) {
                                                        case 0: { // None
                                                            v1205 = Union8{Union8_0{}};
                                                            break;
                                                        }
                                                        case 1: { // Some
                                                            static_array<unsigned char,2> v1109 = v1108.case1.v0; static_array<unsigned char,5> v1110 = v1108.case1.v1;
                                                            static_array<unsigned char,2> v1111;
                                                            static_array<unsigned char,3> v1113;
                                                            int v1115; int v1116; int v1117; unsigned char v1118;
                                                            Tuple12 tmp40 = Tuple12{0, 0, 0, 12u};
                                                            v1115 = tmp40.v0; v1116 = tmp40.v1; v1117 = tmp40.v2; v1118 = tmp40.v3;
                                                            while (while_method_5(v1115)){
                                                                bool v1120;
                                                                v1120 = 0 <= v1115;
                                                                bool v1122;
                                                                if (v1120){
                                                                    bool v1121;
                                                                    v1121 = v1115 < 5;
                                                                    v1122 = v1121;
                                                                } else {
                                                                    v1122 = false;
                                                                }
                                                                bool v1123;
                                                                v1123 = v1122 == false;
                                                                if (v1123){
                                                                    assert("Index must be in range." && v1122);
                                                                } else {
                                                                }
                                                                unsigned char v1125;
                                                                v1125 = v1110[v1115];
                                                                bool v1127;
                                                                v1127 = v1117 < 2;
                                                                int v1135; int v1136; unsigned char v1137;
                                                                if (v1127){
                                                                    unsigned char v1128;
                                                                    v1128 = v1125 / 4u;
                                                                    bool v1129;
                                                                    v1129 = v1118 == v1128;
                                                                    int v1130;
                                                                    if (v1129){
                                                                        v1130 = v1117;
                                                                    } else {
                                                                        v1130 = 0;
                                                                    }
                                                                    v1111[v1130] = v1125;
                                                                    int v1131;
                                                                    v1131 = v1130 + 1;
                                                                    v1135 = v1115; v1136 = v1131; v1137 = v1128;
                                                                } else {
                                                                    break;
                                                                }
                                                                v1116 = v1135;
                                                                v1117 = v1136;
                                                                v1118 = v1137;
                                                                v1115 += 1 ;
                                                            }
                                                            bool v1138;
                                                            v1138 = v1117 == 2;
                                                            Union13 v1154;
                                                            if (v1138){
                                                                int v1139;
                                                                v1139 = 0;
                                                                while (while_method_4(v1139)){
                                                                    int v1141;
                                                                    v1141 = v1116 + -1;
                                                                    bool v1142;
                                                                    v1142 = v1139 < v1141;
                                                                    int v1143;
                                                                    if (v1142){
                                                                        v1143 = 0;
                                                                    } else {
                                                                        v1143 = 2;
                                                                    }
                                                                    int v1144;
                                                                    v1144 = v1143 + v1139;
                                                                    bool v1145;
                                                                    v1145 = 0 <= v1144;
                                                                    bool v1147;
                                                                    if (v1145){
                                                                        bool v1146;
                                                                        v1146 = v1144 < 5;
                                                                        v1147 = v1146;
                                                                    } else {
                                                                        v1147 = false;
                                                                    }
                                                                    bool v1148;
                                                                    v1148 = v1147 == false;
                                                                    if (v1148){
                                                                        assert("Index must be in range." && v1147);
                                                                    } else {
                                                                    }
                                                                    unsigned char v1150;
                                                                    v1150 = v1110[v1144];
                                                                    v1113[v1139] = v1150;
                                                                    v1139 += 1 ;
                                                                }
                                                                v1154 = Union13{Union13_1{v1111, v1113}};
                                                            } else {
                                                                v1154 = Union13{Union13_0{}};
                                                            }
                                                            switch (v1154.tag) {
                                                                case 0: { // None
                                                                    v1205 = Union8{Union8_0{}};
                                                                    break;
                                                                }
                                                                case 1: { // Some
                                                                    static_array<unsigned char,2> v1155 = v1154.case1.v0; static_array<unsigned char,3> v1156 = v1154.case1.v1;
                                                                    static_array<unsigned char,1> v1157;
                                                                    int v1159;
                                                                    v1159 = 0;
                                                                    while (while_method_6(v1159)){
                                                                        bool v1161;
                                                                        v1161 = 0 <= v1159;
                                                                        bool v1163;
                                                                        if (v1161){
                                                                            bool v1162;
                                                                            v1162 = v1159 < 3;
                                                                            v1163 = v1162;
                                                                        } else {
                                                                            v1163 = false;
                                                                        }
                                                                        bool v1164;
                                                                        v1164 = v1163 == false;
                                                                        if (v1164){
                                                                            assert("Index must be in range." && v1163);
                                                                        } else {
                                                                        }
                                                                        unsigned char v1166;
                                                                        v1166 = v1156[v1159];
                                                                        v1157[v1159] = v1166;
                                                                        v1159 += 1 ;
                                                                    }
                                                                    static_array<unsigned char,5> v1168;
                                                                    int v1170;
                                                                    v1170 = 0;
                                                                    while (while_method_2(v1170)){
                                                                        bool v1172;
                                                                        v1172 = 0 <= v1170;
                                                                        bool v1174;
                                                                        if (v1172){
                                                                            bool v1173;
                                                                            v1173 = v1170 < 2;
                                                                            v1174 = v1173;
                                                                        } else {
                                                                            v1174 = false;
                                                                        }
                                                                        bool v1175;
                                                                        v1175 = v1174 == false;
                                                                        if (v1175){
                                                                            assert("Index must be in range." && v1174);
                                                                        } else {
                                                                        }
                                                                        unsigned char v1177;
                                                                        v1177 = v1109[v1170];
                                                                        v1168[v1170] = v1177;
                                                                        v1170 += 1 ;
                                                                    }
                                                                    int v1179;
                                                                    v1179 = 0;
                                                                    while (while_method_2(v1179)){
                                                                        bool v1181;
                                                                        v1181 = 0 <= v1179;
                                                                        bool v1183;
                                                                        if (v1181){
                                                                            bool v1182;
                                                                            v1182 = v1179 < 2;
                                                                            v1183 = v1182;
                                                                        } else {
                                                                            v1183 = false;
                                                                        }
                                                                        bool v1184;
                                                                        v1184 = v1183 == false;
                                                                        if (v1184){
                                                                            assert("Index must be in range." && v1183);
                                                                        } else {
                                                                        }
                                                                        unsigned char v1186;
                                                                        v1186 = v1155[v1179];
                                                                        int v1188;
                                                                        v1188 = 2 + v1179;
                                                                        v1168[v1188] = v1186;
                                                                        v1179 += 1 ;
                                                                    }
                                                                    int v1189;
                                                                    v1189 = 0;
                                                                    while (while_method_6(v1189)){
                                                                        bool v1191;
                                                                        v1191 = 0 <= v1189;
                                                                        bool v1193;
                                                                        if (v1191){
                                                                            bool v1192;
                                                                            v1192 = v1189 < 1;
                                                                            v1193 = v1192;
                                                                        } else {
                                                                            v1193 = false;
                                                                        }
                                                                        bool v1194;
                                                                        v1194 = v1193 == false;
                                                                        if (v1194){
                                                                            assert("Index must be in range." && v1193);
                                                                        } else {
                                                                        }
                                                                        unsigned char v1196;
                                                                        v1196 = v1157[v1189];
                                                                        int v1198;
                                                                        v1198 = 4 + v1189;
                                                                        v1168[v1198] = v1196;
                                                                        v1189 += 1 ;
                                                                    }
                                                                    v1205 = Union8{Union8_1{v1168}};
                                                                    break;
                                                                }
                                                                default: {
                                                                    assert("Invalid tag." && false); __trap();
                                                                }
                                                            }
                                                            break;
                                                        }
                                                        default: {
                                                            assert("Invalid tag." && false); __trap();
                                                        }
                                                    }
                                                    switch (v1205.tag) {
                                                        case 0: { // None
                                                            static_array<unsigned char,2> v1207;
                                                            static_array<unsigned char,5> v1209;
                                                            int v1211; int v1212; int v1213; unsigned char v1214;
                                                            Tuple12 tmp41 = Tuple12{0, 0, 0, 12u};
                                                            v1211 = tmp41.v0; v1212 = tmp41.v1; v1213 = tmp41.v2; v1214 = tmp41.v3;
                                                            while (while_method_10(v1211)){
                                                                bool v1216;
                                                                v1216 = 0 <= v1211;
                                                                bool v1218;
                                                                if (v1216){
                                                                    bool v1217;
                                                                    v1217 = v1211 < 7;
                                                                    v1218 = v1217;
                                                                } else {
                                                                    v1218 = false;
                                                                }
                                                                bool v1219;
                                                                v1219 = v1218 == false;
                                                                if (v1219){
                                                                    assert("Index must be in range." && v1218);
                                                                } else {
                                                                }
                                                                unsigned char v1221;
                                                                v1221 = v129[v1211];
                                                                bool v1223;
                                                                v1223 = v1213 < 2;
                                                                int v1231; int v1232; unsigned char v1233;
                                                                if (v1223){
                                                                    unsigned char v1224;
                                                                    v1224 = v1221 / 4u;
                                                                    bool v1225;
                                                                    v1225 = v1214 == v1224;
                                                                    int v1226;
                                                                    if (v1225){
                                                                        v1226 = v1213;
                                                                    } else {
                                                                        v1226 = 0;
                                                                    }
                                                                    v1207[v1226] = v1221;
                                                                    int v1227;
                                                                    v1227 = v1226 + 1;
                                                                    v1231 = v1211; v1232 = v1227; v1233 = v1224;
                                                                } else {
                                                                    break;
                                                                }
                                                                v1212 = v1231;
                                                                v1213 = v1232;
                                                                v1214 = v1233;
                                                                v1211 += 1 ;
                                                            }
                                                            bool v1234;
                                                            v1234 = v1213 == 2;
                                                            Union12 v1250;
                                                            if (v1234){
                                                                int v1235;
                                                                v1235 = 0;
                                                                while (while_method_5(v1235)){
                                                                    int v1237;
                                                                    v1237 = v1212 + -1;
                                                                    bool v1238;
                                                                    v1238 = v1235 < v1237;
                                                                    int v1239;
                                                                    if (v1238){
                                                                        v1239 = 0;
                                                                    } else {
                                                                        v1239 = 2;
                                                                    }
                                                                    int v1240;
                                                                    v1240 = v1239 + v1235;
                                                                    bool v1241;
                                                                    v1241 = 0 <= v1240;
                                                                    bool v1243;
                                                                    if (v1241){
                                                                        bool v1242;
                                                                        v1242 = v1240 < 7;
                                                                        v1243 = v1242;
                                                                    } else {
                                                                        v1243 = false;
                                                                    }
                                                                    bool v1244;
                                                                    v1244 = v1243 == false;
                                                                    if (v1244){
                                                                        assert("Index must be in range." && v1243);
                                                                    } else {
                                                                    }
                                                                    unsigned char v1246;
                                                                    v1246 = v129[v1240];
                                                                    v1209[v1235] = v1246;
                                                                    v1235 += 1 ;
                                                                }
                                                                v1250 = Union12{Union12_1{v1207, v1209}};
                                                            } else {
                                                                v1250 = Union12{Union12_0{}};
                                                            }
                                                            Union8 v1288;
                                                            switch (v1250.tag) {
                                                                case 0: { // None
                                                                    v1288 = Union8{Union8_0{}};
                                                                    break;
                                                                }
                                                                case 1: { // Some
                                                                    static_array<unsigned char,2> v1251 = v1250.case1.v0; static_array<unsigned char,5> v1252 = v1250.case1.v1;
                                                                    static_array<unsigned char,3> v1253;
                                                                    int v1255;
                                                                    v1255 = 0;
                                                                    while (while_method_4(v1255)){
                                                                        bool v1257;
                                                                        v1257 = 0 <= v1255;
                                                                        bool v1259;
                                                                        if (v1257){
                                                                            bool v1258;
                                                                            v1258 = v1255 < 5;
                                                                            v1259 = v1258;
                                                                        } else {
                                                                            v1259 = false;
                                                                        }
                                                                        bool v1260;
                                                                        v1260 = v1259 == false;
                                                                        if (v1260){
                                                                            assert("Index must be in range." && v1259);
                                                                        } else {
                                                                        }
                                                                        unsigned char v1262;
                                                                        v1262 = v1252[v1255];
                                                                        v1253[v1255] = v1262;
                                                                        v1255 += 1 ;
                                                                    }
                                                                    static_array<unsigned char,5> v1264;
                                                                    int v1266;
                                                                    v1266 = 0;
                                                                    while (while_method_2(v1266)){
                                                                        bool v1268;
                                                                        v1268 = 0 <= v1266;
                                                                        bool v1270;
                                                                        if (v1268){
                                                                            bool v1269;
                                                                            v1269 = v1266 < 2;
                                                                            v1270 = v1269;
                                                                        } else {
                                                                            v1270 = false;
                                                                        }
                                                                        bool v1271;
                                                                        v1271 = v1270 == false;
                                                                        if (v1271){
                                                                            assert("Index must be in range." && v1270);
                                                                        } else {
                                                                        }
                                                                        unsigned char v1273;
                                                                        v1273 = v1251[v1266];
                                                                        v1264[v1266] = v1273;
                                                                        v1266 += 1 ;
                                                                    }
                                                                    int v1275;
                                                                    v1275 = 0;
                                                                    while (while_method_4(v1275)){
                                                                        bool v1277;
                                                                        v1277 = 0 <= v1275;
                                                                        bool v1279;
                                                                        if (v1277){
                                                                            bool v1278;
                                                                            v1278 = v1275 < 3;
                                                                            v1279 = v1278;
                                                                        } else {
                                                                            v1279 = false;
                                                                        }
                                                                        bool v1280;
                                                                        v1280 = v1279 == false;
                                                                        if (v1280){
                                                                            assert("Index must be in range." && v1279);
                                                                        } else {
                                                                        }
                                                                        unsigned char v1282;
                                                                        v1282 = v1253[v1275];
                                                                        int v1284;
                                                                        v1284 = 2 + v1275;
                                                                        v1264[v1284] = v1282;
                                                                        v1275 += 1 ;
                                                                    }
                                                                    v1288 = Union8{Union8_1{v1264}};
                                                                    break;
                                                                }
                                                                default: {
                                                                    assert("Invalid tag." && false); __trap();
                                                                }
                                                            }
                                                            switch (v1288.tag) {
                                                                case 0: { // None
                                                                    static_array<unsigned char,5> v1290;
                                                                    int v1292;
                                                                    v1292 = 0;
                                                                    while (while_method_5(v1292)){
                                                                        bool v1294;
                                                                        v1294 = 0 <= v1292;
                                                                        bool v1296;
                                                                        if (v1294){
                                                                            bool v1295;
                                                                            v1295 = v1292 < 7;
                                                                            v1296 = v1295;
                                                                        } else {
                                                                            v1296 = false;
                                                                        }
                                                                        bool v1297;
                                                                        v1297 = v1296 == false;
                                                                        if (v1297){
                                                                            assert("Index must be in range." && v1296);
                                                                        } else {
                                                                        }
                                                                        unsigned char v1299;
                                                                        v1299 = v129[v1292];
                                                                        v1290[v1292] = v1299;
                                                                        v1292 += 1 ;
                                                                    }
                                                                    v1331 = v1290; v1332 = 0;
                                                                    break;
                                                                }
                                                                case 1: { // Some
                                                                    static_array<unsigned char,5> v1289 = v1288.case1.v0;
                                                                    v1331 = v1289; v1332 = 1;
                                                                    break;
                                                                }
                                                                default: {
                                                                    assert("Invalid tag." && false); __trap();
                                                                }
                                                            }
                                                            break;
                                                        }
                                                        case 1: { // Some
                                                            static_array<unsigned char,5> v1206 = v1205.case1.v0;
                                                            v1331 = v1206; v1332 = 2;
                                                            break;
                                                        }
                                                        default: {
                                                            assert("Invalid tag." && false); __trap();
                                                        }
                                                    }
                                                    break;
                                                }
                                                case 1: { // Some
                                                    static_array<unsigned char,5> v1064 = v1063.case1.v0;
                                                    v1331 = v1064; v1332 = 3;
                                                    break;
                                                }
                                                default: {
                                                    assert("Invalid tag." && false); __trap();
                                                }
                                            }
                                            break;
                                        }
                                        case 1: { // Some
                                            static_array<unsigned char,5> v981 = v980.case1.v0;
                                            v1331 = v981; v1332 = 4;
                                            break;
                                        }
                                        default: {
                                            assert("Invalid tag." && false); __trap();
                                        }
                                    }
                                    break;
                                }
                                case 1: { // Some
                                    static_array<unsigned char,5> v937 = v936.case1.v0;
                                    v1331 = v937; v1332 = 5;
                                    break;
                                }
                                default: {
                                    assert("Invalid tag." && false); __trap();
                                }
                            }
                            break;
                        }
                        case 1: { // Some
                            static_array<unsigned char,5> v737 = v736.case1.v0;
                            v1331 = v737; v1332 = 6;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false); __trap();
                        }
                    }
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,5> v616 = v615.case1.v0;
                    v1331 = v616; v1332 = 7;
                    break;
                }
                default: {
                    assert("Invalid tag." && false); __trap();
                }
            }
            break;
        }
        case 1: { // Some
            static_array<unsigned char,5> v533 = v532.case1.v0;
            v1331 = v533; v1332 = 8;
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    return Tuple0{v1331, v1332};
}
__device__ void method_0(StackMut0 & v0, int v1, Union3 v2){
    v0.v0 = 4503599627370495ull;
    static_array<float,2> v3;
    v3[0] = 0.0f;
    v3[1] = 0.0f;
    v0.v4 = v3;
    static_array_list<Union1,128> & v5 = v0.v2;
    v5.unsafe_set_length(0);
    static_array<Union0,2> v6;
    Union0 v8;
    v8 = Union0{Union0_0{}};
    v6[0] = v8;
    Union0 v10;
    v10 = Union0{Union0_0{}};
    v6[1] = v10;
    int v12;
    v12 = v1 ^ 1;
    Union0 v13;
    v13 = Union0{Union0_2{}};
    v6[v12] = v13;
    v0.v3 = v6;
    static_array_list<Union1,128> & v15 = v0.v2;
    unsigned long long & v16 = v0.v0;
    Union5 v17;
    v17 = Union5{Union5_1{v2}};
    Union5 v18;
    v18 = v17;
    while (while_method_3(v18)){
        Union5 v992;
        switch (v18.tag) {
            case 0: { // None
                v992 = Union5{Union5_0{}};
                break;
            }
            case 1: { // Some
                Union3 v20 = v18.case1.v0;
                Union6 v643;
                switch (v20.tag) {
                    case 0: { // G_Flop
                        int v504 = v20.case0.v0; static_array<static_array<unsigned char,2>,2> v505 = v20.case0.v1; static_array<int,2> v506 = v20.case0.v2; int v507 = v20.case0.v3; static_array<int,2> v508 = v20.case0.v4; Union4 v509 = v20.case0.v5;
                        curandStatePhilox4_32_10_t & v510 = v0.v5;
                        curandStatePhilox4_32_10_t & v511 = v510;
                        static_array<unsigned char,3> v512; unsigned long long v513;
                        Tuple1 tmp2 = draw_cards_1(v511, v16);
                        v512 = tmp2.v0; v513 = tmp2.v1;
                        v0.v0 = v513;
                        static_array_list<unsigned char,5> v514;
                        v514 = get_community_cards_4(v509, v512);
                        Union1 v515;
                        v515 = Union1{Union1_0{v514}};
                        v15.push(v515);
                        Union4 v518;
                        switch (v509.tag) {
                            case 1: { // Preflop
                                v518 = Union4{Union4_0{v512}};
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in flop.");
                                __trap();
                            }
                        }
                        int v519;
                        v519 = 2;
                        int v520;
                        v520 = 0;
                        Union3 v521;
                        v521 = try_round_5(v519, v505, v506, v520, v508, v518);
                        v643 = Union6{Union6_2{v521}};
                        break;
                    }
                    case 1: { // G_Fold
                        int v21 = v20.case1.v0; static_array<static_array<unsigned char,2>,2> v22 = v20.case1.v1; static_array<int,2> v23 = v20.case1.v2; int v24 = v20.case1.v3; static_array<int,2> v25 = v20.case1.v4; Union4 v26 = v20.case1.v5;
                        int v27;
                        v27 = v24 % 2;
                        bool v28;
                        v28 = 0 <= v27;
                        bool v30;
                        if (v28){
                            bool v29;
                            v29 = v27 < 2;
                            v30 = v29;
                        } else {
                            v30 = false;
                        }
                        bool v31;
                        v31 = v30 == false;
                        if (v31){
                            assert("Index must be in range." && v30);
                        } else {
                        }
                        int v33;
                        v33 = v23[v27];
                        int v35;
                        v35 = -v33;
                        float v36;
                        v36 = (float)v35;
                        static_array<float,2> & v37 = v0.v4;
                        v37[v27] = v36;
                        int v38;
                        v38 = v27 ^ 1;
                        float v39;
                        v39 = -v36;
                        v37[v38] = v39;
                        int v40;
                        v40 = v24 + 1;
                        int v41;
                        v41 = v40 % 2;
                        Union1 v42;
                        v42 = Union1{Union1_1{v33, v41}};
                        v15.push(v42);
                        v643 = Union6{Union6_0{}};
                        break;
                    }
                    case 2: { // G_Preflop
                        curandStatePhilox4_32_10_t & v605 = v0.v5;
                        curandStatePhilox4_32_10_t & v606 = v605;
                        static_array<unsigned char,2> v607; unsigned long long v608;
                        Tuple5 tmp7 = draw_cards_8(v606, v16);
                        v607 = tmp7.v0; v608 = tmp7.v1;
                        v0.v0 = v608;
                        curandStatePhilox4_32_10_t & v609 = v0.v5;
                        curandStatePhilox4_32_10_t & v610 = v609;
                        static_array<unsigned char,2> v611; unsigned long long v612;
                        Tuple5 tmp8 = draw_cards_8(v610, v16);
                        v611 = tmp8.v0; v612 = tmp8.v1;
                        v0.v0 = v612;
                        Union1 v613;
                        v613 = Union1{Union1_3{0, v607}};
                        v15.push(v613);
                        Union1 v614;
                        v614 = Union1{Union1_3{1, v611}};
                        v15.push(v614);
                        static_array<static_array<unsigned char,2>,2> v615;
                        v615[0] = v607;
                        v615[1] = v611;
                        static_array<int,2> v617;
                        v617[0] = 2;
                        v617[1] = 1;
                        static_array<int,2> v619;
                        int v621;
                        v621 = 0;
                        while (while_method_2(v621)){
                            bool v623;
                            v623 = 0 <= v621;
                            bool v625;
                            if (v623){
                                bool v624;
                                v624 = v621 < 2;
                                v625 = v624;
                            } else {
                                v625 = false;
                            }
                            bool v626;
                            v626 = v625 == false;
                            if (v626){
                                assert("Index must be in range." && v625);
                            } else {
                            }
                            int v628;
                            v628 = v617[v621];
                            int v630;
                            v630 = 100 - v628;
                            v619[v621] = v630;
                            v621 += 1 ;
                        }
                        int v631;
                        v631 = 2;
                        int v632;
                        v632 = 0;
                        Union4 v633;
                        v633 = Union4{Union4_1{}};
                        Union3 v634;
                        v634 = try_round_5(v631, v615, v617, v632, v619, v633);
                        v643 = Union6{Union6_2{v634}};
                        break;
                    }
                    case 3: { // G_River
                        int v564 = v20.case3.v0; static_array<static_array<unsigned char,2>,2> v565 = v20.case3.v1; static_array<int,2> v566 = v20.case3.v2; int v567 = v20.case3.v3; static_array<int,2> v568 = v20.case3.v4; Union4 v569 = v20.case3.v5;
                        curandStatePhilox4_32_10_t & v570 = v0.v5;
                        curandStatePhilox4_32_10_t & v571 = v570;
                        static_array<unsigned char,1> v572; unsigned long long v573;
                        Tuple6 tmp11 = draw_cards_9(v571, v16);
                        v572 = tmp11.v0; v573 = tmp11.v1;
                        v0.v0 = v573;
                        static_array_list<unsigned char,5> v574;
                        v574 = get_community_cards_10(v569, v572);
                        Union1 v575;
                        v575 = Union1{Union1_0{v574}};
                        v15.push(v575);
                        Union4 v600;
                        switch (v569.tag) {
                            case 3: { // Turn
                                static_array<unsigned char,4> v576 = v569.case3.v0;
                                static_array<unsigned char,5> v577;
                                int v579;
                                v579 = 0;
                                while (while_method_0(v579)){
                                    bool v581;
                                    v581 = 0 <= v579;
                                    bool v583;
                                    if (v581){
                                        bool v582;
                                        v582 = v579 < 4;
                                        v583 = v582;
                                    } else {
                                        v583 = false;
                                    }
                                    bool v584;
                                    v584 = v583 == false;
                                    if (v584){
                                        assert("Index must be in range." && v583);
                                    } else {
                                    }
                                    unsigned char v586;
                                    v586 = v576[v579];
                                    v577[v579] = v586;
                                    v579 += 1 ;
                                }
                                int v588;
                                v588 = 0;
                                while (while_method_6(v588)){
                                    bool v590;
                                    v590 = 0 <= v588;
                                    bool v592;
                                    if (v590){
                                        bool v591;
                                        v591 = v588 < 1;
                                        v592 = v591;
                                    } else {
                                        v592 = false;
                                    }
                                    bool v593;
                                    v593 = v592 == false;
                                    if (v593){
                                        assert("Index must be in range." && v592);
                                    } else {
                                    }
                                    unsigned char v595;
                                    v595 = v572[v588];
                                    int v597;
                                    v597 = 4 + v588;
                                    v577[v597] = v595;
                                    v588 += 1 ;
                                }
                                v600 = Union4{Union4_2{v577}};
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in river.");
                                __trap();
                            }
                        }
                        int v601;
                        v601 = 2;
                        int v602;
                        v602 = 0;
                        Union3 v603;
                        v603 = try_round_5(v601, v565, v566, v602, v568, v600);
                        v643 = Union6{Union6_2{v603}};
                        break;
                    }
                    case 4: { // G_Round
                        int v163 = v20.case4.v0; static_array<static_array<unsigned char,2>,2> v164 = v20.case4.v1; static_array<int,2> v165 = v20.case4.v2; int v166 = v20.case4.v3; static_array<int,2> v167 = v20.case4.v4; Union4 v168 = v20.case4.v5;
                        int v169;
                        v169 = v166 % 2;
                        static_array<Union0,2> & v170 = v0.v3;
                        bool v171;
                        v171 = 0 <= v169;
                        bool v173;
                        if (v171){
                            bool v172;
                            v172 = v169 < 2;
                            v173 = v172;
                        } else {
                            v173 = false;
                        }
                        bool v174;
                        v174 = v173 == false;
                        if (v174){
                            assert("Index must be in range." && v173);
                        } else {
                        }
                        Union0 v176;
                        v176 = v170[v169];
                        Union2 v491;
                        switch (v176.tag) {
                            case 0: { // Computer
                                curandStatePhilox4_32_10_t & v179 = v0.v5;
                                curandStatePhilox4_32_10_t & v180 = v179;
                                static_array<int,2> v181;
                                int v183;
                                v183 = 0;
                                while (while_method_2(v183)){
                                    bool v185;
                                    v185 = 0 <= v183;
                                    bool v187;
                                    if (v185){
                                        bool v186;
                                        v186 = v183 < 2;
                                        v187 = v186;
                                    } else {
                                        v187 = false;
                                    }
                                    bool v188;
                                    v188 = v187 == false;
                                    if (v188){
                                        assert("Index must be in range." && v187);
                                    } else {
                                    }
                                    int v190;
                                    v190 = v167[v183];
                                    bool v193;
                                    if (v185){
                                        bool v192;
                                        v192 = v183 < 2;
                                        v193 = v192;
                                    } else {
                                        v193 = false;
                                    }
                                    bool v194;
                                    v194 = v193 == false;
                                    if (v194){
                                        assert("Index must be in range." && v193);
                                    } else {
                                    }
                                    int v196;
                                    v196 = v165[v183];
                                    int v198;
                                    v198 = v190 + v196;
                                    v181[v183] = v198;
                                    v183 += 1 ;
                                }
                                int v199;
                                v199 = v165[0];
                                int v201; int v202;
                                Tuple4 tmp12 = Tuple4{1, v199};
                                v201 = tmp12.v0; v202 = tmp12.v1;
                                while (while_method_2(v201)){
                                    bool v204;
                                    v204 = 0 <= v201;
                                    bool v206;
                                    if (v204){
                                        bool v205;
                                        v205 = v201 < 2;
                                        v206 = v205;
                                    } else {
                                        v206 = false;
                                    }
                                    bool v207;
                                    v207 = v206 == false;
                                    if (v207){
                                        assert("Index must be in range." && v206);
                                    } else {
                                    }
                                    int v209;
                                    v209 = v165[v201];
                                    bool v211;
                                    v211 = v202 >= v209;
                                    int v212;
                                    if (v211){
                                        v212 = v202;
                                    } else {
                                        v212 = v209;
                                    }
                                    v202 = v212;
                                    v201 += 1 ;
                                }
                                bool v214;
                                if (v171){
                                    bool v213;
                                    v213 = v169 < 2;
                                    v214 = v213;
                                } else {
                                    v214 = false;
                                }
                                bool v215;
                                v215 = v214 == false;
                                if (v215){
                                    assert("Index must be in range." && v214);
                                } else {
                                }
                                int v217;
                                v217 = v181[v169];
                                bool v219;
                                v219 = v202 < v217;
                                int v220;
                                if (v219){
                                    v220 = v202;
                                } else {
                                    v220 = v217;
                                }
                                static_array<int,2> v221;
                                int v223;
                                v223 = 0;
                                while (while_method_2(v223)){
                                    bool v225;
                                    v225 = 0 <= v223;
                                    bool v227;
                                    if (v225){
                                        bool v226;
                                        v226 = v223 < 2;
                                        v227 = v226;
                                    } else {
                                        v227 = false;
                                    }
                                    bool v228;
                                    v228 = v227 == false;
                                    if (v228){
                                        assert("Index must be in range." && v227);
                                    } else {
                                    }
                                    int v230;
                                    v230 = v165[v223];
                                    bool v232;
                                    v232 = v169 == v223;
                                    int v233;
                                    if (v232){
                                        v233 = v220;
                                    } else {
                                        v233 = v230;
                                    }
                                    v221[v223] = v233;
                                    v223 += 1 ;
                                }
                                int v234;
                                v234 = v221[0];
                                int v236; int v237;
                                Tuple4 tmp13 = Tuple4{1, v234};
                                v236 = tmp13.v0; v237 = tmp13.v1;
                                while (while_method_2(v236)){
                                    bool v239;
                                    v239 = 0 <= v236;
                                    bool v241;
                                    if (v239){
                                        bool v240;
                                        v240 = v236 < 2;
                                        v241 = v240;
                                    } else {
                                        v241 = false;
                                    }
                                    bool v242;
                                    v242 = v241 == false;
                                    if (v242){
                                        assert("Index must be in range." && v241);
                                    } else {
                                    }
                                    int v244;
                                    v244 = v221[v236];
                                    int v246;
                                    v246 = v237 + v244;
                                    v237 = v246;
                                    v236 += 1 ;
                                }
                                static_array<int,2> v247;
                                int v249;
                                v249 = 0;
                                while (while_method_2(v249)){
                                    bool v251;
                                    v251 = 0 <= v249;
                                    bool v253;
                                    if (v251){
                                        bool v252;
                                        v252 = v249 < 2;
                                        v253 = v252;
                                    } else {
                                        v253 = false;
                                    }
                                    bool v254;
                                    v254 = v253 == false;
                                    if (v254){
                                        assert("Index must be in range." && v253);
                                    } else {
                                    }
                                    int v256;
                                    v256 = v181[v249];
                                    bool v259;
                                    if (v251){
                                        bool v258;
                                        v258 = v249 < 2;
                                        v259 = v258;
                                    } else {
                                        v259 = false;
                                    }
                                    bool v260;
                                    v260 = v259 == false;
                                    if (v260){
                                        assert("Index must be in range." && v259);
                                    } else {
                                    }
                                    int v262;
                                    v262 = v221[v249];
                                    int v264;
                                    v264 = v256 - v262;
                                    v247[v249] = v264;
                                    v249 += 1 ;
                                }
                                bool v266;
                                if (v171){
                                    bool v265;
                                    v265 = v169 < 2;
                                    v266 = v265;
                                } else {
                                    v266 = false;
                                }
                                bool v267;
                                v267 = v266 == false;
                                if (v267){
                                    assert("Index must be in range." && v266);
                                } else {
                                }
                                int v269;
                                v269 = v165[v169];
                                bool v271;
                                v271 = v269 < v202;
                                float v272;
                                if (v271){
                                    v272 = 1.0f;
                                } else {
                                    v272 = 0.0f;
                                }
                                int v273;
                                v273 = v237 / 3;
                                bool v274;
                                v274 = v163 <= v273;
                                bool v282;
                                if (v274){
                                    bool v276;
                                    if (v171){
                                        bool v275;
                                        v275 = v169 < 2;
                                        v276 = v275;
                                    } else {
                                        v276 = false;
                                    }
                                    bool v277;
                                    v277 = v276 == false;
                                    if (v277){
                                        assert("Index must be in range." && v276);
                                    } else {
                                    }
                                    int v279;
                                    v279 = v247[v169];
                                    bool v281;
                                    v281 = v273 < v279;
                                    v282 = v281;
                                } else {
                                    v282 = false;
                                }
                                float v283;
                                if (v282){
                                    v283 = 1.0f;
                                } else {
                                    v283 = 0.0f;
                                }
                                int v284;
                                v284 = v237 / 2;
                                bool v285;
                                v285 = v163 <= v284;
                                bool v293;
                                if (v285){
                                    bool v287;
                                    if (v171){
                                        bool v286;
                                        v286 = v169 < 2;
                                        v287 = v286;
                                    } else {
                                        v287 = false;
                                    }
                                    bool v288;
                                    v288 = v287 == false;
                                    if (v288){
                                        assert("Index must be in range." && v287);
                                    } else {
                                    }
                                    int v290;
                                    v290 = v247[v169];
                                    bool v292;
                                    v292 = v284 < v290;
                                    v293 = v292;
                                } else {
                                    v293 = false;
                                }
                                float v294;
                                if (v293){
                                    v294 = 1.0f;
                                } else {
                                    v294 = 0.0f;
                                }
                                bool v295;
                                v295 = v163 <= v237;
                                bool v303;
                                if (v295){
                                    bool v297;
                                    if (v171){
                                        bool v296;
                                        v296 = v169 < 2;
                                        v297 = v296;
                                    } else {
                                        v297 = false;
                                    }
                                    bool v298;
                                    v298 = v297 == false;
                                    if (v298){
                                        assert("Index must be in range." && v297);
                                    } else {
                                    }
                                    int v300;
                                    v300 = v247[v169];
                                    bool v302;
                                    v302 = v237 < v300;
                                    v303 = v302;
                                } else {
                                    v303 = false;
                                }
                                float v304;
                                if (v303){
                                    v304 = 1.0f;
                                } else {
                                    v304 = 0.0f;
                                }
                                static_array<Tuple7,6> v305;
                                Union2 v307;
                                v307 = Union2{Union2_2{}};
                                v305[0] = Tuple7{v307, v272};
                                Union2 v309;
                                v309 = Union2{Union2_1{}};
                                v305[1] = Tuple7{v309, 4.0f};
                                Union2 v311;
                                v311 = Union2{Union2_3{v273}};
                                v305[2] = Tuple7{v311, v283};
                                Union2 v313;
                                v313 = Union2{Union2_3{v284}};
                                v305[3] = Tuple7{v313, v294};
                                Union2 v315;
                                v315 = Union2{Union2_3{v237}};
                                v305[4] = Tuple7{v315, v304};
                                Union2 v317;
                                v317 = Union2{Union2_0{}};
                                v305[5] = Tuple7{v317, 1.0f};
                                Union2 v319;
                                v319 = sample_discrete_11(v305, v180);
                                int v320;
                                v320 = sizeof(Union2);
                                unsigned long long v321;
                                v321 = (unsigned long long)v320;
                                bool v322;
                                v322 = v321 <= 98304ull;
                                bool v323;
                                v323 = v322 == false;
                                if (v323){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v322);
                                } else {
                                }
                                extern __shared__ unsigned char v325[];
                                bool v326;
                                v326 = v321 <= v321;
                                bool v327;
                                v327 = v326 == false;
                                if (v327){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v326);
                                } else {
                                }
                                Union2 * v329;
                                v329 = reinterpret_cast<Union2 *>(&v325[0ull]);
                                int v331;
                                v331 = threadIdx.x;
                                bool v332;
                                v332 = v331 == 0;
                                if (v332){
                                    v329[0] = v319;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                Union2 v333;
                                v333 = v329[0];
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                v491 = v333;
                                break;
                            }
                            case 1: { // Human
                                printf("%s\n", "Humans aren't allowed during training.");
                                __trap();
                                break;
                            }
                            case 2: { // Random
                                curandStatePhilox4_32_10_t & v334 = v0.v5;
                                curandStatePhilox4_32_10_t & v335 = v334;
                                static_array<int,2> v336;
                                int v338;
                                v338 = 0;
                                while (while_method_2(v338)){
                                    bool v340;
                                    v340 = 0 <= v338;
                                    bool v342;
                                    if (v340){
                                        bool v341;
                                        v341 = v338 < 2;
                                        v342 = v341;
                                    } else {
                                        v342 = false;
                                    }
                                    bool v343;
                                    v343 = v342 == false;
                                    if (v343){
                                        assert("Index must be in range." && v342);
                                    } else {
                                    }
                                    int v345;
                                    v345 = v167[v338];
                                    bool v348;
                                    if (v340){
                                        bool v347;
                                        v347 = v338 < 2;
                                        v348 = v347;
                                    } else {
                                        v348 = false;
                                    }
                                    bool v349;
                                    v349 = v348 == false;
                                    if (v349){
                                        assert("Index must be in range." && v348);
                                    } else {
                                    }
                                    int v351;
                                    v351 = v165[v338];
                                    int v353;
                                    v353 = v345 + v351;
                                    v336[v338] = v353;
                                    v338 += 1 ;
                                }
                                int v354;
                                v354 = v165[0];
                                int v356; int v357;
                                Tuple4 tmp16 = Tuple4{1, v354};
                                v356 = tmp16.v0; v357 = tmp16.v1;
                                while (while_method_2(v356)){
                                    bool v359;
                                    v359 = 0 <= v356;
                                    bool v361;
                                    if (v359){
                                        bool v360;
                                        v360 = v356 < 2;
                                        v361 = v360;
                                    } else {
                                        v361 = false;
                                    }
                                    bool v362;
                                    v362 = v361 == false;
                                    if (v362){
                                        assert("Index must be in range." && v361);
                                    } else {
                                    }
                                    int v364;
                                    v364 = v165[v356];
                                    bool v366;
                                    v366 = v357 >= v364;
                                    int v367;
                                    if (v366){
                                        v367 = v357;
                                    } else {
                                        v367 = v364;
                                    }
                                    v357 = v367;
                                    v356 += 1 ;
                                }
                                bool v369;
                                if (v171){
                                    bool v368;
                                    v368 = v169 < 2;
                                    v369 = v368;
                                } else {
                                    v369 = false;
                                }
                                bool v370;
                                v370 = v369 == false;
                                if (v370){
                                    assert("Index must be in range." && v369);
                                } else {
                                }
                                int v372;
                                v372 = v336[v169];
                                bool v374;
                                v374 = v357 < v372;
                                int v375;
                                if (v374){
                                    v375 = v357;
                                } else {
                                    v375 = v372;
                                }
                                static_array<int,2> v376;
                                int v378;
                                v378 = 0;
                                while (while_method_2(v378)){
                                    bool v380;
                                    v380 = 0 <= v378;
                                    bool v382;
                                    if (v380){
                                        bool v381;
                                        v381 = v378 < 2;
                                        v382 = v381;
                                    } else {
                                        v382 = false;
                                    }
                                    bool v383;
                                    v383 = v382 == false;
                                    if (v383){
                                        assert("Index must be in range." && v382);
                                    } else {
                                    }
                                    int v385;
                                    v385 = v165[v378];
                                    bool v387;
                                    v387 = v169 == v378;
                                    int v388;
                                    if (v387){
                                        v388 = v375;
                                    } else {
                                        v388 = v385;
                                    }
                                    v376[v378] = v388;
                                    v378 += 1 ;
                                }
                                int v389;
                                v389 = v376[0];
                                int v391; int v392;
                                Tuple4 tmp17 = Tuple4{1, v389};
                                v391 = tmp17.v0; v392 = tmp17.v1;
                                while (while_method_2(v391)){
                                    bool v394;
                                    v394 = 0 <= v391;
                                    bool v396;
                                    if (v394){
                                        bool v395;
                                        v395 = v391 < 2;
                                        v396 = v395;
                                    } else {
                                        v396 = false;
                                    }
                                    bool v397;
                                    v397 = v396 == false;
                                    if (v397){
                                        assert("Index must be in range." && v396);
                                    } else {
                                    }
                                    int v399;
                                    v399 = v376[v391];
                                    int v401;
                                    v401 = v392 + v399;
                                    v392 = v401;
                                    v391 += 1 ;
                                }
                                static_array<int,2> v402;
                                int v404;
                                v404 = 0;
                                while (while_method_2(v404)){
                                    bool v406;
                                    v406 = 0 <= v404;
                                    bool v408;
                                    if (v406){
                                        bool v407;
                                        v407 = v404 < 2;
                                        v408 = v407;
                                    } else {
                                        v408 = false;
                                    }
                                    bool v409;
                                    v409 = v408 == false;
                                    if (v409){
                                        assert("Index must be in range." && v408);
                                    } else {
                                    }
                                    int v411;
                                    v411 = v336[v404];
                                    bool v414;
                                    if (v406){
                                        bool v413;
                                        v413 = v404 < 2;
                                        v414 = v413;
                                    } else {
                                        v414 = false;
                                    }
                                    bool v415;
                                    v415 = v414 == false;
                                    if (v415){
                                        assert("Index must be in range." && v414);
                                    } else {
                                    }
                                    int v417;
                                    v417 = v376[v404];
                                    int v419;
                                    v419 = v411 - v417;
                                    v402[v404] = v419;
                                    v404 += 1 ;
                                }
                                bool v421;
                                if (v171){
                                    bool v420;
                                    v420 = v169 < 2;
                                    v421 = v420;
                                } else {
                                    v421 = false;
                                }
                                bool v422;
                                v422 = v421 == false;
                                if (v422){
                                    assert("Index must be in range." && v421);
                                } else {
                                }
                                int v424;
                                v424 = v165[v169];
                                bool v426;
                                v426 = v424 < v357;
                                float v427;
                                if (v426){
                                    v427 = 1.0f;
                                } else {
                                    v427 = 0.0f;
                                }
                                int v428;
                                v428 = v392 / 3;
                                bool v429;
                                v429 = v163 <= v428;
                                bool v437;
                                if (v429){
                                    bool v431;
                                    if (v171){
                                        bool v430;
                                        v430 = v169 < 2;
                                        v431 = v430;
                                    } else {
                                        v431 = false;
                                    }
                                    bool v432;
                                    v432 = v431 == false;
                                    if (v432){
                                        assert("Index must be in range." && v431);
                                    } else {
                                    }
                                    int v434;
                                    v434 = v402[v169];
                                    bool v436;
                                    v436 = v428 < v434;
                                    v437 = v436;
                                } else {
                                    v437 = false;
                                }
                                float v438;
                                if (v437){
                                    v438 = 1.0f;
                                } else {
                                    v438 = 0.0f;
                                }
                                int v439;
                                v439 = v392 / 2;
                                bool v440;
                                v440 = v163 <= v439;
                                bool v448;
                                if (v440){
                                    bool v442;
                                    if (v171){
                                        bool v441;
                                        v441 = v169 < 2;
                                        v442 = v441;
                                    } else {
                                        v442 = false;
                                    }
                                    bool v443;
                                    v443 = v442 == false;
                                    if (v443){
                                        assert("Index must be in range." && v442);
                                    } else {
                                    }
                                    int v445;
                                    v445 = v402[v169];
                                    bool v447;
                                    v447 = v439 < v445;
                                    v448 = v447;
                                } else {
                                    v448 = false;
                                }
                                float v449;
                                if (v448){
                                    v449 = 1.0f;
                                } else {
                                    v449 = 0.0f;
                                }
                                bool v450;
                                v450 = v163 <= v392;
                                bool v458;
                                if (v450){
                                    bool v452;
                                    if (v171){
                                        bool v451;
                                        v451 = v169 < 2;
                                        v452 = v451;
                                    } else {
                                        v452 = false;
                                    }
                                    bool v453;
                                    v453 = v452 == false;
                                    if (v453){
                                        assert("Index must be in range." && v452);
                                    } else {
                                    }
                                    int v455;
                                    v455 = v402[v169];
                                    bool v457;
                                    v457 = v392 < v455;
                                    v458 = v457;
                                } else {
                                    v458 = false;
                                }
                                float v459;
                                if (v458){
                                    v459 = 1.0f;
                                } else {
                                    v459 = 0.0f;
                                }
                                static_array<Tuple7,6> v460;
                                Union2 v462;
                                v462 = Union2{Union2_2{}};
                                v460[0] = Tuple7{v462, v427};
                                Union2 v464;
                                v464 = Union2{Union2_1{}};
                                v460[1] = Tuple7{v464, 4.0f};
                                Union2 v466;
                                v466 = Union2{Union2_3{v428}};
                                v460[2] = Tuple7{v466, v438};
                                Union2 v468;
                                v468 = Union2{Union2_3{v439}};
                                v460[3] = Tuple7{v468, v449};
                                Union2 v470;
                                v470 = Union2{Union2_3{v392}};
                                v460[4] = Tuple7{v470, v459};
                                Union2 v472;
                                v472 = Union2{Union2_0{}};
                                v460[5] = Tuple7{v472, 1.0f};
                                Union2 v474;
                                v474 = sample_discrete_11(v460, v335);
                                int v475;
                                v475 = sizeof(Union2);
                                unsigned long long v476;
                                v476 = (unsigned long long)v475;
                                bool v477;
                                v477 = v476 <= 98304ull;
                                bool v478;
                                v478 = v477 == false;
                                if (v478){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v477);
                                } else {
                                }
                                extern __shared__ unsigned char v480[];
                                bool v481;
                                v481 = v476 <= v476;
                                bool v482;
                                v482 = v481 == false;
                                if (v482){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v481);
                                } else {
                                }
                                Union2 * v484;
                                v484 = reinterpret_cast<Union2 *>(&v480[0ull]);
                                int v486;
                                v486 = threadIdx.x;
                                bool v487;
                                v487 = v486 == 0;
                                if (v487){
                                    v484[0] = v474;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                Union2 v488;
                                v488 = v484[0];
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                v491 = v488;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union1 v492;
                        v492 = Union1{Union1_2{v169, v491}};
                        v15.push(v492);
                        v643 = Union6{Union6_1{v163, v164, v165, v166, v167, v168, v491}};
                        break;
                    }
                    case 5: { // G_Round'
                        int v494 = v20.case5.v0; static_array<static_array<unsigned char,2>,2> v495 = v20.case5.v1; static_array<int,2> v496 = v20.case5.v2; int v497 = v20.case5.v3; static_array<int,2> v498 = v20.case5.v4; Union4 v499 = v20.case5.v5; Union2 v500 = v20.case5.v6;
                        int v501;
                        v501 = v497 % 2;
                        Union1 v502;
                        v502 = Union1{Union1_2{v501, v500}};
                        v15.push(v502);
                        v643 = Union6{Union6_1{v494, v495, v496, v497, v498, v499, v500}};
                        break;
                    }
                    case 6: { // G_Showdown
                        int v44 = v20.case6.v0; static_array<static_array<unsigned char,2>,2> v45 = v20.case6.v1; static_array<int,2> v46 = v20.case6.v2; int v47 = v20.case6.v3; static_array<int,2> v48 = v20.case6.v4; Union4 v49 = v20.case6.v5;
                        static_array<unsigned char,5> v52;
                        switch (v49.tag) {
                            case 2: { // River
                                static_array<unsigned char,5> v50 = v49.case2.v0;
                                v52 = v50;
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in showdown.");
                                __trap();
                            }
                        }
                        static_array<unsigned char,2> v53;
                        v53 = v45[0];
                        static_array<unsigned char,7> v55;
                        int v57;
                        v57 = 0;
                        while (while_method_2(v57)){
                            bool v59;
                            v59 = 0 <= v57;
                            bool v61;
                            if (v59){
                                bool v60;
                                v60 = v57 < 2;
                                v61 = v60;
                            } else {
                                v61 = false;
                            }
                            bool v62;
                            v62 = v61 == false;
                            if (v62){
                                assert("Index must be in range." && v61);
                            } else {
                            }
                            unsigned char v64;
                            v64 = v53[v57];
                            v55[v57] = v64;
                            v57 += 1 ;
                        }
                        int v66;
                        v66 = 0;
                        while (while_method_5(v66)){
                            bool v68;
                            v68 = 0 <= v66;
                            bool v70;
                            if (v68){
                                bool v69;
                                v69 = v66 < 5;
                                v70 = v69;
                            } else {
                                v70 = false;
                            }
                            bool v71;
                            v71 = v70 == false;
                            if (v71){
                                assert("Index must be in range." && v70);
                            } else {
                            }
                            unsigned char v73;
                            v73 = v52[v66];
                            int v75;
                            v75 = 2 + v66;
                            v55[v75] = v73;
                            v66 += 1 ;
                        }
                        static_array<unsigned char,5> v76; char v77;
                        Tuple0 tmp42 = score_15(v55);
                        v76 = tmp42.v0; v77 = tmp42.v1;
                        static_array<unsigned char,2> v78;
                        v78 = v45[1];
                        static_array<unsigned char,7> v80;
                        int v82;
                        v82 = 0;
                        while (while_method_2(v82)){
                            bool v84;
                            v84 = 0 <= v82;
                            bool v86;
                            if (v84){
                                bool v85;
                                v85 = v82 < 2;
                                v86 = v85;
                            } else {
                                v86 = false;
                            }
                            bool v87;
                            v87 = v86 == false;
                            if (v87){
                                assert("Index must be in range." && v86);
                            } else {
                            }
                            unsigned char v89;
                            v89 = v78[v82];
                            v80[v82] = v89;
                            v82 += 1 ;
                        }
                        int v91;
                        v91 = 0;
                        while (while_method_5(v91)){
                            bool v93;
                            v93 = 0 <= v91;
                            bool v95;
                            if (v93){
                                bool v94;
                                v94 = v91 < 5;
                                v95 = v94;
                            } else {
                                v95 = false;
                            }
                            bool v96;
                            v96 = v95 == false;
                            if (v96){
                                assert("Index must be in range." && v95);
                            } else {
                            }
                            unsigned char v98;
                            v98 = v52[v91];
                            int v100;
                            v100 = 2 + v91;
                            v80[v100] = v98;
                            v91 += 1 ;
                        }
                        static_array<unsigned char,5> v101; char v102;
                        Tuple0 tmp43 = score_15(v80);
                        v101 = tmp43.v0; v102 = tmp43.v1;
                        int v103;
                        v103 = v47 % 2;
                        bool v104;
                        v104 = 0 <= v103;
                        bool v106;
                        if (v104){
                            bool v105;
                            v105 = v103 < 2;
                            v106 = v105;
                        } else {
                            v106 = false;
                        }
                        bool v107;
                        v107 = v106 == false;
                        if (v107){
                            assert("Index must be in range." && v106);
                        } else {
                        }
                        int v109;
                        v109 = v46[v103];
                        bool v111;
                        v111 = v77 < v102;
                        Union7 v117;
                        if (v111){
                            v117 = Union7{Union7_2{}};
                        } else {
                            bool v113;
                            v113 = v77 > v102;
                            if (v113){
                                v117 = Union7{Union7_1{}};
                            } else {
                                v117 = Union7{Union7_0{}};
                            }
                        }
                        Union7 v145;
                        switch (v117.tag) {
                            case 0: { // Eq
                                Union7 v118;
                                v118 = Union7{Union7_0{}};
                                int v119;
                                v119 = 0;
                                while (while_method_5(v119)){
                                    bool v121;
                                    v121 = 0 <= v119;
                                    bool v123;
                                    if (v121){
                                        bool v122;
                                        v122 = v119 < 5;
                                        v123 = v122;
                                    } else {
                                        v123 = false;
                                    }
                                    bool v124;
                                    v124 = v123 == false;
                                    if (v124){
                                        assert("Index must be in range." && v123);
                                    } else {
                                    }
                                    unsigned char v126;
                                    v126 = v76[v119];
                                    bool v129;
                                    if (v121){
                                        bool v128;
                                        v128 = v119 < 5;
                                        v129 = v128;
                                    } else {
                                        v129 = false;
                                    }
                                    bool v130;
                                    v130 = v129 == false;
                                    if (v130){
                                        assert("Index must be in range." && v129);
                                    } else {
                                    }
                                    unsigned char v132;
                                    v132 = v101[v119];
                                    unsigned char v134;
                                    v134 = v126 / 4u;
                                    unsigned char v135;
                                    v135 = v132 / 4u;
                                    bool v136;
                                    v136 = v134 < v135;
                                    Union7 v142;
                                    if (v136){
                                        v142 = Union7{Union7_2{}};
                                    } else {
                                        bool v138;
                                        v138 = v134 > v135;
                                        if (v138){
                                            v142 = Union7{Union7_1{}};
                                        } else {
                                            v142 = Union7{Union7_0{}};
                                        }
                                    }
                                    bool v143;
                                    switch (v142.tag) {
                                        case 0: { // Eq
                                            v143 = true;
                                            break;
                                        }
                                        default: {
                                            v143 = false;
                                        }
                                    }
                                    bool v144;
                                    v144 = v143 == false;
                                    if (v144){
                                        v118 = v142;
                                        break;
                                    } else {
                                    }
                                    v119 += 1 ;
                                }
                                v145 = v118;
                                break;
                            }
                            default: {
                                v145 = v117;
                            }
                        }
                        int v150; int v151;
                        switch (v145.tag) {
                            case 0: { // Eq
                                v150 = 0; v151 = -1;
                                break;
                            }
                            case 1: { // Gt
                                v150 = v109; v151 = 0;
                                break;
                            }
                            case 2: { // Lt
                                v150 = v109; v151 = 1;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        int v152;
                        v152 = -v151;
                        bool v153;
                        v153 = v151 >= v152;
                        int v154;
                        if (v153){
                            v154 = v151;
                        } else {
                            v154 = v152;
                        }
                        float v155;
                        v155 = (float)v150;
                        static_array<float,2> & v156 = v0.v4;
                        v156[v154] = v155;
                        int v157;
                        v157 = v154 ^ 1;
                        float v158;
                        v158 = -v155;
                        v156[v157] = v158;
                        static_array<Tuple0,2> v159;
                        v159[0] = Tuple0{v76, v77};
                        v159[1] = Tuple0{v101, v102};
                        Union1 v161;
                        v161 = Union1{Union1_4{v150, v159, v151}};
                        v15.push(v161);
                        v643 = Union6{Union6_0{}};
                        break;
                    }
                    case 7: { // G_Turn
                        int v523 = v20.case7.v0; static_array<static_array<unsigned char,2>,2> v524 = v20.case7.v1; static_array<int,2> v525 = v20.case7.v2; int v526 = v20.case7.v3; static_array<int,2> v527 = v20.case7.v4; Union4 v528 = v20.case7.v5;
                        curandStatePhilox4_32_10_t & v529 = v0.v5;
                        curandStatePhilox4_32_10_t & v530 = v529;
                        static_array<unsigned char,1> v531; unsigned long long v532;
                        Tuple6 tmp44 = draw_cards_9(v530, v16);
                        v531 = tmp44.v0; v532 = tmp44.v1;
                        v0.v0 = v532;
                        static_array_list<unsigned char,5> v533;
                        v533 = get_community_cards_10(v528, v531);
                        Union1 v534;
                        v534 = Union1{Union1_0{v533}};
                        v15.push(v534);
                        Union4 v559;
                        switch (v528.tag) {
                            case 0: { // Flop
                                static_array<unsigned char,3> v535 = v528.case0.v0;
                                static_array<unsigned char,4> v536;
                                int v538;
                                v538 = 0;
                                while (while_method_4(v538)){
                                    bool v540;
                                    v540 = 0 <= v538;
                                    bool v542;
                                    if (v540){
                                        bool v541;
                                        v541 = v538 < 3;
                                        v542 = v541;
                                    } else {
                                        v542 = false;
                                    }
                                    bool v543;
                                    v543 = v542 == false;
                                    if (v543){
                                        assert("Index must be in range." && v542);
                                    } else {
                                    }
                                    unsigned char v545;
                                    v545 = v535[v538];
                                    v536[v538] = v545;
                                    v538 += 1 ;
                                }
                                int v547;
                                v547 = 0;
                                while (while_method_6(v547)){
                                    bool v549;
                                    v549 = 0 <= v547;
                                    bool v551;
                                    if (v549){
                                        bool v550;
                                        v550 = v547 < 1;
                                        v551 = v550;
                                    } else {
                                        v551 = false;
                                    }
                                    bool v552;
                                    v552 = v551 == false;
                                    if (v552){
                                        assert("Index must be in range." && v551);
                                    } else {
                                    }
                                    unsigned char v554;
                                    v554 = v531[v547];
                                    int v556;
                                    v556 = 3 + v547;
                                    v536[v556] = v554;
                                    v547 += 1 ;
                                }
                                v559 = Union4{Union4_3{v536}};
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in turn.");
                                __trap();
                            }
                        }
                        int v560;
                        v560 = 2;
                        int v561;
                        v561 = 0;
                        Union3 v562;
                        v562 = try_round_5(v560, v524, v525, v561, v527, v559);
                        v643 = Union6{Union6_2{v562}};
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                switch (v643.tag) {
                    case 0: { // T_none
                        v992 = Union5{Union5_0{}};
                        break;
                    }
                    case 1: { // T_round
                        int v647 = v643.case1.v0; static_array<static_array<unsigned char,2>,2> v648 = v643.case1.v1; static_array<int,2> v649 = v643.case1.v2; int v650 = v643.case1.v3; static_array<int,2> v651 = v643.case1.v4; Union4 v652 = v643.case1.v5; Union2 v653 = v643.case1.v6;
                        int v654;
                        v654 = v650 % 2;
                        Union3 v985;
                        switch (v653.tag) {
                            case 0: { // A_All_In
                                static_array<int,2> v860;
                                int v862;
                                v862 = 0;
                                while (while_method_2(v862)){
                                    bool v864;
                                    v864 = 0 <= v862;
                                    bool v866;
                                    if (v864){
                                        bool v865;
                                        v865 = v862 < 2;
                                        v866 = v865;
                                    } else {
                                        v866 = false;
                                    }
                                    bool v867;
                                    v867 = v866 == false;
                                    if (v867){
                                        assert("Index must be in range." && v866);
                                    } else {
                                    }
                                    int v869;
                                    v869 = v651[v862];
                                    bool v872;
                                    if (v864){
                                        bool v871;
                                        v871 = v862 < 2;
                                        v872 = v871;
                                    } else {
                                        v872 = false;
                                    }
                                    bool v873;
                                    v873 = v872 == false;
                                    if (v873){
                                        assert("Index must be in range." && v872);
                                    } else {
                                    }
                                    int v875;
                                    v875 = v649[v862];
                                    int v877;
                                    v877 = v869 + v875;
                                    v860[v862] = v877;
                                    v862 += 1 ;
                                }
                                int v878;
                                v878 = v649[0];
                                int v880; int v881;
                                Tuple4 tmp45 = Tuple4{1, v878};
                                v880 = tmp45.v0; v881 = tmp45.v1;
                                while (while_method_2(v880)){
                                    bool v883;
                                    v883 = 0 <= v880;
                                    bool v885;
                                    if (v883){
                                        bool v884;
                                        v884 = v880 < 2;
                                        v885 = v884;
                                    } else {
                                        v885 = false;
                                    }
                                    bool v886;
                                    v886 = v885 == false;
                                    if (v886){
                                        assert("Index must be in range." && v885);
                                    } else {
                                    }
                                    int v888;
                                    v888 = v649[v880];
                                    bool v890;
                                    v890 = v881 >= v888;
                                    int v891;
                                    if (v890){
                                        v891 = v881;
                                    } else {
                                        v891 = v888;
                                    }
                                    v881 = v891;
                                    v880 += 1 ;
                                }
                                bool v892;
                                v892 = 0 <= v654;
                                bool v894;
                                if (v892){
                                    bool v893;
                                    v893 = v654 < 2;
                                    v894 = v893;
                                } else {
                                    v894 = false;
                                }
                                bool v895;
                                v895 = v894 == false;
                                if (v895){
                                    assert("Index must be in range." && v894);
                                } else {
                                }
                                int v897;
                                v897 = v860[v654];
                                bool v899;
                                v899 = v881 < v897;
                                int v900;
                                if (v899){
                                    v900 = v881;
                                } else {
                                    v900 = v897;
                                }
                                static_array<int,2> v901;
                                int v903;
                                v903 = 0;
                                while (while_method_2(v903)){
                                    bool v905;
                                    v905 = 0 <= v903;
                                    bool v907;
                                    if (v905){
                                        bool v906;
                                        v906 = v903 < 2;
                                        v907 = v906;
                                    } else {
                                        v907 = false;
                                    }
                                    bool v908;
                                    v908 = v907 == false;
                                    if (v908){
                                        assert("Index must be in range." && v907);
                                    } else {
                                    }
                                    int v910;
                                    v910 = v649[v903];
                                    bool v912;
                                    v912 = v654 == v903;
                                    int v913;
                                    if (v912){
                                        v913 = v900;
                                    } else {
                                        v913 = v910;
                                    }
                                    v901[v903] = v913;
                                    v903 += 1 ;
                                }
                                static_array<int,2> v914;
                                int v916;
                                v916 = 0;
                                while (while_method_2(v916)){
                                    bool v918;
                                    v918 = 0 <= v916;
                                    bool v920;
                                    if (v918){
                                        bool v919;
                                        v919 = v916 < 2;
                                        v920 = v919;
                                    } else {
                                        v920 = false;
                                    }
                                    bool v921;
                                    v921 = v920 == false;
                                    if (v921){
                                        assert("Index must be in range." && v920);
                                    } else {
                                    }
                                    int v923;
                                    v923 = v860[v916];
                                    bool v926;
                                    if (v918){
                                        bool v925;
                                        v925 = v916 < 2;
                                        v926 = v925;
                                    } else {
                                        v926 = false;
                                    }
                                    bool v927;
                                    v927 = v926 == false;
                                    if (v927){
                                        assert("Index must be in range." && v926);
                                    } else {
                                    }
                                    int v929;
                                    v929 = v901[v916];
                                    int v931;
                                    v931 = v923 - v929;
                                    v914[v916] = v931;
                                    v916 += 1 ;
                                }
                                bool v933;
                                if (v892){
                                    bool v932;
                                    v932 = v654 < 2;
                                    v933 = v932;
                                } else {
                                    v933 = false;
                                }
                                bool v934;
                                v934 = v933 == false;
                                if (v934){
                                    assert("Index must be in range." && v933);
                                } else {
                                }
                                int v936;
                                v936 = v914[v654];
                                int v938;
                                v938 = v881 + v936;
                                bool v940;
                                if (v892){
                                    bool v939;
                                    v939 = v654 < 2;
                                    v940 = v939;
                                } else {
                                    v940 = false;
                                }
                                bool v941;
                                v941 = v940 == false;
                                if (v941){
                                    assert("Index must be in range." && v940);
                                } else {
                                }
                                int v943;
                                v943 = v860[v654];
                                bool v945;
                                v945 = v938 < v943;
                                int v946;
                                if (v945){
                                    v946 = v938;
                                } else {
                                    v946 = v943;
                                }
                                static_array<int,2> v947;
                                int v949;
                                v949 = 0;
                                while (while_method_2(v949)){
                                    bool v951;
                                    v951 = 0 <= v949;
                                    bool v953;
                                    if (v951){
                                        bool v952;
                                        v952 = v949 < 2;
                                        v953 = v952;
                                    } else {
                                        v953 = false;
                                    }
                                    bool v954;
                                    v954 = v953 == false;
                                    if (v954){
                                        assert("Index must be in range." && v953);
                                    } else {
                                    }
                                    int v956;
                                    v956 = v649[v949];
                                    bool v958;
                                    v958 = v654 == v949;
                                    int v959;
                                    if (v958){
                                        v959 = v946;
                                    } else {
                                        v959 = v956;
                                    }
                                    v947[v949] = v959;
                                    v949 += 1 ;
                                }
                                static_array<int,2> v960;
                                int v962;
                                v962 = 0;
                                while (while_method_2(v962)){
                                    bool v964;
                                    v964 = 0 <= v962;
                                    bool v966;
                                    if (v964){
                                        bool v965;
                                        v965 = v962 < 2;
                                        v966 = v965;
                                    } else {
                                        v966 = false;
                                    }
                                    bool v967;
                                    v967 = v966 == false;
                                    if (v967){
                                        assert("Index must be in range." && v966);
                                    } else {
                                    }
                                    int v969;
                                    v969 = v860[v962];
                                    bool v972;
                                    if (v964){
                                        bool v971;
                                        v971 = v962 < 2;
                                        v972 = v971;
                                    } else {
                                        v972 = false;
                                    }
                                    bool v973;
                                    v973 = v972 == false;
                                    if (v973){
                                        assert("Index must be in range." && v972);
                                    } else {
                                    }
                                    int v975;
                                    v975 = v947[v962];
                                    int v977;
                                    v977 = v969 - v975;
                                    v960[v962] = v977;
                                    v962 += 1 ;
                                }
                                bool v978;
                                v978 = v936 >= v647;
                                int v979;
                                if (v978){
                                    v979 = v936;
                                } else {
                                    v979 = v647;
                                }
                                int v980;
                                v980 = v650 + 1;
                                v985 = try_round_5(v979, v648, v947, v980, v960, v652);
                                break;
                            }
                            case 1: { // A_Call
                                static_array<int,2> v656;
                                int v658;
                                v658 = 0;
                                while (while_method_2(v658)){
                                    bool v660;
                                    v660 = 0 <= v658;
                                    bool v662;
                                    if (v660){
                                        bool v661;
                                        v661 = v658 < 2;
                                        v662 = v661;
                                    } else {
                                        v662 = false;
                                    }
                                    bool v663;
                                    v663 = v662 == false;
                                    if (v663){
                                        assert("Index must be in range." && v662);
                                    } else {
                                    }
                                    int v665;
                                    v665 = v651[v658];
                                    bool v668;
                                    if (v660){
                                        bool v667;
                                        v667 = v658 < 2;
                                        v668 = v667;
                                    } else {
                                        v668 = false;
                                    }
                                    bool v669;
                                    v669 = v668 == false;
                                    if (v669){
                                        assert("Index must be in range." && v668);
                                    } else {
                                    }
                                    int v671;
                                    v671 = v649[v658];
                                    int v673;
                                    v673 = v665 + v671;
                                    v656[v658] = v673;
                                    v658 += 1 ;
                                }
                                int v674;
                                v674 = v649[0];
                                int v676; int v677;
                                Tuple4 tmp46 = Tuple4{1, v674};
                                v676 = tmp46.v0; v677 = tmp46.v1;
                                while (while_method_2(v676)){
                                    bool v679;
                                    v679 = 0 <= v676;
                                    bool v681;
                                    if (v679){
                                        bool v680;
                                        v680 = v676 < 2;
                                        v681 = v680;
                                    } else {
                                        v681 = false;
                                    }
                                    bool v682;
                                    v682 = v681 == false;
                                    if (v682){
                                        assert("Index must be in range." && v681);
                                    } else {
                                    }
                                    int v684;
                                    v684 = v649[v676];
                                    bool v686;
                                    v686 = v677 >= v684;
                                    int v687;
                                    if (v686){
                                        v687 = v677;
                                    } else {
                                        v687 = v684;
                                    }
                                    v677 = v687;
                                    v676 += 1 ;
                                }
                                bool v688;
                                v688 = 0 <= v654;
                                bool v690;
                                if (v688){
                                    bool v689;
                                    v689 = v654 < 2;
                                    v690 = v689;
                                } else {
                                    v690 = false;
                                }
                                bool v691;
                                v691 = v690 == false;
                                if (v691){
                                    assert("Index must be in range." && v690);
                                } else {
                                }
                                int v693;
                                v693 = v656[v654];
                                bool v695;
                                v695 = v677 < v693;
                                int v696;
                                if (v695){
                                    v696 = v677;
                                } else {
                                    v696 = v693;
                                }
                                static_array<int,2> v697;
                                int v699;
                                v699 = 0;
                                while (while_method_2(v699)){
                                    bool v701;
                                    v701 = 0 <= v699;
                                    bool v703;
                                    if (v701){
                                        bool v702;
                                        v702 = v699 < 2;
                                        v703 = v702;
                                    } else {
                                        v703 = false;
                                    }
                                    bool v704;
                                    v704 = v703 == false;
                                    if (v704){
                                        assert("Index must be in range." && v703);
                                    } else {
                                    }
                                    int v706;
                                    v706 = v649[v699];
                                    bool v708;
                                    v708 = v654 == v699;
                                    int v709;
                                    if (v708){
                                        v709 = v696;
                                    } else {
                                        v709 = v706;
                                    }
                                    v697[v699] = v709;
                                    v699 += 1 ;
                                }
                                static_array<int,2> v710;
                                int v712;
                                v712 = 0;
                                while (while_method_2(v712)){
                                    bool v714;
                                    v714 = 0 <= v712;
                                    bool v716;
                                    if (v714){
                                        bool v715;
                                        v715 = v712 < 2;
                                        v716 = v715;
                                    } else {
                                        v716 = false;
                                    }
                                    bool v717;
                                    v717 = v716 == false;
                                    if (v717){
                                        assert("Index must be in range." && v716);
                                    } else {
                                    }
                                    int v719;
                                    v719 = v656[v712];
                                    bool v722;
                                    if (v714){
                                        bool v721;
                                        v721 = v712 < 2;
                                        v722 = v721;
                                    } else {
                                        v722 = false;
                                    }
                                    bool v723;
                                    v723 = v722 == false;
                                    if (v723){
                                        assert("Index must be in range." && v722);
                                    } else {
                                    }
                                    int v725;
                                    v725 = v697[v712];
                                    int v727;
                                    v727 = v719 - v725;
                                    v710[v712] = v727;
                                    v712 += 1 ;
                                }
                                bool v728;
                                v728 = v654 < 2;
                                if (v728){
                                    int v729;
                                    v729 = v650 + 1;
                                    v985 = try_round_5(v647, v648, v697, v729, v710, v652);
                                } else {
                                    v985 = go_next_street_7(v647, v648, v697, v650, v710, v652);
                                }
                                break;
                            }
                            case 2: { // A_Fold
                                v985 = Union3{Union3_1{v647, v648, v649, v650, v651, v652}};
                                break;
                            }
                            case 3: { // A_Raise
                                int v733 = v653.case3.v0;
                                bool v734;
                                v734 = v647 <= v733;
                                bool v735;
                                v735 = v734 == false;
                                if (v735){
                                    assert("The raise amount must match the minimum." && v734);
                                } else {
                                }
                                static_array<int,2> v737;
                                int v739;
                                v739 = 0;
                                while (while_method_2(v739)){
                                    bool v741;
                                    v741 = 0 <= v739;
                                    bool v743;
                                    if (v741){
                                        bool v742;
                                        v742 = v739 < 2;
                                        v743 = v742;
                                    } else {
                                        v743 = false;
                                    }
                                    bool v744;
                                    v744 = v743 == false;
                                    if (v744){
                                        assert("Index must be in range." && v743);
                                    } else {
                                    }
                                    int v746;
                                    v746 = v651[v739];
                                    bool v749;
                                    if (v741){
                                        bool v748;
                                        v748 = v739 < 2;
                                        v749 = v748;
                                    } else {
                                        v749 = false;
                                    }
                                    bool v750;
                                    v750 = v749 == false;
                                    if (v750){
                                        assert("Index must be in range." && v749);
                                    } else {
                                    }
                                    int v752;
                                    v752 = v649[v739];
                                    int v754;
                                    v754 = v746 + v752;
                                    v737[v739] = v754;
                                    v739 += 1 ;
                                }
                                int v755;
                                v755 = v649[0];
                                int v757; int v758;
                                Tuple4 tmp47 = Tuple4{1, v755};
                                v757 = tmp47.v0; v758 = tmp47.v1;
                                while (while_method_2(v757)){
                                    bool v760;
                                    v760 = 0 <= v757;
                                    bool v762;
                                    if (v760){
                                        bool v761;
                                        v761 = v757 < 2;
                                        v762 = v761;
                                    } else {
                                        v762 = false;
                                    }
                                    bool v763;
                                    v763 = v762 == false;
                                    if (v763){
                                        assert("Index must be in range." && v762);
                                    } else {
                                    }
                                    int v765;
                                    v765 = v649[v757];
                                    bool v767;
                                    v767 = v758 >= v765;
                                    int v768;
                                    if (v767){
                                        v768 = v758;
                                    } else {
                                        v768 = v765;
                                    }
                                    v758 = v768;
                                    v757 += 1 ;
                                }
                                bool v769;
                                v769 = 0 <= v654;
                                bool v771;
                                if (v769){
                                    bool v770;
                                    v770 = v654 < 2;
                                    v771 = v770;
                                } else {
                                    v771 = false;
                                }
                                bool v772;
                                v772 = v771 == false;
                                if (v772){
                                    assert("Index must be in range." && v771);
                                } else {
                                }
                                int v774;
                                v774 = v737[v654];
                                bool v776;
                                v776 = v758 < v774;
                                int v777;
                                if (v776){
                                    v777 = v758;
                                } else {
                                    v777 = v774;
                                }
                                static_array<int,2> v778;
                                int v780;
                                v780 = 0;
                                while (while_method_2(v780)){
                                    bool v782;
                                    v782 = 0 <= v780;
                                    bool v784;
                                    if (v782){
                                        bool v783;
                                        v783 = v780 < 2;
                                        v784 = v783;
                                    } else {
                                        v784 = false;
                                    }
                                    bool v785;
                                    v785 = v784 == false;
                                    if (v785){
                                        assert("Index must be in range." && v784);
                                    } else {
                                    }
                                    int v787;
                                    v787 = v649[v780];
                                    bool v789;
                                    v789 = v654 == v780;
                                    int v790;
                                    if (v789){
                                        v790 = v777;
                                    } else {
                                        v790 = v787;
                                    }
                                    v778[v780] = v790;
                                    v780 += 1 ;
                                }
                                static_array<int,2> v791;
                                int v793;
                                v793 = 0;
                                while (while_method_2(v793)){
                                    bool v795;
                                    v795 = 0 <= v793;
                                    bool v797;
                                    if (v795){
                                        bool v796;
                                        v796 = v793 < 2;
                                        v797 = v796;
                                    } else {
                                        v797 = false;
                                    }
                                    bool v798;
                                    v798 = v797 == false;
                                    if (v798){
                                        assert("Index must be in range." && v797);
                                    } else {
                                    }
                                    int v800;
                                    v800 = v737[v793];
                                    bool v803;
                                    if (v795){
                                        bool v802;
                                        v802 = v793 < 2;
                                        v803 = v802;
                                    } else {
                                        v803 = false;
                                    }
                                    bool v804;
                                    v804 = v803 == false;
                                    if (v804){
                                        assert("Index must be in range." && v803);
                                    } else {
                                    }
                                    int v806;
                                    v806 = v778[v793];
                                    int v808;
                                    v808 = v800 - v806;
                                    v791[v793] = v808;
                                    v793 += 1 ;
                                }
                                bool v810;
                                if (v769){
                                    bool v809;
                                    v809 = v654 < 2;
                                    v810 = v809;
                                } else {
                                    v810 = false;
                                }
                                bool v811;
                                v811 = v810 == false;
                                if (v811){
                                    assert("Index must be in range." && v810);
                                } else {
                                }
                                int v813;
                                v813 = v791[v654];
                                bool v815;
                                v815 = v733 < v813;
                                bool v816;
                                v816 = v815 == false;
                                if (v816){
                                    assert("The raise amount must be less than the stack size after calling." && v815);
                                } else {
                                }
                                int v818;
                                v818 = v758 + v733;
                                bool v820;
                                if (v769){
                                    bool v819;
                                    v819 = v654 < 2;
                                    v820 = v819;
                                } else {
                                    v820 = false;
                                }
                                bool v821;
                                v821 = v820 == false;
                                if (v821){
                                    assert("Index must be in range." && v820);
                                } else {
                                }
                                int v823;
                                v823 = v737[v654];
                                bool v825;
                                v825 = v818 < v823;
                                int v826;
                                if (v825){
                                    v826 = v818;
                                } else {
                                    v826 = v823;
                                }
                                static_array<int,2> v827;
                                int v829;
                                v829 = 0;
                                while (while_method_2(v829)){
                                    bool v831;
                                    v831 = 0 <= v829;
                                    bool v833;
                                    if (v831){
                                        bool v832;
                                        v832 = v829 < 2;
                                        v833 = v832;
                                    } else {
                                        v833 = false;
                                    }
                                    bool v834;
                                    v834 = v833 == false;
                                    if (v834){
                                        assert("Index must be in range." && v833);
                                    } else {
                                    }
                                    int v836;
                                    v836 = v649[v829];
                                    bool v838;
                                    v838 = v654 == v829;
                                    int v839;
                                    if (v838){
                                        v839 = v826;
                                    } else {
                                        v839 = v836;
                                    }
                                    v827[v829] = v839;
                                    v829 += 1 ;
                                }
                                static_array<int,2> v840;
                                int v842;
                                v842 = 0;
                                while (while_method_2(v842)){
                                    bool v844;
                                    v844 = 0 <= v842;
                                    bool v846;
                                    if (v844){
                                        bool v845;
                                        v845 = v842 < 2;
                                        v846 = v845;
                                    } else {
                                        v846 = false;
                                    }
                                    bool v847;
                                    v847 = v846 == false;
                                    if (v847){
                                        assert("Index must be in range." && v846);
                                    } else {
                                    }
                                    int v849;
                                    v849 = v737[v842];
                                    bool v852;
                                    if (v844){
                                        bool v851;
                                        v851 = v842 < 2;
                                        v852 = v851;
                                    } else {
                                        v852 = false;
                                    }
                                    bool v853;
                                    v853 = v852 == false;
                                    if (v853){
                                        assert("Index must be in range." && v852);
                                    } else {
                                    }
                                    int v855;
                                    v855 = v827[v842];
                                    int v857;
                                    v857 = v849 - v855;
                                    v840[v842] = v857;
                                    v842 += 1 ;
                                }
                                int v858;
                                v858 = v650 + 1;
                                v985 = try_round_5(v733, v648, v827, v858, v840, v652);
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        v992 = Union5{Union5_1{v985}};
                        break;
                    }
                    case 2: { // T_some
                        Union3 v645 = v643.case2.v0;
                        v992 = Union5{Union5_1{v645}};
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                break;
            }
            default: {
                assert("Invalid tag." && false); __trap();
            }
        }
        v18 = v992;
    }
    return ;
}
__device__ inline bool while_method_14(int v0){
    bool v1;
    v1 = v0 > 0;
    return v1;
}
__device__ int int_range_16(int v0, int v1, curandStatePhilox4_32_10_t & v2){
    int v3;
    v3 = v0 - v1;
    unsigned int v4;
    v4 = (unsigned int)v3;
    unsigned int v5;
    v5 = loop_3(v4, v2);
    unsigned int v6;
    v6 = (unsigned int)v1;
    unsigned int v7;
    v7 = v5 + v6;
    int v8;
    v8 = (int)v7;
    return v8;
}
__device__ inline bool while_method_15(int v0){
    bool v1;
    v1 = v0 < 256;
    return v1;
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1) {
    auto v2 = cooperative_groups::this_grid();
    unsigned long long v3;
    v3 = clock64();
    int v4;
    v4 = threadIdx.x;
    int v5;
    v5 = blockIdx.x;
    int v6;
    v6 = v5 * 256;
    int v7;
    v7 = v4 + v6;
    unsigned long long v8;
    v8 = (unsigned long long)v7;
    curandStatePhilox4_32_10_t v9;
    curand_init(v3,v8,0ull,&v9);
    static_array<Union0,2> v10;
    Union0 v12;
    v12 = Union0{Union0_2{}};
    v10[0] = v12;
    Union0 v14;
    v14 = Union0{Union0_2{}};
    v10[1] = v14;
    static_array_list<Union1,128> v16;
    v16 = static_array_list<Union1,128>{};
    static_array<float,2> v18;
    v18[0] = 0.0f;
    v18[1] = 0.0f;
    cooperative_groups::grid_group & v20 = v2;
    curandStatePhilox4_32_10_t & v21 = v9;
    StackMut0 v22{4503599627370495ull, v20, v16, v10, v18, v21};
    int v23;
    v23 = 0;
    while (while_method_0(v23)){
        int v25;
        v25 = 0;
        while (while_method_1(v25)){
            int v27;
            v27 = 0;
            while (while_method_2(v27)){
                Union3 v29;
                v29 = Union3{Union3_2{}};
                method_0(v22, v27, v29);
                static_array<float,2> & v30 = v22.v4;
                bool v31;
                v31 = 0 <= v27;
                bool v33;
                if (v31){
                    bool v32;
                    v32 = v27 < 2;
                    v33 = v32;
                } else {
                    v33 = false;
                }
                bool v34;
                v34 = v33 == false;
                if (v34){
                    assert("Index must be in range." && v33);
                } else {
                }
                float v36;
                v36 = v30[v27];
                static_array<float,2> & v38 = v22.v4;
                unsigned int * v39;
                v39 = reinterpret_cast<unsigned int *>(&v0[53477376ull]);
                int * v41;
                v41 = reinterpret_cast<int *>(&v1[4194304ull]);
                float * v43;
                v43 = reinterpret_cast<float *>(&v1[4194320ull]);
                float * v45;
                v45 = reinterpret_cast<float *>(&v1[5242896ull]);
                float * v47;
                v47 = reinterpret_cast<float *>(&v1[6291472ull]);
                float * v49;
                v49 = reinterpret_cast<float *>(&v1[7340048ull]);
                float * v51;
                v51 = reinterpret_cast<float *>(&v1[8388624ull]);
                float * v53;
                v53 = reinterpret_cast<float *>(&v1[9437200ull]);
                float * v55;
                v55 = reinterpret_cast<float *>(&v1[10485776ull]);
                int * v57;
                v57 = reinterpret_cast<int *>(&v0[53575680ull]);
                float * v59;
                v59 = reinterpret_cast<float *>(&v0[66158592ull]);
                int * v61;
                v61 = reinterpret_cast<int *>(&v0[78741504ull]);
                int * v63;
                v63 = reinterpret_cast<int *>(&v0[91324416ull]);
                double * v65;
                v65 = reinterpret_cast<double *>(&v0[103907328ull]);
                double * v67;
                v67 = reinterpret_cast<double *>(&v0[154238976ull]);
                double * v69;
                v69 = reinterpret_cast<double *>(&v1[11534352ull]);
                double * v71;
                v71 = reinterpret_cast<double *>(&v1[11927568ull]);
                int * v73;
                v73 = reinterpret_cast<int *>(&v1[12320784ull]);
                int v75;
                v75 = 0;
                while (while_method_0(v75)){
                    int v77;
                    v77 = threadIdx.x;
                    int v78;
                    v78 = blockIdx.x;
                    int v79;
                    v79 = v78 * 256;
                    int v80;
                    v80 = v77 + v79;
                    float v81[2];
                    int v82;
                    v82 = 0;
                    while (while_method_2(v82)){
                        bool v84;
                        v84 = 0 <= v82;
                        bool v86;
                        if (v84){
                            bool v85;
                            v85 = v82 < 2;
                            v86 = v85;
                        } else {
                            v86 = false;
                        }
                        bool v87;
                        v87 = v86 == false;
                        if (v87){
                            assert("Index must be in range." && v86);
                        } else {
                        }
                        float v89;
                        v89 = v38[v82];
                        v81[v82] = v89;
                        v82 += 1 ;
                    }
                    assert("Tensor range check" && 0 <= v75 && v75 < 4);
                    assert("Tensor range check" && 0 <= v80 && v80 < 6144);
                    int v91;
                    v91 = 6144 * v75;
                    int v92;
                    v92 = v91 + v80;
                    int v93;
                    v93 = v73[v92];
                    int v94;
                    v94 = v93;
                    while (while_method_14(v94)){
                        v94 -= 1 ;
                        assert("Tensor range check" && 0 <= v75 && v75 < 4);
                        assert("Tensor range check" && 0 <= v94 && v94 < 128);
                        assert("Tensor range check" && 0 <= v80 && v80 < 6144);
                        int v96;
                        v96 = 6144 * v94;
                        int v97;
                        v97 = v96 + v80;
                        int v98;
                        v98 = 786432 * v75;
                        int v99;
                        v99 = v98 + v97;
                        int v100;
                        v100 = v57[v99];
                        float v101;
                        v101 = v59[v99];
                        int v102;
                        v102 = v61[v99];
                        int v103;
                        v103 = v63[v99];
                        assert("Tensor range check" && 0 <= v102 && v102 < 2);
                        float v104;
                        v104 = v81[v102];
                        assert("Tensor range check" && 0 <= v75 && v75 < 4);
                        int v105;
                        v105 = 65536 * v75;
                        assert("Tensor range check" && 0 <= v103 && v103 < 4096);
                        int v106;
                        v106 = 16 * v103;
                        int v107;
                        v107 = v106 + v105;
                        float * v108;
                        v108 = v43+v107;
                        float * v110;
                        v110 = v45+v107;
                        float * v112;
                        v112 = v47+v107;
                        float * v114;
                        v114 = v49+v107;
                        float * v116;
                        v116 = v51+v107;
                        float * v118;
                        v118 = v53+v107;
                        float * v120;
                        v120 = v55+v107;
                        assert("Tensor range check" && 0 <= v75 && v75 < 4);
                        int v122;
                        v122 = 1572864 * v75;
                        assert("Tensor range check" && 0 <= v94 && v94 < 128);
                        int v123;
                        v123 = 12288 * v94;
                        int v124;
                        v124 = v123 + v122;
                        assert("Tensor range check" && 0 <= v80 && v80 < 6144);
                        int v125;
                        v125 = 2 * v80;
                        int v126;
                        v126 = v125 + v124;
                        double v127[2];
                        int v128;
                        v128 = 0;
                        while (while_method_2(v128)){
                            assert("Tensor range check" && 0 <= v128 && v128 < 2);
                            int v130;
                            v130 = v128 + v126;
                            double v131;
                            v131 = v65[v130];
                            bool v132;
                            v132 = v102 == v128;
                            double v133;
                            if (v132){
                                v133 = 0.0;
                            } else {
                                v133 = v131;
                            }
                            assert("Tensor range check" && 0 <= v128 && v128 < 2);
                            v127[v128] = v133;
                            v128 += 1 ;
                        }
                        double v134;
                        v134 = 0.0;
                        int v135;
                        v135 = 0;
                        while (while_method_2(v135)){
                            assert("Tensor range check" && 0 <= v135 && v135 < 2);
                            double v137;
                            v137 = v127[v135];
                            double v138;
                            v138 = v134 + v137;
                            v134 = v138;
                            v135 += 1 ;
                        }
                        double v139;
                        v139 = 0.0;
                        int v140;
                        v140 = 0;
                        while (while_method_2(v140)){
                            assert("Tensor range check" && 0 <= v140 && v140 < 2);
                            int v142;
                            v142 = v140 + v126;
                            double v143;
                            v143 = v67[v142];
                            double v144;
                            v144 = v139 + v143;
                            v139 = v144;
                            v140 += 1 ;
                        }
                        double v145;
                        v145 = v134 - v139;
                        double v146;
                        v146 = exp(v145);
                        float v147;
                        v147 = (float)v146;
                        float v148;
                        v148 = v104 * v147;
                        assert("Tensor range check" && 0 <= v100 && v100 < 16);
                        float * v149;
                        v149 = v118+v100;
                        float * v151;
                        v151 = v120+v100;
                        float v153;
                        v153 = atomicAdd(v149,v148);
                        float v154;
                        v154 = atomicAdd(v151,v147);
                        float * v155;
                        v155 = v110+0;
                        float * v157;
                        v157 = v114+0;
                        float * v159;
                        v159 = v116+0;
                        int v161;
                        v161 = sizeof(float *);
                        unsigned long long v162;
                        v162 = (unsigned long long)v161;
                        unsigned long long v163;
                        v163 = 256ull * v162;
                        unsigned long long v164;
                        v164 = 4096ull + v163;
                        unsigned long long v165;
                        v165 = v164 + 16ull;
                        unsigned long long v166;
                        v166 = v165 - 1ull;
                        unsigned long long v167;
                        v167 = v166 % 16ull;
                        unsigned long long v168;
                        v168 = v166 - v167;
                        unsigned long long v169;
                        v169 = v168 + v163;
                        unsigned long long v170;
                        v170 = v169 + 16ull;
                        unsigned long long v171;
                        v171 = v170 - 1ull;
                        unsigned long long v172;
                        v172 = v171 % 16ull;
                        unsigned long long v173;
                        v173 = v171 - v172;
                        unsigned long long v174;
                        v174 = v173 + v163;
                        unsigned long long v175;
                        v175 = v174 + 16ull;
                        unsigned long long v176;
                        v176 = v175 - 1ull;
                        unsigned long long v177;
                        v177 = v176 % 16ull;
                        unsigned long long v178;
                        v178 = v176 - v177;
                        unsigned long long v179;
                        v179 = v178 + v163;
                        unsigned long long v180;
                        v180 = v179 + 16ull;
                        unsigned long long v181;
                        v181 = v180 - 1ull;
                        unsigned long long v182;
                        v182 = v181 % 16ull;
                        unsigned long long v183;
                        v183 = v181 - v182;
                        unsigned long long v184;
                        v184 = v183 + 1024ull;
                        bool v185;
                        v185 = v184 <= 98304ull;
                        bool v186;
                        v186 = v185 == false;
                        if (v186){
                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v185);
                        } else {
                        }
                        extern __shared__ unsigned char v188[];
                        bool v189;
                        v189 = v184 <= v184;
                        bool v190;
                        v190 = v189 == false;
                        if (v190){
                            assert("The length of the partition has to be less than or equal to the length of the base array." && v189);
                        } else {
                        }
                        float * v192;
                        v192 = reinterpret_cast<float *>(&v188[0ull]);
                        int * v194;
                        v194 = reinterpret_cast<int *>(&v188[1024ull]);
                        float * v196;
                        v196 = reinterpret_cast<float *>(&v188[2048ull]);
                        float * v198;
                        v198 = reinterpret_cast<float *>(&v188[3072ull]);
                        float * * v200;
                        v200 = reinterpret_cast<float * *>(&v188[4096ull]);
                        float * * v202;
                        v202 = reinterpret_cast<float * *>(&v188[v168]);
                        float * * v204;
                        v204 = reinterpret_cast<float * *>(&v188[v173]);
                        float * * v206;
                        v206 = reinterpret_cast<float * *>(&v188[v178]);
                        float * v208;
                        v208 = reinterpret_cast<float *>(&v188[v183]);
                        int v210;
                        v210 = threadIdx.x;
                        assert("Tensor range check" && 0 <= v210 && v210 < 256);
                        v192[v210] = v101;
                        v194[v210] = v100;
                        v196[v210] = v104;
                        v198[v210] = v147;
                        v200[v210] = v112;
                        v202[v210] = v155;
                        v204[v210] = v157;
                        v206[v210] = v159;
                        asm("barrier.cta.sync %0;" :: "r"(0));
                        bool v211;
                        v211 = 0 <= v210;
                        bool v212;
                        v212 = v211 == false;
                        if (v212){
                            assert("The index needs to be zero or positive." && v211);
                        } else {
                        }
                        int v214;
                        v214 = v210 % 4;
                        int v215;
                        v215 = v210 / 4;
                        bool v216;
                        v216 = v215 < 64;
                        bool v217;
                        v217 = v216 == false;
                        if (v217){
                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v216);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v215 && v215 < 64);
                        int v219;
                        v219 = 0;
                        while (while_method_0(v219)){
                            bool v221;
                            v221 = 0 <= v215;
                            bool v222;
                            v222 = v221 && v216;
                            bool v223;
                            v223 = v222 == false;
                            if (v223){
                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v222);
                            } else {
                            }
                            bool v225;
                            v225 = 0 <= v219;
                            bool v227;
                            if (v225){
                                bool v226;
                                v226 = v219 < 4;
                                v227 = v226;
                            } else {
                                v227 = false;
                            }
                            bool v228;
                            v228 = v227 == false;
                            if (v228){
                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v227);
                            } else {
                            }
                            int v230;
                            v230 = v219 * 64;
                            int v231;
                            v231 = v230 + v215;
                            assert("Tensor range check" && 0 <= v219 && v219 < 4);
                            int v232;
                            v232 = 64 * v219;
                            int v233;
                            v233 = v232 + v215;
                            float v234;
                            v234 = v192[v233];
                            int v235;
                            v235 = v194[v233];
                            float v236;
                            v236 = v196[v233];
                            float v237;
                            v237 = v198[v233];
                            float * v238;
                            v238 = v200[v233];
                            float * v239;
                            v239 = v202[v233];
                            float * v240;
                            v240 = v204[v233];
                            float * v241;
                            v241 = v206[v233];
                            int v242;
                            v242 = blockIdx.x;
                            int v243;
                            v243 = v242 * 256;
                            int v244;
                            v244 = v243 + v231;
                            assert("Tensor range check" && 0 <= v214 && v214 < 4);
                            int v245;
                            v245 = 4 * v214;
                            float v246[4];
                            float v247[4];
                            float v248[4];
                            int v249[4];
                            int v250;
                            v250 = 0;
                            while (while_method_6(v250)){
                                assert("Tensor range check" && 0 <= v250 && v250 < 1);
                                int v252;
                                v252 = 4 * v250;
                                assert("Tensor range check" && 0 <= v250 && v250 < 1);
                                int v253;
                                v253 = 16 * v250;
                                int v254;
                                v254 = v253 + v245;
                                int4* v255;
                                v255 = reinterpret_cast<int4*>(v239 + v254);
                                int4* v256;
                                v256 = reinterpret_cast<int4*>(v246 + v252);
                                assert("Pointer alignment check" && (unsigned long long)(v255) % 4 == 0 && (unsigned long long)(v256) % 4 == 0);
                                *v256 = *v255;
                                int4* v257;
                                v257 = reinterpret_cast<int4*>(v240 + v254);
                                int4* v258;
                                v258 = reinterpret_cast<int4*>(v247 + v252);
                                assert("Pointer alignment check" && (unsigned long long)(v257) % 4 == 0 && (unsigned long long)(v258) % 4 == 0);
                                *v258 = *v257;
                                int4* v259;
                                v259 = reinterpret_cast<int4*>(v241 + v254);
                                int4* v260;
                                v260 = reinterpret_cast<int4*>(v248 + v252);
                                assert("Pointer alignment check" && (unsigned long long)(v259) % 4 == 0 && (unsigned long long)(v260) % 4 == 0);
                                *v260 = *v259;
                                v250 += 1 ;
                            }
                            int v261;
                            v261 = 0;
                            while (while_method_6(v261)){
                                int v263;
                                v263 = 0;
                                while (while_method_0(v263)){
                                    bool v265;
                                    v265 = 0 <= v263;
                                    bool v267;
                                    if (v265){
                                        bool v266;
                                        v266 = v263 < 4;
                                        v267 = v266;
                                    } else {
                                        v267 = false;
                                    }
                                    bool v268;
                                    v268 = v267 == false;
                                    if (v268){
                                        assert("The indices should be inside the range of the dimension." && v267);
                                    } else {
                                    }
                                    bool v270;
                                    v270 = 0 <= v214;
                                    bool v272;
                                    if (v270){
                                        bool v271;
                                        v271 = v214 < 4;
                                        v272 = v271;
                                    } else {
                                        v272 = false;
                                    }
                                    bool v273;
                                    v273 = v272 == false;
                                    if (v273){
                                        assert("The indices should be inside the range of the dimension." && v272);
                                    } else {
                                    }
                                    int v275;
                                    v275 = v214 * 4;
                                    int v276;
                                    v276 = v263 + v275;
                                    bool v277;
                                    v277 = 0 <= v261;
                                    bool v279;
                                    if (v277){
                                        bool v278;
                                        v278 = v261 < 1;
                                        v279 = v278;
                                    } else {
                                        v279 = false;
                                    }
                                    bool v280;
                                    v280 = v279 == false;
                                    if (v280){
                                        assert("The indices should be inside the range of the dimension." && v279);
                                    } else {
                                    }
                                    int v282;
                                    v282 = v261 * 16;
                                    int v283;
                                    v283 = v276 + v282;
                                    assert("Tensor range check" && 0 <= v261 && v261 < 1);
                                    assert("Tensor range check" && 0 <= v263 && v263 < 4);
                                    int v284;
                                    v284 = 4 * v261;
                                    int v285;
                                    v285 = v284 + v263;
                                    v249[v285] = v283;
                                    v263 += 1 ;
                                }
                                v261 += 1 ;
                            }
                            float v286[4];
                            int v287;
                            v287 = 0;
                            while (while_method_6(v287)){
                                int v289;
                                v289 = 0;
                                while (while_method_0(v289)){
                                    assert("Tensor range check" && 0 <= v287 && v287 < 1);
                                    assert("Tensor range check" && 0 <= v289 && v289 < 4);
                                    int v291;
                                    v291 = 4 * v287;
                                    int v292;
                                    v292 = v291 + v289;
                                    float v293;
                                    v293 = v247[v292];
                                    float v294;
                                    v294 = v248[v292];
                                    bool v295;
                                    v295 = v294 == 0.0f;
                                    bool v296;
                                    v296 = v295 != true;
                                    float v298;
                                    if (v296){
                                        float v297;
                                        v297 = v293 / v294;
                                        v298 = v297;
                                    } else {
                                        v298 = 0.0f;
                                    }
                                    assert("Tensor range check" && 0 <= v287 && v287 < 1);
                                    assert("Tensor range check" && 0 <= v289 && v289 < 4);
                                    v286[v292] = v298;
                                    v289 += 1 ;
                                }
                                v287 += 1 ;
                            }
                            bool v299[4];
                            int v300;
                            v300 = 0;
                            while (while_method_6(v300)){
                                int v302;
                                v302 = 0;
                                while (while_method_0(v302)){
                                    assert("Tensor range check" && 0 <= v300 && v300 < 1);
                                    assert("Tensor range check" && 0 <= v302 && v302 < 4);
                                    int v304;
                                    v304 = 4 * v300;
                                    int v305;
                                    v305 = v304 + v302;
                                    float v306;
                                    v306 = v246[v305];
                                    int v307;
                                    v307 = v249[v305];
                                    bool v308;
                                    v308 = v307 < 11;
                                    assert("Tensor range check" && 0 <= v300 && v300 < 1);
                                    assert("Tensor range check" && 0 <= v302 && v302 < 4);
                                    v299[v305] = v308;
                                    v302 += 1 ;
                                }
                                v300 += 1 ;
                            }
                            float v309[4];
                            int v310;
                            v310 = 0;
                            while (while_method_6(v310)){
                                int v312;
                                v312 = 0;
                                while (while_method_0(v312)){
                                    assert("Tensor range check" && 0 <= v310 && v310 < 1);
                                    assert("Tensor range check" && 0 <= v312 && v312 < 4);
                                    int v314;
                                    v314 = 4 * v310;
                                    int v315;
                                    v315 = v314 + v312;
                                    float v316;
                                    v316 = v246[v315];
                                    bool v317;
                                    v317 = v299[v315];
                                    float v320;
                                    if (v317){
                                        bool v318;
                                        v318 = 0.0f >= v316;
                                        if (v318){
                                            v320 = 0.0f;
                                        } else {
                                            v320 = v316;
                                        }
                                    } else {
                                        v320 = 0.0f;
                                    }
                                    assert("Tensor range check" && 0 <= v310 && v310 < 1);
                                    assert("Tensor range check" && 0 <= v312 && v312 < 4);
                                    v309[v315] = v320;
                                    v312 += 1 ;
                                }
                                v310 += 1 ;
                            }
                            float v321;
                            v321 = 0.0f;
                            int v322;
                            v322 = 0;
                            while (while_method_6(v322)){
                                int v324;
                                v324 = 0;
                                while (while_method_0(v324)){
                                    assert("Tensor range check" && 0 <= v322 && v322 < 1);
                                    assert("Tensor range check" && 0 <= v324 && v324 < 4);
                                    int v326;
                                    v326 = 4 * v322;
                                    int v327;
                                    v327 = v326 + v324;
                                    float v328;
                                    v328 = v309[v327];
                                    float v329;
                                    v329 = v321 + v328;
                                    v321 = v329;
                                    v324 += 1 ;
                                }
                                v322 += 1 ;
                            }
                            auto v330 = cooperative_groups::coalesced_threads();
                            int v331;
                            v331 = threadIdx.x;
                            int v332;
                            v332 = v331 / 4;
                            auto v333 = cooperative_groups::labeled_partition(v330,v332);
                            Closure0 v334{};
                            float v335;
                            v335 = cooperative_groups::reduce(v333, v321, v334);
                            int v336[4];
                            int v337;
                            v337 = 0;
                            while (while_method_6(v337)){
                                int v339;
                                v339 = 0;
                                while (while_method_0(v339)){
                                    assert("Tensor range check" && 0 <= v337 && v337 < 1);
                                    assert("Tensor range check" && 0 <= v339 && v339 < 4);
                                    int v341;
                                    v341 = 4 * v337;
                                    int v342;
                                    v342 = v341 + v339;
                                    bool v343;
                                    v343 = v299[v342];
                                    int v344;
                                    if (v343){
                                        v344 = 1;
                                    } else {
                                        v344 = 0;
                                    }
                                    assert("Tensor range check" && 0 <= v337 && v337 < 1);
                                    assert("Tensor range check" && 0 <= v339 && v339 < 4);
                                    v336[v342] = v344;
                                    v339 += 1 ;
                                }
                                v337 += 1 ;
                            }
                            int v345;
                            v345 = 0;
                            int v346;
                            v346 = 0;
                            while (while_method_6(v346)){
                                int v348;
                                v348 = 0;
                                while (while_method_0(v348)){
                                    assert("Tensor range check" && 0 <= v346 && v346 < 1);
                                    assert("Tensor range check" && 0 <= v348 && v348 < 4);
                                    int v350;
                                    v350 = 4 * v346;
                                    int v351;
                                    v351 = v350 + v348;
                                    int v352;
                                    v352 = v336[v351];
                                    int v353;
                                    v353 = v345 + v352;
                                    v345 = v353;
                                    v348 += 1 ;
                                }
                                v346 += 1 ;
                            }
                            auto v354 = cooperative_groups::coalesced_threads();
                            int v355;
                            v355 = threadIdx.x;
                            int v356;
                            v356 = v355 / 4;
                            auto v357 = cooperative_groups::labeled_partition(v354,v356);
                            Closure1 v358{};
                            int v359;
                            v359 = cooperative_groups::reduce(v357, v345, v358);
                            float v360;
                            v360 = (float)v359;
                            float v361;
                            v361 = 1.0f / v360;
                            float v362[4];
                            int v363;
                            v363 = 0;
                            while (while_method_6(v363)){
                                int v365;
                                v365 = 0;
                                while (while_method_0(v365)){
                                    assert("Tensor range check" && 0 <= v363 && v363 < 1);
                                    assert("Tensor range check" && 0 <= v365 && v365 < 4);
                                    int v367;
                                    v367 = 4 * v363;
                                    int v368;
                                    v368 = v367 + v365;
                                    float v369;
                                    v369 = v309[v368];
                                    bool v370;
                                    v370 = v299[v368];
                                    bool v371;
                                    v371 = v370 == false;
                                    float v376;
                                    if (v371){
                                        v376 = 0.0f;
                                    } else {
                                        bool v372;
                                        v372 = v335 == 0.0f;
                                        bool v373;
                                        v373 = v372 != true;
                                        if (v373){
                                            float v374;
                                            v374 = v369 / v335;
                                            v376 = v374;
                                        } else {
                                            v376 = v361;
                                        }
                                    }
                                    assert("Tensor range check" && 0 <= v363 && v363 < 1);
                                    assert("Tensor range check" && 0 <= v365 && v365 < 4);
                                    v362[v368] = v376;
                                    v365 += 1 ;
                                }
                                v363 += 1 ;
                            }
                            float v377[4];
                            int v378;
                            v378 = 0;
                            while (while_method_6(v378)){
                                int v380;
                                v380 = 0;
                                while (while_method_0(v380)){
                                    assert("Tensor range check" && 0 <= v378 && v378 < 1);
                                    assert("Tensor range check" && 0 <= v380 && v380 < 4);
                                    int v382;
                                    v382 = 4 * v378;
                                    int v383;
                                    v383 = v382 + v380;
                                    float v384;
                                    v384 = v286[v383];
                                    int v385;
                                    v385 = v249[v383];
                                    bool v386;
                                    v386 = v235 == v385;
                                    float v389;
                                    if (v386){
                                        float v387;
                                        v387 = v236 - v384;
                                        float v388;
                                        v388 = v387 / v234;
                                        v389 = v388;
                                    } else {
                                        v389 = 0.0f;
                                    }
                                    float v390;
                                    v390 = v389 + v384;
                                    assert("Tensor range check" && 0 <= v378 && v378 < 1);
                                    assert("Tensor range check" && 0 <= v380 && v380 < 4);
                                    v377[v383] = v390;
                                    v380 += 1 ;
                                }
                                v378 += 1 ;
                            }
                            float v391[4];
                            int v392;
                            v392 = 0;
                            while (while_method_6(v392)){
                                int v394;
                                v394 = 0;
                                while (while_method_0(v394)){
                                    assert("Tensor range check" && 0 <= v392 && v392 < 1);
                                    assert("Tensor range check" && 0 <= v394 && v394 < 4);
                                    int v396;
                                    v396 = 4 * v392;
                                    int v397;
                                    v397 = v396 + v394;
                                    float v398;
                                    v398 = v362[v397];
                                    float v399;
                                    v399 = v377[v397];
                                    float v400;
                                    v400 = v398 * v399;
                                    assert("Tensor range check" && 0 <= v392 && v392 < 1);
                                    assert("Tensor range check" && 0 <= v394 && v394 < 4);
                                    v391[v397] = v400;
                                    v394 += 1 ;
                                }
                                v392 += 1 ;
                            }
                            float v401;
                            v401 = 0.0f;
                            int v402;
                            v402 = 0;
                            while (while_method_6(v402)){
                                int v404;
                                v404 = 0;
                                while (while_method_0(v404)){
                                    assert("Tensor range check" && 0 <= v402 && v402 < 1);
                                    assert("Tensor range check" && 0 <= v404 && v404 < 4);
                                    int v406;
                                    v406 = 4 * v402;
                                    int v407;
                                    v407 = v406 + v404;
                                    float v408;
                                    v408 = v391[v407];
                                    float v409;
                                    v409 = v401 + v408;
                                    v401 = v409;
                                    v404 += 1 ;
                                }
                                v402 += 1 ;
                            }
                            auto v410 = cooperative_groups::coalesced_threads();
                            int v411;
                            v411 = threadIdx.x;
                            int v412;
                            v412 = v411 / 4;
                            auto v413 = cooperative_groups::labeled_partition(v410,v412);
                            float v414;
                            v414 = cooperative_groups::reduce(v413, v401, v334);
                            int v415;
                            v415 = 0;
                            while (while_method_6(v415)){
                                int v417;
                                v417 = 0;
                                while (while_method_0(v417)){
                                    assert("Tensor range check" && 0 <= v415 && v415 < 1);
                                    assert("Tensor range check" && 0 <= v417 && v417 < 4);
                                    int v419;
                                    v419 = 4 * v415;
                                    int v420;
                                    v420 = v419 + v417;
                                    float v421;
                                    v421 = v377[v420];
                                    int v422;
                                    v422 = v249[v420];
                                    float v423;
                                    v423 = v421 - v414;
                                    float v424;
                                    v424 = v237 * v423;
                                    assert("Tensor range check" && 0 <= v422 && v422 < 16);
                                    float * v425;
                                    v425 = v238+v422;
                                    float v427;
                                    v427 = atomicAdd(v425,v424);
                                    v417 += 1 ;
                                }
                                v415 += 1 ;
                            }
                            int v428;
                            v428 = 0;
                            while (while_method_6(v428)){
                                assert("Tensor range check" && 0 <= v428 && v428 < 1);
                                assert("Tensor range check" && 0 <= v428 && v428 < 1);
                                v428 += 1 ;
                            }
                            assert("Tensor range check" && 0 <= v231 && v231 < 256);
                            v208[v231] = v414;
                            v219 += 1 ;
                        }
                        asm("barrier.cta.sync %0;" :: "r"(0));
                        assert("Tensor range check" && 0 <= v210 && v210 < 256);
                        float v430;
                        v430 = v208[v210];
                        asm("barrier.cta.sync %0;" :: "r"(0));
                        assert("Tensor range check" && 0 <= v102 && v102 < 2);
                        v81[v102] = v430;
                    }
                    int v431;
                    v431 = threadIdx.x;
                    int v432;
                    v432 = blockIdx.x;
                    int v433;
                    v433 = v432 * 256;
                    int v434;
                    v434 = v431 + v433;
                    assert("Tensor range check" && 0 <= v75 && v75 < 4);
                    int v435;
                    v435 = 12288 * v75;
                    assert("Tensor range check" && 0 <= v434 && v434 < 6144);
                    int v436;
                    v436 = 2 * v434;
                    int v437;
                    v437 = v436 + v435;
                    double * v438;
                    v438 = v69+v437;
                    double * v440;
                    v440 = v71+v437;
                    double * v442;
                    v442 = v438+0;
                    double * v444;
                    v444 = v440+0;
                    double * v446;
                    v446 = v438+0;
                    double * v448;
                    v448 = v440+0;
                    int v450;
                    v450 = sizeof(double *);
                    unsigned long long v451;
                    v451 = (unsigned long long)v450;
                    unsigned long long v452;
                    v452 = 256ull * v451;
                    unsigned long long v453;
                    v453 = v452 + 16ull;
                    unsigned long long v454;
                    v454 = v453 - 1ull;
                    unsigned long long v455;
                    v455 = v454 % 16ull;
                    unsigned long long v456;
                    v456 = v454 - v455;
                    unsigned long long v457;
                    v457 = v456 + v452;
                    unsigned long long v458;
                    v458 = v457 + 16ull;
                    unsigned long long v459;
                    v459 = v458 - 1ull;
                    unsigned long long v460;
                    v460 = v459 % 16ull;
                    unsigned long long v461;
                    v461 = v459 - v460;
                    unsigned long long v462;
                    v462 = v461 + v452;
                    unsigned long long v463;
                    v463 = v462 + 16ull;
                    unsigned long long v464;
                    v464 = v463 - 1ull;
                    unsigned long long v465;
                    v465 = v464 % 16ull;
                    unsigned long long v466;
                    v466 = v464 - v465;
                    unsigned long long v467;
                    v467 = v466 + v452;
                    bool v468;
                    v468 = v467 <= 98304ull;
                    bool v469;
                    v469 = v468 == false;
                    if (v469){
                        assert("The dynamic shared memory is insufficient to allocate the tensor." && v468);
                    } else {
                    }
                    extern __shared__ unsigned char v471[];
                    bool v472;
                    v472 = v467 <= v467;
                    bool v473;
                    v473 = v472 == false;
                    if (v473){
                        assert("The length of the partition has to be less than or equal to the length of the base array." && v472);
                    } else {
                    }
                    double * * v475;
                    v475 = reinterpret_cast<double * *>(&v471[0ull]);
                    double * * v477;
                    v477 = reinterpret_cast<double * *>(&v471[v456]);
                    double * * v479;
                    v479 = reinterpret_cast<double * *>(&v471[v461]);
                    double * * v481;
                    v481 = reinterpret_cast<double * *>(&v471[v466]);
                    int v483;
                    v483 = threadIdx.x;
                    assert("Tensor range check" && 0 <= v483 && v483 < 256);
                    v475[v483] = v442;
                    v477[v483] = v444;
                    v479[v483] = v446;
                    v481[v483] = v448;
                    asm("barrier.cta.sync %0;" :: "r"(0));
                    bool v484;
                    v484 = 0 <= v483;
                    bool v485;
                    v485 = v484 == false;
                    if (v485){
                        assert("The index needs to be zero or positive." && v484);
                    } else {
                    }
                    int v487;
                    v487 = v483 % 1;
                    bool v488;
                    v488 = v483 < 256;
                    bool v489;
                    v489 = v488 == false;
                    if (v489){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v488);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v483 && v483 < 256);
                    int v491;
                    v491 = 0;
                    while (while_method_6(v491)){
                        bool v493;
                        v493 = v484 && v488;
                        bool v494;
                        v494 = v493 == false;
                        if (v494){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v493);
                        } else {
                        }
                        bool v496;
                        v496 = 0 <= v491;
                        bool v498;
                        if (v496){
                            bool v497;
                            v497 = v491 < 1;
                            v498 = v497;
                        } else {
                            v498 = false;
                        }
                        bool v499;
                        v499 = v498 == false;
                        if (v499){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v498);
                        } else {
                        }
                        int v501;
                        v501 = v491 * 256;
                        int v502;
                        v502 = v501 + v483;
                        assert("Tensor range check" && 0 <= v491 && v491 < 1);
                        int v503;
                        v503 = 256 * v491;
                        int v504;
                        v504 = v503 + v483;
                        double * v505;
                        v505 = v475[v504];
                        double * v506;
                        v506 = v477[v504];
                        double * v507;
                        v507 = v479[v504];
                        double * v508;
                        v508 = v481[v504];
                        int v509;
                        v509 = blockIdx.x;
                        int v510;
                        v510 = v509 * 256;
                        int v511;
                        v511 = v510 + v502;
                        assert("Tensor range check" && 0 <= v487 && v487 < 1);
                        int v512;
                        v512 = 2 * v487;
                        double v513[2];
                        double v514[2];
                        int v515[2];
                        int v516;
                        v516 = 0;
                        while (while_method_6(v516)){
                            assert("Tensor range check" && 0 <= v516 && v516 < 1);
                            int v518;
                            v518 = 2 * v516;
                            assert("Tensor range check" && 0 <= v516 && v516 < 1);
                            int v519;
                            v519 = v518 + v512;
                            int4* v520;
                            v520 = reinterpret_cast<int4*>(v505 + v519);
                            int4* v521;
                            v521 = reinterpret_cast<int4*>(v513 + v518);
                            assert("Pointer alignment check" && (unsigned long long)(v520) % 2 == 0 && (unsigned long long)(v521) % 2 == 0);
                            *v521 = *v520;
                            int4* v522;
                            v522 = reinterpret_cast<int4*>(v506 + v519);
                            int4* v523;
                            v523 = reinterpret_cast<int4*>(v514 + v518);
                            assert("Pointer alignment check" && (unsigned long long)(v522) % 2 == 0 && (unsigned long long)(v523) % 2 == 0);
                            *v523 = *v522;
                            v516 += 1 ;
                        }
                        int v524;
                        v524 = 0;
                        while (while_method_6(v524)){
                            int v526;
                            v526 = 0;
                            while (while_method_2(v526)){
                                bool v528;
                                v528 = 0 <= v526;
                                bool v530;
                                if (v528){
                                    bool v529;
                                    v529 = v526 < 2;
                                    v530 = v529;
                                } else {
                                    v530 = false;
                                }
                                bool v531;
                                v531 = v530 == false;
                                if (v531){
                                    assert("The indices should be inside the range of the dimension." && v530);
                                } else {
                                }
                                bool v533;
                                v533 = 0 <= v487;
                                bool v535;
                                if (v533){
                                    bool v534;
                                    v534 = v487 < 1;
                                    v535 = v534;
                                } else {
                                    v535 = false;
                                }
                                bool v536;
                                v536 = v535 == false;
                                if (v536){
                                    assert("The indices should be inside the range of the dimension." && v535);
                                } else {
                                }
                                int v538;
                                v538 = v487 * 2;
                                int v539;
                                v539 = v526 + v538;
                                bool v540;
                                v540 = 0 <= v524;
                                bool v542;
                                if (v540){
                                    bool v541;
                                    v541 = v524 < 1;
                                    v542 = v541;
                                } else {
                                    v542 = false;
                                }
                                bool v543;
                                v543 = v542 == false;
                                if (v543){
                                    assert("The indices should be inside the range of the dimension." && v542);
                                } else {
                                }
                                int v545;
                                v545 = v524 * 2;
                                int v546;
                                v546 = v539 + v545;
                                assert("Tensor range check" && 0 <= v524 && v524 < 1);
                                assert("Tensor range check" && 0 <= v526 && v526 < 2);
                                int v547;
                                v547 = 2 * v524;
                                int v548;
                                v548 = v547 + v526;
                                v515[v548] = v546;
                                v526 += 1 ;
                            }
                            v524 += 1 ;
                        }
                        double v549[2];
                        double v550[2];
                        int v551;
                        v551 = 0;
                        while (while_method_6(v551)){
                            int v553;
                            v553 = 0;
                            while (while_method_2(v553)){
                                assert("Tensor range check" && 0 <= v551 && v551 < 1);
                                assert("Tensor range check" && 0 <= v553 && v553 < 2);
                                int v555;
                                v555 = 2 * v551;
                                int v556;
                                v556 = v555 + v553;
                                double v557;
                                v557 = v513[v556];
                                double v558;
                                v558 = v514[v556];
                                assert("Tensor range check" && 0 <= v551 && v551 < 1);
                                assert("Tensor range check" && 0 <= v553 && v553 < 2);
                                v549[v556] = 0.0;
                                v550[v556] = 0.0;
                                v553 += 1 ;
                            }
                            v551 += 1 ;
                        }
                        int v559;
                        v559 = 0;
                        while (while_method_6(v559)){
                            assert("Tensor range check" && 0 <= v559 && v559 < 1);
                            int v561;
                            v561 = 2 * v559;
                            int v562;
                            v562 = v561 + v512;
                            assert("Tensor range check" && 0 <= v559 && v559 < 1);
                            int4* v563;
                            v563 = reinterpret_cast<int4*>(v549 + v561);
                            int4* v564;
                            v564 = reinterpret_cast<int4*>(v507 + v562);
                            assert("Pointer alignment check" && (unsigned long long)(v563) % 2 == 0 && (unsigned long long)(v564) % 2 == 0);
                            *v564 = *v563;
                            int4* v565;
                            v565 = reinterpret_cast<int4*>(v550 + v561);
                            int4* v566;
                            v566 = reinterpret_cast<int4*>(v508 + v562);
                            assert("Pointer alignment check" && (unsigned long long)(v565) % 2 == 0 && (unsigned long long)(v566) % 2 == 0);
                            *v566 = *v565;
                            v559 += 1 ;
                        }
                        assert("Tensor range check" && 0 <= v502 && v502 < 256);
                        v491 += 1 ;
                    }
                    asm("barrier.cta.sync %0;" :: "r"(0));
                    assert("Tensor range check" && 0 <= v483 && v483 < 256);
                    asm("barrier.cta.sync %0;" :: "r"(0));
                    assert("Tensor range check" && 0 <= v75 && v75 < 4);
                    assert("Tensor range check" && 0 <= v434 && v434 < 6144);
                    int v567;
                    v567 = v91 + v434;
                    v73[v567] = 0;
                    v75 += 1 ;
                }
                v27 += 1 ;
            }
            v25 += 1 ;
        }
        cooperative_groups::grid_group & v568 = v22.v1;
        cooperative_groups::grid_group & v569 = v568;
        curandStatePhilox4_32_10_t & v570 = v22.v5;
        curandStatePhilox4_32_10_t & v571 = v570;
        unsigned int * v572;
        v572 = reinterpret_cast<unsigned int *>(&v0[53477376ull]);
        int * v574;
        v574 = reinterpret_cast<int *>(&v1[4194304ull]);
        float * v576;
        v576 = reinterpret_cast<float *>(&v1[4194320ull]);
        float * v578;
        v578 = reinterpret_cast<float *>(&v1[5242896ull]);
        float * v580;
        v580 = reinterpret_cast<float *>(&v1[6291472ull]);
        float * v582;
        v582 = reinterpret_cast<float *>(&v1[7340048ull]);
        float * v584;
        v584 = reinterpret_cast<float *>(&v1[8388624ull]);
        float * v586;
        v586 = reinterpret_cast<float *>(&v1[9437200ull]);
        float * v588;
        v588 = reinterpret_cast<float *>(&v1[10485776ull]);
        int * v590;
        v590 = reinterpret_cast<int *>(&v0[53575680ull]);
        float * v592;
        v592 = reinterpret_cast<float *>(&v0[66158592ull]);
        int * v594;
        v594 = reinterpret_cast<int *>(&v0[78741504ull]);
        int * v596;
        v596 = reinterpret_cast<int *>(&v0[91324416ull]);
        double * v598;
        v598 = reinterpret_cast<double *>(&v0[103907328ull]);
        double * v600;
        v600 = reinterpret_cast<double *>(&v0[154238976ull]);
        double * v602;
        v602 = reinterpret_cast<double *>(&v1[11534352ull]);
        double * v604;
        v604 = reinterpret_cast<double *>(&v1[11927568ull]);
        int * v606;
        v606 = reinterpret_cast<int *>(&v1[12320784ull]);
        v569.sync() ;
        int v608;
        v608 = threadIdx.x;
        int v609;
        v609 = blockIdx.x;
        int v610;
        v610 = v609 * 256;
        int v611;
        v611 = v608 + v610;
        bool v612;
        v612 = v611 == 0;
        if (v612){
            int v613;
            v613 = 0;
            int v614;
            v614 = 4;
            int v615;
            v615 = int_range_16(v614, v613, v571);
            v574[0] = v615;
        } else {
        }
        __syncwarp();
        int v616;
        v616 = threadIdx.x;
        bool v617;
        v617 = 0 <= v616;
        bool v618;
        v618 = v617 == false;
        if (v618){
            assert("The index needs to be zero or positive." && v617);
        } else {
        }
        int v620;
        v620 = v616 % 4;
        int v621;
        v621 = v616 / 4;
        int v622;
        v622 = v621 % 64;
        int v623;
        v623 = v621 / 64;
        bool v624;
        v624 = v623 < 1;
        bool v625;
        v625 = v624 == false;
        if (v625){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v624);
        } else {
        }
        assert("Tensor range check" && 0 <= v623 && v623 < 1);
        assert("Tensor range check" && 0 <= v622 && v622 < 64);
        assert("Tensor range check" && 0 <= v620 && v620 < 4);
        int v627;
        v627 = 4 * v620;
        int v628;
        v628 = 16 * v622;
        int v629;
        v629 = v628 + v627;
        int v630;
        v630 = 65536 * v623;
        int v631;
        v631 = v630 + v629;
        assert("Tensor range check" && 0 <= v623 && v623 < 1);
        assert("Tensor range check" && 0 <= v622 && v622 < 64);
        assert("Tensor range check" && 0 <= v620 && v620 < 4);
        int v632;
        v632 = blockIdx.x;
        int v633;
        v633 = v632;
        while (while_method_15(v633)){
            bool v635;
            v635 = 0 <= v633;
            bool v636;
            v636 = v635 == false;
            if (v636){
                assert("The index needs to be zero or positive." && v635);
            } else {
            }
            int v638;
            v638 = v633 % 64;
            int v639;
            v639 = v633 / 64;
            bool v640;
            v640 = v639 < 4;
            bool v641;
            v641 = v640 == false;
            if (v641){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v640);
            } else {
            }
            assert("Tensor range check" && 0 <= v639 && v639 < 4);
            assert("Tensor range check" && 0 <= v638 && v638 < 64);
            int v643;
            v643 = 1024 * v638;
            int v644;
            v644 = v643 + v631;
            int v645;
            v645 = 65536 * v639;
            int v646;
            v646 = v645 + v644;
            float v647[4];
            float v648[4];
            float v649[4];
            float v650[4];
            float v651[4];
            float v652[4];
            float v653[4];
            int v654[4];
            int v655;
            v655 = 0;
            while (while_method_6(v655)){
                assert("Tensor range check" && 0 <= v655 && v655 < 1);
                int v657;
                v657 = 4 * v655;
                assert("Tensor range check" && 0 <= v655 && v655 < 1);
                int v658;
                v658 = 16 * v655;
                int v659;
                v659 = v658 + v646;
                int4* v660;
                v660 = reinterpret_cast<int4*>(v576 + v659);
                int4* v661;
                v661 = reinterpret_cast<int4*>(v647 + v657);
                assert("Pointer alignment check" && (unsigned long long)(v660) % 4 == 0 && (unsigned long long)(v661) % 4 == 0);
                *v661 = *v660;
                int4* v662;
                v662 = reinterpret_cast<int4*>(v578 + v659);
                int4* v663;
                v663 = reinterpret_cast<int4*>(v648 + v657);
                assert("Pointer alignment check" && (unsigned long long)(v662) % 4 == 0 && (unsigned long long)(v663) % 4 == 0);
                *v663 = *v662;
                int4* v664;
                v664 = reinterpret_cast<int4*>(v580 + v659);
                int4* v665;
                v665 = reinterpret_cast<int4*>(v649 + v657);
                assert("Pointer alignment check" && (unsigned long long)(v664) % 4 == 0 && (unsigned long long)(v665) % 4 == 0);
                *v665 = *v664;
                int4* v666;
                v666 = reinterpret_cast<int4*>(v582 + v659);
                int4* v667;
                v667 = reinterpret_cast<int4*>(v650 + v657);
                assert("Pointer alignment check" && (unsigned long long)(v666) % 4 == 0 && (unsigned long long)(v667) % 4 == 0);
                *v667 = *v666;
                int4* v668;
                v668 = reinterpret_cast<int4*>(v584 + v659);
                int4* v669;
                v669 = reinterpret_cast<int4*>(v651 + v657);
                assert("Pointer alignment check" && (unsigned long long)(v668) % 4 == 0 && (unsigned long long)(v669) % 4 == 0);
                *v669 = *v668;
                int4* v670;
                v670 = reinterpret_cast<int4*>(v586 + v659);
                int4* v671;
                v671 = reinterpret_cast<int4*>(v652 + v657);
                assert("Pointer alignment check" && (unsigned long long)(v670) % 4 == 0 && (unsigned long long)(v671) % 4 == 0);
                *v671 = *v670;
                int4* v672;
                v672 = reinterpret_cast<int4*>(v588 + v659);
                int4* v673;
                v673 = reinterpret_cast<int4*>(v653 + v657);
                assert("Pointer alignment check" && (unsigned long long)(v672) % 4 == 0 && (unsigned long long)(v673) % 4 == 0);
                *v673 = *v672;
                v655 += 1 ;
            }
            int v674;
            v674 = 0;
            while (while_method_6(v674)){
                int v676;
                v676 = 0;
                while (while_method_0(v676)){
                    bool v678;
                    v678 = 0 <= v676;
                    bool v680;
                    if (v678){
                        bool v679;
                        v679 = v676 < 4;
                        v680 = v679;
                    } else {
                        v680 = false;
                    }
                    bool v681;
                    v681 = v680 == false;
                    if (v681){
                        assert("The indices should be inside the range of the dimension." && v680);
                    } else {
                    }
                    bool v683;
                    v683 = 0 <= v620;
                    bool v685;
                    if (v683){
                        bool v684;
                        v684 = v620 < 4;
                        v685 = v684;
                    } else {
                        v685 = false;
                    }
                    bool v686;
                    v686 = v685 == false;
                    if (v686){
                        assert("The indices should be inside the range of the dimension." && v685);
                    } else {
                    }
                    int v688;
                    v688 = v620 * 4;
                    int v689;
                    v689 = v676 + v688;
                    bool v690;
                    v690 = 0 <= v674;
                    bool v692;
                    if (v690){
                        bool v691;
                        v691 = v674 < 1;
                        v692 = v691;
                    } else {
                        v692 = false;
                    }
                    bool v693;
                    v693 = v692 == false;
                    if (v693){
                        assert("The indices should be inside the range of the dimension." && v692);
                    } else {
                    }
                    int v695;
                    v695 = v674 * 16;
                    int v696;
                    v696 = v689 + v695;
                    assert("Tensor range check" && 0 <= v674 && v674 < 1);
                    assert("Tensor range check" && 0 <= v676 && v676 < 4);
                    int v697;
                    v697 = 4 * v674;
                    int v698;
                    v698 = v697 + v676;
                    v654[v698] = v696;
                    v676 += 1 ;
                }
                v674 += 1 ;
            }
            bool v699;
            v699 = 0 <= v623;
            bool v700;
            v700 = v699 && v624;
            bool v701;
            v701 = v700 == false;
            if (v701){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v700);
            } else {
            }
            bool v703;
            v703 = 0 <= v622;
            bool v705;
            if (v703){
                bool v704;
                v704 = v622 < 64;
                v705 = v704;
            } else {
                v705 = false;
            }
            bool v706;
            v706 = v705 == false;
            if (v706){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v705);
            } else {
            }
            bool v708;
            v708 = 0 <= v639;
            bool v709;
            v709 = v708 && v640;
            bool v710;
            v710 = v709 == false;
            if (v710){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v709);
            } else {
            }
            bool v712;
            v712 = 0 <= v638;
            bool v714;
            if (v712){
                bool v713;
                v713 = v638 < 64;
                v714 = v713;
            } else {
                v714 = false;
            }
            bool v715;
            v715 = v714 == false;
            if (v715){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v714);
            } else {
            }
            int v717;
            v717 = v638 * 64;
            int v718;
            v718 = v639 + v623;
            int v719;
            v719 = v717 + v622;
            bool v720[4];
            int v721;
            v721 = 0;
            while (while_method_6(v721)){
                int v723;
                v723 = 0;
                while (while_method_0(v723)){
                    assert("Tensor range check" && 0 <= v721 && v721 < 1);
                    assert("Tensor range check" && 0 <= v723 && v723 < 4);
                    int v725;
                    v725 = 4 * v721;
                    int v726;
                    v726 = v725 + v723;
                    float v727;
                    v727 = v649[v726];
                    bool v728;
                    v728 = v727 == 0.0f;
                    bool v729;
                    v729 = v728 != true;
                    assert("Tensor range check" && 0 <= v721 && v721 < 1);
                    assert("Tensor range check" && 0 <= v723 && v723 < 4);
                    v720[v726] = v729;
                    v723 += 1 ;
                }
                v721 += 1 ;
            }
            bool v730;
            v730 = false;
            int v731;
            v731 = 0;
            while (while_method_6(v731)){
                int v733;
                v733 = 0;
                while (while_method_0(v733)){
                    assert("Tensor range check" && 0 <= v731 && v731 < 1);
                    assert("Tensor range check" && 0 <= v733 && v733 < 4);
                    int v735;
                    v735 = 4 * v731;
                    int v736;
                    v736 = v735 + v733;
                    bool v737;
                    v737 = v720[v736];
                    bool v738;
                    v738 = v730 || v737;
                    v730 = v738;
                    v733 += 1 ;
                }
                v731 += 1 ;
            }
            auto v739 = cooperative_groups::coalesced_threads();
            int v740;
            v740 = threadIdx.x;
            int v741;
            v741 = v740 / 4;
            auto v742 = cooperative_groups::labeled_partition(v739,v741);
            Closure2 v743{};
            bool v744;
            v744 = cooperative_groups::reduce(v742, v730, v743);
            if (v744){
                float v745[4];
                int v746;
                v746 = 0;
                while (while_method_6(v746)){
                    int v748;
                    v748 = 0;
                    while (while_method_0(v748)){
                        assert("Tensor range check" && 0 <= v746 && v746 < 1);
                        assert("Tensor range check" && 0 <= v748 && v748 < 4);
                        int v750;
                        v750 = 4 * v746;
                        int v751;
                        v751 = v750 + v748;
                        float v752;
                        v752 = v648[v751];
                        float v753;
                        v753 = v649[v751];
                        float v754;
                        v754 = v752 + v753;
                        bool v755;
                        v755 = 0.0f >= v754;
                        float v756;
                        if (v755){
                            v756 = 0.0f;
                        } else {
                            v756 = v754;
                        }
                        assert("Tensor range check" && 0 <= v746 && v746 < 1);
                        assert("Tensor range check" && 0 <= v748 && v748 < 4);
                        v745[v751] = v756;
                        v748 += 1 ;
                    }
                    v746 += 1 ;
                }
                float v757[4];
                int v758;
                v758 = 0;
                while (while_method_6(v758)){
                    int v760;
                    v760 = 0;
                    while (while_method_0(v760)){
                        assert("Tensor range check" && 0 <= v758 && v758 < 1);
                        assert("Tensor range check" && 0 <= v760 && v760 < 4);
                        int v762;
                        v762 = 4 * v758;
                        int v763;
                        v763 = v762 + v760;
                        float v764;
                        v764 = v745[v763];
                        bool v765;
                        v765 = 0.0f >= v764;
                        float v766;
                        if (v765){
                            v766 = 0.0f;
                        } else {
                            v766 = v764;
                        }
                        assert("Tensor range check" && 0 <= v758 && v758 < 1);
                        assert("Tensor range check" && 0 <= v760 && v760 < 4);
                        v757[v763] = v766;
                        v760 += 1 ;
                    }
                    v758 += 1 ;
                }
                float v767;
                v767 = 0.0f;
                int v768;
                v768 = 0;
                while (while_method_6(v768)){
                    int v770;
                    v770 = 0;
                    while (while_method_0(v770)){
                        assert("Tensor range check" && 0 <= v768 && v768 < 1);
                        assert("Tensor range check" && 0 <= v770 && v770 < 4);
                        int v772;
                        v772 = 4 * v768;
                        int v773;
                        v773 = v772 + v770;
                        float v774;
                        v774 = v757[v773];
                        float v775;
                        v775 = v767 + v774;
                        v767 = v775;
                        v770 += 1 ;
                    }
                    v768 += 1 ;
                }
                auto v776 = cooperative_groups::coalesced_threads();
                int v777;
                v777 = threadIdx.x;
                int v778;
                v778 = v777 / 4;
                auto v779 = cooperative_groups::labeled_partition(v776,v778);
                Closure0 v780{};
                float v781;
                v781 = cooperative_groups::reduce(v779, v767, v780);
                float v782[4];
                int v783;
                v783 = 0;
                while (while_method_6(v783)){
                    int v785;
                    v785 = 0;
                    while (while_method_0(v785)){
                        assert("Tensor range check" && 0 <= v783 && v783 < 1);
                        assert("Tensor range check" && 0 <= v785 && v785 < 4);
                        int v787;
                        v787 = 4 * v783;
                        int v788;
                        v788 = v787 + v785;
                        float v789;
                        v789 = v757[v788];
                        bool v790;
                        v790 = v781 == 0.0f;
                        bool v791;
                        v791 = v790 != true;
                        float v793;
                        if (v791){
                            float v792;
                            v792 = v789 / v781;
                            v793 = v792;
                        } else {
                            v793 = 0.0625f;
                        }
                        assert("Tensor range check" && 0 <= v783 && v783 < 1);
                        assert("Tensor range check" && 0 <= v785 && v785 < 4);
                        v782[v788] = v793;
                        v785 += 1 ;
                    }
                    v783 += 1 ;
                }
                float v794[4];
                int v795;
                v795 = 0;
                while (while_method_6(v795)){
                    int v797;
                    v797 = 0;
                    while (while_method_0(v797)){
                        assert("Tensor range check" && 0 <= v795 && v795 < 1);
                        assert("Tensor range check" && 0 <= v797 && v797 < 4);
                        int v799;
                        v799 = 4 * v795;
                        int v800;
                        v800 = v799 + v797;
                        float v801;
                        v801 = v647[v800];
                        float v802;
                        v802 = v782[v800];
                        float v803;
                        v803 = v801 + v802;
                        assert("Tensor range check" && 0 <= v795 && v795 < 1);
                        assert("Tensor range check" && 0 <= v797 && v797 < 4);
                        v794[v800] = v803;
                        v797 += 1 ;
                    }
                    v795 += 1 ;
                }
                float v804[4];
                int v805;
                v805 = 0;
                while (while_method_6(v805)){
                    int v807;
                    v807 = 0;
                    while (while_method_0(v807)){
                        assert("Tensor range check" && 0 <= v805 && v805 < 1);
                        assert("Tensor range check" && 0 <= v807 && v807 < 4);
                        int v809;
                        v809 = 4 * v805;
                        int v810;
                        v810 = v809 + v807;
                        float v811;
                        v811 = v794[v810];
                        float v812;
                        v812 = -v811;
                        bool v813;
                        v813 = v811 >= v812;
                        float v814;
                        if (v813){
                            v814 = v811;
                        } else {
                            v814 = v812;
                        }
                        assert("Tensor range check" && 0 <= v805 && v805 < 1);
                        assert("Tensor range check" && 0 <= v807 && v807 < 4);
                        v804[v810] = v814;
                        v807 += 1 ;
                    }
                    v805 += 1 ;
                }
                float v815;
                v815 = 0.0f;
                int v816;
                v816 = 0;
                while (while_method_6(v816)){
                    int v818;
                    v818 = 0;
                    while (while_method_0(v818)){
                        assert("Tensor range check" && 0 <= v816 && v816 < 1);
                        assert("Tensor range check" && 0 <= v818 && v818 < 4);
                        int v820;
                        v820 = 4 * v816;
                        int v821;
                        v821 = v820 + v818;
                        float v822;
                        v822 = v804[v821];
                        float v823;
                        v823 = v815 + v822;
                        v815 = v823;
                        v818 += 1 ;
                    }
                    v816 += 1 ;
                }
                auto v824 = cooperative_groups::coalesced_threads();
                int v825;
                v825 = threadIdx.x;
                int v826;
                v826 = v825 / 4;
                auto v827 = cooperative_groups::labeled_partition(v824,v826);
                float v828;
                v828 = cooperative_groups::reduce(v827, v815, v780);
                bool v829;
                v829 = v828 > 100.0f;
                float v831;
                if (v829){
                    float v830;
                    v830 = 100.0f / v828;
                    v831 = v830;
                } else {
                    v831 = 1.0f;
                }
                float v832[4];
                int v833;
                v833 = 0;
                while (while_method_6(v833)){
                    int v835;
                    v835 = 0;
                    while (while_method_0(v835)){
                        assert("Tensor range check" && 0 <= v833 && v833 < 1);
                        assert("Tensor range check" && 0 <= v835 && v835 < 4);
                        int v837;
                        v837 = 4 * v833;
                        int v838;
                        v838 = v837 + v835;
                        float v839;
                        v839 = v804[v838];
                        float v840;
                        v840 = v831 * v839;
                        assert("Tensor range check" && 0 <= v833 && v833 < 1);
                        assert("Tensor range check" && 0 <= v835 && v835 < 4);
                        v832[v838] = v840;
                        v835 += 1 ;
                    }
                    v833 += 1 ;
                }
                float v841[4];
                float v842[4];
                int v843;
                v843 = 0;
                while (while_method_6(v843)){
                    int v845;
                    v845 = 0;
                    while (while_method_0(v845)){
                        assert("Tensor range check" && 0 <= v843 && v843 < 1);
                        assert("Tensor range check" && 0 <= v845 && v845 < 4);
                        int v847;
                        v847 = 4 * v843;
                        int v848;
                        v848 = v847 + v845;
                        float v849;
                        v849 = v647[v848];
                        float v850;
                        v850 = v648[v848];
                        float v851;
                        v851 = v649[v848];
                        float v852;
                        v852 = v650[v848];
                        float v853;
                        v853 = v651[v848];
                        float v854;
                        v854 = v652[v848];
                        float v855;
                        v855 = v653[v848];
                        float v856;
                        v856 = v852 + v854;
                        float v857;
                        v857 = v853 + v855;
                        assert("Tensor range check" && 0 <= v843 && v843 < 1);
                        assert("Tensor range check" && 0 <= v845 && v845 < 4);
                        v841[v848] = v856;
                        v842[v848] = v857;
                        v845 += 1 ;
                    }
                    v843 += 1 ;
                }
                int v858;
                v858 = 0;
                while (while_method_6(v858)){
                    int v860;
                    v860 = 0;
                    while (while_method_0(v860)){
                        assert("Tensor range check" && 0 <= v858 && v858 < 1);
                        assert("Tensor range check" && 0 <= v860 && v860 < 4);
                        int v862;
                        v862 = 4 * v858;
                        int v863;
                        v863 = v862 + v860;
                        float v864;
                        v864 = v832[v863];
                        float v865;
                        v865 = v745[v863];
                        float v866;
                        v866 = v841[v863];
                        float v867;
                        v867 = v842[v863];
                        assert("Tensor range check" && 0 <= v858 && v858 < 1);
                        assert("Tensor range check" && 0 <= v860 && v860 < 4);
                        v647[v863] = v864;
                        v648[v863] = v865;
                        v649[v863] = 0.0f;
                        v650[v863] = v866;
                        v651[v863] = v867;
                        v652[v863] = 0.0f;
                        v653[v863] = 0.0f;
                        v860 += 1 ;
                    }
                    v858 += 1 ;
                }
            } else {
            }
            assert("Tensor range check" && 0 <= v639 && v639 < 4);
            assert("Tensor range check" && 0 <= v638 && v638 < 64);
            int v868;
            v868 = 0;
            while (while_method_6(v868)){
                assert("Tensor range check" && 0 <= v868 && v868 < 1);
                int v870;
                v870 = 16 * v868;
                int v871;
                v871 = v870 + v646;
                assert("Tensor range check" && 0 <= v868 && v868 < 1);
                int v872;
                v872 = 4 * v868;
                int4* v873;
                v873 = reinterpret_cast<int4*>(v647 + v872);
                int4* v874;
                v874 = reinterpret_cast<int4*>(v576 + v871);
                assert("Pointer alignment check" && (unsigned long long)(v873) % 4 == 0 && (unsigned long long)(v874) % 4 == 0);
                *v874 = *v873;
                int4* v875;
                v875 = reinterpret_cast<int4*>(v648 + v872);
                int4* v876;
                v876 = reinterpret_cast<int4*>(v578 + v871);
                assert("Pointer alignment check" && (unsigned long long)(v875) % 4 == 0 && (unsigned long long)(v876) % 4 == 0);
                *v876 = *v875;
                int4* v877;
                v877 = reinterpret_cast<int4*>(v649 + v872);
                int4* v878;
                v878 = reinterpret_cast<int4*>(v580 + v871);
                assert("Pointer alignment check" && (unsigned long long)(v877) % 4 == 0 && (unsigned long long)(v878) % 4 == 0);
                *v878 = *v877;
                int4* v879;
                v879 = reinterpret_cast<int4*>(v650 + v872);
                int4* v880;
                v880 = reinterpret_cast<int4*>(v582 + v871);
                assert("Pointer alignment check" && (unsigned long long)(v879) % 4 == 0 && (unsigned long long)(v880) % 4 == 0);
                *v880 = *v879;
                int4* v881;
                v881 = reinterpret_cast<int4*>(v651 + v872);
                int4* v882;
                v882 = reinterpret_cast<int4*>(v584 + v871);
                assert("Pointer alignment check" && (unsigned long long)(v881) % 4 == 0 && (unsigned long long)(v882) % 4 == 0);
                *v882 = *v881;
                int4* v883;
                v883 = reinterpret_cast<int4*>(v652 + v872);
                int4* v884;
                v884 = reinterpret_cast<int4*>(v586 + v871);
                assert("Pointer alignment check" && (unsigned long long)(v883) % 4 == 0 && (unsigned long long)(v884) % 4 == 0);
                *v884 = *v883;
                int4* v885;
                v885 = reinterpret_cast<int4*>(v653 + v872);
                int4* v886;
                v886 = reinterpret_cast<int4*>(v588 + v871);
                assert("Pointer alignment check" && (unsigned long long)(v885) % 4 == 0 && (unsigned long long)(v886) % 4 == 0);
                *v886 = *v885;
                v868 += 1 ;
            }
            v633 += 24 ;
        }
        v569.sync() ;
        v23 += 1 ;
    }
    return ;
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

class dynamic_array(static_array): 
    pass

class dynamic_array_list(static_array_list):
    def length_(self): return self.length

import cupy as cp
from dataclasses import dataclass
from typing import NamedTuple, Union, Callable, Tuple
i8 = int; i16 = int; i32 = int; i64 = int; u8 = int; u16 = int; u32 = int; u64 = int; f32 = float; f64 = float; char = str; string = str

import time
options = []
options.append('--define-macro=NDEBUG')
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39,177')
options.append('--restrict')
options.append('--maxrregcount=255')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
class US0_0(NamedTuple): # Computer
    tag = 0
class US0_1(NamedTuple): # Human
    tag = 1
class US0_2(NamedTuple): # Random
    tag = 2
US0 = Union[US0_0, US0_1, US0_2]
class US3_0(NamedTuple): # Flop
    v0 : static_array
    tag = 0
class US3_1(NamedTuple): # Preflop
    tag = 1
class US3_2(NamedTuple): # River
    v0 : static_array
    tag = 2
class US3_3(NamedTuple): # Turn
    v0 : static_array
    tag = 3
US3 = Union[US3_0, US3_1, US3_2, US3_3]
class US4_0(NamedTuple): # A_All_In
    tag = 0
class US4_1(NamedTuple): # A_Call
    tag = 1
class US4_2(NamedTuple): # A_Fold
    tag = 2
class US4_3(NamedTuple): # A_Raise
    v0 : i32
    tag = 3
US4 = Union[US4_0, US4_1, US4_2, US4_3]
class US2_0(NamedTuple): # G_Flop
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US3
    tag = 0
class US2_1(NamedTuple): # G_Fold
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US3
    tag = 1
class US2_2(NamedTuple): # G_Preflop
    tag = 2
class US2_3(NamedTuple): # G_River
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US3
    tag = 3
class US2_4(NamedTuple): # G_Round
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US3
    tag = 4
class US2_5(NamedTuple): # G_Round'
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US3
    v6 : US4
    tag = 5
class US2_6(NamedTuple): # G_Showdown
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US3
    tag = 6
class US2_7(NamedTuple): # G_Turn
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US3
    tag = 7
US2 = Union[US2_0, US2_1, US2_2, US2_3, US2_4, US2_5, US2_6, US2_7]
class US1_0(NamedTuple): # None
    tag = 0
class US1_1(NamedTuple): # Some
    v0 : US2
    tag = 1
US1 = Union[US1_0, US1_1]
class US5_0(NamedTuple): # GameNotStarted
    tag = 0
class US5_1(NamedTuple): # GameOver
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US3
    tag = 1
class US5_2(NamedTuple): # WaitingForActionFromPlayerId
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US3
    tag = 2
US5 = Union[US5_0, US5_1, US5_2]
class US6_0(NamedTuple): # CommunityCardsAre
    v0 : static_array_list
    tag = 0
class US6_1(NamedTuple): # Fold
    v0 : i32
    v1 : i32
    tag = 1
class US6_2(NamedTuple): # PlayerAction
    v0 : i32
    v1 : US4
    tag = 2
class US6_3(NamedTuple): # PlayerGotCards
    v0 : i32
    v1 : static_array
    tag = 3
class US6_4(NamedTuple): # Showdown
    v0 : i32
    v1 : static_array
    v2 : i32
    tag = 4
US6 = Union[US6_0, US6_1, US6_2, US6_3, US6_4]
class US7_0(NamedTuple): # AddRewardsRando
    v0 : list
    tag = 0
class US7_1(NamedTuple): # AddRewardsSelf
    v0 : list
    tag = 1
US7 = Union[US7_0, US7_1]
def method1(v0 : cp.ndarray, v1 : u64) -> None:
    v3 = v0[0:].view(cp.uint64)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method2(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[8:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method3(v0 : cp.ndarray) -> None:
    del v0
    return 
def method5(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[0:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method7(v0 : i32) -> bool:
    v1 = v0 < 2
    del v0
    return v1
def method10(v0 : cp.ndarray, v1 : u8) -> None:
    v3 = v0[0:].view(cp.uint8)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method9(v0 : cp.ndarray, v1 : u8) -> None:
    return method10(v0, v1)
def method8(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method7(v2):
        v4 = u64(v2)
        v6 = v0[v4:].view(cp.uint8)
        del v4
        v7 = 0 <= v2
        if v7:
            v8 = v2 < 2
            v9 = v8
        else:
            v9 = False
        del v7
        v10 = v9 == False
        if v10:
            v11 = "Index must be in range."
            assert v9, v11
            del v11
        else:
            pass
        del v9, v10
        v13 = v1[v2]
        method9(v6, v13)
        del v6, v13
        v2 += 1 
    del v0, v1, v2
    return 
def method11(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[28:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method13(v0 : i32) -> bool:
    v1 = v0 < 3
    del v0
    return v1
def method12(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method13(v2):
        v4 = u64(v2)
        v6 = v0[v4:].view(cp.uint8)
        del v4
        v7 = 0 <= v2
        if v7:
            v8 = v2 < 3
            v9 = v8
        else:
            v9 = False
        del v7
        v10 = v9 == False
        if v10:
            v11 = "Index must be in range."
            assert v9, v11
            del v11
        else:
            pass
        del v9, v10
        v13 = v1[v2]
        method9(v6, v13)
        del v6, v13
        v2 += 1 
    del v0, v1, v2
    return 
def method15(v0 : i32) -> bool:
    v1 = v0 < 5
    del v0
    return v1
def method14(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method15(v2):
        v4 = u64(v2)
        v6 = v0[v4:].view(cp.uint8)
        del v4
        v7 = 0 <= v2
        if v7:
            v8 = v2 < 5
            v9 = v8
        else:
            v9 = False
        del v7
        v10 = v9 == False
        if v10:
            v11 = "Index must be in range."
            assert v9, v11
            del v11
        else:
            pass
        del v9, v10
        v13 = v1[v2]
        method9(v6, v13)
        del v6, v13
        v2 += 1 
    del v0, v1, v2
    return 
def method17(v0 : i32) -> bool:
    v1 = v0 < 4
    del v0
    return v1
def method16(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method17(v2):
        v4 = u64(v2)
        v6 = v0[v4:].view(cp.uint8)
        del v4
        v7 = 0 <= v2
        if v7:
            v8 = v2 < 4
            v9 = v8
        else:
            v9 = False
        del v7
        v10 = v9 == False
        if v10:
            v11 = "Index must be in range."
            assert v9, v11
            del v11
        else:
            pass
        del v9, v10
        v13 = v1[v2]
        method9(v6, v13)
        del v6, v13
        v2 += 1 
    del v0, v1, v2
    return 
def method6(v0 : cp.ndarray, v1 : i32, v2 : static_array, v3 : static_array, v4 : i32, v5 : static_array, v6 : US3) -> None:
    v8 = v0[0:].view(cp.int32)
    v8[0] = v1
    del v1, v8
    v9 = 0
    while method7(v9):
        v11 = u64(v9)
        v12 = v11 * 2
        del v11
        v13 = 4 + v12
        del v12
        v15 = v0[v13:].view(cp.uint8)
        del v13
        v16 = 0 <= v9
        if v16:
            v17 = v9 < 2
            v18 = v17
        else:
            v18 = False
        del v16
        v19 = v18 == False
        if v19:
            v20 = "Index must be in range."
            assert v18, v20
            del v20
        else:
            pass
        del v18, v19
        v22 = v2[v9]
        method8(v15, v22)
        del v15, v22
        v9 += 1 
    del v2, v9
    v23 = 0
    while method7(v23):
        v25 = u64(v23)
        v26 = v25 * 4
        del v25
        v27 = 8 + v26
        del v26
        v29 = v0[v27:].view(cp.uint8)
        del v27
        v30 = 0 <= v23
        if v30:
            v31 = v23 < 2
            v32 = v31
        else:
            v32 = False
        del v30
        v33 = v32 == False
        if v33:
            v34 = "Index must be in range."
            assert v32, v34
            del v34
        else:
            pass
        del v32, v33
        v36 = v3[v23]
        method5(v29, v36)
        del v29, v36
        v23 += 1 
    del v3, v23
    v38 = v0[16:].view(cp.int32)
    v38[0] = v4
    del v4, v38
    v39 = 0
    while method7(v39):
        v41 = u64(v39)
        v42 = v41 * 4
        del v41
        v43 = 20 + v42
        del v42
        v45 = v0[v43:].view(cp.uint8)
        del v43
        v46 = 0 <= v39
        if v46:
            v47 = v39 < 2
            v48 = v47
        else:
            v48 = False
        del v46
        v49 = v48 == False
        if v49:
            v50 = "Index must be in range."
            assert v48, v50
            del v50
        else:
            pass
        del v48, v49
        v52 = v5[v39]
        method5(v45, v52)
        del v45, v52
        v39 += 1 
    del v5, v39
    v53 = v6.tag
    method11(v0, v53)
    del v53
    v55 = v0[32:].view(cp.uint8)
    del v0
    match v6:
        case US3_0(v56): # Flop
            del v6
            return method12(v55, v56)
        case US3_1(): # Preflop
            del v6
            return method3(v55)
        case US3_2(v57): # River
            del v6
            return method14(v55, v57)
        case US3_3(v58): # Turn
            del v6
            return method16(v55, v58)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method19(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[40:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method18(v0 : cp.ndarray, v1 : i32, v2 : static_array, v3 : static_array, v4 : i32, v5 : static_array, v6 : US3, v7 : US4) -> None:
    v9 = v0[0:].view(cp.int32)
    v9[0] = v1
    del v1, v9
    v10 = 0
    while method7(v10):
        v12 = u64(v10)
        v13 = v12 * 2
        del v12
        v14 = 4 + v13
        del v13
        v16 = v0[v14:].view(cp.uint8)
        del v14
        v17 = 0 <= v10
        if v17:
            v18 = v10 < 2
            v19 = v18
        else:
            v19 = False
        del v17
        v20 = v19 == False
        if v20:
            v21 = "Index must be in range."
            assert v19, v21
            del v21
        else:
            pass
        del v19, v20
        v23 = v2[v10]
        method8(v16, v23)
        del v16, v23
        v10 += 1 
    del v2, v10
    v24 = 0
    while method7(v24):
        v26 = u64(v24)
        v27 = v26 * 4
        del v26
        v28 = 8 + v27
        del v27
        v30 = v0[v28:].view(cp.uint8)
        del v28
        v31 = 0 <= v24
        if v31:
            v32 = v24 < 2
            v33 = v32
        else:
            v33 = False
        del v31
        v34 = v33 == False
        if v34:
            v35 = "Index must be in range."
            assert v33, v35
            del v35
        else:
            pass
        del v33, v34
        v37 = v3[v24]
        method5(v30, v37)
        del v30, v37
        v24 += 1 
    del v3, v24
    v39 = v0[16:].view(cp.int32)
    v39[0] = v4
    del v4, v39
    v40 = 0
    while method7(v40):
        v42 = u64(v40)
        v43 = v42 * 4
        del v42
        v44 = 20 + v43
        del v43
        v46 = v0[v44:].view(cp.uint8)
        del v44
        v47 = 0 <= v40
        if v47:
            v48 = v40 < 2
            v49 = v48
        else:
            v49 = False
        del v47
        v50 = v49 == False
        if v50:
            v51 = "Index must be in range."
            assert v49, v51
            del v51
        else:
            pass
        del v49, v50
        v53 = v5[v40]
        method5(v46, v53)
        del v46, v53
        v40 += 1 
    del v5, v40
    v54 = v6.tag
    method11(v0, v54)
    del v54
    v56 = v0[32:].view(cp.uint8)
    match v6:
        case US3_0(v57): # Flop
            method12(v56, v57)
        case US3_1(): # Preflop
            method3(v56)
        case US3_2(v58): # River
            method14(v56, v58)
        case US3_3(v59): # Turn
            method16(v56, v59)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v6, v56
    v60 = v7.tag
    method19(v0, v60)
    del v60
    v62 = v0[44:].view(cp.uint8)
    del v0
    match v7:
        case US4_0(): # A_All_In
            del v7
            return method3(v62)
        case US4_1(): # A_Call
            del v7
            return method3(v62)
        case US4_2(): # A_Fold
            del v7
            return method3(v62)
        case US4_3(v63): # A_Raise
            del v7
            return method5(v62, v63)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method4(v0 : cp.ndarray, v1 : US2) -> None:
    v2 = v1.tag
    method5(v0, v2)
    del v2
    v4 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US2_0(v5, v6, v7, v8, v9, v10): # G_Flop
            del v1
            return method6(v4, v5, v6, v7, v8, v9, v10)
        case US2_1(v11, v12, v13, v14, v15, v16): # G_Fold
            del v1
            return method6(v4, v11, v12, v13, v14, v15, v16)
        case US2_2(): # G_Preflop
            del v1
            return method3(v4)
        case US2_3(v17, v18, v19, v20, v21, v22): # G_River
            del v1
            return method6(v4, v17, v18, v19, v20, v21, v22)
        case US2_4(v23, v24, v25, v26, v27, v28): # G_Round
            del v1
            return method6(v4, v23, v24, v25, v26, v27, v28)
        case US2_5(v29, v30, v31, v32, v33, v34, v35): # G_Round'
            del v1
            return method18(v4, v29, v30, v31, v32, v33, v34, v35)
        case US2_6(v36, v37, v38, v39, v40, v41): # G_Showdown
            del v1
            return method6(v4, v36, v37, v38, v39, v40, v41)
        case US2_7(v42, v43, v44, v45, v46, v47): # G_Turn
            del v1
            return method6(v4, v42, v43, v44, v45, v46, v47)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method20(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[80:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method21(v0 : i32, v1 : i32) -> bool:
    v2 = v1 < v0
    del v0, v1
    return v2
def method23(v0 : cp.ndarray, v1 : static_array_list) -> None:
    v2 = v1.length
    method5(v0, v2)
    del v2
    v3 = v1.length
    v4 = 0
    while method21(v3, v4):
        v6 = u64(v4)
        v7 = 4 + v6
        del v6
        v9 = v0[v7:].view(cp.uint8)
        del v7
        v11 = v1[v4]
        method9(v9, v11)
        del v9, v11
        v4 += 1 
    del v0, v1, v3, v4
    return 
def method24(v0 : cp.ndarray, v1 : i32, v2 : i32) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v6 = v0[4:].view(cp.int32)
    del v0
    v6[0] = v2
    del v2, v6
    return 
def method26(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[4:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method25(v0 : cp.ndarray, v1 : i32, v2 : US4) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v5 = v2.tag
    method26(v0, v5)
    del v5
    v7 = v0[8:].view(cp.uint8)
    del v0
    match v2:
        case US4_0(): # A_All_In
            del v2
            return method3(v7)
        case US4_1(): # A_Call
            del v2
            return method3(v7)
        case US4_2(): # A_Fold
            del v2
            return method3(v7)
        case US4_3(v8): # A_Raise
            del v2
            return method5(v7, v8)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method27(v0 : cp.ndarray, v1 : i32, v2 : static_array) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v5 = 0
    while method7(v5):
        v7 = u64(v5)
        v8 = 4 + v7
        del v7
        v10 = v0[v8:].view(cp.uint8)
        del v8
        v11 = 0 <= v5
        if v11:
            v12 = v5 < 2
            v13 = v12
        else:
            v13 = False
        del v11
        v14 = v13 == False
        if v14:
            v15 = "Index must be in range."
            assert v13, v15
            del v15
        else:
            pass
        del v13, v14
        v17 = v2[v5]
        method9(v10, v17)
        del v10, v17
        v5 += 1 
    del v0, v2, v5
    return 
def method30(v0 : cp.ndarray, v1 : static_array, v2 : i8) -> None:
    v3 = 0
    while method15(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v8 = 0 <= v3
        if v8:
            v9 = v3 < 5
            v10 = v9
        else:
            v10 = False
        del v8
        v11 = v10 == False
        if v11:
            v12 = "Index must be in range."
            assert v10, v12
            del v12
        else:
            pass
        del v10, v11
        v14 = v1[v3]
        method9(v7, v14)
        del v7, v14
        v3 += 1 
    del v1, v3
    v16 = v0[5:].view(cp.int8)
    del v0
    v16[0] = v2
    del v2, v16
    return 
def method29(v0 : cp.ndarray, v1 : static_array, v2 : i8) -> None:
    return method30(v0, v1, v2)
def method28(v0 : cp.ndarray, v1 : i32, v2 : static_array, v3 : i32) -> None:
    v5 = v0[0:].view(cp.int32)
    v5[0] = v1
    del v1, v5
    v6 = 0
    while method7(v6):
        v8 = u64(v6)
        v9 = v8 * 8
        del v8
        v10 = 8 + v9
        del v9
        v12 = v0[v10:].view(cp.uint8)
        del v10
        v13 = 0 <= v6
        if v13:
            v14 = v6 < 2
            v15 = v14
        else:
            v15 = False
        del v13
        v16 = v15 == False
        if v16:
            v17 = "Index must be in range."
            assert v15, v17
            del v17
        else:
            pass
        del v15, v16
        v20, v21 = v2[v6]
        method29(v12, v20, v21)
        del v12, v20, v21
        v6 += 1 
    del v2, v6
    v23 = v0[24:].view(cp.int32)
    del v0
    v23[0] = v3
    del v3, v23
    return 
def method22(v0 : cp.ndarray, v1 : US6) -> None:
    v2 = v1.tag
    method5(v0, v2)
    del v2
    v4 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US6_0(v5): # CommunityCardsAre
            del v1
            return method23(v4, v5)
        case US6_1(v6, v7): # Fold
            del v1
            return method24(v4, v6, v7)
        case US6_2(v8, v9): # PlayerAction
            del v1
            return method25(v4, v8, v9)
        case US6_3(v10, v11): # PlayerGotCards
            del v1
            return method27(v4, v10, v11)
        case US6_4(v12, v13, v14): # Showdown
            del v1
            return method28(v4, v12, v13, v14)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method31(v0 : cp.ndarray, v1 : US0) -> None:
    v2 = v1.tag
    method5(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US0_0(): # Computer
            del v1
            return method3(v4)
        case US0_1(): # Human
            del v1
            return method3(v4)
        case US0_2(): # Random
            del v1
            return method3(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method32(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[6248:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method0(v0 : cp.ndarray, v1 : u64, v2 : US1, v3 : static_array_list, v4 : static_array, v5 : US5) -> None:
    method1(v0, v1)
    del v1
    v6 = v2.tag
    method2(v0, v6)
    del v6
    v8 = v0[16:].view(cp.uint8)
    match v2:
        case US1_0(): # None
            method3(v8)
        case US1_1(v9): # Some
            method4(v8, v9)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v2, v8
    v10 = v3.length
    method20(v0, v10)
    del v10
    v11 = v3.length
    v12 = 0
    while method21(v11, v12):
        v14 = u64(v12)
        v15 = v14 * 48
        del v14
        v16 = 96 + v15
        del v15
        v18 = v0[v16:].view(cp.uint8)
        del v16
        v20 = v3[v12]
        method22(v18, v20)
        del v18, v20
        v12 += 1 
    del v3, v11, v12
    v21 = 0
    while method7(v21):
        v23 = u64(v21)
        v24 = v23 * 4
        del v23
        v25 = 6240 + v24
        del v24
        v27 = v0[v25:].view(cp.uint8)
        del v25
        v28 = 0 <= v21
        if v28:
            v29 = v21 < 2
            v30 = v29
        else:
            v30 = False
        del v28
        v31 = v30 == False
        if v31:
            v32 = "Index must be in range."
            assert v30, v32
            del v32
        else:
            pass
        del v30, v31
        v34 = v4[v21]
        method31(v27, v34)
        del v27, v34
        v21 += 1 
    del v4, v21
    v35 = v5.tag
    method32(v0, v35)
    del v35
    v37 = v0[6256:].view(cp.uint8)
    del v0
    match v5:
        case US5_0(): # GameNotStarted
            del v5
            return method3(v37)
        case US5_1(v38, v39, v40, v41, v42, v43): # GameOver
            del v5
            return method6(v37, v38, v39, v40, v41, v42, v43)
        case US5_2(v44, v45, v46, v47, v48, v49): # WaitingForActionFromPlayerId
            del v5
            return method6(v37, v44, v45, v46, v47, v48, v49)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method34(v0 : cp.ndarray) -> u64:
    v2 = v0[0:].view(cp.uint64)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method35(v0 : cp.ndarray) -> i32:
    v2 = v0[8:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method36(v0 : cp.ndarray) -> None:
    del v0
    return 
def method38(v0 : cp.ndarray) -> i32:
    v2 = v0[0:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method42(v0 : cp.ndarray) -> u8:
    v2 = v0[0:].view(cp.uint8)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method41(v0 : cp.ndarray) -> u8:
    v1 = method42(v0)
    del v0
    return v1
def method40(v0 : cp.ndarray) -> static_array:
    v2 = static_array(2)
    v3 = 0
    while method7(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v8 = method41(v7)
        del v7
        v2[v3] = v8
        del v8
        v3 += 1 
    del v0, v3
    return v2
def method43(v0 : cp.ndarray) -> i32:
    v2 = v0[28:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method44(v0 : cp.ndarray) -> static_array:
    v2 = static_array(3)
    v3 = 0
    while method13(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v8 = method41(v7)
        del v7
        v2[v3] = v8
        del v8
        v3 += 1 
    del v0, v3
    return v2
def method45(v0 : cp.ndarray) -> static_array:
    v2 = static_array(5)
    v3 = 0
    while method15(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v8 = method41(v7)
        del v7
        v2[v3] = v8
        del v8
        v3 += 1 
    del v0, v3
    return v2
def method46(v0 : cp.ndarray) -> static_array:
    v2 = static_array(4)
    v3 = 0
    while method17(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v8 = method41(v7)
        del v7
        v2[v3] = v8
        del v8
        v3 += 1 
    del v0, v3
    return v2
def method39(v0 : cp.ndarray) -> Tuple[i32, static_array, static_array, i32, static_array, US3]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v5 = static_array(2)
    v6 = 0
    while method7(v6):
        v8 = u64(v6)
        v9 = v8 * 2
        del v8
        v10 = 4 + v9
        del v9
        v12 = v0[v10:].view(cp.uint8)
        del v10
        v13 = method40(v12)
        del v12
        v5[v6] = v13
        del v13
        v6 += 1 
    del v6
    v15 = static_array(2)
    v16 = 0
    while method7(v16):
        v18 = u64(v16)
        v19 = v18 * 4
        del v18
        v20 = 8 + v19
        del v19
        v22 = v0[v20:].view(cp.uint8)
        del v20
        v23 = method38(v22)
        del v22
        v15[v16] = v23
        del v23
        v16 += 1 
    del v16
    v25 = v0[16:].view(cp.int32)
    v26 = v25[0].item()
    del v25
    v28 = static_array(2)
    v29 = 0
    while method7(v29):
        v31 = u64(v29)
        v32 = v31 * 4
        del v31
        v33 = 20 + v32
        del v32
        v35 = v0[v33:].view(cp.uint8)
        del v33
        v36 = method38(v35)
        del v35
        v28[v29] = v36
        del v36
        v29 += 1 
    del v29
    v37 = method43(v0)
    v39 = v0[32:].view(cp.uint8)
    del v0
    if v37 == 0:
        v41 = method44(v39)
        v48 = US3_0(v41)
    elif v37 == 1:
        method36(v39)
        v48 = US3_1()
    elif v37 == 2:
        v44 = method45(v39)
        v48 = US3_2(v44)
    elif v37 == 3:
        v46 = method46(v39)
        v48 = US3_3(v46)
    else:
        raise Exception("Invalid tag.")
    del v37, v39
    return v3, v5, v15, v26, v28, v48
def method48(v0 : cp.ndarray) -> i32:
    v2 = v0[40:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method47(v0 : cp.ndarray) -> Tuple[i32, static_array, static_array, i32, static_array, US3, US4]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v5 = static_array(2)
    v6 = 0
    while method7(v6):
        v8 = u64(v6)
        v9 = v8 * 2
        del v8
        v10 = 4 + v9
        del v9
        v12 = v0[v10:].view(cp.uint8)
        del v10
        v13 = method40(v12)
        del v12
        v5[v6] = v13
        del v13
        v6 += 1 
    del v6
    v15 = static_array(2)
    v16 = 0
    while method7(v16):
        v18 = u64(v16)
        v19 = v18 * 4
        del v18
        v20 = 8 + v19
        del v19
        v22 = v0[v20:].view(cp.uint8)
        del v20
        v23 = method38(v22)
        del v22
        v15[v16] = v23
        del v23
        v16 += 1 
    del v16
    v25 = v0[16:].view(cp.int32)
    v26 = v25[0].item()
    del v25
    v28 = static_array(2)
    v29 = 0
    while method7(v29):
        v31 = u64(v29)
        v32 = v31 * 4
        del v31
        v33 = 20 + v32
        del v32
        v35 = v0[v33:].view(cp.uint8)
        del v33
        v36 = method38(v35)
        del v35
        v28[v29] = v36
        del v36
        v29 += 1 
    del v29
    v37 = method43(v0)
    v39 = v0[32:].view(cp.uint8)
    if v37 == 0:
        v41 = method44(v39)
        v48 = US3_0(v41)
    elif v37 == 1:
        method36(v39)
        v48 = US3_1()
    elif v37 == 2:
        v44 = method45(v39)
        v48 = US3_2(v44)
    elif v37 == 3:
        v46 = method46(v39)
        v48 = US3_3(v46)
    else:
        raise Exception("Invalid tag.")
    del v37, v39
    v49 = method48(v0)
    v51 = v0[44:].view(cp.uint8)
    del v0
    if v49 == 0:
        method36(v51)
        v58 = US4_0()
    elif v49 == 1:
        method36(v51)
        v58 = US4_1()
    elif v49 == 2:
        method36(v51)
        v58 = US4_2()
    elif v49 == 3:
        v56 = method38(v51)
        v58 = US4_3(v56)
    else:
        raise Exception("Invalid tag.")
    del v49, v51
    return v3, v5, v15, v26, v28, v48, v58
def method37(v0 : cp.ndarray) -> US2:
    v1 = method38(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5, v6, v7, v8, v9, v10 = method39(v3)
        del v3
        return US2_0(v5, v6, v7, v8, v9, v10)
    elif v1 == 1:
        del v1
        v12, v13, v14, v15, v16, v17 = method39(v3)
        del v3
        return US2_1(v12, v13, v14, v15, v16, v17)
    elif v1 == 2:
        del v1
        method36(v3)
        del v3
        return US2_2()
    elif v1 == 3:
        del v1
        v20, v21, v22, v23, v24, v25 = method39(v3)
        del v3
        return US2_3(v20, v21, v22, v23, v24, v25)
    elif v1 == 4:
        del v1
        v27, v28, v29, v30, v31, v32 = method39(v3)
        del v3
        return US2_4(v27, v28, v29, v30, v31, v32)
    elif v1 == 5:
        del v1
        v34, v35, v36, v37, v38, v39, v40 = method47(v3)
        del v3
        return US2_5(v34, v35, v36, v37, v38, v39, v40)
    elif v1 == 6:
        del v1
        v42, v43, v44, v45, v46, v47 = method39(v3)
        del v3
        return US2_6(v42, v43, v44, v45, v46, v47)
    elif v1 == 7:
        del v1
        v49, v50, v51, v52, v53, v54 = method39(v3)
        del v3
        return US2_7(v49, v50, v51, v52, v53, v54)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method49(v0 : cp.ndarray) -> i32:
    v2 = v0[80:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method51(v0 : cp.ndarray) -> static_array_list:
    v2 = static_array_list(5)
    v3 = method38(v0)
    v2.unsafe_set_length(v3)
    del v3
    v4 = v2.length
    v5 = 0
    while method21(v4, v5):
        v7 = u64(v5)
        v8 = 4 + v7
        del v7
        v10 = v0[v8:].view(cp.uint8)
        del v8
        v11 = method41(v10)
        del v10
        v2[v5] = v11
        del v11
        v5 += 1 
    del v0, v4, v5
    return v2
def method52(v0 : cp.ndarray) -> Tuple[i32, i32]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v5 = v0[4:].view(cp.int32)
    del v0
    v6 = v5[0].item()
    del v5
    return v3, v6
def method54(v0 : cp.ndarray) -> i32:
    v2 = v0[4:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method53(v0 : cp.ndarray) -> Tuple[i32, US4]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v4 = method54(v0)
    v6 = v0[8:].view(cp.uint8)
    del v0
    if v4 == 0:
        method36(v6)
        v13 = US4_0()
    elif v4 == 1:
        method36(v6)
        v13 = US4_1()
    elif v4 == 2:
        method36(v6)
        v13 = US4_2()
    elif v4 == 3:
        v11 = method38(v6)
        v13 = US4_3(v11)
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    return v3, v13
def method55(v0 : cp.ndarray) -> Tuple[i32, static_array]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v5 = static_array(2)
    v6 = 0
    while method7(v6):
        v8 = u64(v6)
        v9 = 4 + v8
        del v8
        v11 = v0[v9:].view(cp.uint8)
        del v9
        v12 = method41(v11)
        del v11
        v5[v6] = v12
        del v12
        v6 += 1 
    del v0, v6
    return v3, v5
def method58(v0 : cp.ndarray) -> Tuple[static_array, i8]:
    v2 = static_array(5)
    v3 = 0
    while method15(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v8 = method41(v7)
        del v7
        v2[v3] = v8
        del v8
        v3 += 1 
    del v3
    v10 = v0[5:].view(cp.int8)
    del v0
    v11 = v10[0].item()
    del v10
    return v2, v11
def method57(v0 : cp.ndarray) -> Tuple[static_array, i8]:
    v1, v2 = method58(v0)
    del v0
    return v1, v2
def method56(v0 : cp.ndarray) -> Tuple[i32, static_array, i32]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v5 = static_array(2)
    v6 = 0
    while method7(v6):
        v8 = u64(v6)
        v9 = v8 * 8
        del v8
        v10 = 8 + v9
        del v9
        v12 = v0[v10:].view(cp.uint8)
        del v10
        v13, v14 = method57(v12)
        del v12
        v5[v6] = (v13, v14)
        del v13, v14
        v6 += 1 
    del v6
    v16 = v0[24:].view(cp.int32)
    del v0
    v17 = v16[0].item()
    del v16
    return v3, v5, v17
def method50(v0 : cp.ndarray) -> US6:
    v1 = method38(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5 = method51(v3)
        del v3
        return US6_0(v5)
    elif v1 == 1:
        del v1
        v7, v8 = method52(v3)
        del v3
        return US6_1(v7, v8)
    elif v1 == 2:
        del v1
        v10, v11 = method53(v3)
        del v3
        return US6_2(v10, v11)
    elif v1 == 3:
        del v1
        v13, v14 = method55(v3)
        del v3
        return US6_3(v13, v14)
    elif v1 == 4:
        del v1
        v16, v17, v18 = method56(v3)
        del v3
        return US6_4(v16, v17, v18)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method59(v0 : cp.ndarray) -> US0:
    v1 = method38(v0)
    v3 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method36(v3)
        del v3
        return US0_0()
    elif v1 == 1:
        del v1
        method36(v3)
        del v3
        return US0_1()
    elif v1 == 2:
        del v1
        method36(v3)
        del v3
        return US0_2()
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method60(v0 : cp.ndarray) -> i32:
    v2 = v0[6248:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method33(v0 : cp.ndarray) -> Tuple[u64, US1, static_array_list, static_array, US5]:
    v1 = method34(v0)
    v2 = method35(v0)
    v4 = v0[16:].view(cp.uint8)
    if v2 == 0:
        method36(v4)
        v9 = US1_0()
    elif v2 == 1:
        v7 = method37(v4)
        v9 = US1_1(v7)
    else:
        raise Exception("Invalid tag.")
    del v2, v4
    v11 = static_array_list(128)
    v12 = method49(v0)
    v11.unsafe_set_length(v12)
    del v12
    v13 = v11.length
    v14 = 0
    while method21(v13, v14):
        v16 = u64(v14)
        v17 = v16 * 48
        del v16
        v18 = 96 + v17
        del v17
        v20 = v0[v18:].view(cp.uint8)
        del v18
        v21 = method50(v20)
        del v20
        v11[v14] = v21
        del v21
        v14 += 1 
    del v13, v14
    v23 = static_array(2)
    v24 = 0
    while method7(v24):
        v26 = u64(v24)
        v27 = v26 * 4
        del v26
        v28 = 6240 + v27
        del v27
        v30 = v0[v28:].view(cp.uint8)
        del v28
        v31 = method59(v30)
        del v30
        v23[v24] = v31
        del v31
        v24 += 1 
    del v24
    v32 = method60(v0)
    v34 = v0[6256:].view(cp.uint8)
    del v0
    if v32 == 0:
        method36(v34)
        v51 = US5_0()
    elif v32 == 1:
        v37, v38, v39, v40, v41, v42 = method39(v34)
        v51 = US5_1(v37, v38, v39, v40, v41, v42)
    elif v32 == 2:
        v44, v45, v46, v47, v48, v49 = method39(v34)
        v51 = US5_2(v44, v45, v46, v47, v48, v49)
    else:
        raise Exception("Invalid tag.")
    del v32, v34
    return v1, v9, v11, v23, v51
def method67(v0 : u64) -> object:
    v1 = v0
    del v0
    return v1
def method66(v0 : u64) -> object:
    return method67(v0)
def method69() -> object:
    v0 = []
    return v0
def method72(v0 : i32) -> object:
    v1 = v0
    del v0
    return v1
def method76(v0 : u8) -> object:
    v1 = v0
    del v0
    return v1
def method75(v0 : u8) -> object:
    return method76(v0)
def method74(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method7(v2):
        v4 = 0 <= v2
        if v4:
            v5 = v2 < 2
            v6 = v5
        else:
            v6 = False
        del v4
        v7 = v6 == False
        if v7:
            v8 = "Index must be in range."
            assert v6, v8
            del v8
        else:
            pass
        del v6, v7
        v10 = v0[v2]
        v11 = method75(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method73(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method7(v2):
        v4 = 0 <= v2
        if v4:
            v5 = v2 < 2
            v6 = v5
        else:
            v6 = False
        del v4
        v7 = v6 == False
        if v7:
            v8 = "Index must be in range."
            assert v6, v8
            del v8
        else:
            pass
        del v6, v7
        v10 = v0[v2]
        v11 = method74(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method77(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method7(v2):
        v4 = 0 <= v2
        if v4:
            v5 = v2 < 2
            v6 = v5
        else:
            v6 = False
        del v4
        v7 = v6 == False
        if v7:
            v8 = "Index must be in range."
            assert v6, v8
            del v8
        else:
            pass
        del v6, v7
        v10 = v0[v2]
        v11 = method72(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method79(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method13(v2):
        v4 = 0 <= v2
        if v4:
            v5 = v2 < 3
            v6 = v5
        else:
            v6 = False
        del v4
        v7 = v6 == False
        if v7:
            v8 = "Index must be in range."
            assert v6, v8
            del v8
        else:
            pass
        del v6, v7
        v10 = v0[v2]
        v11 = method75(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method80(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method15(v2):
        v4 = 0 <= v2
        if v4:
            v5 = v2 < 5
            v6 = v5
        else:
            v6 = False
        del v4
        v7 = v6 == False
        if v7:
            v8 = "Index must be in range."
            assert v6, v8
            del v8
        else:
            pass
        del v6, v7
        v10 = v0[v2]
        v11 = method75(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method81(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method17(v2):
        v4 = 0 <= v2
        if v4:
            v5 = v2 < 4
            v6 = v5
        else:
            v6 = False
        del v4
        v7 = v6 == False
        if v7:
            v8 = "Index must be in range."
            assert v6, v8
            del v8
        else:
            pass
        del v6, v7
        v10 = v0[v2]
        v11 = method75(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method78(v0 : US3) -> object:
    match v0:
        case US3_0(v1): # Flop
            del v0
            v2 = method79(v1)
            del v1
            v3 = "Flop"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US3_1(): # Preflop
            del v0
            v5 = method69()
            v6 = "Preflop"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case US3_2(v8): # River
            del v0
            v9 = method80(v8)
            del v8
            v10 = "River"
            v11 = [v10,v9]
            del v9, v10
            return v11
        case US3_3(v12): # Turn
            del v0
            v13 = method81(v12)
            del v12
            v14 = "Turn"
            v15 = [v14,v13]
            del v13, v14
            return v15
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method71(v0 : i32, v1 : static_array, v2 : static_array, v3 : i32, v4 : static_array, v5 : US3) -> object:
    v6 = method72(v0)
    del v0
    v7 = method73(v1)
    del v1
    v8 = method77(v2)
    del v2
    v9 = method72(v3)
    del v3
    v10 = method77(v4)
    del v4
    v11 = method78(v5)
    del v5
    v12 = {'min_raise': v6, 'pl_card': v7, 'pot': v8, 'round_turn': v9, 'stack': v10, 'street': v11}
    del v6, v7, v8, v9, v10, v11
    return v12
def method83(v0 : US4) -> object:
    match v0:
        case US4_0(): # A_All_In
            del v0
            v1 = method69()
            v2 = "A_All_In"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US4_1(): # A_Call
            del v0
            v4 = method69()
            v5 = "A_Call"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US4_2(): # A_Fold
            del v0
            v7 = method69()
            v8 = "A_Fold"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US4_3(v10): # A_Raise
            del v0
            v11 = method72(v10)
            del v10
            v12 = "A_Raise"
            v13 = [v12,v11]
            del v11, v12
            return v13
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method82(v0 : i32, v1 : static_array, v2 : static_array, v3 : i32, v4 : static_array, v5 : US3, v6 : US4) -> object:
    v7 = []
    v8 = method71(v0, v1, v2, v3, v4, v5)
    del v0, v1, v2, v3, v4, v5
    v7.append(v8)
    del v8
    v9 = method83(v6)
    del v6
    v7.append(v9)
    del v9
    v10 = v7
    del v7
    return v10
def method70(v0 : US2) -> object:
    match v0:
        case US2_0(v1, v2, v3, v4, v5, v6): # G_Flop
            del v0
            v7 = method71(v1, v2, v3, v4, v5, v6)
            del v1, v2, v3, v4, v5, v6
            v8 = "G_Flop"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US2_1(v10, v11, v12, v13, v14, v15): # G_Fold
            del v0
            v16 = method71(v10, v11, v12, v13, v14, v15)
            del v10, v11, v12, v13, v14, v15
            v17 = "G_Fold"
            v18 = [v17,v16]
            del v16, v17
            return v18
        case US2_2(): # G_Preflop
            del v0
            v19 = method69()
            v20 = "G_Preflop"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case US2_3(v22, v23, v24, v25, v26, v27): # G_River
            del v0
            v28 = method71(v22, v23, v24, v25, v26, v27)
            del v22, v23, v24, v25, v26, v27
            v29 = "G_River"
            v30 = [v29,v28]
            del v28, v29
            return v30
        case US2_4(v31, v32, v33, v34, v35, v36): # G_Round
            del v0
            v37 = method71(v31, v32, v33, v34, v35, v36)
            del v31, v32, v33, v34, v35, v36
            v38 = "G_Round"
            v39 = [v38,v37]
            del v37, v38
            return v39
        case US2_5(v40, v41, v42, v43, v44, v45, v46): # G_Round'
            del v0
            v47 = method82(v40, v41, v42, v43, v44, v45, v46)
            del v40, v41, v42, v43, v44, v45, v46
            v48 = "G_Round'"
            v49 = [v48,v47]
            del v47, v48
            return v49
        case US2_6(v50, v51, v52, v53, v54, v55): # G_Showdown
            del v0
            v56 = method71(v50, v51, v52, v53, v54, v55)
            del v50, v51, v52, v53, v54, v55
            v57 = "G_Showdown"
            v58 = [v57,v56]
            del v56, v57
            return v58
        case US2_7(v59, v60, v61, v62, v63, v64): # G_Turn
            del v0
            v65 = method71(v59, v60, v61, v62, v63, v64)
            del v59, v60, v61, v62, v63, v64
            v66 = "G_Turn"
            v67 = [v66,v65]
            del v65, v66
            return v67
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method68(v0 : US1) -> object:
    match v0:
        case US1_0(): # None
            del v0
            v1 = method69()
            v2 = "None"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US1_1(v4): # Some
            del v0
            v5 = method70(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method65(v0 : u64, v1 : US1) -> object:
    v2 = method66(v0)
    del v0
    v3 = method68(v1)
    del v1
    v4 = {'deck': v2, 'game': v3}
    del v2, v3
    return v4
def method87(v0 : static_array_list) -> object:
    v1 = []
    v2 = v0.length
    v3 = 0
    while method21(v2, v3):
        v6 = v0[v3]
        v7 = method75(v6)
        del v6
        v1.append(v7)
        del v7
        v3 += 1 
    del v0, v2, v3
    return v1
def method88(v0 : i32, v1 : i32) -> object:
    v2 = method72(v0)
    del v0
    v3 = method72(v1)
    del v1
    v4 = {'chips_won': v2, 'winner_id': v3}
    del v2, v3
    return v4
def method89(v0 : i32, v1 : US4) -> object:
    v2 = []
    v3 = method72(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method83(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method90(v0 : i32, v1 : static_array) -> object:
    v2 = []
    v3 = method72(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method74(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method95(v0 : i8) -> object:
    v1 = v0
    del v0
    return v1
def method94(v0 : static_array, v1 : i8) -> object:
    v2 = method80(v0)
    del v0
    v3 = method95(v1)
    del v1
    v4 = {'hand': v2, 'score': v3}
    del v2, v3
    return v4
def method93(v0 : static_array, v1 : i8) -> object:
    return method94(v0, v1)
def method92(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method7(v2):
        v4 = 0 <= v2
        if v4:
            v5 = v2 < 2
            v6 = v5
        else:
            v6 = False
        del v4
        v7 = v6 == False
        if v7:
            v8 = "Index must be in range."
            assert v6, v8
            del v8
        else:
            pass
        del v6, v7
        v11, v12 = v0[v2]
        v13 = method93(v11, v12)
        del v11, v12
        v1.append(v13)
        del v13
        v2 += 1 
    del v0, v2
    return v1
def method91(v0 : i32, v1 : static_array, v2 : i32) -> object:
    v3 = method72(v0)
    del v0
    v4 = method92(v1)
    del v1
    v5 = method72(v2)
    del v2
    v6 = {'chips_won': v3, 'hands_shown': v4, 'winner_id': v5}
    del v3, v4, v5
    return v6
def method86(v0 : US6) -> object:
    match v0:
        case US6_0(v1): # CommunityCardsAre
            del v0
            v2 = method87(v1)
            del v1
            v3 = "CommunityCardsAre"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US6_1(v5, v6): # Fold
            del v0
            v7 = method88(v5, v6)
            del v5, v6
            v8 = "Fold"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US6_2(v10, v11): # PlayerAction
            del v0
            v12 = method89(v10, v11)
            del v10, v11
            v13 = "PlayerAction"
            v14 = [v13,v12]
            del v12, v13
            return v14
        case US6_3(v15, v16): # PlayerGotCards
            del v0
            v17 = method90(v15, v16)
            del v15, v16
            v18 = "PlayerGotCards"
            v19 = [v18,v17]
            del v17, v18
            return v19
        case US6_4(v20, v21, v22): # Showdown
            del v0
            v23 = method91(v20, v21, v22)
            del v20, v21, v22
            v24 = "Showdown"
            v25 = [v24,v23]
            del v23, v24
            return v25
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method85(v0 : static_array_list) -> object:
    v1 = []
    v2 = v0.length
    v3 = 0
    while method21(v2, v3):
        v6 = v0[v3]
        v7 = method86(v6)
        del v6
        v1.append(v7)
        del v7
        v3 += 1 
    del v0, v2, v3
    return v1
def method97(v0 : US0) -> object:
    match v0:
        case US0_0(): # Computer
            del v0
            v1 = method69()
            v2 = "Computer"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US0_1(): # Human
            del v0
            v4 = method69()
            v5 = "Human"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US0_2(): # Random
            del v0
            v7 = method69()
            v8 = "Random"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method96(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method7(v2):
        v4 = 0 <= v2
        if v4:
            v5 = v2 < 2
            v6 = v5
        else:
            v6 = False
        del v4
        v7 = v6 == False
        if v7:
            v8 = "Index must be in range."
            assert v6, v8
            del v8
        else:
            pass
        del v6, v7
        v10 = v0[v2]
        v11 = method97(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method98(v0 : US5) -> object:
    match v0:
        case US5_0(): # GameNotStarted
            del v0
            v1 = method69()
            v2 = "GameNotStarted"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US5_1(v4, v5, v6, v7, v8, v9): # GameOver
            del v0
            v10 = method71(v4, v5, v6, v7, v8, v9)
            del v4, v5, v6, v7, v8, v9
            v11 = "GameOver"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case US5_2(v13, v14, v15, v16, v17, v18): # WaitingForActionFromPlayerId
            del v0
            v19 = method71(v13, v14, v15, v16, v17, v18)
            del v13, v14, v15, v16, v17, v18
            v20 = "WaitingForActionFromPlayerId"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method84(v0 : static_array_list, v1 : static_array, v2 : US5) -> object:
    v3 = method85(v0)
    del v0
    v4 = method96(v1)
    del v1
    v5 = method98(v2)
    del v2
    v6 = {'messages': v3, 'pl_type': v4, 'ui_game_state': v5}
    del v3, v4, v5
    return v6
def method64(v0 : u64, v1 : US1, v2 : static_array_list, v3 : static_array, v4 : US5) -> object:
    v5 = method65(v0, v1)
    del v0, v1
    v6 = method84(v2, v3, v4)
    del v2, v3, v4
    v7 = {'private': v5, 'public': v6}
    del v5, v6
    return v7
def method104(v0 : cp.ndarray) -> object:
    v1 = v0
    del v0
    return v1
def method103(v0 : cp.ndarray) -> object:
    return method104(v0)
def method102(v0 : cp.ndarray, v1 : u64) -> object:
    v2 = []
    v3 = method103(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method67(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method101(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method102(v0, v1)
    del v0, v1
    v5 = method102(v2, v3)
    del v2, v3
    v6 = {'output': v4, 'param': v5}
    del v4, v5
    return v6
def method100(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    return method101(v0, v1, v2, v3)
def method99(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method100(v0, v1, v2, v3)
    del v0, v1, v2, v3
    v5 = {'model_data': v4}
    del v4
    return v5
def method63(v0 : u64, v1 : US1, v2 : static_array_list, v3 : static_array, v4 : US5, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64) -> object:
    v9 = method64(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    v10 = method99(v5, v6, v7, v8)
    del v5, v6, v7, v8
    v11 = {'game': v9, 'neural': v10}
    del v9, v10
    return v11
def method109(v0 : f32) -> object:
    v1 = v0
    del v0
    return v1
def method108(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method21(v2, v3):
        v5 = v0[v3]
        v6 = method109(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method107(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method21(v2, v3):
        v5 = v0[v3]
        v6 = method108(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method106(v0 : US7) -> object:
    match v0:
        case US7_0(v1): # AddRewardsRando
            del v0
            v2 = method107(v1)
            del v1
            v3 = "AddRewardsRando"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US7_1(v5): # AddRewardsSelf
            del v0
            v6 = method107(v5)
            del v5
            v7 = "AddRewardsSelf"
            v8 = [v7,v6]
            del v6, v7
            return v8
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method105(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method21(v2, v3):
        v5 = v0[v3]
        v6 = method106(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method62(v0 : u64, v1 : US1, v2 : static_array_list, v3 : static_array, v4 : US5, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64, v9 : list) -> object:
    v10 = []
    v11 = method63(v0, v1, v2, v3, v4, v5, v6, v7, v8)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8
    v10.append(v11)
    del v11
    v12 = method105(v9)
    del v9
    v10.append(v12)
    del v12
    v13 = v10
    del v10
    return v13
def method61(v0 : u64, v1 : US1, v2 : static_array_list, v3 : static_array, v4 : US5, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64, v9 : list) -> object:
    v10 = method62(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8, v9
    return v10
def main_body():
    v1 = static_array(2)
    v3 = US0_0()
    v1[0] = v3
    del v3
    v5 = US0_1()
    v1[1] = v5
    del v5
    v7 = static_array_list(128)
    v8 = cp.empty(12419088,dtype=cp.uint8)
    v9 = cp.empty(204570624,dtype=cp.uint8)
    v11 = v8[0:0+4*1048576].view(cp.float32)
    v12 = cp.random.normal(0.0,0.0009765625,1048576,dtype=cp.float32) # type: ignore
    cp.copyto(v11[0:0+1048576],v12[0:0+1048576])
    del v11, v12
    v14 = v8[4194304:4194304+4*1].view(cp.int32)
    v16 = v8[4194320:4194320+4*262144].view(cp.float32)
    v18 = v8[5242896:5242896+4*262144].view(cp.float32)
    v20 = v8[6291472:6291472+4*262144].view(cp.float32)
    v22 = v8[7340048:7340048+4*262144].view(cp.float32)
    v24 = v8[8388624:8388624+4*262144].view(cp.float32)
    v26 = v8[9437200:9437200+4*262144].view(cp.float32)
    v28 = v8[10485776:10485776+4*262144].view(cp.float32)
    v14[:] = 0
    del v14
    v16[:] = 0
    del v16
    v18[:] = 0
    del v18
    v20[:] = 0
    del v20
    v22[:] = 0
    del v22
    v24[:] = 0
    del v24
    v26[:] = 0
    del v26
    v28[:] = 0
    del v28
    v30 = v8[11534352:11534352+8*49152].view(cp.float64)
    v32 = v8[11927568:11927568+8*49152].view(cp.float64)
    v34 = v8[12320784:12320784+4*24576].view(cp.int32)
    v30[:] = 0
    del v30
    v32[:] = 0
    del v32
    v34[:] = 0
    del v34
    v35 = cp.empty(16,dtype=cp.uint8)
    del v35
    v36 = cp.empty(6304,dtype=cp.uint8)
    v37 = 4503599627370495
    v38 = US1_0()
    v39 = US5_0()
    method0(v36, v37, v38, v7, v1, v39)
    del v1, v7, v37, v38, v39
    v42 = "{}\n"
    v43 = "Going to run the NL Holdem full kernel."
    print(v42.format(v43),end="")
    del v42, v43
    v44 = time.perf_counter()
    v45 = []
    v46 = cp.zeros(16,dtype=cp.float32) # type: ignore
    del v46
    v47 = cp.zeros(16,dtype=cp.float32) # type: ignore
    del v47
    v48 = cp.cuda.Device().attributes['MultiProcessorCount']
    v49 = v48 == 24
    del v48
    v50 = v49 == False
    if v50:
        v51 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v49, v51
        del v51
    else:
        pass
    del v49, v50
    v52 = 0
    v53 = raw_module.get_function(f"entry{v52}")
    del v52
    v53.max_dynamic_shared_size_bytes = 98304 
    print(f'Threads per block, blocks per grid: {256}, {24}')
    v53((24,),(256,),(v9, v8),shared_mem=98304)
    del v53
    cp.cuda.get_current_stream().synchronize()
    v54 = time.perf_counter()
    v57 = "{}"
    v58 = "The time it took to run the kernel (in seconds) is: "
    print(v57.format(v58),end="")
    del v57, v58
    v59 = v54 - v44
    del v44, v54
    v62 = "{:.6f}\n"
    print(v62.format(v59),end="")
    del v59, v62
    v63, v64, v65, v66, v67 = method33(v36)
    del v36
    v68 = 204570624
    v69 = 12419088
    return method61(v63, v64, v65, v66, v67, v9, v68, v8, v69, v45)

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
