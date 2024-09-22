kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <mma.h>
using namespace nvcuda;
#include <cuda/pipeline>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
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
struct Union7;
struct Tuple7;
__device__ void method_11(unsigned int v0, float * v1, int v2);
__device__ void method_12(unsigned int v0, float * v1, int v2);
struct Union8;
__device__ void method_13(unsigned int * v0, int v1, float * v2);
struct Tuple8;
struct Tuple9;
struct Tuple10;
struct Tuple11;
__device__ Tuple8 method_14(curandStatePhilox4_32_10_t & v0, int * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int v9, int v10);
__device__ float method_15(int * v0, float * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int v8, int v9, int v10);
struct Union9;
struct Tuple12;
__device__ int loop_19(static_array<float,6> v0, float v1, int v2);
__device__ int pick_discrete__18(static_array<float,6> v0, float v1);
__device__ int sample_discrete__17(static_array<float,6> v0, curandStatePhilox4_32_10_t & v1);
__device__ Union2 sample_discrete_16(static_array<Tuple12,6> v0, curandStatePhilox4_32_10_t & v1);
struct Tuple13;
struct Tuple14;
struct Union10;
struct Tuple15;
struct Union11;
struct Tuple16;
struct Tuple17;
struct Union12;
struct Union13;
struct Union14;
struct Union15;
struct Union16;
__device__ Tuple0 score_20(static_array<unsigned char,7> v0);
__device__ void method_0(unsigned char * v0, unsigned char * v1, StackMut0 & v2, int v3, Union3 v4);
struct Tuple18;
__device__ int int_range_21(int v0, int v1, curandStatePhilox4_32_10_t & v2);
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
struct Union7_0 { // None
};
struct Union7_1 { // Some
    Union2 v0;
    __device__ Union7_1(Union2 t0) : v0(t0) {}
    __device__ Union7_1() = delete;
};
struct Union7 {
    union {
        Union7_0 case0; // None
        Union7_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union7() {}
    __device__ Union7(Union7_0 t) : tag(0), case0(t) {} // None
    __device__ Union7(Union7_1 t) : tag(1), case1(t) {} // Some
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
    __device__ Union7 & operator=(Union7 && x) {
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
    __device__ ~Union7() {
        switch(this->tag){
            case 0: this->case0.~Union7_0(); break; // None
            case 1: this->case1.~Union7_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Tuple7 {
    int v0;
    unsigned int v1;
    __device__ Tuple7() = default;
    __device__ Tuple7(int t0, unsigned int t1) : v0(t0), v1(t1) {}
};
struct Union8_0 { // None
};
struct Union8_1 { // Some
    int v0;
    __device__ Union8_1(int t0) : v0(t0) {}
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
struct Closure0 {
    __device__ unsigned int operator()(unsigned int tup0, unsigned int tup1){
        unsigned int v0 = tup0; unsigned int v1 = tup1;
        unsigned int v2;
        v2 = v0 | v1;
        return v2;
    }
};
struct Tuple8 {
    float v0;
    int v1;
    __device__ Tuple8() = default;
    __device__ Tuple8(float t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure1 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Closure2 {
    __device__ int operator()(int tup0, int tup1){
        int v0 = tup0; int v1 = tup1;
        int v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Tuple9 {
    int v0;
    float v1;
    __device__ Tuple9() = default;
    __device__ Tuple9(int t0, float t1) : v0(t0), v1(t1) {}
};
struct Closure3 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Tuple10 {
    float v0;
    bool v1;
    __device__ Tuple10() = default;
    __device__ Tuple10(float t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure4 {
    __device__ Tuple10 operator()(Tuple10 tup0, Tuple10 tup1){
        float v0 = tup0.v0; bool v1 = tup0.v1; float v2 = tup1.v0; bool v3 = tup1.v1;
        if (v1){
            if (v3){
                bool v4;
                v4 = v0 >= v2;
                float v5;
                if (v4){
                    v5 = v0;
                } else {
                    v5 = v2;
                }
                return Tuple10{v5, true};
            } else {
                return Tuple10{v0, v1};
            }
        } else {
            if (v3){
                return Tuple10{v2, v3};
            } else {
                return Tuple10{v0, v1};
            }
        }
    }
};
struct Closure5 {
    __device__ Tuple8 operator()(Tuple8 tup0, Tuple8 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v1 < v3;
        if (v4){
            return Tuple8{v0, v1};
        } else {
            return Tuple8{v2, v3};
        }
    }
};
struct Tuple11 {
    int v0;
    bool v1;
    __device__ Tuple11() = default;
    __device__ Tuple11(int t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure6 {
    __device__ Tuple11 operator()(Tuple11 tup0, Tuple11 tup1){
        int v0 = tup0.v0; bool v1 = tup0.v1; int v2 = tup1.v0; bool v3 = tup1.v1;
        if (v1){
            if (v3){
                bool v4;
                v4 = v0 < v2;
                int v5;
                if (v4){
                    v5 = v0;
                } else {
                    v5 = v2;
                }
                return Tuple11{v5, true};
            } else {
                return Tuple11{v0, v1};
            }
        } else {
            if (v3){
                return Tuple11{v2, v3};
            } else {
                return Tuple11{v0, v1};
            }
        }
    }
};
struct Closure7 {
    int v0;
    __device__ Tuple8 operator()(Tuple8 tup0, Tuple8 tup1){
        int & v0 = this->v0;
        float v1 = tup0.v0; int v2 = tup0.v1; float v3 = tup1.v0; int v4 = tup1.v1;
        bool v5;
        v5 = v2 == v0;
        if (v5){
            return Tuple8{v1, v2};
        } else {
            bool v6;
            v6 = v4 == v0;
            if (v6){
                return Tuple8{v3, v4};
            } else {
                return Tuple8{v1, v2};
            }
        }
    }
    __device__ Closure7(int _v0) : v0(_v0) { }
};
struct Union9_0 { // AA_Call
};
struct Union9_1 { // AA_Fold
};
struct Union9_2 { // AA_Raise
    int v0;
    int v1;
    __device__ Union9_2(int t0, int t1) : v0(t0), v1(t1) {}
    __device__ Union9_2() = delete;
};
struct Union9 {
    union {
        Union9_0 case0; // AA_Call
        Union9_1 case1; // AA_Fold
        Union9_2 case2; // AA_Raise
    };
    unsigned char tag{255};
    __device__ Union9() {}
    __device__ Union9(Union9_0 t) : tag(0), case0(t) {} // AA_Call
    __device__ Union9(Union9_1 t) : tag(1), case1(t) {} // AA_Fold
    __device__ Union9(Union9_2 t) : tag(2), case2(t) {} // AA_Raise
    __device__ Union9(Union9 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union9_0(x.case0); break; // AA_Call
            case 1: new (&this->case1) Union9_1(x.case1); break; // AA_Fold
            case 2: new (&this->case2) Union9_2(x.case2); break; // AA_Raise
        }
    }
    __device__ Union9(Union9 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union9_0(std::move(x.case0)); break; // AA_Call
            case 1: new (&this->case1) Union9_1(std::move(x.case1)); break; // AA_Fold
            case 2: new (&this->case2) Union9_2(std::move(x.case2)); break; // AA_Raise
        }
    }
    __device__ Union9 & operator=(Union9 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // AA_Call
                case 1: this->case1 = x.case1; break; // AA_Fold
                case 2: this->case2 = x.case2; break; // AA_Raise
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
                case 0: this->case0 = std::move(x.case0); break; // AA_Call
                case 1: this->case1 = std::move(x.case1); break; // AA_Fold
                case 2: this->case2 = std::move(x.case2); break; // AA_Raise
            }
        } else {
            this->~Union9();
            new (this) Union9{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union9() {
        switch(this->tag){
            case 0: this->case0.~Union9_0(); break; // AA_Call
            case 1: this->case1.~Union9_1(); break; // AA_Fold
            case 2: this->case2.~Union9_2(); break; // AA_Raise
        }
        this->tag = 255;
    }
};
struct Tuple12 {
    Union2 v0;
    float v1;
    __device__ Tuple12() = default;
    __device__ Tuple12(Union2 t0, float t1) : v0(t0), v1(t1) {}
};
struct Tuple13 {
    int v1;
    bool v0;
    __device__ Tuple13() = default;
    __device__ Tuple13(bool t0, int t1) : v0(t0), v1(t1) {}
};
struct Tuple14 {
    int v0;
    int v1;
    int v2;
    __device__ Tuple14() = default;
    __device__ Tuple14(int t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
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
    __device__ Union10() {}
    __device__ Union10(Union10_0 t) : tag(0), case0(t) {} // Eq
    __device__ Union10(Union10_1 t) : tag(1), case1(t) {} // Gt
    __device__ Union10(Union10_2 t) : tag(2), case2(t) {} // Lt
    __device__ Union10(Union10 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union10_0(x.case0); break; // Eq
            case 1: new (&this->case1) Union10_1(x.case1); break; // Gt
            case 2: new (&this->case2) Union10_2(x.case2); break; // Lt
        }
    }
    __device__ Union10(Union10 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union10_0(std::move(x.case0)); break; // Eq
            case 1: new (&this->case1) Union10_1(std::move(x.case1)); break; // Gt
            case 2: new (&this->case2) Union10_2(std::move(x.case2)); break; // Lt
        }
    }
    __device__ Union10 & operator=(Union10 & x) {
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
    __device__ Union10 & operator=(Union10 && x) {
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
    __device__ ~Union10() {
        switch(this->tag){
            case 0: this->case0.~Union10_0(); break; // Eq
            case 1: this->case1.~Union10_1(); break; // Gt
            case 2: this->case2.~Union10_2(); break; // Lt
        }
        this->tag = 255;
    }
};
struct Tuple15 {
    int v0;
    int v1;
    unsigned char v2;
    __device__ Tuple15() = default;
    __device__ Tuple15(int t0, int t1, unsigned char t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Union11_0 { // None
};
struct Union11_1 { // Some
    static_array<unsigned char,5> v0;
    __device__ Union11_1(static_array<unsigned char,5> t0) : v0(t0) {}
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
struct Tuple16 {
    Union10 v1;
    int v0;
    __device__ Tuple16() = default;
    __device__ Tuple16(int t0, Union10 t1) : v0(t0), v1(t1) {}
};
struct Tuple17 {
    int v0;
    int v1;
    int v2;
    unsigned char v3;
    __device__ Tuple17() = default;
    __device__ Tuple17(int t0, int t1, int t2, unsigned char t3) : v0(t0), v1(t1), v2(t2), v3(t3) {}
};
struct Union12_0 { // None
};
struct Union12_1 { // Some
    static_array<unsigned char,4> v0;
    static_array<unsigned char,3> v1;
    __device__ Union12_1(static_array<unsigned char,4> t0, static_array<unsigned char,3> t1) : v0(t0), v1(t1) {}
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
    static_array<unsigned char,3> v0;
    static_array<unsigned char,4> v1;
    __device__ Union13_1(static_array<unsigned char,3> t0, static_array<unsigned char,4> t1) : v0(t0), v1(t1) {}
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
struct Union14_0 { // None
};
struct Union14_1 { // Some
    static_array<unsigned char,2> v0;
    static_array<unsigned char,2> v1;
    __device__ Union14_1(static_array<unsigned char,2> t0, static_array<unsigned char,2> t1) : v0(t0), v1(t1) {}
    __device__ Union14_1() = delete;
};
struct Union14 {
    union {
        Union14_0 case0; // None
        Union14_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union14() {}
    __device__ Union14(Union14_0 t) : tag(0), case0(t) {} // None
    __device__ Union14(Union14_1 t) : tag(1), case1(t) {} // Some
    __device__ Union14(Union14 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union14_0(x.case0); break; // None
            case 1: new (&this->case1) Union14_1(x.case1); break; // Some
        }
    }
    __device__ Union14(Union14 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union14_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union14_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union14 & operator=(Union14 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union14();
            new (this) Union14{x};
        }
        return *this;
    }
    __device__ Union14 & operator=(Union14 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union14();
            new (this) Union14{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union14() {
        switch(this->tag){
            case 0: this->case0.~Union14_0(); break; // None
            case 1: this->case1.~Union14_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union15_0 { // None
};
struct Union15_1 { // Some
    static_array<unsigned char,2> v0;
    static_array<unsigned char,5> v1;
    __device__ Union15_1(static_array<unsigned char,2> t0, static_array<unsigned char,5> t1) : v0(t0), v1(t1) {}
    __device__ Union15_1() = delete;
};
struct Union15 {
    union {
        Union15_0 case0; // None
        Union15_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union15() {}
    __device__ Union15(Union15_0 t) : tag(0), case0(t) {} // None
    __device__ Union15(Union15_1 t) : tag(1), case1(t) {} // Some
    __device__ Union15(Union15 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union15_0(x.case0); break; // None
            case 1: new (&this->case1) Union15_1(x.case1); break; // Some
        }
    }
    __device__ Union15(Union15 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union15_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union15_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union15 & operator=(Union15 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union15();
            new (this) Union15{x};
        }
        return *this;
    }
    __device__ Union15 & operator=(Union15 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union15();
            new (this) Union15{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union15() {
        switch(this->tag){
            case 0: this->case0.~Union15_0(); break; // None
            case 1: this->case1.~Union15_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Union16_0 { // None
};
struct Union16_1 { // Some
    static_array<unsigned char,2> v0;
    static_array<unsigned char,3> v1;
    __device__ Union16_1(static_array<unsigned char,2> t0, static_array<unsigned char,3> t1) : v0(t0), v1(t1) {}
    __device__ Union16_1() = delete;
};
struct Union16 {
    union {
        Union16_0 case0; // None
        Union16_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union16() {}
    __device__ Union16(Union16_0 t) : tag(0), case0(t) {} // None
    __device__ Union16(Union16_1 t) : tag(1), case1(t) {} // Some
    __device__ Union16(Union16 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union16_0(x.case0); break; // None
            case 1: new (&this->case1) Union16_1(x.case1); break; // Some
        }
    }
    __device__ Union16(Union16 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union16_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union16_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union16 & operator=(Union16 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union16();
            new (this) Union16{x};
        }
        return *this;
    }
    __device__ Union16 & operator=(Union16 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union16();
            new (this) Union16{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union16() {
        switch(this->tag){
            case 0: this->case0.~Union16_0(); break; // None
            case 1: this->case1.~Union16_1(); break; // Some
        }
        this->tag = 255;
    }
};
struct Tuple18 {
    double v1;
    int v0;
    __device__ Tuple18() = default;
    __device__ Tuple18(int t0, double t1) : v0(t0), v1(t1) {}
};
struct Closure8 {
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
    v1 = v0 < 524288;
    return v1;
}
__device__ inline bool while_method_8(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ inline bool while_method_9(int v0){
    bool v1;
    v1 = v0 < 10;
    return v1;
}
__device__ void method_11(unsigned int v0, float * v1, int v2){
    unsigned int v3;
    v3 = v0 + 1u;
    bool v4;
    v4 = v3 == 0u;
    bool v5;
    v5 = v4 != true;
    bool v6;
    v6 = v5 == false;
    if (v6){
        assert("Pickle failure. The input is too large in the binary serializer." && v5);
    } else {
    }
    int v8; unsigned int v9;
    Tuple7 tmp12 = Tuple7{0, v3};
    v8 = tmp12.v0; v9 = tmp12.v1;
    while (while_method_9(v8)){
        unsigned int v11;
        v11 = v9 & 1u;
        int v12;
        v12 = v2 + v8;
        float v13;
        v13 = (float)v11;
        v1[v12] = v13;
        unsigned int v14;
        v14 = v9 >> 1;
        v9 = v14;
        v8 += 1 ;
    }
    bool v15;
    v15 = v9 == 0u;
    bool v16;
    v16 = v15 == false;
    if (v16){
        assert("Picke failure. The remains of the input has to equal zero in the binary pickler." && v15);
        return ;
    } else {
        return ;
    }
}
__device__ inline bool while_method_10(int v0){
    bool v1;
    v1 = v0 < 11;
    return v1;
}
__device__ void method_12(unsigned int v0, float * v1, int v2){
    unsigned int v3;
    v3 = v0 + 1u;
    bool v4;
    v4 = v3 == 0u;
    bool v5;
    v5 = v4 != true;
    bool v6;
    v6 = v5 == false;
    if (v6){
        assert("Pickle failure. The input is too large in the binary serializer." && v5);
    } else {
    }
    int v8; unsigned int v9;
    Tuple7 tmp13 = Tuple7{0, v3};
    v8 = tmp13.v0; v9 = tmp13.v1;
    while (while_method_10(v8)){
        unsigned int v11;
        v11 = v9 & 1u;
        int v12;
        v12 = v2 + v8;
        float v13;
        v13 = (float)v11;
        v1[v12] = v13;
        unsigned int v14;
        v14 = v9 >> 1;
        v9 = v14;
        v8 += 1 ;
    }
    bool v15;
    v15 = v9 == 0u;
    bool v16;
    v16 = v15 == false;
    if (v16){
        assert("Picke failure. The remains of the input has to equal zero in the binary pickler." && v15);
        return ;
    } else {
        return ;
    }
}
__device__ inline bool while_method_11(int v0){
    bool v1;
    v1 = v0 < 32;
    return v1;
}
__device__ inline bool while_method_12(int v0){
    bool v1;
    v1 = v0 < 16;
    return v1;
}
__device__ inline bool while_method_13(int v0){
    bool v1;
    v1 = v0 < 32;
    return v1;
}
__device__ void method_13(unsigned int * v0, int v1, float * v2){
    int v3;
    v3 = blockIdx.x;
    assert("Tensor range check" && 0 <= v3 && v3 < 24);
    int v4;
    v4 = 32768 * v3;
    int v5;
    v5 = blockIdx.x;
    assert("Tensor range check" && 0 <= v5 && v5 < 24);
    int v6;
    v6 = 256 * v5;
    int v7;
    v7 = v6 + v1;
    int v8;
    v8 = threadIdx.x;
    bool v9;
    v9 = 0 <= v8;
    bool v10;
    v10 = v9 == false;
    if (v10){
        assert("The index needs to be zero or positive." && v9);
    } else {
    }
    int v12;
    v12 = v8 % 32;
    int v13;
    v13 = v8 / 32;
    bool v14;
    v14 = v13 < 8;
    bool v15;
    v15 = v14 == false;
    if (v15){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v14);
    } else {
    }
    assert("Tensor range check" && 0 <= v13 && v13 < 8);
    assert("Tensor range check" && 0 <= v12 && v12 < 32);
    int v17;
    v17 = 4 * v12;
    int v18;
    v18 = v17 + v4;
    int v19;
    v19 = 128 * v13;
    int v20;
    v20 = v19 + v18;
    assert("Tensor range check" && 0 <= v13 && v13 < 8);
    int v21;
    v21 = v13 + v7;
    int v22;
    v22 = 0;
    while (while_method_13(v22)){
        assert("Tensor range check" && 0 <= v22 && v22 < 32);
        int v24;
        v24 = 1024 * v22;
        int v25;
        v25 = v24 + v20;
        float v26[4];
        int v27[4];
        int v28;
        v28 = 0;
        while (while_method_6(v28)){
            assert("Tensor range check" && 0 <= v28 && v28 < 1);
            int v30;
            v30 = 4 * v28;
            assert("Tensor range check" && 0 <= v28 && v28 < 1);
            int v31;
            v31 = 128 * v28;
            int v32;
            v32 = v31 + v25;
            int4* v33;
            v33 = reinterpret_cast<int4*>(v2 + v32);
            int4* v34;
            v34 = reinterpret_cast<int4*>(v26 + v30);
            assert("Pointer alignment check" && (unsigned long long)(v33) % 4 == 0 && (unsigned long long)(v34) % 4 == 0);
            *v34 = *v33;
            v28 += 1 ;
        }
        int v35;
        v35 = 0;
        while (while_method_6(v35)){
            int v37;
            v37 = 0;
            while (while_method_0(v37)){
                bool v39;
                v39 = 0 <= v37;
                bool v41;
                if (v39){
                    bool v40;
                    v40 = v37 < 4;
                    v41 = v40;
                } else {
                    v41 = false;
                }
                bool v42;
                v42 = v41 == false;
                if (v42){
                    assert("The indices should be inside the range of the dimension." && v41);
                } else {
                }
                bool v44;
                v44 = 0 <= v12;
                bool v46;
                if (v44){
                    bool v45;
                    v45 = v12 < 32;
                    v46 = v45;
                } else {
                    v46 = false;
                }
                bool v47;
                v47 = v46 == false;
                if (v47){
                    assert("The indices should be inside the range of the dimension." && v46);
                } else {
                }
                int v49;
                v49 = v12 * 4;
                int v50;
                v50 = v37 + v49;
                bool v51;
                v51 = 0 <= v35;
                bool v53;
                if (v51){
                    bool v52;
                    v52 = v35 < 1;
                    v53 = v52;
                } else {
                    v53 = false;
                }
                bool v54;
                v54 = v53 == false;
                if (v54){
                    assert("The indices should be inside the range of the dimension." && v53);
                } else {
                }
                int v56;
                v56 = v35 * 128;
                int v57;
                v57 = v50 + v56;
                assert("Tensor range check" && 0 <= v35 && v35 < 1);
                assert("Tensor range check" && 0 <= v37 && v37 < 4);
                int v58;
                v58 = 4 * v35;
                int v59;
                v59 = v58 + v37;
                v27[v59] = v57;
                v37 += 1 ;
            }
            v35 += 1 ;
        }
        bool v60;
        v60 = 0 <= v13;
        bool v61;
        v61 = v60 && v14;
        bool v62;
        v62 = v61 == false;
        if (v62){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v61);
        } else {
        }
        bool v64;
        v64 = 0 <= v22;
        bool v66;
        if (v64){
            bool v65;
            v65 = v22 < 32;
            v66 = v65;
        } else {
            v66 = false;
        }
        bool v67;
        v67 = v66 == false;
        if (v67){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v66);
        } else {
        }
        int v69;
        v69 = v22 * 8;
        int v70;
        v70 = v69 + v13;
        unsigned int v71[4];
        int v72;
        v72 = 0;
        while (while_method_6(v72)){
            int v74;
            v74 = 0;
            while (while_method_0(v74)){
                assert("Tensor range check" && 0 <= v72 && v72 < 1);
                assert("Tensor range check" && 0 <= v74 && v74 < 4);
                int v76;
                v76 = 4 * v72;
                int v77;
                v77 = v76 + v74;
                float v78;
                v78 = v26[v77];
                int v79;
                v79 = v27[v77];
                bool v80;
                v80 = v78 <= 0.0f;
                unsigned int v82;
                if (v80){
                    v82 = 0u;
                } else {
                    unsigned int v81;
                    v81 = 1u << v79;
                    v82 = v81;
                }
                assert("Tensor range check" && 0 <= v72 && v72 < 1);
                assert("Tensor range check" && 0 <= v74 && v74 < 4);
                v71[v77] = v82;
                v74 += 1 ;
            }
            v72 += 1 ;
        }
        unsigned int v83;
        v83 = 0u;
        int v84;
        v84 = 0;
        while (while_method_6(v84)){
            int v86;
            v86 = 0;
            while (while_method_0(v86)){
                assert("Tensor range check" && 0 <= v84 && v84 < 1);
                assert("Tensor range check" && 0 <= v86 && v86 < 4);
                int v88;
                v88 = 4 * v84;
                int v89;
                v89 = v88 + v86;
                unsigned int v90;
                v90 = v71[v89];
                unsigned int v91;
                v91 = v83 | v90;
                v83 = v91;
                v86 += 1 ;
            }
            v84 += 1 ;
        }
        auto v92 = cooperative_groups::coalesced_threads();
        int v93;
        v93 = threadIdx.x;
        int v94;
        v94 = v93 / 32;
        auto v95 = cooperative_groups::labeled_partition(v92,v94);
        Closure0 v96{};
        unsigned int v97;
        v97 = cooperative_groups::reduce(v95, v83, v96);
        unsigned int v98;
        v98 = v97 % 4096u;
        assert("Tensor range check" && 0 <= v22 && v22 < 32);
        int v99;
        v99 = 8 * v22;
        int v100;
        v100 = v99 + v21;
        v0[v100] = v98;
        v22 += 1 ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0));
    return ;
}
__device__ Tuple8 method_14(curandStatePhilox4_32_10_t & v0, int * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int v9, int v10){
    assert("Tensor range check" && 0 <= v10 && v10 < 4);
    int v11;
    v11 = 65536 * v10;
    assert("Tensor range check" && 0 <= v9 && v9 < 4096);
    int v12;
    v12 = 16 * v9;
    int v13;
    v13 = v12 + v11;
    float * v14;
    v14 = v2+v13;
    float * v16;
    v16 = v3+v13;
    int v18;
    v18 = sizeof(float *);
    unsigned long long v19;
    v19 = (unsigned long long)v18;
    unsigned long long v20;
    v20 = 256ull * v19;
    unsigned long long v21;
    v21 = v20 + 16ull;
    unsigned long long v22;
    v22 = v21 - 1ull;
    unsigned long long v23;
    v23 = v22 % 16ull;
    unsigned long long v24;
    v24 = v22 - v23;
    unsigned long long v25;
    v25 = v24 + v20;
    unsigned long long v26;
    v26 = v25 + 16ull;
    unsigned long long v27;
    v27 = v26 - 1ull;
    unsigned long long v28;
    v28 = v27 % 16ull;
    unsigned long long v29;
    v29 = v27 - v28;
    unsigned long long v30;
    v30 = v29 + 1024ull;
    unsigned long long v31;
    v31 = v30 + 16ull;
    unsigned long long v32;
    v32 = v31 - 1ull;
    unsigned long long v33;
    v33 = v32 % 16ull;
    unsigned long long v34;
    v34 = v32 - v33;
    unsigned long long v35;
    v35 = v34 + 1024ull;
    bool v36;
    v36 = v35 <= 98304ull;
    bool v37;
    v37 = v36 == false;
    if (v37){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v36);
    } else {
    }
    extern __shared__ unsigned char v39[];
    bool v40;
    v40 = v35 <= v35;
    bool v41;
    v41 = v40 == false;
    if (v41){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v40);
    } else {
    }
    float * * v43;
    v43 = reinterpret_cast<float * *>(&v39[0ull]);
    float * * v45;
    v45 = reinterpret_cast<float * *>(&v39[v24]);
    float * v47;
    v47 = reinterpret_cast<float *>(&v39[v29]);
    int * v49;
    v49 = reinterpret_cast<int *>(&v39[v34]);
    int v51;
    v51 = threadIdx.x;
    assert("Tensor range check" && 0 <= v51 && v51 < 256);
    v43[v51] = v14;
    v45[v51] = v16;
    asm("barrier.cta.sync %0;" :: "r"(0));
    bool v52;
    v52 = 0 <= v51;
    bool v53;
    v53 = v52 == false;
    if (v53){
        assert("The index needs to be zero or positive." && v52);
    } else {
    }
    int v55;
    v55 = v51 % 4;
    int v56;
    v56 = v51 / 4;
    bool v57;
    v57 = v56 < 64;
    bool v58;
    v58 = v57 == false;
    if (v58){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v57);
    } else {
    }
    assert("Tensor range check" && 0 <= v56 && v56 < 64);
    int v60;
    v60 = 0;
    while (while_method_0(v60)){
        bool v62;
        v62 = 0 <= v56;
        bool v63;
        v63 = v62 && v57;
        bool v64;
        v64 = v63 == false;
        if (v64){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v63);
        } else {
        }
        bool v66;
        v66 = 0 <= v60;
        bool v68;
        if (v66){
            bool v67;
            v67 = v60 < 4;
            v68 = v67;
        } else {
            v68 = false;
        }
        bool v69;
        v69 = v68 == false;
        if (v69){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v68);
        } else {
        }
        int v71;
        v71 = v60 * 64;
        int v72;
        v72 = v71 + v56;
        assert("Tensor range check" && 0 <= v60 && v60 < 4);
        int v73;
        v73 = 64 * v60;
        int v74;
        v74 = v73 + v56;
        float * v75;
        v75 = v43[v74];
        float * v76;
        v76 = v45[v74];
        int v77;
        v77 = blockIdx.x;
        int v78;
        v78 = v77 * 256;
        int v79;
        v79 = v78 + v72;
        assert("Tensor range check" && 0 <= v55 && v55 < 4);
        int v80;
        v80 = 4 * v55;
        float v81[4];
        float v82[4];
        int v83[4];
        int v84;
        v84 = 0;
        while (while_method_6(v84)){
            assert("Tensor range check" && 0 <= v84 && v84 < 1);
            int v86;
            v86 = 4 * v84;
            assert("Tensor range check" && 0 <= v84 && v84 < 1);
            int v87;
            v87 = 16 * v84;
            int v88;
            v88 = v87 + v80;
            int4* v89;
            v89 = reinterpret_cast<int4*>(v75 + v88);
            int4* v90;
            v90 = reinterpret_cast<int4*>(v81 + v86);
            assert("Pointer alignment check" && (unsigned long long)(v89) % 4 == 0 && (unsigned long long)(v90) % 4 == 0);
            *v90 = *v89;
            int4* v91;
            v91 = reinterpret_cast<int4*>(v76 + v88);
            int4* v92;
            v92 = reinterpret_cast<int4*>(v82 + v86);
            assert("Pointer alignment check" && (unsigned long long)(v91) % 4 == 0 && (unsigned long long)(v92) % 4 == 0);
            *v92 = *v91;
            v84 += 1 ;
        }
        int v93;
        v93 = 0;
        while (while_method_6(v93)){
            int v95;
            v95 = 0;
            while (while_method_0(v95)){
                bool v97;
                v97 = 0 <= v95;
                bool v99;
                if (v97){
                    bool v98;
                    v98 = v95 < 4;
                    v99 = v98;
                } else {
                    v99 = false;
                }
                bool v100;
                v100 = v99 == false;
                if (v100){
                    assert("The indices should be inside the range of the dimension." && v99);
                } else {
                }
                bool v102;
                v102 = 0 <= v55;
                bool v104;
                if (v102){
                    bool v103;
                    v103 = v55 < 4;
                    v104 = v103;
                } else {
                    v104 = false;
                }
                bool v105;
                v105 = v104 == false;
                if (v105){
                    assert("The indices should be inside the range of the dimension." && v104);
                } else {
                }
                int v107;
                v107 = v55 * 4;
                int v108;
                v108 = v95 + v107;
                bool v109;
                v109 = 0 <= v93;
                bool v111;
                if (v109){
                    bool v110;
                    v110 = v93 < 1;
                    v111 = v110;
                } else {
                    v111 = false;
                }
                bool v112;
                v112 = v111 == false;
                if (v112){
                    assert("The indices should be inside the range of the dimension." && v111);
                } else {
                }
                int v114;
                v114 = v93 * 16;
                int v115;
                v115 = v108 + v114;
                assert("Tensor range check" && 0 <= v93 && v93 < 1);
                assert("Tensor range check" && 0 <= v95 && v95 < 4);
                int v116;
                v116 = 4 * v93;
                int v117;
                v117 = v116 + v95;
                v83[v117] = v115;
                v95 += 1 ;
            }
            v93 += 1 ;
        }
        bool v118[4];
        int v119;
        v119 = 0;
        while (while_method_6(v119)){
            int v121;
            v121 = 0;
            while (while_method_0(v121)){
                assert("Tensor range check" && 0 <= v119 && v119 < 1);
                assert("Tensor range check" && 0 <= v121 && v121 < 4);
                int v123;
                v123 = 4 * v119;
                int v124;
                v124 = v123 + v121;
                float v125;
                v125 = v81[v124];
                int v126;
                v126 = v83[v124];
                bool v127;
                v127 = v126 < 11;
                assert("Tensor range check" && 0 <= v119 && v119 < 1);
                assert("Tensor range check" && 0 <= v121 && v121 < 4);
                v118[v124] = v127;
                v121 += 1 ;
            }
            v119 += 1 ;
        }
        float v128[4];
        int v129;
        v129 = 0;
        while (while_method_6(v129)){
            int v131;
            v131 = 0;
            while (while_method_0(v131)){
                assert("Tensor range check" && 0 <= v129 && v129 < 1);
                assert("Tensor range check" && 0 <= v131 && v131 < 4);
                int v133;
                v133 = 4 * v129;
                int v134;
                v134 = v133 + v131;
                float v135;
                v135 = v81[v134];
                bool v136;
                v136 = v118[v134];
                float v139;
                if (v136){
                    bool v137;
                    v137 = 0.0f >= v135;
                    if (v137){
                        v139 = 0.0f;
                    } else {
                        v139 = v135;
                    }
                } else {
                    v139 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v129 && v129 < 1);
                assert("Tensor range check" && 0 <= v131 && v131 < 4);
                v128[v134] = v139;
                v131 += 1 ;
            }
            v129 += 1 ;
        }
        float v140;
        v140 = 0.0f;
        int v141;
        v141 = 0;
        while (while_method_6(v141)){
            int v143;
            v143 = 0;
            while (while_method_0(v143)){
                assert("Tensor range check" && 0 <= v141 && v141 < 1);
                assert("Tensor range check" && 0 <= v143 && v143 < 4);
                int v145;
                v145 = 4 * v141;
                int v146;
                v146 = v145 + v143;
                float v147;
                v147 = v128[v146];
                float v148;
                v148 = v140 + v147;
                v140 = v148;
                v143 += 1 ;
            }
            v141 += 1 ;
        }
        auto v149 = cooperative_groups::coalesced_threads();
        int v150;
        v150 = threadIdx.x;
        int v151;
        v151 = v150 / 4;
        auto v152 = cooperative_groups::labeled_partition(v149,v151);
        Closure1 v153{};
        float v154;
        v154 = cooperative_groups::reduce(v152, v140, v153);
        int v155[4];
        int v156;
        v156 = 0;
        while (while_method_6(v156)){
            int v158;
            v158 = 0;
            while (while_method_0(v158)){
                assert("Tensor range check" && 0 <= v156 && v156 < 1);
                assert("Tensor range check" && 0 <= v158 && v158 < 4);
                int v160;
                v160 = 4 * v156;
                int v161;
                v161 = v160 + v158;
                bool v162;
                v162 = v118[v161];
                int v163;
                if (v162){
                    v163 = 1;
                } else {
                    v163 = 0;
                }
                assert("Tensor range check" && 0 <= v156 && v156 < 1);
                assert("Tensor range check" && 0 <= v158 && v158 < 4);
                v155[v161] = v163;
                v158 += 1 ;
            }
            v156 += 1 ;
        }
        int v164;
        v164 = 0;
        int v165;
        v165 = 0;
        while (while_method_6(v165)){
            int v167;
            v167 = 0;
            while (while_method_0(v167)){
                assert("Tensor range check" && 0 <= v165 && v165 < 1);
                assert("Tensor range check" && 0 <= v167 && v167 < 4);
                int v169;
                v169 = 4 * v165;
                int v170;
                v170 = v169 + v167;
                int v171;
                v171 = v155[v170];
                int v172;
                v172 = v164 + v171;
                v164 = v172;
                v167 += 1 ;
            }
            v165 += 1 ;
        }
        auto v173 = cooperative_groups::coalesced_threads();
        int v174;
        v174 = threadIdx.x;
        int v175;
        v175 = v174 / 4;
        auto v176 = cooperative_groups::labeled_partition(v173,v175);
        Closure2 v177{};
        int v178;
        v178 = cooperative_groups::reduce(v176, v164, v177);
        float v179;
        v179 = (float)v178;
        float v180;
        v180 = 1.0f / v179;
        float v181[4];
        int v182;
        v182 = 0;
        while (while_method_6(v182)){
            int v184;
            v184 = 0;
            while (while_method_0(v184)){
                assert("Tensor range check" && 0 <= v182 && v182 < 1);
                assert("Tensor range check" && 0 <= v184 && v184 < 4);
                int v186;
                v186 = 4 * v182;
                int v187;
                v187 = v186 + v184;
                float v188;
                v188 = v128[v187];
                bool v189;
                v189 = v118[v187];
                bool v190;
                v190 = v189 == false;
                float v195;
                if (v190){
                    v195 = 0.0f;
                } else {
                    bool v191;
                    v191 = v154 == 0.0f;
                    bool v192;
                    v192 = v191 != true;
                    if (v192){
                        float v193;
                        v193 = v188 / v154;
                        v195 = v193;
                    } else {
                        v195 = v180;
                    }
                }
                assert("Tensor range check" && 0 <= v182 && v182 < 1);
                assert("Tensor range check" && 0 <= v184 && v184 < 4);
                v181[v187] = v195;
                v184 += 1 ;
            }
            v182 += 1 ;
        }
        float v196[4];
        float v197;
        v197 = 0.0f;
        int v198;
        v198 = 0;
        while (while_method_6(v198)){
            assert("Tensor range check" && 0 <= v198 && v198 < 1);
            int v200;
            v200 = 4 * v198;
            assert("Tensor range check" && 0 <= v198 && v198 < 1);
            int v201; float v202;
            Tuple9 tmp14 = Tuple9{0, 0.0f};
            v201 = tmp14.v0; v202 = tmp14.v1;
            while (while_method_0(v201)){
                assert("Tensor range check" && 0 <= v201 && v201 < 4);
                int v204;
                v204 = v201 + v200;
                float v205;
                v205 = v181[v204];
                float v206;
                v206 = v202 + v205;
                v202 = v206;
                v201 += 1 ;
            }
            auto v207 = cooperative_groups::coalesced_threads();
            int v208;
            v208 = threadIdx.x;
            int v209;
            v209 = v208 / 4;
            auto v210 = cooperative_groups::labeled_partition(v207,v209);
            Closure3 v211{};
            float v212;
            v212 = cooperative_groups::inclusive_scan(v210, v202, v211);
            float v213;
            v213 = v210.shfl_up(v212,1);
            bool v214;
            v214 = v210.thread_rank() == 0;
            float v215;
            if (v214){
                v215 = 0.0f;
            } else {
                v215 = v213;
            }
            float v216;
            v216 = v210.shfl(v212,v210.num_threads()-1);
            float v217;
            v217 = v197 + v215;
            int v218; float v219;
            Tuple9 tmp15 = Tuple9{0, v217};
            v218 = tmp15.v0; v219 = tmp15.v1;
            while (while_method_0(v218)){
                assert("Tensor range check" && 0 <= v218 && v218 < 4);
                int v221;
                v221 = v218 + v200;
                float v222;
                v222 = v181[v221];
                float v223;
                v223 = v219 + v222;
                assert("Tensor range check" && 0 <= v218 && v218 < 4);
                v196[v221] = v223;
                v219 = v223;
                v218 += 1 ;
            }
            float v224;
            v224 = v197 + v216;
            v197 = v224;
            v198 += 1 ;
        }
        float v225[4];
        bool v226[4];
        int v227;
        v227 = 0;
        while (while_method_6(v227)){
            int v229;
            v229 = 0;
            while (while_method_0(v229)){
                assert("Tensor range check" && 0 <= v227 && v227 < 1);
                assert("Tensor range check" && 0 <= v229 && v229 < 4);
                int v231;
                v231 = 4 * v227;
                int v232;
                v232 = v231 + v229;
                float v233;
                v233 = v196[v232];
                float v234;
                v234 = v181[v232];
                bool v235;
                v235 = v234 > 0.0f;
                assert("Tensor range check" && 0 <= v227 && v227 < 1);
                assert("Tensor range check" && 0 <= v229 && v229 < 4);
                v225[v232] = v233;
                v226[v232] = v235;
                v229 += 1 ;
            }
            v227 += 1 ;
        }
        float v236; bool v237;
        Tuple10 tmp16 = Tuple10{-1.0f / 0.0f, false};
        v236 = tmp16.v0; v237 = tmp16.v1;
        int v238;
        v238 = 0;
        while (while_method_6(v238)){
            int v240;
            v240 = 0;
            while (while_method_0(v240)){
                assert("Tensor range check" && 0 <= v238 && v238 < 1);
                assert("Tensor range check" && 0 <= v240 && v240 < 4);
                int v242;
                v242 = 4 * v238;
                int v243;
                v243 = v242 + v240;
                float v244;
                v244 = v225[v243];
                bool v245;
                v245 = v226[v243];
                float v252; bool v253;
                if (v237){
                    if (v245){
                        bool v246;
                        v246 = v236 >= v244;
                        float v247;
                        if (v246){
                            v247 = v236;
                        } else {
                            v247 = v244;
                        }
                        v252 = v247; v253 = true;
                    } else {
                        v252 = v236; v253 = v237;
                    }
                } else {
                    if (v245){
                        v252 = v244; v253 = v245;
                    } else {
                        v252 = v236; v253 = v237;
                    }
                }
                v236 = v252;
                v237 = v253;
                v240 += 1 ;
            }
            v238 += 1 ;
        }
        auto v254 = cooperative_groups::coalesced_threads();
        int v255;
        v255 = threadIdx.x;
        int v256;
        v256 = v255 / 4;
        auto v257 = cooperative_groups::labeled_partition(v254,v256);
        Closure4 v258{};
        float v259; bool v260;
        Tuple10 tmp17 = cooperative_groups::reduce(v257, Tuple10{v236, v237}, v258);
        v259 = tmp17.v0; v260 = tmp17.v1;
        bool v261;
        v261 = v260 == false;
        if (v261){
            assert("The local reduce must be true." && v260);
        } else {
        }
        float v263[4];
        int v264[4];
        int v265;
        v265 = 0;
        while (while_method_6(v265)){
            int v267;
            v267 = 0;
            while (while_method_0(v267)){
                assert("Tensor range check" && 0 <= v265 && v265 < 1);
                assert("Tensor range check" && 0 <= v267 && v267 < 4);
                int v269;
                v269 = 4 * v265;
                int v270;
                v270 = v269 + v267;
                int v271;
                v271 = v83[v270];
                float v272;
                v272 = curand_uniform(&v0);
                assert("Tensor range check" && 0 <= v265 && v265 < 1);
                assert("Tensor range check" && 0 <= v267 && v267 < 4);
                v263[v270] = v272;
                v264[v270] = v271;
                v267 += 1 ;
            }
            v265 += 1 ;
        }
        float v273; int v274;
        Tuple8 tmp18 = Tuple8{0.0f, 2147483647};
        v273 = tmp18.v0; v274 = tmp18.v1;
        int v275;
        v275 = 0;
        while (while_method_6(v275)){
            int v277;
            v277 = 0;
            while (while_method_0(v277)){
                assert("Tensor range check" && 0 <= v275 && v275 < 1);
                assert("Tensor range check" && 0 <= v277 && v277 < 4);
                int v279;
                v279 = 4 * v275;
                int v280;
                v280 = v279 + v277;
                float v281;
                v281 = v263[v280];
                int v282;
                v282 = v264[v280];
                bool v283;
                v283 = v274 < v282;
                float v284; int v285;
                if (v283){
                    v284 = v273; v285 = v274;
                } else {
                    v284 = v281; v285 = v282;
                }
                v273 = v284;
                v274 = v285;
                v277 += 1 ;
            }
            v275 += 1 ;
        }
        auto v286 = cooperative_groups::coalesced_threads();
        int v287;
        v287 = threadIdx.x;
        int v288;
        v288 = v287 / 4;
        auto v289 = cooperative_groups::labeled_partition(v286,v288);
        Closure5 v290{};
        float v291; int v292;
        Tuple8 tmp19 = cooperative_groups::reduce(v289, Tuple8{v273, v274}, v290);
        v291 = tmp19.v0; v292 = tmp19.v1;
        float v293;
        v293 = v259 * v291;
        int v294[4];
        bool v295[4];
        int v296;
        v296 = 0;
        while (while_method_6(v296)){
            int v298;
            v298 = 0;
            while (while_method_0(v298)){
                assert("Tensor range check" && 0 <= v296 && v296 < 1);
                assert("Tensor range check" && 0 <= v298 && v298 < 4);
                int v300;
                v300 = 4 * v296;
                int v301;
                v301 = v300 + v298;
                float v302;
                v302 = v225[v301];
                bool v303;
                v303 = v226[v301];
                int v304;
                v304 = v83[v301];
                int v307; bool v308;
                if (v303){
                    float v305;
                    v305 = v302 - v293;
                    bool v306;
                    v306 = v305 >= 0.0f;
                    v307 = v304; v308 = v306;
                } else {
                    v307 = 2147483647; v308 = false;
                }
                assert("Tensor range check" && 0 <= v296 && v296 < 1);
                assert("Tensor range check" && 0 <= v298 && v298 < 4);
                v294[v301] = v307;
                v295[v301] = v308;
                v298 += 1 ;
            }
            v296 += 1 ;
        }
        int v309; bool v310;
        Tuple11 tmp20 = Tuple11{2147483647, false};
        v309 = tmp20.v0; v310 = tmp20.v1;
        int v311;
        v311 = 0;
        while (while_method_6(v311)){
            int v313;
            v313 = 0;
            while (while_method_0(v313)){
                assert("Tensor range check" && 0 <= v311 && v311 < 1);
                assert("Tensor range check" && 0 <= v313 && v313 < 4);
                int v315;
                v315 = 4 * v311;
                int v316;
                v316 = v315 + v313;
                int v317;
                v317 = v294[v316];
                bool v318;
                v318 = v295[v316];
                int v325; bool v326;
                if (v310){
                    if (v318){
                        bool v319;
                        v319 = v309 < v317;
                        int v320;
                        if (v319){
                            v320 = v309;
                        } else {
                            v320 = v317;
                        }
                        v325 = v320; v326 = true;
                    } else {
                        v325 = v309; v326 = v310;
                    }
                } else {
                    if (v318){
                        v325 = v317; v326 = v318;
                    } else {
                        v325 = v309; v326 = v310;
                    }
                }
                v309 = v325;
                v310 = v326;
                v313 += 1 ;
            }
            v311 += 1 ;
        }
        auto v327 = cooperative_groups::coalesced_threads();
        int v328;
        v328 = threadIdx.x;
        int v329;
        v329 = v328 / 4;
        auto v330 = cooperative_groups::labeled_partition(v327,v329);
        Closure6 v331{};
        int v332; bool v333;
        Tuple11 tmp21 = cooperative_groups::reduce(v330, Tuple11{v309, v310}, v331);
        v332 = tmp21.v0; v333 = tmp21.v1;
        bool v334;
        v334 = v333 == false;
        if (v334){
            assert("The local reduce must be true." && v333);
        } else {
        }
        bool v336[4];
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
                float v343;
                v343 = v82[v342];
                int v344;
                v344 = v83[v342];
                bool v345;
                v345 = v344 < 11;
                assert("Tensor range check" && 0 <= v337 && v337 < 1);
                assert("Tensor range check" && 0 <= v339 && v339 < 4);
                v336[v342] = v345;
                v339 += 1 ;
            }
            v337 += 1 ;
        }
        float v346[4];
        int v347;
        v347 = 0;
        while (while_method_6(v347)){
            int v349;
            v349 = 0;
            while (while_method_0(v349)){
                assert("Tensor range check" && 0 <= v347 && v347 < 1);
                assert("Tensor range check" && 0 <= v349 && v349 < 4);
                int v351;
                v351 = 4 * v347;
                int v352;
                v352 = v351 + v349;
                float v353;
                v353 = v82[v352];
                bool v354;
                v354 = v336[v352];
                float v357;
                if (v354){
                    bool v355;
                    v355 = 0.0f >= v353;
                    if (v355){
                        v357 = 0.0f;
                    } else {
                        v357 = v353;
                    }
                } else {
                    v357 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v347 && v347 < 1);
                assert("Tensor range check" && 0 <= v349 && v349 < 4);
                v346[v352] = v357;
                v349 += 1 ;
            }
            v347 += 1 ;
        }
        float v358;
        v358 = 0.0f;
        int v359;
        v359 = 0;
        while (while_method_6(v359)){
            int v361;
            v361 = 0;
            while (while_method_0(v361)){
                assert("Tensor range check" && 0 <= v359 && v359 < 1);
                assert("Tensor range check" && 0 <= v361 && v361 < 4);
                int v363;
                v363 = 4 * v359;
                int v364;
                v364 = v363 + v361;
                float v365;
                v365 = v346[v364];
                float v366;
                v366 = v358 + v365;
                v358 = v366;
                v361 += 1 ;
            }
            v359 += 1 ;
        }
        auto v367 = cooperative_groups::coalesced_threads();
        int v368;
        v368 = threadIdx.x;
        int v369;
        v369 = v368 / 4;
        auto v370 = cooperative_groups::labeled_partition(v367,v369);
        float v371;
        v371 = cooperative_groups::reduce(v370, v358, v153);
        int v372[4];
        int v373;
        v373 = 0;
        while (while_method_6(v373)){
            int v375;
            v375 = 0;
            while (while_method_0(v375)){
                assert("Tensor range check" && 0 <= v373 && v373 < 1);
                assert("Tensor range check" && 0 <= v375 && v375 < 4);
                int v377;
                v377 = 4 * v373;
                int v378;
                v378 = v377 + v375;
                bool v379;
                v379 = v336[v378];
                int v380;
                if (v379){
                    v380 = 1;
                } else {
                    v380 = 0;
                }
                assert("Tensor range check" && 0 <= v373 && v373 < 1);
                assert("Tensor range check" && 0 <= v375 && v375 < 4);
                v372[v378] = v380;
                v375 += 1 ;
            }
            v373 += 1 ;
        }
        int v381;
        v381 = 0;
        int v382;
        v382 = 0;
        while (while_method_6(v382)){
            int v384;
            v384 = 0;
            while (while_method_0(v384)){
                assert("Tensor range check" && 0 <= v382 && v382 < 1);
                assert("Tensor range check" && 0 <= v384 && v384 < 4);
                int v386;
                v386 = 4 * v382;
                int v387;
                v387 = v386 + v384;
                int v388;
                v388 = v372[v387];
                int v389;
                v389 = v381 + v388;
                v381 = v389;
                v384 += 1 ;
            }
            v382 += 1 ;
        }
        auto v390 = cooperative_groups::coalesced_threads();
        int v391;
        v391 = threadIdx.x;
        int v392;
        v392 = v391 / 4;
        auto v393 = cooperative_groups::labeled_partition(v390,v392);
        int v394;
        v394 = cooperative_groups::reduce(v393, v381, v177);
        float v395;
        v395 = (float)v394;
        float v396;
        v396 = 1.0f / v395;
        float v397[4];
        int v398;
        v398 = 0;
        while (while_method_6(v398)){
            int v400;
            v400 = 0;
            while (while_method_0(v400)){
                assert("Tensor range check" && 0 <= v398 && v398 < 1);
                assert("Tensor range check" && 0 <= v400 && v400 < 4);
                int v402;
                v402 = 4 * v398;
                int v403;
                v403 = v402 + v400;
                float v404;
                v404 = v346[v403];
                bool v405;
                v405 = v336[v403];
                bool v406;
                v406 = v405 == false;
                float v411;
                if (v406){
                    v411 = 0.0f;
                } else {
                    bool v407;
                    v407 = v371 == 0.0f;
                    bool v408;
                    v408 = v407 != true;
                    if (v408){
                        float v409;
                        v409 = v404 / v371;
                        v411 = v409;
                    } else {
                        v411 = v396;
                    }
                }
                assert("Tensor range check" && 0 <= v398 && v398 < 1);
                assert("Tensor range check" && 0 <= v400 && v400 < 4);
                v397[v403] = v411;
                v400 += 1 ;
            }
            v398 += 1 ;
        }
        float v412; int v413;
        Tuple8 tmp22 = Tuple8{0.0f, 2147483647};
        v412 = tmp22.v0; v413 = tmp22.v1;
        int v414;
        v414 = 0;
        while (while_method_6(v414)){
            int v416;
            v416 = 0;
            while (while_method_0(v416)){
                assert("Tensor range check" && 0 <= v414 && v414 < 1);
                assert("Tensor range check" && 0 <= v416 && v416 < 4);
                int v418;
                v418 = 4 * v414;
                int v419;
                v419 = v418 + v416;
                float v420;
                v420 = v181[v419];
                int v421;
                v421 = v83[v419];
                bool v422;
                v422 = v413 == v332;
                float v426; int v427;
                if (v422){
                    v426 = v412; v427 = v413;
                } else {
                    bool v423;
                    v423 = v421 == v332;
                    if (v423){
                        v426 = v420; v427 = v421;
                    } else {
                        v426 = v412; v427 = v413;
                    }
                }
                v412 = v426;
                v413 = v427;
                v416 += 1 ;
            }
            v414 += 1 ;
        }
        auto v428 = cooperative_groups::coalesced_threads();
        int v429;
        v429 = threadIdx.x;
        int v430;
        v430 = v429 / 4;
        auto v431 = cooperative_groups::labeled_partition(v428,v430);
        Closure7 v432{v332};
        float v433; int v434;
        Tuple8 tmp23 = cooperative_groups::reduce(v431, Tuple8{v412, v413}, v432);
        v433 = tmp23.v0; v434 = tmp23.v1;
        bool v435;
        v435 = v434 == 2147483647;
        bool v436;
        v436 = v435 != true;
        bool v437;
        v437 = v436 == false;
        if (v437){
            assert("Expected a valid action id in get_action." && v436);
        } else {
        }
        int v439;
        v439 = 0;
        while (while_method_6(v439)){
            assert("Tensor range check" && 0 <= v439 && v439 < 1);
            assert("Tensor range check" && 0 <= v439 && v439 < 1);
            v439 += 1 ;
        }
        assert("Tensor range check" && 0 <= v72 && v72 < 256);
        v47[v72] = v433;
        v49[v72] = v332;
        v60 += 1 ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0));
    assert("Tensor range check" && 0 <= v51 && v51 < 256);
    float v441;
    v441 = v47[v51];
    int v442;
    v442 = v49[v51];
    asm("barrier.cta.sync %0;" :: "r"(0));
    return Tuple8{v441, v442};
}
__device__ float method_15(int * v0, float * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, int v8, int v9, int v10){
    assert("Tensor range check" && 0 <= v9 && v9 < 4);
    int v11;
    v11 = 65536 * v9;
    assert("Tensor range check" && 0 <= v8 && v8 < 4096);
    int v12;
    v12 = 16 * v8;
    int v13;
    v13 = v12 + v11;
    float * v14;
    v14 = v2+v13;
    int v16;
    v16 = sizeof(float *);
    unsigned long long v17;
    v17 = (unsigned long long)v16;
    unsigned long long v18;
    v18 = 256ull * v17;
    unsigned long long v19;
    v19 = 1024ull + v18;
    unsigned long long v20;
    v20 = v19 + 16ull;
    unsigned long long v21;
    v21 = v20 - 1ull;
    unsigned long long v22;
    v22 = v21 % 16ull;
    unsigned long long v23;
    v23 = v21 - v22;
    unsigned long long v24;
    v24 = v23 + 1024ull;
    bool v25;
    v25 = v24 <= 98304ull;
    bool v26;
    v26 = v25 == false;
    if (v26){
        assert("The dynamic shared memory is insufficient to allocate the tensor." && v25);
    } else {
    }
    extern __shared__ unsigned char v28[];
    bool v29;
    v29 = v24 <= v24;
    bool v30;
    v30 = v29 == false;
    if (v30){
        assert("The length of the partition has to be less than or equal to the length of the base array." && v29);
    } else {
    }
    int * v32;
    v32 = reinterpret_cast<int *>(&v28[0ull]);
    float * * v34;
    v34 = reinterpret_cast<float * *>(&v28[1024ull]);
    float * v36;
    v36 = reinterpret_cast<float *>(&v28[v23]);
    int v38;
    v38 = threadIdx.x;
    assert("Tensor range check" && 0 <= v38 && v38 < 256);
    v32[v38] = v10;
    v34[v38] = v14;
    asm("barrier.cta.sync %0;" :: "r"(0));
    bool v39;
    v39 = 0 <= v38;
    bool v40;
    v40 = v39 == false;
    if (v40){
        assert("The index needs to be zero or positive." && v39);
    } else {
    }
    int v42;
    v42 = v38 % 4;
    int v43;
    v43 = v38 / 4;
    bool v44;
    v44 = v43 < 64;
    bool v45;
    v45 = v44 == false;
    if (v45){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v44);
    } else {
    }
    assert("Tensor range check" && 0 <= v43 && v43 < 64);
    int v47;
    v47 = 0;
    while (while_method_0(v47)){
        bool v49;
        v49 = 0 <= v43;
        bool v50;
        v50 = v49 && v44;
        bool v51;
        v51 = v50 == false;
        if (v51){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v50);
        } else {
        }
        bool v53;
        v53 = 0 <= v47;
        bool v55;
        if (v53){
            bool v54;
            v54 = v47 < 4;
            v55 = v54;
        } else {
            v55 = false;
        }
        bool v56;
        v56 = v55 == false;
        if (v56){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v55);
        } else {
        }
        int v58;
        v58 = v47 * 64;
        int v59;
        v59 = v58 + v43;
        assert("Tensor range check" && 0 <= v47 && v47 < 4);
        int v60;
        v60 = 64 * v47;
        int v61;
        v61 = v60 + v43;
        int v62;
        v62 = v32[v61];
        float * v63;
        v63 = v34[v61];
        int v64;
        v64 = blockIdx.x;
        int v65;
        v65 = v64 * 256;
        int v66;
        v66 = v65 + v59;
        assert("Tensor range check" && 0 <= v42 && v42 < 4);
        int v67;
        v67 = 4 * v42;
        float v68[4];
        int v69[4];
        int v70;
        v70 = 0;
        while (while_method_6(v70)){
            assert("Tensor range check" && 0 <= v70 && v70 < 1);
            int v72;
            v72 = 4 * v70;
            assert("Tensor range check" && 0 <= v70 && v70 < 1);
            int v73;
            v73 = 16 * v70;
            int v74;
            v74 = v73 + v67;
            int4* v75;
            v75 = reinterpret_cast<int4*>(v63 + v74);
            int4* v76;
            v76 = reinterpret_cast<int4*>(v68 + v72);
            assert("Pointer alignment check" && (unsigned long long)(v75) % 4 == 0 && (unsigned long long)(v76) % 4 == 0);
            *v76 = *v75;
            v70 += 1 ;
        }
        int v77;
        v77 = 0;
        while (while_method_6(v77)){
            int v79;
            v79 = 0;
            while (while_method_0(v79)){
                bool v81;
                v81 = 0 <= v79;
                bool v83;
                if (v81){
                    bool v82;
                    v82 = v79 < 4;
                    v83 = v82;
                } else {
                    v83 = false;
                }
                bool v84;
                v84 = v83 == false;
                if (v84){
                    assert("The indices should be inside the range of the dimension." && v83);
                } else {
                }
                bool v86;
                v86 = 0 <= v42;
                bool v88;
                if (v86){
                    bool v87;
                    v87 = v42 < 4;
                    v88 = v87;
                } else {
                    v88 = false;
                }
                bool v89;
                v89 = v88 == false;
                if (v89){
                    assert("The indices should be inside the range of the dimension." && v88);
                } else {
                }
                int v91;
                v91 = v42 * 4;
                int v92;
                v92 = v79 + v91;
                bool v93;
                v93 = 0 <= v77;
                bool v95;
                if (v93){
                    bool v94;
                    v94 = v77 < 1;
                    v95 = v94;
                } else {
                    v95 = false;
                }
                bool v96;
                v96 = v95 == false;
                if (v96){
                    assert("The indices should be inside the range of the dimension." && v95);
                } else {
                }
                int v98;
                v98 = v77 * 16;
                int v99;
                v99 = v92 + v98;
                assert("Tensor range check" && 0 <= v77 && v77 < 1);
                assert("Tensor range check" && 0 <= v79 && v79 < 4);
                int v100;
                v100 = 4 * v77;
                int v101;
                v101 = v100 + v79;
                v69[v101] = v99;
                v79 += 1 ;
            }
            v77 += 1 ;
        }
        bool v102[4];
        int v103;
        v103 = 0;
        while (while_method_6(v103)){
            int v105;
            v105 = 0;
            while (while_method_0(v105)){
                assert("Tensor range check" && 0 <= v103 && v103 < 1);
                assert("Tensor range check" && 0 <= v105 && v105 < 4);
                int v107;
                v107 = 4 * v103;
                int v108;
                v108 = v107 + v105;
                float v109;
                v109 = v68[v108];
                int v110;
                v110 = v69[v108];
                bool v111;
                v111 = v110 < 11;
                assert("Tensor range check" && 0 <= v103 && v103 < 1);
                assert("Tensor range check" && 0 <= v105 && v105 < 4);
                v102[v108] = v111;
                v105 += 1 ;
            }
            v103 += 1 ;
        }
        float v112[4];
        int v113;
        v113 = 0;
        while (while_method_6(v113)){
            int v115;
            v115 = 0;
            while (while_method_0(v115)){
                assert("Tensor range check" && 0 <= v113 && v113 < 1);
                assert("Tensor range check" && 0 <= v115 && v115 < 4);
                int v117;
                v117 = 4 * v113;
                int v118;
                v118 = v117 + v115;
                float v119;
                v119 = v68[v118];
                bool v120;
                v120 = v102[v118];
                float v123;
                if (v120){
                    bool v121;
                    v121 = 0.0f >= v119;
                    if (v121){
                        v123 = 0.0f;
                    } else {
                        v123 = v119;
                    }
                } else {
                    v123 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v113 && v113 < 1);
                assert("Tensor range check" && 0 <= v115 && v115 < 4);
                v112[v118] = v123;
                v115 += 1 ;
            }
            v113 += 1 ;
        }
        float v124;
        v124 = 0.0f;
        int v125;
        v125 = 0;
        while (while_method_6(v125)){
            int v127;
            v127 = 0;
            while (while_method_0(v127)){
                assert("Tensor range check" && 0 <= v125 && v125 < 1);
                assert("Tensor range check" && 0 <= v127 && v127 < 4);
                int v129;
                v129 = 4 * v125;
                int v130;
                v130 = v129 + v127;
                float v131;
                v131 = v112[v130];
                float v132;
                v132 = v124 + v131;
                v124 = v132;
                v127 += 1 ;
            }
            v125 += 1 ;
        }
        auto v133 = cooperative_groups::coalesced_threads();
        int v134;
        v134 = threadIdx.x;
        int v135;
        v135 = v134 / 4;
        auto v136 = cooperative_groups::labeled_partition(v133,v135);
        Closure1 v137{};
        float v138;
        v138 = cooperative_groups::reduce(v136, v124, v137);
        int v139[4];
        int v140;
        v140 = 0;
        while (while_method_6(v140)){
            int v142;
            v142 = 0;
            while (while_method_0(v142)){
                assert("Tensor range check" && 0 <= v140 && v140 < 1);
                assert("Tensor range check" && 0 <= v142 && v142 < 4);
                int v144;
                v144 = 4 * v140;
                int v145;
                v145 = v144 + v142;
                bool v146;
                v146 = v102[v145];
                int v147;
                if (v146){
                    v147 = 1;
                } else {
                    v147 = 0;
                }
                assert("Tensor range check" && 0 <= v140 && v140 < 1);
                assert("Tensor range check" && 0 <= v142 && v142 < 4);
                v139[v145] = v147;
                v142 += 1 ;
            }
            v140 += 1 ;
        }
        int v148;
        v148 = 0;
        int v149;
        v149 = 0;
        while (while_method_6(v149)){
            int v151;
            v151 = 0;
            while (while_method_0(v151)){
                assert("Tensor range check" && 0 <= v149 && v149 < 1);
                assert("Tensor range check" && 0 <= v151 && v151 < 4);
                int v153;
                v153 = 4 * v149;
                int v154;
                v154 = v153 + v151;
                int v155;
                v155 = v139[v154];
                int v156;
                v156 = v148 + v155;
                v148 = v156;
                v151 += 1 ;
            }
            v149 += 1 ;
        }
        auto v157 = cooperative_groups::coalesced_threads();
        int v158;
        v158 = threadIdx.x;
        int v159;
        v159 = v158 / 4;
        auto v160 = cooperative_groups::labeled_partition(v157,v159);
        Closure2 v161{};
        int v162;
        v162 = cooperative_groups::reduce(v160, v148, v161);
        float v163;
        v163 = (float)v162;
        float v164;
        v164 = 1.0f / v163;
        float v165[4];
        int v166;
        v166 = 0;
        while (while_method_6(v166)){
            int v168;
            v168 = 0;
            while (while_method_0(v168)){
                assert("Tensor range check" && 0 <= v166 && v166 < 1);
                assert("Tensor range check" && 0 <= v168 && v168 < 4);
                int v170;
                v170 = 4 * v166;
                int v171;
                v171 = v170 + v168;
                float v172;
                v172 = v112[v171];
                bool v173;
                v173 = v102[v171];
                bool v174;
                v174 = v173 == false;
                float v179;
                if (v174){
                    v179 = 0.0f;
                } else {
                    bool v175;
                    v175 = v138 == 0.0f;
                    bool v176;
                    v176 = v175 != true;
                    if (v176){
                        float v177;
                        v177 = v172 / v138;
                        v179 = v177;
                    } else {
                        v179 = v164;
                    }
                }
                assert("Tensor range check" && 0 <= v166 && v166 < 1);
                assert("Tensor range check" && 0 <= v168 && v168 < 4);
                v165[v171] = v179;
                v168 += 1 ;
            }
            v166 += 1 ;
        }
        float v180; int v181;
        Tuple8 tmp25 = Tuple8{0.0f, 2147483647};
        v180 = tmp25.v0; v181 = tmp25.v1;
        int v182;
        v182 = 0;
        while (while_method_6(v182)){
            int v184;
            v184 = 0;
            while (while_method_0(v184)){
                assert("Tensor range check" && 0 <= v182 && v182 < 1);
                assert("Tensor range check" && 0 <= v184 && v184 < 4);
                int v186;
                v186 = 4 * v182;
                int v187;
                v187 = v186 + v184;
                float v188;
                v188 = v165[v187];
                int v189;
                v189 = v69[v187];
                bool v190;
                v190 = v181 == v62;
                float v194; int v195;
                if (v190){
                    v194 = v180; v195 = v181;
                } else {
                    bool v191;
                    v191 = v189 == v62;
                    if (v191){
                        v194 = v188; v195 = v189;
                    } else {
                        v194 = v180; v195 = v181;
                    }
                }
                v180 = v194;
                v181 = v195;
                v184 += 1 ;
            }
            v182 += 1 ;
        }
        auto v196 = cooperative_groups::coalesced_threads();
        int v197;
        v197 = threadIdx.x;
        int v198;
        v198 = v197 / 4;
        auto v199 = cooperative_groups::labeled_partition(v196,v198);
        Closure7 v200{v62};
        float v201; int v202;
        Tuple8 tmp26 = cooperative_groups::reduce(v199, Tuple8{v180, v181}, v200);
        v201 = tmp26.v0; v202 = tmp26.v1;
        bool v203;
        v203 = v202 == 2147483647;
        bool v204;
        v204 = v203 != true;
        bool v205;
        v205 = v204 == false;
        if (v205){
            assert("Expected a valid action id in get_action." && v204);
        } else {
        }
        int v207;
        v207 = 0;
        while (while_method_6(v207)){
            assert("Tensor range check" && 0 <= v207 && v207 < 1);
            assert("Tensor range check" && 0 <= v207 && v207 < 1);
            v207 += 1 ;
        }
        assert("Tensor range check" && 0 <= v59 && v59 < 256);
        v36[v59] = v201;
        v47 += 1 ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0));
    assert("Tensor range check" && 0 <= v38 && v38 < 256);
    float v209;
    v209 = v36[v38];
    asm("barrier.cta.sync %0;" :: "r"(0));
    return v209;
}
__device__ inline bool while_method_14(int v0){
    bool v1;
    v1 = v0 < 6;
    return v1;
}
__device__ inline bool while_method_15(static_array<float,6> v0, int v1){
    bool v2;
    v2 = v1 < 6;
    return v2;
}
__device__ inline bool while_method_16(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    return v2;
}
__device__ int loop_19(static_array<float,6> v0, float v1, int v2){
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
            return loop_19(v0, v1, v11);
        }
    } else {
        return 5;
    }
}
__device__ int pick_discrete__18(static_array<float,6> v0, float v1){
    static_array<float,6> v2;
    int v4;
    v4 = 0;
    while (while_method_14(v4)){
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
    while (while_method_15(v2, v13)){
        int v15;
        v15 = 6;
        while (while_method_16(v13, v15)){
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
    return loop_19(v2, v36, v37);
}
__device__ int sample_discrete__17(static_array<float,6> v0, curandStatePhilox4_32_10_t & v1){
    float v2;
    v2 = curand_uniform(&v1);
    return pick_discrete__18(v0, v2);
}
__device__ Union2 sample_discrete_16(static_array<Tuple12,6> v0, curandStatePhilox4_32_10_t & v1){
    static_array<float,6> v2;
    int v4;
    v4 = 0;
    while (while_method_14(v4)){
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
        Tuple12 tmp32 = v0[v4];
        v11 = tmp32.v0; v12 = tmp32.v1;
        v2[v4] = v12;
        v4 += 1 ;
    }
    int v15;
    v15 = sample_discrete__17(v2, v1);
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
    Tuple12 tmp33 = v0[v15];
    v21 = tmp33.v0; v22 = tmp33.v1;
    return v21;
}
__device__ inline bool while_method_17(int v0){
    bool v1;
    v1 = v0 < 7;
    return v1;
}
__device__ inline bool while_method_18(static_array<unsigned char,7> v0, bool v1, int v2){
    bool v3;
    v3 = v2 < 7;
    return v3;
}
__device__ inline bool while_method_19(static_array<unsigned char,7> v0, int v1){
    bool v2;
    v2 = v1 < 7;
    return v2;
}
__device__ inline bool while_method_20(int v0, int v1, int v2, int v3){
    bool v4;
    v4 = v3 < v0;
    return v4;
}
__device__ Tuple0 score_20(static_array<unsigned char,7> v0){
    static_array<unsigned char,7> v1;
    int v3;
    v3 = 0;
    while (while_method_17(v3)){
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
    Tuple13 tmp34 = Tuple13{true, 1};
    v14 = tmp34.v0; v15 = tmp34.v1;
    while (while_method_18(v1, v14, v15)){
        int v17;
        v17 = 0;
        while (while_method_19(v1, v17)){
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
            Tuple14 tmp35 = Tuple14{v17, v21, v17};
            v26 = tmp35.v0; v27 = tmp35.v1; v28 = tmp35.v2;
            while (while_method_20(v25, v26, v27, v28)){
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
                    Union10 v71;
                    if (v65){
                        v71 = Union10{Union10_2{}};
                    } else {
                        bool v67;
                        v67 = v63 > v64;
                        if (v67){
                            v71 = Union10{Union10_1{}};
                        } else {
                            v71 = Union10{Union10_0{}};
                        }
                    }
                    Union10 v81;
                    switch (v71.tag) {
                        case 0: { // Eq
                            unsigned char v72;
                            v72 = v47 % 4u;
                            unsigned char v73;
                            v73 = v62 % 4u;
                            bool v74;
                            v74 = v72 < v73;
                            if (v74){
                                v81 = Union10{Union10_2{}};
                            } else {
                                bool v76;
                                v76 = v72 > v73;
                                if (v76){
                                    v81 = Union10{Union10_1{}};
                                } else {
                                    v81 = Union10{Union10_0{}};
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
    Tuple15 tmp36 = Tuple15{0, 0, 12u};
    v132 = tmp36.v0; v133 = tmp36.v1; v134 = tmp36.v2;
    while (while_method_17(v132)){
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
    Union11 v202;
    if (v196){
        v202 = Union11{Union11_1{v130}};
    } else {
        bool v198;
        v198 = v133 == 5;
        if (v198){
            v202 = Union11{Union11_1{v130}};
        } else {
            v202 = Union11{Union11_0{}};
        }
    }
    static_array<unsigned char,5> v203;
    int v205; int v206; unsigned char v207;
    Tuple15 tmp37 = Tuple15{0, 0, 12u};
    v205 = tmp37.v0; v206 = tmp37.v1; v207 = tmp37.v2;
    while (while_method_17(v205)){
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
    Union11 v275;
    if (v269){
        v275 = Union11{Union11_1{v203}};
    } else {
        bool v271;
        v271 = v206 == 5;
        if (v271){
            v275 = Union11{Union11_1{v203}};
        } else {
            v275 = Union11{Union11_0{}};
        }
    }
    Union11 v312;
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
                    Union10 v278;
                    v278 = Union10{Union10_0{}};
                    int v279; Union10 v280;
                    Tuple16 tmp38 = Tuple16{0, v278};
                    v279 = tmp38.v0; v280 = tmp38.v1;
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
                        Union10 v305;
                        switch (v280.tag) {
                            case 0: { // Eq
                                unsigned char v295;
                                v295 = v287 / 4u;
                                unsigned char v296;
                                v296 = v293 / 4u;
                                bool v297;
                                v297 = v295 < v296;
                                if (v297){
                                    v305 = Union10{Union10_2{}};
                                } else {
                                    bool v299;
                                    v299 = v295 > v296;
                                    if (v299){
                                        v305 = Union10{Union10_1{}};
                                    } else {
                                        v305 = Union10{Union10_0{}};
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
                    v312 = Union11{Union11_1{v307}};
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
    Tuple15 tmp39 = Tuple15{0, 0, 12u};
    v315 = tmp39.v0; v316 = tmp39.v1; v317 = tmp39.v2;
    while (while_method_17(v315)){
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
    Union11 v385;
    if (v379){
        v385 = Union11{Union11_1{v313}};
    } else {
        bool v381;
        v381 = v316 == 5;
        if (v381){
            v385 = Union11{Union11_1{v313}};
        } else {
            v385 = Union11{Union11_0{}};
        }
    }
    Union11 v422;
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
                    Union10 v388;
                    v388 = Union10{Union10_0{}};
                    int v389; Union10 v390;
                    Tuple16 tmp40 = Tuple16{0, v388};
                    v389 = tmp40.v0; v390 = tmp40.v1;
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
                        Union10 v415;
                        switch (v390.tag) {
                            case 0: { // Eq
                                unsigned char v405;
                                v405 = v397 / 4u;
                                unsigned char v406;
                                v406 = v403 / 4u;
                                bool v407;
                                v407 = v405 < v406;
                                if (v407){
                                    v415 = Union10{Union10_2{}};
                                } else {
                                    bool v409;
                                    v409 = v405 > v406;
                                    if (v409){
                                        v415 = Union10{Union10_1{}};
                                    } else {
                                        v415 = Union10{Union10_0{}};
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
                    v422 = Union11{Union11_1{v417}};
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
    Tuple15 tmp41 = Tuple15{0, 0, 12u};
    v425 = tmp41.v0; v426 = tmp41.v1; v427 = tmp41.v2;
    while (while_method_17(v425)){
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
    Union11 v495;
    if (v489){
        v495 = Union11{Union11_1{v423}};
    } else {
        bool v491;
        v491 = v426 == 5;
        if (v491){
            v495 = Union11{Union11_1{v423}};
        } else {
            v495 = Union11{Union11_0{}};
        }
    }
    Union11 v532;
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
                    Union10 v498;
                    v498 = Union10{Union10_0{}};
                    int v499; Union10 v500;
                    Tuple16 tmp42 = Tuple16{0, v498};
                    v499 = tmp42.v0; v500 = tmp42.v1;
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
                        Union10 v525;
                        switch (v500.tag) {
                            case 0: { // Eq
                                unsigned char v515;
                                v515 = v507 / 4u;
                                unsigned char v516;
                                v516 = v513 / 4u;
                                bool v517;
                                v517 = v515 < v516;
                                if (v517){
                                    v525 = Union10{Union10_2{}};
                                } else {
                                    bool v519;
                                    v519 = v515 > v516;
                                    if (v519){
                                        v525 = Union10{Union10_1{}};
                                    } else {
                                        v525 = Union10{Union10_0{}};
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
                    v532 = Union11{Union11_1{v527}};
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
            Tuple17 tmp43 = Tuple17{0, 0, 0, 12u};
            v538 = tmp43.v0; v539 = tmp43.v1; v540 = tmp43.v2; v541 = tmp43.v3;
            while (while_method_17(v538)){
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
            Union12 v577;
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
                v577 = Union12{Union12_1{v534, v536}};
            } else {
                v577 = Union12{Union12_0{}};
            }
            Union11 v615;
            switch (v577.tag) {
                case 0: { // None
                    v615 = Union11{Union11_0{}};
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
                    v615 = Union11{Union11_1{v591}};
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
                    Tuple17 tmp44 = Tuple17{0, 0, 0, 12u};
                    v621 = tmp44.v0; v622 = tmp44.v1; v623 = tmp44.v2; v624 = tmp44.v3;
                    while (while_method_17(v621)){
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
                    Union13 v660;
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
                        v660 = Union13{Union13_1{v617, v619}};
                    } else {
                        v660 = Union13{Union13_0{}};
                    }
                    Union11 v736;
                    switch (v660.tag) {
                        case 0: { // None
                            v736 = Union11{Union11_0{}};
                            break;
                        }
                        case 1: { // Some
                            static_array<unsigned char,3> v661 = v660.case1.v0; static_array<unsigned char,4> v662 = v660.case1.v1;
                            static_array<unsigned char,2> v663;
                            static_array<unsigned char,2> v665;
                            int v667; int v668; int v669; unsigned char v670;
                            Tuple17 tmp45 = Tuple17{0, 0, 0, 12u};
                            v667 = tmp45.v0; v668 = tmp45.v1; v669 = tmp45.v2; v670 = tmp45.v3;
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
                            Union14 v706;
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
                                v706 = Union14{Union14_1{v663, v665}};
                            } else {
                                v706 = Union14{Union14_0{}};
                            }
                            switch (v706.tag) {
                                case 0: { // None
                                    v736 = Union11{Union11_0{}};
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
                                    v736 = Union11{Union11_1{v709}};
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
                            Tuple4 tmp46 = Tuple4{0, 0};
                            v740 = tmp46.v0; v741 = tmp46.v1;
                            while (while_method_17(v740)){
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
                            Union11 v759;
                            if (v756){
                                v759 = Union11{Union11_1{v738}};
                            } else {
                                v759 = Union11{Union11_0{}};
                            }
                            static_array<unsigned char,5> v760;
                            int v762; int v763;
                            Tuple4 tmp47 = Tuple4{0, 0};
                            v762 = tmp47.v0; v763 = tmp47.v1;
                            while (while_method_17(v762)){
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
                            Union11 v781;
                            if (v778){
                                v781 = Union11{Union11_1{v760}};
                            } else {
                                v781 = Union11{Union11_0{}};
                            }
                            Union11 v818;
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
                                            Union10 v784;
                                            v784 = Union10{Union10_0{}};
                                            int v785; Union10 v786;
                                            Tuple16 tmp48 = Tuple16{0, v784};
                                            v785 = tmp48.v0; v786 = tmp48.v1;
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
                                                Union10 v811;
                                                switch (v786.tag) {
                                                    case 0: { // Eq
                                                        unsigned char v801;
                                                        v801 = v793 / 4u;
                                                        unsigned char v802;
                                                        v802 = v799 / 4u;
                                                        bool v803;
                                                        v803 = v801 < v802;
                                                        if (v803){
                                                            v811 = Union10{Union10_2{}};
                                                        } else {
                                                            bool v805;
                                                            v805 = v801 > v802;
                                                            if (v805){
                                                                v811 = Union10{Union10_1{}};
                                                            } else {
                                                                v811 = Union10{Union10_0{}};
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
                                            v818 = Union11{Union11_1{v813}};
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
                            Tuple4 tmp49 = Tuple4{0, 0};
                            v821 = tmp49.v0; v822 = tmp49.v1;
                            while (while_method_17(v821)){
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
                            Union11 v840;
                            if (v837){
                                v840 = Union11{Union11_1{v819}};
                            } else {
                                v840 = Union11{Union11_0{}};
                            }
                            Union11 v877;
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
                                            Union10 v843;
                                            v843 = Union10{Union10_0{}};
                                            int v844; Union10 v845;
                                            Tuple16 tmp50 = Tuple16{0, v843};
                                            v844 = tmp50.v0; v845 = tmp50.v1;
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
                                                Union10 v870;
                                                switch (v845.tag) {
                                                    case 0: { // Eq
                                                        unsigned char v860;
                                                        v860 = v852 / 4u;
                                                        unsigned char v861;
                                                        v861 = v858 / 4u;
                                                        bool v862;
                                                        v862 = v860 < v861;
                                                        if (v862){
                                                            v870 = Union10{Union10_2{}};
                                                        } else {
                                                            bool v864;
                                                            v864 = v860 > v861;
                                                            if (v864){
                                                                v870 = Union10{Union10_1{}};
                                                            } else {
                                                                v870 = Union10{Union10_0{}};
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
                                            v877 = Union11{Union11_1{v872}};
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
                            Tuple4 tmp51 = Tuple4{0, 0};
                            v880 = tmp51.v0; v881 = tmp51.v1;
                            while (while_method_17(v880)){
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
                            Union11 v899;
                            if (v896){
                                v899 = Union11{Union11_1{v878}};
                            } else {
                                v899 = Union11{Union11_0{}};
                            }
                            Union11 v936;
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
                                            Union10 v902;
                                            v902 = Union10{Union10_0{}};
                                            int v903; Union10 v904;
                                            Tuple16 tmp52 = Tuple16{0, v902};
                                            v903 = tmp52.v0; v904 = tmp52.v1;
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
                                                Union10 v929;
                                                switch (v904.tag) {
                                                    case 0: { // Eq
                                                        unsigned char v919;
                                                        v919 = v911 / 4u;
                                                        unsigned char v920;
                                                        v920 = v917 / 4u;
                                                        bool v921;
                                                        v921 = v919 < v920;
                                                        if (v921){
                                                            v929 = Union10{Union10_2{}};
                                                        } else {
                                                            bool v923;
                                                            v923 = v919 > v920;
                                                            if (v923){
                                                                v929 = Union10{Union10_1{}};
                                                            } else {
                                                                v929 = Union10{Union10_0{}};
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
                                            v936 = Union11{Union11_1{v931}};
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
                                    Tuple15 tmp53 = Tuple15{0, 0, 12u};
                                    v940 = tmp53.v0; v941 = tmp53.v1; v942 = tmp53.v2;
                                    while (while_method_17(v940)){
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
                                    Union11 v980;
                                    if (v974){
                                        v980 = Union11{Union11_1{v938}};
                                    } else {
                                        bool v976;
                                        v976 = v941 == 5;
                                        if (v976){
                                            v980 = Union11{Union11_1{v938}};
                                        } else {
                                            v980 = Union11{Union11_0{}};
                                        }
                                    }
                                    switch (v980.tag) {
                                        case 0: { // None
                                            static_array<unsigned char,3> v982;
                                            static_array<unsigned char,4> v984;
                                            int v986; int v987; int v988; unsigned char v989;
                                            Tuple17 tmp54 = Tuple17{0, 0, 0, 12u};
                                            v986 = tmp54.v0; v987 = tmp54.v1; v988 = tmp54.v2; v989 = tmp54.v3;
                                            while (while_method_17(v986)){
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
                                            Union13 v1025;
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
                                                v1025 = Union13{Union13_1{v982, v984}};
                                            } else {
                                                v1025 = Union13{Union13_0{}};
                                            }
                                            Union11 v1063;
                                            switch (v1025.tag) {
                                                case 0: { // None
                                                    v1063 = Union11{Union11_0{}};
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
                                                    v1063 = Union11{Union11_1{v1039}};
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
                                                    Tuple17 tmp55 = Tuple17{0, 0, 0, 12u};
                                                    v1069 = tmp55.v0; v1070 = tmp55.v1; v1071 = tmp55.v2; v1072 = tmp55.v3;
                                                    while (while_method_17(v1069)){
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
                                                    Union15 v1108;
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
                                                        v1108 = Union15{Union15_1{v1065, v1067}};
                                                    } else {
                                                        v1108 = Union15{Union15_0{}};
                                                    }
                                                    Union11 v1205;
                                                    switch (v1108.tag) {
                                                        case 0: { // None
                                                            v1205 = Union11{Union11_0{}};
                                                            break;
                                                        }
                                                        case 1: { // Some
                                                            static_array<unsigned char,2> v1109 = v1108.case1.v0; static_array<unsigned char,5> v1110 = v1108.case1.v1;
                                                            static_array<unsigned char,2> v1111;
                                                            static_array<unsigned char,3> v1113;
                                                            int v1115; int v1116; int v1117; unsigned char v1118;
                                                            Tuple17 tmp56 = Tuple17{0, 0, 0, 12u};
                                                            v1115 = tmp56.v0; v1116 = tmp56.v1; v1117 = tmp56.v2; v1118 = tmp56.v3;
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
                                                            Union16 v1154;
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
                                                                v1154 = Union16{Union16_1{v1111, v1113}};
                                                            } else {
                                                                v1154 = Union16{Union16_0{}};
                                                            }
                                                            switch (v1154.tag) {
                                                                case 0: { // None
                                                                    v1205 = Union11{Union11_0{}};
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
                                                                    v1205 = Union11{Union11_1{v1168}};
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
                                                            Tuple17 tmp57 = Tuple17{0, 0, 0, 12u};
                                                            v1211 = tmp57.v0; v1212 = tmp57.v1; v1213 = tmp57.v2; v1214 = tmp57.v3;
                                                            while (while_method_17(v1211)){
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
                                                            Union15 v1250;
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
                                                                v1250 = Union15{Union15_1{v1207, v1209}};
                                                            } else {
                                                                v1250 = Union15{Union15_0{}};
                                                            }
                                                            Union11 v1288;
                                                            switch (v1250.tag) {
                                                                case 0: { // None
                                                                    v1288 = Union11{Union11_0{}};
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
                                                                    v1288 = Union11{Union11_1{v1264}};
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
__device__ void method_0(unsigned char * v0, unsigned char * v1, StackMut0 & v2, int v3, Union3 v4){
    v2.v0 = 4503599627370495ull;
    static_array<float,2> v5;
    v5[0] = 0.0f;
    v5[1] = 0.0f;
    v2.v4 = v5;
    static_array_list<Union1,128> & v7 = v2.v2;
    v7.unsafe_set_length(0);
    static_array<Union0,2> v8;
    Union0 v10;
    v10 = Union0{Union0_0{}};
    v8[0] = v10;
    Union0 v12;
    v12 = Union0{Union0_0{}};
    v8[1] = v12;
    int v14;
    v14 = v3 ^ 1;
    Union0 v15;
    v15 = Union0{Union0_2{}};
    v8[v14] = v15;
    v2.v3 = v8;
    static_array_list<Union1,128> & v17 = v2.v2;
    unsigned long long & v18 = v2.v0;
    Union5 v19;
    v19 = Union5{Union5_1{v4}};
    Union5 v20;
    v20 = v19;
    while (while_method_3(v20)){
        Union5 v1768;
        switch (v20.tag) {
            case 0: { // None
                v1768 = Union5{Union5_0{}};
                break;
            }
            case 1: { // Some
                Union3 v22 = v20.case1.v0;
                Union6 v1419;
                switch (v22.tag) {
                    case 0: { // G_Flop
                        int v1280 = v22.case0.v0; static_array<static_array<unsigned char,2>,2> v1281 = v22.case0.v1; static_array<int,2> v1282 = v22.case0.v2; int v1283 = v22.case0.v3; static_array<int,2> v1284 = v22.case0.v4; Union4 v1285 = v22.case0.v5;
                        curandStatePhilox4_32_10_t & v1286 = v2.v5;
                        curandStatePhilox4_32_10_t & v1287 = v1286;
                        static_array<unsigned char,3> v1288; unsigned long long v1289;
                        Tuple1 tmp2 = draw_cards_1(v1287, v18);
                        v1288 = tmp2.v0; v1289 = tmp2.v1;
                        v2.v0 = v1289;
                        static_array_list<unsigned char,5> v1290;
                        v1290 = get_community_cards_4(v1285, v1288);
                        Union1 v1291;
                        v1291 = Union1{Union1_0{v1290}};
                        v17.push(v1291);
                        Union4 v1294;
                        switch (v1285.tag) {
                            case 1: { // Preflop
                                v1294 = Union4{Union4_0{v1288}};
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in flop.");
                                __trap();
                            }
                        }
                        int v1295;
                        v1295 = 2;
                        int v1296;
                        v1296 = 0;
                        Union3 v1297;
                        v1297 = try_round_5(v1295, v1281, v1282, v1296, v1284, v1294);
                        v1419 = Union6{Union6_2{v1297}};
                        break;
                    }
                    case 1: { // G_Fold
                        int v23 = v22.case1.v0; static_array<static_array<unsigned char,2>,2> v24 = v22.case1.v1; static_array<int,2> v25 = v22.case1.v2; int v26 = v22.case1.v3; static_array<int,2> v27 = v22.case1.v4; Union4 v28 = v22.case1.v5;
                        int v29;
                        v29 = v26 % 2;
                        bool v30;
                        v30 = 0 <= v29;
                        bool v32;
                        if (v30){
                            bool v31;
                            v31 = v29 < 2;
                            v32 = v31;
                        } else {
                            v32 = false;
                        }
                        bool v33;
                        v33 = v32 == false;
                        if (v33){
                            assert("Index must be in range." && v32);
                        } else {
                        }
                        int v35;
                        v35 = v25[v29];
                        int v37;
                        v37 = v26 + 1;
                        int v38;
                        v38 = v37 % 2;
                        Union1 v39;
                        v39 = Union1{Union1_1{v35, v38}};
                        v17.push(v39);
                        v1419 = Union6{Union6_0{}};
                        break;
                    }
                    case 2: { // G_Preflop
                        curandStatePhilox4_32_10_t & v1381 = v2.v5;
                        curandStatePhilox4_32_10_t & v1382 = v1381;
                        static_array<unsigned char,2> v1383; unsigned long long v1384;
                        Tuple5 tmp7 = draw_cards_8(v1382, v18);
                        v1383 = tmp7.v0; v1384 = tmp7.v1;
                        v2.v0 = v1384;
                        curandStatePhilox4_32_10_t & v1385 = v2.v5;
                        curandStatePhilox4_32_10_t & v1386 = v1385;
                        static_array<unsigned char,2> v1387; unsigned long long v1388;
                        Tuple5 tmp8 = draw_cards_8(v1386, v18);
                        v1387 = tmp8.v0; v1388 = tmp8.v1;
                        v2.v0 = v1388;
                        Union1 v1389;
                        v1389 = Union1{Union1_3{0, v1383}};
                        v17.push(v1389);
                        Union1 v1390;
                        v1390 = Union1{Union1_3{1, v1387}};
                        v17.push(v1390);
                        static_array<static_array<unsigned char,2>,2> v1391;
                        v1391[0] = v1383;
                        v1391[1] = v1387;
                        static_array<int,2> v1393;
                        v1393[0] = 2;
                        v1393[1] = 1;
                        static_array<int,2> v1395;
                        int v1397;
                        v1397 = 0;
                        while (while_method_2(v1397)){
                            bool v1399;
                            v1399 = 0 <= v1397;
                            bool v1401;
                            if (v1399){
                                bool v1400;
                                v1400 = v1397 < 2;
                                v1401 = v1400;
                            } else {
                                v1401 = false;
                            }
                            bool v1402;
                            v1402 = v1401 == false;
                            if (v1402){
                                assert("Index must be in range." && v1401);
                            } else {
                            }
                            int v1404;
                            v1404 = v1393[v1397];
                            int v1406;
                            v1406 = 100 - v1404;
                            v1395[v1397] = v1406;
                            v1397 += 1 ;
                        }
                        int v1407;
                        v1407 = 2;
                        int v1408;
                        v1408 = 0;
                        Union4 v1409;
                        v1409 = Union4{Union4_1{}};
                        Union3 v1410;
                        v1410 = try_round_5(v1407, v1391, v1393, v1408, v1395, v1409);
                        v1419 = Union6{Union6_2{v1410}};
                        break;
                    }
                    case 3: { // G_River
                        int v1340 = v22.case3.v0; static_array<static_array<unsigned char,2>,2> v1341 = v22.case3.v1; static_array<int,2> v1342 = v22.case3.v2; int v1343 = v22.case3.v3; static_array<int,2> v1344 = v22.case3.v4; Union4 v1345 = v22.case3.v5;
                        curandStatePhilox4_32_10_t & v1346 = v2.v5;
                        curandStatePhilox4_32_10_t & v1347 = v1346;
                        static_array<unsigned char,1> v1348; unsigned long long v1349;
                        Tuple6 tmp11 = draw_cards_9(v1347, v18);
                        v1348 = tmp11.v0; v1349 = tmp11.v1;
                        v2.v0 = v1349;
                        static_array_list<unsigned char,5> v1350;
                        v1350 = get_community_cards_10(v1345, v1348);
                        Union1 v1351;
                        v1351 = Union1{Union1_0{v1350}};
                        v17.push(v1351);
                        Union4 v1376;
                        switch (v1345.tag) {
                            case 3: { // Turn
                                static_array<unsigned char,4> v1352 = v1345.case3.v0;
                                static_array<unsigned char,5> v1353;
                                int v1355;
                                v1355 = 0;
                                while (while_method_0(v1355)){
                                    bool v1357;
                                    v1357 = 0 <= v1355;
                                    bool v1359;
                                    if (v1357){
                                        bool v1358;
                                        v1358 = v1355 < 4;
                                        v1359 = v1358;
                                    } else {
                                        v1359 = false;
                                    }
                                    bool v1360;
                                    v1360 = v1359 == false;
                                    if (v1360){
                                        assert("Index must be in range." && v1359);
                                    } else {
                                    }
                                    unsigned char v1362;
                                    v1362 = v1352[v1355];
                                    v1353[v1355] = v1362;
                                    v1355 += 1 ;
                                }
                                int v1364;
                                v1364 = 0;
                                while (while_method_6(v1364)){
                                    bool v1366;
                                    v1366 = 0 <= v1364;
                                    bool v1368;
                                    if (v1366){
                                        bool v1367;
                                        v1367 = v1364 < 1;
                                        v1368 = v1367;
                                    } else {
                                        v1368 = false;
                                    }
                                    bool v1369;
                                    v1369 = v1368 == false;
                                    if (v1369){
                                        assert("Index must be in range." && v1368);
                                    } else {
                                    }
                                    unsigned char v1371;
                                    v1371 = v1348[v1364];
                                    int v1373;
                                    v1373 = 4 + v1364;
                                    v1353[v1373] = v1371;
                                    v1364 += 1 ;
                                }
                                v1376 = Union4{Union4_2{v1353}};
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in river.");
                                __trap();
                            }
                        }
                        int v1377;
                        v1377 = 2;
                        int v1378;
                        v1378 = 0;
                        Union3 v1379;
                        v1379 = try_round_5(v1377, v1341, v1342, v1378, v1344, v1376);
                        v1419 = Union6{Union6_2{v1379}};
                        break;
                    }
                    case 4: { // G_Round
                        int v153 = v22.case4.v0; static_array<static_array<unsigned char,2>,2> v154 = v22.case4.v1; static_array<int,2> v155 = v22.case4.v2; int v156 = v22.case4.v3; static_array<int,2> v157 = v22.case4.v4; Union4 v158 = v22.case4.v5;
                        int v159;
                        v159 = v156 % 2;
                        static_array<Union0,2> & v160 = v2.v3;
                        bool v161;
                        v161 = 0 <= v159;
                        bool v163;
                        if (v161){
                            bool v162;
                            v162 = v159 < 2;
                            v163 = v162;
                        } else {
                            v163 = false;
                        }
                        bool v164;
                        v164 = v163 == false;
                        if (v164){
                            assert("Index must be in range." && v163);
                        } else {
                        }
                        Union0 v166;
                        v166 = v160[v159];
                        Union2 v1267;
                        switch (v166.tag) {
                            case 0: { // Computer
                                static_array_list<Union1,128> & v169 = v2.v2;
                                // qwe;
                                curandStatePhilox4_32_10_t & v170 = v2.v5;
                                curandStatePhilox4_32_10_t & v171 = v170;
                                unsigned int * v172;
                                v172 = reinterpret_cast<unsigned int *>(&v0[53477376ull]);
                                float * v174;
                                v174 = reinterpret_cast<float *>(&v0[0ull]);
                                float * v176;
                                v176 = reinterpret_cast<float *>(&v0[0ull]);
                                int v178;
                                v178 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v178 && v178 < 24);
                                int v179;
                                v179 = 524288 * v178;
                                int v180;
                                v180 = threadIdx.x;
                                int v181;
                                v181 = v180;
                                while (while_method_7(v181)){
                                    bool v183;
                                    v183 = 0 <= v181;
                                    bool v184;
                                    v184 = v183 == false;
                                    if (v184){
                                        assert("The index needs to be zero or positive." && v183);
                                    } else {
                                    }
                                    int v186;
                                    v186 = v181 % 2048;
                                    int v187;
                                    v187 = v181 / 2048;
                                    bool v188;
                                    v188 = v187 < 256;
                                    bool v189;
                                    v189 = v188 == false;
                                    if (v189){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v188);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v187 && v187 < 256);
                                    assert("Tensor range check" && 0 <= v186 && v186 < 2048);
                                    int v191;
                                    v191 = v186 + v179;
                                    int v192;
                                    v192 = 2048 * v187;
                                    int v193;
                                    v193 = v192 + v191;
                                    v176[v193] = 0.0f;
                                    v181 += 256 ;
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                int v194;
                                v194 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v194 && v194 < 256);
                                int v195;
                                v195 = 2048 * v194;
                                int v196;
                                v196 = v195 + v179;
                                int v197;
                                v197 = v169.length;
                                bool v198;
                                v198 = 128 >= v197;
                                bool v199;
                                v199 = v198 == false;
                                if (v199){
                                    assert("The type level dimension has to equal the value passed at runtime into create." && v198);
                                } else {
                                }
                                static_array_list<Union7,128> v201;
                                v201 = static_array_list<Union7,128>{};
                                v201.unsafe_set_length(v197);
                                int v203;
                                v203 = 0;
                                while (while_method_8(v197, v203)){
                                    Union1 v205;
                                    v205 = v169[v203];
                                    Union7 v211;
                                    switch (v205.tag) {
                                        case 2: { // PlayerAction
                                            int v207 = v205.case2.v0; Union2 v208 = v205.case2.v1;
                                            v211 = Union7{Union7_1{v208}};
                                            break;
                                        }
                                        default: {
                                            v211 = Union7{Union7_0{}};
                                        }
                                    }
                                    v201[v203] = v211;
                                    v203 += 1 ;
                                }
                                static_array<int,2> v212;
                                int v214;
                                v214 = 0;
                                while (while_method_2(v214)){
                                    int v216;
                                    v216 = v214 + v159;
                                    bool v217;
                                    v217 = 0 <= v216;
                                    bool v219;
                                    if (v217){
                                        bool v218;
                                        v218 = v216 < 2;
                                        v219 = v218;
                                    } else {
                                        v219 = false;
                                    }
                                    bool v220;
                                    v220 = v219 == false;
                                    if (v220){
                                        assert("Index must be in range." && v219);
                                    } else {
                                    }
                                    int v222;
                                    v222 = v155[v216];
                                    v212[v214] = v222;
                                    v214 += 1 ;
                                }
                                static_array<int,2> v224;
                                int v226;
                                v226 = 0;
                                while (while_method_2(v226)){
                                    int v228;
                                    v228 = v226 + v159;
                                    bool v229;
                                    v229 = 0 <= v228;
                                    bool v231;
                                    if (v229){
                                        bool v230;
                                        v230 = v228 < 2;
                                        v231 = v230;
                                    } else {
                                        v231 = false;
                                    }
                                    bool v232;
                                    v232 = v231 == false;
                                    if (v232){
                                        assert("Index must be in range." && v231);
                                    } else {
                                    }
                                    int v234;
                                    v234 = v157[v228];
                                    v224[v226] = v234;
                                    v226 += 1 ;
                                }
                                bool v237;
                                if (v161){
                                    bool v236;
                                    v236 = v159 < 2;
                                    v237 = v236;
                                } else {
                                    v237 = false;
                                }
                                bool v238;
                                v238 = v237 == false;
                                if (v238){
                                    assert("Index must be in range." && v237);
                                } else {
                                }
                                static_array<unsigned char,2> v240;
                                v240 = v154[v159];
                                static_array_list<unsigned char,5> v242;
                                v242 = static_array_list<unsigned char,5>{};
                                switch (v158.tag) {
                                    case 0: { // Flop
                                        static_array<unsigned char,3> v244 = v158.case0.v0;
                                        int v245;
                                        v245 = 0;
                                        while (while_method_4(v245)){
                                            bool v247;
                                            v247 = 0 <= v245;
                                            bool v249;
                                            if (v247){
                                                bool v248;
                                                v248 = v245 < 3;
                                                v249 = v248;
                                            } else {
                                                v249 = false;
                                            }
                                            bool v250;
                                            v250 = v249 == false;
                                            if (v250){
                                                assert("Index must be in range." && v249);
                                            } else {
                                            }
                                            unsigned char v252;
                                            v252 = v244[v245];
                                            v242.push(v252);
                                            v245 += 1 ;
                                        }
                                        break;
                                    }
                                    case 1: { // Preflop
                                        break;
                                    }
                                    case 2: { // River
                                        static_array<unsigned char,5> v264 = v158.case2.v0;
                                        int v265;
                                        v265 = 0;
                                        while (while_method_5(v265)){
                                            bool v267;
                                            v267 = 0 <= v265;
                                            bool v269;
                                            if (v267){
                                                bool v268;
                                                v268 = v265 < 5;
                                                v269 = v268;
                                            } else {
                                                v269 = false;
                                            }
                                            bool v270;
                                            v270 = v269 == false;
                                            if (v270){
                                                assert("Index must be in range." && v269);
                                            } else {
                                            }
                                            unsigned char v272;
                                            v272 = v264[v265];
                                            v242.push(v272);
                                            v265 += 1 ;
                                        }
                                        break;
                                    }
                                    case 3: { // Turn
                                        static_array<unsigned char,4> v254 = v158.case3.v0;
                                        int v255;
                                        v255 = 0;
                                        while (while_method_0(v255)){
                                            bool v257;
                                            v257 = 0 <= v255;
                                            bool v259;
                                            if (v257){
                                                bool v258;
                                                v258 = v255 < 4;
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
                                            unsigned char v262;
                                            v262 = v254[v255];
                                            v242.push(v262);
                                            v255 += 1 ;
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                float * v274;
                                v274 = v176+v196;
                                int v276;
                                v276 = v201.length;
                                bool v277;
                                v277 = v276 == 0;
                                if (v277){
                                    v274[0] = 1.0f;
                                } else {
                                }
                                int v278;
                                v278 = v201.length;
                                int v279;
                                v279 = 0;
                                while (while_method_8(v278, v279)){
                                    Union7 v281;
                                    v281 = v201[v279];
                                    int v283;
                                    v283 = v279 * 14;
                                    int v284;
                                    v284 = 1 + v283;
                                    switch (v281.tag) {
                                        case 0: { // None
                                            v274[v284] = 1.0f;
                                            break;
                                        }
                                        case 1: { // Some
                                            Union2 v285 = v281.case1.v0;
                                            int v286;
                                            v286 = v284 + 1;
                                            switch (v285.tag) {
                                                case 0: { // A_All_In
                                                    v274[v286] = 1.0f;
                                                    break;
                                                }
                                                case 1: { // A_Call
                                                    int v287;
                                                    v287 = v286 + 1;
                                                    v274[v287] = 1.0f;
                                                    break;
                                                }
                                                case 2: { // A_Fold
                                                    int v288;
                                                    v288 = v286 + 2;
                                                    v274[v288] = 1.0f;
                                                    break;
                                                }
                                                case 3: { // A_Raise
                                                    int v289 = v285.case3.v0;
                                                    int v290;
                                                    v290 = v286 + 3;
                                                    bool v291;
                                                    v291 = 1 <= v289;
                                                    bool v293;
                                                    if (v291){
                                                        bool v292;
                                                        v292 = v289 < 1023;
                                                        v293 = v292;
                                                    } else {
                                                        v293 = false;
                                                    }
                                                    bool v294;
                                                    v294 = v293 == false;
                                                    if (v294){
                                                        assert("Pickle failure. The input is out of the bounds of the given range." && v293);
                                                    } else {
                                                    }
                                                    int v296;
                                                    v296 = v289 - 1;
                                                    unsigned int v297;
                                                    v297 = (unsigned int)v296;
                                                    method_11(v297, v274, v290);
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
                                    v279 += 1 ;
                                }
                                int v298;
                                v298 = 0;
                                while (while_method_2(v298)){
                                    bool v300;
                                    v300 = 0 <= v298;
                                    bool v302;
                                    if (v300){
                                        bool v301;
                                        v301 = v298 < 2;
                                        v302 = v301;
                                    } else {
                                        v302 = false;
                                    }
                                    bool v303;
                                    v303 = v302 == false;
                                    if (v303){
                                        assert("Index must be in range." && v302);
                                    } else {
                                    }
                                    int v305;
                                    v305 = v212[v298];
                                    int v307;
                                    v307 = v298 * 11;
                                    int v308;
                                    v308 = 1794 + v307;
                                    bool v309;
                                    v309 = 0 <= v305;
                                    bool v311;
                                    if (v309){
                                        bool v310;
                                        v310 = v305 < 1023;
                                        v311 = v310;
                                    } else {
                                        v311 = false;
                                    }
                                    bool v312;
                                    v312 = v311 == false;
                                    if (v312){
                                        assert("Pickle failure. The input is out of the bounds of the given range." && v311);
                                    } else {
                                    }
                                    unsigned int v314;
                                    v314 = (unsigned int)v305;
                                    method_12(v314, v274, v308);
                                    v298 += 1 ;
                                }
                                int v315;
                                v315 = 0;
                                while (while_method_2(v315)){
                                    bool v317;
                                    v317 = 0 <= v315;
                                    bool v319;
                                    if (v317){
                                        bool v318;
                                        v318 = v315 < 2;
                                        v319 = v318;
                                    } else {
                                        v319 = false;
                                    }
                                    bool v320;
                                    v320 = v319 == false;
                                    if (v320){
                                        assert("Index must be in range." && v319);
                                    } else {
                                    }
                                    int v322;
                                    v322 = v224[v315];
                                    int v324;
                                    v324 = v315 * 11;
                                    int v325;
                                    v325 = 1817 + v324;
                                    bool v326;
                                    v326 = 0 <= v322;
                                    bool v328;
                                    if (v326){
                                        bool v327;
                                        v327 = v322 < 1023;
                                        v328 = v327;
                                    } else {
                                        v328 = false;
                                    }
                                    bool v329;
                                    v329 = v328 == false;
                                    if (v329){
                                        assert("Pickle failure. The input is out of the bounds of the given range." && v328);
                                    } else {
                                    }
                                    unsigned int v331;
                                    v331 = (unsigned int)v322;
                                    method_12(v331, v274, v325);
                                    v315 += 1 ;
                                }
                                int v332;
                                v332 = 0;
                                while (while_method_2(v332)){
                                    bool v334;
                                    v334 = 0 <= v332;
                                    bool v336;
                                    if (v334){
                                        bool v335;
                                        v335 = v332 < 2;
                                        v336 = v335;
                                    } else {
                                        v336 = false;
                                    }
                                    bool v337;
                                    v337 = v336 == false;
                                    if (v337){
                                        assert("Index must be in range." && v336);
                                    } else {
                                    }
                                    unsigned char v339;
                                    v339 = v240[v332];
                                    int v341;
                                    v341 = v332 * 17;
                                    int v342;
                                    v342 = 1840 + v341;
                                    unsigned char v343;
                                    v343 = v339 % 4u;
                                    int v344;
                                    v344 = (int)v343;
                                    unsigned char v345;
                                    v345 = v339 / 4u;
                                    int v346;
                                    v346 = (int)v345;
                                    unsigned int v347;
                                    v347 = (unsigned int)v344;
                                    int v348;
                                    v348 = (int)v347;
                                    bool v349;
                                    v349 = v348 < 4;
                                    bool v350;
                                    v350 = v349 == false;
                                    if (v350){
                                        assert("Pickle failure. Int value out of bounds." && v349);
                                    } else {
                                    }
                                    int v352;
                                    v352 = v342 + v348;
                                    v274[v352] = 1.0f;
                                    int v353;
                                    v353 = v342 + 4;
                                    unsigned int v354;
                                    v354 = (unsigned int)v346;
                                    int v355;
                                    v355 = (int)v354;
                                    bool v356;
                                    v356 = v355 < 13;
                                    bool v357;
                                    v357 = v356 == false;
                                    if (v357){
                                        assert("Pickle failure. Int value out of bounds." && v356);
                                    } else {
                                    }
                                    int v359;
                                    v359 = v353 + v355;
                                    v274[v359] = 1.0f;
                                    v332 += 1 ;
                                }
                                int v360;
                                v360 = v242.length;
                                bool v361;
                                v361 = v360 == 0;
                                if (v361){
                                    v274[1874] = 1.0f;
                                } else {
                                }
                                int v362;
                                v362 = v242.length;
                                int v363;
                                v363 = 0;
                                while (while_method_8(v362, v363)){
                                    unsigned char v365;
                                    v365 = v242[v363];
                                    int v367;
                                    v367 = v363 * 17;
                                    int v368;
                                    v368 = 1875 + v367;
                                    unsigned char v369;
                                    v369 = v365 % 4u;
                                    int v370;
                                    v370 = (int)v369;
                                    unsigned char v371;
                                    v371 = v365 / 4u;
                                    int v372;
                                    v372 = (int)v371;
                                    unsigned int v373;
                                    v373 = (unsigned int)v370;
                                    int v374;
                                    v374 = (int)v373;
                                    bool v375;
                                    v375 = v374 < 4;
                                    bool v376;
                                    v376 = v375 == false;
                                    if (v376){
                                        assert("Pickle failure. Int value out of bounds." && v375);
                                    } else {
                                    }
                                    int v378;
                                    v378 = v368 + v374;
                                    v274[v378] = 1.0f;
                                    int v379;
                                    v379 = v368 + 4;
                                    unsigned int v380;
                                    v380 = (unsigned int)v372;
                                    int v381;
                                    v381 = (int)v380;
                                    bool v382;
                                    v382 = v381 < 13;
                                    bool v383;
                                    v383 = v382 == false;
                                    if (v383){
                                        assert("Pickle failure. Int value out of bounds." && v382);
                                    } else {
                                    }
                                    int v385;
                                    v385 = v379 + v381;
                                    v274[v385] = 1.0f;
                                    v363 += 1 ;
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                int v386;
                                v386 = 0;
                                while (while_method_0(v386)){
                                    float * v388;
                                    v388 = reinterpret_cast<float *>(&v0[0ull]);
                                    float * v390;
                                    v390 = reinterpret_cast<float *>(&v1[0ull]);
                                    assert("Tensor range check" && 0 <= v386 && v386 < 4);
                                    int v392;
                                    v392 = 262144 * v386;
                                    float * v393;
                                    v393 = reinterpret_cast<float *>(&v0[50331648ull]);
                                    int v395;
                                    v395 = blockIdx.x;
                                    assert("Tensor range check" && 0 <= v395 && v395 < 24);
                                    int v396;
                                    v396 = 524288 * v395;
                                    int v397;
                                    v397 = blockIdx.x;
                                    assert("Tensor range check" && 0 <= v397 && v397 < 24);
                                    int v398;
                                    v398 = 32768 * v397;
                                    cuda::pipeline<cuda::thread_scope_thread> v399 = cuda::make_pipeline();
                                    extern __shared__ unsigned char v400[];
                                    float * v401;
                                    v401 = reinterpret_cast<float *>(&v400[0ull]);
                                    float * v403;
                                    v403 = reinterpret_cast<float *>(&v400[34816ull]);
                                    float * v405;
                                    v405 = reinterpret_cast<float *>(&v400[0ull]);
                                    int v407;
                                    v407 = threadIdx.x;
                                    int v408;
                                    v408 = v407 / 32;
                                    bool v409;
                                    v409 = 0 <= v408;
                                    bool v410;
                                    v410 = v409 == false;
                                    if (v410){
                                        assert("The index needs to be zero or positive." && v409);
                                    } else {
                                    }
                                    int v412;
                                    v412 = v408 % 8;
                                    int v413;
                                    v413 = v408 / 8;
                                    bool v414;
                                    v414 = v413 < 1;
                                    bool v415;
                                    v415 = v414 == false;
                                    if (v415){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v414);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v413 && v413 < 1);
                                    assert("Tensor range check" && 0 <= v412 && v412 < 8);
                                    int v417;
                                    v417 = 16 * v412;
                                    int v418;
                                    v418 = 17408 * v413;
                                    int v419;
                                    v419 = v418 + v417;
                                    float * v420;
                                    v420 = v405+v419;
                                    assert("Tensor range check" && 0 <= v413 && v413 < 1);
                                    int v422;
                                    v422 = 8704 * v413;
                                    int v423;
                                    v423 = threadIdx.x;
                                    int v424;
                                    v424 = v423 % 32;
                                    bool v425;
                                    v425 = 0 <= v424;
                                    bool v426;
                                    v426 = v425 == false;
                                    if (v426){
                                        assert("The index needs to be zero or positive." && v425);
                                    } else {
                                    }
                                    int v428;
                                    v428 = v424 % 4;
                                    int v429;
                                    v429 = v424 / 4;
                                    bool v430;
                                    v430 = v429 < 8;
                                    bool v431;
                                    v431 = v430 == false;
                                    if (v431){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v430);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v429 && v429 < 8);
                                    assert("Tensor range check" && 0 <= v428 && v428 < 4);
                                    int v433;
                                    v433 = v428 + v422;
                                    int v434;
                                    v434 = 68 * v429;
                                    int v435;
                                    v435 = v434 + v433;
                                    float * v436;
                                    v436 = v401+v435;
                                    assert("Tensor range check" && 0 <= v412 && v412 < 8);
                                    int v438;
                                    v438 = 1088 * v412;
                                    int v439;
                                    v439 = threadIdx.x;
                                    int v440;
                                    v440 = v439 % 32;
                                    bool v441;
                                    v441 = 0 <= v440;
                                    bool v442;
                                    v442 = v441 == false;
                                    if (v442){
                                        assert("The index needs to be zero or positive." && v441);
                                    } else {
                                    }
                                    int v444;
                                    v444 = v440 % 4;
                                    int v445;
                                    v445 = v440 / 4;
                                    bool v446;
                                    v446 = v445 < 8;
                                    bool v447;
                                    v447 = v446 == false;
                                    if (v447){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v446);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v445 && v445 < 8);
                                    assert("Tensor range check" && 0 <= v444 && v444 < 4);
                                    int v449;
                                    v449 = v444 + v438;
                                    int v450;
                                    v450 = 68 * v445;
                                    int v451;
                                    v451 = v450 + v449;
                                    float * v452;
                                    v452 = v403+v451;
                                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> v454[8];
                                    int v455;
                                    v455 = 0;
                                    while (while_method_2(v455)){
                                        int v457;
                                        v457 = 0;
                                        while (while_method_6(v457)){
                                            assert("Tensor range check" && 0 <= v455 && v455 < 2);
                                            assert("Tensor range check" && 0 <= v457 && v457 < 1);
                                            int v459;
                                            v459 = 128 * v457;
                                            int v460;
                                            v460 = v459 + v398;
                                            int v461;
                                            v461 = 16384 * v455;
                                            int v462;
                                            v462 = v461 + v460;
                                            float * v463;
                                            v463 = v393+v462;
                                            // Pushing the loop unrolling to: 0
                                            int v465;
                                            v465 = 0;
                                            #pragma unroll
                                            while (while_method_1(v465)){
                                                int v467;
                                                v467 = 0;
                                                #pragma unroll
                                                while (while_method_6(v467)){
                                                    assert("Tensor range check" && 0 <= v465 && v465 < 8);
                                                    assert("Tensor range check" && 0 <= v467 && v467 < 1);
                                                    int v469;
                                                    v469 = v465 + v467;
                                                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v470 = v454[v469];
                                                    wmma::fill_fragment(v470, 0.0f);
                                                    v467 += 1 ;
                                                }
                                                v465 += 1 ;
                                            }
                                            // Poping the loop unrolling to: 0
                                            int v471;
                                            v471 = 0;
                                            while (while_method_11(v471)){
                                                int v473;
                                                v473 = v471 + 1;
                                                bool v474;
                                                v474 = v471 == 0;
                                                int v475;
                                                v475 = v471 % 2;
                                                bool v476;
                                                v476 = 0 <= v471;
                                                bool v477;
                                                v477 = v476 == false;
                                                if (v477){
                                                    assert("The index needs to be zero or positive." && v476);
                                                } else {
                                                }
                                                bool v479;
                                                v479 = v471 < 32;
                                                bool v480;
                                                v480 = v479 == false;
                                                if (v480){
                                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v479);
                                                } else {
                                                }
                                                bool v482;
                                                v482 = v473 < 32;
                                                Union8 v488;
                                                if (v482){
                                                    bool v483;
                                                    v483 = 0 <= v473;
                                                    bool v484;
                                                    v484 = v483 == false;
                                                    if (v484){
                                                        assert("The index needs to be zero or positive." && v483);
                                                    } else {
                                                    }
                                                    v488 = Union8{Union8_1{v473}};
                                                } else {
                                                    v488 = Union8{Union8_0{}};
                                                }
                                                assert("Tensor range check" && 0 <= v455 && v455 < 2);
                                                int v489;
                                                v489 = 262144 * v455;
                                                int v490;
                                                v490 = v489 + v396;
                                                assert("Tensor range check" && 0 <= v471 && v471 < 32);
                                                int v491;
                                                v491 = 64 * v471;
                                                int v492;
                                                v492 = v491 + v490;
                                                float * v493;
                                                v493 = v388+v492;
                                                assert("Tensor range check" && 0 <= v457 && v457 < 1);
                                                int v495;
                                                v495 = 262144 * v457;
                                                int v496;
                                                v496 = v495 + v392;
                                                if (v474){
                                                    assert("Tensor range check" && 0 <= v471 && v471 < 32);
                                                    int v497;
                                                    v497 = v491 + v496;
                                                    float * v498;
                                                    v498 = v390+v497;
                                                    // Pushing the loop unrolling to: 0
                                                    v399.producer_acquire();
                                                    int v500;
                                                    v500 = threadIdx.x;
                                                    bool v501;
                                                    v501 = 0 <= v500;
                                                    bool v502;
                                                    v502 = v501 == false;
                                                    if (v502){
                                                        assert("The index needs to be zero or positive." && v501);
                                                    } else {
                                                    }
                                                    int v504;
                                                    v504 = v500 % 16;
                                                    int v505;
                                                    v505 = v500 / 16;
                                                    bool v506;
                                                    v506 = v505 < 16;
                                                    bool v507;
                                                    v507 = v506 == false;
                                                    if (v507){
                                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v506);
                                                    } else {
                                                    }
                                                    assert("Tensor range check" && 0 <= v505 && v505 < 16);
                                                    assert("Tensor range check" && 0 <= v504 && v504 < 16);
                                                    int v509;
                                                    v509 = 4 * v504;
                                                    int v510;
                                                    v510 = 68 * v505;
                                                    int v511;
                                                    v511 = v510 + v509;
                                                    int v512;
                                                    v512 = 2048 * v505;
                                                    int v513;
                                                    v513 = v512 + v509;
                                                    float * v514;
                                                    v514 = v403+v511;
                                                    float * v516;
                                                    v516 = v498+v513;
                                                    int v518;
                                                    v518 = 0;
                                                    #pragma unroll
                                                    while (while_method_1(v518)){
                                                        int v520;
                                                        v520 = 0;
                                                        #pragma unroll
                                                        while (while_method_6(v520)){
                                                            assert("Tensor range check" && 0 <= v518 && v518 < 8);
                                                            assert("Tensor range check" && 0 <= v520 && v520 < 1);
                                                            int v522;
                                                            v522 = 64 * v520;
                                                            int v523;
                                                            v523 = 1088 * v518;
                                                            int v524;
                                                            v524 = v523 + v522;
                                                            int v525;
                                                            v525 = 32768 * v518;
                                                            int v526;
                                                            v526 = v525 + v522;
                                                            constexpr int v527 = sizeof(float) * 4;
                                                            assert("Pointer alignment check" && (unsigned long long)(v516 + v526) % v527 == 0 && (unsigned long long)(v514 + v524) % v527 == 0);
                                                            cuda::memcpy_async(v514 + v524, v516 + v526, cuda::aligned_size_t<v527>(v527), v399);
                                                            v520 += 1 ;
                                                        }
                                                        v518 += 1 ;
                                                    }
                                                    v399.producer_commit();
                                                    // Poping the loop unrolling to: 0
                                                } else {
                                                }
                                                // Pushing the loop unrolling to: 0
                                                int v528;
                                                v528 = threadIdx.x;
                                                bool v529;
                                                v529 = 0 <= v528;
                                                bool v530;
                                                v530 = v529 == false;
                                                if (v530){
                                                    assert("The index needs to be zero or positive." && v529);
                                                } else {
                                                }
                                                int v532;
                                                v532 = v528 % 16;
                                                int v533;
                                                v533 = v528 / 16;
                                                bool v534;
                                                v534 = v533 < 16;
                                                bool v535;
                                                v535 = v534 == false;
                                                if (v535){
                                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v534);
                                                } else {
                                                }
                                                assert("Tensor range check" && 0 <= v533 && v533 < 16);
                                                assert("Tensor range check" && 0 <= v532 && v532 < 16);
                                                int v537;
                                                v537 = 4 * v532;
                                                int v538;
                                                v538 = 68 * v533;
                                                int v539;
                                                v539 = v538 + v537;
                                                int v540;
                                                v540 = 2048 * v533;
                                                int v541;
                                                v541 = v540 + v537;
                                                float * v542;
                                                v542 = v401+v539;
                                                float * v544;
                                                v544 = v493+v541;
                                                int v546;
                                                v546 = 0;
                                                #pragma unroll
                                                while (while_method_1(v546)){
                                                    int v548;
                                                    v548 = 0;
                                                    #pragma unroll
                                                    while (while_method_6(v548)){
                                                        assert("Tensor range check" && 0 <= v546 && v546 < 8);
                                                        assert("Tensor range check" && 0 <= v548 && v548 < 1);
                                                        int v550;
                                                        v550 = 64 * v548;
                                                        int v551;
                                                        v551 = 1088 * v546;
                                                        int v552;
                                                        v552 = v551 + v550;
                                                        int v553;
                                                        v553 = 32768 * v546;
                                                        int v554;
                                                        v554 = v553 + v550;
                                                        int4* v555;
                                                        v555 = reinterpret_cast<int4*>(v544 + v554);
                                                        int4* v556;
                                                        v556 = reinterpret_cast<int4*>(v542 + v552);
                                                        assert("Pointer alignment check" && (unsigned long long)(v555) % 4 == 0 && (unsigned long long)(v556) % 4 == 0);
                                                        *v556 = *v555;
                                                        v548 += 1 ;
                                                    }
                                                    v546 += 1 ;
                                                }
                                                // Poping the loop unrolling to: 0
                                                wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v557[1];
                                                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v558[8];
                                                cuda::pipeline_consumer_wait_prior<0>(v399);;
                                                asm("barrier.cta.sync %0;" :: "r"(0));
                                                // Pushing the loop unrolling to: 0
                                                int v559;
                                                v559 = 0;
                                                #pragma unroll
                                                while (while_method_6(v559)){
                                                    int v561;
                                                    v561 = 0;
                                                    #pragma unroll
                                                    while (while_method_1(v561)){
                                                        assert("Tensor range check" && 0 <= v559 && v559 < 1);
                                                        assert("Tensor range check" && 0 <= v561 && v561 < 8);
                                                        int v563;
                                                        v563 = 8 * v559;
                                                        int v564;
                                                        v564 = v563 + v561;
                                                        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v565 = v558[v564];
                                                        assert("Tensor range check" && 0 <= v559 && v559 < 1);
                                                        int v566;
                                                        v566 = 1088 * v559;
                                                        assert("Tensor range check" && 0 <= v561 && v561 < 8);
                                                        int v567;
                                                        v567 = 8 * v561;
                                                        int v568;
                                                        v568 = v567 + v566;
                                                        int v569;
                                                        v569 = 0;
                                                        #pragma unroll
                                                        while (while_method_2(v569)){
                                                            int v571;
                                                            v571 = 0;
                                                            #pragma unroll
                                                            while (while_method_2(v571)){
                                                                assert("Tensor range check" && 0 <= v569 && v569 < 2);
                                                                assert("Tensor range check" && 0 <= v571 && v571 < 2);
                                                                int v573;
                                                                v573 = 4 * v571;
                                                                int v574;
                                                                v574 = v573 + v568;
                                                                int v575;
                                                                v575 = 544 * v569;
                                                                int v576;
                                                                v576 = v575 + v574;
                                                                float v577;
                                                                v577 = v452[v576];
                                                                bool v578;
                                                                v578 = 0 <= v571;
                                                                bool v580;
                                                                if (v578){
                                                                    bool v579;
                                                                    v579 = v571 < 2;
                                                                    v580 = v579;
                                                                } else {
                                                                    v580 = false;
                                                                }
                                                                bool v581;
                                                                v581 = v580 == false;
                                                                if (v581){
                                                                    assert("The indices should be inside the range of the dimension." && v580);
                                                                } else {
                                                                }
                                                                bool v583;
                                                                v583 = 0 <= v569;
                                                                bool v585;
                                                                if (v583){
                                                                    bool v584;
                                                                    v584 = v569 < 2;
                                                                    v585 = v584;
                                                                } else {
                                                                    v585 = false;
                                                                }
                                                                bool v586;
                                                                v586 = v585 == false;
                                                                if (v586){
                                                                    assert("The indices should be inside the range of the dimension." && v585);
                                                                } else {
                                                                }
                                                                int v588;
                                                                v588 = v569 * 2;
                                                                int v589;
                                                                v589 = v571 + v588;
                                                                v565.x[v589] = wmma::__float_to_tf32(v577);
                                                                v571 += 1 ;
                                                            }
                                                            v569 += 1 ;
                                                        }
                                                        v561 += 1 ;
                                                    }
                                                    v559 += 1 ;
                                                }
                                                // Poping the loop unrolling to: 0
                                                v399.consumer_release();
                                                switch (v488.tag) {
                                                    case 0: { // None
                                                        break;
                                                    }
                                                    case 1: { // Some
                                                        int v590 = v488.case1.v0;
                                                        assert("Tensor range check" && 0 <= v590 && v590 < 32);
                                                        int v591;
                                                        v591 = 64 * v590;
                                                        int v592;
                                                        v592 = v591 + v496;
                                                        float * v593;
                                                        v593 = v390+v592;
                                                        asm("barrier.cta.sync %0;" :: "r"(0));
                                                        // Pushing the loop unrolling to: 0
                                                        v399.producer_acquire();
                                                        int v595;
                                                        v595 = threadIdx.x;
                                                        bool v596;
                                                        v596 = 0 <= v595;
                                                        bool v597;
                                                        v597 = v596 == false;
                                                        if (v597){
                                                            assert("The index needs to be zero or positive." && v596);
                                                        } else {
                                                        }
                                                        int v599;
                                                        v599 = v595 % 16;
                                                        int v600;
                                                        v600 = v595 / 16;
                                                        bool v601;
                                                        v601 = v600 < 16;
                                                        bool v602;
                                                        v602 = v601 == false;
                                                        if (v602){
                                                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v601);
                                                        } else {
                                                        }
                                                        assert("Tensor range check" && 0 <= v600 && v600 < 16);
                                                        assert("Tensor range check" && 0 <= v599 && v599 < 16);
                                                        int v604;
                                                        v604 = 4 * v599;
                                                        int v605;
                                                        v605 = 68 * v600;
                                                        int v606;
                                                        v606 = v605 + v604;
                                                        int v607;
                                                        v607 = 2048 * v600;
                                                        int v608;
                                                        v608 = v607 + v604;
                                                        float * v609;
                                                        v609 = v403+v606;
                                                        float * v611;
                                                        v611 = v593+v608;
                                                        int v613;
                                                        v613 = 0;
                                                        #pragma unroll
                                                        while (while_method_1(v613)){
                                                            int v615;
                                                            v615 = 0;
                                                            #pragma unroll
                                                            while (while_method_6(v615)){
                                                                assert("Tensor range check" && 0 <= v613 && v613 < 8);
                                                                assert("Tensor range check" && 0 <= v615 && v615 < 1);
                                                                int v617;
                                                                v617 = 64 * v615;
                                                                int v618;
                                                                v618 = 1088 * v613;
                                                                int v619;
                                                                v619 = v618 + v617;
                                                                int v620;
                                                                v620 = 32768 * v613;
                                                                int v621;
                                                                v621 = v620 + v617;
                                                                constexpr int v622 = sizeof(float) * 4;
                                                                assert("Pointer alignment check" && (unsigned long long)(v611 + v621) % v622 == 0 && (unsigned long long)(v609 + v619) % v622 == 0);
                                                                cuda::memcpy_async(v609 + v619, v611 + v621, cuda::aligned_size_t<v622>(v622), v399);
                                                                v615 += 1 ;
                                                            }
                                                            v613 += 1 ;
                                                        }
                                                        v399.producer_commit();
                                                        // Poping the loop unrolling to: 0
                                                        break;
                                                    }
                                                    default: {
                                                        assert("Invalid tag." && false); __trap();
                                                    }
                                                }
                                                // Pushing the loop unrolling to: 0
                                                int v623;
                                                v623 = 0;
                                                #pragma unroll
                                                while (while_method_1(v623)){
                                                    int v625;
                                                    v625 = 0;
                                                    #pragma unroll
                                                    while (while_method_1(v625)){
                                                        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v627 = v557[0];
                                                        assert("Tensor range check" && 0 <= v623 && v623 < 8);
                                                        int v628;
                                                        v628 = 1088 * v623;
                                                        assert("Tensor range check" && 0 <= v625 && v625 < 8);
                                                        int v629;
                                                        v629 = 8 * v625;
                                                        int v630;
                                                        v630 = v629 + v628;
                                                        int v631;
                                                        v631 = 0;
                                                        #pragma unroll
                                                        while (while_method_2(v631)){
                                                            int v633;
                                                            v633 = 0;
                                                            #pragma unroll
                                                            while (while_method_2(v633)){
                                                                assert("Tensor range check" && 0 <= v631 && v631 < 2);
                                                                assert("Tensor range check" && 0 <= v633 && v633 < 2);
                                                                int v635;
                                                                v635 = 544 * v633;
                                                                int v636;
                                                                v636 = v635 + v630;
                                                                int v637;
                                                                v637 = 4 * v631;
                                                                int v638;
                                                                v638 = v637 + v636;
                                                                float v639;
                                                                v639 = v436[v638];
                                                                bool v640;
                                                                v640 = 0 <= v633;
                                                                bool v642;
                                                                if (v640){
                                                                    bool v641;
                                                                    v641 = v633 < 2;
                                                                    v642 = v641;
                                                                } else {
                                                                    v642 = false;
                                                                }
                                                                bool v643;
                                                                v643 = v642 == false;
                                                                if (v643){
                                                                    assert("The indices should be inside the range of the dimension." && v642);
                                                                } else {
                                                                }
                                                                bool v645;
                                                                v645 = 0 <= v631;
                                                                bool v647;
                                                                if (v645){
                                                                    bool v646;
                                                                    v646 = v631 < 2;
                                                                    v647 = v646;
                                                                } else {
                                                                    v647 = false;
                                                                }
                                                                bool v648;
                                                                v648 = v647 == false;
                                                                if (v648){
                                                                    assert("The indices should be inside the range of the dimension." && v647);
                                                                } else {
                                                                }
                                                                int v650;
                                                                v650 = v631 * 2;
                                                                int v651;
                                                                v651 = v633 + v650;
                                                                v627.x[v651] = wmma::__float_to_tf32(v639);
                                                                v633 += 1 ;
                                                            }
                                                            v631 += 1 ;
                                                        }
                                                        int v652;
                                                        v652 = 0;
                                                        #pragma unroll
                                                        while (while_method_6(v652)){
                                                            assert("Tensor range check" && 0 <= v623 && v623 < 8);
                                                            assert("Tensor range check" && 0 <= v652 && v652 < 1);
                                                            int v654;
                                                            v654 = v623 + v652;
                                                            wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v655 = v454[v654];
                                                            assert("Tensor range check" && 0 <= v652 && v652 < 1);
                                                            assert("Tensor range check" && 0 <= v625 && v625 < 8);
                                                            int v656;
                                                            v656 = 8 * v652;
                                                            int v657;
                                                            v657 = v656 + v625;
                                                            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v658 = v558[v657];
                                                            wmma::mma_sync(v655, v627, v658, v655);
                                                            v652 += 1 ;
                                                        }
                                                        v625 += 1 ;
                                                    }
                                                    v623 += 1 ;
                                                }
                                                // Poping the loop unrolling to: 0
                                                asm("barrier.cta.sync %0;" :: "r"(0));
                                                v471 = v473;
                                            }
                                            // Pushing the loop unrolling to: 0
                                            int v659;
                                            v659 = 0;
                                            #pragma unroll
                                            while (while_method_1(v659)){
                                                int v661;
                                                v661 = 0;
                                                #pragma unroll
                                                while (while_method_6(v661)){
                                                    assert("Tensor range check" && 0 <= v659 && v659 < 8);
                                                    assert("Tensor range check" && 0 <= v661 && v661 < 1);
                                                    int v663;
                                                    v663 = v659 + v661;
                                                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v664 = v454[v663];
                                                    assert("Tensor range check" && 0 <= v659 && v659 < 8);
                                                    assert("Tensor range check" && 0 <= v661 && v661 < 1);
                                                    int v665;
                                                    v665 = 16 * v661;
                                                    int v666;
                                                    v666 = 2176 * v659;
                                                    int v667;
                                                    v667 = v666 + v665;
                                                    float * v668;
                                                    v668 = v420+v667;
                                                    wmma::store_matrix_sync(v668, v664, 136, wmma::mem_row_major);
                                                    v661 += 1 ;
                                                }
                                                v659 += 1 ;
                                            }
                                            // Poping the loop unrolling to: 0
                                            asm("barrier.cta.sync %0;" :: "r"(0));
                                            // Pushing the loop unrolling to: 0
                                            int v670;
                                            v670 = threadIdx.x;
                                            bool v671;
                                            v671 = 0 <= v670;
                                            bool v672;
                                            v672 = v671 == false;
                                            if (v672){
                                                assert("The index needs to be zero or positive." && v671);
                                            } else {
                                            }
                                            int v674;
                                            v674 = v670 % 32;
                                            int v675;
                                            v675 = v670 / 32;
                                            bool v676;
                                            v676 = v675 < 8;
                                            bool v677;
                                            v677 = v676 == false;
                                            if (v677){
                                                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v676);
                                            } else {
                                            }
                                            assert("Tensor range check" && 0 <= v675 && v675 < 8);
                                            assert("Tensor range check" && 0 <= v674 && v674 < 32);
                                            int v679;
                                            v679 = 4 * v674;
                                            int v680;
                                            v680 = 128 * v675;
                                            int v681;
                                            v681 = v680 + v679;
                                            int v682;
                                            v682 = 136 * v675;
                                            int v683;
                                            v683 = v682 + v679;
                                            float * v684;
                                            v684 = v463+v681;
                                            float * v686;
                                            v686 = v405+v683;
                                            int v688;
                                            v688 = 0;
                                            #pragma unroll
                                            while (while_method_12(v688)){
                                                int v690;
                                                v690 = 0;
                                                #pragma unroll
                                                while (while_method_6(v690)){
                                                    assert("Tensor range check" && 0 <= v688 && v688 < 16);
                                                    assert("Tensor range check" && 0 <= v690 && v690 < 1);
                                                    int v692;
                                                    v692 = 128 * v690;
                                                    int v693;
                                                    v693 = 1024 * v688;
                                                    int v694;
                                                    v694 = v693 + v692;
                                                    int v695;
                                                    v695 = 1088 * v688;
                                                    int v696;
                                                    v696 = v695 + v692;
                                                    int4* v697;
                                                    v697 = reinterpret_cast<int4*>(v686 + v696);
                                                    int4* v698;
                                                    v698 = reinterpret_cast<int4*>(v684 + v694);
                                                    assert("Pointer alignment check" && (unsigned long long)(v697) % 4 == 0 && (unsigned long long)(v698) % 4 == 0);
                                                    *v698 = *v697;
                                                    v690 += 1 ;
                                                }
                                                v688 += 1 ;
                                            }
                                            // Poping the loop unrolling to: 0
                                            asm("barrier.cta.sync %0;" :: "r"(0));
                                            v457 += 1 ;
                                        }
                                        v455 += 1 ;
                                    }
                                    unsigned int * v699;
                                    v699 = reinterpret_cast<unsigned int *>(&v0[53477376ull]);
                                    assert("Tensor range check" && 0 <= v386 && v386 < 4);
                                    int v701;
                                    v701 = 6144 * v386;
                                    method_13(v699, v701, v393);
                                    int * v702;
                                    v702 = reinterpret_cast<int *>(&v1[4194304ull]);
                                    float * v704;
                                    v704 = reinterpret_cast<float *>(&v1[4194320ull]);
                                    float * v706;
                                    v706 = reinterpret_cast<float *>(&v1[5242896ull]);
                                    float * v708;
                                    v708 = reinterpret_cast<float *>(&v1[6291472ull]);
                                    float * v710;
                                    v710 = reinterpret_cast<float *>(&v1[7340048ull]);
                                    float * v712;
                                    v712 = reinterpret_cast<float *>(&v1[8388624ull]);
                                    float * v714;
                                    v714 = reinterpret_cast<float *>(&v1[9437200ull]);
                                    float * v716;
                                    v716 = reinterpret_cast<float *>(&v1[10485776ull]);
                                    int * v718;
                                    v718 = reinterpret_cast<int *>(&v0[53575680ull]);
                                    float * v720;
                                    v720 = reinterpret_cast<float *>(&v0[66158592ull]);
                                    int * v722;
                                    v722 = reinterpret_cast<int *>(&v0[78741504ull]);
                                    int * v724;
                                    v724 = reinterpret_cast<int *>(&v0[91324416ull]);
                                    double * v726;
                                    v726 = reinterpret_cast<double *>(&v0[103907328ull]);
                                    double * v728;
                                    v728 = reinterpret_cast<double *>(&v0[154238976ull]);
                                    double * v730;
                                    v730 = reinterpret_cast<double *>(&v1[11534352ull]);
                                    double * v732;
                                    v732 = reinterpret_cast<double *>(&v1[11927568ull]);
                                    int * v734;
                                    v734 = reinterpret_cast<int *>(&v1[12320784ull]);
                                    v386 += 1 ;
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                int * v736;
                                v736 = reinterpret_cast<int *>(&v1[4194304ull]);
                                float * v738;
                                v738 = reinterpret_cast<float *>(&v1[4194320ull]);
                                float * v740;
                                v740 = reinterpret_cast<float *>(&v1[5242896ull]);
                                float * v742;
                                v742 = reinterpret_cast<float *>(&v1[6291472ull]);
                                float * v744;
                                v744 = reinterpret_cast<float *>(&v1[7340048ull]);
                                float * v746;
                                v746 = reinterpret_cast<float *>(&v1[8388624ull]);
                                float * v748;
                                v748 = reinterpret_cast<float *>(&v1[9437200ull]);
                                float * v750;
                                v750 = reinterpret_cast<float *>(&v1[10485776ull]);
                                int v752;
                                v752 = v736[0];
                                unsigned int * v753;
                                v753 = reinterpret_cast<unsigned int *>(&v0[53477376ull]);
                                int v755;
                                v755 = blockIdx.x;
                                int v756;
                                v756 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v752 && v752 < 4);
                                assert("Tensor range check" && 0 <= v755 && v755 < 24);
                                assert("Tensor range check" && 0 <= v756 && v756 < 256);
                                int v757;
                                v757 = 256 * v755;
                                int v758;
                                v758 = v757 + v756;
                                int v759;
                                v759 = 6144 * v752;
                                int v760;
                                v760 = v759 + v758;
                                unsigned int v761;
                                v761 = v753[v760];
                                int v762;
                                v762 = (int)v761;
                                float v763; int v764;
                                Tuple8 tmp24 = method_14(v171, v736, v738, v740, v742, v744, v746, v748, v750, v762, v752);
                                v763 = tmp24.v0; v764 = tmp24.v1;
                                extern __shared__ unsigned char v765[];
                                float * v766;
                                v766 = reinterpret_cast<float *>(&v765[0ull]);
                                int * v768;
                                v768 = reinterpret_cast<int *>(&v765[16ull]);
                                int v770;
                                v770 = threadIdx.x;
                                bool v771;
                                v771 = v770 == 0;
                                if (v771){
                                    v766[0] = v763;
                                    v768[0] = v764;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                float v772;
                                v772 = v766[0];
                                int v773;
                                v773 = v768[0];
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                double * v774;
                                v774 = reinterpret_cast<double *>(&v1[11534352ull]);
                                double * v776;
                                v776 = reinterpret_cast<double *>(&v1[11927568ull]);
                                int * v778;
                                v778 = reinterpret_cast<int *>(&v1[12320784ull]);
                                int * v780;
                                v780 = reinterpret_cast<int *>(&v0[53575680ull]);
                                float * v782;
                                v782 = reinterpret_cast<float *>(&v0[66158592ull]);
                                int * v784;
                                v784 = reinterpret_cast<int *>(&v0[78741504ull]);
                                int * v786;
                                v786 = reinterpret_cast<int *>(&v0[91324416ull]);
                                double * v788;
                                v788 = reinterpret_cast<double *>(&v0[103907328ull]);
                                double * v790;
                                v790 = reinterpret_cast<double *>(&v0[154238976ull]);
                                int v792;
                                v792 = threadIdx.x;
                                int v793;
                                v793 = blockIdx.x;
                                int v794;
                                v794 = v793 * 256;
                                int v795;
                                v795 = v792 + v794;
                                int v796;
                                v796 = 0;
                                while (while_method_0(v796)){
                                    unsigned int * v798;
                                    v798 = reinterpret_cast<unsigned int *>(&v0[53477376ull]);
                                    int v800;
                                    v800 = blockIdx.x;
                                    int v801;
                                    v801 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v796 && v796 < 4);
                                    assert("Tensor range check" && 0 <= v800 && v800 < 24);
                                    assert("Tensor range check" && 0 <= v801 && v801 < 256);
                                    int v802;
                                    v802 = 256 * v800;
                                    int v803;
                                    v803 = v802 + v801;
                                    int v804;
                                    v804 = 6144 * v796;
                                    int v805;
                                    v805 = v804 + v803;
                                    unsigned int v806;
                                    v806 = v798[v805];
                                    int v807;
                                    v807 = (int)v806;
                                    float v808;
                                    v808 = method_15(v736, v738, v740, v742, v744, v746, v748, v750, v807, v796, v773);
                                    assert("Tensor range check" && 0 <= v796 && v796 < 4);
                                    assert("Tensor range check" && 0 <= v795 && v795 < 6144);
                                    int v809;
                                    v809 = v804 + v795;
                                    int v810;
                                    v810 = v778[v809];
                                    int v811;
                                    v811 = v810 + 1;
                                    assert("Tensor range check" && 0 <= v796 && v796 < 4);
                                    assert("Tensor range check" && 0 <= v795 && v795 < 6144);
                                    v778[v809] = v811;
                                    assert("Tensor range check" && 0 <= v796 && v796 < 4);
                                    assert("Tensor range check" && 0 <= v810 && v810 < 128);
                                    assert("Tensor range check" && 0 <= v795 && v795 < 6144);
                                    int v812;
                                    v812 = 6144 * v810;
                                    int v813;
                                    v813 = v812 + v795;
                                    int v814;
                                    v814 = 786432 * v796;
                                    int v815;
                                    v815 = v814 + v813;
                                    v780[v815] = v773;
                                    v782[v815] = v772;
                                    v784[v815] = v159;
                                    v786[v815] = v807;
                                    assert("Tensor range check" && 0 <= v796 && v796 < 4);
                                    int v816;
                                    v816 = 12288 * v796;
                                    assert("Tensor range check" && 0 <= v795 && v795 < 6144);
                                    int v817;
                                    v817 = 2 * v795;
                                    int v818;
                                    v818 = v817 + v816;
                                    assert("Tensor range check" && 0 <= v796 && v796 < 4);
                                    int v819;
                                    v819 = 1572864 * v796;
                                    assert("Tensor range check" && 0 <= v810 && v810 < 128);
                                    int v820;
                                    v820 = 12288 * v810;
                                    int v821;
                                    v821 = v820 + v819;
                                    assert("Tensor range check" && 0 <= v795 && v795 < 6144);
                                    int v822;
                                    v822 = v817 + v821;
                                    double * v823;
                                    v823 = v774+v818;
                                    double * v825;
                                    v825 = v776+v818;
                                    double * v827;
                                    v827 = v788+v822;
                                    double * v829;
                                    v829 = v790+v822;
                                    int v831;
                                    v831 = sizeof(double *);
                                    unsigned long long v832;
                                    v832 = (unsigned long long)v831;
                                    unsigned long long v833;
                                    v833 = 256ull * v832;
                                    unsigned long long v834;
                                    v834 = v833 + 16ull;
                                    unsigned long long v835;
                                    v835 = v834 - 1ull;
                                    unsigned long long v836;
                                    v836 = v835 % 16ull;
                                    unsigned long long v837;
                                    v837 = v835 - v836;
                                    unsigned long long v838;
                                    v838 = v837 + v833;
                                    unsigned long long v839;
                                    v839 = v838 + 16ull;
                                    unsigned long long v840;
                                    v840 = v839 - 1ull;
                                    unsigned long long v841;
                                    v841 = v840 % 16ull;
                                    unsigned long long v842;
                                    v842 = v840 - v841;
                                    unsigned long long v843;
                                    v843 = v842 + v833;
                                    unsigned long long v844;
                                    v844 = v843 + 16ull;
                                    unsigned long long v845;
                                    v845 = v844 - 1ull;
                                    unsigned long long v846;
                                    v846 = v845 % 16ull;
                                    unsigned long long v847;
                                    v847 = v845 - v846;
                                    unsigned long long v848;
                                    v848 = v847 + v833;
                                    bool v849;
                                    v849 = v848 <= 98304ull;
                                    bool v850;
                                    v850 = v849 == false;
                                    if (v850){
                                        assert("The dynamic shared memory is insufficient to allocate the tensor." && v849);
                                    } else {
                                    }
                                    extern __shared__ unsigned char v852[];
                                    bool v853;
                                    v853 = v848 <= v848;
                                    bool v854;
                                    v854 = v853 == false;
                                    if (v854){
                                        assert("The length of the partition has to be less than or equal to the length of the base array." && v853);
                                    } else {
                                    }
                                    double * * v856;
                                    v856 = reinterpret_cast<double * *>(&v852[0ull]);
                                    double * * v858;
                                    v858 = reinterpret_cast<double * *>(&v852[v837]);
                                    double * * v860;
                                    v860 = reinterpret_cast<double * *>(&v852[v842]);
                                    double * * v862;
                                    v862 = reinterpret_cast<double * *>(&v852[v847]);
                                    int v864;
                                    v864 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v864 && v864 < 256);
                                    v856[v864] = v823;
                                    v858[v864] = v825;
                                    v860[v864] = v827;
                                    v862[v864] = v829;
                                    asm("barrier.cta.sync %0;" :: "r"(0));
                                    bool v865;
                                    v865 = 0 <= v864;
                                    bool v866;
                                    v866 = v865 == false;
                                    if (v866){
                                        assert("The index needs to be zero or positive." && v865);
                                    } else {
                                    }
                                    int v868;
                                    v868 = v864 % 1;
                                    bool v869;
                                    v869 = v864 < 256;
                                    bool v870;
                                    v870 = v869 == false;
                                    if (v870){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v869);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v864 && v864 < 256);
                                    int v872;
                                    v872 = 0;
                                    while (while_method_6(v872)){
                                        bool v874;
                                        v874 = v865 && v869;
                                        bool v875;
                                        v875 = v874 == false;
                                        if (v875){
                                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v874);
                                        } else {
                                        }
                                        bool v877;
                                        v877 = 0 <= v872;
                                        bool v879;
                                        if (v877){
                                            bool v878;
                                            v878 = v872 < 1;
                                            v879 = v878;
                                        } else {
                                            v879 = false;
                                        }
                                        bool v880;
                                        v880 = v879 == false;
                                        if (v880){
                                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v879);
                                        } else {
                                        }
                                        int v882;
                                        v882 = v872 * 256;
                                        int v883;
                                        v883 = v882 + v864;
                                        assert("Tensor range check" && 0 <= v872 && v872 < 1);
                                        int v884;
                                        v884 = 256 * v872;
                                        int v885;
                                        v885 = v884 + v864;
                                        double * v886;
                                        v886 = v856[v885];
                                        double * v887;
                                        v887 = v858[v885];
                                        double * v888;
                                        v888 = v860[v885];
                                        double * v889;
                                        v889 = v862[v885];
                                        int v890;
                                        v890 = blockIdx.x;
                                        int v891;
                                        v891 = v890 * 256;
                                        int v892;
                                        v892 = v891 + v883;
                                        assert("Tensor range check" && 0 <= v868 && v868 < 1);
                                        int v893;
                                        v893 = 2 * v868;
                                        double v894[2];
                                        double v895[2];
                                        int v896[2];
                                        int v897;
                                        v897 = 0;
                                        while (while_method_6(v897)){
                                            assert("Tensor range check" && 0 <= v897 && v897 < 1);
                                            int v899;
                                            v899 = 2 * v897;
                                            assert("Tensor range check" && 0 <= v897 && v897 < 1);
                                            int v900;
                                            v900 = v899 + v893;
                                            int4* v901;
                                            v901 = reinterpret_cast<int4*>(v886 + v900);
                                            int4* v902;
                                            v902 = reinterpret_cast<int4*>(v894 + v899);
                                            assert("Pointer alignment check" && (unsigned long long)(v901) % 2 == 0 && (unsigned long long)(v902) % 2 == 0);
                                            *v902 = *v901;
                                            int4* v903;
                                            v903 = reinterpret_cast<int4*>(v887 + v900);
                                            int4* v904;
                                            v904 = reinterpret_cast<int4*>(v895 + v899);
                                            assert("Pointer alignment check" && (unsigned long long)(v903) % 2 == 0 && (unsigned long long)(v904) % 2 == 0);
                                            *v904 = *v903;
                                            v897 += 1 ;
                                        }
                                        int v905;
                                        v905 = 0;
                                        while (while_method_6(v905)){
                                            int v907;
                                            v907 = 0;
                                            while (while_method_2(v907)){
                                                bool v909;
                                                v909 = 0 <= v907;
                                                bool v911;
                                                if (v909){
                                                    bool v910;
                                                    v910 = v907 < 2;
                                                    v911 = v910;
                                                } else {
                                                    v911 = false;
                                                }
                                                bool v912;
                                                v912 = v911 == false;
                                                if (v912){
                                                    assert("The indices should be inside the range of the dimension." && v911);
                                                } else {
                                                }
                                                bool v914;
                                                v914 = 0 <= v868;
                                                bool v916;
                                                if (v914){
                                                    bool v915;
                                                    v915 = v868 < 1;
                                                    v916 = v915;
                                                } else {
                                                    v916 = false;
                                                }
                                                bool v917;
                                                v917 = v916 == false;
                                                if (v917){
                                                    assert("The indices should be inside the range of the dimension." && v916);
                                                } else {
                                                }
                                                int v919;
                                                v919 = v868 * 2;
                                                int v920;
                                                v920 = v907 + v919;
                                                bool v921;
                                                v921 = 0 <= v905;
                                                bool v923;
                                                if (v921){
                                                    bool v922;
                                                    v922 = v905 < 1;
                                                    v923 = v922;
                                                } else {
                                                    v923 = false;
                                                }
                                                bool v924;
                                                v924 = v923 == false;
                                                if (v924){
                                                    assert("The indices should be inside the range of the dimension." && v923);
                                                } else {
                                                }
                                                int v926;
                                                v926 = v905 * 2;
                                                int v927;
                                                v927 = v920 + v926;
                                                assert("Tensor range check" && 0 <= v905 && v905 < 1);
                                                assert("Tensor range check" && 0 <= v907 && v907 < 2);
                                                int v928;
                                                v928 = 2 * v905;
                                                int v929;
                                                v929 = v928 + v907;
                                                v896[v929] = v927;
                                                v907 += 1 ;
                                            }
                                            v905 += 1 ;
                                        }
                                        int v930;
                                        v930 = 0;
                                        while (while_method_6(v930)){
                                            assert("Tensor range check" && 0 <= v930 && v930 < 1);
                                            int v932;
                                            v932 = 2 * v930;
                                            int v933;
                                            v933 = v932 + v893;
                                            assert("Tensor range check" && 0 <= v930 && v930 < 1);
                                            int4* v934;
                                            v934 = reinterpret_cast<int4*>(v894 + v932);
                                            int4* v935;
                                            v935 = reinterpret_cast<int4*>(v888 + v933);
                                            assert("Pointer alignment check" && (unsigned long long)(v934) % 2 == 0 && (unsigned long long)(v935) % 2 == 0);
                                            *v935 = *v934;
                                            int4* v936;
                                            v936 = reinterpret_cast<int4*>(v895 + v932);
                                            int4* v937;
                                            v937 = reinterpret_cast<int4*>(v889 + v933);
                                            assert("Pointer alignment check" && (unsigned long long)(v936) % 2 == 0 && (unsigned long long)(v937) % 2 == 0);
                                            *v937 = *v936;
                                            v930 += 1 ;
                                        }
                                        assert("Tensor range check" && 0 <= v883 && v883 < 256);
                                        v872 += 1 ;
                                    }
                                    asm("barrier.cta.sync %0;" :: "r"(0));
                                    assert("Tensor range check" && 0 <= v864 && v864 < 256);
                                    asm("barrier.cta.sync %0;" :: "r"(0));
                                    double v938;
                                    v938 = (double)v772;
                                    double v939;
                                    v939 = log(v938);
                                    double v940;
                                    v940 = (double)v808;
                                    double v941;
                                    v941 = log(v940);
                                    assert("Tensor range check" && 0 <= v796 && v796 < 4);
                                    assert("Tensor range check" && 0 <= v795 && v795 < 6144);
                                    assert("Tensor range check" && 0 <= v159 && v159 < 2);
                                    int v942;
                                    v942 = v817 + v159;
                                    int v943;
                                    v943 = v816 + v942;
                                    double v944;
                                    v944 = v774[v943];
                                    double v945;
                                    v945 = v776[v943];
                                    double v946;
                                    v946 = v941 + v944;
                                    double v947;
                                    v947 = v939 + v945;
                                    assert("Tensor range check" && 0 <= v796 && v796 < 4);
                                    assert("Tensor range check" && 0 <= v795 && v795 < 6144);
                                    assert("Tensor range check" && 0 <= v159 && v159 < 2);
                                    v774[v943] = v946;
                                    v776[v943] = v947;
                                    v796 += 1 ;
                                }
                                bool v948;
                                v948 = 0 == v773;
                                Union9 v981;
                                if (v948){
                                    v981 = Union9{Union9_1{}};
                                } else {
                                    bool v950;
                                    v950 = 1 == v773;
                                    if (v950){
                                        v981 = Union9{Union9_0{}};
                                    } else {
                                        bool v952;
                                        v952 = 2 == v773;
                                        if (v952){
                                            v981 = Union9{Union9_2{1, 3}};
                                        } else {
                                            bool v954;
                                            v954 = 3 == v773;
                                            if (v954){
                                                v981 = Union9{Union9_2{1, 2}};
                                            } else {
                                                bool v956;
                                                v956 = 4 == v773;
                                                if (v956){
                                                    v981 = Union9{Union9_2{2, 3}};
                                                } else {
                                                    bool v958;
                                                    v958 = 5 == v773;
                                                    if (v958){
                                                        v981 = Union9{Union9_2{3, 4}};
                                                    } else {
                                                        bool v960;
                                                        v960 = 6 == v773;
                                                        if (v960){
                                                            v981 = Union9{Union9_2{1, 1}};
                                                        } else {
                                                            bool v962;
                                                            v962 = 7 == v773;
                                                            if (v962){
                                                                v981 = Union9{Union9_2{3, 2}};
                                                            } else {
                                                                bool v964;
                                                                v964 = 8 == v773;
                                                                if (v964){
                                                                    v981 = Union9{Union9_2{2, 1}};
                                                                } else {
                                                                    bool v966;
                                                                    v966 = 9 == v773;
                                                                    if (v966){
                                                                        v981 = Union9{Union9_2{3, 1}};
                                                                    } else {
                                                                        bool v968;
                                                                        v968 = 10 == v773;
                                                                        if (v968){
                                                                            v981 = Union9{Union9_2{2147483647, 1}};
                                                                        } else {
                                                                            printf("%s\n", "Invalid output id in the NL Holdem model.");
                                                                            __trap();
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                switch (v981.tag) {
                                    case 0: { // AA_Call
                                        v1267 = Union2{Union2_1{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v982;
                                        v982 = v155[0];
                                        int v984; int v985;
                                        Tuple4 tmp27 = Tuple4{1, v982};
                                        v984 = tmp27.v0; v985 = tmp27.v1;
                                        while (while_method_2(v984)){
                                            bool v987;
                                            v987 = 0 <= v984;
                                            bool v989;
                                            if (v987){
                                                bool v988;
                                                v988 = v984 < 2;
                                                v989 = v988;
                                            } else {
                                                v989 = false;
                                            }
                                            bool v990;
                                            v990 = v989 == false;
                                            if (v990){
                                                assert("Index must be in range." && v989);
                                            } else {
                                            }
                                            int v992;
                                            v992 = v155[v984];
                                            bool v994;
                                            v994 = v985 >= v992;
                                            int v995;
                                            if (v994){
                                                v995 = v985;
                                            } else {
                                                v995 = v992;
                                            }
                                            v985 = v995;
                                            v984 += 1 ;
                                        }
                                        bool v997;
                                        if (v161){
                                            bool v996;
                                            v996 = v159 < 2;
                                            v997 = v996;
                                        } else {
                                            v997 = false;
                                        }
                                        bool v998;
                                        v998 = v997 == false;
                                        if (v998){
                                            assert("Index must be in range." && v997);
                                        } else {
                                        }
                                        int v1000;
                                        v1000 = v155[v159];
                                        bool v1002;
                                        v1002 = v1000 == v985;
                                        if (v1002){
                                            v1267 = Union2{Union2_1{}};
                                        } else {
                                            v1267 = Union2{Union2_2{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        int v1007 = v981.case2.v0; int v1008 = v981.case2.v1;
                                        static_array<int,2> v1009;
                                        int v1011;
                                        v1011 = 0;
                                        while (while_method_2(v1011)){
                                            bool v1013;
                                            v1013 = 0 <= v1011;
                                            bool v1015;
                                            if (v1013){
                                                bool v1014;
                                                v1014 = v1011 < 2;
                                                v1015 = v1014;
                                            } else {
                                                v1015 = false;
                                            }
                                            bool v1016;
                                            v1016 = v1015 == false;
                                            if (v1016){
                                                assert("Index must be in range." && v1015);
                                            } else {
                                            }
                                            int v1018;
                                            v1018 = v157[v1011];
                                            bool v1021;
                                            if (v1013){
                                                bool v1020;
                                                v1020 = v1011 < 2;
                                                v1021 = v1020;
                                            } else {
                                                v1021 = false;
                                            }
                                            bool v1022;
                                            v1022 = v1021 == false;
                                            if (v1022){
                                                assert("Index must be in range." && v1021);
                                            } else {
                                            }
                                            int v1024;
                                            v1024 = v155[v1011];
                                            int v1026;
                                            v1026 = v1018 + v1024;
                                            v1009[v1011] = v1026;
                                            v1011 += 1 ;
                                        }
                                        int v1027;
                                        v1027 = v155[0];
                                        int v1029; int v1030;
                                        Tuple4 tmp28 = Tuple4{1, v1027};
                                        v1029 = tmp28.v0; v1030 = tmp28.v1;
                                        while (while_method_2(v1029)){
                                            bool v1032;
                                            v1032 = 0 <= v1029;
                                            bool v1034;
                                            if (v1032){
                                                bool v1033;
                                                v1033 = v1029 < 2;
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
                                            int v1037;
                                            v1037 = v155[v1029];
                                            bool v1039;
                                            v1039 = v1030 >= v1037;
                                            int v1040;
                                            if (v1039){
                                                v1040 = v1030;
                                            } else {
                                                v1040 = v1037;
                                            }
                                            v1030 = v1040;
                                            v1029 += 1 ;
                                        }
                                        bool v1042;
                                        if (v161){
                                            bool v1041;
                                            v1041 = v159 < 2;
                                            v1042 = v1041;
                                        } else {
                                            v1042 = false;
                                        }
                                        bool v1043;
                                        v1043 = v1042 == false;
                                        if (v1043){
                                            assert("Index must be in range." && v1042);
                                        } else {
                                        }
                                        int v1045;
                                        v1045 = v1009[v159];
                                        bool v1047;
                                        v1047 = v1030 < v1045;
                                        int v1048;
                                        if (v1047){
                                            v1048 = v1030;
                                        } else {
                                            v1048 = v1045;
                                        }
                                        static_array<int,2> v1049;
                                        int v1051;
                                        v1051 = 0;
                                        while (while_method_2(v1051)){
                                            bool v1053;
                                            v1053 = 0 <= v1051;
                                            bool v1055;
                                            if (v1053){
                                                bool v1054;
                                                v1054 = v1051 < 2;
                                                v1055 = v1054;
                                            } else {
                                                v1055 = false;
                                            }
                                            bool v1056;
                                            v1056 = v1055 == false;
                                            if (v1056){
                                                assert("Index must be in range." && v1055);
                                            } else {
                                            }
                                            int v1058;
                                            v1058 = v155[v1051];
                                            bool v1060;
                                            v1060 = v159 == v1051;
                                            int v1061;
                                            if (v1060){
                                                v1061 = v1048;
                                            } else {
                                                v1061 = v1058;
                                            }
                                            v1049[v1051] = v1061;
                                            v1051 += 1 ;
                                        }
                                        int v1062;
                                        v1062 = v1049[0];
                                        int v1064; int v1065;
                                        Tuple4 tmp29 = Tuple4{1, v1062};
                                        v1064 = tmp29.v0; v1065 = tmp29.v1;
                                        while (while_method_2(v1064)){
                                            bool v1067;
                                            v1067 = 0 <= v1064;
                                            bool v1069;
                                            if (v1067){
                                                bool v1068;
                                                v1068 = v1064 < 2;
                                                v1069 = v1068;
                                            } else {
                                                v1069 = false;
                                            }
                                            bool v1070;
                                            v1070 = v1069 == false;
                                            if (v1070){
                                                assert("Index must be in range." && v1069);
                                            } else {
                                            }
                                            int v1072;
                                            v1072 = v1049[v1064];
                                            int v1074;
                                            v1074 = v1065 + v1072;
                                            v1065 = v1074;
                                            v1064 += 1 ;
                                        }
                                        static_array<int,2> v1075;
                                        int v1077;
                                        v1077 = 0;
                                        while (while_method_2(v1077)){
                                            bool v1079;
                                            v1079 = 0 <= v1077;
                                            bool v1081;
                                            if (v1079){
                                                bool v1080;
                                                v1080 = v1077 < 2;
                                                v1081 = v1080;
                                            } else {
                                                v1081 = false;
                                            }
                                            bool v1082;
                                            v1082 = v1081 == false;
                                            if (v1082){
                                                assert("Index must be in range." && v1081);
                                            } else {
                                            }
                                            int v1084;
                                            v1084 = v1009[v1077];
                                            bool v1087;
                                            if (v1079){
                                                bool v1086;
                                                v1086 = v1077 < 2;
                                                v1087 = v1086;
                                            } else {
                                                v1087 = false;
                                            }
                                            bool v1088;
                                            v1088 = v1087 == false;
                                            if (v1088){
                                                assert("Index must be in range." && v1087);
                                            } else {
                                            }
                                            int v1090;
                                            v1090 = v1049[v1077];
                                            int v1092;
                                            v1092 = v1084 - v1090;
                                            v1075[v1077] = v1092;
                                            v1077 += 1 ;
                                        }
                                        int v1093;
                                        v1093 = v1007 * v1065;
                                        int v1094;
                                        v1094 = v1093 / v1008;
                                        bool v1095;
                                        v1095 = v153 >= v1094;
                                        int v1096;
                                        if (v1095){
                                            v1096 = v153;
                                        } else {
                                            v1096 = v1094;
                                        }
                                        bool v1098;
                                        if (v161){
                                            bool v1097;
                                            v1097 = v159 < 2;
                                            v1098 = v1097;
                                        } else {
                                            v1098 = false;
                                        }
                                        bool v1099;
                                        v1099 = v1098 == false;
                                        if (v1099){
                                            assert("Index must be in range." && v1098);
                                        } else {
                                        }
                                        int v1101;
                                        v1101 = v1075[v159];
                                        bool v1103;
                                        v1103 = v1096 >= v1101;
                                        if (v1103){
                                            v1267 = Union2{Union2_0{}};
                                        } else {
                                            v1267 = Union2{Union2_3{v1096}};
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                break;
                            }
                            case 1: { // Human
                                printf("%s\n", "Humans aren't allowed during training.");
                                __trap();
                                break;
                            }
                            case 2: { // Random
                                curandStatePhilox4_32_10_t & v1110 = v2.v5;
                                curandStatePhilox4_32_10_t & v1111 = v1110;
                                static_array<int,2> v1112;
                                int v1114;
                                v1114 = 0;
                                while (while_method_2(v1114)){
                                    bool v1116;
                                    v1116 = 0 <= v1114;
                                    bool v1118;
                                    if (v1116){
                                        bool v1117;
                                        v1117 = v1114 < 2;
                                        v1118 = v1117;
                                    } else {
                                        v1118 = false;
                                    }
                                    bool v1119;
                                    v1119 = v1118 == false;
                                    if (v1119){
                                        assert("Index must be in range." && v1118);
                                    } else {
                                    }
                                    int v1121;
                                    v1121 = v157[v1114];
                                    bool v1124;
                                    if (v1116){
                                        bool v1123;
                                        v1123 = v1114 < 2;
                                        v1124 = v1123;
                                    } else {
                                        v1124 = false;
                                    }
                                    bool v1125;
                                    v1125 = v1124 == false;
                                    if (v1125){
                                        assert("Index must be in range." && v1124);
                                    } else {
                                    }
                                    int v1127;
                                    v1127 = v155[v1114];
                                    int v1129;
                                    v1129 = v1121 + v1127;
                                    v1112[v1114] = v1129;
                                    v1114 += 1 ;
                                }
                                int v1130;
                                v1130 = v155[0];
                                int v1132; int v1133;
                                Tuple4 tmp30 = Tuple4{1, v1130};
                                v1132 = tmp30.v0; v1133 = tmp30.v1;
                                while (while_method_2(v1132)){
                                    bool v1135;
                                    v1135 = 0 <= v1132;
                                    bool v1137;
                                    if (v1135){
                                        bool v1136;
                                        v1136 = v1132 < 2;
                                        v1137 = v1136;
                                    } else {
                                        v1137 = false;
                                    }
                                    bool v1138;
                                    v1138 = v1137 == false;
                                    if (v1138){
                                        assert("Index must be in range." && v1137);
                                    } else {
                                    }
                                    int v1140;
                                    v1140 = v155[v1132];
                                    bool v1142;
                                    v1142 = v1133 >= v1140;
                                    int v1143;
                                    if (v1142){
                                        v1143 = v1133;
                                    } else {
                                        v1143 = v1140;
                                    }
                                    v1133 = v1143;
                                    v1132 += 1 ;
                                }
                                bool v1145;
                                if (v161){
                                    bool v1144;
                                    v1144 = v159 < 2;
                                    v1145 = v1144;
                                } else {
                                    v1145 = false;
                                }
                                bool v1146;
                                v1146 = v1145 == false;
                                if (v1146){
                                    assert("Index must be in range." && v1145);
                                } else {
                                }
                                int v1148;
                                v1148 = v1112[v159];
                                bool v1150;
                                v1150 = v1133 < v1148;
                                int v1151;
                                if (v1150){
                                    v1151 = v1133;
                                } else {
                                    v1151 = v1148;
                                }
                                static_array<int,2> v1152;
                                int v1154;
                                v1154 = 0;
                                while (while_method_2(v1154)){
                                    bool v1156;
                                    v1156 = 0 <= v1154;
                                    bool v1158;
                                    if (v1156){
                                        bool v1157;
                                        v1157 = v1154 < 2;
                                        v1158 = v1157;
                                    } else {
                                        v1158 = false;
                                    }
                                    bool v1159;
                                    v1159 = v1158 == false;
                                    if (v1159){
                                        assert("Index must be in range." && v1158);
                                    } else {
                                    }
                                    int v1161;
                                    v1161 = v155[v1154];
                                    bool v1163;
                                    v1163 = v159 == v1154;
                                    int v1164;
                                    if (v1163){
                                        v1164 = v1151;
                                    } else {
                                        v1164 = v1161;
                                    }
                                    v1152[v1154] = v1164;
                                    v1154 += 1 ;
                                }
                                int v1165;
                                v1165 = v1152[0];
                                int v1167; int v1168;
                                Tuple4 tmp31 = Tuple4{1, v1165};
                                v1167 = tmp31.v0; v1168 = tmp31.v1;
                                while (while_method_2(v1167)){
                                    bool v1170;
                                    v1170 = 0 <= v1167;
                                    bool v1172;
                                    if (v1170){
                                        bool v1171;
                                        v1171 = v1167 < 2;
                                        v1172 = v1171;
                                    } else {
                                        v1172 = false;
                                    }
                                    bool v1173;
                                    v1173 = v1172 == false;
                                    if (v1173){
                                        assert("Index must be in range." && v1172);
                                    } else {
                                    }
                                    int v1175;
                                    v1175 = v1152[v1167];
                                    int v1177;
                                    v1177 = v1168 + v1175;
                                    v1168 = v1177;
                                    v1167 += 1 ;
                                }
                                static_array<int,2> v1178;
                                int v1180;
                                v1180 = 0;
                                while (while_method_2(v1180)){
                                    bool v1182;
                                    v1182 = 0 <= v1180;
                                    bool v1184;
                                    if (v1182){
                                        bool v1183;
                                        v1183 = v1180 < 2;
                                        v1184 = v1183;
                                    } else {
                                        v1184 = false;
                                    }
                                    bool v1185;
                                    v1185 = v1184 == false;
                                    if (v1185){
                                        assert("Index must be in range." && v1184);
                                    } else {
                                    }
                                    int v1187;
                                    v1187 = v1112[v1180];
                                    bool v1190;
                                    if (v1182){
                                        bool v1189;
                                        v1189 = v1180 < 2;
                                        v1190 = v1189;
                                    } else {
                                        v1190 = false;
                                    }
                                    bool v1191;
                                    v1191 = v1190 == false;
                                    if (v1191){
                                        assert("Index must be in range." && v1190);
                                    } else {
                                    }
                                    int v1193;
                                    v1193 = v1152[v1180];
                                    int v1195;
                                    v1195 = v1187 - v1193;
                                    v1178[v1180] = v1195;
                                    v1180 += 1 ;
                                }
                                bool v1197;
                                if (v161){
                                    bool v1196;
                                    v1196 = v159 < 2;
                                    v1197 = v1196;
                                } else {
                                    v1197 = false;
                                }
                                bool v1198;
                                v1198 = v1197 == false;
                                if (v1198){
                                    assert("Index must be in range." && v1197);
                                } else {
                                }
                                int v1200;
                                v1200 = v155[v159];
                                bool v1202;
                                v1202 = v1200 < v1133;
                                float v1203;
                                if (v1202){
                                    v1203 = 1.0f;
                                } else {
                                    v1203 = 0.0f;
                                }
                                int v1204;
                                v1204 = v1168 / 3;
                                bool v1205;
                                v1205 = v153 <= v1204;
                                bool v1213;
                                if (v1205){
                                    bool v1207;
                                    if (v161){
                                        bool v1206;
                                        v1206 = v159 < 2;
                                        v1207 = v1206;
                                    } else {
                                        v1207 = false;
                                    }
                                    bool v1208;
                                    v1208 = v1207 == false;
                                    if (v1208){
                                        assert("Index must be in range." && v1207);
                                    } else {
                                    }
                                    int v1210;
                                    v1210 = v1178[v159];
                                    bool v1212;
                                    v1212 = v1204 < v1210;
                                    v1213 = v1212;
                                } else {
                                    v1213 = false;
                                }
                                float v1214;
                                if (v1213){
                                    v1214 = 1.0f;
                                } else {
                                    v1214 = 0.0f;
                                }
                                int v1215;
                                v1215 = v1168 / 2;
                                bool v1216;
                                v1216 = v153 <= v1215;
                                bool v1224;
                                if (v1216){
                                    bool v1218;
                                    if (v161){
                                        bool v1217;
                                        v1217 = v159 < 2;
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
                                    int v1221;
                                    v1221 = v1178[v159];
                                    bool v1223;
                                    v1223 = v1215 < v1221;
                                    v1224 = v1223;
                                } else {
                                    v1224 = false;
                                }
                                float v1225;
                                if (v1224){
                                    v1225 = 1.0f;
                                } else {
                                    v1225 = 0.0f;
                                }
                                bool v1226;
                                v1226 = v153 <= v1168;
                                bool v1234;
                                if (v1226){
                                    bool v1228;
                                    if (v161){
                                        bool v1227;
                                        v1227 = v159 < 2;
                                        v1228 = v1227;
                                    } else {
                                        v1228 = false;
                                    }
                                    bool v1229;
                                    v1229 = v1228 == false;
                                    if (v1229){
                                        assert("Index must be in range." && v1228);
                                    } else {
                                    }
                                    int v1231;
                                    v1231 = v1178[v159];
                                    bool v1233;
                                    v1233 = v1168 < v1231;
                                    v1234 = v1233;
                                } else {
                                    v1234 = false;
                                }
                                float v1235;
                                if (v1234){
                                    v1235 = 1.0f;
                                } else {
                                    v1235 = 0.0f;
                                }
                                static_array<Tuple12,6> v1236;
                                Union2 v1238;
                                v1238 = Union2{Union2_2{}};
                                v1236[0] = Tuple12{v1238, v1203};
                                Union2 v1240;
                                v1240 = Union2{Union2_1{}};
                                v1236[1] = Tuple12{v1240, 4.0f};
                                Union2 v1242;
                                v1242 = Union2{Union2_3{v1204}};
                                v1236[2] = Tuple12{v1242, v1214};
                                Union2 v1244;
                                v1244 = Union2{Union2_3{v1215}};
                                v1236[3] = Tuple12{v1244, v1225};
                                Union2 v1246;
                                v1246 = Union2{Union2_3{v1168}};
                                v1236[4] = Tuple12{v1246, v1235};
                                Union2 v1248;
                                v1248 = Union2{Union2_0{}};
                                v1236[5] = Tuple12{v1248, 1.0f};
                                Union2 v1250;
                                v1250 = sample_discrete_16(v1236, v1111);
                                int v1251;
                                v1251 = sizeof(Union2);
                                unsigned long long v1252;
                                v1252 = (unsigned long long)v1251;
                                bool v1253;
                                v1253 = v1252 <= 98304ull;
                                bool v1254;
                                v1254 = v1253 == false;
                                if (v1254){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v1253);
                                } else {
                                }
                                extern __shared__ unsigned char v1256[];
                                bool v1257;
                                v1257 = v1252 <= v1252;
                                bool v1258;
                                v1258 = v1257 == false;
                                if (v1258){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v1257);
                                } else {
                                }
                                Union2 * v1260;
                                v1260 = reinterpret_cast<Union2 *>(&v1256[0ull]);
                                int v1262;
                                v1262 = threadIdx.x;
                                bool v1263;
                                v1263 = v1262 == 0;
                                if (v1263){
                                    v1260[0] = v1250;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                Union2 v1264;
                                v1264 = v1260[0];
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                v1267 = v1264;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union1 v1268;
                        v1268 = Union1{Union1_2{v159, v1267}};
                        v17.push(v1268);
                        v1419 = Union6{Union6_1{v153, v154, v155, v156, v157, v158, v1267}};
                        break;
                    }
                    case 5: { // G_Round'
                        int v1270 = v22.case5.v0; static_array<static_array<unsigned char,2>,2> v1271 = v22.case5.v1; static_array<int,2> v1272 = v22.case5.v2; int v1273 = v22.case5.v3; static_array<int,2> v1274 = v22.case5.v4; Union4 v1275 = v22.case5.v5; Union2 v1276 = v22.case5.v6;
                        int v1277;
                        v1277 = v1273 % 2;
                        Union1 v1278;
                        v1278 = Union1{Union1_2{v1277, v1276}};
                        v17.push(v1278);
                        v1419 = Union6{Union6_1{v1270, v1271, v1272, v1273, v1274, v1275, v1276}};
                        break;
                    }
                    case 6: { // G_Showdown
                        int v41 = v22.case6.v0; static_array<static_array<unsigned char,2>,2> v42 = v22.case6.v1; static_array<int,2> v43 = v22.case6.v2; int v44 = v22.case6.v3; static_array<int,2> v45 = v22.case6.v4; Union4 v46 = v22.case6.v5;
                        static_array<unsigned char,5> v49;
                        switch (v46.tag) {
                            case 2: { // River
                                static_array<unsigned char,5> v47 = v46.case2.v0;
                                v49 = v47;
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in showdown.");
                                __trap();
                            }
                        }
                        static_array<unsigned char,2> v50;
                        v50 = v42[0];
                        static_array<unsigned char,7> v52;
                        int v54;
                        v54 = 0;
                        while (while_method_2(v54)){
                            bool v56;
                            v56 = 0 <= v54;
                            bool v58;
                            if (v56){
                                bool v57;
                                v57 = v54 < 2;
                                v58 = v57;
                            } else {
                                v58 = false;
                            }
                            bool v59;
                            v59 = v58 == false;
                            if (v59){
                                assert("Index must be in range." && v58);
                            } else {
                            }
                            unsigned char v61;
                            v61 = v50[v54];
                            v52[v54] = v61;
                            v54 += 1 ;
                        }
                        int v63;
                        v63 = 0;
                        while (while_method_5(v63)){
                            bool v65;
                            v65 = 0 <= v63;
                            bool v67;
                            if (v65){
                                bool v66;
                                v66 = v63 < 5;
                                v67 = v66;
                            } else {
                                v67 = false;
                            }
                            bool v68;
                            v68 = v67 == false;
                            if (v68){
                                assert("Index must be in range." && v67);
                            } else {
                            }
                            unsigned char v70;
                            v70 = v49[v63];
                            int v72;
                            v72 = 2 + v63;
                            v52[v72] = v70;
                            v63 += 1 ;
                        }
                        static_array<unsigned char,5> v73; char v74;
                        Tuple0 tmp58 = score_20(v52);
                        v73 = tmp58.v0; v74 = tmp58.v1;
                        static_array<unsigned char,2> v75;
                        v75 = v42[1];
                        static_array<unsigned char,7> v77;
                        int v79;
                        v79 = 0;
                        while (while_method_2(v79)){
                            bool v81;
                            v81 = 0 <= v79;
                            bool v83;
                            if (v81){
                                bool v82;
                                v82 = v79 < 2;
                                v83 = v82;
                            } else {
                                v83 = false;
                            }
                            bool v84;
                            v84 = v83 == false;
                            if (v84){
                                assert("Index must be in range." && v83);
                            } else {
                            }
                            unsigned char v86;
                            v86 = v75[v79];
                            v77[v79] = v86;
                            v79 += 1 ;
                        }
                        int v88;
                        v88 = 0;
                        while (while_method_5(v88)){
                            bool v90;
                            v90 = 0 <= v88;
                            bool v92;
                            if (v90){
                                bool v91;
                                v91 = v88 < 5;
                                v92 = v91;
                            } else {
                                v92 = false;
                            }
                            bool v93;
                            v93 = v92 == false;
                            if (v93){
                                assert("Index must be in range." && v92);
                            } else {
                            }
                            unsigned char v95;
                            v95 = v49[v88];
                            int v97;
                            v97 = 2 + v88;
                            v77[v97] = v95;
                            v88 += 1 ;
                        }
                        static_array<unsigned char,5> v98; char v99;
                        Tuple0 tmp59 = score_20(v77);
                        v98 = tmp59.v0; v99 = tmp59.v1;
                        int v100;
                        v100 = v44 % 2;
                        bool v101;
                        v101 = 0 <= v100;
                        bool v103;
                        if (v101){
                            bool v102;
                            v102 = v100 < 2;
                            v103 = v102;
                        } else {
                            v103 = false;
                        }
                        bool v104;
                        v104 = v103 == false;
                        if (v104){
                            assert("Index must be in range." && v103);
                        } else {
                        }
                        int v106;
                        v106 = v43[v100];
                        bool v108;
                        v108 = v74 < v99;
                        Union10 v114;
                        if (v108){
                            v114 = Union10{Union10_2{}};
                        } else {
                            bool v110;
                            v110 = v74 > v99;
                            if (v110){
                                v114 = Union10{Union10_1{}};
                            } else {
                                v114 = Union10{Union10_0{}};
                            }
                        }
                        Union10 v142;
                        switch (v114.tag) {
                            case 0: { // Eq
                                Union10 v115;
                                v115 = Union10{Union10_0{}};
                                int v116;
                                v116 = 0;
                                while (while_method_5(v116)){
                                    bool v118;
                                    v118 = 0 <= v116;
                                    bool v120;
                                    if (v118){
                                        bool v119;
                                        v119 = v116 < 5;
                                        v120 = v119;
                                    } else {
                                        v120 = false;
                                    }
                                    bool v121;
                                    v121 = v120 == false;
                                    if (v121){
                                        assert("Index must be in range." && v120);
                                    } else {
                                    }
                                    unsigned char v123;
                                    v123 = v73[v116];
                                    bool v126;
                                    if (v118){
                                        bool v125;
                                        v125 = v116 < 5;
                                        v126 = v125;
                                    } else {
                                        v126 = false;
                                    }
                                    bool v127;
                                    v127 = v126 == false;
                                    if (v127){
                                        assert("Index must be in range." && v126);
                                    } else {
                                    }
                                    unsigned char v129;
                                    v129 = v98[v116];
                                    unsigned char v131;
                                    v131 = v123 / 4u;
                                    unsigned char v132;
                                    v132 = v129 / 4u;
                                    bool v133;
                                    v133 = v131 < v132;
                                    Union10 v139;
                                    if (v133){
                                        v139 = Union10{Union10_2{}};
                                    } else {
                                        bool v135;
                                        v135 = v131 > v132;
                                        if (v135){
                                            v139 = Union10{Union10_1{}};
                                        } else {
                                            v139 = Union10{Union10_0{}};
                                        }
                                    }
                                    bool v140;
                                    switch (v139.tag) {
                                        case 0: { // Eq
                                            v140 = true;
                                            break;
                                        }
                                        default: {
                                            v140 = false;
                                        }
                                    }
                                    bool v141;
                                    v141 = v140 == false;
                                    if (v141){
                                        v115 = v139;
                                        break;
                                    } else {
                                    }
                                    v116 += 1 ;
                                }
                                v142 = v115;
                                break;
                            }
                            default: {
                                v142 = v114;
                            }
                        }
                        int v147; int v148;
                        switch (v142.tag) {
                            case 0: { // Eq
                                v147 = 0; v148 = -1;
                                break;
                            }
                            case 1: { // Gt
                                v147 = v106; v148 = 0;
                                break;
                            }
                            case 2: { // Lt
                                v147 = v106; v148 = 1;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        static_array<Tuple0,2> v149;
                        v149[0] = Tuple0{v73, v74};
                        v149[1] = Tuple0{v98, v99};
                        Union1 v151;
                        v151 = Union1{Union1_4{v147, v149, v148}};
                        v17.push(v151);
                        v1419 = Union6{Union6_0{}};
                        break;
                    }
                    case 7: { // G_Turn
                        int v1299 = v22.case7.v0; static_array<static_array<unsigned char,2>,2> v1300 = v22.case7.v1; static_array<int,2> v1301 = v22.case7.v2; int v1302 = v22.case7.v3; static_array<int,2> v1303 = v22.case7.v4; Union4 v1304 = v22.case7.v5;
                        curandStatePhilox4_32_10_t & v1305 = v2.v5;
                        curandStatePhilox4_32_10_t & v1306 = v1305;
                        static_array<unsigned char,1> v1307; unsigned long long v1308;
                        Tuple6 tmp60 = draw_cards_9(v1306, v18);
                        v1307 = tmp60.v0; v1308 = tmp60.v1;
                        v2.v0 = v1308;
                        static_array_list<unsigned char,5> v1309;
                        v1309 = get_community_cards_10(v1304, v1307);
                        Union1 v1310;
                        v1310 = Union1{Union1_0{v1309}};
                        v17.push(v1310);
                        Union4 v1335;
                        switch (v1304.tag) {
                            case 0: { // Flop
                                static_array<unsigned char,3> v1311 = v1304.case0.v0;
                                static_array<unsigned char,4> v1312;
                                int v1314;
                                v1314 = 0;
                                while (while_method_4(v1314)){
                                    bool v1316;
                                    v1316 = 0 <= v1314;
                                    bool v1318;
                                    if (v1316){
                                        bool v1317;
                                        v1317 = v1314 < 3;
                                        v1318 = v1317;
                                    } else {
                                        v1318 = false;
                                    }
                                    bool v1319;
                                    v1319 = v1318 == false;
                                    if (v1319){
                                        assert("Index must be in range." && v1318);
                                    } else {
                                    }
                                    unsigned char v1321;
                                    v1321 = v1311[v1314];
                                    v1312[v1314] = v1321;
                                    v1314 += 1 ;
                                }
                                int v1323;
                                v1323 = 0;
                                while (while_method_6(v1323)){
                                    bool v1325;
                                    v1325 = 0 <= v1323;
                                    bool v1327;
                                    if (v1325){
                                        bool v1326;
                                        v1326 = v1323 < 1;
                                        v1327 = v1326;
                                    } else {
                                        v1327 = false;
                                    }
                                    bool v1328;
                                    v1328 = v1327 == false;
                                    if (v1328){
                                        assert("Index must be in range." && v1327);
                                    } else {
                                    }
                                    unsigned char v1330;
                                    v1330 = v1307[v1323];
                                    int v1332;
                                    v1332 = 3 + v1323;
                                    v1312[v1332] = v1330;
                                    v1323 += 1 ;
                                }
                                v1335 = Union4{Union4_3{v1312}};
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in turn.");
                                __trap();
                            }
                        }
                        int v1336;
                        v1336 = 2;
                        int v1337;
                        v1337 = 0;
                        Union3 v1338;
                        v1338 = try_round_5(v1336, v1300, v1301, v1337, v1303, v1335);
                        v1419 = Union6{Union6_2{v1338}};
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                switch (v1419.tag) {
                    case 0: { // T_none
                        v1768 = Union5{Union5_0{}};
                        break;
                    }
                    case 1: { // T_round
                        int v1423 = v1419.case1.v0; static_array<static_array<unsigned char,2>,2> v1424 = v1419.case1.v1; static_array<int,2> v1425 = v1419.case1.v2; int v1426 = v1419.case1.v3; static_array<int,2> v1427 = v1419.case1.v4; Union4 v1428 = v1419.case1.v5; Union2 v1429 = v1419.case1.v6;
                        int v1430;
                        v1430 = v1426 % 2;
                        Union3 v1761;
                        switch (v1429.tag) {
                            case 0: { // A_All_In
                                static_array<int,2> v1636;
                                int v1638;
                                v1638 = 0;
                                while (while_method_2(v1638)){
                                    bool v1640;
                                    v1640 = 0 <= v1638;
                                    bool v1642;
                                    if (v1640){
                                        bool v1641;
                                        v1641 = v1638 < 2;
                                        v1642 = v1641;
                                    } else {
                                        v1642 = false;
                                    }
                                    bool v1643;
                                    v1643 = v1642 == false;
                                    if (v1643){
                                        assert("Index must be in range." && v1642);
                                    } else {
                                    }
                                    int v1645;
                                    v1645 = v1427[v1638];
                                    bool v1648;
                                    if (v1640){
                                        bool v1647;
                                        v1647 = v1638 < 2;
                                        v1648 = v1647;
                                    } else {
                                        v1648 = false;
                                    }
                                    bool v1649;
                                    v1649 = v1648 == false;
                                    if (v1649){
                                        assert("Index must be in range." && v1648);
                                    } else {
                                    }
                                    int v1651;
                                    v1651 = v1425[v1638];
                                    int v1653;
                                    v1653 = v1645 + v1651;
                                    v1636[v1638] = v1653;
                                    v1638 += 1 ;
                                }
                                int v1654;
                                v1654 = v1425[0];
                                int v1656; int v1657;
                                Tuple4 tmp61 = Tuple4{1, v1654};
                                v1656 = tmp61.v0; v1657 = tmp61.v1;
                                while (while_method_2(v1656)){
                                    bool v1659;
                                    v1659 = 0 <= v1656;
                                    bool v1661;
                                    if (v1659){
                                        bool v1660;
                                        v1660 = v1656 < 2;
                                        v1661 = v1660;
                                    } else {
                                        v1661 = false;
                                    }
                                    bool v1662;
                                    v1662 = v1661 == false;
                                    if (v1662){
                                        assert("Index must be in range." && v1661);
                                    } else {
                                    }
                                    int v1664;
                                    v1664 = v1425[v1656];
                                    bool v1666;
                                    v1666 = v1657 >= v1664;
                                    int v1667;
                                    if (v1666){
                                        v1667 = v1657;
                                    } else {
                                        v1667 = v1664;
                                    }
                                    v1657 = v1667;
                                    v1656 += 1 ;
                                }
                                bool v1668;
                                v1668 = 0 <= v1430;
                                bool v1670;
                                if (v1668){
                                    bool v1669;
                                    v1669 = v1430 < 2;
                                    v1670 = v1669;
                                } else {
                                    v1670 = false;
                                }
                                bool v1671;
                                v1671 = v1670 == false;
                                if (v1671){
                                    assert("Index must be in range." && v1670);
                                } else {
                                }
                                int v1673;
                                v1673 = v1636[v1430];
                                bool v1675;
                                v1675 = v1657 < v1673;
                                int v1676;
                                if (v1675){
                                    v1676 = v1657;
                                } else {
                                    v1676 = v1673;
                                }
                                static_array<int,2> v1677;
                                int v1679;
                                v1679 = 0;
                                while (while_method_2(v1679)){
                                    bool v1681;
                                    v1681 = 0 <= v1679;
                                    bool v1683;
                                    if (v1681){
                                        bool v1682;
                                        v1682 = v1679 < 2;
                                        v1683 = v1682;
                                    } else {
                                        v1683 = false;
                                    }
                                    bool v1684;
                                    v1684 = v1683 == false;
                                    if (v1684){
                                        assert("Index must be in range." && v1683);
                                    } else {
                                    }
                                    int v1686;
                                    v1686 = v1425[v1679];
                                    bool v1688;
                                    v1688 = v1430 == v1679;
                                    int v1689;
                                    if (v1688){
                                        v1689 = v1676;
                                    } else {
                                        v1689 = v1686;
                                    }
                                    v1677[v1679] = v1689;
                                    v1679 += 1 ;
                                }
                                static_array<int,2> v1690;
                                int v1692;
                                v1692 = 0;
                                while (while_method_2(v1692)){
                                    bool v1694;
                                    v1694 = 0 <= v1692;
                                    bool v1696;
                                    if (v1694){
                                        bool v1695;
                                        v1695 = v1692 < 2;
                                        v1696 = v1695;
                                    } else {
                                        v1696 = false;
                                    }
                                    bool v1697;
                                    v1697 = v1696 == false;
                                    if (v1697){
                                        assert("Index must be in range." && v1696);
                                    } else {
                                    }
                                    int v1699;
                                    v1699 = v1636[v1692];
                                    bool v1702;
                                    if (v1694){
                                        bool v1701;
                                        v1701 = v1692 < 2;
                                        v1702 = v1701;
                                    } else {
                                        v1702 = false;
                                    }
                                    bool v1703;
                                    v1703 = v1702 == false;
                                    if (v1703){
                                        assert("Index must be in range." && v1702);
                                    } else {
                                    }
                                    int v1705;
                                    v1705 = v1677[v1692];
                                    int v1707;
                                    v1707 = v1699 - v1705;
                                    v1690[v1692] = v1707;
                                    v1692 += 1 ;
                                }
                                bool v1709;
                                if (v1668){
                                    bool v1708;
                                    v1708 = v1430 < 2;
                                    v1709 = v1708;
                                } else {
                                    v1709 = false;
                                }
                                bool v1710;
                                v1710 = v1709 == false;
                                if (v1710){
                                    assert("Index must be in range." && v1709);
                                } else {
                                }
                                int v1712;
                                v1712 = v1690[v1430];
                                int v1714;
                                v1714 = v1657 + v1712;
                                bool v1716;
                                if (v1668){
                                    bool v1715;
                                    v1715 = v1430 < 2;
                                    v1716 = v1715;
                                } else {
                                    v1716 = false;
                                }
                                bool v1717;
                                v1717 = v1716 == false;
                                if (v1717){
                                    assert("Index must be in range." && v1716);
                                } else {
                                }
                                int v1719;
                                v1719 = v1636[v1430];
                                bool v1721;
                                v1721 = v1714 < v1719;
                                int v1722;
                                if (v1721){
                                    v1722 = v1714;
                                } else {
                                    v1722 = v1719;
                                }
                                static_array<int,2> v1723;
                                int v1725;
                                v1725 = 0;
                                while (while_method_2(v1725)){
                                    bool v1727;
                                    v1727 = 0 <= v1725;
                                    bool v1729;
                                    if (v1727){
                                        bool v1728;
                                        v1728 = v1725 < 2;
                                        v1729 = v1728;
                                    } else {
                                        v1729 = false;
                                    }
                                    bool v1730;
                                    v1730 = v1729 == false;
                                    if (v1730){
                                        assert("Index must be in range." && v1729);
                                    } else {
                                    }
                                    int v1732;
                                    v1732 = v1425[v1725];
                                    bool v1734;
                                    v1734 = v1430 == v1725;
                                    int v1735;
                                    if (v1734){
                                        v1735 = v1722;
                                    } else {
                                        v1735 = v1732;
                                    }
                                    v1723[v1725] = v1735;
                                    v1725 += 1 ;
                                }
                                static_array<int,2> v1736;
                                int v1738;
                                v1738 = 0;
                                while (while_method_2(v1738)){
                                    bool v1740;
                                    v1740 = 0 <= v1738;
                                    bool v1742;
                                    if (v1740){
                                        bool v1741;
                                        v1741 = v1738 < 2;
                                        v1742 = v1741;
                                    } else {
                                        v1742 = false;
                                    }
                                    bool v1743;
                                    v1743 = v1742 == false;
                                    if (v1743){
                                        assert("Index must be in range." && v1742);
                                    } else {
                                    }
                                    int v1745;
                                    v1745 = v1636[v1738];
                                    bool v1748;
                                    if (v1740){
                                        bool v1747;
                                        v1747 = v1738 < 2;
                                        v1748 = v1747;
                                    } else {
                                        v1748 = false;
                                    }
                                    bool v1749;
                                    v1749 = v1748 == false;
                                    if (v1749){
                                        assert("Index must be in range." && v1748);
                                    } else {
                                    }
                                    int v1751;
                                    v1751 = v1723[v1738];
                                    int v1753;
                                    v1753 = v1745 - v1751;
                                    v1736[v1738] = v1753;
                                    v1738 += 1 ;
                                }
                                bool v1754;
                                v1754 = v1712 >= v1423;
                                int v1755;
                                if (v1754){
                                    v1755 = v1712;
                                } else {
                                    v1755 = v1423;
                                }
                                int v1756;
                                v1756 = v1426 + 1;
                                v1761 = try_round_5(v1755, v1424, v1723, v1756, v1736, v1428);
                                break;
                            }
                            case 1: { // A_Call
                                static_array<int,2> v1432;
                                int v1434;
                                v1434 = 0;
                                while (while_method_2(v1434)){
                                    bool v1436;
                                    v1436 = 0 <= v1434;
                                    bool v1438;
                                    if (v1436){
                                        bool v1437;
                                        v1437 = v1434 < 2;
                                        v1438 = v1437;
                                    } else {
                                        v1438 = false;
                                    }
                                    bool v1439;
                                    v1439 = v1438 == false;
                                    if (v1439){
                                        assert("Index must be in range." && v1438);
                                    } else {
                                    }
                                    int v1441;
                                    v1441 = v1427[v1434];
                                    bool v1444;
                                    if (v1436){
                                        bool v1443;
                                        v1443 = v1434 < 2;
                                        v1444 = v1443;
                                    } else {
                                        v1444 = false;
                                    }
                                    bool v1445;
                                    v1445 = v1444 == false;
                                    if (v1445){
                                        assert("Index must be in range." && v1444);
                                    } else {
                                    }
                                    int v1447;
                                    v1447 = v1425[v1434];
                                    int v1449;
                                    v1449 = v1441 + v1447;
                                    v1432[v1434] = v1449;
                                    v1434 += 1 ;
                                }
                                int v1450;
                                v1450 = v1425[0];
                                int v1452; int v1453;
                                Tuple4 tmp62 = Tuple4{1, v1450};
                                v1452 = tmp62.v0; v1453 = tmp62.v1;
                                while (while_method_2(v1452)){
                                    bool v1455;
                                    v1455 = 0 <= v1452;
                                    bool v1457;
                                    if (v1455){
                                        bool v1456;
                                        v1456 = v1452 < 2;
                                        v1457 = v1456;
                                    } else {
                                        v1457 = false;
                                    }
                                    bool v1458;
                                    v1458 = v1457 == false;
                                    if (v1458){
                                        assert("Index must be in range." && v1457);
                                    } else {
                                    }
                                    int v1460;
                                    v1460 = v1425[v1452];
                                    bool v1462;
                                    v1462 = v1453 >= v1460;
                                    int v1463;
                                    if (v1462){
                                        v1463 = v1453;
                                    } else {
                                        v1463 = v1460;
                                    }
                                    v1453 = v1463;
                                    v1452 += 1 ;
                                }
                                bool v1464;
                                v1464 = 0 <= v1430;
                                bool v1466;
                                if (v1464){
                                    bool v1465;
                                    v1465 = v1430 < 2;
                                    v1466 = v1465;
                                } else {
                                    v1466 = false;
                                }
                                bool v1467;
                                v1467 = v1466 == false;
                                if (v1467){
                                    assert("Index must be in range." && v1466);
                                } else {
                                }
                                int v1469;
                                v1469 = v1432[v1430];
                                bool v1471;
                                v1471 = v1453 < v1469;
                                int v1472;
                                if (v1471){
                                    v1472 = v1453;
                                } else {
                                    v1472 = v1469;
                                }
                                static_array<int,2> v1473;
                                int v1475;
                                v1475 = 0;
                                while (while_method_2(v1475)){
                                    bool v1477;
                                    v1477 = 0 <= v1475;
                                    bool v1479;
                                    if (v1477){
                                        bool v1478;
                                        v1478 = v1475 < 2;
                                        v1479 = v1478;
                                    } else {
                                        v1479 = false;
                                    }
                                    bool v1480;
                                    v1480 = v1479 == false;
                                    if (v1480){
                                        assert("Index must be in range." && v1479);
                                    } else {
                                    }
                                    int v1482;
                                    v1482 = v1425[v1475];
                                    bool v1484;
                                    v1484 = v1430 == v1475;
                                    int v1485;
                                    if (v1484){
                                        v1485 = v1472;
                                    } else {
                                        v1485 = v1482;
                                    }
                                    v1473[v1475] = v1485;
                                    v1475 += 1 ;
                                }
                                static_array<int,2> v1486;
                                int v1488;
                                v1488 = 0;
                                while (while_method_2(v1488)){
                                    bool v1490;
                                    v1490 = 0 <= v1488;
                                    bool v1492;
                                    if (v1490){
                                        bool v1491;
                                        v1491 = v1488 < 2;
                                        v1492 = v1491;
                                    } else {
                                        v1492 = false;
                                    }
                                    bool v1493;
                                    v1493 = v1492 == false;
                                    if (v1493){
                                        assert("Index must be in range." && v1492);
                                    } else {
                                    }
                                    int v1495;
                                    v1495 = v1432[v1488];
                                    bool v1498;
                                    if (v1490){
                                        bool v1497;
                                        v1497 = v1488 < 2;
                                        v1498 = v1497;
                                    } else {
                                        v1498 = false;
                                    }
                                    bool v1499;
                                    v1499 = v1498 == false;
                                    if (v1499){
                                        assert("Index must be in range." && v1498);
                                    } else {
                                    }
                                    int v1501;
                                    v1501 = v1473[v1488];
                                    int v1503;
                                    v1503 = v1495 - v1501;
                                    v1486[v1488] = v1503;
                                    v1488 += 1 ;
                                }
                                bool v1504;
                                v1504 = v1430 < 2;
                                if (v1504){
                                    int v1505;
                                    v1505 = v1426 + 1;
                                    v1761 = try_round_5(v1423, v1424, v1473, v1505, v1486, v1428);
                                } else {
                                    v1761 = go_next_street_7(v1423, v1424, v1473, v1426, v1486, v1428);
                                }
                                break;
                            }
                            case 2: { // A_Fold
                                v1761 = Union3{Union3_1{v1423, v1424, v1425, v1426, v1427, v1428}};
                                break;
                            }
                            case 3: { // A_Raise
                                int v1509 = v1429.case3.v0;
                                bool v1510;
                                v1510 = v1423 <= v1509;
                                bool v1511;
                                v1511 = v1510 == false;
                                if (v1511){
                                    assert("The raise amount must match the minimum." && v1510);
                                } else {
                                }
                                static_array<int,2> v1513;
                                int v1515;
                                v1515 = 0;
                                while (while_method_2(v1515)){
                                    bool v1517;
                                    v1517 = 0 <= v1515;
                                    bool v1519;
                                    if (v1517){
                                        bool v1518;
                                        v1518 = v1515 < 2;
                                        v1519 = v1518;
                                    } else {
                                        v1519 = false;
                                    }
                                    bool v1520;
                                    v1520 = v1519 == false;
                                    if (v1520){
                                        assert("Index must be in range." && v1519);
                                    } else {
                                    }
                                    int v1522;
                                    v1522 = v1427[v1515];
                                    bool v1525;
                                    if (v1517){
                                        bool v1524;
                                        v1524 = v1515 < 2;
                                        v1525 = v1524;
                                    } else {
                                        v1525 = false;
                                    }
                                    bool v1526;
                                    v1526 = v1525 == false;
                                    if (v1526){
                                        assert("Index must be in range." && v1525);
                                    } else {
                                    }
                                    int v1528;
                                    v1528 = v1425[v1515];
                                    int v1530;
                                    v1530 = v1522 + v1528;
                                    v1513[v1515] = v1530;
                                    v1515 += 1 ;
                                }
                                int v1531;
                                v1531 = v1425[0];
                                int v1533; int v1534;
                                Tuple4 tmp63 = Tuple4{1, v1531};
                                v1533 = tmp63.v0; v1534 = tmp63.v1;
                                while (while_method_2(v1533)){
                                    bool v1536;
                                    v1536 = 0 <= v1533;
                                    bool v1538;
                                    if (v1536){
                                        bool v1537;
                                        v1537 = v1533 < 2;
                                        v1538 = v1537;
                                    } else {
                                        v1538 = false;
                                    }
                                    bool v1539;
                                    v1539 = v1538 == false;
                                    if (v1539){
                                        assert("Index must be in range." && v1538);
                                    } else {
                                    }
                                    int v1541;
                                    v1541 = v1425[v1533];
                                    bool v1543;
                                    v1543 = v1534 >= v1541;
                                    int v1544;
                                    if (v1543){
                                        v1544 = v1534;
                                    } else {
                                        v1544 = v1541;
                                    }
                                    v1534 = v1544;
                                    v1533 += 1 ;
                                }
                                bool v1545;
                                v1545 = 0 <= v1430;
                                bool v1547;
                                if (v1545){
                                    bool v1546;
                                    v1546 = v1430 < 2;
                                    v1547 = v1546;
                                } else {
                                    v1547 = false;
                                }
                                bool v1548;
                                v1548 = v1547 == false;
                                if (v1548){
                                    assert("Index must be in range." && v1547);
                                } else {
                                }
                                int v1550;
                                v1550 = v1513[v1430];
                                bool v1552;
                                v1552 = v1534 < v1550;
                                int v1553;
                                if (v1552){
                                    v1553 = v1534;
                                } else {
                                    v1553 = v1550;
                                }
                                static_array<int,2> v1554;
                                int v1556;
                                v1556 = 0;
                                while (while_method_2(v1556)){
                                    bool v1558;
                                    v1558 = 0 <= v1556;
                                    bool v1560;
                                    if (v1558){
                                        bool v1559;
                                        v1559 = v1556 < 2;
                                        v1560 = v1559;
                                    } else {
                                        v1560 = false;
                                    }
                                    bool v1561;
                                    v1561 = v1560 == false;
                                    if (v1561){
                                        assert("Index must be in range." && v1560);
                                    } else {
                                    }
                                    int v1563;
                                    v1563 = v1425[v1556];
                                    bool v1565;
                                    v1565 = v1430 == v1556;
                                    int v1566;
                                    if (v1565){
                                        v1566 = v1553;
                                    } else {
                                        v1566 = v1563;
                                    }
                                    v1554[v1556] = v1566;
                                    v1556 += 1 ;
                                }
                                static_array<int,2> v1567;
                                int v1569;
                                v1569 = 0;
                                while (while_method_2(v1569)){
                                    bool v1571;
                                    v1571 = 0 <= v1569;
                                    bool v1573;
                                    if (v1571){
                                        bool v1572;
                                        v1572 = v1569 < 2;
                                        v1573 = v1572;
                                    } else {
                                        v1573 = false;
                                    }
                                    bool v1574;
                                    v1574 = v1573 == false;
                                    if (v1574){
                                        assert("Index must be in range." && v1573);
                                    } else {
                                    }
                                    int v1576;
                                    v1576 = v1513[v1569];
                                    bool v1579;
                                    if (v1571){
                                        bool v1578;
                                        v1578 = v1569 < 2;
                                        v1579 = v1578;
                                    } else {
                                        v1579 = false;
                                    }
                                    bool v1580;
                                    v1580 = v1579 == false;
                                    if (v1580){
                                        assert("Index must be in range." && v1579);
                                    } else {
                                    }
                                    int v1582;
                                    v1582 = v1554[v1569];
                                    int v1584;
                                    v1584 = v1576 - v1582;
                                    v1567[v1569] = v1584;
                                    v1569 += 1 ;
                                }
                                bool v1586;
                                if (v1545){
                                    bool v1585;
                                    v1585 = v1430 < 2;
                                    v1586 = v1585;
                                } else {
                                    v1586 = false;
                                }
                                bool v1587;
                                v1587 = v1586 == false;
                                if (v1587){
                                    assert("Index must be in range." && v1586);
                                } else {
                                }
                                int v1589;
                                v1589 = v1567[v1430];
                                bool v1591;
                                v1591 = v1509 < v1589;
                                bool v1592;
                                v1592 = v1591 == false;
                                if (v1592){
                                    assert("The raise amount must be less than the stack size after calling." && v1591);
                                } else {
                                }
                                int v1594;
                                v1594 = v1534 + v1509;
                                bool v1596;
                                if (v1545){
                                    bool v1595;
                                    v1595 = v1430 < 2;
                                    v1596 = v1595;
                                } else {
                                    v1596 = false;
                                }
                                bool v1597;
                                v1597 = v1596 == false;
                                if (v1597){
                                    assert("Index must be in range." && v1596);
                                } else {
                                }
                                int v1599;
                                v1599 = v1513[v1430];
                                bool v1601;
                                v1601 = v1594 < v1599;
                                int v1602;
                                if (v1601){
                                    v1602 = v1594;
                                } else {
                                    v1602 = v1599;
                                }
                                static_array<int,2> v1603;
                                int v1605;
                                v1605 = 0;
                                while (while_method_2(v1605)){
                                    bool v1607;
                                    v1607 = 0 <= v1605;
                                    bool v1609;
                                    if (v1607){
                                        bool v1608;
                                        v1608 = v1605 < 2;
                                        v1609 = v1608;
                                    } else {
                                        v1609 = false;
                                    }
                                    bool v1610;
                                    v1610 = v1609 == false;
                                    if (v1610){
                                        assert("Index must be in range." && v1609);
                                    } else {
                                    }
                                    int v1612;
                                    v1612 = v1425[v1605];
                                    bool v1614;
                                    v1614 = v1430 == v1605;
                                    int v1615;
                                    if (v1614){
                                        v1615 = v1602;
                                    } else {
                                        v1615 = v1612;
                                    }
                                    v1603[v1605] = v1615;
                                    v1605 += 1 ;
                                }
                                static_array<int,2> v1616;
                                int v1618;
                                v1618 = 0;
                                while (while_method_2(v1618)){
                                    bool v1620;
                                    v1620 = 0 <= v1618;
                                    bool v1622;
                                    if (v1620){
                                        bool v1621;
                                        v1621 = v1618 < 2;
                                        v1622 = v1621;
                                    } else {
                                        v1622 = false;
                                    }
                                    bool v1623;
                                    v1623 = v1622 == false;
                                    if (v1623){
                                        assert("Index must be in range." && v1622);
                                    } else {
                                    }
                                    int v1625;
                                    v1625 = v1513[v1618];
                                    bool v1628;
                                    if (v1620){
                                        bool v1627;
                                        v1627 = v1618 < 2;
                                        v1628 = v1627;
                                    } else {
                                        v1628 = false;
                                    }
                                    bool v1629;
                                    v1629 = v1628 == false;
                                    if (v1629){
                                        assert("Index must be in range." && v1628);
                                    } else {
                                    }
                                    int v1631;
                                    v1631 = v1603[v1618];
                                    int v1633;
                                    v1633 = v1625 - v1631;
                                    v1616[v1618] = v1633;
                                    v1618 += 1 ;
                                }
                                int v1634;
                                v1634 = v1426 + 1;
                                v1761 = try_round_5(v1509, v1424, v1603, v1634, v1616, v1428);
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        v1768 = Union5{Union5_1{v1761}};
                        break;
                    }
                    case 2: { // T_some
                        Union3 v1421 = v1419.case2.v0;
                        v1768 = Union5{Union5_1{v1421}};
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
        v20 = v1768;
    }
    return ;
}
__device__ inline bool while_method_21(int v0){
    bool v1;
    v1 = v0 > 0;
    return v1;
}
__device__ int int_range_21(int v0, int v1, curandStatePhilox4_32_10_t & v2){
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
__device__ inline bool while_method_22(int v0){
    bool v1;
    v1 = v0 < 256;
    return v1;
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned long long v1, unsigned char * v2, unsigned long long v3, float * v4, float * v5, float * v6) {
    auto v7 = cooperative_groups::this_grid();
    unsigned long long v8;
    v8 = clock64();
    int v9;
    v9 = threadIdx.x;
    int v10;
    v10 = blockIdx.x;
    int v11;
    v11 = v10 * 256;
    int v12;
    v12 = v9 + v11;
    unsigned long long v13;
    v13 = (unsigned long long)v12;
    curandStatePhilox4_32_10_t v14;
    curand_init(v8,v13,0ull,&v14);
    static_array<Union0,2> v15;
    Union0 v17;
    v17 = Union0{Union0_2{}};
    v15[0] = v17;
    Union0 v19;
    v19 = Union0{Union0_2{}};
    v15[1] = v19;
    static_array_list<Union1,128> v21;
    v21 = static_array_list<Union1,128>{};
    static_array<float,2> v23;
    v23[0] = 0.0f;
    v23[1] = 0.0f;
    cooperative_groups::grid_group & v25 = v7;
    curandStatePhilox4_32_10_t & v26 = v14;
    StackMut0 v27{4503599627370495ull, v25, v21, v15, v23, v26};
    bool v28;
    v28 = 12419088ull == v3;
    bool v29;
    v29 = v28 == false;
    if (v29){
        assert("The params needs to have matching offsets." && v28);
    } else {
    }
    bool v31;
    v31 = 204570624ull == v1;
    bool v32;
    v32 = v31 == false;
    if (v32){
        assert("The outputs needs to have matching offsets." && v31);
    } else {
    }
    int v34;
    v34 = 0;
    while (while_method_0(v34)){
        int v36;
        v36 = 0;
        while (while_method_1(v36)){
            int v38;
            v38 = 0;
            while (while_method_2(v38)){
                Union3 v40;
                v40 = Union3{Union3_2{}};
                method_0(v0, v2, v27, v38, v40);
                static_array<float,2> & v41 = v27.v4;
                bool v42;
                v42 = 0 <= v38;
                bool v44;
                if (v42){
                    bool v43;
                    v43 = v38 < 2;
                    v44 = v43;
                } else {
                    v44 = false;
                }
                bool v45;
                v45 = v44 == false;
                if (v45){
                    assert("Index must be in range." && v44);
                } else {
                }
                float v47;
                v47 = v41[v38];
                double * v49;
                v49 = reinterpret_cast<double *>(&v2[11534352ull]);
                double * v51;
                v51 = reinterpret_cast<double *>(&v2[11927568ull]);
                int * v53;
                v53 = reinterpret_cast<int *>(&v2[12320784ull]);
                int v55;
                v55 = threadIdx.x;
                int v56;
                v56 = blockIdx.x;
                int v57;
                v57 = v56 * 256;
                int v58;
                v58 = v55 + v57;
                assert("Tensor range check" && 0 <= v58 && v58 < 6144);
                int v59;
                v59 = 2 * v58;
                int v60; double v61;
                Tuple18 tmp64 = Tuple18{0, 1.0};
                v60 = tmp64.v0; v61 = tmp64.v1;
                while (while_method_2(v60)){
                    assert("Tensor range check" && 0 <= v60 && v60 < 2);
                    int v63;
                    v63 = v60 + v59;
                    int v64; double v65;
                    Tuple18 tmp65 = Tuple18{0, 0.0};
                    v64 = tmp65.v0; v65 = tmp65.v1;
                    while (while_method_0(v64)){
                        assert("Tensor range check" && 0 <= v64 && v64 < 4);
                        int v67;
                        v67 = 12288 * v64;
                        int v68;
                        v68 = v67 + v63;
                        double v69;
                        v69 = v49[v68];
                        double v70;
                        v70 = v51[v68];
                        double v71;
                        v71 = v69 - v70;
                        double v72;
                        v72 = exp(v71);
                        double v73;
                        v73 = v65 + v72;
                        v65 = v73;
                        v64 += 1 ;
                    }
                    double v74;
                    v74 = v61 * v65;
                    v61 = v74;
                    v60 += 1 ;
                }
                float v75;
                v75 = (float)v61;
                int v76;
                v76 = 0;
                while (while_method_0(v76)){
                    double * v78;
                    v78 = reinterpret_cast<double *>(&v2[11534352ull]);
                    double * v80;
                    v80 = reinterpret_cast<double *>(&v2[11927568ull]);
                    int * v82;
                    v82 = reinterpret_cast<int *>(&v2[12320784ull]);
                    int v84;
                    v84 = threadIdx.x;
                    int v85;
                    v85 = blockIdx.x;
                    int v86;
                    v86 = v85 * 256;
                    int v87;
                    v87 = v84 + v86;
                    assert("Tensor range check" && 0 <= v87 && v87 < 6144);
                    int v88;
                    v88 = 2 * v87;
                    int v89; double v90;
                    Tuple18 tmp66 = Tuple18{0, 1.0};
                    v89 = tmp66.v0; v90 = tmp66.v1;
                    while (while_method_2(v89)){
                        assert("Tensor range check" && 0 <= v89 && v89 < 2);
                        int v92;
                        v92 = v89 + v88;
                        int v93; double v94;
                        Tuple18 tmp67 = Tuple18{0, 0.0};
                        v93 = tmp67.v0; v94 = tmp67.v1;
                        while (while_method_0(v93)){
                            assert("Tensor range check" && 0 <= v93 && v93 < 4);
                            int v96;
                            v96 = 12288 * v93;
                            int v97;
                            v97 = v96 + v92;
                            double v98;
                            v98 = v78[v97];
                            double v99;
                            v99 = v80[v97];
                            double v100;
                            v100 = v98 - v99;
                            double v101;
                            v101 = exp(v100);
                            bool v102;
                            v102 = v76 == v93;
                            bool v103;
                            v103 = v102 != true;
                            double v104;
                            if (v103){
                                v104 = v101;
                            } else {
                                v104 = 0.0;
                            }
                            double v105;
                            v105 = v94 + v104;
                            v94 = v105;
                            v93 += 1 ;
                        }
                        double v106;
                        v106 = v90 * v94;
                        v90 = v106;
                        v89 += 1 ;
                    }
                    float v107;
                    v107 = (float)v90;
                    float v108;
                    v108 = v75 - v107;
                    float v109;
                    v109 = v47 * v108;
                    assert("Tensor range check" && 0 <= v76 && v76 < 4);
                    assert("Tensor range check" && 0 <= v34 && v34 < 4);
                    int v110;
                    v110 = 4 * v76;
                    int v111;
                    v111 = v110 + v34;
                    float * v112;
                    v112 = v4+v111;
                    float * v114;
                    v114 = v5+v111;
                    float v116;
                    v116 = atomicAdd(v112,v109);
                    float v117;
                    v117 = atomicAdd(v114,v108);
                    v76 += 1 ;
                }
                static_array<float,2> & v118 = v27.v4;
                unsigned int * v119;
                v119 = reinterpret_cast<unsigned int *>(&v0[53477376ull]);
                int * v121;
                v121 = reinterpret_cast<int *>(&v2[4194304ull]);
                float * v123;
                v123 = reinterpret_cast<float *>(&v2[4194320ull]);
                float * v125;
                v125 = reinterpret_cast<float *>(&v2[5242896ull]);
                float * v127;
                v127 = reinterpret_cast<float *>(&v2[6291472ull]);
                float * v129;
                v129 = reinterpret_cast<float *>(&v2[7340048ull]);
                float * v131;
                v131 = reinterpret_cast<float *>(&v2[8388624ull]);
                float * v133;
                v133 = reinterpret_cast<float *>(&v2[9437200ull]);
                float * v135;
                v135 = reinterpret_cast<float *>(&v2[10485776ull]);
                int * v137;
                v137 = reinterpret_cast<int *>(&v0[53575680ull]);
                float * v139;
                v139 = reinterpret_cast<float *>(&v0[66158592ull]);
                int * v141;
                v141 = reinterpret_cast<int *>(&v0[78741504ull]);
                int * v143;
                v143 = reinterpret_cast<int *>(&v0[91324416ull]);
                double * v145;
                v145 = reinterpret_cast<double *>(&v0[103907328ull]);
                double * v147;
                v147 = reinterpret_cast<double *>(&v0[154238976ull]);
                double * v149;
                v149 = reinterpret_cast<double *>(&v2[11534352ull]);
                double * v151;
                v151 = reinterpret_cast<double *>(&v2[11927568ull]);
                int * v153;
                v153 = reinterpret_cast<int *>(&v2[12320784ull]);
                int v155;
                v155 = 0;
                while (while_method_0(v155)){
                    int v157;
                    v157 = threadIdx.x;
                    int v158;
                    v158 = blockIdx.x;
                    int v159;
                    v159 = v158 * 256;
                    int v160;
                    v160 = v157 + v159;
                    float v161[2];
                    int v162;
                    v162 = 0;
                    while (while_method_2(v162)){
                        bool v164;
                        v164 = 0 <= v162;
                        bool v166;
                        if (v164){
                            bool v165;
                            v165 = v162 < 2;
                            v166 = v165;
                        } else {
                            v166 = false;
                        }
                        bool v167;
                        v167 = v166 == false;
                        if (v167){
                            assert("Index must be in range." && v166);
                        } else {
                        }
                        float v169;
                        v169 = v118[v162];
                        v161[v162] = v169;
                        v162 += 1 ;
                    }
                    assert("Tensor range check" && 0 <= v155 && v155 < 4);
                    assert("Tensor range check" && 0 <= v160 && v160 < 6144);
                    int v171;
                    v171 = 6144 * v155;
                    int v172;
                    v172 = v171 + v160;
                    int v173;
                    v173 = v153[v172];
                    int v174;
                    v174 = v173;
                    while (while_method_21(v174)){
                        v174 -= 1 ;
                        assert("Tensor range check" && 0 <= v155 && v155 < 4);
                        assert("Tensor range check" && 0 <= v174 && v174 < 128);
                        assert("Tensor range check" && 0 <= v160 && v160 < 6144);
                        int v176;
                        v176 = 6144 * v174;
                        int v177;
                        v177 = v176 + v160;
                        int v178;
                        v178 = 786432 * v155;
                        int v179;
                        v179 = v178 + v177;
                        int v180;
                        v180 = v137[v179];
                        float v181;
                        v181 = v139[v179];
                        int v182;
                        v182 = v141[v179];
                        int v183;
                        v183 = v143[v179];
                        assert("Tensor range check" && 0 <= v182 && v182 < 2);
                        float v184;
                        v184 = v161[v182];
                        assert("Tensor range check" && 0 <= v155 && v155 < 4);
                        int v185;
                        v185 = 65536 * v155;
                        assert("Tensor range check" && 0 <= v183 && v183 < 4096);
                        int v186;
                        v186 = 16 * v183;
                        int v187;
                        v187 = v186 + v185;
                        float * v188;
                        v188 = v123+v187;
                        float * v190;
                        v190 = v125+v187;
                        float * v192;
                        v192 = v127+v187;
                        float * v194;
                        v194 = v129+v187;
                        float * v196;
                        v196 = v131+v187;
                        float * v198;
                        v198 = v133+v187;
                        float * v200;
                        v200 = v135+v187;
                        assert("Tensor range check" && 0 <= v155 && v155 < 4);
                        int v202;
                        v202 = 1572864 * v155;
                        assert("Tensor range check" && 0 <= v174 && v174 < 128);
                        int v203;
                        v203 = 12288 * v174;
                        int v204;
                        v204 = v203 + v202;
                        assert("Tensor range check" && 0 <= v160 && v160 < 6144);
                        int v205;
                        v205 = 2 * v160;
                        int v206;
                        v206 = v205 + v204;
                        double v207[2];
                        int v208;
                        v208 = 0;
                        while (while_method_2(v208)){
                            assert("Tensor range check" && 0 <= v208 && v208 < 2);
                            int v210;
                            v210 = v208 + v206;
                            double v211;
                            v211 = v145[v210];
                            bool v212;
                            v212 = v182 == v208;
                            double v213;
                            if (v212){
                                v213 = 0.0;
                            } else {
                                v213 = v211;
                            }
                            assert("Tensor range check" && 0 <= v208 && v208 < 2);
                            v207[v208] = v213;
                            v208 += 1 ;
                        }
                        double v214;
                        v214 = 0.0;
                        int v215;
                        v215 = 0;
                        while (while_method_2(v215)){
                            assert("Tensor range check" && 0 <= v215 && v215 < 2);
                            double v217;
                            v217 = v207[v215];
                            double v218;
                            v218 = v214 + v217;
                            v214 = v218;
                            v215 += 1 ;
                        }
                        double v219;
                        v219 = 0.0;
                        int v220;
                        v220 = 0;
                        while (while_method_2(v220)){
                            assert("Tensor range check" && 0 <= v220 && v220 < 2);
                            int v222;
                            v222 = v220 + v206;
                            double v223;
                            v223 = v147[v222];
                            double v224;
                            v224 = v219 + v223;
                            v219 = v224;
                            v220 += 1 ;
                        }
                        double v225;
                        v225 = v214 - v219;
                        double v226;
                        v226 = exp(v225);
                        float v227;
                        v227 = (float)v226;
                        float v228;
                        v228 = v184 * v227;
                        assert("Tensor range check" && 0 <= v180 && v180 < 16);
                        float * v229;
                        v229 = v198+v180;
                        float * v231;
                        v231 = v200+v180;
                        float v233;
                        v233 = atomicAdd(v229,v228);
                        float v234;
                        v234 = atomicAdd(v231,v227);
                        float * v235;
                        v235 = v190+0;
                        float * v237;
                        v237 = v194+0;
                        float * v239;
                        v239 = v196+0;
                        int v241;
                        v241 = sizeof(float *);
                        unsigned long long v242;
                        v242 = (unsigned long long)v241;
                        unsigned long long v243;
                        v243 = 256ull * v242;
                        unsigned long long v244;
                        v244 = 4096ull + v243;
                        unsigned long long v245;
                        v245 = v244 + 16ull;
                        unsigned long long v246;
                        v246 = v245 - 1ull;
                        unsigned long long v247;
                        v247 = v246 % 16ull;
                        unsigned long long v248;
                        v248 = v246 - v247;
                        unsigned long long v249;
                        v249 = v248 + v243;
                        unsigned long long v250;
                        v250 = v249 + 16ull;
                        unsigned long long v251;
                        v251 = v250 - 1ull;
                        unsigned long long v252;
                        v252 = v251 % 16ull;
                        unsigned long long v253;
                        v253 = v251 - v252;
                        unsigned long long v254;
                        v254 = v253 + v243;
                        unsigned long long v255;
                        v255 = v254 + 16ull;
                        unsigned long long v256;
                        v256 = v255 - 1ull;
                        unsigned long long v257;
                        v257 = v256 % 16ull;
                        unsigned long long v258;
                        v258 = v256 - v257;
                        unsigned long long v259;
                        v259 = v258 + v243;
                        unsigned long long v260;
                        v260 = v259 + 16ull;
                        unsigned long long v261;
                        v261 = v260 - 1ull;
                        unsigned long long v262;
                        v262 = v261 % 16ull;
                        unsigned long long v263;
                        v263 = v261 - v262;
                        unsigned long long v264;
                        v264 = v263 + 1024ull;
                        bool v265;
                        v265 = v264 <= 98304ull;
                        bool v266;
                        v266 = v265 == false;
                        if (v266){
                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v265);
                        } else {
                        }
                        extern __shared__ unsigned char v268[];
                        bool v269;
                        v269 = v264 <= v264;
                        bool v270;
                        v270 = v269 == false;
                        if (v270){
                            assert("The length of the partition has to be less than or equal to the length of the base array." && v269);
                        } else {
                        }
                        float * v272;
                        v272 = reinterpret_cast<float *>(&v268[0ull]);
                        int * v274;
                        v274 = reinterpret_cast<int *>(&v268[1024ull]);
                        float * v276;
                        v276 = reinterpret_cast<float *>(&v268[2048ull]);
                        float * v278;
                        v278 = reinterpret_cast<float *>(&v268[3072ull]);
                        float * * v280;
                        v280 = reinterpret_cast<float * *>(&v268[4096ull]);
                        float * * v282;
                        v282 = reinterpret_cast<float * *>(&v268[v248]);
                        float * * v284;
                        v284 = reinterpret_cast<float * *>(&v268[v253]);
                        float * * v286;
                        v286 = reinterpret_cast<float * *>(&v268[v258]);
                        float * v288;
                        v288 = reinterpret_cast<float *>(&v268[v263]);
                        int v290;
                        v290 = threadIdx.x;
                        assert("Tensor range check" && 0 <= v290 && v290 < 256);
                        v272[v290] = v181;
                        v274[v290] = v180;
                        v276[v290] = v184;
                        v278[v290] = v227;
                        v280[v290] = v192;
                        v282[v290] = v235;
                        v284[v290] = v237;
                        v286[v290] = v239;
                        asm("barrier.cta.sync %0;" :: "r"(0));
                        bool v291;
                        v291 = 0 <= v290;
                        bool v292;
                        v292 = v291 == false;
                        if (v292){
                            assert("The index needs to be zero or positive." && v291);
                        } else {
                        }
                        int v294;
                        v294 = v290 % 4;
                        int v295;
                        v295 = v290 / 4;
                        bool v296;
                        v296 = v295 < 64;
                        bool v297;
                        v297 = v296 == false;
                        if (v297){
                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v296);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v295 && v295 < 64);
                        int v299;
                        v299 = 0;
                        while (while_method_0(v299)){
                            bool v301;
                            v301 = 0 <= v295;
                            bool v302;
                            v302 = v301 && v296;
                            bool v303;
                            v303 = v302 == false;
                            if (v303){
                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v302);
                            } else {
                            }
                            bool v305;
                            v305 = 0 <= v299;
                            bool v307;
                            if (v305){
                                bool v306;
                                v306 = v299 < 4;
                                v307 = v306;
                            } else {
                                v307 = false;
                            }
                            bool v308;
                            v308 = v307 == false;
                            if (v308){
                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v307);
                            } else {
                            }
                            int v310;
                            v310 = v299 * 64;
                            int v311;
                            v311 = v310 + v295;
                            assert("Tensor range check" && 0 <= v299 && v299 < 4);
                            int v312;
                            v312 = 64 * v299;
                            int v313;
                            v313 = v312 + v295;
                            float v314;
                            v314 = v272[v313];
                            int v315;
                            v315 = v274[v313];
                            float v316;
                            v316 = v276[v313];
                            float v317;
                            v317 = v278[v313];
                            float * v318;
                            v318 = v280[v313];
                            float * v319;
                            v319 = v282[v313];
                            float * v320;
                            v320 = v284[v313];
                            float * v321;
                            v321 = v286[v313];
                            int v322;
                            v322 = blockIdx.x;
                            int v323;
                            v323 = v322 * 256;
                            int v324;
                            v324 = v323 + v311;
                            assert("Tensor range check" && 0 <= v294 && v294 < 4);
                            int v325;
                            v325 = 4 * v294;
                            float v326[4];
                            float v327[4];
                            float v328[4];
                            int v329[4];
                            int v330;
                            v330 = 0;
                            while (while_method_6(v330)){
                                assert("Tensor range check" && 0 <= v330 && v330 < 1);
                                int v332;
                                v332 = 4 * v330;
                                assert("Tensor range check" && 0 <= v330 && v330 < 1);
                                int v333;
                                v333 = 16 * v330;
                                int v334;
                                v334 = v333 + v325;
                                int4* v335;
                                v335 = reinterpret_cast<int4*>(v319 + v334);
                                int4* v336;
                                v336 = reinterpret_cast<int4*>(v326 + v332);
                                assert("Pointer alignment check" && (unsigned long long)(v335) % 4 == 0 && (unsigned long long)(v336) % 4 == 0);
                                *v336 = *v335;
                                int4* v337;
                                v337 = reinterpret_cast<int4*>(v320 + v334);
                                int4* v338;
                                v338 = reinterpret_cast<int4*>(v327 + v332);
                                assert("Pointer alignment check" && (unsigned long long)(v337) % 4 == 0 && (unsigned long long)(v338) % 4 == 0);
                                *v338 = *v337;
                                int4* v339;
                                v339 = reinterpret_cast<int4*>(v321 + v334);
                                int4* v340;
                                v340 = reinterpret_cast<int4*>(v328 + v332);
                                assert("Pointer alignment check" && (unsigned long long)(v339) % 4 == 0 && (unsigned long long)(v340) % 4 == 0);
                                *v340 = *v339;
                                v330 += 1 ;
                            }
                            int v341;
                            v341 = 0;
                            while (while_method_6(v341)){
                                int v343;
                                v343 = 0;
                                while (while_method_0(v343)){
                                    bool v345;
                                    v345 = 0 <= v343;
                                    bool v347;
                                    if (v345){
                                        bool v346;
                                        v346 = v343 < 4;
                                        v347 = v346;
                                    } else {
                                        v347 = false;
                                    }
                                    bool v348;
                                    v348 = v347 == false;
                                    if (v348){
                                        assert("The indices should be inside the range of the dimension." && v347);
                                    } else {
                                    }
                                    bool v350;
                                    v350 = 0 <= v294;
                                    bool v352;
                                    if (v350){
                                        bool v351;
                                        v351 = v294 < 4;
                                        v352 = v351;
                                    } else {
                                        v352 = false;
                                    }
                                    bool v353;
                                    v353 = v352 == false;
                                    if (v353){
                                        assert("The indices should be inside the range of the dimension." && v352);
                                    } else {
                                    }
                                    int v355;
                                    v355 = v294 * 4;
                                    int v356;
                                    v356 = v343 + v355;
                                    bool v357;
                                    v357 = 0 <= v341;
                                    bool v359;
                                    if (v357){
                                        bool v358;
                                        v358 = v341 < 1;
                                        v359 = v358;
                                    } else {
                                        v359 = false;
                                    }
                                    bool v360;
                                    v360 = v359 == false;
                                    if (v360){
                                        assert("The indices should be inside the range of the dimension." && v359);
                                    } else {
                                    }
                                    int v362;
                                    v362 = v341 * 16;
                                    int v363;
                                    v363 = v356 + v362;
                                    assert("Tensor range check" && 0 <= v341 && v341 < 1);
                                    assert("Tensor range check" && 0 <= v343 && v343 < 4);
                                    int v364;
                                    v364 = 4 * v341;
                                    int v365;
                                    v365 = v364 + v343;
                                    v329[v365] = v363;
                                    v343 += 1 ;
                                }
                                v341 += 1 ;
                            }
                            float v366[4];
                            int v367;
                            v367 = 0;
                            while (while_method_6(v367)){
                                int v369;
                                v369 = 0;
                                while (while_method_0(v369)){
                                    assert("Tensor range check" && 0 <= v367 && v367 < 1);
                                    assert("Tensor range check" && 0 <= v369 && v369 < 4);
                                    int v371;
                                    v371 = 4 * v367;
                                    int v372;
                                    v372 = v371 + v369;
                                    float v373;
                                    v373 = v327[v372];
                                    float v374;
                                    v374 = v328[v372];
                                    bool v375;
                                    v375 = v374 == 0.0f;
                                    bool v376;
                                    v376 = v375 != true;
                                    float v378;
                                    if (v376){
                                        float v377;
                                        v377 = v373 / v374;
                                        v378 = v377;
                                    } else {
                                        v378 = 0.0f;
                                    }
                                    assert("Tensor range check" && 0 <= v367 && v367 < 1);
                                    assert("Tensor range check" && 0 <= v369 && v369 < 4);
                                    v366[v372] = v378;
                                    v369 += 1 ;
                                }
                                v367 += 1 ;
                            }
                            bool v379[4];
                            int v380;
                            v380 = 0;
                            while (while_method_6(v380)){
                                int v382;
                                v382 = 0;
                                while (while_method_0(v382)){
                                    assert("Tensor range check" && 0 <= v380 && v380 < 1);
                                    assert("Tensor range check" && 0 <= v382 && v382 < 4);
                                    int v384;
                                    v384 = 4 * v380;
                                    int v385;
                                    v385 = v384 + v382;
                                    float v386;
                                    v386 = v326[v385];
                                    int v387;
                                    v387 = v329[v385];
                                    bool v388;
                                    v388 = v387 < 11;
                                    assert("Tensor range check" && 0 <= v380 && v380 < 1);
                                    assert("Tensor range check" && 0 <= v382 && v382 < 4);
                                    v379[v385] = v388;
                                    v382 += 1 ;
                                }
                                v380 += 1 ;
                            }
                            float v389[4];
                            int v390;
                            v390 = 0;
                            while (while_method_6(v390)){
                                int v392;
                                v392 = 0;
                                while (while_method_0(v392)){
                                    assert("Tensor range check" && 0 <= v390 && v390 < 1);
                                    assert("Tensor range check" && 0 <= v392 && v392 < 4);
                                    int v394;
                                    v394 = 4 * v390;
                                    int v395;
                                    v395 = v394 + v392;
                                    float v396;
                                    v396 = v326[v395];
                                    bool v397;
                                    v397 = v379[v395];
                                    float v400;
                                    if (v397){
                                        bool v398;
                                        v398 = 0.0f >= v396;
                                        if (v398){
                                            v400 = 0.0f;
                                        } else {
                                            v400 = v396;
                                        }
                                    } else {
                                        v400 = 0.0f;
                                    }
                                    assert("Tensor range check" && 0 <= v390 && v390 < 1);
                                    assert("Tensor range check" && 0 <= v392 && v392 < 4);
                                    v389[v395] = v400;
                                    v392 += 1 ;
                                }
                                v390 += 1 ;
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
                                    v408 = v389[v407];
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
                            Closure1 v414{};
                            float v415;
                            v415 = cooperative_groups::reduce(v413, v401, v414);
                            int v416[4];
                            int v417;
                            v417 = 0;
                            while (while_method_6(v417)){
                                int v419;
                                v419 = 0;
                                while (while_method_0(v419)){
                                    assert("Tensor range check" && 0 <= v417 && v417 < 1);
                                    assert("Tensor range check" && 0 <= v419 && v419 < 4);
                                    int v421;
                                    v421 = 4 * v417;
                                    int v422;
                                    v422 = v421 + v419;
                                    bool v423;
                                    v423 = v379[v422];
                                    int v424;
                                    if (v423){
                                        v424 = 1;
                                    } else {
                                        v424 = 0;
                                    }
                                    assert("Tensor range check" && 0 <= v417 && v417 < 1);
                                    assert("Tensor range check" && 0 <= v419 && v419 < 4);
                                    v416[v422] = v424;
                                    v419 += 1 ;
                                }
                                v417 += 1 ;
                            }
                            int v425;
                            v425 = 0;
                            int v426;
                            v426 = 0;
                            while (while_method_6(v426)){
                                int v428;
                                v428 = 0;
                                while (while_method_0(v428)){
                                    assert("Tensor range check" && 0 <= v426 && v426 < 1);
                                    assert("Tensor range check" && 0 <= v428 && v428 < 4);
                                    int v430;
                                    v430 = 4 * v426;
                                    int v431;
                                    v431 = v430 + v428;
                                    int v432;
                                    v432 = v416[v431];
                                    int v433;
                                    v433 = v425 + v432;
                                    v425 = v433;
                                    v428 += 1 ;
                                }
                                v426 += 1 ;
                            }
                            auto v434 = cooperative_groups::coalesced_threads();
                            int v435;
                            v435 = threadIdx.x;
                            int v436;
                            v436 = v435 / 4;
                            auto v437 = cooperative_groups::labeled_partition(v434,v436);
                            Closure2 v438{};
                            int v439;
                            v439 = cooperative_groups::reduce(v437, v425, v438);
                            float v440;
                            v440 = (float)v439;
                            float v441;
                            v441 = 1.0f / v440;
                            float v442[4];
                            int v443;
                            v443 = 0;
                            while (while_method_6(v443)){
                                int v445;
                                v445 = 0;
                                while (while_method_0(v445)){
                                    assert("Tensor range check" && 0 <= v443 && v443 < 1);
                                    assert("Tensor range check" && 0 <= v445 && v445 < 4);
                                    int v447;
                                    v447 = 4 * v443;
                                    int v448;
                                    v448 = v447 + v445;
                                    float v449;
                                    v449 = v389[v448];
                                    bool v450;
                                    v450 = v379[v448];
                                    bool v451;
                                    v451 = v450 == false;
                                    float v456;
                                    if (v451){
                                        v456 = 0.0f;
                                    } else {
                                        bool v452;
                                        v452 = v415 == 0.0f;
                                        bool v453;
                                        v453 = v452 != true;
                                        if (v453){
                                            float v454;
                                            v454 = v449 / v415;
                                            v456 = v454;
                                        } else {
                                            v456 = v441;
                                        }
                                    }
                                    assert("Tensor range check" && 0 <= v443 && v443 < 1);
                                    assert("Tensor range check" && 0 <= v445 && v445 < 4);
                                    v442[v448] = v456;
                                    v445 += 1 ;
                                }
                                v443 += 1 ;
                            }
                            float v457[4];
                            int v458;
                            v458 = 0;
                            while (while_method_6(v458)){
                                int v460;
                                v460 = 0;
                                while (while_method_0(v460)){
                                    assert("Tensor range check" && 0 <= v458 && v458 < 1);
                                    assert("Tensor range check" && 0 <= v460 && v460 < 4);
                                    int v462;
                                    v462 = 4 * v458;
                                    int v463;
                                    v463 = v462 + v460;
                                    float v464;
                                    v464 = v366[v463];
                                    int v465;
                                    v465 = v329[v463];
                                    bool v466;
                                    v466 = v315 == v465;
                                    float v469;
                                    if (v466){
                                        float v467;
                                        v467 = v316 - v464;
                                        float v468;
                                        v468 = v467 / v314;
                                        v469 = v468;
                                    } else {
                                        v469 = 0.0f;
                                    }
                                    float v470;
                                    v470 = v469 + v464;
                                    assert("Tensor range check" && 0 <= v458 && v458 < 1);
                                    assert("Tensor range check" && 0 <= v460 && v460 < 4);
                                    v457[v463] = v470;
                                    v460 += 1 ;
                                }
                                v458 += 1 ;
                            }
                            float v471[4];
                            int v472;
                            v472 = 0;
                            while (while_method_6(v472)){
                                int v474;
                                v474 = 0;
                                while (while_method_0(v474)){
                                    assert("Tensor range check" && 0 <= v472 && v472 < 1);
                                    assert("Tensor range check" && 0 <= v474 && v474 < 4);
                                    int v476;
                                    v476 = 4 * v472;
                                    int v477;
                                    v477 = v476 + v474;
                                    float v478;
                                    v478 = v442[v477];
                                    float v479;
                                    v479 = v457[v477];
                                    float v480;
                                    v480 = v478 * v479;
                                    assert("Tensor range check" && 0 <= v472 && v472 < 1);
                                    assert("Tensor range check" && 0 <= v474 && v474 < 4);
                                    v471[v477] = v480;
                                    v474 += 1 ;
                                }
                                v472 += 1 ;
                            }
                            float v481;
                            v481 = 0.0f;
                            int v482;
                            v482 = 0;
                            while (while_method_6(v482)){
                                int v484;
                                v484 = 0;
                                while (while_method_0(v484)){
                                    assert("Tensor range check" && 0 <= v482 && v482 < 1);
                                    assert("Tensor range check" && 0 <= v484 && v484 < 4);
                                    int v486;
                                    v486 = 4 * v482;
                                    int v487;
                                    v487 = v486 + v484;
                                    float v488;
                                    v488 = v471[v487];
                                    float v489;
                                    v489 = v481 + v488;
                                    v481 = v489;
                                    v484 += 1 ;
                                }
                                v482 += 1 ;
                            }
                            auto v490 = cooperative_groups::coalesced_threads();
                            int v491;
                            v491 = threadIdx.x;
                            int v492;
                            v492 = v491 / 4;
                            auto v493 = cooperative_groups::labeled_partition(v490,v492);
                            float v494;
                            v494 = cooperative_groups::reduce(v493, v481, v414);
                            int v495;
                            v495 = 0;
                            while (while_method_6(v495)){
                                int v497;
                                v497 = 0;
                                while (while_method_0(v497)){
                                    assert("Tensor range check" && 0 <= v495 && v495 < 1);
                                    assert("Tensor range check" && 0 <= v497 && v497 < 4);
                                    int v499;
                                    v499 = 4 * v495;
                                    int v500;
                                    v500 = v499 + v497;
                                    float v501;
                                    v501 = v457[v500];
                                    int v502;
                                    v502 = v329[v500];
                                    float v503;
                                    v503 = v501 - v494;
                                    float v504;
                                    v504 = v317 * v503;
                                    assert("Tensor range check" && 0 <= v502 && v502 < 16);
                                    float * v505;
                                    v505 = v318+v502;
                                    float v507;
                                    v507 = atomicAdd(v505,v504);
                                    v497 += 1 ;
                                }
                                v495 += 1 ;
                            }
                            int v508;
                            v508 = 0;
                            while (while_method_6(v508)){
                                assert("Tensor range check" && 0 <= v508 && v508 < 1);
                                assert("Tensor range check" && 0 <= v508 && v508 < 1);
                                v508 += 1 ;
                            }
                            assert("Tensor range check" && 0 <= v311 && v311 < 256);
                            v288[v311] = v494;
                            v299 += 1 ;
                        }
                        asm("barrier.cta.sync %0;" :: "r"(0));
                        assert("Tensor range check" && 0 <= v290 && v290 < 256);
                        float v510;
                        v510 = v288[v290];
                        asm("barrier.cta.sync %0;" :: "r"(0));
                        assert("Tensor range check" && 0 <= v182 && v182 < 2);
                        v161[v182] = v510;
                    }
                    int v511;
                    v511 = threadIdx.x;
                    int v512;
                    v512 = blockIdx.x;
                    int v513;
                    v513 = v512 * 256;
                    int v514;
                    v514 = v511 + v513;
                    assert("Tensor range check" && 0 <= v155 && v155 < 4);
                    int v515;
                    v515 = 12288 * v155;
                    assert("Tensor range check" && 0 <= v514 && v514 < 6144);
                    int v516;
                    v516 = 2 * v514;
                    int v517;
                    v517 = v516 + v515;
                    double * v518;
                    v518 = v149+v517;
                    double * v520;
                    v520 = v151+v517;
                    double * v522;
                    v522 = v518+0;
                    double * v524;
                    v524 = v520+0;
                    double * v526;
                    v526 = v518+0;
                    double * v528;
                    v528 = v520+0;
                    int v530;
                    v530 = sizeof(double *);
                    unsigned long long v531;
                    v531 = (unsigned long long)v530;
                    unsigned long long v532;
                    v532 = 256ull * v531;
                    unsigned long long v533;
                    v533 = v532 + 16ull;
                    unsigned long long v534;
                    v534 = v533 - 1ull;
                    unsigned long long v535;
                    v535 = v534 % 16ull;
                    unsigned long long v536;
                    v536 = v534 - v535;
                    unsigned long long v537;
                    v537 = v536 + v532;
                    unsigned long long v538;
                    v538 = v537 + 16ull;
                    unsigned long long v539;
                    v539 = v538 - 1ull;
                    unsigned long long v540;
                    v540 = v539 % 16ull;
                    unsigned long long v541;
                    v541 = v539 - v540;
                    unsigned long long v542;
                    v542 = v541 + v532;
                    unsigned long long v543;
                    v543 = v542 + 16ull;
                    unsigned long long v544;
                    v544 = v543 - 1ull;
                    unsigned long long v545;
                    v545 = v544 % 16ull;
                    unsigned long long v546;
                    v546 = v544 - v545;
                    unsigned long long v547;
                    v547 = v546 + v532;
                    bool v548;
                    v548 = v547 <= 98304ull;
                    bool v549;
                    v549 = v548 == false;
                    if (v549){
                        assert("The dynamic shared memory is insufficient to allocate the tensor." && v548);
                    } else {
                    }
                    extern __shared__ unsigned char v551[];
                    bool v552;
                    v552 = v547 <= v547;
                    bool v553;
                    v553 = v552 == false;
                    if (v553){
                        assert("The length of the partition has to be less than or equal to the length of the base array." && v552);
                    } else {
                    }
                    double * * v555;
                    v555 = reinterpret_cast<double * *>(&v551[0ull]);
                    double * * v557;
                    v557 = reinterpret_cast<double * *>(&v551[v536]);
                    double * * v559;
                    v559 = reinterpret_cast<double * *>(&v551[v541]);
                    double * * v561;
                    v561 = reinterpret_cast<double * *>(&v551[v546]);
                    int v563;
                    v563 = threadIdx.x;
                    assert("Tensor range check" && 0 <= v563 && v563 < 256);
                    v555[v563] = v522;
                    v557[v563] = v524;
                    v559[v563] = v526;
                    v561[v563] = v528;
                    asm("barrier.cta.sync %0;" :: "r"(0));
                    bool v564;
                    v564 = 0 <= v563;
                    bool v565;
                    v565 = v564 == false;
                    if (v565){
                        assert("The index needs to be zero or positive." && v564);
                    } else {
                    }
                    int v567;
                    v567 = v563 % 1;
                    bool v568;
                    v568 = v563 < 256;
                    bool v569;
                    v569 = v568 == false;
                    if (v569){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v568);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v563 && v563 < 256);
                    int v571;
                    v571 = 0;
                    while (while_method_6(v571)){
                        bool v573;
                        v573 = v564 && v568;
                        bool v574;
                        v574 = v573 == false;
                        if (v574){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v573);
                        } else {
                        }
                        bool v576;
                        v576 = 0 <= v571;
                        bool v578;
                        if (v576){
                            bool v577;
                            v577 = v571 < 1;
                            v578 = v577;
                        } else {
                            v578 = false;
                        }
                        bool v579;
                        v579 = v578 == false;
                        if (v579){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v578);
                        } else {
                        }
                        int v581;
                        v581 = v571 * 256;
                        int v582;
                        v582 = v581 + v563;
                        assert("Tensor range check" && 0 <= v571 && v571 < 1);
                        int v583;
                        v583 = 256 * v571;
                        int v584;
                        v584 = v583 + v563;
                        double * v585;
                        v585 = v555[v584];
                        double * v586;
                        v586 = v557[v584];
                        double * v587;
                        v587 = v559[v584];
                        double * v588;
                        v588 = v561[v584];
                        int v589;
                        v589 = blockIdx.x;
                        int v590;
                        v590 = v589 * 256;
                        int v591;
                        v591 = v590 + v582;
                        assert("Tensor range check" && 0 <= v567 && v567 < 1);
                        int v592;
                        v592 = 2 * v567;
                        double v593[2];
                        double v594[2];
                        int v595[2];
                        int v596;
                        v596 = 0;
                        while (while_method_6(v596)){
                            assert("Tensor range check" && 0 <= v596 && v596 < 1);
                            int v598;
                            v598 = 2 * v596;
                            assert("Tensor range check" && 0 <= v596 && v596 < 1);
                            int v599;
                            v599 = v598 + v592;
                            int4* v600;
                            v600 = reinterpret_cast<int4*>(v585 + v599);
                            int4* v601;
                            v601 = reinterpret_cast<int4*>(v593 + v598);
                            assert("Pointer alignment check" && (unsigned long long)(v600) % 2 == 0 && (unsigned long long)(v601) % 2 == 0);
                            *v601 = *v600;
                            int4* v602;
                            v602 = reinterpret_cast<int4*>(v586 + v599);
                            int4* v603;
                            v603 = reinterpret_cast<int4*>(v594 + v598);
                            assert("Pointer alignment check" && (unsigned long long)(v602) % 2 == 0 && (unsigned long long)(v603) % 2 == 0);
                            *v603 = *v602;
                            v596 += 1 ;
                        }
                        int v604;
                        v604 = 0;
                        while (while_method_6(v604)){
                            int v606;
                            v606 = 0;
                            while (while_method_2(v606)){
                                bool v608;
                                v608 = 0 <= v606;
                                bool v610;
                                if (v608){
                                    bool v609;
                                    v609 = v606 < 2;
                                    v610 = v609;
                                } else {
                                    v610 = false;
                                }
                                bool v611;
                                v611 = v610 == false;
                                if (v611){
                                    assert("The indices should be inside the range of the dimension." && v610);
                                } else {
                                }
                                bool v613;
                                v613 = 0 <= v567;
                                bool v615;
                                if (v613){
                                    bool v614;
                                    v614 = v567 < 1;
                                    v615 = v614;
                                } else {
                                    v615 = false;
                                }
                                bool v616;
                                v616 = v615 == false;
                                if (v616){
                                    assert("The indices should be inside the range of the dimension." && v615);
                                } else {
                                }
                                int v618;
                                v618 = v567 * 2;
                                int v619;
                                v619 = v606 + v618;
                                bool v620;
                                v620 = 0 <= v604;
                                bool v622;
                                if (v620){
                                    bool v621;
                                    v621 = v604 < 1;
                                    v622 = v621;
                                } else {
                                    v622 = false;
                                }
                                bool v623;
                                v623 = v622 == false;
                                if (v623){
                                    assert("The indices should be inside the range of the dimension." && v622);
                                } else {
                                }
                                int v625;
                                v625 = v604 * 2;
                                int v626;
                                v626 = v619 + v625;
                                assert("Tensor range check" && 0 <= v604 && v604 < 1);
                                assert("Tensor range check" && 0 <= v606 && v606 < 2);
                                int v627;
                                v627 = 2 * v604;
                                int v628;
                                v628 = v627 + v606;
                                v595[v628] = v626;
                                v606 += 1 ;
                            }
                            v604 += 1 ;
                        }
                        double v629[2];
                        double v630[2];
                        int v631;
                        v631 = 0;
                        while (while_method_6(v631)){
                            int v633;
                            v633 = 0;
                            while (while_method_2(v633)){
                                assert("Tensor range check" && 0 <= v631 && v631 < 1);
                                assert("Tensor range check" && 0 <= v633 && v633 < 2);
                                int v635;
                                v635 = 2 * v631;
                                int v636;
                                v636 = v635 + v633;
                                double v637;
                                v637 = v593[v636];
                                double v638;
                                v638 = v594[v636];
                                assert("Tensor range check" && 0 <= v631 && v631 < 1);
                                assert("Tensor range check" && 0 <= v633 && v633 < 2);
                                v629[v636] = 0.0;
                                v630[v636] = 0.0;
                                v633 += 1 ;
                            }
                            v631 += 1 ;
                        }
                        int v639;
                        v639 = 0;
                        while (while_method_6(v639)){
                            assert("Tensor range check" && 0 <= v639 && v639 < 1);
                            int v641;
                            v641 = 2 * v639;
                            int v642;
                            v642 = v641 + v592;
                            assert("Tensor range check" && 0 <= v639 && v639 < 1);
                            int4* v643;
                            v643 = reinterpret_cast<int4*>(v629 + v641);
                            int4* v644;
                            v644 = reinterpret_cast<int4*>(v587 + v642);
                            assert("Pointer alignment check" && (unsigned long long)(v643) % 2 == 0 && (unsigned long long)(v644) % 2 == 0);
                            *v644 = *v643;
                            int4* v645;
                            v645 = reinterpret_cast<int4*>(v630 + v641);
                            int4* v646;
                            v646 = reinterpret_cast<int4*>(v588 + v642);
                            assert("Pointer alignment check" && (unsigned long long)(v645) % 2 == 0 && (unsigned long long)(v646) % 2 == 0);
                            *v646 = *v645;
                            v639 += 1 ;
                        }
                        assert("Tensor range check" && 0 <= v582 && v582 < 256);
                        v571 += 1 ;
                    }
                    asm("barrier.cta.sync %0;" :: "r"(0));
                    assert("Tensor range check" && 0 <= v563 && v563 < 256);
                    asm("barrier.cta.sync %0;" :: "r"(0));
                    assert("Tensor range check" && 0 <= v155 && v155 < 4);
                    assert("Tensor range check" && 0 <= v514 && v514 < 6144);
                    int v647;
                    v647 = v171 + v514;
                    v153[v647] = 0;
                    v155 += 1 ;
                }
                v38 += 1 ;
            }
            v36 += 1 ;
        }
        cooperative_groups::grid_group & v648 = v27.v1;
        cooperative_groups::grid_group & v649 = v648;
        curandStatePhilox4_32_10_t & v650 = v27.v5;
        curandStatePhilox4_32_10_t & v651 = v650;
        unsigned int * v652;
        v652 = reinterpret_cast<unsigned int *>(&v0[53477376ull]);
        int * v654;
        v654 = reinterpret_cast<int *>(&v2[4194304ull]);
        float * v656;
        v656 = reinterpret_cast<float *>(&v2[4194320ull]);
        float * v658;
        v658 = reinterpret_cast<float *>(&v2[5242896ull]);
        float * v660;
        v660 = reinterpret_cast<float *>(&v2[6291472ull]);
        float * v662;
        v662 = reinterpret_cast<float *>(&v2[7340048ull]);
        float * v664;
        v664 = reinterpret_cast<float *>(&v2[8388624ull]);
        float * v666;
        v666 = reinterpret_cast<float *>(&v2[9437200ull]);
        float * v668;
        v668 = reinterpret_cast<float *>(&v2[10485776ull]);
        int * v670;
        v670 = reinterpret_cast<int *>(&v0[53575680ull]);
        float * v672;
        v672 = reinterpret_cast<float *>(&v0[66158592ull]);
        int * v674;
        v674 = reinterpret_cast<int *>(&v0[78741504ull]);
        int * v676;
        v676 = reinterpret_cast<int *>(&v0[91324416ull]);
        double * v678;
        v678 = reinterpret_cast<double *>(&v0[103907328ull]);
        double * v680;
        v680 = reinterpret_cast<double *>(&v0[154238976ull]);
        double * v682;
        v682 = reinterpret_cast<double *>(&v2[11534352ull]);
        double * v684;
        v684 = reinterpret_cast<double *>(&v2[11927568ull]);
        int * v686;
        v686 = reinterpret_cast<int *>(&v2[12320784ull]);
        v649.sync() ;
        int v688;
        v688 = threadIdx.x;
        int v689;
        v689 = blockIdx.x;
        int v690;
        v690 = v689 * 256;
        int v691;
        v691 = v688 + v690;
        bool v692;
        v692 = v691 == 0;
        if (v692){
            int v693;
            v693 = 0;
            int v694;
            v694 = 4;
            int v695;
            v695 = int_range_21(v694, v693, v651);
            v654[0] = v695;
        } else {
        }
        __syncwarp();
        int v696;
        v696 = threadIdx.x;
        bool v697;
        v697 = 0 <= v696;
        bool v698;
        v698 = v697 == false;
        if (v698){
            assert("The index needs to be zero or positive." && v697);
        } else {
        }
        int v700;
        v700 = v696 % 4;
        int v701;
        v701 = v696 / 4;
        int v702;
        v702 = v701 % 64;
        int v703;
        v703 = v701 / 64;
        bool v704;
        v704 = v703 < 1;
        bool v705;
        v705 = v704 == false;
        if (v705){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v704);
        } else {
        }
        assert("Tensor range check" && 0 <= v703 && v703 < 1);
        assert("Tensor range check" && 0 <= v702 && v702 < 64);
        assert("Tensor range check" && 0 <= v700 && v700 < 4);
        int v707;
        v707 = 4 * v700;
        int v708;
        v708 = 16 * v702;
        int v709;
        v709 = v708 + v707;
        int v710;
        v710 = 65536 * v703;
        int v711;
        v711 = v710 + v709;
        assert("Tensor range check" && 0 <= v703 && v703 < 1);
        assert("Tensor range check" && 0 <= v702 && v702 < 64);
        assert("Tensor range check" && 0 <= v700 && v700 < 4);
        int v712;
        v712 = blockIdx.x;
        int v713;
        v713 = v712;
        while (while_method_22(v713)){
            bool v715;
            v715 = 0 <= v713;
            bool v716;
            v716 = v715 == false;
            if (v716){
                assert("The index needs to be zero or positive." && v715);
            } else {
            }
            int v718;
            v718 = v713 % 64;
            int v719;
            v719 = v713 / 64;
            bool v720;
            v720 = v719 < 4;
            bool v721;
            v721 = v720 == false;
            if (v721){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v720);
            } else {
            }
            assert("Tensor range check" && 0 <= v719 && v719 < 4);
            assert("Tensor range check" && 0 <= v718 && v718 < 64);
            int v723;
            v723 = 1024 * v718;
            int v724;
            v724 = v723 + v711;
            int v725;
            v725 = 65536 * v719;
            int v726;
            v726 = v725 + v724;
            float v727[4];
            float v728[4];
            float v729[4];
            float v730[4];
            float v731[4];
            float v732[4];
            float v733[4];
            int v734[4];
            int v735;
            v735 = 0;
            while (while_method_6(v735)){
                assert("Tensor range check" && 0 <= v735 && v735 < 1);
                int v737;
                v737 = 4 * v735;
                assert("Tensor range check" && 0 <= v735 && v735 < 1);
                int v738;
                v738 = 16 * v735;
                int v739;
                v739 = v738 + v726;
                int4* v740;
                v740 = reinterpret_cast<int4*>(v656 + v739);
                int4* v741;
                v741 = reinterpret_cast<int4*>(v727 + v737);
                assert("Pointer alignment check" && (unsigned long long)(v740) % 4 == 0 && (unsigned long long)(v741) % 4 == 0);
                *v741 = *v740;
                int4* v742;
                v742 = reinterpret_cast<int4*>(v658 + v739);
                int4* v743;
                v743 = reinterpret_cast<int4*>(v728 + v737);
                assert("Pointer alignment check" && (unsigned long long)(v742) % 4 == 0 && (unsigned long long)(v743) % 4 == 0);
                *v743 = *v742;
                int4* v744;
                v744 = reinterpret_cast<int4*>(v660 + v739);
                int4* v745;
                v745 = reinterpret_cast<int4*>(v729 + v737);
                assert("Pointer alignment check" && (unsigned long long)(v744) % 4 == 0 && (unsigned long long)(v745) % 4 == 0);
                *v745 = *v744;
                int4* v746;
                v746 = reinterpret_cast<int4*>(v662 + v739);
                int4* v747;
                v747 = reinterpret_cast<int4*>(v730 + v737);
                assert("Pointer alignment check" && (unsigned long long)(v746) % 4 == 0 && (unsigned long long)(v747) % 4 == 0);
                *v747 = *v746;
                int4* v748;
                v748 = reinterpret_cast<int4*>(v664 + v739);
                int4* v749;
                v749 = reinterpret_cast<int4*>(v731 + v737);
                assert("Pointer alignment check" && (unsigned long long)(v748) % 4 == 0 && (unsigned long long)(v749) % 4 == 0);
                *v749 = *v748;
                int4* v750;
                v750 = reinterpret_cast<int4*>(v666 + v739);
                int4* v751;
                v751 = reinterpret_cast<int4*>(v732 + v737);
                assert("Pointer alignment check" && (unsigned long long)(v750) % 4 == 0 && (unsigned long long)(v751) % 4 == 0);
                *v751 = *v750;
                int4* v752;
                v752 = reinterpret_cast<int4*>(v668 + v739);
                int4* v753;
                v753 = reinterpret_cast<int4*>(v733 + v737);
                assert("Pointer alignment check" && (unsigned long long)(v752) % 4 == 0 && (unsigned long long)(v753) % 4 == 0);
                *v753 = *v752;
                v735 += 1 ;
            }
            int v754;
            v754 = 0;
            while (while_method_6(v754)){
                int v756;
                v756 = 0;
                while (while_method_0(v756)){
                    bool v758;
                    v758 = 0 <= v756;
                    bool v760;
                    if (v758){
                        bool v759;
                        v759 = v756 < 4;
                        v760 = v759;
                    } else {
                        v760 = false;
                    }
                    bool v761;
                    v761 = v760 == false;
                    if (v761){
                        assert("The indices should be inside the range of the dimension." && v760);
                    } else {
                    }
                    bool v763;
                    v763 = 0 <= v700;
                    bool v765;
                    if (v763){
                        bool v764;
                        v764 = v700 < 4;
                        v765 = v764;
                    } else {
                        v765 = false;
                    }
                    bool v766;
                    v766 = v765 == false;
                    if (v766){
                        assert("The indices should be inside the range of the dimension." && v765);
                    } else {
                    }
                    int v768;
                    v768 = v700 * 4;
                    int v769;
                    v769 = v756 + v768;
                    bool v770;
                    v770 = 0 <= v754;
                    bool v772;
                    if (v770){
                        bool v771;
                        v771 = v754 < 1;
                        v772 = v771;
                    } else {
                        v772 = false;
                    }
                    bool v773;
                    v773 = v772 == false;
                    if (v773){
                        assert("The indices should be inside the range of the dimension." && v772);
                    } else {
                    }
                    int v775;
                    v775 = v754 * 16;
                    int v776;
                    v776 = v769 + v775;
                    assert("Tensor range check" && 0 <= v754 && v754 < 1);
                    assert("Tensor range check" && 0 <= v756 && v756 < 4);
                    int v777;
                    v777 = 4 * v754;
                    int v778;
                    v778 = v777 + v756;
                    v734[v778] = v776;
                    v756 += 1 ;
                }
                v754 += 1 ;
            }
            bool v779;
            v779 = 0 <= v703;
            bool v780;
            v780 = v779 && v704;
            bool v781;
            v781 = v780 == false;
            if (v781){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v780);
            } else {
            }
            bool v783;
            v783 = 0 <= v702;
            bool v785;
            if (v783){
                bool v784;
                v784 = v702 < 64;
                v785 = v784;
            } else {
                v785 = false;
            }
            bool v786;
            v786 = v785 == false;
            if (v786){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v785);
            } else {
            }
            bool v788;
            v788 = 0 <= v719;
            bool v789;
            v789 = v788 && v720;
            bool v790;
            v790 = v789 == false;
            if (v790){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v789);
            } else {
            }
            bool v792;
            v792 = 0 <= v718;
            bool v794;
            if (v792){
                bool v793;
                v793 = v718 < 64;
                v794 = v793;
            } else {
                v794 = false;
            }
            bool v795;
            v795 = v794 == false;
            if (v795){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v794);
            } else {
            }
            int v797;
            v797 = v718 * 64;
            int v798;
            v798 = v719 + v703;
            int v799;
            v799 = v797 + v702;
            bool v800[4];
            int v801;
            v801 = 0;
            while (while_method_6(v801)){
                int v803;
                v803 = 0;
                while (while_method_0(v803)){
                    assert("Tensor range check" && 0 <= v801 && v801 < 1);
                    assert("Tensor range check" && 0 <= v803 && v803 < 4);
                    int v805;
                    v805 = 4 * v801;
                    int v806;
                    v806 = v805 + v803;
                    float v807;
                    v807 = v729[v806];
                    bool v808;
                    v808 = v807 == 0.0f;
                    bool v809;
                    v809 = v808 != true;
                    assert("Tensor range check" && 0 <= v801 && v801 < 1);
                    assert("Tensor range check" && 0 <= v803 && v803 < 4);
                    v800[v806] = v809;
                    v803 += 1 ;
                }
                v801 += 1 ;
            }
            bool v810;
            v810 = false;
            int v811;
            v811 = 0;
            while (while_method_6(v811)){
                int v813;
                v813 = 0;
                while (while_method_0(v813)){
                    assert("Tensor range check" && 0 <= v811 && v811 < 1);
                    assert("Tensor range check" && 0 <= v813 && v813 < 4);
                    int v815;
                    v815 = 4 * v811;
                    int v816;
                    v816 = v815 + v813;
                    bool v817;
                    v817 = v800[v816];
                    bool v818;
                    v818 = v810 || v817;
                    v810 = v818;
                    v813 += 1 ;
                }
                v811 += 1 ;
            }
            auto v819 = cooperative_groups::coalesced_threads();
            int v820;
            v820 = threadIdx.x;
            int v821;
            v821 = v820 / 4;
            auto v822 = cooperative_groups::labeled_partition(v819,v821);
            Closure8 v823{};
            bool v824;
            v824 = cooperative_groups::reduce(v822, v810, v823);
            if (v824){
                float v825[4];
                int v826;
                v826 = 0;
                while (while_method_6(v826)){
                    int v828;
                    v828 = 0;
                    while (while_method_0(v828)){
                        assert("Tensor range check" && 0 <= v826 && v826 < 1);
                        assert("Tensor range check" && 0 <= v828 && v828 < 4);
                        int v830;
                        v830 = 4 * v826;
                        int v831;
                        v831 = v830 + v828;
                        float v832;
                        v832 = v728[v831];
                        float v833;
                        v833 = v729[v831];
                        float v834;
                        v834 = v832 + v833;
                        bool v835;
                        v835 = 0.0f >= v834;
                        float v836;
                        if (v835){
                            v836 = 0.0f;
                        } else {
                            v836 = v834;
                        }
                        assert("Tensor range check" && 0 <= v826 && v826 < 1);
                        assert("Tensor range check" && 0 <= v828 && v828 < 4);
                        v825[v831] = v836;
                        v828 += 1 ;
                    }
                    v826 += 1 ;
                }
                float v837[4];
                int v838;
                v838 = 0;
                while (while_method_6(v838)){
                    int v840;
                    v840 = 0;
                    while (while_method_0(v840)){
                        assert("Tensor range check" && 0 <= v838 && v838 < 1);
                        assert("Tensor range check" && 0 <= v840 && v840 < 4);
                        int v842;
                        v842 = 4 * v838;
                        int v843;
                        v843 = v842 + v840;
                        float v844;
                        v844 = v825[v843];
                        bool v845;
                        v845 = 0.0f >= v844;
                        float v846;
                        if (v845){
                            v846 = 0.0f;
                        } else {
                            v846 = v844;
                        }
                        assert("Tensor range check" && 0 <= v838 && v838 < 1);
                        assert("Tensor range check" && 0 <= v840 && v840 < 4);
                        v837[v843] = v846;
                        v840 += 1 ;
                    }
                    v838 += 1 ;
                }
                float v847;
                v847 = 0.0f;
                int v848;
                v848 = 0;
                while (while_method_6(v848)){
                    int v850;
                    v850 = 0;
                    while (while_method_0(v850)){
                        assert("Tensor range check" && 0 <= v848 && v848 < 1);
                        assert("Tensor range check" && 0 <= v850 && v850 < 4);
                        int v852;
                        v852 = 4 * v848;
                        int v853;
                        v853 = v852 + v850;
                        float v854;
                        v854 = v837[v853];
                        float v855;
                        v855 = v847 + v854;
                        v847 = v855;
                        v850 += 1 ;
                    }
                    v848 += 1 ;
                }
                auto v856 = cooperative_groups::coalesced_threads();
                int v857;
                v857 = threadIdx.x;
                int v858;
                v858 = v857 / 4;
                auto v859 = cooperative_groups::labeled_partition(v856,v858);
                Closure1 v860{};
                float v861;
                v861 = cooperative_groups::reduce(v859, v847, v860);
                float v862[4];
                int v863;
                v863 = 0;
                while (while_method_6(v863)){
                    int v865;
                    v865 = 0;
                    while (while_method_0(v865)){
                        assert("Tensor range check" && 0 <= v863 && v863 < 1);
                        assert("Tensor range check" && 0 <= v865 && v865 < 4);
                        int v867;
                        v867 = 4 * v863;
                        int v868;
                        v868 = v867 + v865;
                        float v869;
                        v869 = v837[v868];
                        bool v870;
                        v870 = v861 == 0.0f;
                        bool v871;
                        v871 = v870 != true;
                        float v873;
                        if (v871){
                            float v872;
                            v872 = v869 / v861;
                            v873 = v872;
                        } else {
                            v873 = 0.0625f;
                        }
                        assert("Tensor range check" && 0 <= v863 && v863 < 1);
                        assert("Tensor range check" && 0 <= v865 && v865 < 4);
                        v862[v868] = v873;
                        v865 += 1 ;
                    }
                    v863 += 1 ;
                }
                float v874[4];
                int v875;
                v875 = 0;
                while (while_method_6(v875)){
                    int v877;
                    v877 = 0;
                    while (while_method_0(v877)){
                        assert("Tensor range check" && 0 <= v875 && v875 < 1);
                        assert("Tensor range check" && 0 <= v877 && v877 < 4);
                        int v879;
                        v879 = 4 * v875;
                        int v880;
                        v880 = v879 + v877;
                        float v881;
                        v881 = v727[v880];
                        float v882;
                        v882 = v862[v880];
                        float v883;
                        v883 = v881 + v882;
                        assert("Tensor range check" && 0 <= v875 && v875 < 1);
                        assert("Tensor range check" && 0 <= v877 && v877 < 4);
                        v874[v880] = v883;
                        v877 += 1 ;
                    }
                    v875 += 1 ;
                }
                float v884[4];
                int v885;
                v885 = 0;
                while (while_method_6(v885)){
                    int v887;
                    v887 = 0;
                    while (while_method_0(v887)){
                        assert("Tensor range check" && 0 <= v885 && v885 < 1);
                        assert("Tensor range check" && 0 <= v887 && v887 < 4);
                        int v889;
                        v889 = 4 * v885;
                        int v890;
                        v890 = v889 + v887;
                        float v891;
                        v891 = v874[v890];
                        float v892;
                        v892 = -v891;
                        bool v893;
                        v893 = v891 >= v892;
                        float v894;
                        if (v893){
                            v894 = v891;
                        } else {
                            v894 = v892;
                        }
                        assert("Tensor range check" && 0 <= v885 && v885 < 1);
                        assert("Tensor range check" && 0 <= v887 && v887 < 4);
                        v884[v890] = v894;
                        v887 += 1 ;
                    }
                    v885 += 1 ;
                }
                float v895;
                v895 = 0.0f;
                int v896;
                v896 = 0;
                while (while_method_6(v896)){
                    int v898;
                    v898 = 0;
                    while (while_method_0(v898)){
                        assert("Tensor range check" && 0 <= v896 && v896 < 1);
                        assert("Tensor range check" && 0 <= v898 && v898 < 4);
                        int v900;
                        v900 = 4 * v896;
                        int v901;
                        v901 = v900 + v898;
                        float v902;
                        v902 = v884[v901];
                        float v903;
                        v903 = v895 + v902;
                        v895 = v903;
                        v898 += 1 ;
                    }
                    v896 += 1 ;
                }
                auto v904 = cooperative_groups::coalesced_threads();
                int v905;
                v905 = threadIdx.x;
                int v906;
                v906 = v905 / 4;
                auto v907 = cooperative_groups::labeled_partition(v904,v906);
                float v908;
                v908 = cooperative_groups::reduce(v907, v895, v860);
                bool v909;
                v909 = v908 > 100.0f;
                float v911;
                if (v909){
                    float v910;
                    v910 = 100.0f / v908;
                    v911 = v910;
                } else {
                    v911 = 1.0f;
                }
                float v912[4];
                int v913;
                v913 = 0;
                while (while_method_6(v913)){
                    int v915;
                    v915 = 0;
                    while (while_method_0(v915)){
                        assert("Tensor range check" && 0 <= v913 && v913 < 1);
                        assert("Tensor range check" && 0 <= v915 && v915 < 4);
                        int v917;
                        v917 = 4 * v913;
                        int v918;
                        v918 = v917 + v915;
                        float v919;
                        v919 = v884[v918];
                        float v920;
                        v920 = v911 * v919;
                        assert("Tensor range check" && 0 <= v913 && v913 < 1);
                        assert("Tensor range check" && 0 <= v915 && v915 < 4);
                        v912[v918] = v920;
                        v915 += 1 ;
                    }
                    v913 += 1 ;
                }
                float v921[4];
                float v922[4];
                int v923;
                v923 = 0;
                while (while_method_6(v923)){
                    int v925;
                    v925 = 0;
                    while (while_method_0(v925)){
                        assert("Tensor range check" && 0 <= v923 && v923 < 1);
                        assert("Tensor range check" && 0 <= v925 && v925 < 4);
                        int v927;
                        v927 = 4 * v923;
                        int v928;
                        v928 = v927 + v925;
                        float v929;
                        v929 = v727[v928];
                        float v930;
                        v930 = v728[v928];
                        float v931;
                        v931 = v729[v928];
                        float v932;
                        v932 = v730[v928];
                        float v933;
                        v933 = v731[v928];
                        float v934;
                        v934 = v732[v928];
                        float v935;
                        v935 = v733[v928];
                        float v936;
                        v936 = v932 + v934;
                        float v937;
                        v937 = v933 + v935;
                        assert("Tensor range check" && 0 <= v923 && v923 < 1);
                        assert("Tensor range check" && 0 <= v925 && v925 < 4);
                        v921[v928] = v936;
                        v922[v928] = v937;
                        v925 += 1 ;
                    }
                    v923 += 1 ;
                }
                int v938;
                v938 = 0;
                while (while_method_6(v938)){
                    int v940;
                    v940 = 0;
                    while (while_method_0(v940)){
                        assert("Tensor range check" && 0 <= v938 && v938 < 1);
                        assert("Tensor range check" && 0 <= v940 && v940 < 4);
                        int v942;
                        v942 = 4 * v938;
                        int v943;
                        v943 = v942 + v940;
                        float v944;
                        v944 = v912[v943];
                        float v945;
                        v945 = v825[v943];
                        float v946;
                        v946 = v921[v943];
                        float v947;
                        v947 = v922[v943];
                        assert("Tensor range check" && 0 <= v938 && v938 < 1);
                        assert("Tensor range check" && 0 <= v940 && v940 < 4);
                        v727[v943] = v944;
                        v728[v943] = v945;
                        v729[v943] = 0.0f;
                        v730[v943] = v946;
                        v731[v943] = v947;
                        v732[v943] = 0.0f;
                        v733[v943] = 0.0f;
                        v940 += 1 ;
                    }
                    v938 += 1 ;
                }
            } else {
            }
            assert("Tensor range check" && 0 <= v719 && v719 < 4);
            assert("Tensor range check" && 0 <= v718 && v718 < 64);
            int v948;
            v948 = 0;
            while (while_method_6(v948)){
                assert("Tensor range check" && 0 <= v948 && v948 < 1);
                int v950;
                v950 = 16 * v948;
                int v951;
                v951 = v950 + v726;
                assert("Tensor range check" && 0 <= v948 && v948 < 1);
                int v952;
                v952 = 4 * v948;
                int4* v953;
                v953 = reinterpret_cast<int4*>(v727 + v952);
                int4* v954;
                v954 = reinterpret_cast<int4*>(v656 + v951);
                assert("Pointer alignment check" && (unsigned long long)(v953) % 4 == 0 && (unsigned long long)(v954) % 4 == 0);
                *v954 = *v953;
                int4* v955;
                v955 = reinterpret_cast<int4*>(v728 + v952);
                int4* v956;
                v956 = reinterpret_cast<int4*>(v658 + v951);
                assert("Pointer alignment check" && (unsigned long long)(v955) % 4 == 0 && (unsigned long long)(v956) % 4 == 0);
                *v956 = *v955;
                int4* v957;
                v957 = reinterpret_cast<int4*>(v729 + v952);
                int4* v958;
                v958 = reinterpret_cast<int4*>(v660 + v951);
                assert("Pointer alignment check" && (unsigned long long)(v957) % 4 == 0 && (unsigned long long)(v958) % 4 == 0);
                *v958 = *v957;
                int4* v959;
                v959 = reinterpret_cast<int4*>(v730 + v952);
                int4* v960;
                v960 = reinterpret_cast<int4*>(v662 + v951);
                assert("Pointer alignment check" && (unsigned long long)(v959) % 4 == 0 && (unsigned long long)(v960) % 4 == 0);
                *v960 = *v959;
                int4* v961;
                v961 = reinterpret_cast<int4*>(v731 + v952);
                int4* v962;
                v962 = reinterpret_cast<int4*>(v664 + v951);
                assert("Pointer alignment check" && (unsigned long long)(v961) % 4 == 0 && (unsigned long long)(v962) % 4 == 0);
                *v962 = *v961;
                int4* v963;
                v963 = reinterpret_cast<int4*>(v732 + v952);
                int4* v964;
                v964 = reinterpret_cast<int4*>(v666 + v951);
                assert("Pointer alignment check" && (unsigned long long)(v963) % 4 == 0 && (unsigned long long)(v964) % 4 == 0);
                *v964 = *v963;
                int4* v965;
                v965 = reinterpret_cast<int4*>(v733 + v952);
                int4* v966;
                v966 = reinterpret_cast<int4*>(v668 + v951);
                assert("Pointer alignment check" && (unsigned long long)(v965) % 4 == 0 && (unsigned long long)(v966) % 4 == 0);
                *v966 = *v965;
                v948 += 1 ;
            }
            v713 += 24 ;
        }
        v649.sync() ;
        v34 += 1 ;
    }
    cooperative_groups::grid_group & v967 = v27.v1;
    cooperative_groups::grid_group & v968 = v967;
    int v969;
    v969 = threadIdx.x;
    int v970;
    v970 = blockIdx.x;
    int v971;
    v971 = v970 * 256;
    int v972;
    v972 = v969 + v971;
    int v973;
    v973 = v972;
    while (while_method_0(v973)){
        bool v975;
        v975 = 0 <= v973;
        bool v976;
        v976 = v975 == false;
        if (v976){
            assert("The index needs to be zero or positive." && v975);
        } else {
        }
        int v978;
        v978 = v973 % 1;
        bool v979;
        v979 = v973 < 4;
        bool v980;
        v980 = v979 == false;
        if (v980){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v979);
        } else {
        }
        assert("Tensor range check" && 0 <= v973 && v973 < 4);
        assert("Tensor range check" && 0 <= v978 && v978 < 1);
        int v982;
        v982 = 4 * v978;
        int v983;
        v983 = 4 * v973;
        int v984;
        v984 = v983 + v982;
        assert("Tensor range check" && 0 <= v973 && v973 < 4);
        assert("Tensor range check" && 0 <= v978 && v978 < 1);
        float v985[4];
        float v986[4];
        float v987[4];
        int4* v988;
        v988 = reinterpret_cast<int4*>(v4 + v984);
        int4* v989;
        v989 = reinterpret_cast<int4*>(v985 + 0);
        assert("Pointer alignment check" && (unsigned long long)(v988) % 4 == 0 && (unsigned long long)(v989) % 4 == 0);
        *v989 = *v988;
        int4* v990;
        v990 = reinterpret_cast<int4*>(v5 + v984);
        int4* v991;
        v991 = reinterpret_cast<int4*>(v986 + 0);
        assert("Pointer alignment check" && (unsigned long long)(v990) % 4 == 0 && (unsigned long long)(v991) % 4 == 0);
        *v991 = *v990;
        // Pushing the loop unrolling to: 0
        int v992;
        v992 = 0;
        #pragma unroll
        while (while_method_0(v992)){
            assert("Tensor range check" && 0 <= v992 && v992 < 4);
            float v994;
            v994 = v985[v992];
            float v995;
            v995 = v986[v992];
            bool v996;
            v996 = v995 == 0.0f;
            bool v997;
            v997 = v996 != true;
            float v999;
            if (v997){
                float v998;
                v998 = v994 / v995;
                v999 = v998;
            } else {
                v999 = 0.0f;
            }
            assert("Tensor range check" && 0 <= v992 && v992 < 4);
            v987[v992] = v999;
            v992 += 1 ;
        }
        // Poping the loop unrolling to: 0
        int4* v1000;
        v1000 = reinterpret_cast<int4*>(v987 + 0);
        int4* v1001;
        v1001 = reinterpret_cast<int4*>(v6 + v984);
        assert("Pointer alignment check" && (unsigned long long)(v1000) % 4 == 0 && (unsigned long long)(v1001) % 4 == 0);
        *v1001 = *v1000;
        v973 += 6144 ;
    }
    v968.sync() ;
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
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39,177')
options.append('--restrict')
options.append('--maxrregcount=255')
options.append('--std=c++20')
options.append('-D__CUDA_NO_HALF_CONVERSIONS__')
raw_module = cp.RawModule(code=kernel, backend='nvcc', enable_cooperative_groups=True, options=tuple(options))
import collections
class US1_0(NamedTuple): # A_All_In
    tag = 0
class US1_1(NamedTuple): # A_Call
    tag = 1
class US1_2(NamedTuple): # A_Fold
    tag = 2
class US1_3(NamedTuple): # A_Raise
    v0 : i32
    tag = 3
US1 = Union[US1_0, US1_1, US1_2, US1_3]
class US0_0(NamedTuple): # ActionSelected
    v0 : US1
    tag = 0
class US0_1(NamedTuple): # PlayerChanged
    v0 : static_array
    tag = 1
class US0_2(NamedTuple): # StartGame
    tag = 2
class US0_3(NamedTuple): # StartTrainingVsRando
    tag = 3
class US0_4(NamedTuple): # StartTrainingVsSelf
    tag = 4
US0 = Union[US0_0, US0_1, US0_2, US0_3, US0_4]
class US2_0(NamedTuple): # Computer
    tag = 0
class US2_1(NamedTuple): # Human
    tag = 1
class US2_2(NamedTuple): # Random
    tag = 2
US2 = Union[US2_0, US2_1, US2_2]
class US5_0(NamedTuple): # Flop
    v0 : static_array
    tag = 0
class US5_1(NamedTuple): # Preflop
    tag = 1
class US5_2(NamedTuple): # River
    v0 : static_array
    tag = 2
class US5_3(NamedTuple): # Turn
    v0 : static_array
    tag = 3
US5 = Union[US5_0, US5_1, US5_2, US5_3]
class US4_0(NamedTuple): # G_Flop
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US5
    tag = 0
class US4_1(NamedTuple): # G_Fold
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US5
    tag = 1
class US4_2(NamedTuple): # G_Preflop
    tag = 2
class US4_3(NamedTuple): # G_River
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US5
    tag = 3
class US4_4(NamedTuple): # G_Round
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US5
    tag = 4
class US4_5(NamedTuple): # G_Round'
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US5
    v6 : US1
    tag = 5
class US4_6(NamedTuple): # G_Showdown
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US5
    tag = 6
class US4_7(NamedTuple): # G_Turn
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US5
    tag = 7
US4 = Union[US4_0, US4_1, US4_2, US4_3, US4_4, US4_5, US4_6, US4_7]
class US3_0(NamedTuple): # None
    tag = 0
class US3_1(NamedTuple): # Some
    v0 : US4
    tag = 1
US3 = Union[US3_0, US3_1]
class US6_0(NamedTuple): # GameNotStarted
    tag = 0
class US6_1(NamedTuple): # GameOver
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US5
    tag = 1
class US6_2(NamedTuple): # WaitingForActionFromPlayerId
    v0 : i32
    v1 : static_array
    v2 : static_array
    v3 : i32
    v4 : static_array
    v5 : US5
    tag = 2
US6 = Union[US6_0, US6_1, US6_2]
class US7_0(NamedTuple): # CommunityCardsAre
    v0 : static_array_list
    tag = 0
class US7_1(NamedTuple): # Fold
    v0 : i32
    v1 : i32
    tag = 1
class US7_2(NamedTuple): # PlayerAction
    v0 : i32
    v1 : US1
    tag = 2
class US7_3(NamedTuple): # PlayerGotCards
    v0 : i32
    v1 : static_array
    tag = 3
class US7_4(NamedTuple): # Showdown
    v0 : i32
    v1 : static_array
    v2 : i32
    tag = 4
US7 = Union[US7_0, US7_1, US7_2, US7_3, US7_4]
class US8_0(NamedTuple): # AddRewardsRando
    v0 : list
    tag = 0
class US8_1(NamedTuple): # AddRewardsSelf
    v0 : list
    tag = 1
US8 = Union[US8_0, US8_1]
def Closure0():
    def inner(v0 : object, v1 : object) -> object:
        v2 = method0(v0)
        v3, v4, v5, v6, v7, v8, v9, v10, v11 = method8(v1)
        v12 = cp.empty(16,dtype=cp.uint8)
        del v12
        v13 = cp.empty(6304,dtype=cp.uint8)
        method46(v13, v3, v4, v5, v6, v7)
        del v3, v4, v5, v6, v7
        v16 = "{}\n"
        v17 = "Going to run the NL Holdem full kernel."
        print(v16.format(v17),end="")
        del v16, v17
        v18 = time.perf_counter()
        v19 = []
        match v2:
            case US0_3(): # StartTrainingVsRando
                v20 = cp.zeros(16,dtype=cp.float32) # type: ignore
                v21 = cp.zeros(16,dtype=cp.float32) # type: ignore
                v22 = cp.empty(16,dtype=cp.float32)
                v23 = cp.cuda.Device().attributes['MultiProcessorCount']
                v24 = v23 == 24
                del v23
                v25 = v24 == False
                if v25:
                    v26 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
                    assert v24, v26
                    del v26
                else:
                    pass
                del v24, v25
                v27 = 0
                v28 = raw_module.get_function(f"entry{v27}")
                del v27
                v28.max_dynamic_shared_size_bytes = 98304 
                print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
                v28((24,),(256,),(v8, v9, v10, v11, v20, v21, v22),shared_mem=98304)
                del v20, v21, v28
                v29 = []
                v31 = v22[0:]
                del v22
                v32 = v31.get()
                del v31
                v33 = 0
                while method63(v33):
                    v35 = []
                    v36 = 0
                    while method63(v36):
                        assert 0 <= v33 < 4, 'Tensor range check'
                        assert 0 <= v36 < 4, 'Tensor range check'
                        v38 = 4 * v33
                        v39 = v38 + v36
                        del v38
                        v40 = v32[v39].item()
                        del v39
                        v35.append(v40)
                        del v40
                        v36 += 1 
                    del v36
                    v29.append(v35)
                    del v35
                    v33 += 1 
                del v32, v33
                v41 = US8_0(v29)
                del v29
                v19.append(v41)
                del v41
            case t:
                raise Exception("Temporarily disabled for debugging.")
        del v2
        cp.cuda.get_current_stream().synchronize()
        v42 = time.perf_counter()
        v45 = "{}"
        v46 = "The time it took to run the kernel (in seconds) is: "
        print(v45.format(v46),end="")
        del v45, v46
        v47 = v42 - v18
        del v18, v42
        v50 = "{:.6f}\n"
        print(v50.format(v47),end="")
        del v47, v50
        v51, v52, v53, v54, v55 = method78(v13)
        del v13
        return method106(v51, v52, v53, v54, v55, v8, v9, v10, v11, v19)
    return inner
def Closure1():
    def inner() -> object:
        v1 = static_array(2)
        v3 = US2_0()
        v1[0] = v3
        del v3
        v5 = US2_1()
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
        v35 = 4503599627370495
        v36 = US3_0()
        v37 = US6_0()
        v38 = 204570624
        v39 = 12419088
        return method155(v35, v36, v7, v1, v37, v9, v38, v8, v39)
    return inner
def method3(v0 : object) -> None:
    assert v0 == [], f'Expected an unit type. Got: {v0}'
    del v0
    return 
def method4(v0 : object) -> i32:
    assert isinstance(v0,i32), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method2(v0 : object) -> US1:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "A_All_In" == v1
    if v3:
        del v1, v3
        method3(v2)
        del v2
        return US1_0()
    else:
        del v3
        v5 = "A_Call" == v1
        if v5:
            del v1, v5
            method3(v2)
            del v2
            return US1_1()
        else:
            del v5
            v7 = "A_Fold" == v1
            if v7:
                del v1, v7
                method3(v2)
                del v2
                return US1_2()
            else:
                del v7
                v9 = "A_Raise" == v1
                if v9:
                    del v1, v9
                    v10 = method4(v2)
                    del v2
                    return US1_3(v10)
                else:
                    del v2, v9
                    raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                    del v1
                    raise Exception("Error")
def method6(v0 : i32, v1 : i32) -> bool:
    v2 = v1 < v0
    del v0, v1
    return v2
def method7(v0 : object) -> US2:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "Computer" == v1
    if v3:
        del v1, v3
        method3(v2)
        del v2
        return US2_0()
    else:
        del v3
        v5 = "Human" == v1
        if v5:
            del v1, v5
            method3(v2)
            del v2
            return US2_1()
        else:
            del v5
            v7 = "Random" == v1
            if v7:
                del v1, v7
                method3(v2)
                del v2
                return US2_2()
            else:
                del v2, v7
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method5(v0 : object) -> static_array:
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
    v6 = static_array(2)
    v7 = 0
    while method6(v1, v7):
        v9 = v0[v7]
        v10 = method7(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
def method1(v0 : object) -> US0:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "ActionSelected" == v1
    if v3:
        del v1, v3
        v4 = method2(v2)
        del v2
        return US0_0(v4)
    else:
        del v3
        v6 = "PlayerChanged" == v1
        if v6:
            del v1, v6
            v7 = method5(v2)
            del v2
            return US0_1(v7)
        else:
            del v6
            v9 = "StartGame" == v1
            if v9:
                del v1, v9
                method3(v2)
                del v2
                return US0_2()
            else:
                del v9
                v11 = "StartTrainingVsRando" == v1
                if v11:
                    del v1, v11
                    method3(v2)
                    del v2
                    return US0_3()
                else:
                    del v11
                    v13 = "StartTrainingVsSelf" == v1
                    if v13:
                        del v1, v13
                        method3(v2)
                        del v2
                        return US0_4()
                    else:
                        del v2, v13
                        raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                        del v1
                        raise Exception("Error")
def method0(v0 : object) -> US0:
    return method1(v0)
def method13(v0 : object) -> u64:
    assert isinstance(v0,u64), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method12(v0 : object) -> u64:
    v1 = method13(v0)
    del v0
    return v1
def method20(v0 : object) -> u8:
    assert isinstance(v0,u8), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method19(v0 : object) -> u8:
    v1 = method20(v0)
    del v0
    return v1
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
    v6 = static_array(2)
    v7 = 0
    while method6(v1, v7):
        v9 = v0[v7]
        v10 = method19(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
def method17(v0 : object) -> static_array:
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
    v6 = static_array(2)
    v7 = 0
    while method6(v1, v7):
        v9 = v0[v7]
        v10 = method18(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
def method21(v0 : object) -> static_array:
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
    v6 = static_array(2)
    v7 = 0
    while method6(v1, v7):
        v9 = v0[v7]
        v10 = method4(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
def method23(v0 : object) -> static_array:
    assert isinstance(v0,list), f'The object needs to be a Python list. Got: {v0}'
    v1 = len(v0) # type: ignore
    v2 = 3 == v1
    v3 = v2 == False
    if v3:
        v4 = "The type level dimension has to equal the value passed at runtime into create."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v6 = static_array(3)
    v7 = 0
    while method6(v1, v7):
        v9 = v0[v7]
        v10 = method19(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
def method24(v0 : object) -> static_array:
    assert isinstance(v0,list), f'The object needs to be a Python list. Got: {v0}'
    v1 = len(v0) # type: ignore
    v2 = 5 == v1
    v3 = v2 == False
    if v3:
        v4 = "The type level dimension has to equal the value passed at runtime into create."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v6 = static_array(5)
    v7 = 0
    while method6(v1, v7):
        v9 = v0[v7]
        v10 = method19(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
def method25(v0 : object) -> static_array:
    assert isinstance(v0,list), f'The object needs to be a Python list. Got: {v0}'
    v1 = len(v0) # type: ignore
    v2 = 4 == v1
    v3 = v2 == False
    if v3:
        v4 = "The type level dimension has to equal the value passed at runtime into create."
        assert v2, v4
        del v4
    else:
        pass
    del v2, v3
    v6 = static_array(4)
    v7 = 0
    while method6(v1, v7):
        v9 = v0[v7]
        v10 = method19(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
def method22(v0 : object) -> US5:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "Flop" == v1
    if v3:
        del v1, v3
        v4 = method23(v2)
        del v2
        return US5_0(v4)
    else:
        del v3
        v6 = "Preflop" == v1
        if v6:
            del v1, v6
            method3(v2)
            del v2
            return US5_1()
        else:
            del v6
            v8 = "River" == v1
            if v8:
                del v1, v8
                v9 = method24(v2)
                del v2
                return US5_2(v9)
            else:
                del v8
                v11 = "Turn" == v1
                if v11:
                    del v1, v11
                    v12 = method25(v2)
                    del v2
                    return US5_3(v12)
                else:
                    del v2, v11
                    raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                    del v1
                    raise Exception("Error")
def method16(v0 : object) -> Tuple[i32, static_array, static_array, i32, static_array, US5]:
    v1 = v0["min_raise"] # type: ignore
    v2 = method4(v1)
    del v1
    v3 = v0["pl_card"] # type: ignore
    v4 = method17(v3)
    del v3
    v5 = v0["pot"] # type: ignore
    v6 = method21(v5)
    del v5
    v7 = v0["round_turn"] # type: ignore
    v8 = method4(v7)
    del v7
    v9 = v0["stack"] # type: ignore
    v10 = method21(v9)
    del v9
    v11 = v0["street"] # type: ignore
    del v0
    v12 = method22(v11)
    del v11
    return v2, v4, v6, v8, v10, v12
def method26(v0 : object) -> Tuple[i32, static_array, static_array, i32, static_array, US5, US1]:
    v1 = v0[0] # type: ignore
    v2, v3, v4, v5, v6, v7 = method16(v1)
    del v1
    v8 = v0[1] # type: ignore
    del v0
    v9 = method2(v8)
    del v8
    return v2, v3, v4, v5, v6, v7, v9
def method15(v0 : object) -> US4:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "G_Flop" == v1
    if v3:
        del v1, v3
        v4, v5, v6, v7, v8, v9 = method16(v2)
        del v2
        return US4_0(v4, v5, v6, v7, v8, v9)
    else:
        del v3
        v11 = "G_Fold" == v1
        if v11:
            del v1, v11
            v12, v13, v14, v15, v16, v17 = method16(v2)
            del v2
            return US4_1(v12, v13, v14, v15, v16, v17)
        else:
            del v11
            v19 = "G_Preflop" == v1
            if v19:
                del v1, v19
                method3(v2)
                del v2
                return US4_2()
            else:
                del v19
                v21 = "G_River" == v1
                if v21:
                    del v1, v21
                    v22, v23, v24, v25, v26, v27 = method16(v2)
                    del v2
                    return US4_3(v22, v23, v24, v25, v26, v27)
                else:
                    del v21
                    v29 = "G_Round" == v1
                    if v29:
                        del v1, v29
                        v30, v31, v32, v33, v34, v35 = method16(v2)
                        del v2
                        return US4_4(v30, v31, v32, v33, v34, v35)
                    else:
                        del v29
                        v37 = "G_Round'" == v1
                        if v37:
                            del v1, v37
                            v38, v39, v40, v41, v42, v43, v44 = method26(v2)
                            del v2
                            return US4_5(v38, v39, v40, v41, v42, v43, v44)
                        else:
                            del v37
                            v46 = "G_Showdown" == v1
                            if v46:
                                del v1, v46
                                v47, v48, v49, v50, v51, v52 = method16(v2)
                                del v2
                                return US4_6(v47, v48, v49, v50, v51, v52)
                            else:
                                del v46
                                v54 = "G_Turn" == v1
                                if v54:
                                    del v1, v54
                                    v55, v56, v57, v58, v59, v60 = method16(v2)
                                    del v2
                                    return US4_7(v55, v56, v57, v58, v59, v60)
                                else:
                                    del v2, v54
                                    raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                                    del v1
                                    raise Exception("Error")
def method14(v0 : object) -> US3:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "None" == v1
    if v3:
        del v1, v3
        method3(v2)
        del v2
        return US3_0()
    else:
        del v3
        v5 = "Some" == v1
        if v5:
            del v1, v5
            v6 = method15(v2)
            del v2
            return US3_1(v6)
        else:
            del v2, v5
            raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
            del v1
            raise Exception("Error")
def method11(v0 : object) -> Tuple[u64, US3]:
    v1 = v0["deck"] # type: ignore
    v2 = method12(v1)
    del v1
    v3 = v0["game"] # type: ignore
    del v0
    v4 = method14(v3)
    del v3
    return v2, v4
def method30(v0 : object) -> static_array_list:
    v1 = len(v0) # type: ignore
    assert (5 >= v1), f'The length of the original object has to be greater than or equal to the static array dimension.\nExpected: 5\nGot: {v1} '
    del v1
    assert isinstance(v0,list), f'The object needs to be a Python list. Got: {v0}'
    v2 = len(v0) # type: ignore
    v3 = 5 >= v2
    v4 = v3 == False
    if v4:
        v5 = "The type level dimension has to equal the value passed at runtime into create."
        assert v3, v5
        del v5
    else:
        pass
    del v3, v4
    v7 = static_array_list(5)
    v7.unsafe_set_length(v2)
    v8 = 0
    while method6(v2, v8):
        v10 = v0[v8]
        v11 = method19(v10)
        del v10
        v7[v8] = v11
        del v11
        v8 += 1 
    del v0, v2, v8
    return v7
def method31(v0 : object) -> Tuple[i32, i32]:
    v1 = v0["chips_won"] # type: ignore
    v2 = method4(v1)
    del v1
    v3 = v0["winner_id"] # type: ignore
    del v0
    v4 = method4(v3)
    del v3
    return v2, v4
def method32(v0 : object) -> Tuple[i32, US1]:
    v1 = v0[0] # type: ignore
    v2 = method4(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method2(v3)
    del v3
    return v2, v4
def method33(v0 : object) -> Tuple[i32, static_array]:
    v1 = v0[0] # type: ignore
    v2 = method4(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method18(v3)
    del v3
    return v2, v4
def method38(v0 : object) -> i8:
    assert isinstance(v0,i8), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method37(v0 : object) -> Tuple[static_array, i8]:
    v1 = v0["hand"] # type: ignore
    v2 = method24(v1)
    del v1
    v3 = v0["score"] # type: ignore
    del v0
    v4 = method38(v3)
    del v3
    return v2, v4
def method36(v0 : object) -> Tuple[static_array, i8]:
    v1, v2 = method37(v0)
    del v0
    return v1, v2
def method35(v0 : object) -> static_array:
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
    v6 = static_array(2)
    v7 = 0
    while method6(v1, v7):
        v9 = v0[v7]
        v10, v11 = method36(v9)
        del v9
        v6[v7] = (v10, v11)
        del v10, v11
        v7 += 1 
    del v0, v1, v7
    return v6
def method34(v0 : object) -> Tuple[i32, static_array, i32]:
    v1 = v0["chips_won"] # type: ignore
    v2 = method4(v1)
    del v1
    v3 = v0["hands_shown"] # type: ignore
    v4 = method35(v3)
    del v3
    v5 = v0["winner_id"] # type: ignore
    del v0
    v6 = method4(v5)
    del v5
    return v2, v4, v6
def method29(v0 : object) -> US7:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "CommunityCardsAre" == v1
    if v3:
        del v1, v3
        v4 = method30(v2)
        del v2
        return US7_0(v4)
    else:
        del v3
        v6 = "Fold" == v1
        if v6:
            del v1, v6
            v7, v8 = method31(v2)
            del v2
            return US7_1(v7, v8)
        else:
            del v6
            v10 = "PlayerAction" == v1
            if v10:
                del v1, v10
                v11, v12 = method32(v2)
                del v2
                return US7_2(v11, v12)
            else:
                del v10
                v14 = "PlayerGotCards" == v1
                if v14:
                    del v1, v14
                    v15, v16 = method33(v2)
                    del v2
                    return US7_3(v15, v16)
                else:
                    del v14
                    v18 = "Showdown" == v1
                    if v18:
                        del v1, v18
                        v19, v20, v21 = method34(v2)
                        del v2
                        return US7_4(v19, v20, v21)
                    else:
                        del v2, v18
                        raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                        del v1
                        raise Exception("Error")
def method28(v0 : object) -> static_array_list:
    v1 = len(v0) # type: ignore
    assert (128 >= v1), f'The length of the original object has to be greater than or equal to the static array dimension.\nExpected: 128\nGot: {v1} '
    del v1
    assert isinstance(v0,list), f'The object needs to be a Python list. Got: {v0}'
    v2 = len(v0) # type: ignore
    v3 = 128 >= v2
    v4 = v3 == False
    if v4:
        v5 = "The type level dimension has to equal the value passed at runtime into create."
        assert v3, v5
        del v5
    else:
        pass
    del v3, v4
    v7 = static_array_list(128)
    v7.unsafe_set_length(v2)
    v8 = 0
    while method6(v2, v8):
        v10 = v0[v8]
        v11 = method29(v10)
        del v10
        v7[v8] = v11
        del v11
        v8 += 1 
    del v0, v2, v8
    return v7
def method39(v0 : object) -> US6:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "GameNotStarted" == v1
    if v3:
        del v1, v3
        method3(v2)
        del v2
        return US6_0()
    else:
        del v3
        v5 = "GameOver" == v1
        if v5:
            del v1, v5
            v6, v7, v8, v9, v10, v11 = method16(v2)
            del v2
            return US6_1(v6, v7, v8, v9, v10, v11)
        else:
            del v5
            v13 = "WaitingForActionFromPlayerId" == v1
            if v13:
                del v1, v13
                v14, v15, v16, v17, v18, v19 = method16(v2)
                del v2
                return US6_2(v14, v15, v16, v17, v18, v19)
            else:
                del v2, v13
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method27(v0 : object) -> Tuple[static_array_list, static_array, US6]:
    v1 = v0["messages"] # type: ignore
    v2 = method28(v1)
    del v1
    v3 = v0["pl_type"] # type: ignore
    v4 = method5(v3)
    del v3
    v5 = v0["ui_game_state"] # type: ignore
    del v0
    v6 = method39(v5)
    del v5
    return v2, v4, v6
def method10(v0 : object) -> Tuple[u64, US3, static_array_list, static_array, US6]:
    v1 = v0["private"] # type: ignore
    v2, v3 = method11(v1)
    del v1
    v4 = v0["public"] # type: ignore
    del v0
    v5, v6, v7 = method27(v4)
    del v4
    return v2, v3, v5, v6, v7
def method45(v0 : object) -> cp.ndarray:
    assert isinstance(v0,cp.ndarray), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method44(v0 : object) -> cp.ndarray:
    v1 = method45(v0)
    del v0
    return v1
def method43(v0 : object) -> Tuple[cp.ndarray, u64]:
    v1 = v0[0] # type: ignore
    v2 = method44(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method13(v3)
    del v3
    return v2, v4
def method42(v0 : object) -> Tuple[cp.ndarray, u64, cp.ndarray, u64]:
    v1 = v0["output"] # type: ignore
    v2, v3 = method43(v1)
    del v1
    v4 = v0["param"] # type: ignore
    del v0
    v5, v6 = method43(v4)
    del v4
    return v2, v3, v5, v6
def method41(v0 : object) -> Tuple[cp.ndarray, u64, cp.ndarray, u64]:
    v1, v2, v3, v4 = method42(v0)
    del v0
    return v1, v2, v3, v4
def method40(v0 : object) -> Tuple[cp.ndarray, u64, cp.ndarray, u64]:
    v1 = v0["model_data"] # type: ignore
    del v0
    v2, v3, v4, v5 = method41(v1)
    del v1
    return v2, v3, v4, v5
def method9(v0 : object) -> Tuple[u64, US3, static_array_list, static_array, US6, cp.ndarray, u64, cp.ndarray, u64]:
    v1 = v0["game"] # type: ignore
    v2, v3, v4, v5, v6 = method10(v1)
    del v1
    v7 = v0["neural"] # type: ignore
    del v0
    v8, v9, v10, v11 = method40(v7)
    del v7
    return v2, v3, v4, v5, v6, v8, v9, v10, v11
def method8(v0 : object) -> Tuple[u64, US3, static_array_list, static_array, US6, cp.ndarray, u64, cp.ndarray, u64]:
    return method9(v0)
def method47(v0 : cp.ndarray, v1 : u64) -> None:
    v3 = v0[0:].view(cp.uint64)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method48(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[8:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method49(v0 : cp.ndarray) -> None:
    del v0
    return 
def method51(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[0:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method53(v0 : i32) -> bool:
    v1 = v0 < 2
    del v0
    return v1
def method56(v0 : cp.ndarray, v1 : u8) -> None:
    v3 = v0[0:].view(cp.uint8)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method55(v0 : cp.ndarray, v1 : u8) -> None:
    return method56(v0, v1)
def method54(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method53(v2):
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
        method55(v6, v13)
        del v6, v13
        v2 += 1 
    del v0, v1, v2
    return 
def method57(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[28:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method59(v0 : i32) -> bool:
    v1 = v0 < 3
    del v0
    return v1
def method58(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method59(v2):
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
        method55(v6, v13)
        del v6, v13
        v2 += 1 
    del v0, v1, v2
    return 
def method61(v0 : i32) -> bool:
    v1 = v0 < 5
    del v0
    return v1
def method60(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method61(v2):
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
        method55(v6, v13)
        del v6, v13
        v2 += 1 
    del v0, v1, v2
    return 
def method63(v0 : i32) -> bool:
    v1 = v0 < 4
    del v0
    return v1
def method62(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method63(v2):
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
        method55(v6, v13)
        del v6, v13
        v2 += 1 
    del v0, v1, v2
    return 
def method52(v0 : cp.ndarray, v1 : i32, v2 : static_array, v3 : static_array, v4 : i32, v5 : static_array, v6 : US5) -> None:
    v8 = v0[0:].view(cp.int32)
    v8[0] = v1
    del v1, v8
    v9 = 0
    while method53(v9):
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
        method54(v15, v22)
        del v15, v22
        v9 += 1 
    del v2, v9
    v23 = 0
    while method53(v23):
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
        method51(v29, v36)
        del v29, v36
        v23 += 1 
    del v3, v23
    v38 = v0[16:].view(cp.int32)
    v38[0] = v4
    del v4, v38
    v39 = 0
    while method53(v39):
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
        method51(v45, v52)
        del v45, v52
        v39 += 1 
    del v5, v39
    v53 = v6.tag
    method57(v0, v53)
    del v53
    v55 = v0[32:].view(cp.uint8)
    del v0
    match v6:
        case US5_0(v56): # Flop
            del v6
            return method58(v55, v56)
        case US5_1(): # Preflop
            del v6
            return method49(v55)
        case US5_2(v57): # River
            del v6
            return method60(v55, v57)
        case US5_3(v58): # Turn
            del v6
            return method62(v55, v58)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method65(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[40:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method64(v0 : cp.ndarray, v1 : i32, v2 : static_array, v3 : static_array, v4 : i32, v5 : static_array, v6 : US5, v7 : US1) -> None:
    v9 = v0[0:].view(cp.int32)
    v9[0] = v1
    del v1, v9
    v10 = 0
    while method53(v10):
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
        method54(v16, v23)
        del v16, v23
        v10 += 1 
    del v2, v10
    v24 = 0
    while method53(v24):
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
        method51(v30, v37)
        del v30, v37
        v24 += 1 
    del v3, v24
    v39 = v0[16:].view(cp.int32)
    v39[0] = v4
    del v4, v39
    v40 = 0
    while method53(v40):
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
        method51(v46, v53)
        del v46, v53
        v40 += 1 
    del v5, v40
    v54 = v6.tag
    method57(v0, v54)
    del v54
    v56 = v0[32:].view(cp.uint8)
    match v6:
        case US5_0(v57): # Flop
            method58(v56, v57)
        case US5_1(): # Preflop
            method49(v56)
        case US5_2(v58): # River
            method60(v56, v58)
        case US5_3(v59): # Turn
            method62(v56, v59)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v6, v56
    v60 = v7.tag
    method65(v0, v60)
    del v60
    v62 = v0[44:].view(cp.uint8)
    del v0
    match v7:
        case US1_0(): # A_All_In
            del v7
            return method49(v62)
        case US1_1(): # A_Call
            del v7
            return method49(v62)
        case US1_2(): # A_Fold
            del v7
            return method49(v62)
        case US1_3(v63): # A_Raise
            del v7
            return method51(v62, v63)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method50(v0 : cp.ndarray, v1 : US4) -> None:
    v2 = v1.tag
    method51(v0, v2)
    del v2
    v4 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US4_0(v5, v6, v7, v8, v9, v10): # G_Flop
            del v1
            return method52(v4, v5, v6, v7, v8, v9, v10)
        case US4_1(v11, v12, v13, v14, v15, v16): # G_Fold
            del v1
            return method52(v4, v11, v12, v13, v14, v15, v16)
        case US4_2(): # G_Preflop
            del v1
            return method49(v4)
        case US4_3(v17, v18, v19, v20, v21, v22): # G_River
            del v1
            return method52(v4, v17, v18, v19, v20, v21, v22)
        case US4_4(v23, v24, v25, v26, v27, v28): # G_Round
            del v1
            return method52(v4, v23, v24, v25, v26, v27, v28)
        case US4_5(v29, v30, v31, v32, v33, v34, v35): # G_Round'
            del v1
            return method64(v4, v29, v30, v31, v32, v33, v34, v35)
        case US4_6(v36, v37, v38, v39, v40, v41): # G_Showdown
            del v1
            return method52(v4, v36, v37, v38, v39, v40, v41)
        case US4_7(v42, v43, v44, v45, v46, v47): # G_Turn
            del v1
            return method52(v4, v42, v43, v44, v45, v46, v47)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method66(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[80:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method68(v0 : cp.ndarray, v1 : static_array_list) -> None:
    v2 = v1.length
    method51(v0, v2)
    del v2
    v3 = v1.length
    v4 = 0
    while method6(v3, v4):
        v6 = u64(v4)
        v7 = 4 + v6
        del v6
        v9 = v0[v7:].view(cp.uint8)
        del v7
        v11 = v1[v4]
        method55(v9, v11)
        del v9, v11
        v4 += 1 
    del v0, v1, v3, v4
    return 
def method69(v0 : cp.ndarray, v1 : i32, v2 : i32) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v6 = v0[4:].view(cp.int32)
    del v0
    v6[0] = v2
    del v2, v6
    return 
def method71(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[4:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method70(v0 : cp.ndarray, v1 : i32, v2 : US1) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v5 = v2.tag
    method71(v0, v5)
    del v5
    v7 = v0[8:].view(cp.uint8)
    del v0
    match v2:
        case US1_0(): # A_All_In
            del v2
            return method49(v7)
        case US1_1(): # A_Call
            del v2
            return method49(v7)
        case US1_2(): # A_Fold
            del v2
            return method49(v7)
        case US1_3(v8): # A_Raise
            del v2
            return method51(v7, v8)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method72(v0 : cp.ndarray, v1 : i32, v2 : static_array) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v5 = 0
    while method53(v5):
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
        method55(v10, v17)
        del v10, v17
        v5 += 1 
    del v0, v2, v5
    return 
def method75(v0 : cp.ndarray, v1 : static_array, v2 : i8) -> None:
    v3 = 0
    while method61(v3):
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
        method55(v7, v14)
        del v7, v14
        v3 += 1 
    del v1, v3
    v16 = v0[5:].view(cp.int8)
    del v0
    v16[0] = v2
    del v2, v16
    return 
def method74(v0 : cp.ndarray, v1 : static_array, v2 : i8) -> None:
    return method75(v0, v1, v2)
def method73(v0 : cp.ndarray, v1 : i32, v2 : static_array, v3 : i32) -> None:
    v5 = v0[0:].view(cp.int32)
    v5[0] = v1
    del v1, v5
    v6 = 0
    while method53(v6):
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
        method74(v12, v20, v21)
        del v12, v20, v21
        v6 += 1 
    del v2, v6
    v23 = v0[24:].view(cp.int32)
    del v0
    v23[0] = v3
    del v3, v23
    return 
def method67(v0 : cp.ndarray, v1 : US7) -> None:
    v2 = v1.tag
    method51(v0, v2)
    del v2
    v4 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US7_0(v5): # CommunityCardsAre
            del v1
            return method68(v4, v5)
        case US7_1(v6, v7): # Fold
            del v1
            return method69(v4, v6, v7)
        case US7_2(v8, v9): # PlayerAction
            del v1
            return method70(v4, v8, v9)
        case US7_3(v10, v11): # PlayerGotCards
            del v1
            return method72(v4, v10, v11)
        case US7_4(v12, v13, v14): # Showdown
            del v1
            return method73(v4, v12, v13, v14)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method76(v0 : cp.ndarray, v1 : US2) -> None:
    v2 = v1.tag
    method51(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US2_0(): # Computer
            del v1
            return method49(v4)
        case US2_1(): # Human
            del v1
            return method49(v4)
        case US2_2(): # Random
            del v1
            return method49(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method77(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[6248:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method46(v0 : cp.ndarray, v1 : u64, v2 : US3, v3 : static_array_list, v4 : static_array, v5 : US6) -> None:
    method47(v0, v1)
    del v1
    v6 = v2.tag
    method48(v0, v6)
    del v6
    v8 = v0[16:].view(cp.uint8)
    match v2:
        case US3_0(): # None
            method49(v8)
        case US3_1(v9): # Some
            method50(v8, v9)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v2, v8
    v10 = v3.length
    method66(v0, v10)
    del v10
    v11 = v3.length
    v12 = 0
    while method6(v11, v12):
        v14 = u64(v12)
        v15 = v14 * 48
        del v14
        v16 = 96 + v15
        del v15
        v18 = v0[v16:].view(cp.uint8)
        del v16
        v20 = v3[v12]
        method67(v18, v20)
        del v18, v20
        v12 += 1 
    del v3, v11, v12
    v21 = 0
    while method53(v21):
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
        method76(v27, v34)
        del v27, v34
        v21 += 1 
    del v4, v21
    v35 = v5.tag
    method77(v0, v35)
    del v35
    v37 = v0[6256:].view(cp.uint8)
    del v0
    match v5:
        case US6_0(): # GameNotStarted
            del v5
            return method49(v37)
        case US6_1(v38, v39, v40, v41, v42, v43): # GameOver
            del v5
            return method52(v37, v38, v39, v40, v41, v42, v43)
        case US6_2(v44, v45, v46, v47, v48, v49): # WaitingForActionFromPlayerId
            del v5
            return method52(v37, v44, v45, v46, v47, v48, v49)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method79(v0 : cp.ndarray) -> u64:
    v2 = v0[0:].view(cp.uint64)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method80(v0 : cp.ndarray) -> i32:
    v2 = v0[8:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method81(v0 : cp.ndarray) -> None:
    del v0
    return 
def method83(v0 : cp.ndarray) -> i32:
    v2 = v0[0:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method87(v0 : cp.ndarray) -> u8:
    v2 = v0[0:].view(cp.uint8)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method86(v0 : cp.ndarray) -> u8:
    v1 = method87(v0)
    del v0
    return v1
def method85(v0 : cp.ndarray) -> static_array:
    v2 = static_array(2)
    v3 = 0
    while method53(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v8 = method86(v7)
        del v7
        v2[v3] = v8
        del v8
        v3 += 1 
    del v0, v3
    return v2
def method88(v0 : cp.ndarray) -> i32:
    v2 = v0[28:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method89(v0 : cp.ndarray) -> static_array:
    v2 = static_array(3)
    v3 = 0
    while method59(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v8 = method86(v7)
        del v7
        v2[v3] = v8
        del v8
        v3 += 1 
    del v0, v3
    return v2
def method90(v0 : cp.ndarray) -> static_array:
    v2 = static_array(5)
    v3 = 0
    while method61(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v8 = method86(v7)
        del v7
        v2[v3] = v8
        del v8
        v3 += 1 
    del v0, v3
    return v2
def method91(v0 : cp.ndarray) -> static_array:
    v2 = static_array(4)
    v3 = 0
    while method63(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v8 = method86(v7)
        del v7
        v2[v3] = v8
        del v8
        v3 += 1 
    del v0, v3
    return v2
def method84(v0 : cp.ndarray) -> Tuple[i32, static_array, static_array, i32, static_array, US5]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v5 = static_array(2)
    v6 = 0
    while method53(v6):
        v8 = u64(v6)
        v9 = v8 * 2
        del v8
        v10 = 4 + v9
        del v9
        v12 = v0[v10:].view(cp.uint8)
        del v10
        v13 = method85(v12)
        del v12
        v5[v6] = v13
        del v13
        v6 += 1 
    del v6
    v15 = static_array(2)
    v16 = 0
    while method53(v16):
        v18 = u64(v16)
        v19 = v18 * 4
        del v18
        v20 = 8 + v19
        del v19
        v22 = v0[v20:].view(cp.uint8)
        del v20
        v23 = method83(v22)
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
    while method53(v29):
        v31 = u64(v29)
        v32 = v31 * 4
        del v31
        v33 = 20 + v32
        del v32
        v35 = v0[v33:].view(cp.uint8)
        del v33
        v36 = method83(v35)
        del v35
        v28[v29] = v36
        del v36
        v29 += 1 
    del v29
    v37 = method88(v0)
    v39 = v0[32:].view(cp.uint8)
    del v0
    if v37 == 0:
        v41 = method89(v39)
        v48 = US5_0(v41)
    elif v37 == 1:
        method81(v39)
        v48 = US5_1()
    elif v37 == 2:
        v44 = method90(v39)
        v48 = US5_2(v44)
    elif v37 == 3:
        v46 = method91(v39)
        v48 = US5_3(v46)
    else:
        raise Exception("Invalid tag.")
    del v37, v39
    return v3, v5, v15, v26, v28, v48
def method93(v0 : cp.ndarray) -> i32:
    v2 = v0[40:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method92(v0 : cp.ndarray) -> Tuple[i32, static_array, static_array, i32, static_array, US5, US1]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v5 = static_array(2)
    v6 = 0
    while method53(v6):
        v8 = u64(v6)
        v9 = v8 * 2
        del v8
        v10 = 4 + v9
        del v9
        v12 = v0[v10:].view(cp.uint8)
        del v10
        v13 = method85(v12)
        del v12
        v5[v6] = v13
        del v13
        v6 += 1 
    del v6
    v15 = static_array(2)
    v16 = 0
    while method53(v16):
        v18 = u64(v16)
        v19 = v18 * 4
        del v18
        v20 = 8 + v19
        del v19
        v22 = v0[v20:].view(cp.uint8)
        del v20
        v23 = method83(v22)
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
    while method53(v29):
        v31 = u64(v29)
        v32 = v31 * 4
        del v31
        v33 = 20 + v32
        del v32
        v35 = v0[v33:].view(cp.uint8)
        del v33
        v36 = method83(v35)
        del v35
        v28[v29] = v36
        del v36
        v29 += 1 
    del v29
    v37 = method88(v0)
    v39 = v0[32:].view(cp.uint8)
    if v37 == 0:
        v41 = method89(v39)
        v48 = US5_0(v41)
    elif v37 == 1:
        method81(v39)
        v48 = US5_1()
    elif v37 == 2:
        v44 = method90(v39)
        v48 = US5_2(v44)
    elif v37 == 3:
        v46 = method91(v39)
        v48 = US5_3(v46)
    else:
        raise Exception("Invalid tag.")
    del v37, v39
    v49 = method93(v0)
    v51 = v0[44:].view(cp.uint8)
    del v0
    if v49 == 0:
        method81(v51)
        v58 = US1_0()
    elif v49 == 1:
        method81(v51)
        v58 = US1_1()
    elif v49 == 2:
        method81(v51)
        v58 = US1_2()
    elif v49 == 3:
        v56 = method83(v51)
        v58 = US1_3(v56)
    else:
        raise Exception("Invalid tag.")
    del v49, v51
    return v3, v5, v15, v26, v28, v48, v58
def method82(v0 : cp.ndarray) -> US4:
    v1 = method83(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5, v6, v7, v8, v9, v10 = method84(v3)
        del v3
        return US4_0(v5, v6, v7, v8, v9, v10)
    elif v1 == 1:
        del v1
        v12, v13, v14, v15, v16, v17 = method84(v3)
        del v3
        return US4_1(v12, v13, v14, v15, v16, v17)
    elif v1 == 2:
        del v1
        method81(v3)
        del v3
        return US4_2()
    elif v1 == 3:
        del v1
        v20, v21, v22, v23, v24, v25 = method84(v3)
        del v3
        return US4_3(v20, v21, v22, v23, v24, v25)
    elif v1 == 4:
        del v1
        v27, v28, v29, v30, v31, v32 = method84(v3)
        del v3
        return US4_4(v27, v28, v29, v30, v31, v32)
    elif v1 == 5:
        del v1
        v34, v35, v36, v37, v38, v39, v40 = method92(v3)
        del v3
        return US4_5(v34, v35, v36, v37, v38, v39, v40)
    elif v1 == 6:
        del v1
        v42, v43, v44, v45, v46, v47 = method84(v3)
        del v3
        return US4_6(v42, v43, v44, v45, v46, v47)
    elif v1 == 7:
        del v1
        v49, v50, v51, v52, v53, v54 = method84(v3)
        del v3
        return US4_7(v49, v50, v51, v52, v53, v54)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method94(v0 : cp.ndarray) -> i32:
    v2 = v0[80:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method96(v0 : cp.ndarray) -> static_array_list:
    v2 = static_array_list(5)
    v3 = method83(v0)
    v2.unsafe_set_length(v3)
    del v3
    v4 = v2.length
    v5 = 0
    while method6(v4, v5):
        v7 = u64(v5)
        v8 = 4 + v7
        del v7
        v10 = v0[v8:].view(cp.uint8)
        del v8
        v11 = method86(v10)
        del v10
        v2[v5] = v11
        del v11
        v5 += 1 
    del v0, v4, v5
    return v2
def method97(v0 : cp.ndarray) -> Tuple[i32, i32]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v5 = v0[4:].view(cp.int32)
    del v0
    v6 = v5[0].item()
    del v5
    return v3, v6
def method99(v0 : cp.ndarray) -> i32:
    v2 = v0[4:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method98(v0 : cp.ndarray) -> Tuple[i32, US1]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v4 = method99(v0)
    v6 = v0[8:].view(cp.uint8)
    del v0
    if v4 == 0:
        method81(v6)
        v13 = US1_0()
    elif v4 == 1:
        method81(v6)
        v13 = US1_1()
    elif v4 == 2:
        method81(v6)
        v13 = US1_2()
    elif v4 == 3:
        v11 = method83(v6)
        v13 = US1_3(v11)
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    return v3, v13
def method100(v0 : cp.ndarray) -> Tuple[i32, static_array]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v5 = static_array(2)
    v6 = 0
    while method53(v6):
        v8 = u64(v6)
        v9 = 4 + v8
        del v8
        v11 = v0[v9:].view(cp.uint8)
        del v9
        v12 = method86(v11)
        del v11
        v5[v6] = v12
        del v12
        v6 += 1 
    del v0, v6
    return v3, v5
def method103(v0 : cp.ndarray) -> Tuple[static_array, i8]:
    v2 = static_array(5)
    v3 = 0
    while method61(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v8 = method86(v7)
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
def method102(v0 : cp.ndarray) -> Tuple[static_array, i8]:
    v1, v2 = method103(v0)
    del v0
    return v1, v2
def method101(v0 : cp.ndarray) -> Tuple[i32, static_array, i32]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v5 = static_array(2)
    v6 = 0
    while method53(v6):
        v8 = u64(v6)
        v9 = v8 * 8
        del v8
        v10 = 8 + v9
        del v9
        v12 = v0[v10:].view(cp.uint8)
        del v10
        v13, v14 = method102(v12)
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
def method95(v0 : cp.ndarray) -> US7:
    v1 = method83(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5 = method96(v3)
        del v3
        return US7_0(v5)
    elif v1 == 1:
        del v1
        v7, v8 = method97(v3)
        del v3
        return US7_1(v7, v8)
    elif v1 == 2:
        del v1
        v10, v11 = method98(v3)
        del v3
        return US7_2(v10, v11)
    elif v1 == 3:
        del v1
        v13, v14 = method100(v3)
        del v3
        return US7_3(v13, v14)
    elif v1 == 4:
        del v1
        v16, v17, v18 = method101(v3)
        del v3
        return US7_4(v16, v17, v18)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method104(v0 : cp.ndarray) -> US2:
    v1 = method83(v0)
    v3 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method81(v3)
        del v3
        return US2_0()
    elif v1 == 1:
        del v1
        method81(v3)
        del v3
        return US2_1()
    elif v1 == 2:
        del v1
        method81(v3)
        del v3
        return US2_2()
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method105(v0 : cp.ndarray) -> i32:
    v2 = v0[6248:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method78(v0 : cp.ndarray) -> Tuple[u64, US3, static_array_list, static_array, US6]:
    v1 = method79(v0)
    v2 = method80(v0)
    v4 = v0[16:].view(cp.uint8)
    if v2 == 0:
        method81(v4)
        v9 = US3_0()
    elif v2 == 1:
        v7 = method82(v4)
        v9 = US3_1(v7)
    else:
        raise Exception("Invalid tag.")
    del v2, v4
    v11 = static_array_list(128)
    v12 = method94(v0)
    v11.unsafe_set_length(v12)
    del v12
    v13 = v11.length
    v14 = 0
    while method6(v13, v14):
        v16 = u64(v14)
        v17 = v16 * 48
        del v16
        v18 = 96 + v17
        del v17
        v20 = v0[v18:].view(cp.uint8)
        del v18
        v21 = method95(v20)
        del v20
        v11[v14] = v21
        del v21
        v14 += 1 
    del v13, v14
    v23 = static_array(2)
    v24 = 0
    while method53(v24):
        v26 = u64(v24)
        v27 = v26 * 4
        del v26
        v28 = 6240 + v27
        del v27
        v30 = v0[v28:].view(cp.uint8)
        del v28
        v31 = method104(v30)
        del v30
        v23[v24] = v31
        del v31
        v24 += 1 
    del v24
    v32 = method105(v0)
    v34 = v0[6256:].view(cp.uint8)
    del v0
    if v32 == 0:
        method81(v34)
        v51 = US6_0()
    elif v32 == 1:
        v37, v38, v39, v40, v41, v42 = method84(v34)
        v51 = US6_1(v37, v38, v39, v40, v41, v42)
    elif v32 == 2:
        v44, v45, v46, v47, v48, v49 = method84(v34)
        v51 = US6_2(v44, v45, v46, v47, v48, v49)
    else:
        raise Exception("Invalid tag.")
    del v32, v34
    return v1, v9, v11, v23, v51
def method112(v0 : u64) -> object:
    v1 = v0
    del v0
    return v1
def method111(v0 : u64) -> object:
    return method112(v0)
def method114() -> object:
    v0 = []
    return v0
def method117(v0 : i32) -> object:
    v1 = v0
    del v0
    return v1
def method121(v0 : u8) -> object:
    v1 = v0
    del v0
    return v1
def method120(v0 : u8) -> object:
    return method121(v0)
def method119(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method53(v2):
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
        v11 = method120(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method118(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method53(v2):
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
        v11 = method119(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method122(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method53(v2):
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
        v11 = method117(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method124(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method59(v2):
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
        v11 = method120(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method125(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method61(v2):
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
        v11 = method120(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method126(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method63(v2):
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
        v11 = method120(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method123(v0 : US5) -> object:
    match v0:
        case US5_0(v1): # Flop
            del v0
            v2 = method124(v1)
            del v1
            v3 = "Flop"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US5_1(): # Preflop
            del v0
            v5 = method114()
            v6 = "Preflop"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case US5_2(v8): # River
            del v0
            v9 = method125(v8)
            del v8
            v10 = "River"
            v11 = [v10,v9]
            del v9, v10
            return v11
        case US5_3(v12): # Turn
            del v0
            v13 = method126(v12)
            del v12
            v14 = "Turn"
            v15 = [v14,v13]
            del v13, v14
            return v15
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method116(v0 : i32, v1 : static_array, v2 : static_array, v3 : i32, v4 : static_array, v5 : US5) -> object:
    v6 = method117(v0)
    del v0
    v7 = method118(v1)
    del v1
    v8 = method122(v2)
    del v2
    v9 = method117(v3)
    del v3
    v10 = method122(v4)
    del v4
    v11 = method123(v5)
    del v5
    v12 = {'min_raise': v6, 'pl_card': v7, 'pot': v8, 'round_turn': v9, 'stack': v10, 'street': v11}
    del v6, v7, v8, v9, v10, v11
    return v12
def method128(v0 : US1) -> object:
    match v0:
        case US1_0(): # A_All_In
            del v0
            v1 = method114()
            v2 = "A_All_In"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US1_1(): # A_Call
            del v0
            v4 = method114()
            v5 = "A_Call"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US1_2(): # A_Fold
            del v0
            v7 = method114()
            v8 = "A_Fold"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US1_3(v10): # A_Raise
            del v0
            v11 = method117(v10)
            del v10
            v12 = "A_Raise"
            v13 = [v12,v11]
            del v11, v12
            return v13
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method127(v0 : i32, v1 : static_array, v2 : static_array, v3 : i32, v4 : static_array, v5 : US5, v6 : US1) -> object:
    v7 = []
    v8 = method116(v0, v1, v2, v3, v4, v5)
    del v0, v1, v2, v3, v4, v5
    v7.append(v8)
    del v8
    v9 = method128(v6)
    del v6
    v7.append(v9)
    del v9
    v10 = v7
    del v7
    return v10
def method115(v0 : US4) -> object:
    match v0:
        case US4_0(v1, v2, v3, v4, v5, v6): # G_Flop
            del v0
            v7 = method116(v1, v2, v3, v4, v5, v6)
            del v1, v2, v3, v4, v5, v6
            v8 = "G_Flop"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US4_1(v10, v11, v12, v13, v14, v15): # G_Fold
            del v0
            v16 = method116(v10, v11, v12, v13, v14, v15)
            del v10, v11, v12, v13, v14, v15
            v17 = "G_Fold"
            v18 = [v17,v16]
            del v16, v17
            return v18
        case US4_2(): # G_Preflop
            del v0
            v19 = method114()
            v20 = "G_Preflop"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case US4_3(v22, v23, v24, v25, v26, v27): # G_River
            del v0
            v28 = method116(v22, v23, v24, v25, v26, v27)
            del v22, v23, v24, v25, v26, v27
            v29 = "G_River"
            v30 = [v29,v28]
            del v28, v29
            return v30
        case US4_4(v31, v32, v33, v34, v35, v36): # G_Round
            del v0
            v37 = method116(v31, v32, v33, v34, v35, v36)
            del v31, v32, v33, v34, v35, v36
            v38 = "G_Round"
            v39 = [v38,v37]
            del v37, v38
            return v39
        case US4_5(v40, v41, v42, v43, v44, v45, v46): # G_Round'
            del v0
            v47 = method127(v40, v41, v42, v43, v44, v45, v46)
            del v40, v41, v42, v43, v44, v45, v46
            v48 = "G_Round'"
            v49 = [v48,v47]
            del v47, v48
            return v49
        case US4_6(v50, v51, v52, v53, v54, v55): # G_Showdown
            del v0
            v56 = method116(v50, v51, v52, v53, v54, v55)
            del v50, v51, v52, v53, v54, v55
            v57 = "G_Showdown"
            v58 = [v57,v56]
            del v56, v57
            return v58
        case US4_7(v59, v60, v61, v62, v63, v64): # G_Turn
            del v0
            v65 = method116(v59, v60, v61, v62, v63, v64)
            del v59, v60, v61, v62, v63, v64
            v66 = "G_Turn"
            v67 = [v66,v65]
            del v65, v66
            return v67
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method113(v0 : US3) -> object:
    match v0:
        case US3_0(): # None
            del v0
            v1 = method114()
            v2 = "None"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US3_1(v4): # Some
            del v0
            v5 = method115(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method110(v0 : u64, v1 : US3) -> object:
    v2 = method111(v0)
    del v0
    v3 = method113(v1)
    del v1
    v4 = {'deck': v2, 'game': v3}
    del v2, v3
    return v4
def method132(v0 : static_array_list) -> object:
    v1 = []
    v2 = v0.length
    v3 = 0
    while method6(v2, v3):
        v6 = v0[v3]
        v7 = method120(v6)
        del v6
        v1.append(v7)
        del v7
        v3 += 1 
    del v0, v2, v3
    return v1
def method133(v0 : i32, v1 : i32) -> object:
    v2 = method117(v0)
    del v0
    v3 = method117(v1)
    del v1
    v4 = {'chips_won': v2, 'winner_id': v3}
    del v2, v3
    return v4
def method134(v0 : i32, v1 : US1) -> object:
    v2 = []
    v3 = method117(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method128(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method135(v0 : i32, v1 : static_array) -> object:
    v2 = []
    v3 = method117(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method119(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method140(v0 : i8) -> object:
    v1 = v0
    del v0
    return v1
def method139(v0 : static_array, v1 : i8) -> object:
    v2 = method125(v0)
    del v0
    v3 = method140(v1)
    del v1
    v4 = {'hand': v2, 'score': v3}
    del v2, v3
    return v4
def method138(v0 : static_array, v1 : i8) -> object:
    return method139(v0, v1)
def method137(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method53(v2):
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
        v13 = method138(v11, v12)
        del v11, v12
        v1.append(v13)
        del v13
        v2 += 1 
    del v0, v2
    return v1
def method136(v0 : i32, v1 : static_array, v2 : i32) -> object:
    v3 = method117(v0)
    del v0
    v4 = method137(v1)
    del v1
    v5 = method117(v2)
    del v2
    v6 = {'chips_won': v3, 'hands_shown': v4, 'winner_id': v5}
    del v3, v4, v5
    return v6
def method131(v0 : US7) -> object:
    match v0:
        case US7_0(v1): # CommunityCardsAre
            del v0
            v2 = method132(v1)
            del v1
            v3 = "CommunityCardsAre"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US7_1(v5, v6): # Fold
            del v0
            v7 = method133(v5, v6)
            del v5, v6
            v8 = "Fold"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US7_2(v10, v11): # PlayerAction
            del v0
            v12 = method134(v10, v11)
            del v10, v11
            v13 = "PlayerAction"
            v14 = [v13,v12]
            del v12, v13
            return v14
        case US7_3(v15, v16): # PlayerGotCards
            del v0
            v17 = method135(v15, v16)
            del v15, v16
            v18 = "PlayerGotCards"
            v19 = [v18,v17]
            del v17, v18
            return v19
        case US7_4(v20, v21, v22): # Showdown
            del v0
            v23 = method136(v20, v21, v22)
            del v20, v21, v22
            v24 = "Showdown"
            v25 = [v24,v23]
            del v23, v24
            return v25
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method130(v0 : static_array_list) -> object:
    v1 = []
    v2 = v0.length
    v3 = 0
    while method6(v2, v3):
        v6 = v0[v3]
        v7 = method131(v6)
        del v6
        v1.append(v7)
        del v7
        v3 += 1 
    del v0, v2, v3
    return v1
def method142(v0 : US2) -> object:
    match v0:
        case US2_0(): # Computer
            del v0
            v1 = method114()
            v2 = "Computer"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US2_1(): # Human
            del v0
            v4 = method114()
            v5 = "Human"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US2_2(): # Random
            del v0
            v7 = method114()
            v8 = "Random"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method141(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method53(v2):
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
        v11 = method142(v10)
        del v10
        v1.append(v11)
        del v11
        v2 += 1 
    del v0, v2
    return v1
def method143(v0 : US6) -> object:
    match v0:
        case US6_0(): # GameNotStarted
            del v0
            v1 = method114()
            v2 = "GameNotStarted"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US6_1(v4, v5, v6, v7, v8, v9): # GameOver
            del v0
            v10 = method116(v4, v5, v6, v7, v8, v9)
            del v4, v5, v6, v7, v8, v9
            v11 = "GameOver"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case US6_2(v13, v14, v15, v16, v17, v18): # WaitingForActionFromPlayerId
            del v0
            v19 = method116(v13, v14, v15, v16, v17, v18)
            del v13, v14, v15, v16, v17, v18
            v20 = "WaitingForActionFromPlayerId"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method129(v0 : static_array_list, v1 : static_array, v2 : US6) -> object:
    v3 = method130(v0)
    del v0
    v4 = method141(v1)
    del v1
    v5 = method143(v2)
    del v2
    v6 = {'messages': v3, 'pl_type': v4, 'ui_game_state': v5}
    del v3, v4, v5
    return v6
def method109(v0 : u64, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US6) -> object:
    v5 = method110(v0, v1)
    del v0, v1
    v6 = method129(v2, v3, v4)
    del v2, v3, v4
    v7 = {'private': v5, 'public': v6}
    del v5, v6
    return v7
def method149(v0 : cp.ndarray) -> object:
    v1 = v0
    del v0
    return v1
def method148(v0 : cp.ndarray) -> object:
    return method149(v0)
def method147(v0 : cp.ndarray, v1 : u64) -> object:
    v2 = []
    v3 = method148(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method112(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method146(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method147(v0, v1)
    del v0, v1
    v5 = method147(v2, v3)
    del v2, v3
    v6 = {'output': v4, 'param': v5}
    del v4, v5
    return v6
def method145(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    return method146(v0, v1, v2, v3)
def method144(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method145(v0, v1, v2, v3)
    del v0, v1, v2, v3
    v5 = {'model_data': v4}
    del v4
    return v5
def method108(v0 : u64, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US6, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64) -> object:
    v9 = method109(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    v10 = method144(v5, v6, v7, v8)
    del v5, v6, v7, v8
    v11 = {'game': v9, 'neural': v10}
    del v9, v10
    return v11
def method154(v0 : f32) -> object:
    v1 = v0
    del v0
    return v1
def method153(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method6(v2, v3):
        v5 = v0[v3]
        v6 = method154(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method152(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method6(v2, v3):
        v5 = v0[v3]
        v6 = method153(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method151(v0 : US8) -> object:
    match v0:
        case US8_0(v1): # AddRewardsRando
            del v0
            v2 = method152(v1)
            del v1
            v3 = "AddRewardsRando"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US8_1(v5): # AddRewardsSelf
            del v0
            v6 = method152(v5)
            del v5
            v7 = "AddRewardsSelf"
            v8 = [v7,v6]
            del v6, v7
            return v8
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method150(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method6(v2, v3):
        v5 = v0[v3]
        v6 = method151(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method107(v0 : u64, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US6, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64, v9 : list) -> object:
    v10 = []
    v11 = method108(v0, v1, v2, v3, v4, v5, v6, v7, v8)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8
    v10.append(v11)
    del v11
    v12 = method150(v9)
    del v9
    v10.append(v12)
    del v12
    v13 = v10
    del v10
    return v13
def method106(v0 : u64, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US6, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64, v9 : list) -> object:
    v10 = method107(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8, v9
    return v10
def method155(v0 : u64, v1 : US3, v2 : static_array_list, v3 : static_array, v4 : US6, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64) -> object:
    v9 = method108(v0, v1, v2, v3, v4, v5, v6, v7, v8)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8
    return v9
def main_body():
    v0 = Closure0()
    v1 = Closure1()
    v2 = collections.namedtuple("Holdem_Full",['event_loop_gpu', 'init'])(v0, v1)
    del v0, v1
    return v2

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())