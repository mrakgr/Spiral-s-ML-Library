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
    __syncthreads();
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
    __syncthreads();
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
    __syncthreads();
    assert("Tensor range check" && 0 <= v51 && v51 < 256);
    float v441;
    v441 = v47[v51];
    int v442;
    v442 = v49[v51];
    __syncthreads();
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
    __syncthreads();
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
    __syncthreads();
    assert("Tensor range check" && 0 <= v38 && v38 < 256);
    float v209;
    v209 = v36[v38];
    __syncthreads();
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
        Union5 v1782;
        switch (v20.tag) {
            case 0: { // None
                v1782 = Union5{Union5_0{}};
                break;
            }
            case 1: { // Some
                Union3 v22 = v20.case1.v0;
                Union6 v1433;
                switch (v22.tag) {
                    case 0: { // G_Flop
                        int v1294 = v22.case0.v0; static_array<static_array<unsigned char,2>,2> v1295 = v22.case0.v1; static_array<int,2> v1296 = v22.case0.v2; int v1297 = v22.case0.v3; static_array<int,2> v1298 = v22.case0.v4; Union4 v1299 = v22.case0.v5;
                        curandStatePhilox4_32_10_t & v1300 = v2.v5;
                        curandStatePhilox4_32_10_t & v1301 = v1300;
                        static_array<unsigned char,3> v1302; unsigned long long v1303;
                        Tuple1 tmp2 = draw_cards_1(v1301, v18);
                        v1302 = tmp2.v0; v1303 = tmp2.v1;
                        v2.v0 = v1303;
                        static_array_list<unsigned char,5> v1304;
                        v1304 = get_community_cards_4(v1299, v1302);
                        Union1 v1305;
                        v1305 = Union1{Union1_0{v1304}};
                        v17.push(v1305);
                        Union4 v1308;
                        switch (v1299.tag) {
                            case 1: { // Preflop
                                v1308 = Union4{Union4_0{v1302}};
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in flop.");
                                __trap();
                            }
                        }
                        int v1309;
                        v1309 = 2;
                        int v1310;
                        v1310 = 0;
                        Union3 v1311;
                        v1311 = try_round_5(v1309, v1295, v1296, v1310, v1298, v1308);
                        v1433 = Union6{Union6_2{v1311}};
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
                        v37 = -v35;
                        float v38;
                        v38 = (float)v37;
                        static_array<float,2> & v39 = v2.v4;
                        v39[v29] = v38;
                        int v40;
                        v40 = v29 ^ 1;
                        float v41;
                        v41 = -v38;
                        v39[v40] = v41;
                        int v42;
                        v42 = v26 + 1;
                        int v43;
                        v43 = v42 % 2;
                        Union1 v44;
                        v44 = Union1{Union1_1{v35, v43}};
                        v17.push(v44);
                        v1433 = Union6{Union6_0{}};
                        break;
                    }
                    case 2: { // G_Preflop
                        curandStatePhilox4_32_10_t & v1395 = v2.v5;
                        curandStatePhilox4_32_10_t & v1396 = v1395;
                        static_array<unsigned char,2> v1397; unsigned long long v1398;
                        Tuple5 tmp7 = draw_cards_8(v1396, v18);
                        v1397 = tmp7.v0; v1398 = tmp7.v1;
                        v2.v0 = v1398;
                        curandStatePhilox4_32_10_t & v1399 = v2.v5;
                        curandStatePhilox4_32_10_t & v1400 = v1399;
                        static_array<unsigned char,2> v1401; unsigned long long v1402;
                        Tuple5 tmp8 = draw_cards_8(v1400, v18);
                        v1401 = tmp8.v0; v1402 = tmp8.v1;
                        v2.v0 = v1402;
                        Union1 v1403;
                        v1403 = Union1{Union1_3{0, v1397}};
                        v17.push(v1403);
                        Union1 v1404;
                        v1404 = Union1{Union1_3{1, v1401}};
                        v17.push(v1404);
                        static_array<static_array<unsigned char,2>,2> v1405;
                        v1405[0] = v1397;
                        v1405[1] = v1401;
                        static_array<int,2> v1407;
                        v1407[0] = 2;
                        v1407[1] = 1;
                        static_array<int,2> v1409;
                        int v1411;
                        v1411 = 0;
                        while (while_method_2(v1411)){
                            bool v1413;
                            v1413 = 0 <= v1411;
                            bool v1415;
                            if (v1413){
                                bool v1414;
                                v1414 = v1411 < 2;
                                v1415 = v1414;
                            } else {
                                v1415 = false;
                            }
                            bool v1416;
                            v1416 = v1415 == false;
                            if (v1416){
                                assert("Index must be in range." && v1415);
                            } else {
                            }
                            int v1418;
                            v1418 = v1407[v1411];
                            int v1420;
                            v1420 = 100 - v1418;
                            v1409[v1411] = v1420;
                            v1411 += 1 ;
                        }
                        int v1421;
                        v1421 = 2;
                        int v1422;
                        v1422 = 0;
                        Union4 v1423;
                        v1423 = Union4{Union4_1{}};
                        Union3 v1424;
                        v1424 = try_round_5(v1421, v1405, v1407, v1422, v1409, v1423);
                        v1433 = Union6{Union6_2{v1424}};
                        break;
                    }
                    case 3: { // G_River
                        int v1354 = v22.case3.v0; static_array<static_array<unsigned char,2>,2> v1355 = v22.case3.v1; static_array<int,2> v1356 = v22.case3.v2; int v1357 = v22.case3.v3; static_array<int,2> v1358 = v22.case3.v4; Union4 v1359 = v22.case3.v5;
                        curandStatePhilox4_32_10_t & v1360 = v2.v5;
                        curandStatePhilox4_32_10_t & v1361 = v1360;
                        static_array<unsigned char,1> v1362; unsigned long long v1363;
                        Tuple6 tmp11 = draw_cards_9(v1361, v18);
                        v1362 = tmp11.v0; v1363 = tmp11.v1;
                        v2.v0 = v1363;
                        static_array_list<unsigned char,5> v1364;
                        v1364 = get_community_cards_10(v1359, v1362);
                        Union1 v1365;
                        v1365 = Union1{Union1_0{v1364}};
                        v17.push(v1365);
                        Union4 v1390;
                        switch (v1359.tag) {
                            case 3: { // Turn
                                static_array<unsigned char,4> v1366 = v1359.case3.v0;
                                static_array<unsigned char,5> v1367;
                                int v1369;
                                v1369 = 0;
                                while (while_method_0(v1369)){
                                    bool v1371;
                                    v1371 = 0 <= v1369;
                                    bool v1373;
                                    if (v1371){
                                        bool v1372;
                                        v1372 = v1369 < 4;
                                        v1373 = v1372;
                                    } else {
                                        v1373 = false;
                                    }
                                    bool v1374;
                                    v1374 = v1373 == false;
                                    if (v1374){
                                        assert("Index must be in range." && v1373);
                                    } else {
                                    }
                                    unsigned char v1376;
                                    v1376 = v1366[v1369];
                                    v1367[v1369] = v1376;
                                    v1369 += 1 ;
                                }
                                int v1378;
                                v1378 = 0;
                                while (while_method_6(v1378)){
                                    bool v1380;
                                    v1380 = 0 <= v1378;
                                    bool v1382;
                                    if (v1380){
                                        bool v1381;
                                        v1381 = v1378 < 1;
                                        v1382 = v1381;
                                    } else {
                                        v1382 = false;
                                    }
                                    bool v1383;
                                    v1383 = v1382 == false;
                                    if (v1383){
                                        assert("Index must be in range." && v1382);
                                    } else {
                                    }
                                    unsigned char v1385;
                                    v1385 = v1362[v1378];
                                    int v1387;
                                    v1387 = 4 + v1378;
                                    v1367[v1387] = v1385;
                                    v1378 += 1 ;
                                }
                                v1390 = Union4{Union4_2{v1367}};
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in river.");
                                __trap();
                            }
                        }
                        int v1391;
                        v1391 = 2;
                        int v1392;
                        v1392 = 0;
                        Union3 v1393;
                        v1393 = try_round_5(v1391, v1355, v1356, v1392, v1358, v1390);
                        v1433 = Union6{Union6_2{v1393}};
                        break;
                    }
                    case 4: { // G_Round
                        int v165 = v22.case4.v0; static_array<static_array<unsigned char,2>,2> v166 = v22.case4.v1; static_array<int,2> v167 = v22.case4.v2; int v168 = v22.case4.v3; static_array<int,2> v169 = v22.case4.v4; Union4 v170 = v22.case4.v5;
                        int v171;
                        v171 = v168 % 2;
                        static_array<Union0,2> & v172 = v2.v3;
                        bool v173;
                        v173 = 0 <= v171;
                        bool v175;
                        if (v173){
                            bool v174;
                            v174 = v171 < 2;
                            v175 = v174;
                        } else {
                            v175 = false;
                        }
                        bool v176;
                        v176 = v175 == false;
                        if (v176){
                            assert("Index must be in range." && v175);
                        } else {
                        }
                        Union0 v178;
                        v178 = v172[v171];
                        Union2 v1281;
                        switch (v178.tag) {
                            case 0: { // Computer
                                static_array_list<Union1,128> & v181 = v2.v2;
                                curandStatePhilox4_32_10_t & v182 = v2.v5;
                                curandStatePhilox4_32_10_t & v183 = v182;
                                unsigned int * v184;
                                v184 = reinterpret_cast<unsigned int *>(&v0[53477376ull]);
                                float * v186;
                                v186 = reinterpret_cast<float *>(&v0[0ull]);
                                float * v188;
                                v188 = reinterpret_cast<float *>(&v0[0ull]);
                                int v190;
                                v190 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v190 && v190 < 24);
                                int v191;
                                v191 = 524288 * v190;
                                int v192;
                                v192 = threadIdx.x;
                                int v193;
                                v193 = v192;
                                while (while_method_7(v193)){
                                    bool v195;
                                    v195 = 0 <= v193;
                                    bool v196;
                                    v196 = v195 == false;
                                    if (v196){
                                        assert("The index needs to be zero or positive." && v195);
                                    } else {
                                    }
                                    int v198;
                                    v198 = v193 % 2048;
                                    int v199;
                                    v199 = v193 / 2048;
                                    bool v200;
                                    v200 = v199 < 256;
                                    bool v201;
                                    v201 = v200 == false;
                                    if (v201){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v200);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v199 && v199 < 256);
                                    assert("Tensor range check" && 0 <= v198 && v198 < 2048);
                                    int v203;
                                    v203 = v198 + v191;
                                    int v204;
                                    v204 = 2048 * v199;
                                    int v205;
                                    v205 = v204 + v203;
                                    v188[v205] = 0.0f;
                                    v193 += 256 ;
                                }
                                __syncthreads();
                                int v206;
                                v206 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v206 && v206 < 256);
                                int v207;
                                v207 = 2048 * v206;
                                int v208;
                                v208 = v207 + v191;
                                int v209;
                                v209 = v181.length;
                                bool v210;
                                v210 = 128 >= v209;
                                bool v211;
                                v211 = v210 == false;
                                if (v211){
                                    assert("The type level dimension has to equal the value passed at runtime into create." && v210);
                                } else {
                                }
                                static_array_list<Union7,128> v213;
                                v213 = static_array_list<Union7,128>{};
                                v213.unsafe_set_length(v209);
                                int v215;
                                v215 = 0;
                                while (while_method_8(v209, v215)){
                                    Union1 v217;
                                    v217 = v181[v215];
                                    Union7 v223;
                                    switch (v217.tag) {
                                        case 2: { // PlayerAction
                                            int v219 = v217.case2.v0; Union2 v220 = v217.case2.v1;
                                            v223 = Union7{Union7_1{v220}};
                                            break;
                                        }
                                        default: {
                                            v223 = Union7{Union7_0{}};
                                        }
                                    }
                                    v213[v215] = v223;
                                    v215 += 1 ;
                                }
                                static_array<int,2> v224;
                                int v226;
                                v226 = 0;
                                while (while_method_2(v226)){
                                    int v228;
                                    v228 = v226 + v168;
                                    int v229;
                                    v229 = v228 % 2;
                                    bool v230;
                                    v230 = 0 <= v229;
                                    bool v232;
                                    if (v230){
                                        bool v231;
                                        v231 = v229 < 2;
                                        v232 = v231;
                                    } else {
                                        v232 = false;
                                    }
                                    bool v233;
                                    v233 = v232 == false;
                                    if (v233){
                                        assert("Index must be in range." && v232);
                                    } else {
                                    }
                                    int v235;
                                    v235 = v167[v229];
                                    v224[v226] = v235;
                                    v226 += 1 ;
                                }
                                static_array<int,2> v237;
                                int v239;
                                v239 = 0;
                                while (while_method_2(v239)){
                                    int v241;
                                    v241 = v239 + v168;
                                    int v242;
                                    v242 = v241 % 2;
                                    bool v243;
                                    v243 = 0 <= v242;
                                    bool v245;
                                    if (v243){
                                        bool v244;
                                        v244 = v242 < 2;
                                        v245 = v244;
                                    } else {
                                        v245 = false;
                                    }
                                    bool v246;
                                    v246 = v245 == false;
                                    if (v246){
                                        assert("Index must be in range." && v245);
                                    } else {
                                    }
                                    int v248;
                                    v248 = v169[v242];
                                    v237[v239] = v248;
                                    v239 += 1 ;
                                }
                                bool v251;
                                if (v173){
                                    bool v250;
                                    v250 = v171 < 2;
                                    v251 = v250;
                                } else {
                                    v251 = false;
                                }
                                bool v252;
                                v252 = v251 == false;
                                if (v252){
                                    assert("Index must be in range." && v251);
                                } else {
                                }
                                static_array<unsigned char,2> v254;
                                v254 = v166[v171];
                                static_array_list<unsigned char,5> v256;
                                v256 = static_array_list<unsigned char,5>{};
                                switch (v170.tag) {
                                    case 0: { // Flop
                                        static_array<unsigned char,3> v258 = v170.case0.v0;
                                        int v259;
                                        v259 = 0;
                                        while (while_method_4(v259)){
                                            bool v261;
                                            v261 = 0 <= v259;
                                            bool v263;
                                            if (v261){
                                                bool v262;
                                                v262 = v259 < 3;
                                                v263 = v262;
                                            } else {
                                                v263 = false;
                                            }
                                            bool v264;
                                            v264 = v263 == false;
                                            if (v264){
                                                assert("Index must be in range." && v263);
                                            } else {
                                            }
                                            unsigned char v266;
                                            v266 = v258[v259];
                                            v256.push(v266);
                                            v259 += 1 ;
                                        }
                                        break;
                                    }
                                    case 1: { // Preflop
                                        break;
                                    }
                                    case 2: { // River
                                        static_array<unsigned char,5> v278 = v170.case2.v0;
                                        int v279;
                                        v279 = 0;
                                        while (while_method_5(v279)){
                                            bool v281;
                                            v281 = 0 <= v279;
                                            bool v283;
                                            if (v281){
                                                bool v282;
                                                v282 = v279 < 5;
                                                v283 = v282;
                                            } else {
                                                v283 = false;
                                            }
                                            bool v284;
                                            v284 = v283 == false;
                                            if (v284){
                                                assert("Index must be in range." && v283);
                                            } else {
                                            }
                                            unsigned char v286;
                                            v286 = v278[v279];
                                            v256.push(v286);
                                            v279 += 1 ;
                                        }
                                        break;
                                    }
                                    case 3: { // Turn
                                        static_array<unsigned char,4> v268 = v170.case3.v0;
                                        int v269;
                                        v269 = 0;
                                        while (while_method_0(v269)){
                                            bool v271;
                                            v271 = 0 <= v269;
                                            bool v273;
                                            if (v271){
                                                bool v272;
                                                v272 = v269 < 4;
                                                v273 = v272;
                                            } else {
                                                v273 = false;
                                            }
                                            bool v274;
                                            v274 = v273 == false;
                                            if (v274){
                                                assert("Index must be in range." && v273);
                                            } else {
                                            }
                                            unsigned char v276;
                                            v276 = v268[v269];
                                            v256.push(v276);
                                            v269 += 1 ;
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                float * v288;
                                v288 = v188+v208;
                                int v290;
                                v290 = v213.length;
                                bool v291;
                                v291 = v290 == 0;
                                if (v291){
                                    v288[0] = 1.0f;
                                } else {
                                }
                                int v292;
                                v292 = v213.length;
                                int v293;
                                v293 = 0;
                                while (while_method_8(v292, v293)){
                                    Union7 v295;
                                    v295 = v213[v293];
                                    int v297;
                                    v297 = v293 * 14;
                                    int v298;
                                    v298 = 1 + v297;
                                    switch (v295.tag) {
                                        case 0: { // None
                                            v288[v298] = 1.0f;
                                            break;
                                        }
                                        case 1: { // Some
                                            Union2 v299 = v295.case1.v0;
                                            int v300;
                                            v300 = v298 + 1;
                                            switch (v299.tag) {
                                                case 0: { // A_All_In
                                                    v288[v300] = 1.0f;
                                                    break;
                                                }
                                                case 1: { // A_Call
                                                    int v301;
                                                    v301 = v300 + 1;
                                                    v288[v301] = 1.0f;
                                                    break;
                                                }
                                                case 2: { // A_Fold
                                                    int v302;
                                                    v302 = v300 + 2;
                                                    v288[v302] = 1.0f;
                                                    break;
                                                }
                                                case 3: { // A_Raise
                                                    int v303 = v299.case3.v0;
                                                    int v304;
                                                    v304 = v300 + 3;
                                                    bool v305;
                                                    v305 = 1 <= v303;
                                                    bool v307;
                                                    if (v305){
                                                        bool v306;
                                                        v306 = v303 < 1023;
                                                        v307 = v306;
                                                    } else {
                                                        v307 = false;
                                                    }
                                                    bool v308;
                                                    v308 = v307 == false;
                                                    if (v308){
                                                        assert("Pickle failure. The input is out of the bounds of the given range." && v307);
                                                    } else {
                                                    }
                                                    int v310;
                                                    v310 = v303 - 1;
                                                    unsigned int v311;
                                                    v311 = (unsigned int)v310;
                                                    method_11(v311, v288, v304);
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
                                    v293 += 1 ;
                                }
                                int v312;
                                v312 = 0;
                                while (while_method_2(v312)){
                                    // 222;
                                    bool v314;
                                    v314 = 0 <= v312;
                                    bool v316;
                                    if (v314){
                                        bool v315;
                                        v315 = v312 < 2;
                                        v316 = v315;
                                    } else {
                                        v316 = false;
                                    }
                                    bool v317;
                                    v317 = v316 == false;
                                    if (v317){
                                        assert("Index must be in range." && v316);
                                    } else {
                                    }
                                    int v319;
                                    v319 = v224[v312];
                                    int v321;
                                    v321 = v312 * 11;
                                    int v322;
                                    v322 = 1794 + v321;
                                    bool v323;
                                    v323 = 0 <= v319;
                                    bool v325;
                                    if (v323){
                                        bool v324;
                                        v324 = v319 < 1023;
                                        v325 = v324;
                                    } else {
                                        v325 = false;
                                    }
                                    bool v326;
                                    v326 = v325 == false;
                                    if (v326){
                                        assert("Pickle failure. The input is out of the bounds of the given range." && v325);
                                    } else {
                                    }
                                    unsigned int v328;
                                    v328 = (unsigned int)v319;
                                    method_12(v328, v288, v322);
                                    v312 += 1 ;
                                }
                                int v329;
                                v329 = 0;
                                while (while_method_2(v329)){
                                    // 222;
                                    bool v331;
                                    v331 = 0 <= v329;
                                    bool v333;
                                    if (v331){
                                        bool v332;
                                        v332 = v329 < 2;
                                        v333 = v332;
                                    } else {
                                        v333 = false;
                                    }
                                    bool v334;
                                    v334 = v333 == false;
                                    if (v334){
                                        assert("Index must be in range." && v333);
                                    } else {
                                    }
                                    int v336;
                                    v336 = v237[v329];
                                    int v338;
                                    v338 = v329 * 11;
                                    int v339;
                                    v339 = 1817 + v338;
                                    bool v340;
                                    v340 = 0 <= v336;
                                    bool v342;
                                    if (v340){
                                        bool v341;
                                        v341 = v336 < 1023;
                                        v342 = v341;
                                    } else {
                                        v342 = false;
                                    }
                                    bool v343;
                                    v343 = v342 == false;
                                    if (v343){
                                        assert("Pickle failure. The input is out of the bounds of the given range." && v342);
                                    } else {
                                    }
                                    unsigned int v345;
                                    v345 = (unsigned int)v336;
                                    method_12(v345, v288, v339);
                                    v329 += 1 ;
                                }
                                int v346;
                                v346 = 0;
                                while (while_method_2(v346)){
                                    // 222;
                                    bool v348;
                                    v348 = 0 <= v346;
                                    bool v350;
                                    if (v348){
                                        bool v349;
                                        v349 = v346 < 2;
                                        v350 = v349;
                                    } else {
                                        v350 = false;
                                    }
                                    bool v351;
                                    v351 = v350 == false;
                                    if (v351){
                                        assert("Index must be in range." && v350);
                                    } else {
                                    }
                                    unsigned char v353;
                                    v353 = v254[v346];
                                    int v355;
                                    v355 = v346 * 17;
                                    int v356;
                                    v356 = 1840 + v355;
                                    unsigned char v357;
                                    v357 = v353 % 4u;
                                    int v358;
                                    v358 = (int)v357;
                                    unsigned char v359;
                                    v359 = v353 / 4u;
                                    int v360;
                                    v360 = (int)v359;
                                    unsigned int v361;
                                    v361 = (unsigned int)v358;
                                    int v362;
                                    v362 = (int)v361;
                                    bool v363;
                                    v363 = v362 < 4;
                                    bool v364;
                                    v364 = v363 == false;
                                    if (v364){
                                        assert("Pickle failure. Int value out of bounds." && v363);
                                    } else {
                                    }
                                    int v366;
                                    v366 = v356 + v362;
                                    v288[v366] = 1.0f;
                                    int v367;
                                    v367 = v356 + 4;
                                    unsigned int v368;
                                    v368 = (unsigned int)v360;
                                    int v369;
                                    v369 = (int)v368;
                                    bool v370;
                                    v370 = v369 < 13;
                                    bool v371;
                                    v371 = v370 == false;
                                    if (v371){
                                        assert("Pickle failure. Int value out of bounds." && v370);
                                    } else {
                                    }
                                    int v373;
                                    v373 = v367 + v369;
                                    v288[v373] = 1.0f;
                                    v346 += 1 ;
                                }
                                int v374;
                                v374 = v256.length;
                                bool v375;
                                v375 = v374 == 0;
                                if (v375){
                                    v288[1874] = 1.0f;
                                } else {
                                }
                                int v376;
                                v376 = v256.length;
                                int v377;
                                v377 = 0;
                                while (while_method_8(v376, v377)){
                                    unsigned char v379;
                                    v379 = v256[v377];
                                    int v381;
                                    v381 = v377 * 17;
                                    int v382;
                                    v382 = 1875 + v381;
                                    unsigned char v383;
                                    v383 = v379 % 4u;
                                    int v384;
                                    v384 = (int)v383;
                                    unsigned char v385;
                                    v385 = v379 / 4u;
                                    int v386;
                                    v386 = (int)v385;
                                    unsigned int v387;
                                    v387 = (unsigned int)v384;
                                    int v388;
                                    v388 = (int)v387;
                                    bool v389;
                                    v389 = v388 < 4;
                                    bool v390;
                                    v390 = v389 == false;
                                    if (v390){
                                        assert("Pickle failure. Int value out of bounds." && v389);
                                    } else {
                                    }
                                    int v392;
                                    v392 = v382 + v388;
                                    v288[v392] = 1.0f;
                                    int v393;
                                    v393 = v382 + 4;
                                    unsigned int v394;
                                    v394 = (unsigned int)v386;
                                    int v395;
                                    v395 = (int)v394;
                                    bool v396;
                                    v396 = v395 < 13;
                                    bool v397;
                                    v397 = v396 == false;
                                    if (v397){
                                        assert("Pickle failure. Int value out of bounds." && v396);
                                    } else {
                                    }
                                    int v399;
                                    v399 = v393 + v395;
                                    v288[v399] = 1.0f;
                                    v377 += 1 ;
                                }
                                __syncthreads();
                                int v400;
                                v400 = 0;
                                while (while_method_0(v400)){
                                    float * v402;
                                    v402 = reinterpret_cast<float *>(&v0[0ull]);
                                    float * v404;
                                    v404 = reinterpret_cast<float *>(&v1[0ull]);
                                    assert("Tensor range check" && 0 <= v400 && v400 < 4);
                                    int v406;
                                    v406 = 262144 * v400;
                                    float * v407;
                                    v407 = reinterpret_cast<float *>(&v0[50331648ull]);
                                    int v409;
                                    v409 = blockIdx.x;
                                    assert("Tensor range check" && 0 <= v409 && v409 < 24);
                                    int v410;
                                    v410 = 524288 * v409;
                                    int v411;
                                    v411 = blockIdx.x;
                                    assert("Tensor range check" && 0 <= v411 && v411 < 24);
                                    int v412;
                                    v412 = 32768 * v411;
                                    cuda::pipeline<cuda::thread_scope_thread> v413 = cuda::make_pipeline();
                                    extern __shared__ unsigned char v414[];
                                    float * v415;
                                    v415 = reinterpret_cast<float *>(&v414[0ull]);
                                    float * v417;
                                    v417 = reinterpret_cast<float *>(&v414[34816ull]);
                                    float * v419;
                                    v419 = reinterpret_cast<float *>(&v414[0ull]);
                                    int v421;
                                    v421 = threadIdx.x;
                                    int v422;
                                    v422 = v421 / 32;
                                    bool v423;
                                    v423 = 0 <= v422;
                                    bool v424;
                                    v424 = v423 == false;
                                    if (v424){
                                        assert("The index needs to be zero or positive." && v423);
                                    } else {
                                    }
                                    int v426;
                                    v426 = v422 % 8;
                                    int v427;
                                    v427 = v422 / 8;
                                    bool v428;
                                    v428 = v427 < 1;
                                    bool v429;
                                    v429 = v428 == false;
                                    if (v429){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v428);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v427 && v427 < 1);
                                    assert("Tensor range check" && 0 <= v426 && v426 < 8);
                                    int v431;
                                    v431 = 16 * v426;
                                    int v432;
                                    v432 = 17408 * v427;
                                    int v433;
                                    v433 = v432 + v431;
                                    float * v434;
                                    v434 = v419+v433;
                                    assert("Tensor range check" && 0 <= v427 && v427 < 1);
                                    int v436;
                                    v436 = 8704 * v427;
                                    int v437;
                                    v437 = threadIdx.x;
                                    int v438;
                                    v438 = v437 % 32;
                                    bool v439;
                                    v439 = 0 <= v438;
                                    bool v440;
                                    v440 = v439 == false;
                                    if (v440){
                                        assert("The index needs to be zero or positive." && v439);
                                    } else {
                                    }
                                    int v442;
                                    v442 = v438 % 4;
                                    int v443;
                                    v443 = v438 / 4;
                                    bool v444;
                                    v444 = v443 < 8;
                                    bool v445;
                                    v445 = v444 == false;
                                    if (v445){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v444);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v443 && v443 < 8);
                                    assert("Tensor range check" && 0 <= v442 && v442 < 4);
                                    int v447;
                                    v447 = v442 + v436;
                                    int v448;
                                    v448 = 68 * v443;
                                    int v449;
                                    v449 = v448 + v447;
                                    float * v450;
                                    v450 = v415+v449;
                                    assert("Tensor range check" && 0 <= v426 && v426 < 8);
                                    int v452;
                                    v452 = 1088 * v426;
                                    int v453;
                                    v453 = threadIdx.x;
                                    int v454;
                                    v454 = v453 % 32;
                                    bool v455;
                                    v455 = 0 <= v454;
                                    bool v456;
                                    v456 = v455 == false;
                                    if (v456){
                                        assert("The index needs to be zero or positive." && v455);
                                    } else {
                                    }
                                    int v458;
                                    v458 = v454 % 4;
                                    int v459;
                                    v459 = v454 / 4;
                                    bool v460;
                                    v460 = v459 < 8;
                                    bool v461;
                                    v461 = v460 == false;
                                    if (v461){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v460);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v459 && v459 < 8);
                                    assert("Tensor range check" && 0 <= v458 && v458 < 4);
                                    int v463;
                                    v463 = v458 + v452;
                                    int v464;
                                    v464 = 68 * v459;
                                    int v465;
                                    v465 = v464 + v463;
                                    float * v466;
                                    v466 = v417+v465;
                                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> v468[8];
                                    int v469;
                                    v469 = 0;
                                    while (while_method_2(v469)){
                                        int v471;
                                        v471 = 0;
                                        while (while_method_6(v471)){
                                            assert("Tensor range check" && 0 <= v469 && v469 < 2);
                                            assert("Tensor range check" && 0 <= v471 && v471 < 1);
                                            int v473;
                                            v473 = 128 * v471;
                                            int v474;
                                            v474 = v473 + v412;
                                            int v475;
                                            v475 = 16384 * v469;
                                            int v476;
                                            v476 = v475 + v474;
                                            float * v477;
                                            v477 = v407+v476;
                                            // Pushing the loop unrolling to: 0
                                            int v479;
                                            v479 = 0;
                                            #pragma unroll
                                            while (while_method_1(v479)){
                                                int v481;
                                                v481 = 0;
                                                #pragma unroll
                                                while (while_method_6(v481)){
                                                    assert("Tensor range check" && 0 <= v479 && v479 < 8);
                                                    assert("Tensor range check" && 0 <= v481 && v481 < 1);
                                                    int v483;
                                                    v483 = v479 + v481;
                                                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v484 = v468[v483];
                                                    wmma::fill_fragment(v484, 0.0f);
                                                    v481 += 1 ;
                                                }
                                                v479 += 1 ;
                                            }
                                            // Poping the loop unrolling to: 0
                                            int v485;
                                            v485 = 0;
                                            while (while_method_11(v485)){
                                                int v487;
                                                v487 = v485 + 1;
                                                bool v488;
                                                v488 = v485 == 0;
                                                int v489;
                                                v489 = v485 % 2;
                                                bool v490;
                                                v490 = 0 <= v485;
                                                bool v491;
                                                v491 = v490 == false;
                                                if (v491){
                                                    assert("The index needs to be zero or positive." && v490);
                                                } else {
                                                }
                                                bool v493;
                                                v493 = v485 < 32;
                                                bool v494;
                                                v494 = v493 == false;
                                                if (v494){
                                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v493);
                                                } else {
                                                }
                                                bool v496;
                                                v496 = v487 < 32;
                                                Union8 v502;
                                                if (v496){
                                                    bool v497;
                                                    v497 = 0 <= v487;
                                                    bool v498;
                                                    v498 = v497 == false;
                                                    if (v498){
                                                        assert("The index needs to be zero or positive." && v497);
                                                    } else {
                                                    }
                                                    v502 = Union8{Union8_1{v487}};
                                                } else {
                                                    v502 = Union8{Union8_0{}};
                                                }
                                                assert("Tensor range check" && 0 <= v469 && v469 < 2);
                                                int v503;
                                                v503 = 262144 * v469;
                                                int v504;
                                                v504 = v503 + v410;
                                                assert("Tensor range check" && 0 <= v485 && v485 < 32);
                                                int v505;
                                                v505 = 64 * v485;
                                                int v506;
                                                v506 = v505 + v504;
                                                float * v507;
                                                v507 = v402+v506;
                                                assert("Tensor range check" && 0 <= v471 && v471 < 1);
                                                int v509;
                                                v509 = 262144 * v471;
                                                int v510;
                                                v510 = v509 + v406;
                                                if (v488){
                                                    assert("Tensor range check" && 0 <= v485 && v485 < 32);
                                                    int v511;
                                                    v511 = v505 + v510;
                                                    float * v512;
                                                    v512 = v404+v511;
                                                    // Pushing the loop unrolling to: 0
                                                    v413.producer_acquire();
                                                    int v514;
                                                    v514 = threadIdx.x;
                                                    bool v515;
                                                    v515 = 0 <= v514;
                                                    bool v516;
                                                    v516 = v515 == false;
                                                    if (v516){
                                                        assert("The index needs to be zero or positive." && v515);
                                                    } else {
                                                    }
                                                    int v518;
                                                    v518 = v514 % 16;
                                                    int v519;
                                                    v519 = v514 / 16;
                                                    bool v520;
                                                    v520 = v519 < 16;
                                                    bool v521;
                                                    v521 = v520 == false;
                                                    if (v521){
                                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v520);
                                                    } else {
                                                    }
                                                    assert("Tensor range check" && 0 <= v519 && v519 < 16);
                                                    assert("Tensor range check" && 0 <= v518 && v518 < 16);
                                                    int v523;
                                                    v523 = 4 * v518;
                                                    int v524;
                                                    v524 = 68 * v519;
                                                    int v525;
                                                    v525 = v524 + v523;
                                                    int v526;
                                                    v526 = 2048 * v519;
                                                    int v527;
                                                    v527 = v526 + v523;
                                                    float * v528;
                                                    v528 = v417+v525;
                                                    float * v530;
                                                    v530 = v512+v527;
                                                    int v532;
                                                    v532 = 0;
                                                    #pragma unroll
                                                    while (while_method_1(v532)){
                                                        int v534;
                                                        v534 = 0;
                                                        #pragma unroll
                                                        while (while_method_6(v534)){
                                                            assert("Tensor range check" && 0 <= v532 && v532 < 8);
                                                            assert("Tensor range check" && 0 <= v534 && v534 < 1);
                                                            int v536;
                                                            v536 = 64 * v534;
                                                            int v537;
                                                            v537 = 1088 * v532;
                                                            int v538;
                                                            v538 = v537 + v536;
                                                            int v539;
                                                            v539 = 32768 * v532;
                                                            int v540;
                                                            v540 = v539 + v536;
                                                            constexpr int v541 = sizeof(float) * 4;
                                                            assert("Pointer alignment check" && (unsigned long long)(v530 + v540) % v541 == 0 && (unsigned long long)(v528 + v538) % v541 == 0);
                                                            cuda::memcpy_async(v528 + v538, v530 + v540, cuda::aligned_size_t<v541>(v541), v413);
                                                            v534 += 1 ;
                                                        }
                                                        v532 += 1 ;
                                                    }
                                                    v413.producer_commit();
                                                    // Poping the loop unrolling to: 0
                                                } else {
                                                }
                                                // Pushing the loop unrolling to: 0
                                                int v542;
                                                v542 = threadIdx.x;
                                                bool v543;
                                                v543 = 0 <= v542;
                                                bool v544;
                                                v544 = v543 == false;
                                                if (v544){
                                                    assert("The index needs to be zero or positive." && v543);
                                                } else {
                                                }
                                                int v546;
                                                v546 = v542 % 16;
                                                int v547;
                                                v547 = v542 / 16;
                                                bool v548;
                                                v548 = v547 < 16;
                                                bool v549;
                                                v549 = v548 == false;
                                                if (v549){
                                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v548);
                                                } else {
                                                }
                                                assert("Tensor range check" && 0 <= v547 && v547 < 16);
                                                assert("Tensor range check" && 0 <= v546 && v546 < 16);
                                                int v551;
                                                v551 = 4 * v546;
                                                int v552;
                                                v552 = 68 * v547;
                                                int v553;
                                                v553 = v552 + v551;
                                                int v554;
                                                v554 = 2048 * v547;
                                                int v555;
                                                v555 = v554 + v551;
                                                float * v556;
                                                v556 = v415+v553;
                                                float * v558;
                                                v558 = v507+v555;
                                                int v560;
                                                v560 = 0;
                                                #pragma unroll
                                                while (while_method_1(v560)){
                                                    int v562;
                                                    v562 = 0;
                                                    #pragma unroll
                                                    while (while_method_6(v562)){
                                                        assert("Tensor range check" && 0 <= v560 && v560 < 8);
                                                        assert("Tensor range check" && 0 <= v562 && v562 < 1);
                                                        int v564;
                                                        v564 = 64 * v562;
                                                        int v565;
                                                        v565 = 1088 * v560;
                                                        int v566;
                                                        v566 = v565 + v564;
                                                        int v567;
                                                        v567 = 32768 * v560;
                                                        int v568;
                                                        v568 = v567 + v564;
                                                        int4* v569;
                                                        v569 = reinterpret_cast<int4*>(v558 + v568);
                                                        int4* v570;
                                                        v570 = reinterpret_cast<int4*>(v556 + v566);
                                                        assert("Pointer alignment check" && (unsigned long long)(v569) % 4 == 0 && (unsigned long long)(v570) % 4 == 0);
                                                        *v570 = *v569;
                                                        v562 += 1 ;
                                                    }
                                                    v560 += 1 ;
                                                }
                                                // Poping the loop unrolling to: 0
                                                wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v571[1];
                                                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v572[8];
                                                cuda::pipeline_consumer_wait_prior<0>(v413);;
                                                __syncthreads();
                                                // Pushing the loop unrolling to: 0
                                                int v573;
                                                v573 = 0;
                                                #pragma unroll
                                                while (while_method_6(v573)){
                                                    int v575;
                                                    v575 = 0;
                                                    #pragma unroll
                                                    while (while_method_1(v575)){
                                                        assert("Tensor range check" && 0 <= v573 && v573 < 1);
                                                        assert("Tensor range check" && 0 <= v575 && v575 < 8);
                                                        int v577;
                                                        v577 = 8 * v573;
                                                        int v578;
                                                        v578 = v577 + v575;
                                                        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v579 = v572[v578];
                                                        assert("Tensor range check" && 0 <= v573 && v573 < 1);
                                                        int v580;
                                                        v580 = 1088 * v573;
                                                        assert("Tensor range check" && 0 <= v575 && v575 < 8);
                                                        int v581;
                                                        v581 = 8 * v575;
                                                        int v582;
                                                        v582 = v581 + v580;
                                                        int v583;
                                                        v583 = 0;
                                                        #pragma unroll
                                                        while (while_method_2(v583)){
                                                            int v585;
                                                            v585 = 0;
                                                            #pragma unroll
                                                            while (while_method_2(v585)){
                                                                assert("Tensor range check" && 0 <= v583 && v583 < 2);
                                                                assert("Tensor range check" && 0 <= v585 && v585 < 2);
                                                                int v587;
                                                                v587 = 4 * v585;
                                                                int v588;
                                                                v588 = v587 + v582;
                                                                int v589;
                                                                v589 = 544 * v583;
                                                                int v590;
                                                                v590 = v589 + v588;
                                                                float v591;
                                                                v591 = v466[v590];
                                                                bool v592;
                                                                v592 = 0 <= v585;
                                                                bool v594;
                                                                if (v592){
                                                                    bool v593;
                                                                    v593 = v585 < 2;
                                                                    v594 = v593;
                                                                } else {
                                                                    v594 = false;
                                                                }
                                                                bool v595;
                                                                v595 = v594 == false;
                                                                if (v595){
                                                                    assert("The indices should be inside the range of the dimension." && v594);
                                                                } else {
                                                                }
                                                                bool v597;
                                                                v597 = 0 <= v583;
                                                                bool v599;
                                                                if (v597){
                                                                    bool v598;
                                                                    v598 = v583 < 2;
                                                                    v599 = v598;
                                                                } else {
                                                                    v599 = false;
                                                                }
                                                                bool v600;
                                                                v600 = v599 == false;
                                                                if (v600){
                                                                    assert("The indices should be inside the range of the dimension." && v599);
                                                                } else {
                                                                }
                                                                int v602;
                                                                v602 = v583 * 2;
                                                                int v603;
                                                                v603 = v585 + v602;
                                                                v579.x[v603] = wmma::__float_to_tf32(v591);
                                                                v585 += 1 ;
                                                            }
                                                            v583 += 1 ;
                                                        }
                                                        v575 += 1 ;
                                                    }
                                                    v573 += 1 ;
                                                }
                                                // Poping the loop unrolling to: 0
                                                v413.consumer_release();
                                                switch (v502.tag) {
                                                    case 0: { // None
                                                        break;
                                                    }
                                                    case 1: { // Some
                                                        int v604 = v502.case1.v0;
                                                        assert("Tensor range check" && 0 <= v604 && v604 < 32);
                                                        int v605;
                                                        v605 = 64 * v604;
                                                        int v606;
                                                        v606 = v605 + v510;
                                                        float * v607;
                                                        v607 = v404+v606;
                                                        __syncthreads();
                                                        // Pushing the loop unrolling to: 0
                                                        v413.producer_acquire();
                                                        int v609;
                                                        v609 = threadIdx.x;
                                                        bool v610;
                                                        v610 = 0 <= v609;
                                                        bool v611;
                                                        v611 = v610 == false;
                                                        if (v611){
                                                            assert("The index needs to be zero or positive." && v610);
                                                        } else {
                                                        }
                                                        int v613;
                                                        v613 = v609 % 16;
                                                        int v614;
                                                        v614 = v609 / 16;
                                                        bool v615;
                                                        v615 = v614 < 16;
                                                        bool v616;
                                                        v616 = v615 == false;
                                                        if (v616){
                                                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v615);
                                                        } else {
                                                        }
                                                        assert("Tensor range check" && 0 <= v614 && v614 < 16);
                                                        assert("Tensor range check" && 0 <= v613 && v613 < 16);
                                                        int v618;
                                                        v618 = 4 * v613;
                                                        int v619;
                                                        v619 = 68 * v614;
                                                        int v620;
                                                        v620 = v619 + v618;
                                                        int v621;
                                                        v621 = 2048 * v614;
                                                        int v622;
                                                        v622 = v621 + v618;
                                                        float * v623;
                                                        v623 = v417+v620;
                                                        float * v625;
                                                        v625 = v607+v622;
                                                        int v627;
                                                        v627 = 0;
                                                        #pragma unroll
                                                        while (while_method_1(v627)){
                                                            int v629;
                                                            v629 = 0;
                                                            #pragma unroll
                                                            while (while_method_6(v629)){
                                                                assert("Tensor range check" && 0 <= v627 && v627 < 8);
                                                                assert("Tensor range check" && 0 <= v629 && v629 < 1);
                                                                int v631;
                                                                v631 = 64 * v629;
                                                                int v632;
                                                                v632 = 1088 * v627;
                                                                int v633;
                                                                v633 = v632 + v631;
                                                                int v634;
                                                                v634 = 32768 * v627;
                                                                int v635;
                                                                v635 = v634 + v631;
                                                                constexpr int v636 = sizeof(float) * 4;
                                                                assert("Pointer alignment check" && (unsigned long long)(v625 + v635) % v636 == 0 && (unsigned long long)(v623 + v633) % v636 == 0);
                                                                cuda::memcpy_async(v623 + v633, v625 + v635, cuda::aligned_size_t<v636>(v636), v413);
                                                                v629 += 1 ;
                                                            }
                                                            v627 += 1 ;
                                                        }
                                                        v413.producer_commit();
                                                        // Poping the loop unrolling to: 0
                                                        break;
                                                    }
                                                    default: {
                                                        assert("Invalid tag." && false); __trap();
                                                    }
                                                }
                                                // Pushing the loop unrolling to: 0
                                                int v637;
                                                v637 = 0;
                                                #pragma unroll
                                                while (while_method_1(v637)){
                                                    int v639;
                                                    v639 = 0;
                                                    #pragma unroll
                                                    while (while_method_1(v639)){
                                                        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v641 = v571[0];
                                                        assert("Tensor range check" && 0 <= v637 && v637 < 8);
                                                        int v642;
                                                        v642 = 1088 * v637;
                                                        assert("Tensor range check" && 0 <= v639 && v639 < 8);
                                                        int v643;
                                                        v643 = 8 * v639;
                                                        int v644;
                                                        v644 = v643 + v642;
                                                        int v645;
                                                        v645 = 0;
                                                        #pragma unroll
                                                        while (while_method_2(v645)){
                                                            int v647;
                                                            v647 = 0;
                                                            #pragma unroll
                                                            while (while_method_2(v647)){
                                                                assert("Tensor range check" && 0 <= v645 && v645 < 2);
                                                                assert("Tensor range check" && 0 <= v647 && v647 < 2);
                                                                int v649;
                                                                v649 = 544 * v647;
                                                                int v650;
                                                                v650 = v649 + v644;
                                                                int v651;
                                                                v651 = 4 * v645;
                                                                int v652;
                                                                v652 = v651 + v650;
                                                                float v653;
                                                                v653 = v450[v652];
                                                                bool v654;
                                                                v654 = 0 <= v647;
                                                                bool v656;
                                                                if (v654){
                                                                    bool v655;
                                                                    v655 = v647 < 2;
                                                                    v656 = v655;
                                                                } else {
                                                                    v656 = false;
                                                                }
                                                                bool v657;
                                                                v657 = v656 == false;
                                                                if (v657){
                                                                    assert("The indices should be inside the range of the dimension." && v656);
                                                                } else {
                                                                }
                                                                bool v659;
                                                                v659 = 0 <= v645;
                                                                bool v661;
                                                                if (v659){
                                                                    bool v660;
                                                                    v660 = v645 < 2;
                                                                    v661 = v660;
                                                                } else {
                                                                    v661 = false;
                                                                }
                                                                bool v662;
                                                                v662 = v661 == false;
                                                                if (v662){
                                                                    assert("The indices should be inside the range of the dimension." && v661);
                                                                } else {
                                                                }
                                                                int v664;
                                                                v664 = v645 * 2;
                                                                int v665;
                                                                v665 = v647 + v664;
                                                                v641.x[v665] = wmma::__float_to_tf32(v653);
                                                                v647 += 1 ;
                                                            }
                                                            v645 += 1 ;
                                                        }
                                                        int v666;
                                                        v666 = 0;
                                                        #pragma unroll
                                                        while (while_method_6(v666)){
                                                            assert("Tensor range check" && 0 <= v637 && v637 < 8);
                                                            assert("Tensor range check" && 0 <= v666 && v666 < 1);
                                                            int v668;
                                                            v668 = v637 + v666;
                                                            wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v669 = v468[v668];
                                                            assert("Tensor range check" && 0 <= v666 && v666 < 1);
                                                            assert("Tensor range check" && 0 <= v639 && v639 < 8);
                                                            int v670;
                                                            v670 = 8 * v666;
                                                            int v671;
                                                            v671 = v670 + v639;
                                                            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v672 = v572[v671];
                                                            wmma::mma_sync(v669, v641, v672, v669);
                                                            v666 += 1 ;
                                                        }
                                                        v639 += 1 ;
                                                    }
                                                    v637 += 1 ;
                                                }
                                                // Poping the loop unrolling to: 0
                                                __syncthreads();
                                                v485 = v487;
                                            }
                                            // Pushing the loop unrolling to: 0
                                            int v673;
                                            v673 = 0;
                                            #pragma unroll
                                            while (while_method_1(v673)){
                                                int v675;
                                                v675 = 0;
                                                #pragma unroll
                                                while (while_method_6(v675)){
                                                    assert("Tensor range check" && 0 <= v673 && v673 < 8);
                                                    assert("Tensor range check" && 0 <= v675 && v675 < 1);
                                                    int v677;
                                                    v677 = v673 + v675;
                                                    wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v678 = v468[v677];
                                                    assert("Tensor range check" && 0 <= v673 && v673 < 8);
                                                    assert("Tensor range check" && 0 <= v675 && v675 < 1);
                                                    int v679;
                                                    v679 = 16 * v675;
                                                    int v680;
                                                    v680 = 2176 * v673;
                                                    int v681;
                                                    v681 = v680 + v679;
                                                    float * v682;
                                                    v682 = v434+v681;
                                                    wmma::store_matrix_sync(v682, v678, 136, wmma::mem_row_major);
                                                    v675 += 1 ;
                                                }
                                                v673 += 1 ;
                                            }
                                            // Poping the loop unrolling to: 0
                                            __syncthreads();
                                            // Pushing the loop unrolling to: 0
                                            int v684;
                                            v684 = threadIdx.x;
                                            bool v685;
                                            v685 = 0 <= v684;
                                            bool v686;
                                            v686 = v685 == false;
                                            if (v686){
                                                assert("The index needs to be zero or positive." && v685);
                                            } else {
                                            }
                                            int v688;
                                            v688 = v684 % 32;
                                            int v689;
                                            v689 = v684 / 32;
                                            bool v690;
                                            v690 = v689 < 8;
                                            bool v691;
                                            v691 = v690 == false;
                                            if (v691){
                                                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v690);
                                            } else {
                                            }
                                            assert("Tensor range check" && 0 <= v689 && v689 < 8);
                                            assert("Tensor range check" && 0 <= v688 && v688 < 32);
                                            int v693;
                                            v693 = 4 * v688;
                                            int v694;
                                            v694 = 128 * v689;
                                            int v695;
                                            v695 = v694 + v693;
                                            int v696;
                                            v696 = 136 * v689;
                                            int v697;
                                            v697 = v696 + v693;
                                            float * v698;
                                            v698 = v477+v695;
                                            float * v700;
                                            v700 = v419+v697;
                                            int v702;
                                            v702 = 0;
                                            #pragma unroll
                                            while (while_method_12(v702)){
                                                int v704;
                                                v704 = 0;
                                                #pragma unroll
                                                while (while_method_6(v704)){
                                                    assert("Tensor range check" && 0 <= v702 && v702 < 16);
                                                    assert("Tensor range check" && 0 <= v704 && v704 < 1);
                                                    int v706;
                                                    v706 = 128 * v704;
                                                    int v707;
                                                    v707 = 1024 * v702;
                                                    int v708;
                                                    v708 = v707 + v706;
                                                    int v709;
                                                    v709 = 1088 * v702;
                                                    int v710;
                                                    v710 = v709 + v706;
                                                    int4* v711;
                                                    v711 = reinterpret_cast<int4*>(v700 + v710);
                                                    int4* v712;
                                                    v712 = reinterpret_cast<int4*>(v698 + v708);
                                                    assert("Pointer alignment check" && (unsigned long long)(v711) % 4 == 0 && (unsigned long long)(v712) % 4 == 0);
                                                    *v712 = *v711;
                                                    v704 += 1 ;
                                                }
                                                v702 += 1 ;
                                            }
                                            // Poping the loop unrolling to: 0
                                            __syncthreads();
                                            v471 += 1 ;
                                        }
                                        v469 += 1 ;
                                    }
                                    unsigned int * v713;
                                    v713 = reinterpret_cast<unsigned int *>(&v0[53477376ull]);
                                    assert("Tensor range check" && 0 <= v400 && v400 < 4);
                                    int v715;
                                    v715 = 6144 * v400;
                                    method_13(v713, v715, v407);
                                    int * v716;
                                    v716 = reinterpret_cast<int *>(&v1[4194304ull]);
                                    float * v718;
                                    v718 = reinterpret_cast<float *>(&v1[4194320ull]);
                                    float * v720;
                                    v720 = reinterpret_cast<float *>(&v1[5242896ull]);
                                    float * v722;
                                    v722 = reinterpret_cast<float *>(&v1[6291472ull]);
                                    float * v724;
                                    v724 = reinterpret_cast<float *>(&v1[7340048ull]);
                                    float * v726;
                                    v726 = reinterpret_cast<float *>(&v1[8388624ull]);
                                    float * v728;
                                    v728 = reinterpret_cast<float *>(&v1[9437200ull]);
                                    float * v730;
                                    v730 = reinterpret_cast<float *>(&v1[10485776ull]);
                                    int * v732;
                                    v732 = reinterpret_cast<int *>(&v0[53575680ull]);
                                    float * v734;
                                    v734 = reinterpret_cast<float *>(&v0[66158592ull]);
                                    int * v736;
                                    v736 = reinterpret_cast<int *>(&v0[78741504ull]);
                                    int * v738;
                                    v738 = reinterpret_cast<int *>(&v0[91324416ull]);
                                    double * v740;
                                    v740 = reinterpret_cast<double *>(&v0[103907328ull]);
                                    double * v742;
                                    v742 = reinterpret_cast<double *>(&v0[154238976ull]);
                                    double * v744;
                                    v744 = reinterpret_cast<double *>(&v1[11534352ull]);
                                    double * v746;
                                    v746 = reinterpret_cast<double *>(&v1[11927568ull]);
                                    int * v748;
                                    v748 = reinterpret_cast<int *>(&v1[12320784ull]);
                                    v400 += 1 ;
                                }
                                __syncthreads();
                                int * v750;
                                v750 = reinterpret_cast<int *>(&v1[4194304ull]);
                                float * v752;
                                v752 = reinterpret_cast<float *>(&v1[4194320ull]);
                                float * v754;
                                v754 = reinterpret_cast<float *>(&v1[5242896ull]);
                                float * v756;
                                v756 = reinterpret_cast<float *>(&v1[6291472ull]);
                                float * v758;
                                v758 = reinterpret_cast<float *>(&v1[7340048ull]);
                                float * v760;
                                v760 = reinterpret_cast<float *>(&v1[8388624ull]);
                                float * v762;
                                v762 = reinterpret_cast<float *>(&v1[9437200ull]);
                                float * v764;
                                v764 = reinterpret_cast<float *>(&v1[10485776ull]);
                                int v766;
                                v766 = v750[0];
                                unsigned int * v767;
                                v767 = reinterpret_cast<unsigned int *>(&v0[53477376ull]);
                                int v769;
                                v769 = blockIdx.x;
                                int v770;
                                v770 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v766 && v766 < 4);
                                assert("Tensor range check" && 0 <= v769 && v769 < 24);
                                assert("Tensor range check" && 0 <= v770 && v770 < 256);
                                int v771;
                                v771 = 256 * v769;
                                int v772;
                                v772 = v771 + v770;
                                int v773;
                                v773 = 6144 * v766;
                                int v774;
                                v774 = v773 + v772;
                                unsigned int v775;
                                v775 = v767[v774];
                                int v776;
                                v776 = (int)v775;
                                float v777; int v778;
                                Tuple8 tmp24 = method_14(v183, v750, v752, v754, v756, v758, v760, v762, v764, v776, v766);
                                v777 = tmp24.v0; v778 = tmp24.v1;
                                extern __shared__ unsigned char v779[];
                                float * v780;
                                v780 = reinterpret_cast<float *>(&v779[0ull]);
                                int * v782;
                                v782 = reinterpret_cast<int *>(&v779[16ull]);
                                int v784;
                                v784 = threadIdx.x;
                                bool v785;
                                v785 = v784 == 0;
                                if (v785){
                                    v780[0] = v777;
                                    v782[0] = v778;
                                } else {
                                }
                                __syncthreads();
                                float v786;
                                v786 = v780[0];
                                int v787;
                                v787 = v782[0];
                                __syncthreads();
                                double * v788;
                                v788 = reinterpret_cast<double *>(&v1[11534352ull]);
                                double * v790;
                                v790 = reinterpret_cast<double *>(&v1[11927568ull]);
                                int * v792;
                                v792 = reinterpret_cast<int *>(&v1[12320784ull]);
                                int * v794;
                                v794 = reinterpret_cast<int *>(&v0[53575680ull]);
                                float * v796;
                                v796 = reinterpret_cast<float *>(&v0[66158592ull]);
                                int * v798;
                                v798 = reinterpret_cast<int *>(&v0[78741504ull]);
                                int * v800;
                                v800 = reinterpret_cast<int *>(&v0[91324416ull]);
                                double * v802;
                                v802 = reinterpret_cast<double *>(&v0[103907328ull]);
                                double * v804;
                                v804 = reinterpret_cast<double *>(&v0[154238976ull]);
                                int v806;
                                v806 = threadIdx.x;
                                int v807;
                                v807 = blockIdx.x;
                                int v808;
                                v808 = v807 * 256;
                                int v809;
                                v809 = v806 + v808;
                                int v810;
                                v810 = 0;
                                while (while_method_0(v810)){
                                    unsigned int * v812;
                                    v812 = reinterpret_cast<unsigned int *>(&v0[53477376ull]);
                                    int v814;
                                    v814 = blockIdx.x;
                                    int v815;
                                    v815 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v810 && v810 < 4);
                                    assert("Tensor range check" && 0 <= v814 && v814 < 24);
                                    assert("Tensor range check" && 0 <= v815 && v815 < 256);
                                    int v816;
                                    v816 = 256 * v814;
                                    int v817;
                                    v817 = v816 + v815;
                                    int v818;
                                    v818 = 6144 * v810;
                                    int v819;
                                    v819 = v818 + v817;
                                    unsigned int v820;
                                    v820 = v812[v819];
                                    int v821;
                                    v821 = (int)v820;
                                    float v822;
                                    v822 = method_15(v750, v752, v754, v756, v758, v760, v762, v764, v821, v810, v787);
                                    assert("Tensor range check" && 0 <= v810 && v810 < 4);
                                    assert("Tensor range check" && 0 <= v809 && v809 < 6144);
                                    int v823;
                                    v823 = v818 + v809;
                                    int v824;
                                    v824 = v792[v823];
                                    int v825;
                                    v825 = v824 + 1;
                                    assert("Tensor range check" && 0 <= v810 && v810 < 4);
                                    assert("Tensor range check" && 0 <= v809 && v809 < 6144);
                                    v792[v823] = v825;
                                    assert("Tensor range check" && 0 <= v810 && v810 < 4);
                                    assert("Tensor range check" && 0 <= v824 && v824 < 128);
                                    assert("Tensor range check" && 0 <= v809 && v809 < 6144);
                                    int v826;
                                    v826 = 6144 * v824;
                                    int v827;
                                    v827 = v826 + v809;
                                    int v828;
                                    v828 = 786432 * v810;
                                    int v829;
                                    v829 = v828 + v827;
                                    v794[v829] = v787;
                                    v796[v829] = v786;
                                    v798[v829] = v171;
                                    v800[v829] = v821;
                                    assert("Tensor range check" && 0 <= v810 && v810 < 4);
                                    int v830;
                                    v830 = 12288 * v810;
                                    assert("Tensor range check" && 0 <= v809 && v809 < 6144);
                                    int v831;
                                    v831 = 2 * v809;
                                    int v832;
                                    v832 = v831 + v830;
                                    assert("Tensor range check" && 0 <= v810 && v810 < 4);
                                    int v833;
                                    v833 = 1572864 * v810;
                                    assert("Tensor range check" && 0 <= v824 && v824 < 128);
                                    int v834;
                                    v834 = 12288 * v824;
                                    int v835;
                                    v835 = v834 + v833;
                                    assert("Tensor range check" && 0 <= v809 && v809 < 6144);
                                    int v836;
                                    v836 = v831 + v835;
                                    double * v837;
                                    v837 = v788+v832;
                                    double * v839;
                                    v839 = v790+v832;
                                    double * v841;
                                    v841 = v802+v836;
                                    double * v843;
                                    v843 = v804+v836;
                                    int v845;
                                    v845 = sizeof(double *);
                                    unsigned long long v846;
                                    v846 = (unsigned long long)v845;
                                    unsigned long long v847;
                                    v847 = 256ull * v846;
                                    unsigned long long v848;
                                    v848 = v847 + 16ull;
                                    unsigned long long v849;
                                    v849 = v848 - 1ull;
                                    unsigned long long v850;
                                    v850 = v849 % 16ull;
                                    unsigned long long v851;
                                    v851 = v849 - v850;
                                    unsigned long long v852;
                                    v852 = v851 + v847;
                                    unsigned long long v853;
                                    v853 = v852 + 16ull;
                                    unsigned long long v854;
                                    v854 = v853 - 1ull;
                                    unsigned long long v855;
                                    v855 = v854 % 16ull;
                                    unsigned long long v856;
                                    v856 = v854 - v855;
                                    unsigned long long v857;
                                    v857 = v856 + v847;
                                    unsigned long long v858;
                                    v858 = v857 + 16ull;
                                    unsigned long long v859;
                                    v859 = v858 - 1ull;
                                    unsigned long long v860;
                                    v860 = v859 % 16ull;
                                    unsigned long long v861;
                                    v861 = v859 - v860;
                                    unsigned long long v862;
                                    v862 = v861 + v847;
                                    bool v863;
                                    v863 = v862 <= 98304ull;
                                    bool v864;
                                    v864 = v863 == false;
                                    if (v864){
                                        assert("The dynamic shared memory is insufficient to allocate the tensor." && v863);
                                    } else {
                                    }
                                    extern __shared__ unsigned char v866[];
                                    bool v867;
                                    v867 = v862 <= v862;
                                    bool v868;
                                    v868 = v867 == false;
                                    if (v868){
                                        assert("The length of the partition has to be less than or equal to the length of the base array." && v867);
                                    } else {
                                    }
                                    double * * v870;
                                    v870 = reinterpret_cast<double * *>(&v866[0ull]);
                                    double * * v872;
                                    v872 = reinterpret_cast<double * *>(&v866[v851]);
                                    double * * v874;
                                    v874 = reinterpret_cast<double * *>(&v866[v856]);
                                    double * * v876;
                                    v876 = reinterpret_cast<double * *>(&v866[v861]);
                                    int v878;
                                    v878 = threadIdx.x;
                                    assert("Tensor range check" && 0 <= v878 && v878 < 256);
                                    v870[v878] = v837;
                                    v872[v878] = v839;
                                    v874[v878] = v841;
                                    v876[v878] = v843;
                                    __syncthreads();
                                    bool v879;
                                    v879 = 0 <= v878;
                                    bool v880;
                                    v880 = v879 == false;
                                    if (v880){
                                        assert("The index needs to be zero or positive." && v879);
                                    } else {
                                    }
                                    int v882;
                                    v882 = v878 % 1;
                                    bool v883;
                                    v883 = v878 < 256;
                                    bool v884;
                                    v884 = v883 == false;
                                    if (v884){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v883);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v878 && v878 < 256);
                                    int v886;
                                    v886 = 0;
                                    while (while_method_6(v886)){
                                        bool v888;
                                        v888 = v879 && v883;
                                        bool v889;
                                        v889 = v888 == false;
                                        if (v889){
                                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v888);
                                        } else {
                                        }
                                        bool v891;
                                        v891 = 0 <= v886;
                                        bool v893;
                                        if (v891){
                                            bool v892;
                                            v892 = v886 < 1;
                                            v893 = v892;
                                        } else {
                                            v893 = false;
                                        }
                                        bool v894;
                                        v894 = v893 == false;
                                        if (v894){
                                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v893);
                                        } else {
                                        }
                                        int v896;
                                        v896 = v886 * 256;
                                        int v897;
                                        v897 = v896 + v878;
                                        assert("Tensor range check" && 0 <= v886 && v886 < 1);
                                        int v898;
                                        v898 = 256 * v886;
                                        int v899;
                                        v899 = v898 + v878;
                                        double * v900;
                                        v900 = v870[v899];
                                        double * v901;
                                        v901 = v872[v899];
                                        double * v902;
                                        v902 = v874[v899];
                                        double * v903;
                                        v903 = v876[v899];
                                        int v904;
                                        v904 = blockIdx.x;
                                        int v905;
                                        v905 = v904 * 256;
                                        int v906;
                                        v906 = v905 + v897;
                                        assert("Tensor range check" && 0 <= v882 && v882 < 1);
                                        int v907;
                                        v907 = 2 * v882;
                                        double v908[2];
                                        double v909[2];
                                        int v910[2];
                                        int v911;
                                        v911 = 0;
                                        while (while_method_6(v911)){
                                            assert("Tensor range check" && 0 <= v911 && v911 < 1);
                                            int v913;
                                            v913 = 2 * v911;
                                            assert("Tensor range check" && 0 <= v911 && v911 < 1);
                                            int v914;
                                            v914 = v913 + v907;
                                            int4* v915;
                                            v915 = reinterpret_cast<int4*>(v900 + v914);
                                            int4* v916;
                                            v916 = reinterpret_cast<int4*>(v908 + v913);
                                            assert("Pointer alignment check" && (unsigned long long)(v915) % 2 == 0 && (unsigned long long)(v916) % 2 == 0);
                                            *v916 = *v915;
                                            int4* v917;
                                            v917 = reinterpret_cast<int4*>(v901 + v914);
                                            int4* v918;
                                            v918 = reinterpret_cast<int4*>(v909 + v913);
                                            assert("Pointer alignment check" && (unsigned long long)(v917) % 2 == 0 && (unsigned long long)(v918) % 2 == 0);
                                            *v918 = *v917;
                                            v911 += 1 ;
                                        }
                                        int v919;
                                        v919 = 0;
                                        while (while_method_6(v919)){
                                            int v921;
                                            v921 = 0;
                                            while (while_method_2(v921)){
                                                bool v923;
                                                v923 = 0 <= v921;
                                                bool v925;
                                                if (v923){
                                                    bool v924;
                                                    v924 = v921 < 2;
                                                    v925 = v924;
                                                } else {
                                                    v925 = false;
                                                }
                                                bool v926;
                                                v926 = v925 == false;
                                                if (v926){
                                                    assert("The indices should be inside the range of the dimension." && v925);
                                                } else {
                                                }
                                                bool v928;
                                                v928 = 0 <= v882;
                                                bool v930;
                                                if (v928){
                                                    bool v929;
                                                    v929 = v882 < 1;
                                                    v930 = v929;
                                                } else {
                                                    v930 = false;
                                                }
                                                bool v931;
                                                v931 = v930 == false;
                                                if (v931){
                                                    assert("The indices should be inside the range of the dimension." && v930);
                                                } else {
                                                }
                                                int v933;
                                                v933 = v882 * 2;
                                                int v934;
                                                v934 = v921 + v933;
                                                bool v935;
                                                v935 = 0 <= v919;
                                                bool v937;
                                                if (v935){
                                                    bool v936;
                                                    v936 = v919 < 1;
                                                    v937 = v936;
                                                } else {
                                                    v937 = false;
                                                }
                                                bool v938;
                                                v938 = v937 == false;
                                                if (v938){
                                                    assert("The indices should be inside the range of the dimension." && v937);
                                                } else {
                                                }
                                                int v940;
                                                v940 = v919 * 2;
                                                int v941;
                                                v941 = v934 + v940;
                                                assert("Tensor range check" && 0 <= v919 && v919 < 1);
                                                assert("Tensor range check" && 0 <= v921 && v921 < 2);
                                                int v942;
                                                v942 = 2 * v919;
                                                int v943;
                                                v943 = v942 + v921;
                                                v910[v943] = v941;
                                                v921 += 1 ;
                                            }
                                            v919 += 1 ;
                                        }
                                        int v944;
                                        v944 = 0;
                                        while (while_method_6(v944)){
                                            assert("Tensor range check" && 0 <= v944 && v944 < 1);
                                            int v946;
                                            v946 = 2 * v944;
                                            int v947;
                                            v947 = v946 + v907;
                                            assert("Tensor range check" && 0 <= v944 && v944 < 1);
                                            int4* v948;
                                            v948 = reinterpret_cast<int4*>(v908 + v946);
                                            int4* v949;
                                            v949 = reinterpret_cast<int4*>(v902 + v947);
                                            assert("Pointer alignment check" && (unsigned long long)(v948) % 2 == 0 && (unsigned long long)(v949) % 2 == 0);
                                            *v949 = *v948;
                                            int4* v950;
                                            v950 = reinterpret_cast<int4*>(v909 + v946);
                                            int4* v951;
                                            v951 = reinterpret_cast<int4*>(v903 + v947);
                                            assert("Pointer alignment check" && (unsigned long long)(v950) % 2 == 0 && (unsigned long long)(v951) % 2 == 0);
                                            *v951 = *v950;
                                            v944 += 1 ;
                                        }
                                        assert("Tensor range check" && 0 <= v897 && v897 < 256);
                                        v886 += 1 ;
                                    }
                                    __syncthreads();
                                    assert("Tensor range check" && 0 <= v878 && v878 < 256);
                                    __syncthreads();
                                    double v952;
                                    v952 = (double)v786;
                                    double v953;
                                    v953 = log(v952);
                                    double v954;
                                    v954 = (double)v822;
                                    double v955;
                                    v955 = log(v954);
                                    assert("Tensor range check" && 0 <= v810 && v810 < 4);
                                    assert("Tensor range check" && 0 <= v809 && v809 < 6144);
                                    assert("Tensor range check" && 0 <= v171 && v171 < 2);
                                    int v956;
                                    v956 = v831 + v171;
                                    int v957;
                                    v957 = v830 + v956;
                                    double v958;
                                    v958 = v788[v957];
                                    double v959;
                                    v959 = v790[v957];
                                    double v960;
                                    v960 = v955 + v958;
                                    double v961;
                                    v961 = v953 + v959;
                                    assert("Tensor range check" && 0 <= v810 && v810 < 4);
                                    assert("Tensor range check" && 0 <= v809 && v809 < 6144);
                                    assert("Tensor range check" && 0 <= v171 && v171 < 2);
                                    v788[v957] = v960;
                                    v790[v957] = v961;
                                    v810 += 1 ;
                                }
                                bool v962;
                                v962 = 0 == v787;
                                Union9 v995;
                                if (v962){
                                    v995 = Union9{Union9_1{}};
                                } else {
                                    bool v964;
                                    v964 = 1 == v787;
                                    if (v964){
                                        v995 = Union9{Union9_0{}};
                                    } else {
                                        bool v966;
                                        v966 = 2 == v787;
                                        if (v966){
                                            v995 = Union9{Union9_2{1, 3}};
                                        } else {
                                            bool v968;
                                            v968 = 3 == v787;
                                            if (v968){
                                                v995 = Union9{Union9_2{1, 2}};
                                            } else {
                                                bool v970;
                                                v970 = 4 == v787;
                                                if (v970){
                                                    v995 = Union9{Union9_2{2, 3}};
                                                } else {
                                                    bool v972;
                                                    v972 = 5 == v787;
                                                    if (v972){
                                                        v995 = Union9{Union9_2{3, 4}};
                                                    } else {
                                                        bool v974;
                                                        v974 = 6 == v787;
                                                        if (v974){
                                                            v995 = Union9{Union9_2{1, 1}};
                                                        } else {
                                                            bool v976;
                                                            v976 = 7 == v787;
                                                            if (v976){
                                                                v995 = Union9{Union9_2{3, 2}};
                                                            } else {
                                                                bool v978;
                                                                v978 = 8 == v787;
                                                                if (v978){
                                                                    v995 = Union9{Union9_2{2, 1}};
                                                                } else {
                                                                    bool v980;
                                                                    v980 = 9 == v787;
                                                                    if (v980){
                                                                        v995 = Union9{Union9_2{3, 1}};
                                                                    } else {
                                                                        bool v982;
                                                                        v982 = 10 == v787;
                                                                        if (v982){
                                                                            v995 = Union9{Union9_2{2147483647, 1}};
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
                                switch (v995.tag) {
                                    case 0: { // AA_Call
                                        v1281 = Union2{Union2_1{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v996;
                                        v996 = v167[0];
                                        int v998; int v999;
                                        Tuple4 tmp27 = Tuple4{1, v996};
                                        v998 = tmp27.v0; v999 = tmp27.v1;
                                        while (while_method_2(v998)){
                                            bool v1001;
                                            v1001 = 0 <= v998;
                                            bool v1003;
                                            if (v1001){
                                                bool v1002;
                                                v1002 = v998 < 2;
                                                v1003 = v1002;
                                            } else {
                                                v1003 = false;
                                            }
                                            bool v1004;
                                            v1004 = v1003 == false;
                                            if (v1004){
                                                assert("Index must be in range." && v1003);
                                            } else {
                                            }
                                            int v1006;
                                            v1006 = v167[v998];
                                            bool v1008;
                                            v1008 = v999 >= v1006;
                                            int v1009;
                                            if (v1008){
                                                v1009 = v999;
                                            } else {
                                                v1009 = v1006;
                                            }
                                            v999 = v1009;
                                            v998 += 1 ;
                                        }
                                        bool v1011;
                                        if (v173){
                                            bool v1010;
                                            v1010 = v171 < 2;
                                            v1011 = v1010;
                                        } else {
                                            v1011 = false;
                                        }
                                        bool v1012;
                                        v1012 = v1011 == false;
                                        if (v1012){
                                            assert("Index must be in range." && v1011);
                                        } else {
                                        }
                                        int v1014;
                                        v1014 = v167[v171];
                                        bool v1016;
                                        v1016 = v1014 == v999;
                                        if (v1016){
                                            v1281 = Union2{Union2_1{}};
                                        } else {
                                            v1281 = Union2{Union2_2{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        int v1021 = v995.case2.v0; int v1022 = v995.case2.v1;
                                        static_array<int,2> v1023;
                                        int v1025;
                                        v1025 = 0;
                                        while (while_method_2(v1025)){
                                            bool v1027;
                                            v1027 = 0 <= v1025;
                                            bool v1029;
                                            if (v1027){
                                                bool v1028;
                                                v1028 = v1025 < 2;
                                                v1029 = v1028;
                                            } else {
                                                v1029 = false;
                                            }
                                            bool v1030;
                                            v1030 = v1029 == false;
                                            if (v1030){
                                                assert("Index must be in range." && v1029);
                                            } else {
                                            }
                                            int v1032;
                                            v1032 = v169[v1025];
                                            bool v1035;
                                            if (v1027){
                                                bool v1034;
                                                v1034 = v1025 < 2;
                                                v1035 = v1034;
                                            } else {
                                                v1035 = false;
                                            }
                                            bool v1036;
                                            v1036 = v1035 == false;
                                            if (v1036){
                                                assert("Index must be in range." && v1035);
                                            } else {
                                            }
                                            int v1038;
                                            v1038 = v167[v1025];
                                            int v1040;
                                            v1040 = v1032 + v1038;
                                            v1023[v1025] = v1040;
                                            v1025 += 1 ;
                                        }
                                        int v1041;
                                        v1041 = v167[0];
                                        int v1043; int v1044;
                                        Tuple4 tmp28 = Tuple4{1, v1041};
                                        v1043 = tmp28.v0; v1044 = tmp28.v1;
                                        while (while_method_2(v1043)){
                                            bool v1046;
                                            v1046 = 0 <= v1043;
                                            bool v1048;
                                            if (v1046){
                                                bool v1047;
                                                v1047 = v1043 < 2;
                                                v1048 = v1047;
                                            } else {
                                                v1048 = false;
                                            }
                                            bool v1049;
                                            v1049 = v1048 == false;
                                            if (v1049){
                                                assert("Index must be in range." && v1048);
                                            } else {
                                            }
                                            int v1051;
                                            v1051 = v167[v1043];
                                            bool v1053;
                                            v1053 = v1044 >= v1051;
                                            int v1054;
                                            if (v1053){
                                                v1054 = v1044;
                                            } else {
                                                v1054 = v1051;
                                            }
                                            v1044 = v1054;
                                            v1043 += 1 ;
                                        }
                                        bool v1056;
                                        if (v173){
                                            bool v1055;
                                            v1055 = v171 < 2;
                                            v1056 = v1055;
                                        } else {
                                            v1056 = false;
                                        }
                                        bool v1057;
                                        v1057 = v1056 == false;
                                        if (v1057){
                                            assert("Index must be in range." && v1056);
                                        } else {
                                        }
                                        int v1059;
                                        v1059 = v1023[v171];
                                        bool v1061;
                                        v1061 = v1044 < v1059;
                                        int v1062;
                                        if (v1061){
                                            v1062 = v1044;
                                        } else {
                                            v1062 = v1059;
                                        }
                                        static_array<int,2> v1063;
                                        int v1065;
                                        v1065 = 0;
                                        while (while_method_2(v1065)){
                                            bool v1067;
                                            v1067 = 0 <= v1065;
                                            bool v1069;
                                            if (v1067){
                                                bool v1068;
                                                v1068 = v1065 < 2;
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
                                            v1072 = v167[v1065];
                                            bool v1074;
                                            v1074 = v171 == v1065;
                                            int v1075;
                                            if (v1074){
                                                v1075 = v1062;
                                            } else {
                                                v1075 = v1072;
                                            }
                                            v1063[v1065] = v1075;
                                            v1065 += 1 ;
                                        }
                                        int v1076;
                                        v1076 = v1063[0];
                                        int v1078; int v1079;
                                        Tuple4 tmp29 = Tuple4{1, v1076};
                                        v1078 = tmp29.v0; v1079 = tmp29.v1;
                                        while (while_method_2(v1078)){
                                            bool v1081;
                                            v1081 = 0 <= v1078;
                                            bool v1083;
                                            if (v1081){
                                                bool v1082;
                                                v1082 = v1078 < 2;
                                                v1083 = v1082;
                                            } else {
                                                v1083 = false;
                                            }
                                            bool v1084;
                                            v1084 = v1083 == false;
                                            if (v1084){
                                                assert("Index must be in range." && v1083);
                                            } else {
                                            }
                                            int v1086;
                                            v1086 = v1063[v1078];
                                            int v1088;
                                            v1088 = v1079 + v1086;
                                            v1079 = v1088;
                                            v1078 += 1 ;
                                        }
                                        static_array<int,2> v1089;
                                        int v1091;
                                        v1091 = 0;
                                        while (while_method_2(v1091)){
                                            bool v1093;
                                            v1093 = 0 <= v1091;
                                            bool v1095;
                                            if (v1093){
                                                bool v1094;
                                                v1094 = v1091 < 2;
                                                v1095 = v1094;
                                            } else {
                                                v1095 = false;
                                            }
                                            bool v1096;
                                            v1096 = v1095 == false;
                                            if (v1096){
                                                assert("Index must be in range." && v1095);
                                            } else {
                                            }
                                            int v1098;
                                            v1098 = v1023[v1091];
                                            bool v1101;
                                            if (v1093){
                                                bool v1100;
                                                v1100 = v1091 < 2;
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
                                            int v1104;
                                            v1104 = v1063[v1091];
                                            int v1106;
                                            v1106 = v1098 - v1104;
                                            v1089[v1091] = v1106;
                                            v1091 += 1 ;
                                        }
                                        int v1107;
                                        v1107 = v1021 * v1079;
                                        int v1108;
                                        v1108 = v1107 / v1022;
                                        bool v1109;
                                        v1109 = v165 >= v1108;
                                        int v1110;
                                        if (v1109){
                                            v1110 = v165;
                                        } else {
                                            v1110 = v1108;
                                        }
                                        bool v1112;
                                        if (v173){
                                            bool v1111;
                                            v1111 = v171 < 2;
                                            v1112 = v1111;
                                        } else {
                                            v1112 = false;
                                        }
                                        bool v1113;
                                        v1113 = v1112 == false;
                                        if (v1113){
                                            assert("Index must be in range." && v1112);
                                        } else {
                                        }
                                        int v1115;
                                        v1115 = v1089[v171];
                                        bool v1117;
                                        v1117 = v1110 >= v1115;
                                        if (v1117){
                                            v1281 = Union2{Union2_0{}};
                                        } else {
                                            v1281 = Union2{Union2_3{v1110}};
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
                                curandStatePhilox4_32_10_t & v1124 = v2.v5;
                                curandStatePhilox4_32_10_t & v1125 = v1124;
                                static_array<int,2> v1126;
                                int v1128;
                                v1128 = 0;
                                while (while_method_2(v1128)){
                                    bool v1130;
                                    v1130 = 0 <= v1128;
                                    bool v1132;
                                    if (v1130){
                                        bool v1131;
                                        v1131 = v1128 < 2;
                                        v1132 = v1131;
                                    } else {
                                        v1132 = false;
                                    }
                                    bool v1133;
                                    v1133 = v1132 == false;
                                    if (v1133){
                                        assert("Index must be in range." && v1132);
                                    } else {
                                    }
                                    int v1135;
                                    v1135 = v169[v1128];
                                    bool v1138;
                                    if (v1130){
                                        bool v1137;
                                        v1137 = v1128 < 2;
                                        v1138 = v1137;
                                    } else {
                                        v1138 = false;
                                    }
                                    bool v1139;
                                    v1139 = v1138 == false;
                                    if (v1139){
                                        assert("Index must be in range." && v1138);
                                    } else {
                                    }
                                    int v1141;
                                    v1141 = v167[v1128];
                                    int v1143;
                                    v1143 = v1135 + v1141;
                                    v1126[v1128] = v1143;
                                    v1128 += 1 ;
                                }
                                int v1144;
                                v1144 = v167[0];
                                int v1146; int v1147;
                                Tuple4 tmp30 = Tuple4{1, v1144};
                                v1146 = tmp30.v0; v1147 = tmp30.v1;
                                while (while_method_2(v1146)){
                                    bool v1149;
                                    v1149 = 0 <= v1146;
                                    bool v1151;
                                    if (v1149){
                                        bool v1150;
                                        v1150 = v1146 < 2;
                                        v1151 = v1150;
                                    } else {
                                        v1151 = false;
                                    }
                                    bool v1152;
                                    v1152 = v1151 == false;
                                    if (v1152){
                                        assert("Index must be in range." && v1151);
                                    } else {
                                    }
                                    int v1154;
                                    v1154 = v167[v1146];
                                    bool v1156;
                                    v1156 = v1147 >= v1154;
                                    int v1157;
                                    if (v1156){
                                        v1157 = v1147;
                                    } else {
                                        v1157 = v1154;
                                    }
                                    v1147 = v1157;
                                    v1146 += 1 ;
                                }
                                bool v1159;
                                if (v173){
                                    bool v1158;
                                    v1158 = v171 < 2;
                                    v1159 = v1158;
                                } else {
                                    v1159 = false;
                                }
                                bool v1160;
                                v1160 = v1159 == false;
                                if (v1160){
                                    assert("Index must be in range." && v1159);
                                } else {
                                }
                                int v1162;
                                v1162 = v1126[v171];
                                bool v1164;
                                v1164 = v1147 < v1162;
                                int v1165;
                                if (v1164){
                                    v1165 = v1147;
                                } else {
                                    v1165 = v1162;
                                }
                                static_array<int,2> v1166;
                                int v1168;
                                v1168 = 0;
                                while (while_method_2(v1168)){
                                    bool v1170;
                                    v1170 = 0 <= v1168;
                                    bool v1172;
                                    if (v1170){
                                        bool v1171;
                                        v1171 = v1168 < 2;
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
                                    v1175 = v167[v1168];
                                    bool v1177;
                                    v1177 = v171 == v1168;
                                    int v1178;
                                    if (v1177){
                                        v1178 = v1165;
                                    } else {
                                        v1178 = v1175;
                                    }
                                    v1166[v1168] = v1178;
                                    v1168 += 1 ;
                                }
                                int v1179;
                                v1179 = v1166[0];
                                int v1181; int v1182;
                                Tuple4 tmp31 = Tuple4{1, v1179};
                                v1181 = tmp31.v0; v1182 = tmp31.v1;
                                while (while_method_2(v1181)){
                                    bool v1184;
                                    v1184 = 0 <= v1181;
                                    bool v1186;
                                    if (v1184){
                                        bool v1185;
                                        v1185 = v1181 < 2;
                                        v1186 = v1185;
                                    } else {
                                        v1186 = false;
                                    }
                                    bool v1187;
                                    v1187 = v1186 == false;
                                    if (v1187){
                                        assert("Index must be in range." && v1186);
                                    } else {
                                    }
                                    int v1189;
                                    v1189 = v1166[v1181];
                                    int v1191;
                                    v1191 = v1182 + v1189;
                                    v1182 = v1191;
                                    v1181 += 1 ;
                                }
                                static_array<int,2> v1192;
                                int v1194;
                                v1194 = 0;
                                while (while_method_2(v1194)){
                                    bool v1196;
                                    v1196 = 0 <= v1194;
                                    bool v1198;
                                    if (v1196){
                                        bool v1197;
                                        v1197 = v1194 < 2;
                                        v1198 = v1197;
                                    } else {
                                        v1198 = false;
                                    }
                                    bool v1199;
                                    v1199 = v1198 == false;
                                    if (v1199){
                                        assert("Index must be in range." && v1198);
                                    } else {
                                    }
                                    int v1201;
                                    v1201 = v1126[v1194];
                                    bool v1204;
                                    if (v1196){
                                        bool v1203;
                                        v1203 = v1194 < 2;
                                        v1204 = v1203;
                                    } else {
                                        v1204 = false;
                                    }
                                    bool v1205;
                                    v1205 = v1204 == false;
                                    if (v1205){
                                        assert("Index must be in range." && v1204);
                                    } else {
                                    }
                                    int v1207;
                                    v1207 = v1166[v1194];
                                    int v1209;
                                    v1209 = v1201 - v1207;
                                    v1192[v1194] = v1209;
                                    v1194 += 1 ;
                                }
                                bool v1211;
                                if (v173){
                                    bool v1210;
                                    v1210 = v171 < 2;
                                    v1211 = v1210;
                                } else {
                                    v1211 = false;
                                }
                                bool v1212;
                                v1212 = v1211 == false;
                                if (v1212){
                                    assert("Index must be in range." && v1211);
                                } else {
                                }
                                int v1214;
                                v1214 = v167[v171];
                                bool v1216;
                                v1216 = v1214 < v1147;
                                float v1217;
                                if (v1216){
                                    v1217 = 1.0f;
                                } else {
                                    v1217 = 0.0f;
                                }
                                int v1218;
                                v1218 = v1182 / 3;
                                bool v1219;
                                v1219 = v165 <= v1218;
                                bool v1227;
                                if (v1219){
                                    bool v1221;
                                    if (v173){
                                        bool v1220;
                                        v1220 = v171 < 2;
                                        v1221 = v1220;
                                    } else {
                                        v1221 = false;
                                    }
                                    bool v1222;
                                    v1222 = v1221 == false;
                                    if (v1222){
                                        assert("Index must be in range." && v1221);
                                    } else {
                                    }
                                    int v1224;
                                    v1224 = v1192[v171];
                                    bool v1226;
                                    v1226 = v1218 < v1224;
                                    v1227 = v1226;
                                } else {
                                    v1227 = false;
                                }
                                float v1228;
                                if (v1227){
                                    v1228 = 1.0f;
                                } else {
                                    v1228 = 0.0f;
                                }
                                int v1229;
                                v1229 = v1182 / 2;
                                bool v1230;
                                v1230 = v165 <= v1229;
                                bool v1238;
                                if (v1230){
                                    bool v1232;
                                    if (v173){
                                        bool v1231;
                                        v1231 = v171 < 2;
                                        v1232 = v1231;
                                    } else {
                                        v1232 = false;
                                    }
                                    bool v1233;
                                    v1233 = v1232 == false;
                                    if (v1233){
                                        assert("Index must be in range." && v1232);
                                    } else {
                                    }
                                    int v1235;
                                    v1235 = v1192[v171];
                                    bool v1237;
                                    v1237 = v1229 < v1235;
                                    v1238 = v1237;
                                } else {
                                    v1238 = false;
                                }
                                float v1239;
                                if (v1238){
                                    v1239 = 1.0f;
                                } else {
                                    v1239 = 0.0f;
                                }
                                bool v1240;
                                v1240 = v165 <= v1182;
                                bool v1248;
                                if (v1240){
                                    bool v1242;
                                    if (v173){
                                        bool v1241;
                                        v1241 = v171 < 2;
                                        v1242 = v1241;
                                    } else {
                                        v1242 = false;
                                    }
                                    bool v1243;
                                    v1243 = v1242 == false;
                                    if (v1243){
                                        assert("Index must be in range." && v1242);
                                    } else {
                                    }
                                    int v1245;
                                    v1245 = v1192[v171];
                                    bool v1247;
                                    v1247 = v1182 < v1245;
                                    v1248 = v1247;
                                } else {
                                    v1248 = false;
                                }
                                float v1249;
                                if (v1248){
                                    v1249 = 1.0f;
                                } else {
                                    v1249 = 0.0f;
                                }
                                static_array<Tuple12,6> v1250;
                                Union2 v1252;
                                v1252 = Union2{Union2_2{}};
                                v1250[0] = Tuple12{v1252, v1217};
                                Union2 v1254;
                                v1254 = Union2{Union2_1{}};
                                v1250[1] = Tuple12{v1254, 4.0f};
                                Union2 v1256;
                                v1256 = Union2{Union2_3{v1218}};
                                v1250[2] = Tuple12{v1256, v1228};
                                Union2 v1258;
                                v1258 = Union2{Union2_3{v1229}};
                                v1250[3] = Tuple12{v1258, v1239};
                                Union2 v1260;
                                v1260 = Union2{Union2_3{v1182}};
                                v1250[4] = Tuple12{v1260, v1249};
                                Union2 v1262;
                                v1262 = Union2{Union2_0{}};
                                v1250[5] = Tuple12{v1262, 1.0f};
                                Union2 v1264;
                                v1264 = sample_discrete_16(v1250, v1125);
                                int v1265;
                                v1265 = sizeof(Union2);
                                unsigned long long v1266;
                                v1266 = (unsigned long long)v1265;
                                bool v1267;
                                v1267 = v1266 <= 98304ull;
                                bool v1268;
                                v1268 = v1267 == false;
                                if (v1268){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v1267);
                                } else {
                                }
                                extern __shared__ unsigned char v1270[];
                                bool v1271;
                                v1271 = v1266 <= v1266;
                                bool v1272;
                                v1272 = v1271 == false;
                                if (v1272){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v1271);
                                } else {
                                }
                                Union2 * v1274;
                                v1274 = reinterpret_cast<Union2 *>(&v1270[0ull]);
                                int v1276;
                                v1276 = threadIdx.x;
                                bool v1277;
                                v1277 = v1276 == 0;
                                if (v1277){
                                    v1274[0] = v1264;
                                } else {
                                }
                                __syncthreads();
                                Union2 v1278;
                                v1278 = v1274[0];
                                __syncthreads();
                                v1281 = v1278;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        Union1 v1282;
                        v1282 = Union1{Union1_2{v171, v1281}};
                        v17.push(v1282);
                        v1433 = Union6{Union6_1{v165, v166, v167, v168, v169, v170, v1281}};
                        break;
                    }
                    case 5: { // G_Round'
                        int v1284 = v22.case5.v0; static_array<static_array<unsigned char,2>,2> v1285 = v22.case5.v1; static_array<int,2> v1286 = v22.case5.v2; int v1287 = v22.case5.v3; static_array<int,2> v1288 = v22.case5.v4; Union4 v1289 = v22.case5.v5; Union2 v1290 = v22.case5.v6;
                        int v1291;
                        v1291 = v1287 % 2;
                        Union1 v1292;
                        v1292 = Union1{Union1_2{v1291, v1290}};
                        v17.push(v1292);
                        v1433 = Union6{Union6_1{v1284, v1285, v1286, v1287, v1288, v1289, v1290}};
                        break;
                    }
                    case 6: { // G_Showdown
                        int v46 = v22.case6.v0; static_array<static_array<unsigned char,2>,2> v47 = v22.case6.v1; static_array<int,2> v48 = v22.case6.v2; int v49 = v22.case6.v3; static_array<int,2> v50 = v22.case6.v4; Union4 v51 = v22.case6.v5;
                        static_array<unsigned char,5> v54;
                        switch (v51.tag) {
                            case 2: { // River
                                static_array<unsigned char,5> v52 = v51.case2.v0;
                                v54 = v52;
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in showdown.");
                                __trap();
                            }
                        }
                        static_array<unsigned char,2> v55;
                        v55 = v47[0];
                        static_array<unsigned char,7> v57;
                        int v59;
                        v59 = 0;
                        while (while_method_2(v59)){
                            bool v61;
                            v61 = 0 <= v59;
                            bool v63;
                            if (v61){
                                bool v62;
                                v62 = v59 < 2;
                                v63 = v62;
                            } else {
                                v63 = false;
                            }
                            bool v64;
                            v64 = v63 == false;
                            if (v64){
                                assert("Index must be in range." && v63);
                            } else {
                            }
                            unsigned char v66;
                            v66 = v55[v59];
                            v57[v59] = v66;
                            v59 += 1 ;
                        }
                        int v68;
                        v68 = 0;
                        while (while_method_5(v68)){
                            bool v70;
                            v70 = 0 <= v68;
                            bool v72;
                            if (v70){
                                bool v71;
                                v71 = v68 < 5;
                                v72 = v71;
                            } else {
                                v72 = false;
                            }
                            bool v73;
                            v73 = v72 == false;
                            if (v73){
                                assert("Index must be in range." && v72);
                            } else {
                            }
                            unsigned char v75;
                            v75 = v54[v68];
                            int v77;
                            v77 = 2 + v68;
                            v57[v77] = v75;
                            v68 += 1 ;
                        }
                        static_array<unsigned char,5> v78; char v79;
                        Tuple0 tmp58 = score_20(v57);
                        v78 = tmp58.v0; v79 = tmp58.v1;
                        static_array<unsigned char,2> v80;
                        v80 = v47[1];
                        static_array<unsigned char,7> v82;
                        int v84;
                        v84 = 0;
                        while (while_method_2(v84)){
                            bool v86;
                            v86 = 0 <= v84;
                            bool v88;
                            if (v86){
                                bool v87;
                                v87 = v84 < 2;
                                v88 = v87;
                            } else {
                                v88 = false;
                            }
                            bool v89;
                            v89 = v88 == false;
                            if (v89){
                                assert("Index must be in range." && v88);
                            } else {
                            }
                            unsigned char v91;
                            v91 = v80[v84];
                            v82[v84] = v91;
                            v84 += 1 ;
                        }
                        int v93;
                        v93 = 0;
                        while (while_method_5(v93)){
                            bool v95;
                            v95 = 0 <= v93;
                            bool v97;
                            if (v95){
                                bool v96;
                                v96 = v93 < 5;
                                v97 = v96;
                            } else {
                                v97 = false;
                            }
                            bool v98;
                            v98 = v97 == false;
                            if (v98){
                                assert("Index must be in range." && v97);
                            } else {
                            }
                            unsigned char v100;
                            v100 = v54[v93];
                            int v102;
                            v102 = 2 + v93;
                            v82[v102] = v100;
                            v93 += 1 ;
                        }
                        static_array<unsigned char,5> v103; char v104;
                        Tuple0 tmp59 = score_20(v82);
                        v103 = tmp59.v0; v104 = tmp59.v1;
                        int v105;
                        v105 = v49 % 2;
                        bool v106;
                        v106 = 0 <= v105;
                        bool v108;
                        if (v106){
                            bool v107;
                            v107 = v105 < 2;
                            v108 = v107;
                        } else {
                            v108 = false;
                        }
                        bool v109;
                        v109 = v108 == false;
                        if (v109){
                            assert("Index must be in range." && v108);
                        } else {
                        }
                        int v111;
                        v111 = v48[v105];
                        bool v113;
                        v113 = v79 < v104;
                        Union10 v119;
                        if (v113){
                            v119 = Union10{Union10_2{}};
                        } else {
                            bool v115;
                            v115 = v79 > v104;
                            if (v115){
                                v119 = Union10{Union10_1{}};
                            } else {
                                v119 = Union10{Union10_0{}};
                            }
                        }
                        Union10 v147;
                        switch (v119.tag) {
                            case 0: { // Eq
                                Union10 v120;
                                v120 = Union10{Union10_0{}};
                                int v121;
                                v121 = 0;
                                while (while_method_5(v121)){
                                    bool v123;
                                    v123 = 0 <= v121;
                                    bool v125;
                                    if (v123){
                                        bool v124;
                                        v124 = v121 < 5;
                                        v125 = v124;
                                    } else {
                                        v125 = false;
                                    }
                                    bool v126;
                                    v126 = v125 == false;
                                    if (v126){
                                        assert("Index must be in range." && v125);
                                    } else {
                                    }
                                    unsigned char v128;
                                    v128 = v78[v121];
                                    bool v131;
                                    if (v123){
                                        bool v130;
                                        v130 = v121 < 5;
                                        v131 = v130;
                                    } else {
                                        v131 = false;
                                    }
                                    bool v132;
                                    v132 = v131 == false;
                                    if (v132){
                                        assert("Index must be in range." && v131);
                                    } else {
                                    }
                                    unsigned char v134;
                                    v134 = v103[v121];
                                    unsigned char v136;
                                    v136 = v128 / 4u;
                                    unsigned char v137;
                                    v137 = v134 / 4u;
                                    bool v138;
                                    v138 = v136 < v137;
                                    Union10 v144;
                                    if (v138){
                                        v144 = Union10{Union10_2{}};
                                    } else {
                                        bool v140;
                                        v140 = v136 > v137;
                                        if (v140){
                                            v144 = Union10{Union10_1{}};
                                        } else {
                                            v144 = Union10{Union10_0{}};
                                        }
                                    }
                                    bool v145;
                                    switch (v144.tag) {
                                        case 0: { // Eq
                                            v145 = true;
                                            break;
                                        }
                                        default: {
                                            v145 = false;
                                        }
                                    }
                                    bool v146;
                                    v146 = v145 == false;
                                    if (v146){
                                        v120 = v144;
                                        break;
                                    } else {
                                    }
                                    v121 += 1 ;
                                }
                                v147 = v120;
                                break;
                            }
                            default: {
                                v147 = v119;
                            }
                        }
                        int v152; int v153;
                        switch (v147.tag) {
                            case 0: { // Eq
                                v152 = 0; v153 = -1;
                                break;
                            }
                            case 1: { // Gt
                                v152 = v111; v153 = 0;
                                break;
                            }
                            case 2: { // Lt
                                v152 = v111; v153 = 1;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        int v154;
                        v154 = -v153;
                        bool v155;
                        v155 = v153 >= v154;
                        int v156;
                        if (v155){
                            v156 = v153;
                        } else {
                            v156 = v154;
                        }
                        float v157;
                        v157 = (float)v152;
                        static_array<float,2> & v158 = v2.v4;
                        v158[v156] = v157;
                        int v159;
                        v159 = v156 ^ 1;
                        float v160;
                        v160 = -v157;
                        v158[v159] = v160;
                        static_array<Tuple0,2> v161;
                        v161[0] = Tuple0{v78, v79};
                        v161[1] = Tuple0{v103, v104};
                        Union1 v163;
                        v163 = Union1{Union1_4{v152, v161, v153}};
                        v17.push(v163);
                        v1433 = Union6{Union6_0{}};
                        break;
                    }
                    case 7: { // G_Turn
                        int v1313 = v22.case7.v0; static_array<static_array<unsigned char,2>,2> v1314 = v22.case7.v1; static_array<int,2> v1315 = v22.case7.v2; int v1316 = v22.case7.v3; static_array<int,2> v1317 = v22.case7.v4; Union4 v1318 = v22.case7.v5;
                        curandStatePhilox4_32_10_t & v1319 = v2.v5;
                        curandStatePhilox4_32_10_t & v1320 = v1319;
                        static_array<unsigned char,1> v1321; unsigned long long v1322;
                        Tuple6 tmp60 = draw_cards_9(v1320, v18);
                        v1321 = tmp60.v0; v1322 = tmp60.v1;
                        v2.v0 = v1322;
                        static_array_list<unsigned char,5> v1323;
                        v1323 = get_community_cards_10(v1318, v1321);
                        Union1 v1324;
                        v1324 = Union1{Union1_0{v1323}};
                        v17.push(v1324);
                        Union4 v1349;
                        switch (v1318.tag) {
                            case 0: { // Flop
                                static_array<unsigned char,3> v1325 = v1318.case0.v0;
                                static_array<unsigned char,4> v1326;
                                int v1328;
                                v1328 = 0;
                                while (while_method_4(v1328)){
                                    bool v1330;
                                    v1330 = 0 <= v1328;
                                    bool v1332;
                                    if (v1330){
                                        bool v1331;
                                        v1331 = v1328 < 3;
                                        v1332 = v1331;
                                    } else {
                                        v1332 = false;
                                    }
                                    bool v1333;
                                    v1333 = v1332 == false;
                                    if (v1333){
                                        assert("Index must be in range." && v1332);
                                    } else {
                                    }
                                    unsigned char v1335;
                                    v1335 = v1325[v1328];
                                    v1326[v1328] = v1335;
                                    v1328 += 1 ;
                                }
                                int v1337;
                                v1337 = 0;
                                while (while_method_6(v1337)){
                                    bool v1339;
                                    v1339 = 0 <= v1337;
                                    bool v1341;
                                    if (v1339){
                                        bool v1340;
                                        v1340 = v1337 < 1;
                                        v1341 = v1340;
                                    } else {
                                        v1341 = false;
                                    }
                                    bool v1342;
                                    v1342 = v1341 == false;
                                    if (v1342){
                                        assert("Index must be in range." && v1341);
                                    } else {
                                    }
                                    unsigned char v1344;
                                    v1344 = v1321[v1337];
                                    int v1346;
                                    v1346 = 3 + v1337;
                                    v1326[v1346] = v1344;
                                    v1337 += 1 ;
                                }
                                v1349 = Union4{Union4_3{v1326}};
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in turn.");
                                __trap();
                            }
                        }
                        int v1350;
                        v1350 = 2;
                        int v1351;
                        v1351 = 0;
                        Union3 v1352;
                        v1352 = try_round_5(v1350, v1314, v1315, v1351, v1317, v1349);
                        v1433 = Union6{Union6_2{v1352}};
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false); __trap();
                    }
                }
                switch (v1433.tag) {
                    case 0: { // T_none
                        v1782 = Union5{Union5_0{}};
                        break;
                    }
                    case 1: { // T_round
                        int v1437 = v1433.case1.v0; static_array<static_array<unsigned char,2>,2> v1438 = v1433.case1.v1; static_array<int,2> v1439 = v1433.case1.v2; int v1440 = v1433.case1.v3; static_array<int,2> v1441 = v1433.case1.v4; Union4 v1442 = v1433.case1.v5; Union2 v1443 = v1433.case1.v6;
                        int v1444;
                        v1444 = v1440 % 2;
                        Union3 v1775;
                        switch (v1443.tag) {
                            case 0: { // A_All_In
                                static_array<int,2> v1650;
                                int v1652;
                                v1652 = 0;
                                while (while_method_2(v1652)){
                                    bool v1654;
                                    v1654 = 0 <= v1652;
                                    bool v1656;
                                    if (v1654){
                                        bool v1655;
                                        v1655 = v1652 < 2;
                                        v1656 = v1655;
                                    } else {
                                        v1656 = false;
                                    }
                                    bool v1657;
                                    v1657 = v1656 == false;
                                    if (v1657){
                                        assert("Index must be in range." && v1656);
                                    } else {
                                    }
                                    int v1659;
                                    v1659 = v1441[v1652];
                                    bool v1662;
                                    if (v1654){
                                        bool v1661;
                                        v1661 = v1652 < 2;
                                        v1662 = v1661;
                                    } else {
                                        v1662 = false;
                                    }
                                    bool v1663;
                                    v1663 = v1662 == false;
                                    if (v1663){
                                        assert("Index must be in range." && v1662);
                                    } else {
                                    }
                                    int v1665;
                                    v1665 = v1439[v1652];
                                    int v1667;
                                    v1667 = v1659 + v1665;
                                    v1650[v1652] = v1667;
                                    v1652 += 1 ;
                                }
                                int v1668;
                                v1668 = v1439[0];
                                int v1670; int v1671;
                                Tuple4 tmp61 = Tuple4{1, v1668};
                                v1670 = tmp61.v0; v1671 = tmp61.v1;
                                while (while_method_2(v1670)){
                                    bool v1673;
                                    v1673 = 0 <= v1670;
                                    bool v1675;
                                    if (v1673){
                                        bool v1674;
                                        v1674 = v1670 < 2;
                                        v1675 = v1674;
                                    } else {
                                        v1675 = false;
                                    }
                                    bool v1676;
                                    v1676 = v1675 == false;
                                    if (v1676){
                                        assert("Index must be in range." && v1675);
                                    } else {
                                    }
                                    int v1678;
                                    v1678 = v1439[v1670];
                                    bool v1680;
                                    v1680 = v1671 >= v1678;
                                    int v1681;
                                    if (v1680){
                                        v1681 = v1671;
                                    } else {
                                        v1681 = v1678;
                                    }
                                    v1671 = v1681;
                                    v1670 += 1 ;
                                }
                                bool v1682;
                                v1682 = 0 <= v1444;
                                bool v1684;
                                if (v1682){
                                    bool v1683;
                                    v1683 = v1444 < 2;
                                    v1684 = v1683;
                                } else {
                                    v1684 = false;
                                }
                                bool v1685;
                                v1685 = v1684 == false;
                                if (v1685){
                                    assert("Index must be in range." && v1684);
                                } else {
                                }
                                int v1687;
                                v1687 = v1650[v1444];
                                bool v1689;
                                v1689 = v1671 < v1687;
                                int v1690;
                                if (v1689){
                                    v1690 = v1671;
                                } else {
                                    v1690 = v1687;
                                }
                                static_array<int,2> v1691;
                                int v1693;
                                v1693 = 0;
                                while (while_method_2(v1693)){
                                    bool v1695;
                                    v1695 = 0 <= v1693;
                                    bool v1697;
                                    if (v1695){
                                        bool v1696;
                                        v1696 = v1693 < 2;
                                        v1697 = v1696;
                                    } else {
                                        v1697 = false;
                                    }
                                    bool v1698;
                                    v1698 = v1697 == false;
                                    if (v1698){
                                        assert("Index must be in range." && v1697);
                                    } else {
                                    }
                                    int v1700;
                                    v1700 = v1439[v1693];
                                    bool v1702;
                                    v1702 = v1444 == v1693;
                                    int v1703;
                                    if (v1702){
                                        v1703 = v1690;
                                    } else {
                                        v1703 = v1700;
                                    }
                                    v1691[v1693] = v1703;
                                    v1693 += 1 ;
                                }
                                static_array<int,2> v1704;
                                int v1706;
                                v1706 = 0;
                                while (while_method_2(v1706)){
                                    bool v1708;
                                    v1708 = 0 <= v1706;
                                    bool v1710;
                                    if (v1708){
                                        bool v1709;
                                        v1709 = v1706 < 2;
                                        v1710 = v1709;
                                    } else {
                                        v1710 = false;
                                    }
                                    bool v1711;
                                    v1711 = v1710 == false;
                                    if (v1711){
                                        assert("Index must be in range." && v1710);
                                    } else {
                                    }
                                    int v1713;
                                    v1713 = v1650[v1706];
                                    bool v1716;
                                    if (v1708){
                                        bool v1715;
                                        v1715 = v1706 < 2;
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
                                    v1719 = v1691[v1706];
                                    int v1721;
                                    v1721 = v1713 - v1719;
                                    v1704[v1706] = v1721;
                                    v1706 += 1 ;
                                }
                                bool v1723;
                                if (v1682){
                                    bool v1722;
                                    v1722 = v1444 < 2;
                                    v1723 = v1722;
                                } else {
                                    v1723 = false;
                                }
                                bool v1724;
                                v1724 = v1723 == false;
                                if (v1724){
                                    assert("Index must be in range." && v1723);
                                } else {
                                }
                                int v1726;
                                v1726 = v1704[v1444];
                                int v1728;
                                v1728 = v1671 + v1726;
                                bool v1730;
                                if (v1682){
                                    bool v1729;
                                    v1729 = v1444 < 2;
                                    v1730 = v1729;
                                } else {
                                    v1730 = false;
                                }
                                bool v1731;
                                v1731 = v1730 == false;
                                if (v1731){
                                    assert("Index must be in range." && v1730);
                                } else {
                                }
                                int v1733;
                                v1733 = v1650[v1444];
                                bool v1735;
                                v1735 = v1728 < v1733;
                                int v1736;
                                if (v1735){
                                    v1736 = v1728;
                                } else {
                                    v1736 = v1733;
                                }
                                static_array<int,2> v1737;
                                int v1739;
                                v1739 = 0;
                                while (while_method_2(v1739)){
                                    bool v1741;
                                    v1741 = 0 <= v1739;
                                    bool v1743;
                                    if (v1741){
                                        bool v1742;
                                        v1742 = v1739 < 2;
                                        v1743 = v1742;
                                    } else {
                                        v1743 = false;
                                    }
                                    bool v1744;
                                    v1744 = v1743 == false;
                                    if (v1744){
                                        assert("Index must be in range." && v1743);
                                    } else {
                                    }
                                    int v1746;
                                    v1746 = v1439[v1739];
                                    bool v1748;
                                    v1748 = v1444 == v1739;
                                    int v1749;
                                    if (v1748){
                                        v1749 = v1736;
                                    } else {
                                        v1749 = v1746;
                                    }
                                    v1737[v1739] = v1749;
                                    v1739 += 1 ;
                                }
                                static_array<int,2> v1750;
                                int v1752;
                                v1752 = 0;
                                while (while_method_2(v1752)){
                                    bool v1754;
                                    v1754 = 0 <= v1752;
                                    bool v1756;
                                    if (v1754){
                                        bool v1755;
                                        v1755 = v1752 < 2;
                                        v1756 = v1755;
                                    } else {
                                        v1756 = false;
                                    }
                                    bool v1757;
                                    v1757 = v1756 == false;
                                    if (v1757){
                                        assert("Index must be in range." && v1756);
                                    } else {
                                    }
                                    int v1759;
                                    v1759 = v1650[v1752];
                                    bool v1762;
                                    if (v1754){
                                        bool v1761;
                                        v1761 = v1752 < 2;
                                        v1762 = v1761;
                                    } else {
                                        v1762 = false;
                                    }
                                    bool v1763;
                                    v1763 = v1762 == false;
                                    if (v1763){
                                        assert("Index must be in range." && v1762);
                                    } else {
                                    }
                                    int v1765;
                                    v1765 = v1737[v1752];
                                    int v1767;
                                    v1767 = v1759 - v1765;
                                    v1750[v1752] = v1767;
                                    v1752 += 1 ;
                                }
                                bool v1768;
                                v1768 = v1726 >= v1437;
                                int v1769;
                                if (v1768){
                                    v1769 = v1726;
                                } else {
                                    v1769 = v1437;
                                }
                                int v1770;
                                v1770 = v1440 + 1;
                                v1775 = try_round_5(v1769, v1438, v1737, v1770, v1750, v1442);
                                break;
                            }
                            case 1: { // A_Call
                                static_array<int,2> v1446;
                                int v1448;
                                v1448 = 0;
                                while (while_method_2(v1448)){
                                    bool v1450;
                                    v1450 = 0 <= v1448;
                                    bool v1452;
                                    if (v1450){
                                        bool v1451;
                                        v1451 = v1448 < 2;
                                        v1452 = v1451;
                                    } else {
                                        v1452 = false;
                                    }
                                    bool v1453;
                                    v1453 = v1452 == false;
                                    if (v1453){
                                        assert("Index must be in range." && v1452);
                                    } else {
                                    }
                                    int v1455;
                                    v1455 = v1441[v1448];
                                    bool v1458;
                                    if (v1450){
                                        bool v1457;
                                        v1457 = v1448 < 2;
                                        v1458 = v1457;
                                    } else {
                                        v1458 = false;
                                    }
                                    bool v1459;
                                    v1459 = v1458 == false;
                                    if (v1459){
                                        assert("Index must be in range." && v1458);
                                    } else {
                                    }
                                    int v1461;
                                    v1461 = v1439[v1448];
                                    int v1463;
                                    v1463 = v1455 + v1461;
                                    v1446[v1448] = v1463;
                                    v1448 += 1 ;
                                }
                                int v1464;
                                v1464 = v1439[0];
                                int v1466; int v1467;
                                Tuple4 tmp62 = Tuple4{1, v1464};
                                v1466 = tmp62.v0; v1467 = tmp62.v1;
                                while (while_method_2(v1466)){
                                    bool v1469;
                                    v1469 = 0 <= v1466;
                                    bool v1471;
                                    if (v1469){
                                        bool v1470;
                                        v1470 = v1466 < 2;
                                        v1471 = v1470;
                                    } else {
                                        v1471 = false;
                                    }
                                    bool v1472;
                                    v1472 = v1471 == false;
                                    if (v1472){
                                        assert("Index must be in range." && v1471);
                                    } else {
                                    }
                                    int v1474;
                                    v1474 = v1439[v1466];
                                    bool v1476;
                                    v1476 = v1467 >= v1474;
                                    int v1477;
                                    if (v1476){
                                        v1477 = v1467;
                                    } else {
                                        v1477 = v1474;
                                    }
                                    v1467 = v1477;
                                    v1466 += 1 ;
                                }
                                bool v1478;
                                v1478 = 0 <= v1444;
                                bool v1480;
                                if (v1478){
                                    bool v1479;
                                    v1479 = v1444 < 2;
                                    v1480 = v1479;
                                } else {
                                    v1480 = false;
                                }
                                bool v1481;
                                v1481 = v1480 == false;
                                if (v1481){
                                    assert("Index must be in range." && v1480);
                                } else {
                                }
                                int v1483;
                                v1483 = v1446[v1444];
                                bool v1485;
                                v1485 = v1467 < v1483;
                                int v1486;
                                if (v1485){
                                    v1486 = v1467;
                                } else {
                                    v1486 = v1483;
                                }
                                static_array<int,2> v1487;
                                int v1489;
                                v1489 = 0;
                                while (while_method_2(v1489)){
                                    bool v1491;
                                    v1491 = 0 <= v1489;
                                    bool v1493;
                                    if (v1491){
                                        bool v1492;
                                        v1492 = v1489 < 2;
                                        v1493 = v1492;
                                    } else {
                                        v1493 = false;
                                    }
                                    bool v1494;
                                    v1494 = v1493 == false;
                                    if (v1494){
                                        assert("Index must be in range." && v1493);
                                    } else {
                                    }
                                    int v1496;
                                    v1496 = v1439[v1489];
                                    bool v1498;
                                    v1498 = v1444 == v1489;
                                    int v1499;
                                    if (v1498){
                                        v1499 = v1486;
                                    } else {
                                        v1499 = v1496;
                                    }
                                    v1487[v1489] = v1499;
                                    v1489 += 1 ;
                                }
                                static_array<int,2> v1500;
                                int v1502;
                                v1502 = 0;
                                while (while_method_2(v1502)){
                                    bool v1504;
                                    v1504 = 0 <= v1502;
                                    bool v1506;
                                    if (v1504){
                                        bool v1505;
                                        v1505 = v1502 < 2;
                                        v1506 = v1505;
                                    } else {
                                        v1506 = false;
                                    }
                                    bool v1507;
                                    v1507 = v1506 == false;
                                    if (v1507){
                                        assert("Index must be in range." && v1506);
                                    } else {
                                    }
                                    int v1509;
                                    v1509 = v1446[v1502];
                                    bool v1512;
                                    if (v1504){
                                        bool v1511;
                                        v1511 = v1502 < 2;
                                        v1512 = v1511;
                                    } else {
                                        v1512 = false;
                                    }
                                    bool v1513;
                                    v1513 = v1512 == false;
                                    if (v1513){
                                        assert("Index must be in range." && v1512);
                                    } else {
                                    }
                                    int v1515;
                                    v1515 = v1487[v1502];
                                    int v1517;
                                    v1517 = v1509 - v1515;
                                    v1500[v1502] = v1517;
                                    v1502 += 1 ;
                                }
                                bool v1518;
                                v1518 = v1444 < 2;
                                if (v1518){
                                    int v1519;
                                    v1519 = v1440 + 1;
                                    v1775 = try_round_5(v1437, v1438, v1487, v1519, v1500, v1442);
                                } else {
                                    v1775 = go_next_street_7(v1437, v1438, v1487, v1440, v1500, v1442);
                                }
                                break;
                            }
                            case 2: { // A_Fold
                                v1775 = Union3{Union3_1{v1437, v1438, v1439, v1440, v1441, v1442}};
                                break;
                            }
                            case 3: { // A_Raise
                                int v1523 = v1443.case3.v0;
                                bool v1524;
                                v1524 = v1437 <= v1523;
                                bool v1525;
                                v1525 = v1524 == false;
                                if (v1525){
                                    assert("The raise amount must match the minimum." && v1524);
                                } else {
                                }
                                static_array<int,2> v1527;
                                int v1529;
                                v1529 = 0;
                                while (while_method_2(v1529)){
                                    bool v1531;
                                    v1531 = 0 <= v1529;
                                    bool v1533;
                                    if (v1531){
                                        bool v1532;
                                        v1532 = v1529 < 2;
                                        v1533 = v1532;
                                    } else {
                                        v1533 = false;
                                    }
                                    bool v1534;
                                    v1534 = v1533 == false;
                                    if (v1534){
                                        assert("Index must be in range." && v1533);
                                    } else {
                                    }
                                    int v1536;
                                    v1536 = v1441[v1529];
                                    bool v1539;
                                    if (v1531){
                                        bool v1538;
                                        v1538 = v1529 < 2;
                                        v1539 = v1538;
                                    } else {
                                        v1539 = false;
                                    }
                                    bool v1540;
                                    v1540 = v1539 == false;
                                    if (v1540){
                                        assert("Index must be in range." && v1539);
                                    } else {
                                    }
                                    int v1542;
                                    v1542 = v1439[v1529];
                                    int v1544;
                                    v1544 = v1536 + v1542;
                                    v1527[v1529] = v1544;
                                    v1529 += 1 ;
                                }
                                int v1545;
                                v1545 = v1439[0];
                                int v1547; int v1548;
                                Tuple4 tmp63 = Tuple4{1, v1545};
                                v1547 = tmp63.v0; v1548 = tmp63.v1;
                                while (while_method_2(v1547)){
                                    bool v1550;
                                    v1550 = 0 <= v1547;
                                    bool v1552;
                                    if (v1550){
                                        bool v1551;
                                        v1551 = v1547 < 2;
                                        v1552 = v1551;
                                    } else {
                                        v1552 = false;
                                    }
                                    bool v1553;
                                    v1553 = v1552 == false;
                                    if (v1553){
                                        assert("Index must be in range." && v1552);
                                    } else {
                                    }
                                    int v1555;
                                    v1555 = v1439[v1547];
                                    bool v1557;
                                    v1557 = v1548 >= v1555;
                                    int v1558;
                                    if (v1557){
                                        v1558 = v1548;
                                    } else {
                                        v1558 = v1555;
                                    }
                                    v1548 = v1558;
                                    v1547 += 1 ;
                                }
                                bool v1559;
                                v1559 = 0 <= v1444;
                                bool v1561;
                                if (v1559){
                                    bool v1560;
                                    v1560 = v1444 < 2;
                                    v1561 = v1560;
                                } else {
                                    v1561 = false;
                                }
                                bool v1562;
                                v1562 = v1561 == false;
                                if (v1562){
                                    assert("Index must be in range." && v1561);
                                } else {
                                }
                                int v1564;
                                v1564 = v1527[v1444];
                                bool v1566;
                                v1566 = v1548 < v1564;
                                int v1567;
                                if (v1566){
                                    v1567 = v1548;
                                } else {
                                    v1567 = v1564;
                                }
                                static_array<int,2> v1568;
                                int v1570;
                                v1570 = 0;
                                while (while_method_2(v1570)){
                                    bool v1572;
                                    v1572 = 0 <= v1570;
                                    bool v1574;
                                    if (v1572){
                                        bool v1573;
                                        v1573 = v1570 < 2;
                                        v1574 = v1573;
                                    } else {
                                        v1574 = false;
                                    }
                                    bool v1575;
                                    v1575 = v1574 == false;
                                    if (v1575){
                                        assert("Index must be in range." && v1574);
                                    } else {
                                    }
                                    int v1577;
                                    v1577 = v1439[v1570];
                                    bool v1579;
                                    v1579 = v1444 == v1570;
                                    int v1580;
                                    if (v1579){
                                        v1580 = v1567;
                                    } else {
                                        v1580 = v1577;
                                    }
                                    v1568[v1570] = v1580;
                                    v1570 += 1 ;
                                }
                                static_array<int,2> v1581;
                                int v1583;
                                v1583 = 0;
                                while (while_method_2(v1583)){
                                    bool v1585;
                                    v1585 = 0 <= v1583;
                                    bool v1587;
                                    if (v1585){
                                        bool v1586;
                                        v1586 = v1583 < 2;
                                        v1587 = v1586;
                                    } else {
                                        v1587 = false;
                                    }
                                    bool v1588;
                                    v1588 = v1587 == false;
                                    if (v1588){
                                        assert("Index must be in range." && v1587);
                                    } else {
                                    }
                                    int v1590;
                                    v1590 = v1527[v1583];
                                    bool v1593;
                                    if (v1585){
                                        bool v1592;
                                        v1592 = v1583 < 2;
                                        v1593 = v1592;
                                    } else {
                                        v1593 = false;
                                    }
                                    bool v1594;
                                    v1594 = v1593 == false;
                                    if (v1594){
                                        assert("Index must be in range." && v1593);
                                    } else {
                                    }
                                    int v1596;
                                    v1596 = v1568[v1583];
                                    int v1598;
                                    v1598 = v1590 - v1596;
                                    v1581[v1583] = v1598;
                                    v1583 += 1 ;
                                }
                                bool v1600;
                                if (v1559){
                                    bool v1599;
                                    v1599 = v1444 < 2;
                                    v1600 = v1599;
                                } else {
                                    v1600 = false;
                                }
                                bool v1601;
                                v1601 = v1600 == false;
                                if (v1601){
                                    assert("Index must be in range." && v1600);
                                } else {
                                }
                                int v1603;
                                v1603 = v1581[v1444];
                                bool v1605;
                                v1605 = v1523 < v1603;
                                bool v1606;
                                v1606 = v1605 == false;
                                if (v1606){
                                    assert("The raise amount must be less than the stack size after calling." && v1605);
                                } else {
                                }
                                int v1608;
                                v1608 = v1548 + v1523;
                                bool v1610;
                                if (v1559){
                                    bool v1609;
                                    v1609 = v1444 < 2;
                                    v1610 = v1609;
                                } else {
                                    v1610 = false;
                                }
                                bool v1611;
                                v1611 = v1610 == false;
                                if (v1611){
                                    assert("Index must be in range." && v1610);
                                } else {
                                }
                                int v1613;
                                v1613 = v1527[v1444];
                                bool v1615;
                                v1615 = v1608 < v1613;
                                int v1616;
                                if (v1615){
                                    v1616 = v1608;
                                } else {
                                    v1616 = v1613;
                                }
                                static_array<int,2> v1617;
                                int v1619;
                                v1619 = 0;
                                while (while_method_2(v1619)){
                                    bool v1621;
                                    v1621 = 0 <= v1619;
                                    bool v1623;
                                    if (v1621){
                                        bool v1622;
                                        v1622 = v1619 < 2;
                                        v1623 = v1622;
                                    } else {
                                        v1623 = false;
                                    }
                                    bool v1624;
                                    v1624 = v1623 == false;
                                    if (v1624){
                                        assert("Index must be in range." && v1623);
                                    } else {
                                    }
                                    int v1626;
                                    v1626 = v1439[v1619];
                                    bool v1628;
                                    v1628 = v1444 == v1619;
                                    int v1629;
                                    if (v1628){
                                        v1629 = v1616;
                                    } else {
                                        v1629 = v1626;
                                    }
                                    v1617[v1619] = v1629;
                                    v1619 += 1 ;
                                }
                                static_array<int,2> v1630;
                                int v1632;
                                v1632 = 0;
                                while (while_method_2(v1632)){
                                    bool v1634;
                                    v1634 = 0 <= v1632;
                                    bool v1636;
                                    if (v1634){
                                        bool v1635;
                                        v1635 = v1632 < 2;
                                        v1636 = v1635;
                                    } else {
                                        v1636 = false;
                                    }
                                    bool v1637;
                                    v1637 = v1636 == false;
                                    if (v1637){
                                        assert("Index must be in range." && v1636);
                                    } else {
                                    }
                                    int v1639;
                                    v1639 = v1527[v1632];
                                    bool v1642;
                                    if (v1634){
                                        bool v1641;
                                        v1641 = v1632 < 2;
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
                                    v1645 = v1617[v1632];
                                    int v1647;
                                    v1647 = v1639 - v1645;
                                    v1630[v1632] = v1647;
                                    v1632 += 1 ;
                                }
                                int v1648;
                                v1648 = v1440 + 1;
                                v1775 = try_round_5(v1523, v1438, v1617, v1648, v1630, v1442);
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        v1782 = Union5{Union5_1{v1775}};
                        break;
                    }
                    case 2: { // T_some
                        Union3 v1435 = v1433.case2.v0;
                        v1782 = Union5{Union5_1{v1435}};
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
        v20 = v1782;
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
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1, float * v2, float * v3, float * v4) {
    auto v5 = cooperative_groups::this_grid();
    unsigned long long v6;
    v6 = clock64();
    int v7;
    v7 = threadIdx.x;
    int v8;
    v8 = blockIdx.x;
    int v9;
    v9 = v8 * 256;
    int v10;
    v10 = v7 + v9;
    unsigned long long v11;
    v11 = (unsigned long long)v10;
    curandStatePhilox4_32_10_t v12;
    curand_init(v6,v11,0ull,&v12);
    static_array<Union0,2> v13;
    Union0 v15;
    v15 = Union0{Union0_2{}};
    v13[0] = v15;
    Union0 v17;
    v17 = Union0{Union0_2{}};
    v13[1] = v17;
    static_array_list<Union1,128> v19;
    v19 = static_array_list<Union1,128>{};
    static_array<float,2> v21;
    v21[0] = 0.0f;
    v21[1] = 0.0f;
    cooperative_groups::grid_group & v23 = v5;
    curandStatePhilox4_32_10_t & v24 = v12;
    StackMut0 v25{4503599627370495ull, v23, v19, v13, v21, v24};
    int v26;
    v26 = 0;
    while (while_method_0(v26)){
        int v28;
        v28 = 0;
        while (while_method_1(v28)){
            int v30;
            v30 = 0;
            while (while_method_2(v30)){
                Union3 v32;
                v32 = Union3{Union3_2{}};
                method_0(v0, v1, v25, v30, v32);
                static_array<float,2> & v33 = v25.v4;
                bool v34;
                v34 = 0 <= v30;
                bool v36;
                if (v34){
                    bool v35;
                    v35 = v30 < 2;
                    v36 = v35;
                } else {
                    v36 = false;
                }
                bool v37;
                v37 = v36 == false;
                if (v37){
                    assert("Index must be in range." && v36);
                } else {
                }
                float v39;
                v39 = v33[v30];
                double * v41;
                v41 = reinterpret_cast<double *>(&v1[11534352ull]);
                double * v43;
                v43 = reinterpret_cast<double *>(&v1[11927568ull]);
                int * v45;
                v45 = reinterpret_cast<int *>(&v1[12320784ull]);
                int v47;
                v47 = threadIdx.x;
                int v48;
                v48 = blockIdx.x;
                int v49;
                v49 = v48 * 256;
                int v50;
                v50 = v47 + v49;
                assert("Tensor range check" && 0 <= v50 && v50 < 6144);
                int v51;
                v51 = 2 * v50;
                int v52; double v53;
                Tuple18 tmp64 = Tuple18{0, 1.0};
                v52 = tmp64.v0; v53 = tmp64.v1;
                while (while_method_2(v52)){
                    assert("Tensor range check" && 0 <= v52 && v52 < 2);
                    int v55;
                    v55 = v52 + v51;
                    int v56; double v57;
                    Tuple18 tmp65 = Tuple18{0, 0.0};
                    v56 = tmp65.v0; v57 = tmp65.v1;
                    while (while_method_0(v56)){
                        assert("Tensor range check" && 0 <= v56 && v56 < 4);
                        int v59;
                        v59 = 12288 * v56;
                        int v60;
                        v60 = v59 + v55;
                        double v61;
                        v61 = v41[v60];
                        double v62;
                        v62 = v43[v60];
                        double v63;
                        v63 = v61 - v62;
                        double v64;
                        v64 = exp(v63);
                        double v65;
                        v65 = v57 + v64;
                        v57 = v65;
                        v56 += 1 ;
                    }
                    double v66;
                    v66 = v53 * v57;
                    v53 = v66;
                    v52 += 1 ;
                }
                float v67;
                v67 = (float)v53;
                int v68;
                v68 = 0;
                while (while_method_0(v68)){
                    double * v70;
                    v70 = reinterpret_cast<double *>(&v1[11534352ull]);
                    double * v72;
                    v72 = reinterpret_cast<double *>(&v1[11927568ull]);
                    int * v74;
                    v74 = reinterpret_cast<int *>(&v1[12320784ull]);
                    int v76;
                    v76 = threadIdx.x;
                    int v77;
                    v77 = blockIdx.x;
                    int v78;
                    v78 = v77 * 256;
                    int v79;
                    v79 = v76 + v78;
                    assert("Tensor range check" && 0 <= v79 && v79 < 6144);
                    int v80;
                    v80 = 2 * v79;
                    int v81; double v82;
                    Tuple18 tmp66 = Tuple18{0, 1.0};
                    v81 = tmp66.v0; v82 = tmp66.v1;
                    while (while_method_2(v81)){
                        assert("Tensor range check" && 0 <= v81 && v81 < 2);
                        int v84;
                        v84 = v81 + v80;
                        int v85; double v86;
                        Tuple18 tmp67 = Tuple18{0, 0.0};
                        v85 = tmp67.v0; v86 = tmp67.v1;
                        while (while_method_0(v85)){
                            assert("Tensor range check" && 0 <= v85 && v85 < 4);
                            int v88;
                            v88 = 12288 * v85;
                            int v89;
                            v89 = v88 + v84;
                            double v90;
                            v90 = v70[v89];
                            double v91;
                            v91 = v72[v89];
                            double v92;
                            v92 = v90 - v91;
                            double v93;
                            v93 = exp(v92);
                            bool v94;
                            v94 = v68 == v85;
                            bool v95;
                            v95 = v94 != true;
                            double v96;
                            if (v95){
                                v96 = v93;
                            } else {
                                v96 = 0.0;
                            }
                            double v97;
                            v97 = v86 + v96;
                            v86 = v97;
                            v85 += 1 ;
                        }
                        double v98;
                        v98 = v82 * v86;
                        v82 = v98;
                        v81 += 1 ;
                    }
                    float v99;
                    v99 = (float)v82;
                    float v100;
                    v100 = v67 - v99;
                    float v101;
                    v101 = v39 * v100;
                    assert("Tensor range check" && 0 <= v68 && v68 < 4);
                    assert("Tensor range check" && 0 <= v26 && v26 < 4);
                    int v102;
                    v102 = 4 * v68;
                    int v103;
                    v103 = v102 + v26;
                    float * v104;
                    v104 = v2+v103;
                    float * v106;
                    v106 = v3+v103;
                    float v108;
                    v108 = atomicAdd(v104,v101);
                    float v109;
                    v109 = atomicAdd(v106,v100);
                    v68 += 1 ;
                }
                static_array<float,2> & v110 = v25.v4;
                unsigned int * v111;
                v111 = reinterpret_cast<unsigned int *>(&v0[53477376ull]);
                int * v113;
                v113 = reinterpret_cast<int *>(&v1[4194304ull]);
                float * v115;
                v115 = reinterpret_cast<float *>(&v1[4194320ull]);
                float * v117;
                v117 = reinterpret_cast<float *>(&v1[5242896ull]);
                float * v119;
                v119 = reinterpret_cast<float *>(&v1[6291472ull]);
                float * v121;
                v121 = reinterpret_cast<float *>(&v1[7340048ull]);
                float * v123;
                v123 = reinterpret_cast<float *>(&v1[8388624ull]);
                float * v125;
                v125 = reinterpret_cast<float *>(&v1[9437200ull]);
                float * v127;
                v127 = reinterpret_cast<float *>(&v1[10485776ull]);
                int * v129;
                v129 = reinterpret_cast<int *>(&v0[53575680ull]);
                float * v131;
                v131 = reinterpret_cast<float *>(&v0[66158592ull]);
                int * v133;
                v133 = reinterpret_cast<int *>(&v0[78741504ull]);
                int * v135;
                v135 = reinterpret_cast<int *>(&v0[91324416ull]);
                double * v137;
                v137 = reinterpret_cast<double *>(&v0[103907328ull]);
                double * v139;
                v139 = reinterpret_cast<double *>(&v0[154238976ull]);
                double * v141;
                v141 = reinterpret_cast<double *>(&v1[11534352ull]);
                double * v143;
                v143 = reinterpret_cast<double *>(&v1[11927568ull]);
                int * v145;
                v145 = reinterpret_cast<int *>(&v1[12320784ull]);
                int v147;
                v147 = 0;
                while (while_method_0(v147)){
                    int v149;
                    v149 = threadIdx.x;
                    int v150;
                    v150 = blockIdx.x;
                    int v151;
                    v151 = v150 * 256;
                    int v152;
                    v152 = v149 + v151;
                    float v153[2];
                    int v154;
                    v154 = 0;
                    while (while_method_2(v154)){
                        bool v156;
                        v156 = 0 <= v154;
                        bool v158;
                        if (v156){
                            bool v157;
                            v157 = v154 < 2;
                            v158 = v157;
                        } else {
                            v158 = false;
                        }
                        bool v159;
                        v159 = v158 == false;
                        if (v159){
                            assert("Index must be in range." && v158);
                        } else {
                        }
                        float v161;
                        v161 = v110[v154];
                        v153[v154] = v161;
                        v154 += 1 ;
                    }
                    assert("Tensor range check" && 0 <= v147 && v147 < 4);
                    assert("Tensor range check" && 0 <= v152 && v152 < 6144);
                    int v163;
                    v163 = 6144 * v147;
                    int v164;
                    v164 = v163 + v152;
                    int v165;
                    v165 = v145[v164];
                    int v166;
                    v166 = v165;
                    while (while_method_21(v166)){
                        v166 -= 1 ;
                        assert("Tensor range check" && 0 <= v147 && v147 < 4);
                        assert("Tensor range check" && 0 <= v166 && v166 < 128);
                        assert("Tensor range check" && 0 <= v152 && v152 < 6144);
                        int v168;
                        v168 = 6144 * v166;
                        int v169;
                        v169 = v168 + v152;
                        int v170;
                        v170 = 786432 * v147;
                        int v171;
                        v171 = v170 + v169;
                        int v172;
                        v172 = v129[v171];
                        float v173;
                        v173 = v131[v171];
                        int v174;
                        v174 = v133[v171];
                        int v175;
                        v175 = v135[v171];
                        assert("Tensor range check" && 0 <= v174 && v174 < 2);
                        float v176;
                        v176 = v153[v174];
                        assert("Tensor range check" && 0 <= v147 && v147 < 4);
                        int v177;
                        v177 = 65536 * v147;
                        assert("Tensor range check" && 0 <= v175 && v175 < 4096);
                        int v178;
                        v178 = 16 * v175;
                        int v179;
                        v179 = v178 + v177;
                        float * v180;
                        v180 = v115+v179;
                        float * v182;
                        v182 = v117+v179;
                        float * v184;
                        v184 = v119+v179;
                        float * v186;
                        v186 = v121+v179;
                        float * v188;
                        v188 = v123+v179;
                        float * v190;
                        v190 = v125+v179;
                        float * v192;
                        v192 = v127+v179;
                        assert("Tensor range check" && 0 <= v147 && v147 < 4);
                        int v194;
                        v194 = 1572864 * v147;
                        assert("Tensor range check" && 0 <= v166 && v166 < 128);
                        int v195;
                        v195 = 12288 * v166;
                        int v196;
                        v196 = v195 + v194;
                        assert("Tensor range check" && 0 <= v152 && v152 < 6144);
                        int v197;
                        v197 = 2 * v152;
                        int v198;
                        v198 = v197 + v196;
                        double v199[2];
                        int v200;
                        v200 = 0;
                        while (while_method_2(v200)){
                            assert("Tensor range check" && 0 <= v200 && v200 < 2);
                            int v202;
                            v202 = v200 + v198;
                            double v203;
                            v203 = v137[v202];
                            bool v204;
                            v204 = v174 == v200;
                            double v205;
                            if (v204){
                                v205 = 0.0;
                            } else {
                                v205 = v203;
                            }
                            assert("Tensor range check" && 0 <= v200 && v200 < 2);
                            v199[v200] = v205;
                            v200 += 1 ;
                        }
                        double v206;
                        v206 = 0.0;
                        int v207;
                        v207 = 0;
                        while (while_method_2(v207)){
                            assert("Tensor range check" && 0 <= v207 && v207 < 2);
                            double v209;
                            v209 = v199[v207];
                            double v210;
                            v210 = v206 + v209;
                            v206 = v210;
                            v207 += 1 ;
                        }
                        double v211;
                        v211 = 0.0;
                        int v212;
                        v212 = 0;
                        while (while_method_2(v212)){
                            assert("Tensor range check" && 0 <= v212 && v212 < 2);
                            int v214;
                            v214 = v212 + v198;
                            double v215;
                            v215 = v139[v214];
                            double v216;
                            v216 = v211 + v215;
                            v211 = v216;
                            v212 += 1 ;
                        }
                        double v217;
                        v217 = v206 - v211;
                        double v218;
                        v218 = exp(v217);
                        float v219;
                        v219 = (float)v218;
                        float v220;
                        v220 = v176 * v219;
                        assert("Tensor range check" && 0 <= v172 && v172 < 16);
                        float * v221;
                        v221 = v190+v172;
                        float * v223;
                        v223 = v192+v172;
                        float v225;
                        v225 = atomicAdd(v221,v220);
                        float v226;
                        v226 = atomicAdd(v223,v219);
                        float * v227;
                        v227 = v182+0;
                        float * v229;
                        v229 = v186+0;
                        float * v231;
                        v231 = v188+0;
                        int v233;
                        v233 = sizeof(float *);
                        unsigned long long v234;
                        v234 = (unsigned long long)v233;
                        unsigned long long v235;
                        v235 = 256ull * v234;
                        unsigned long long v236;
                        v236 = 4096ull + v235;
                        unsigned long long v237;
                        v237 = v236 + 16ull;
                        unsigned long long v238;
                        v238 = v237 - 1ull;
                        unsigned long long v239;
                        v239 = v238 % 16ull;
                        unsigned long long v240;
                        v240 = v238 - v239;
                        unsigned long long v241;
                        v241 = v240 + v235;
                        unsigned long long v242;
                        v242 = v241 + 16ull;
                        unsigned long long v243;
                        v243 = v242 - 1ull;
                        unsigned long long v244;
                        v244 = v243 % 16ull;
                        unsigned long long v245;
                        v245 = v243 - v244;
                        unsigned long long v246;
                        v246 = v245 + v235;
                        unsigned long long v247;
                        v247 = v246 + 16ull;
                        unsigned long long v248;
                        v248 = v247 - 1ull;
                        unsigned long long v249;
                        v249 = v248 % 16ull;
                        unsigned long long v250;
                        v250 = v248 - v249;
                        unsigned long long v251;
                        v251 = v250 + v235;
                        unsigned long long v252;
                        v252 = v251 + 16ull;
                        unsigned long long v253;
                        v253 = v252 - 1ull;
                        unsigned long long v254;
                        v254 = v253 % 16ull;
                        unsigned long long v255;
                        v255 = v253 - v254;
                        unsigned long long v256;
                        v256 = v255 + 1024ull;
                        bool v257;
                        v257 = v256 <= 98304ull;
                        bool v258;
                        v258 = v257 == false;
                        if (v258){
                            assert("The dynamic shared memory is insufficient to allocate the tensor." && v257);
                        } else {
                        }
                        extern __shared__ unsigned char v260[];
                        bool v261;
                        v261 = v256 <= v256;
                        bool v262;
                        v262 = v261 == false;
                        if (v262){
                            assert("The length of the partition has to be less than or equal to the length of the base array." && v261);
                        } else {
                        }
                        float * v264;
                        v264 = reinterpret_cast<float *>(&v260[0ull]);
                        int * v266;
                        v266 = reinterpret_cast<int *>(&v260[1024ull]);
                        float * v268;
                        v268 = reinterpret_cast<float *>(&v260[2048ull]);
                        float * v270;
                        v270 = reinterpret_cast<float *>(&v260[3072ull]);
                        float * * v272;
                        v272 = reinterpret_cast<float * *>(&v260[4096ull]);
                        float * * v274;
                        v274 = reinterpret_cast<float * *>(&v260[v240]);
                        float * * v276;
                        v276 = reinterpret_cast<float * *>(&v260[v245]);
                        float * * v278;
                        v278 = reinterpret_cast<float * *>(&v260[v250]);
                        float * v280;
                        v280 = reinterpret_cast<float *>(&v260[v255]);
                        int v282;
                        v282 = threadIdx.x;
                        assert("Tensor range check" && 0 <= v282 && v282 < 256);
                        v264[v282] = v173;
                        v266[v282] = v172;
                        v268[v282] = v176;
                        v270[v282] = v219;
                        v272[v282] = v184;
                        v274[v282] = v227;
                        v276[v282] = v229;
                        v278[v282] = v231;
                        __syncthreads();
                        bool v283;
                        v283 = 0 <= v282;
                        bool v284;
                        v284 = v283 == false;
                        if (v284){
                            assert("The index needs to be zero or positive." && v283);
                        } else {
                        }
                        int v286;
                        v286 = v282 % 4;
                        int v287;
                        v287 = v282 / 4;
                        bool v288;
                        v288 = v287 < 64;
                        bool v289;
                        v289 = v288 == false;
                        if (v289){
                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v288);
                        } else {
                        }
                        assert("Tensor range check" && 0 <= v287 && v287 < 64);
                        int v291;
                        v291 = 0;
                        while (while_method_0(v291)){
                            bool v293;
                            v293 = 0 <= v287;
                            bool v294;
                            v294 = v293 && v288;
                            bool v295;
                            v295 = v294 == false;
                            if (v295){
                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v294);
                            } else {
                            }
                            bool v297;
                            v297 = 0 <= v291;
                            bool v299;
                            if (v297){
                                bool v298;
                                v298 = v291 < 4;
                                v299 = v298;
                            } else {
                                v299 = false;
                            }
                            bool v300;
                            v300 = v299 == false;
                            if (v300){
                                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v299);
                            } else {
                            }
                            int v302;
                            v302 = v291 * 64;
                            int v303;
                            v303 = v302 + v287;
                            assert("Tensor range check" && 0 <= v291 && v291 < 4);
                            int v304;
                            v304 = 64 * v291;
                            int v305;
                            v305 = v304 + v287;
                            float v306;
                            v306 = v264[v305];
                            int v307;
                            v307 = v266[v305];
                            float v308;
                            v308 = v268[v305];
                            float v309;
                            v309 = v270[v305];
                            float * v310;
                            v310 = v272[v305];
                            float * v311;
                            v311 = v274[v305];
                            float * v312;
                            v312 = v276[v305];
                            float * v313;
                            v313 = v278[v305];
                            int v314;
                            v314 = blockIdx.x;
                            int v315;
                            v315 = v314 * 256;
                            int v316;
                            v316 = v315 + v303;
                            assert("Tensor range check" && 0 <= v286 && v286 < 4);
                            int v317;
                            v317 = 4 * v286;
                            float v318[4];
                            float v319[4];
                            float v320[4];
                            int v321[4];
                            int v322;
                            v322 = 0;
                            while (while_method_6(v322)){
                                assert("Tensor range check" && 0 <= v322 && v322 < 1);
                                int v324;
                                v324 = 4 * v322;
                                assert("Tensor range check" && 0 <= v322 && v322 < 1);
                                int v325;
                                v325 = 16 * v322;
                                int v326;
                                v326 = v325 + v317;
                                int4* v327;
                                v327 = reinterpret_cast<int4*>(v311 + v326);
                                int4* v328;
                                v328 = reinterpret_cast<int4*>(v318 + v324);
                                assert("Pointer alignment check" && (unsigned long long)(v327) % 4 == 0 && (unsigned long long)(v328) % 4 == 0);
                                *v328 = *v327;
                                int4* v329;
                                v329 = reinterpret_cast<int4*>(v312 + v326);
                                int4* v330;
                                v330 = reinterpret_cast<int4*>(v319 + v324);
                                assert("Pointer alignment check" && (unsigned long long)(v329) % 4 == 0 && (unsigned long long)(v330) % 4 == 0);
                                *v330 = *v329;
                                int4* v331;
                                v331 = reinterpret_cast<int4*>(v313 + v326);
                                int4* v332;
                                v332 = reinterpret_cast<int4*>(v320 + v324);
                                assert("Pointer alignment check" && (unsigned long long)(v331) % 4 == 0 && (unsigned long long)(v332) % 4 == 0);
                                *v332 = *v331;
                                v322 += 1 ;
                            }
                            int v333;
                            v333 = 0;
                            while (while_method_6(v333)){
                                int v335;
                                v335 = 0;
                                while (while_method_0(v335)){
                                    bool v337;
                                    v337 = 0 <= v335;
                                    bool v339;
                                    if (v337){
                                        bool v338;
                                        v338 = v335 < 4;
                                        v339 = v338;
                                    } else {
                                        v339 = false;
                                    }
                                    bool v340;
                                    v340 = v339 == false;
                                    if (v340){
                                        assert("The indices should be inside the range of the dimension." && v339);
                                    } else {
                                    }
                                    bool v342;
                                    v342 = 0 <= v286;
                                    bool v344;
                                    if (v342){
                                        bool v343;
                                        v343 = v286 < 4;
                                        v344 = v343;
                                    } else {
                                        v344 = false;
                                    }
                                    bool v345;
                                    v345 = v344 == false;
                                    if (v345){
                                        assert("The indices should be inside the range of the dimension." && v344);
                                    } else {
                                    }
                                    int v347;
                                    v347 = v286 * 4;
                                    int v348;
                                    v348 = v335 + v347;
                                    bool v349;
                                    v349 = 0 <= v333;
                                    bool v351;
                                    if (v349){
                                        bool v350;
                                        v350 = v333 < 1;
                                        v351 = v350;
                                    } else {
                                        v351 = false;
                                    }
                                    bool v352;
                                    v352 = v351 == false;
                                    if (v352){
                                        assert("The indices should be inside the range of the dimension." && v351);
                                    } else {
                                    }
                                    int v354;
                                    v354 = v333 * 16;
                                    int v355;
                                    v355 = v348 + v354;
                                    assert("Tensor range check" && 0 <= v333 && v333 < 1);
                                    assert("Tensor range check" && 0 <= v335 && v335 < 4);
                                    int v356;
                                    v356 = 4 * v333;
                                    int v357;
                                    v357 = v356 + v335;
                                    v321[v357] = v355;
                                    v335 += 1 ;
                                }
                                v333 += 1 ;
                            }
                            float v358[4];
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
                                    v365 = v319[v364];
                                    float v366;
                                    v366 = v320[v364];
                                    bool v367;
                                    v367 = v366 == 0.0f;
                                    bool v368;
                                    v368 = v367 != true;
                                    float v370;
                                    if (v368){
                                        float v369;
                                        v369 = v365 / v366;
                                        v370 = v369;
                                    } else {
                                        v370 = 0.0f;
                                    }
                                    assert("Tensor range check" && 0 <= v359 && v359 < 1);
                                    assert("Tensor range check" && 0 <= v361 && v361 < 4);
                                    v358[v364] = v370;
                                    v361 += 1 ;
                                }
                                v359 += 1 ;
                            }
                            bool v371[4];
                            int v372;
                            v372 = 0;
                            while (while_method_6(v372)){
                                int v374;
                                v374 = 0;
                                while (while_method_0(v374)){
                                    assert("Tensor range check" && 0 <= v372 && v372 < 1);
                                    assert("Tensor range check" && 0 <= v374 && v374 < 4);
                                    int v376;
                                    v376 = 4 * v372;
                                    int v377;
                                    v377 = v376 + v374;
                                    float v378;
                                    v378 = v318[v377];
                                    int v379;
                                    v379 = v321[v377];
                                    bool v380;
                                    v380 = v379 < 11;
                                    assert("Tensor range check" && 0 <= v372 && v372 < 1);
                                    assert("Tensor range check" && 0 <= v374 && v374 < 4);
                                    v371[v377] = v380;
                                    v374 += 1 ;
                                }
                                v372 += 1 ;
                            }
                            float v381[4];
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
                                    float v388;
                                    v388 = v318[v387];
                                    bool v389;
                                    v389 = v371[v387];
                                    float v392;
                                    if (v389){
                                        bool v390;
                                        v390 = 0.0f >= v388;
                                        if (v390){
                                            v392 = 0.0f;
                                        } else {
                                            v392 = v388;
                                        }
                                    } else {
                                        v392 = 0.0f;
                                    }
                                    assert("Tensor range check" && 0 <= v382 && v382 < 1);
                                    assert("Tensor range check" && 0 <= v384 && v384 < 4);
                                    v381[v387] = v392;
                                    v384 += 1 ;
                                }
                                v382 += 1 ;
                            }
                            float v393;
                            v393 = 0.0f;
                            int v394;
                            v394 = 0;
                            while (while_method_6(v394)){
                                int v396;
                                v396 = 0;
                                while (while_method_0(v396)){
                                    assert("Tensor range check" && 0 <= v394 && v394 < 1);
                                    assert("Tensor range check" && 0 <= v396 && v396 < 4);
                                    int v398;
                                    v398 = 4 * v394;
                                    int v399;
                                    v399 = v398 + v396;
                                    float v400;
                                    v400 = v381[v399];
                                    float v401;
                                    v401 = v393 + v400;
                                    v393 = v401;
                                    v396 += 1 ;
                                }
                                v394 += 1 ;
                            }
                            auto v402 = cooperative_groups::coalesced_threads();
                            int v403;
                            v403 = threadIdx.x;
                            int v404;
                            v404 = v403 / 4;
                            auto v405 = cooperative_groups::labeled_partition(v402,v404);
                            Closure1 v406{};
                            float v407;
                            v407 = cooperative_groups::reduce(v405, v393, v406);
                            int v408[4];
                            int v409;
                            v409 = 0;
                            while (while_method_6(v409)){
                                int v411;
                                v411 = 0;
                                while (while_method_0(v411)){
                                    assert("Tensor range check" && 0 <= v409 && v409 < 1);
                                    assert("Tensor range check" && 0 <= v411 && v411 < 4);
                                    int v413;
                                    v413 = 4 * v409;
                                    int v414;
                                    v414 = v413 + v411;
                                    bool v415;
                                    v415 = v371[v414];
                                    int v416;
                                    if (v415){
                                        v416 = 1;
                                    } else {
                                        v416 = 0;
                                    }
                                    assert("Tensor range check" && 0 <= v409 && v409 < 1);
                                    assert("Tensor range check" && 0 <= v411 && v411 < 4);
                                    v408[v414] = v416;
                                    v411 += 1 ;
                                }
                                v409 += 1 ;
                            }
                            int v417;
                            v417 = 0;
                            int v418;
                            v418 = 0;
                            while (while_method_6(v418)){
                                int v420;
                                v420 = 0;
                                while (while_method_0(v420)){
                                    assert("Tensor range check" && 0 <= v418 && v418 < 1);
                                    assert("Tensor range check" && 0 <= v420 && v420 < 4);
                                    int v422;
                                    v422 = 4 * v418;
                                    int v423;
                                    v423 = v422 + v420;
                                    int v424;
                                    v424 = v408[v423];
                                    int v425;
                                    v425 = v417 + v424;
                                    v417 = v425;
                                    v420 += 1 ;
                                }
                                v418 += 1 ;
                            }
                            auto v426 = cooperative_groups::coalesced_threads();
                            int v427;
                            v427 = threadIdx.x;
                            int v428;
                            v428 = v427 / 4;
                            auto v429 = cooperative_groups::labeled_partition(v426,v428);
                            Closure2 v430{};
                            int v431;
                            v431 = cooperative_groups::reduce(v429, v417, v430);
                            float v432;
                            v432 = (float)v431;
                            float v433;
                            v433 = 1.0f / v432;
                            float v434[4];
                            int v435;
                            v435 = 0;
                            while (while_method_6(v435)){
                                int v437;
                                v437 = 0;
                                while (while_method_0(v437)){
                                    assert("Tensor range check" && 0 <= v435 && v435 < 1);
                                    assert("Tensor range check" && 0 <= v437 && v437 < 4);
                                    int v439;
                                    v439 = 4 * v435;
                                    int v440;
                                    v440 = v439 + v437;
                                    float v441;
                                    v441 = v381[v440];
                                    bool v442;
                                    v442 = v371[v440];
                                    bool v443;
                                    v443 = v442 == false;
                                    float v448;
                                    if (v443){
                                        v448 = 0.0f;
                                    } else {
                                        bool v444;
                                        v444 = v407 == 0.0f;
                                        bool v445;
                                        v445 = v444 != true;
                                        if (v445){
                                            float v446;
                                            v446 = v441 / v407;
                                            v448 = v446;
                                        } else {
                                            v448 = v433;
                                        }
                                    }
                                    assert("Tensor range check" && 0 <= v435 && v435 < 1);
                                    assert("Tensor range check" && 0 <= v437 && v437 < 4);
                                    v434[v440] = v448;
                                    v437 += 1 ;
                                }
                                v435 += 1 ;
                            }
                            float v449[4];
                            int v450;
                            v450 = 0;
                            while (while_method_6(v450)){
                                int v452;
                                v452 = 0;
                                while (while_method_0(v452)){
                                    assert("Tensor range check" && 0 <= v450 && v450 < 1);
                                    assert("Tensor range check" && 0 <= v452 && v452 < 4);
                                    int v454;
                                    v454 = 4 * v450;
                                    int v455;
                                    v455 = v454 + v452;
                                    float v456;
                                    v456 = v358[v455];
                                    int v457;
                                    v457 = v321[v455];
                                    bool v458;
                                    v458 = v307 == v457;
                                    float v461;
                                    if (v458){
                                        float v459;
                                        v459 = v308 - v456;
                                        float v460;
                                        v460 = v459 / v306;
                                        v461 = v460;
                                    } else {
                                        v461 = 0.0f;
                                    }
                                    float v462;
                                    v462 = v461 + v456;
                                    assert("Tensor range check" && 0 <= v450 && v450 < 1);
                                    assert("Tensor range check" && 0 <= v452 && v452 < 4);
                                    v449[v455] = v462;
                                    v452 += 1 ;
                                }
                                v450 += 1 ;
                            }
                            float v463[4];
                            int v464;
                            v464 = 0;
                            while (while_method_6(v464)){
                                int v466;
                                v466 = 0;
                                while (while_method_0(v466)){
                                    assert("Tensor range check" && 0 <= v464 && v464 < 1);
                                    assert("Tensor range check" && 0 <= v466 && v466 < 4);
                                    int v468;
                                    v468 = 4 * v464;
                                    int v469;
                                    v469 = v468 + v466;
                                    float v470;
                                    v470 = v434[v469];
                                    float v471;
                                    v471 = v449[v469];
                                    float v472;
                                    v472 = v470 * v471;
                                    assert("Tensor range check" && 0 <= v464 && v464 < 1);
                                    assert("Tensor range check" && 0 <= v466 && v466 < 4);
                                    v463[v469] = v472;
                                    v466 += 1 ;
                                }
                                v464 += 1 ;
                            }
                            float v473;
                            v473 = 0.0f;
                            int v474;
                            v474 = 0;
                            while (while_method_6(v474)){
                                int v476;
                                v476 = 0;
                                while (while_method_0(v476)){
                                    assert("Tensor range check" && 0 <= v474 && v474 < 1);
                                    assert("Tensor range check" && 0 <= v476 && v476 < 4);
                                    int v478;
                                    v478 = 4 * v474;
                                    int v479;
                                    v479 = v478 + v476;
                                    float v480;
                                    v480 = v463[v479];
                                    float v481;
                                    v481 = v473 + v480;
                                    v473 = v481;
                                    v476 += 1 ;
                                }
                                v474 += 1 ;
                            }
                            auto v482 = cooperative_groups::coalesced_threads();
                            int v483;
                            v483 = threadIdx.x;
                            int v484;
                            v484 = v483 / 4;
                            auto v485 = cooperative_groups::labeled_partition(v482,v484);
                            float v486;
                            v486 = cooperative_groups::reduce(v485, v473, v406);
                            int v487;
                            v487 = 0;
                            while (while_method_6(v487)){
                                int v489;
                                v489 = 0;
                                while (while_method_0(v489)){
                                    assert("Tensor range check" && 0 <= v487 && v487 < 1);
                                    assert("Tensor range check" && 0 <= v489 && v489 < 4);
                                    int v491;
                                    v491 = 4 * v487;
                                    int v492;
                                    v492 = v491 + v489;
                                    float v493;
                                    v493 = v449[v492];
                                    int v494;
                                    v494 = v321[v492];
                                    float v495;
                                    v495 = v493 - v486;
                                    float v496;
                                    v496 = v309 * v495;
                                    assert("Tensor range check" && 0 <= v494 && v494 < 16);
                                    float * v497;
                                    v497 = v310+v494;
                                    float v499;
                                    v499 = atomicAdd(v497,v496);
                                    v489 += 1 ;
                                }
                                v487 += 1 ;
                            }
                            int v500;
                            v500 = 0;
                            while (while_method_6(v500)){
                                assert("Tensor range check" && 0 <= v500 && v500 < 1);
                                assert("Tensor range check" && 0 <= v500 && v500 < 1);
                                v500 += 1 ;
                            }
                            assert("Tensor range check" && 0 <= v303 && v303 < 256);
                            v280[v303] = v486;
                            v291 += 1 ;
                        }
                        __syncthreads();
                        assert("Tensor range check" && 0 <= v282 && v282 < 256);
                        float v502;
                        v502 = v280[v282];
                        __syncthreads();
                        assert("Tensor range check" && 0 <= v174 && v174 < 2);
                        v153[v174] = v502;
                    }
                    int v503;
                    v503 = threadIdx.x;
                    int v504;
                    v504 = blockIdx.x;
                    int v505;
                    v505 = v504 * 256;
                    int v506;
                    v506 = v503 + v505;
                    assert("Tensor range check" && 0 <= v147 && v147 < 4);
                    int v507;
                    v507 = 12288 * v147;
                    assert("Tensor range check" && 0 <= v506 && v506 < 6144);
                    int v508;
                    v508 = 2 * v506;
                    int v509;
                    v509 = v508 + v507;
                    double * v510;
                    v510 = v141+v509;
                    double * v512;
                    v512 = v143+v509;
                    double * v514;
                    v514 = v510+0;
                    double * v516;
                    v516 = v512+0;
                    double * v518;
                    v518 = v510+0;
                    double * v520;
                    v520 = v512+0;
                    int v522;
                    v522 = sizeof(double *);
                    unsigned long long v523;
                    v523 = (unsigned long long)v522;
                    unsigned long long v524;
                    v524 = 256ull * v523;
                    unsigned long long v525;
                    v525 = v524 + 16ull;
                    unsigned long long v526;
                    v526 = v525 - 1ull;
                    unsigned long long v527;
                    v527 = v526 % 16ull;
                    unsigned long long v528;
                    v528 = v526 - v527;
                    unsigned long long v529;
                    v529 = v528 + v524;
                    unsigned long long v530;
                    v530 = v529 + 16ull;
                    unsigned long long v531;
                    v531 = v530 - 1ull;
                    unsigned long long v532;
                    v532 = v531 % 16ull;
                    unsigned long long v533;
                    v533 = v531 - v532;
                    unsigned long long v534;
                    v534 = v533 + v524;
                    unsigned long long v535;
                    v535 = v534 + 16ull;
                    unsigned long long v536;
                    v536 = v535 - 1ull;
                    unsigned long long v537;
                    v537 = v536 % 16ull;
                    unsigned long long v538;
                    v538 = v536 - v537;
                    unsigned long long v539;
                    v539 = v538 + v524;
                    bool v540;
                    v540 = v539 <= 98304ull;
                    bool v541;
                    v541 = v540 == false;
                    if (v541){
                        assert("The dynamic shared memory is insufficient to allocate the tensor." && v540);
                    } else {
                    }
                    extern __shared__ unsigned char v543[];
                    bool v544;
                    v544 = v539 <= v539;
                    bool v545;
                    v545 = v544 == false;
                    if (v545){
                        assert("The length of the partition has to be less than or equal to the length of the base array." && v544);
                    } else {
                    }
                    double * * v547;
                    v547 = reinterpret_cast<double * *>(&v543[0ull]);
                    double * * v549;
                    v549 = reinterpret_cast<double * *>(&v543[v528]);
                    double * * v551;
                    v551 = reinterpret_cast<double * *>(&v543[v533]);
                    double * * v553;
                    v553 = reinterpret_cast<double * *>(&v543[v538]);
                    int v555;
                    v555 = threadIdx.x;
                    assert("Tensor range check" && 0 <= v555 && v555 < 256);
                    v547[v555] = v514;
                    v549[v555] = v516;
                    v551[v555] = v518;
                    v553[v555] = v520;
                    __syncthreads();
                    bool v556;
                    v556 = 0 <= v555;
                    bool v557;
                    v557 = v556 == false;
                    if (v557){
                        assert("The index needs to be zero or positive." && v556);
                    } else {
                    }
                    int v559;
                    v559 = v555 % 1;
                    bool v560;
                    v560 = v555 < 256;
                    bool v561;
                    v561 = v560 == false;
                    if (v561){
                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v560);
                    } else {
                    }
                    assert("Tensor range check" && 0 <= v555 && v555 < 256);
                    int v563;
                    v563 = 0;
                    while (while_method_6(v563)){
                        bool v565;
                        v565 = v556 && v560;
                        bool v566;
                        v566 = v565 == false;
                        if (v566){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v565);
                        } else {
                        }
                        bool v568;
                        v568 = 0 <= v563;
                        bool v570;
                        if (v568){
                            bool v569;
                            v569 = v563 < 1;
                            v570 = v569;
                        } else {
                            v570 = false;
                        }
                        bool v571;
                        v571 = v570 == false;
                        if (v571){
                            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v570);
                        } else {
                        }
                        int v573;
                        v573 = v563 * 256;
                        int v574;
                        v574 = v573 + v555;
                        assert("Tensor range check" && 0 <= v563 && v563 < 1);
                        int v575;
                        v575 = 256 * v563;
                        int v576;
                        v576 = v575 + v555;
                        double * v577;
                        v577 = v547[v576];
                        double * v578;
                        v578 = v549[v576];
                        double * v579;
                        v579 = v551[v576];
                        double * v580;
                        v580 = v553[v576];
                        int v581;
                        v581 = blockIdx.x;
                        int v582;
                        v582 = v581 * 256;
                        int v583;
                        v583 = v582 + v574;
                        assert("Tensor range check" && 0 <= v559 && v559 < 1);
                        int v584;
                        v584 = 2 * v559;
                        double v585[2];
                        double v586[2];
                        int v587[2];
                        int v588;
                        v588 = 0;
                        while (while_method_6(v588)){
                            assert("Tensor range check" && 0 <= v588 && v588 < 1);
                            int v590;
                            v590 = 2 * v588;
                            assert("Tensor range check" && 0 <= v588 && v588 < 1);
                            int v591;
                            v591 = v590 + v584;
                            int4* v592;
                            v592 = reinterpret_cast<int4*>(v577 + v591);
                            int4* v593;
                            v593 = reinterpret_cast<int4*>(v585 + v590);
                            assert("Pointer alignment check" && (unsigned long long)(v592) % 2 == 0 && (unsigned long long)(v593) % 2 == 0);
                            *v593 = *v592;
                            int4* v594;
                            v594 = reinterpret_cast<int4*>(v578 + v591);
                            int4* v595;
                            v595 = reinterpret_cast<int4*>(v586 + v590);
                            assert("Pointer alignment check" && (unsigned long long)(v594) % 2 == 0 && (unsigned long long)(v595) % 2 == 0);
                            *v595 = *v594;
                            v588 += 1 ;
                        }
                        int v596;
                        v596 = 0;
                        while (while_method_6(v596)){
                            int v598;
                            v598 = 0;
                            while (while_method_2(v598)){
                                bool v600;
                                v600 = 0 <= v598;
                                bool v602;
                                if (v600){
                                    bool v601;
                                    v601 = v598 < 2;
                                    v602 = v601;
                                } else {
                                    v602 = false;
                                }
                                bool v603;
                                v603 = v602 == false;
                                if (v603){
                                    assert("The indices should be inside the range of the dimension." && v602);
                                } else {
                                }
                                bool v605;
                                v605 = 0 <= v559;
                                bool v607;
                                if (v605){
                                    bool v606;
                                    v606 = v559 < 1;
                                    v607 = v606;
                                } else {
                                    v607 = false;
                                }
                                bool v608;
                                v608 = v607 == false;
                                if (v608){
                                    assert("The indices should be inside the range of the dimension." && v607);
                                } else {
                                }
                                int v610;
                                v610 = v559 * 2;
                                int v611;
                                v611 = v598 + v610;
                                bool v612;
                                v612 = 0 <= v596;
                                bool v614;
                                if (v612){
                                    bool v613;
                                    v613 = v596 < 1;
                                    v614 = v613;
                                } else {
                                    v614 = false;
                                }
                                bool v615;
                                v615 = v614 == false;
                                if (v615){
                                    assert("The indices should be inside the range of the dimension." && v614);
                                } else {
                                }
                                int v617;
                                v617 = v596 * 2;
                                int v618;
                                v618 = v611 + v617;
                                assert("Tensor range check" && 0 <= v596 && v596 < 1);
                                assert("Tensor range check" && 0 <= v598 && v598 < 2);
                                int v619;
                                v619 = 2 * v596;
                                int v620;
                                v620 = v619 + v598;
                                v587[v620] = v618;
                                v598 += 1 ;
                            }
                            v596 += 1 ;
                        }
                        double v621[2];
                        double v622[2];
                        int v623;
                        v623 = 0;
                        while (while_method_6(v623)){
                            int v625;
                            v625 = 0;
                            while (while_method_2(v625)){
                                assert("Tensor range check" && 0 <= v623 && v623 < 1);
                                assert("Tensor range check" && 0 <= v625 && v625 < 2);
                                int v627;
                                v627 = 2 * v623;
                                int v628;
                                v628 = v627 + v625;
                                double v629;
                                v629 = v585[v628];
                                double v630;
                                v630 = v586[v628];
                                assert("Tensor range check" && 0 <= v623 && v623 < 1);
                                assert("Tensor range check" && 0 <= v625 && v625 < 2);
                                v621[v628] = 0.0;
                                v622[v628] = 0.0;
                                v625 += 1 ;
                            }
                            v623 += 1 ;
                        }
                        int v631;
                        v631 = 0;
                        while (while_method_6(v631)){
                            assert("Tensor range check" && 0 <= v631 && v631 < 1);
                            int v633;
                            v633 = 2 * v631;
                            int v634;
                            v634 = v633 + v584;
                            assert("Tensor range check" && 0 <= v631 && v631 < 1);
                            int4* v635;
                            v635 = reinterpret_cast<int4*>(v621 + v633);
                            int4* v636;
                            v636 = reinterpret_cast<int4*>(v579 + v634);
                            assert("Pointer alignment check" && (unsigned long long)(v635) % 2 == 0 && (unsigned long long)(v636) % 2 == 0);
                            *v636 = *v635;
                            int4* v637;
                            v637 = reinterpret_cast<int4*>(v622 + v633);
                            int4* v638;
                            v638 = reinterpret_cast<int4*>(v580 + v634);
                            assert("Pointer alignment check" && (unsigned long long)(v637) % 2 == 0 && (unsigned long long)(v638) % 2 == 0);
                            *v638 = *v637;
                            v631 += 1 ;
                        }
                        assert("Tensor range check" && 0 <= v574 && v574 < 256);
                        v563 += 1 ;
                    }
                    __syncthreads();
                    assert("Tensor range check" && 0 <= v555 && v555 < 256);
                    __syncthreads();
                    assert("Tensor range check" && 0 <= v147 && v147 < 4);
                    assert("Tensor range check" && 0 <= v506 && v506 < 6144);
                    int v639;
                    v639 = v163 + v506;
                    v145[v639] = 0;
                    v147 += 1 ;
                }
                v30 += 1 ;
            }
            v28 += 1 ;
        }
        cooperative_groups::grid_group & v640 = v25.v1;
        cooperative_groups::grid_group & v641 = v640;
        curandStatePhilox4_32_10_t & v642 = v25.v5;
        curandStatePhilox4_32_10_t & v643 = v642;
        unsigned int * v644;
        v644 = reinterpret_cast<unsigned int *>(&v0[53477376ull]);
        int * v646;
        v646 = reinterpret_cast<int *>(&v1[4194304ull]);
        float * v648;
        v648 = reinterpret_cast<float *>(&v1[4194320ull]);
        float * v650;
        v650 = reinterpret_cast<float *>(&v1[5242896ull]);
        float * v652;
        v652 = reinterpret_cast<float *>(&v1[6291472ull]);
        float * v654;
        v654 = reinterpret_cast<float *>(&v1[7340048ull]);
        float * v656;
        v656 = reinterpret_cast<float *>(&v1[8388624ull]);
        float * v658;
        v658 = reinterpret_cast<float *>(&v1[9437200ull]);
        float * v660;
        v660 = reinterpret_cast<float *>(&v1[10485776ull]);
        int * v662;
        v662 = reinterpret_cast<int *>(&v0[53575680ull]);
        float * v664;
        v664 = reinterpret_cast<float *>(&v0[66158592ull]);
        int * v666;
        v666 = reinterpret_cast<int *>(&v0[78741504ull]);
        int * v668;
        v668 = reinterpret_cast<int *>(&v0[91324416ull]);
        double * v670;
        v670 = reinterpret_cast<double *>(&v0[103907328ull]);
        double * v672;
        v672 = reinterpret_cast<double *>(&v0[154238976ull]);
        double * v674;
        v674 = reinterpret_cast<double *>(&v1[11534352ull]);
        double * v676;
        v676 = reinterpret_cast<double *>(&v1[11927568ull]);
        int * v678;
        v678 = reinterpret_cast<int *>(&v1[12320784ull]);
        v641.sync() ;
        int v680;
        v680 = threadIdx.x;
        int v681;
        v681 = blockIdx.x;
        int v682;
        v682 = v681 * 256;
        int v683;
        v683 = v680 + v682;
        bool v684;
        v684 = v683 == 0;
        if (v684){
            int v685;
            v685 = 0;
            int v686;
            v686 = 4;
            int v687;
            v687 = int_range_21(v686, v685, v643);
            v646[0] = v687;
        } else {
        }
        __syncwarp();
        int v688;
        v688 = threadIdx.x;
        bool v689;
        v689 = 0 <= v688;
        bool v690;
        v690 = v689 == false;
        if (v690){
            assert("The index needs to be zero or positive." && v689);
        } else {
        }
        int v692;
        v692 = v688 % 4;
        int v693;
        v693 = v688 / 4;
        int v694;
        v694 = v693 % 64;
        int v695;
        v695 = v693 / 64;
        bool v696;
        v696 = v695 < 1;
        bool v697;
        v697 = v696 == false;
        if (v697){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v696);
        } else {
        }
        assert("Tensor range check" && 0 <= v695 && v695 < 1);
        assert("Tensor range check" && 0 <= v694 && v694 < 64);
        assert("Tensor range check" && 0 <= v692 && v692 < 4);
        int v699;
        v699 = 4 * v692;
        int v700;
        v700 = 16 * v694;
        int v701;
        v701 = v700 + v699;
        int v702;
        v702 = 65536 * v695;
        int v703;
        v703 = v702 + v701;
        assert("Tensor range check" && 0 <= v695 && v695 < 1);
        assert("Tensor range check" && 0 <= v694 && v694 < 64);
        assert("Tensor range check" && 0 <= v692 && v692 < 4);
        int v704;
        v704 = blockIdx.x;
        int v705;
        v705 = v704;
        while (while_method_22(v705)){
            bool v707;
            v707 = 0 <= v705;
            bool v708;
            v708 = v707 == false;
            if (v708){
                assert("The index needs to be zero or positive." && v707);
            } else {
            }
            int v710;
            v710 = v705 % 64;
            int v711;
            v711 = v705 / 64;
            bool v712;
            v712 = v711 < 4;
            bool v713;
            v713 = v712 == false;
            if (v713){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v712);
            } else {
            }
            assert("Tensor range check" && 0 <= v711 && v711 < 4);
            assert("Tensor range check" && 0 <= v710 && v710 < 64);
            int v715;
            v715 = 1024 * v710;
            int v716;
            v716 = v715 + v703;
            int v717;
            v717 = 65536 * v711;
            int v718;
            v718 = v717 + v716;
            float v719[4];
            float v720[4];
            float v721[4];
            float v722[4];
            float v723[4];
            float v724[4];
            float v725[4];
            int v726[4];
            int v727;
            v727 = 0;
            while (while_method_6(v727)){
                assert("Tensor range check" && 0 <= v727 && v727 < 1);
                int v729;
                v729 = 4 * v727;
                assert("Tensor range check" && 0 <= v727 && v727 < 1);
                int v730;
                v730 = 16 * v727;
                int v731;
                v731 = v730 + v718;
                int4* v732;
                v732 = reinterpret_cast<int4*>(v648 + v731);
                int4* v733;
                v733 = reinterpret_cast<int4*>(v719 + v729);
                assert("Pointer alignment check" && (unsigned long long)(v732) % 4 == 0 && (unsigned long long)(v733) % 4 == 0);
                *v733 = *v732;
                int4* v734;
                v734 = reinterpret_cast<int4*>(v650 + v731);
                int4* v735;
                v735 = reinterpret_cast<int4*>(v720 + v729);
                assert("Pointer alignment check" && (unsigned long long)(v734) % 4 == 0 && (unsigned long long)(v735) % 4 == 0);
                *v735 = *v734;
                int4* v736;
                v736 = reinterpret_cast<int4*>(v652 + v731);
                int4* v737;
                v737 = reinterpret_cast<int4*>(v721 + v729);
                assert("Pointer alignment check" && (unsigned long long)(v736) % 4 == 0 && (unsigned long long)(v737) % 4 == 0);
                *v737 = *v736;
                int4* v738;
                v738 = reinterpret_cast<int4*>(v654 + v731);
                int4* v739;
                v739 = reinterpret_cast<int4*>(v722 + v729);
                assert("Pointer alignment check" && (unsigned long long)(v738) % 4 == 0 && (unsigned long long)(v739) % 4 == 0);
                *v739 = *v738;
                int4* v740;
                v740 = reinterpret_cast<int4*>(v656 + v731);
                int4* v741;
                v741 = reinterpret_cast<int4*>(v723 + v729);
                assert("Pointer alignment check" && (unsigned long long)(v740) % 4 == 0 && (unsigned long long)(v741) % 4 == 0);
                *v741 = *v740;
                int4* v742;
                v742 = reinterpret_cast<int4*>(v658 + v731);
                int4* v743;
                v743 = reinterpret_cast<int4*>(v724 + v729);
                assert("Pointer alignment check" && (unsigned long long)(v742) % 4 == 0 && (unsigned long long)(v743) % 4 == 0);
                *v743 = *v742;
                int4* v744;
                v744 = reinterpret_cast<int4*>(v660 + v731);
                int4* v745;
                v745 = reinterpret_cast<int4*>(v725 + v729);
                assert("Pointer alignment check" && (unsigned long long)(v744) % 4 == 0 && (unsigned long long)(v745) % 4 == 0);
                *v745 = *v744;
                v727 += 1 ;
            }
            int v746;
            v746 = 0;
            while (while_method_6(v746)){
                int v748;
                v748 = 0;
                while (while_method_0(v748)){
                    bool v750;
                    v750 = 0 <= v748;
                    bool v752;
                    if (v750){
                        bool v751;
                        v751 = v748 < 4;
                        v752 = v751;
                    } else {
                        v752 = false;
                    }
                    bool v753;
                    v753 = v752 == false;
                    if (v753){
                        assert("The indices should be inside the range of the dimension." && v752);
                    } else {
                    }
                    bool v755;
                    v755 = 0 <= v692;
                    bool v757;
                    if (v755){
                        bool v756;
                        v756 = v692 < 4;
                        v757 = v756;
                    } else {
                        v757 = false;
                    }
                    bool v758;
                    v758 = v757 == false;
                    if (v758){
                        assert("The indices should be inside the range of the dimension." && v757);
                    } else {
                    }
                    int v760;
                    v760 = v692 * 4;
                    int v761;
                    v761 = v748 + v760;
                    bool v762;
                    v762 = 0 <= v746;
                    bool v764;
                    if (v762){
                        bool v763;
                        v763 = v746 < 1;
                        v764 = v763;
                    } else {
                        v764 = false;
                    }
                    bool v765;
                    v765 = v764 == false;
                    if (v765){
                        assert("The indices should be inside the range of the dimension." && v764);
                    } else {
                    }
                    int v767;
                    v767 = v746 * 16;
                    int v768;
                    v768 = v761 + v767;
                    assert("Tensor range check" && 0 <= v746 && v746 < 1);
                    assert("Tensor range check" && 0 <= v748 && v748 < 4);
                    int v769;
                    v769 = 4 * v746;
                    int v770;
                    v770 = v769 + v748;
                    v726[v770] = v768;
                    v748 += 1 ;
                }
                v746 += 1 ;
            }
            bool v771;
            v771 = 0 <= v695;
            bool v772;
            v772 = v771 && v696;
            bool v773;
            v773 = v772 == false;
            if (v773){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v772);
            } else {
            }
            bool v775;
            v775 = 0 <= v694;
            bool v777;
            if (v775){
                bool v776;
                v776 = v694 < 64;
                v777 = v776;
            } else {
                v777 = false;
            }
            bool v778;
            v778 = v777 == false;
            if (v778){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v777);
            } else {
            }
            bool v780;
            v780 = 0 <= v711;
            bool v781;
            v781 = v780 && v712;
            bool v782;
            v782 = v781 == false;
            if (v782){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v781);
            } else {
            }
            bool v784;
            v784 = 0 <= v710;
            bool v786;
            if (v784){
                bool v785;
                v785 = v710 < 64;
                v786 = v785;
            } else {
                v786 = false;
            }
            bool v787;
            v787 = v786 == false;
            if (v787){
                assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v786);
            } else {
            }
            int v789;
            v789 = v710 * 64;
            int v790;
            v790 = v711 + v695;
            int v791;
            v791 = v789 + v694;
            bool v792[4];
            int v793;
            v793 = 0;
            while (while_method_6(v793)){
                int v795;
                v795 = 0;
                while (while_method_0(v795)){
                    assert("Tensor range check" && 0 <= v793 && v793 < 1);
                    assert("Tensor range check" && 0 <= v795 && v795 < 4);
                    int v797;
                    v797 = 4 * v793;
                    int v798;
                    v798 = v797 + v795;
                    float v799;
                    v799 = v721[v798];
                    bool v800;
                    v800 = v799 == 0.0f;
                    bool v801;
                    v801 = v800 != true;
                    assert("Tensor range check" && 0 <= v793 && v793 < 1);
                    assert("Tensor range check" && 0 <= v795 && v795 < 4);
                    v792[v798] = v801;
                    v795 += 1 ;
                }
                v793 += 1 ;
            }
            bool v802;
            v802 = false;
            int v803;
            v803 = 0;
            while (while_method_6(v803)){
                int v805;
                v805 = 0;
                while (while_method_0(v805)){
                    assert("Tensor range check" && 0 <= v803 && v803 < 1);
                    assert("Tensor range check" && 0 <= v805 && v805 < 4);
                    int v807;
                    v807 = 4 * v803;
                    int v808;
                    v808 = v807 + v805;
                    bool v809;
                    v809 = v792[v808];
                    bool v810;
                    v810 = v802 || v809;
                    v802 = v810;
                    v805 += 1 ;
                }
                v803 += 1 ;
            }
            auto v811 = cooperative_groups::coalesced_threads();
            int v812;
            v812 = threadIdx.x;
            int v813;
            v813 = v812 / 4;
            auto v814 = cooperative_groups::labeled_partition(v811,v813);
            Closure8 v815{};
            bool v816;
            v816 = cooperative_groups::reduce(v814, v802, v815);
            if (v816){
                float v817[4];
                int v818;
                v818 = 0;
                while (while_method_6(v818)){
                    int v820;
                    v820 = 0;
                    while (while_method_0(v820)){
                        assert("Tensor range check" && 0 <= v818 && v818 < 1);
                        assert("Tensor range check" && 0 <= v820 && v820 < 4);
                        int v822;
                        v822 = 4 * v818;
                        int v823;
                        v823 = v822 + v820;
                        float v824;
                        v824 = v720[v823];
                        float v825;
                        v825 = v721[v823];
                        float v826;
                        v826 = v824 + v825;
                        bool v827;
                        v827 = 0.0f >= v826;
                        float v828;
                        if (v827){
                            v828 = 0.0f;
                        } else {
                            v828 = v826;
                        }
                        assert("Tensor range check" && 0 <= v818 && v818 < 1);
                        assert("Tensor range check" && 0 <= v820 && v820 < 4);
                        v817[v823] = v828;
                        v820 += 1 ;
                    }
                    v818 += 1 ;
                }
                float v829[4];
                int v830;
                v830 = 0;
                while (while_method_6(v830)){
                    int v832;
                    v832 = 0;
                    while (while_method_0(v832)){
                        assert("Tensor range check" && 0 <= v830 && v830 < 1);
                        assert("Tensor range check" && 0 <= v832 && v832 < 4);
                        int v834;
                        v834 = 4 * v830;
                        int v835;
                        v835 = v834 + v832;
                        float v836;
                        v836 = v817[v835];
                        bool v837;
                        v837 = 0.0f >= v836;
                        float v838;
                        if (v837){
                            v838 = 0.0f;
                        } else {
                            v838 = v836;
                        }
                        assert("Tensor range check" && 0 <= v830 && v830 < 1);
                        assert("Tensor range check" && 0 <= v832 && v832 < 4);
                        v829[v835] = v838;
                        v832 += 1 ;
                    }
                    v830 += 1 ;
                }
                float v839;
                v839 = 0.0f;
                int v840;
                v840 = 0;
                while (while_method_6(v840)){
                    int v842;
                    v842 = 0;
                    while (while_method_0(v842)){
                        assert("Tensor range check" && 0 <= v840 && v840 < 1);
                        assert("Tensor range check" && 0 <= v842 && v842 < 4);
                        int v844;
                        v844 = 4 * v840;
                        int v845;
                        v845 = v844 + v842;
                        float v846;
                        v846 = v829[v845];
                        float v847;
                        v847 = v839 + v846;
                        v839 = v847;
                        v842 += 1 ;
                    }
                    v840 += 1 ;
                }
                auto v848 = cooperative_groups::coalesced_threads();
                int v849;
                v849 = threadIdx.x;
                int v850;
                v850 = v849 / 4;
                auto v851 = cooperative_groups::labeled_partition(v848,v850);
                Closure1 v852{};
                float v853;
                v853 = cooperative_groups::reduce(v851, v839, v852);
                float v854[4];
                int v855;
                v855 = 0;
                while (while_method_6(v855)){
                    int v857;
                    v857 = 0;
                    while (while_method_0(v857)){
                        assert("Tensor range check" && 0 <= v855 && v855 < 1);
                        assert("Tensor range check" && 0 <= v857 && v857 < 4);
                        int v859;
                        v859 = 4 * v855;
                        int v860;
                        v860 = v859 + v857;
                        float v861;
                        v861 = v829[v860];
                        bool v862;
                        v862 = v853 == 0.0f;
                        bool v863;
                        v863 = v862 != true;
                        float v865;
                        if (v863){
                            float v864;
                            v864 = v861 / v853;
                            v865 = v864;
                        } else {
                            v865 = 0.0625f;
                        }
                        assert("Tensor range check" && 0 <= v855 && v855 < 1);
                        assert("Tensor range check" && 0 <= v857 && v857 < 4);
                        v854[v860] = v865;
                        v857 += 1 ;
                    }
                    v855 += 1 ;
                }
                float v866[4];
                int v867;
                v867 = 0;
                while (while_method_6(v867)){
                    int v869;
                    v869 = 0;
                    while (while_method_0(v869)){
                        assert("Tensor range check" && 0 <= v867 && v867 < 1);
                        assert("Tensor range check" && 0 <= v869 && v869 < 4);
                        int v871;
                        v871 = 4 * v867;
                        int v872;
                        v872 = v871 + v869;
                        float v873;
                        v873 = v719[v872];
                        float v874;
                        v874 = v854[v872];
                        float v875;
                        v875 = v873 + v874;
                        assert("Tensor range check" && 0 <= v867 && v867 < 1);
                        assert("Tensor range check" && 0 <= v869 && v869 < 4);
                        v866[v872] = v875;
                        v869 += 1 ;
                    }
                    v867 += 1 ;
                }
                float v876[4];
                int v877;
                v877 = 0;
                while (while_method_6(v877)){
                    int v879;
                    v879 = 0;
                    while (while_method_0(v879)){
                        assert("Tensor range check" && 0 <= v877 && v877 < 1);
                        assert("Tensor range check" && 0 <= v879 && v879 < 4);
                        int v881;
                        v881 = 4 * v877;
                        int v882;
                        v882 = v881 + v879;
                        float v883;
                        v883 = v866[v882];
                        float v884;
                        v884 = -v883;
                        bool v885;
                        v885 = v883 >= v884;
                        float v886;
                        if (v885){
                            v886 = v883;
                        } else {
                            v886 = v884;
                        }
                        assert("Tensor range check" && 0 <= v877 && v877 < 1);
                        assert("Tensor range check" && 0 <= v879 && v879 < 4);
                        v876[v882] = v886;
                        v879 += 1 ;
                    }
                    v877 += 1 ;
                }
                float v887;
                v887 = 0.0f;
                int v888;
                v888 = 0;
                while (while_method_6(v888)){
                    int v890;
                    v890 = 0;
                    while (while_method_0(v890)){
                        assert("Tensor range check" && 0 <= v888 && v888 < 1);
                        assert("Tensor range check" && 0 <= v890 && v890 < 4);
                        int v892;
                        v892 = 4 * v888;
                        int v893;
                        v893 = v892 + v890;
                        float v894;
                        v894 = v876[v893];
                        float v895;
                        v895 = v887 + v894;
                        v887 = v895;
                        v890 += 1 ;
                    }
                    v888 += 1 ;
                }
                auto v896 = cooperative_groups::coalesced_threads();
                int v897;
                v897 = threadIdx.x;
                int v898;
                v898 = v897 / 4;
                auto v899 = cooperative_groups::labeled_partition(v896,v898);
                float v900;
                v900 = cooperative_groups::reduce(v899, v887, v852);
                bool v901;
                v901 = v900 > 100.0f;
                float v903;
                if (v901){
                    float v902;
                    v902 = 100.0f / v900;
                    v903 = v902;
                } else {
                    v903 = 1.0f;
                }
                float v904[4];
                int v905;
                v905 = 0;
                while (while_method_6(v905)){
                    int v907;
                    v907 = 0;
                    while (while_method_0(v907)){
                        assert("Tensor range check" && 0 <= v905 && v905 < 1);
                        assert("Tensor range check" && 0 <= v907 && v907 < 4);
                        int v909;
                        v909 = 4 * v905;
                        int v910;
                        v910 = v909 + v907;
                        float v911;
                        v911 = v876[v910];
                        float v912;
                        v912 = v903 * v911;
                        assert("Tensor range check" && 0 <= v905 && v905 < 1);
                        assert("Tensor range check" && 0 <= v907 && v907 < 4);
                        v904[v910] = v912;
                        v907 += 1 ;
                    }
                    v905 += 1 ;
                }
                float v913[4];
                float v914[4];
                int v915;
                v915 = 0;
                while (while_method_6(v915)){
                    int v917;
                    v917 = 0;
                    while (while_method_0(v917)){
                        assert("Tensor range check" && 0 <= v915 && v915 < 1);
                        assert("Tensor range check" && 0 <= v917 && v917 < 4);
                        int v919;
                        v919 = 4 * v915;
                        int v920;
                        v920 = v919 + v917;
                        float v921;
                        v921 = v719[v920];
                        float v922;
                        v922 = v720[v920];
                        float v923;
                        v923 = v721[v920];
                        float v924;
                        v924 = v722[v920];
                        float v925;
                        v925 = v723[v920];
                        float v926;
                        v926 = v724[v920];
                        float v927;
                        v927 = v725[v920];
                        float v928;
                        v928 = v924 + v926;
                        float v929;
                        v929 = v925 + v927;
                        assert("Tensor range check" && 0 <= v915 && v915 < 1);
                        assert("Tensor range check" && 0 <= v917 && v917 < 4);
                        v913[v920] = v928;
                        v914[v920] = v929;
                        v917 += 1 ;
                    }
                    v915 += 1 ;
                }
                int v930;
                v930 = 0;
                while (while_method_6(v930)){
                    int v932;
                    v932 = 0;
                    while (while_method_0(v932)){
                        assert("Tensor range check" && 0 <= v930 && v930 < 1);
                        assert("Tensor range check" && 0 <= v932 && v932 < 4);
                        int v934;
                        v934 = 4 * v930;
                        int v935;
                        v935 = v934 + v932;
                        float v936;
                        v936 = v904[v935];
                        float v937;
                        v937 = v817[v935];
                        float v938;
                        v938 = v913[v935];
                        float v939;
                        v939 = v914[v935];
                        assert("Tensor range check" && 0 <= v930 && v930 < 1);
                        assert("Tensor range check" && 0 <= v932 && v932 < 4);
                        v719[v935] = v936;
                        v720[v935] = v937;
                        v721[v935] = 0.0f;
                        v722[v935] = v938;
                        v723[v935] = v939;
                        v724[v935] = 0.0f;
                        v725[v935] = 0.0f;
                        v932 += 1 ;
                    }
                    v930 += 1 ;
                }
            } else {
            }
            assert("Tensor range check" && 0 <= v711 && v711 < 4);
            assert("Tensor range check" && 0 <= v710 && v710 < 64);
            int v940;
            v940 = 0;
            while (while_method_6(v940)){
                assert("Tensor range check" && 0 <= v940 && v940 < 1);
                int v942;
                v942 = 16 * v940;
                int v943;
                v943 = v942 + v718;
                assert("Tensor range check" && 0 <= v940 && v940 < 1);
                int v944;
                v944 = 4 * v940;
                int4* v945;
                v945 = reinterpret_cast<int4*>(v719 + v944);
                int4* v946;
                v946 = reinterpret_cast<int4*>(v648 + v943);
                assert("Pointer alignment check" && (unsigned long long)(v945) % 4 == 0 && (unsigned long long)(v946) % 4 == 0);
                *v946 = *v945;
                int4* v947;
                v947 = reinterpret_cast<int4*>(v720 + v944);
                int4* v948;
                v948 = reinterpret_cast<int4*>(v650 + v943);
                assert("Pointer alignment check" && (unsigned long long)(v947) % 4 == 0 && (unsigned long long)(v948) % 4 == 0);
                *v948 = *v947;
                int4* v949;
                v949 = reinterpret_cast<int4*>(v721 + v944);
                int4* v950;
                v950 = reinterpret_cast<int4*>(v652 + v943);
                assert("Pointer alignment check" && (unsigned long long)(v949) % 4 == 0 && (unsigned long long)(v950) % 4 == 0);
                *v950 = *v949;
                int4* v951;
                v951 = reinterpret_cast<int4*>(v722 + v944);
                int4* v952;
                v952 = reinterpret_cast<int4*>(v654 + v943);
                assert("Pointer alignment check" && (unsigned long long)(v951) % 4 == 0 && (unsigned long long)(v952) % 4 == 0);
                *v952 = *v951;
                int4* v953;
                v953 = reinterpret_cast<int4*>(v723 + v944);
                int4* v954;
                v954 = reinterpret_cast<int4*>(v656 + v943);
                assert("Pointer alignment check" && (unsigned long long)(v953) % 4 == 0 && (unsigned long long)(v954) % 4 == 0);
                *v954 = *v953;
                int4* v955;
                v955 = reinterpret_cast<int4*>(v724 + v944);
                int4* v956;
                v956 = reinterpret_cast<int4*>(v658 + v943);
                assert("Pointer alignment check" && (unsigned long long)(v955) % 4 == 0 && (unsigned long long)(v956) % 4 == 0);
                *v956 = *v955;
                int4* v957;
                v957 = reinterpret_cast<int4*>(v725 + v944);
                int4* v958;
                v958 = reinterpret_cast<int4*>(v660 + v943);
                assert("Pointer alignment check" && (unsigned long long)(v957) % 4 == 0 && (unsigned long long)(v958) % 4 == 0);
                *v958 = *v957;
                v940 += 1 ;
            }
            v705 += 24 ;
        }
        v641.sync() ;
        v26 += 1 ;
    }
    cooperative_groups::grid_group & v959 = v25.v1;
    cooperative_groups::grid_group & v960 = v959;
    int v961;
    v961 = threadIdx.x;
    int v962;
    v962 = blockIdx.x;
    int v963;
    v963 = v962 * 256;
    int v964;
    v964 = v961 + v963;
    int v965;
    v965 = v964;
    while (while_method_0(v965)){
        bool v967;
        v967 = 0 <= v965;
        bool v968;
        v968 = v967 == false;
        if (v968){
            assert("The index needs to be zero or positive." && v967);
        } else {
        }
        int v970;
        v970 = v965 % 1;
        bool v971;
        v971 = v965 < 4;
        bool v972;
        v972 = v971 == false;
        if (v972){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v971);
        } else {
        }
        assert("Tensor range check" && 0 <= v965 && v965 < 4);
        assert("Tensor range check" && 0 <= v970 && v970 < 1);
        int v974;
        v974 = 4 * v970;
        int v975;
        v975 = 4 * v965;
        int v976;
        v976 = v975 + v974;
        assert("Tensor range check" && 0 <= v965 && v965 < 4);
        assert("Tensor range check" && 0 <= v970 && v970 < 1);
        float v977[4];
        float v978[4];
        float v979[4];
        int4* v980;
        v980 = reinterpret_cast<int4*>(v2 + v976);
        int4* v981;
        v981 = reinterpret_cast<int4*>(v977 + 0);
        assert("Pointer alignment check" && (unsigned long long)(v980) % 4 == 0 && (unsigned long long)(v981) % 4 == 0);
        *v981 = *v980;
        int4* v982;
        v982 = reinterpret_cast<int4*>(v3 + v976);
        int4* v983;
        v983 = reinterpret_cast<int4*>(v978 + 0);
        assert("Pointer alignment check" && (unsigned long long)(v982) % 4 == 0 && (unsigned long long)(v983) % 4 == 0);
        *v983 = *v982;
        // Pushing the loop unrolling to: 0
        int v984;
        v984 = 0;
        #pragma unroll
        while (while_method_0(v984)){
            assert("Tensor range check" && 0 <= v984 && v984 < 4);
            float v986;
            v986 = v977[v984];
            float v987;
            v987 = v978[v984];
            bool v988;
            v988 = v987 == 0.0f;
            bool v989;
            v989 = v988 != true;
            float v991;
            if (v989){
                float v990;
                v990 = v986 / v987;
                v991 = v990;
            } else {
                v991 = 0.0f;
            }
            assert("Tensor range check" && 0 <= v984 && v984 < 4);
            v979[v984] = v991;
            v984 += 1 ;
        }
        // Poping the loop unrolling to: 0
        int4* v992;
        v992 = reinterpret_cast<int4*>(v979 + 0);
        int4* v993;
        v993 = reinterpret_cast<int4*>(v4 + v976);
        assert("Pointer alignment check" && (unsigned long long)(v992) % 4 == 0 && (unsigned long long)(v993) % 4 == 0);
        *v993 = *v992;
        v965 += 6144 ;
    }
    v960.sync() ;
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
    v5 = {'model_ptrs': v4}
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
    v43 = "Going to run the NL Holdem pefromance kernel."
    print(v42.format(v43),end="")
    del v42, v43
    v44 = time.perf_counter()
    v45 = []
    v46 = cp.zeros(16,dtype=cp.float32) # type: ignore
    v47 = cp.zeros(16,dtype=cp.float32) # type: ignore
    v48 = cp.empty(16,dtype=cp.float32)
    v49 = cp.cuda.Device().attributes['MultiProcessorCount']
    v50 = v49 == 24
    del v49
    v51 = v50 == False
    if v51:
        v52 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
        assert v50, v52
        del v52
    else:
        pass
    del v50, v51
    v53 = 0
    v54 = raw_module.get_function(f"entry{v53}")
    del v53
    v54.max_dynamic_shared_size_bytes = 98304 
    print(f'Threads per block, blocks per grid: {256}, {24}')
    v54((24,),(256,),(v9, v8, v46, v47, v48),shared_mem=98304)
    del v46, v47, v48, v54
    cp.cuda.get_current_stream().synchronize()
    v55 = time.perf_counter()
    v58 = "{}"
    v59 = "The time it took to run the kernel (in seconds) is: "
    print(v58.format(v59),end="")
    del v58, v59
    v60 = v55 - v44
    del v44, v55
    v63 = "{:.6f}\n"
    print(v63.format(v60),end="")
    del v60, v63
    v64, v65, v66, v67, v68 = method33(v36)
    del v36
    v69 = 204570624
    v70 = 12419088
    return method61(v64, v65, v66, v67, v68, v9, v69, v8, v70, v45)

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
