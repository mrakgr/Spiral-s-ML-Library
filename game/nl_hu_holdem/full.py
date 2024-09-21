kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <mma.h>
using namespace nvcuda;
#include <cuda/pipeline>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups.h>
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

struct Union1;
struct Union2;
struct Union0;
__device__ int f_1(unsigned char * v0);
__device__ void f_3(unsigned char * v0);
__device__ Union1 f_2(unsigned char * v0);
__device__ Union2 f_5(unsigned char * v0);
__device__ static_array<Union2,2> f_4(unsigned char * v0);
__device__ Union0 f_0(unsigned char * v0);
struct Union5;
struct Union4;
struct Union3;
struct Tuple0;
struct Union6;
struct Union7;
struct Tuple1;
__device__ unsigned long long f_7(unsigned char * v0);
__device__ int f_8(unsigned char * v0);
struct Tuple2;
__device__ unsigned char f_13(unsigned char * v0);
__device__ unsigned char f_12(unsigned char * v0);
__device__ static_array<unsigned char,2> f_11(unsigned char * v0);
__device__ int f_14(unsigned char * v0);
__device__ static_array<unsigned char,3> f_15(unsigned char * v0);
__device__ static_array<unsigned char,5> f_16(unsigned char * v0);
__device__ static_array<unsigned char,4> f_17(unsigned char * v0);
__device__ Tuple2 f_10(unsigned char * v0);
struct Tuple3;
__device__ int f_19(unsigned char * v0);
__device__ Tuple3 f_18(unsigned char * v0);
__device__ Union4 f_9(unsigned char * v0);
__device__ int f_20(unsigned char * v0);
__device__ static_array_list<unsigned char,5> f_22(unsigned char * v0);
struct Tuple4;
__device__ Tuple4 f_23(unsigned char * v0);
struct Tuple5;
__device__ int f_25(unsigned char * v0);
__device__ Tuple5 f_24(unsigned char * v0);
struct Tuple6;
__device__ Tuple6 f_26(unsigned char * v0);
struct Tuple7;
__device__ Tuple0 f_29(unsigned char * v0);
__device__ Tuple0 f_28(unsigned char * v0);
__device__ Tuple7 f_27(unsigned char * v0);
__device__ Union6 f_21(unsigned char * v0);
__device__ int f_30(unsigned char * v0);
__device__ Tuple1 f_6(unsigned char * v0);
struct StackMut0;
struct Tuple8;
struct Tuple9;
struct Tuple10;
__device__ unsigned int loop_34(unsigned int v0, curandStatePhilox4_32_10_t & v1);
__device__ Tuple10 draw_card_33(curandStatePhilox4_32_10_t & v0, unsigned long long v1);
__device__ Tuple8 draw_cards_32(curandStatePhilox4_32_10_t & v0, unsigned long long v1);
__device__ static_array_list<unsigned char,5> get_community_cards_35(Union5 v0, static_array<unsigned char,3> v1);
__device__ bool player_can_act_37(int v0, static_array<static_array<unsigned char,2>,2> v1, static_array<int,2> v2, int v3, static_array<int,2> v4, Union5 v5);
__device__ Union4 go_next_street_38(int v0, static_array<static_array<unsigned char,2>,2> v1, static_array<int,2> v2, int v3, static_array<int,2> v4, Union5 v5);
__device__ Union4 try_round_36(int v0, static_array<static_array<unsigned char,2>,2> v1, static_array<int,2> v2, int v3, static_array<int,2> v4, Union5 v5);
struct Tuple11;
__device__ Tuple11 draw_cards_39(curandStatePhilox4_32_10_t & v0, unsigned long long v1);
struct Tuple12;
__device__ Tuple12 draw_cards_40(curandStatePhilox4_32_10_t & v0, unsigned long long v1);
__device__ static_array_list<unsigned char,5> get_community_cards_41(Union5 v0, static_array<unsigned char,1> v1);
struct Union8;
struct Tuple13;
__device__ void method_42(unsigned int v0, float * v1, int v2);
__device__ void method_43(unsigned int v0, float * v1, int v2);
__device__ int int_range_44(int v0, int v1, curandStatePhilox4_32_10_t & v2);
struct Union9;
__device__ void method_45(unsigned int * v0, int v1, float * v2);
struct Tuple14;
struct Tuple15;
struct Tuple16;
struct Tuple17;
__device__ Tuple14 method_46(curandStatePhilox4_32_10_t & v0, int * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int v9, int v10);
struct Union10;
struct Tuple18;
__device__ int loop_50(static_array<float,6> v0, float v1, int v2);
__device__ int pick_discrete__49(static_array<float,6> v0, float v1);
__device__ int sample_discrete__48(static_array<float,6> v0, curandStatePhilox4_32_10_t & v1);
__device__ Union1 sample_discrete_47(static_array<Tuple18,6> v0, curandStatePhilox4_32_10_t & v1);
struct Tuple19;
struct Tuple20;
struct Union11;
struct Tuple21;
struct Union12;
struct Tuple22;
struct Tuple23;
struct Union13;
struct Union14;
struct Union15;
struct Union16;
struct Union17;
__device__ Tuple0 score_51(static_array<unsigned char,7> v0);
__device__ void play_loop_31(unsigned char * v0, unsigned long long v1, unsigned char * v2, unsigned long long v3, StackMut0 & v4, Union4 v5);
__device__ void f_53(unsigned char * v0, unsigned long long v1);
__device__ void f_54(unsigned char * v0, int v1);
__device__ void f_55(unsigned char * v0);
__device__ void f_57(unsigned char * v0, int v1);
__device__ void f_61(unsigned char * v0, unsigned char v1);
__device__ void f_60(unsigned char * v0, unsigned char v1);
__device__ void f_59(unsigned char * v0, static_array<unsigned char,2> v1);
__device__ void f_62(unsigned char * v0, int v1);
__device__ void f_63(unsigned char * v0, static_array<unsigned char,3> v1);
__device__ void f_64(unsigned char * v0, static_array<unsigned char,5> v1);
__device__ void f_65(unsigned char * v0, static_array<unsigned char,4> v1);
__device__ void f_58(unsigned char * v0, int v1, static_array<static_array<unsigned char,2>,2> v2, static_array<int,2> v3, int v4, static_array<int,2> v5, Union5 v6);
__device__ void f_67(unsigned char * v0, int v1);
__device__ void f_66(unsigned char * v0, int v1, static_array<static_array<unsigned char,2>,2> v2, static_array<int,2> v3, int v4, static_array<int,2> v5, Union5 v6, Union1 v7);
__device__ void f_56(unsigned char * v0, Union4 v1);
__device__ void f_68(unsigned char * v0, int v1);
__device__ void f_70(unsigned char * v0, static_array_list<unsigned char,5> v1);
__device__ void f_71(unsigned char * v0, int v1, int v2);
__device__ void f_73(unsigned char * v0, int v1);
__device__ void f_72(unsigned char * v0, int v1, Union1 v2);
__device__ void f_74(unsigned char * v0, int v1, static_array<unsigned char,2> v2);
__device__ void f_77(unsigned char * v0, static_array<unsigned char,5> v1, char v2);
__device__ void f_76(unsigned char * v0, static_array<unsigned char,5> v1, char v2);
__device__ void f_75(unsigned char * v0, int v1, static_array<Tuple0,2> v2, int v3);
__device__ void f_69(unsigned char * v0, Union6 v1);
__device__ void f_78(unsigned char * v0, Union2 v1);
__device__ void f_79(unsigned char * v0, int v1);
__device__ void f_52(unsigned char * v0, unsigned long long v1, Union3 v2, dynamic_array_list<Union6,128> v3, static_array<Union2,2> v4, Union7 v5);
struct Union1_0 { // A_All_In
};
struct Union1_1 { // A_Call
};
struct Union1_2 { // A_Fold
};
struct Union1_3 { // A_Raise
    int v0;
    __device__ Union1_3(int t0) : v0(t0) {}
    __device__ Union1_3() = delete;
};
struct Union1 {
    union {
        Union1_0 case0; // A_All_In
        Union1_1 case1; // A_Call
        Union1_2 case2; // A_Fold
        Union1_3 case3; // A_Raise
    };
    unsigned char tag{255};
    __device__ Union1() {}
    __device__ Union1(Union1_0 t) : tag(0), case0(t) {} // A_All_In
    __device__ Union1(Union1_1 t) : tag(1), case1(t) {} // A_Call
    __device__ Union1(Union1_2 t) : tag(2), case2(t) {} // A_Fold
    __device__ Union1(Union1_3 t) : tag(3), case3(t) {} // A_Raise
    __device__ Union1(Union1 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(x.case0); break; // A_All_In
            case 1: new (&this->case1) Union1_1(x.case1); break; // A_Call
            case 2: new (&this->case2) Union1_2(x.case2); break; // A_Fold
            case 3: new (&this->case3) Union1_3(x.case3); break; // A_Raise
        }
    }
    __device__ Union1(Union1 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union1_0(std::move(x.case0)); break; // A_All_In
            case 1: new (&this->case1) Union1_1(std::move(x.case1)); break; // A_Call
            case 2: new (&this->case2) Union1_2(std::move(x.case2)); break; // A_Fold
            case 3: new (&this->case3) Union1_3(std::move(x.case3)); break; // A_Raise
        }
    }
    __device__ Union1 & operator=(Union1 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // A_All_In
                case 1: this->case1 = x.case1; break; // A_Call
                case 2: this->case2 = x.case2; break; // A_Fold
                case 3: this->case3 = x.case3; break; // A_Raise
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
                case 0: this->case0 = std::move(x.case0); break; // A_All_In
                case 1: this->case1 = std::move(x.case1); break; // A_Call
                case 2: this->case2 = std::move(x.case2); break; // A_Fold
                case 3: this->case3 = std::move(x.case3); break; // A_Raise
            }
        } else {
            this->~Union1();
            new (this) Union1{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union1() {
        switch(this->tag){
            case 0: this->case0.~Union1_0(); break; // A_All_In
            case 1: this->case1.~Union1_1(); break; // A_Call
            case 2: this->case2.~Union1_2(); break; // A_Fold
            case 3: this->case3.~Union1_3(); break; // A_Raise
        }
        this->tag = 255;
    }
};
struct Union2_0 { // Computer
};
struct Union2_1 { // Human
};
struct Union2_2 { // Random
};
struct Union2 {
    union {
        Union2_0 case0; // Computer
        Union2_1 case1; // Human
        Union2_2 case2; // Random
    };
    unsigned char tag{255};
    __device__ Union2() {}
    __device__ Union2(Union2_0 t) : tag(0), case0(t) {} // Computer
    __device__ Union2(Union2_1 t) : tag(1), case1(t) {} // Human
    __device__ Union2(Union2_2 t) : tag(2), case2(t) {} // Random
    __device__ Union2(Union2 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(x.case0); break; // Computer
            case 1: new (&this->case1) Union2_1(x.case1); break; // Human
            case 2: new (&this->case2) Union2_2(x.case2); break; // Random
        }
    }
    __device__ Union2(Union2 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union2_0(std::move(x.case0)); break; // Computer
            case 1: new (&this->case1) Union2_1(std::move(x.case1)); break; // Human
            case 2: new (&this->case2) Union2_2(std::move(x.case2)); break; // Random
        }
    }
    __device__ Union2 & operator=(Union2 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Computer
                case 1: this->case1 = x.case1; break; // Human
                case 2: this->case2 = x.case2; break; // Random
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
                case 0: this->case0 = std::move(x.case0); break; // Computer
                case 1: this->case1 = std::move(x.case1); break; // Human
                case 2: this->case2 = std::move(x.case2); break; // Random
            }
        } else {
            this->~Union2();
            new (this) Union2{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union2() {
        switch(this->tag){
            case 0: this->case0.~Union2_0(); break; // Computer
            case 1: this->case1.~Union2_1(); break; // Human
            case 2: this->case2.~Union2_2(); break; // Random
        }
        this->tag = 255;
    }
};
struct Union0_0 { // ActionSelected
    Union1 v0;
    __device__ Union0_0(Union1 t0) : v0(t0) {}
    __device__ Union0_0() = delete;
};
struct Union0_1 { // PlayerChanged
    static_array<Union2,2> v0;
    __device__ Union0_1(static_array<Union2,2> t0) : v0(t0) {}
    __device__ Union0_1() = delete;
};
struct Union0_2 { // StartGame
};
struct Union0 {
    union {
        Union0_0 case0; // ActionSelected
        Union0_1 case1; // PlayerChanged
        Union0_2 case2; // StartGame
    };
    unsigned char tag{255};
    __device__ Union0() {}
    __device__ Union0(Union0_0 t) : tag(0), case0(t) {} // ActionSelected
    __device__ Union0(Union0_1 t) : tag(1), case1(t) {} // PlayerChanged
    __device__ Union0(Union0_2 t) : tag(2), case2(t) {} // StartGame
    __device__ Union0(Union0 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(x.case0); break; // ActionSelected
            case 1: new (&this->case1) Union0_1(x.case1); break; // PlayerChanged
            case 2: new (&this->case2) Union0_2(x.case2); break; // StartGame
        }
    }
    __device__ Union0(Union0 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union0_0(std::move(x.case0)); break; // ActionSelected
            case 1: new (&this->case1) Union0_1(std::move(x.case1)); break; // PlayerChanged
            case 2: new (&this->case2) Union0_2(std::move(x.case2)); break; // StartGame
        }
    }
    __device__ Union0 & operator=(Union0 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // ActionSelected
                case 1: this->case1 = x.case1; break; // PlayerChanged
                case 2: this->case2 = x.case2; break; // StartGame
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
                case 0: this->case0 = std::move(x.case0); break; // ActionSelected
                case 1: this->case1 = std::move(x.case1); break; // PlayerChanged
                case 2: this->case2 = std::move(x.case2); break; // StartGame
            }
        } else {
            this->~Union0();
            new (this) Union0{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union0() {
        switch(this->tag){
            case 0: this->case0.~Union0_0(); break; // ActionSelected
            case 1: this->case1.~Union0_1(); break; // PlayerChanged
            case 2: this->case2.~Union0_2(); break; // StartGame
        }
        this->tag = 255;
    }
};
struct Union5_0 { // Flop
    static_array<unsigned char,3> v0;
    __device__ Union5_0(static_array<unsigned char,3> t0) : v0(t0) {}
    __device__ Union5_0() = delete;
};
struct Union5_1 { // Preflop
};
struct Union5_2 { // River
    static_array<unsigned char,5> v0;
    __device__ Union5_2(static_array<unsigned char,5> t0) : v0(t0) {}
    __device__ Union5_2() = delete;
};
struct Union5_3 { // Turn
    static_array<unsigned char,4> v0;
    __device__ Union5_3(static_array<unsigned char,4> t0) : v0(t0) {}
    __device__ Union5_3() = delete;
};
struct Union5 {
    union {
        Union5_0 case0; // Flop
        Union5_1 case1; // Preflop
        Union5_2 case2; // River
        Union5_3 case3; // Turn
    };
    unsigned char tag{255};
    __device__ Union5() {}
    __device__ Union5(Union5_0 t) : tag(0), case0(t) {} // Flop
    __device__ Union5(Union5_1 t) : tag(1), case1(t) {} // Preflop
    __device__ Union5(Union5_2 t) : tag(2), case2(t) {} // River
    __device__ Union5(Union5_3 t) : tag(3), case3(t) {} // Turn
    __device__ Union5(Union5 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union5_0(x.case0); break; // Flop
            case 1: new (&this->case1) Union5_1(x.case1); break; // Preflop
            case 2: new (&this->case2) Union5_2(x.case2); break; // River
            case 3: new (&this->case3) Union5_3(x.case3); break; // Turn
        }
    }
    __device__ Union5(Union5 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union5_0(std::move(x.case0)); break; // Flop
            case 1: new (&this->case1) Union5_1(std::move(x.case1)); break; // Preflop
            case 2: new (&this->case2) Union5_2(std::move(x.case2)); break; // River
            case 3: new (&this->case3) Union5_3(std::move(x.case3)); break; // Turn
        }
    }
    __device__ Union5 & operator=(Union5 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Flop
                case 1: this->case1 = x.case1; break; // Preflop
                case 2: this->case2 = x.case2; break; // River
                case 3: this->case3 = x.case3; break; // Turn
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
                case 0: this->case0 = std::move(x.case0); break; // Flop
                case 1: this->case1 = std::move(x.case1); break; // Preflop
                case 2: this->case2 = std::move(x.case2); break; // River
                case 3: this->case3 = std::move(x.case3); break; // Turn
            }
        } else {
            this->~Union5();
            new (this) Union5{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union5() {
        switch(this->tag){
            case 0: this->case0.~Union5_0(); break; // Flop
            case 1: this->case1.~Union5_1(); break; // Preflop
            case 2: this->case2.~Union5_2(); break; // River
            case 3: this->case3.~Union5_3(); break; // Turn
        }
        this->tag = 255;
    }
};
struct Union4_0 { // G_Flop
    static_array<static_array<unsigned char,2>,2> v1;
    static_array<int,2> v2;
    static_array<int,2> v4;
    Union5 v5;
    int v0;
    int v3;
    __device__ Union4_0(int t0, static_array<static_array<unsigned char,2>,2> t1, static_array<int,2> t2, int t3, static_array<int,2> t4, Union5 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_0() = delete;
};
struct Union4_1 { // G_Fold
    static_array<static_array<unsigned char,2>,2> v1;
    static_array<int,2> v2;
    static_array<int,2> v4;
    Union5 v5;
    int v0;
    int v3;
    __device__ Union4_1(int t0, static_array<static_array<unsigned char,2>,2> t1, static_array<int,2> t2, int t3, static_array<int,2> t4, Union5 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_1() = delete;
};
struct Union4_2 { // G_Preflop
};
struct Union4_3 { // G_River
    static_array<static_array<unsigned char,2>,2> v1;
    static_array<int,2> v2;
    static_array<int,2> v4;
    Union5 v5;
    int v0;
    int v3;
    __device__ Union4_3(int t0, static_array<static_array<unsigned char,2>,2> t1, static_array<int,2> t2, int t3, static_array<int,2> t4, Union5 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_3() = delete;
};
struct Union4_4 { // G_Round
    static_array<static_array<unsigned char,2>,2> v1;
    static_array<int,2> v2;
    static_array<int,2> v4;
    Union5 v5;
    int v0;
    int v3;
    __device__ Union4_4(int t0, static_array<static_array<unsigned char,2>,2> t1, static_array<int,2> t2, int t3, static_array<int,2> t4, Union5 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_4() = delete;
};
struct Union4_5 { // G_Round'
    static_array<static_array<unsigned char,2>,2> v1;
    static_array<int,2> v2;
    static_array<int,2> v4;
    Union5 v5;
    Union1 v6;
    int v0;
    int v3;
    __device__ Union4_5(int t0, static_array<static_array<unsigned char,2>,2> t1, static_array<int,2> t2, int t3, static_array<int,2> t4, Union5 t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __device__ Union4_5() = delete;
};
struct Union4_6 { // G_Showdown
    static_array<static_array<unsigned char,2>,2> v1;
    static_array<int,2> v2;
    static_array<int,2> v4;
    Union5 v5;
    int v0;
    int v3;
    __device__ Union4_6(int t0, static_array<static_array<unsigned char,2>,2> t1, static_array<int,2> t2, int t3, static_array<int,2> t4, Union5 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_6() = delete;
};
struct Union4_7 { // G_Turn
    static_array<static_array<unsigned char,2>,2> v1;
    static_array<int,2> v2;
    static_array<int,2> v4;
    Union5 v5;
    int v0;
    int v3;
    __device__ Union4_7(int t0, static_array<static_array<unsigned char,2>,2> t1, static_array<int,2> t2, int t3, static_array<int,2> t4, Union5 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_7() = delete;
};
struct Union4 {
    union {
        Union4_0 case0; // G_Flop
        Union4_1 case1; // G_Fold
        Union4_2 case2; // G_Preflop
        Union4_3 case3; // G_River
        Union4_4 case4; // G_Round
        Union4_5 case5; // G_Round'
        Union4_6 case6; // G_Showdown
        Union4_7 case7; // G_Turn
    };
    unsigned char tag{255};
    __device__ Union4() {}
    __device__ Union4(Union4_0 t) : tag(0), case0(t) {} // G_Flop
    __device__ Union4(Union4_1 t) : tag(1), case1(t) {} // G_Fold
    __device__ Union4(Union4_2 t) : tag(2), case2(t) {} // G_Preflop
    __device__ Union4(Union4_3 t) : tag(3), case3(t) {} // G_River
    __device__ Union4(Union4_4 t) : tag(4), case4(t) {} // G_Round
    __device__ Union4(Union4_5 t) : tag(5), case5(t) {} // G_Round'
    __device__ Union4(Union4_6 t) : tag(6), case6(t) {} // G_Showdown
    __device__ Union4(Union4_7 t) : tag(7), case7(t) {} // G_Turn
    __device__ Union4(Union4 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(x.case0); break; // G_Flop
            case 1: new (&this->case1) Union4_1(x.case1); break; // G_Fold
            case 2: new (&this->case2) Union4_2(x.case2); break; // G_Preflop
            case 3: new (&this->case3) Union4_3(x.case3); break; // G_River
            case 4: new (&this->case4) Union4_4(x.case4); break; // G_Round
            case 5: new (&this->case5) Union4_5(x.case5); break; // G_Round'
            case 6: new (&this->case6) Union4_6(x.case6); break; // G_Showdown
            case 7: new (&this->case7) Union4_7(x.case7); break; // G_Turn
        }
    }
    __device__ Union4(Union4 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union4_0(std::move(x.case0)); break; // G_Flop
            case 1: new (&this->case1) Union4_1(std::move(x.case1)); break; // G_Fold
            case 2: new (&this->case2) Union4_2(std::move(x.case2)); break; // G_Preflop
            case 3: new (&this->case3) Union4_3(std::move(x.case3)); break; // G_River
            case 4: new (&this->case4) Union4_4(std::move(x.case4)); break; // G_Round
            case 5: new (&this->case5) Union4_5(std::move(x.case5)); break; // G_Round'
            case 6: new (&this->case6) Union4_6(std::move(x.case6)); break; // G_Showdown
            case 7: new (&this->case7) Union4_7(std::move(x.case7)); break; // G_Turn
        }
    }
    __device__ Union4 & operator=(Union4 & x) {
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
            this->~Union4();
            new (this) Union4{x};
        }
        return *this;
    }
    __device__ Union4 & operator=(Union4 && x) {
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
            this->~Union4();
            new (this) Union4{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union4() {
        switch(this->tag){
            case 0: this->case0.~Union4_0(); break; // G_Flop
            case 1: this->case1.~Union4_1(); break; // G_Fold
            case 2: this->case2.~Union4_2(); break; // G_Preflop
            case 3: this->case3.~Union4_3(); break; // G_River
            case 4: this->case4.~Union4_4(); break; // G_Round
            case 5: this->case5.~Union4_5(); break; // G_Round'
            case 6: this->case6.~Union4_6(); break; // G_Showdown
            case 7: this->case7.~Union4_7(); break; // G_Turn
        }
        this->tag = 255;
    }
};
struct Union3_0 { // None
};
struct Union3_1 { // Some
    Union4 v0;
    __device__ Union3_1(Union4 t0) : v0(t0) {}
    __device__ Union3_1() = delete;
};
struct Union3 {
    union {
        Union3_0 case0; // None
        Union3_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union3() {}
    __device__ Union3(Union3_0 t) : tag(0), case0(t) {} // None
    __device__ Union3(Union3_1 t) : tag(1), case1(t) {} // Some
    __device__ Union3(Union3 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(x.case0); break; // None
            case 1: new (&this->case1) Union3_1(x.case1); break; // Some
        }
    }
    __device__ Union3(Union3 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union3_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union3_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union3 & operator=(Union3 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
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
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union3();
            new (this) Union3{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union3() {
        switch(this->tag){
            case 0: this->case0.~Union3_0(); break; // None
            case 1: this->case1.~Union3_1(); break; // Some
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
struct Union6_0 { // CommunityCardsAre
    static_array_list<unsigned char,5> v0;
    __device__ Union6_0(static_array_list<unsigned char,5> t0) : v0(t0) {}
    __device__ Union6_0() = delete;
};
struct Union6_1 { // Fold
    int v0;
    int v1;
    __device__ Union6_1(int t0, int t1) : v0(t0), v1(t1) {}
    __device__ Union6_1() = delete;
};
struct Union6_2 { // PlayerAction
    Union1 v1;
    int v0;
    __device__ Union6_2(int t0, Union1 t1) : v0(t0), v1(t1) {}
    __device__ Union6_2() = delete;
};
struct Union6_3 { // PlayerGotCards
    static_array<unsigned char,2> v1;
    int v0;
    __device__ Union6_3(int t0, static_array<unsigned char,2> t1) : v0(t0), v1(t1) {}
    __device__ Union6_3() = delete;
};
struct Union6_4 { // Showdown
    static_array<Tuple0,2> v1;
    int v0;
    int v2;
    __device__ Union6_4(int t0, static_array<Tuple0,2> t1, int t2) : v0(t0), v1(t1), v2(t2) {}
    __device__ Union6_4() = delete;
};
struct Union6 {
    union {
        Union6_0 case0; // CommunityCardsAre
        Union6_1 case1; // Fold
        Union6_2 case2; // PlayerAction
        Union6_3 case3; // PlayerGotCards
        Union6_4 case4; // Showdown
    };
    unsigned char tag{255};
    __device__ Union6() {}
    __device__ Union6(Union6_0 t) : tag(0), case0(t) {} // CommunityCardsAre
    __device__ Union6(Union6_1 t) : tag(1), case1(t) {} // Fold
    __device__ Union6(Union6_2 t) : tag(2), case2(t) {} // PlayerAction
    __device__ Union6(Union6_3 t) : tag(3), case3(t) {} // PlayerGotCards
    __device__ Union6(Union6_4 t) : tag(4), case4(t) {} // Showdown
    __device__ Union6(Union6 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(x.case0); break; // CommunityCardsAre
            case 1: new (&this->case1) Union6_1(x.case1); break; // Fold
            case 2: new (&this->case2) Union6_2(x.case2); break; // PlayerAction
            case 3: new (&this->case3) Union6_3(x.case3); break; // PlayerGotCards
            case 4: new (&this->case4) Union6_4(x.case4); break; // Showdown
        }
    }
    __device__ Union6(Union6 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union6_0(std::move(x.case0)); break; // CommunityCardsAre
            case 1: new (&this->case1) Union6_1(std::move(x.case1)); break; // Fold
            case 2: new (&this->case2) Union6_2(std::move(x.case2)); break; // PlayerAction
            case 3: new (&this->case3) Union6_3(std::move(x.case3)); break; // PlayerGotCards
            case 4: new (&this->case4) Union6_4(std::move(x.case4)); break; // Showdown
        }
    }
    __device__ Union6 & operator=(Union6 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // CommunityCardsAre
                case 1: this->case1 = x.case1; break; // Fold
                case 2: this->case2 = x.case2; break; // PlayerAction
                case 3: this->case3 = x.case3; break; // PlayerGotCards
                case 4: this->case4 = x.case4; break; // Showdown
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
                case 0: this->case0 = std::move(x.case0); break; // CommunityCardsAre
                case 1: this->case1 = std::move(x.case1); break; // Fold
                case 2: this->case2 = std::move(x.case2); break; // PlayerAction
                case 3: this->case3 = std::move(x.case3); break; // PlayerGotCards
                case 4: this->case4 = std::move(x.case4); break; // Showdown
            }
        } else {
            this->~Union6();
            new (this) Union6{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union6() {
        switch(this->tag){
            case 0: this->case0.~Union6_0(); break; // CommunityCardsAre
            case 1: this->case1.~Union6_1(); break; // Fold
            case 2: this->case2.~Union6_2(); break; // PlayerAction
            case 3: this->case3.~Union6_3(); break; // PlayerGotCards
            case 4: this->case4.~Union6_4(); break; // Showdown
        }
        this->tag = 255;
    }
};
struct Union7_0 { // GameNotStarted
};
struct Union7_1 { // GameOver
    static_array<static_array<unsigned char,2>,2> v1;
    static_array<int,2> v2;
    static_array<int,2> v4;
    Union5 v5;
    int v0;
    int v3;
    __device__ Union7_1(int t0, static_array<static_array<unsigned char,2>,2> t1, static_array<int,2> t2, int t3, static_array<int,2> t4, Union5 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union7_1() = delete;
};
struct Union7_2 { // WaitingForActionFromPlayerId
    static_array<static_array<unsigned char,2>,2> v1;
    static_array<int,2> v2;
    static_array<int,2> v4;
    Union5 v5;
    int v0;
    int v3;
    __device__ Union7_2(int t0, static_array<static_array<unsigned char,2>,2> t1, static_array<int,2> t2, int t3, static_array<int,2> t4, Union5 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union7_2() = delete;
};
struct Union7 {
    union {
        Union7_0 case0; // GameNotStarted
        Union7_1 case1; // GameOver
        Union7_2 case2; // WaitingForActionFromPlayerId
    };
    unsigned char tag{255};
    __device__ Union7() {}
    __device__ Union7(Union7_0 t) : tag(0), case0(t) {} // GameNotStarted
    __device__ Union7(Union7_1 t) : tag(1), case1(t) {} // GameOver
    __device__ Union7(Union7_2 t) : tag(2), case2(t) {} // WaitingForActionFromPlayerId
    __device__ Union7(Union7 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(x.case0); break; // GameNotStarted
            case 1: new (&this->case1) Union7_1(x.case1); break; // GameOver
            case 2: new (&this->case2) Union7_2(x.case2); break; // WaitingForActionFromPlayerId
        }
    }
    __device__ Union7(Union7 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union7_0(std::move(x.case0)); break; // GameNotStarted
            case 1: new (&this->case1) Union7_1(std::move(x.case1)); break; // GameOver
            case 2: new (&this->case2) Union7_2(std::move(x.case2)); break; // WaitingForActionFromPlayerId
        }
    }
    __device__ Union7 & operator=(Union7 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // GameNotStarted
                case 1: this->case1 = x.case1; break; // GameOver
                case 2: this->case2 = x.case2; break; // WaitingForActionFromPlayerId
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
                case 0: this->case0 = std::move(x.case0); break; // GameNotStarted
                case 1: this->case1 = std::move(x.case1); break; // GameOver
                case 2: this->case2 = std::move(x.case2); break; // WaitingForActionFromPlayerId
            }
        } else {
            this->~Union7();
            new (this) Union7{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union7() {
        switch(this->tag){
            case 0: this->case0.~Union7_0(); break; // GameNotStarted
            case 1: this->case1.~Union7_1(); break; // GameOver
            case 2: this->case2.~Union7_2(); break; // WaitingForActionFromPlayerId
        }
        this->tag = 255;
    }
};
struct Tuple1 {
    unsigned long long v0;
    Union3 v1;
    dynamic_array_list<Union6,128> v2;
    static_array<Union2,2> v3;
    Union7 v4;
    __device__ Tuple1() = default;
    __device__ Tuple1(unsigned long long t0, Union3 t1, dynamic_array_list<Union6,128> t2, static_array<Union2,2> t3, Union7 t4) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4) {}
};
struct Tuple2 {
    static_array<static_array<unsigned char,2>,2> v1;
    static_array<int,2> v2;
    static_array<int,2> v4;
    Union5 v5;
    int v0;
    int v3;
    __device__ Tuple2() = default;
    __device__ Tuple2(int t0, static_array<static_array<unsigned char,2>,2> t1, static_array<int,2> t2, int t3, static_array<int,2> t4, Union5 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
};
struct Tuple3 {
    static_array<static_array<unsigned char,2>,2> v1;
    static_array<int,2> v2;
    static_array<int,2> v4;
    Union5 v5;
    Union1 v6;
    int v0;
    int v3;
    __device__ Tuple3() = default;
    __device__ Tuple3(int t0, static_array<static_array<unsigned char,2>,2> t1, static_array<int,2> t2, int t3, static_array<int,2> t4, Union5 t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
};
struct Tuple4 {
    int v0;
    int v1;
    __device__ Tuple4() = default;
    __device__ Tuple4(int t0, int t1) : v0(t0), v1(t1) {}
};
struct Tuple5 {
    Union1 v1;
    int v0;
    __device__ Tuple5() = default;
    __device__ Tuple5(int t0, Union1 t1) : v0(t0), v1(t1) {}
};
struct Tuple6 {
    static_array<unsigned char,2> v1;
    int v0;
    __device__ Tuple6() = default;
    __device__ Tuple6(int t0, static_array<unsigned char,2> t1) : v0(t0), v1(t1) {}
};
struct Tuple7 {
    static_array<Tuple0,2> v1;
    int v0;
    int v2;
    __device__ Tuple7() = default;
    __device__ Tuple7(int t0, static_array<Tuple0,2> t1, int t2) : v0(t0), v1(t1), v2(t2) {}
};
struct StackMut0 {
    unsigned long long v0;
    Union3 v1;
    dynamic_array_list<Union6,128> v2;
    static_array<Union2,2> v3;
    curandStatePhilox4_32_10_t v4;
    Union7 v5;
    __device__ StackMut0() = default;
    __device__ StackMut0(unsigned long long t0, Union3 t1, dynamic_array_list<Union6,128> t2, static_array<Union2,2> t3, curandStatePhilox4_32_10_t t4, Union7 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
};
struct Tuple8 {
    static_array<unsigned char,3> v0;
    unsigned long long v1;
    __device__ Tuple8() = default;
    __device__ Tuple8(static_array<unsigned char,3> t0, unsigned long long t1) : v0(t0), v1(t1) {}
};
struct Tuple9 {
    unsigned long long v1;
    int v0;
    __device__ Tuple9() = default;
    __device__ Tuple9(int t0, unsigned long long t1) : v0(t0), v1(t1) {}
};
struct Tuple10 {
    unsigned long long v1;
    unsigned char v0;
    __device__ Tuple10() = default;
    __device__ Tuple10(unsigned char t0, unsigned long long t1) : v0(t0), v1(t1) {}
};
struct Tuple11 {
    static_array<unsigned char,2> v0;
    unsigned long long v1;
    __device__ Tuple11() = default;
    __device__ Tuple11(static_array<unsigned char,2> t0, unsigned long long t1) : v0(t0), v1(t1) {}
};
struct Tuple12 {
    static_array<unsigned char,1> v0;
    unsigned long long v1;
    __device__ Tuple12() = default;
    __device__ Tuple12(static_array<unsigned char,1> t0, unsigned long long t1) : v0(t0), v1(t1) {}
};
struct Union8_0 { // None
};
struct Union8_1 { // Some
    Union1 v0;
    __device__ Union8_1(Union1 t0) : v0(t0) {}
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
struct Tuple13 {
    int v0;
    unsigned int v1;
    __device__ Tuple13() = default;
    __device__ Tuple13(int t0, unsigned int t1) : v0(t0), v1(t1) {}
};
struct Union9_0 { // None
};
struct Union9_1 { // Some
    int v0;
    __device__ Union9_1(int t0) : v0(t0) {}
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
struct Closure0 {
    __device__ unsigned int operator()(unsigned int tup0, unsigned int tup1){
        unsigned int v0 = tup0; unsigned int v1 = tup1;
        unsigned int v2;
        v2 = v0 | v1;
        return v2;
    }
};
struct Tuple14 {
    float v0;
    int v1;
    __device__ Tuple14() = default;
    __device__ Tuple14(float t0, int t1) : v0(t0), v1(t1) {}
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
struct Tuple15 {
    int v0;
    float v1;
    __device__ Tuple15() = default;
    __device__ Tuple15(int t0, float t1) : v0(t0), v1(t1) {}
};
struct Closure3 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Tuple16 {
    float v0;
    bool v1;
    __device__ Tuple16() = default;
    __device__ Tuple16(float t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure4 {
    __device__ Tuple16 operator()(Tuple16 tup0, Tuple16 tup1){
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
                return Tuple16{v5, true};
            } else {
                return Tuple16{v0, v1};
            }
        } else {
            if (v3){
                return Tuple16{v2, v3};
            } else {
                return Tuple16{v0, v1};
            }
        }
    }
};
struct Closure5 {
    __device__ Tuple14 operator()(Tuple14 tup0, Tuple14 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v1 < v3;
        if (v4){
            return Tuple14{v0, v1};
        } else {
            return Tuple14{v2, v3};
        }
    }
};
struct Tuple17 {
    int v0;
    bool v1;
    __device__ Tuple17() = default;
    __device__ Tuple17(int t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure6 {
    __device__ Tuple17 operator()(Tuple17 tup0, Tuple17 tup1){
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
                return Tuple17{v5, true};
            } else {
                return Tuple17{v0, v1};
            }
        } else {
            if (v3){
                return Tuple17{v2, v3};
            } else {
                return Tuple17{v0, v1};
            }
        }
    }
};
struct Closure7 {
    int v0;
    __device__ Tuple14 operator()(Tuple14 tup0, Tuple14 tup1){
        int & v0 = this->v0;
        float v1 = tup0.v0; int v2 = tup0.v1; float v3 = tup1.v0; int v4 = tup1.v1;
        bool v5;
        v5 = v2 == v0;
        if (v5){
            return Tuple14{v1, v2};
        } else {
            bool v6;
            v6 = v4 == v0;
            if (v6){
                return Tuple14{v3, v4};
            } else {
                return Tuple14{v1, v2};
            }
        }
    }
    __device__ Closure7(int _v0) : v0(_v0) { }
};
struct Union10_0 { // AA_Call
};
struct Union10_1 { // AA_Fold
};
struct Union10_2 { // AA_Raise
    int v0;
    int v1;
    __device__ Union10_2(int t0, int t1) : v0(t0), v1(t1) {}
    __device__ Union10_2() = delete;
};
struct Union10 {
    union {
        Union10_0 case0; // AA_Call
        Union10_1 case1; // AA_Fold
        Union10_2 case2; // AA_Raise
    };
    unsigned char tag{255};
    __device__ Union10() {}
    __device__ Union10(Union10_0 t) : tag(0), case0(t) {} // AA_Call
    __device__ Union10(Union10_1 t) : tag(1), case1(t) {} // AA_Fold
    __device__ Union10(Union10_2 t) : tag(2), case2(t) {} // AA_Raise
    __device__ Union10(Union10 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union10_0(x.case0); break; // AA_Call
            case 1: new (&this->case1) Union10_1(x.case1); break; // AA_Fold
            case 2: new (&this->case2) Union10_2(x.case2); break; // AA_Raise
        }
    }
    __device__ Union10(Union10 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union10_0(std::move(x.case0)); break; // AA_Call
            case 1: new (&this->case1) Union10_1(std::move(x.case1)); break; // AA_Fold
            case 2: new (&this->case2) Union10_2(std::move(x.case2)); break; // AA_Raise
        }
    }
    __device__ Union10 & operator=(Union10 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // AA_Call
                case 1: this->case1 = x.case1; break; // AA_Fold
                case 2: this->case2 = x.case2; break; // AA_Raise
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
                case 0: this->case0 = std::move(x.case0); break; // AA_Call
                case 1: this->case1 = std::move(x.case1); break; // AA_Fold
                case 2: this->case2 = std::move(x.case2); break; // AA_Raise
            }
        } else {
            this->~Union10();
            new (this) Union10{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union10() {
        switch(this->tag){
            case 0: this->case0.~Union10_0(); break; // AA_Call
            case 1: this->case1.~Union10_1(); break; // AA_Fold
            case 2: this->case2.~Union10_2(); break; // AA_Raise
        }
        this->tag = 255;
    }
};
struct Tuple18 {
    Union1 v0;
    float v1;
    __device__ Tuple18() = default;
    __device__ Tuple18(Union1 t0, float t1) : v0(t0), v1(t1) {}
};
struct Tuple19 {
    int v1;
    bool v0;
    __device__ Tuple19() = default;
    __device__ Tuple19(bool t0, int t1) : v0(t0), v1(t1) {}
};
struct Tuple20 {
    int v0;
    int v1;
    int v2;
    __device__ Tuple20() = default;
    __device__ Tuple20(int t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Union11_0 { // Eq
};
struct Union11_1 { // Gt
};
struct Union11_2 { // Lt
};
struct Union11 {
    union {
        Union11_0 case0; // Eq
        Union11_1 case1; // Gt
        Union11_2 case2; // Lt
    };
    unsigned char tag{255};
    __device__ Union11() {}
    __device__ Union11(Union11_0 t) : tag(0), case0(t) {} // Eq
    __device__ Union11(Union11_1 t) : tag(1), case1(t) {} // Gt
    __device__ Union11(Union11_2 t) : tag(2), case2(t) {} // Lt
    __device__ Union11(Union11 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union11_0(x.case0); break; // Eq
            case 1: new (&this->case1) Union11_1(x.case1); break; // Gt
            case 2: new (&this->case2) Union11_2(x.case2); break; // Lt
        }
    }
    __device__ Union11(Union11 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union11_0(std::move(x.case0)); break; // Eq
            case 1: new (&this->case1) Union11_1(std::move(x.case1)); break; // Gt
            case 2: new (&this->case2) Union11_2(std::move(x.case2)); break; // Lt
        }
    }
    __device__ Union11 & operator=(Union11 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // Eq
                case 1: this->case1 = x.case1; break; // Gt
                case 2: this->case2 = x.case2; break; // Lt
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
                case 0: this->case0 = std::move(x.case0); break; // Eq
                case 1: this->case1 = std::move(x.case1); break; // Gt
                case 2: this->case2 = std::move(x.case2); break; // Lt
            }
        } else {
            this->~Union11();
            new (this) Union11{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union11() {
        switch(this->tag){
            case 0: this->case0.~Union11_0(); break; // Eq
            case 1: this->case1.~Union11_1(); break; // Gt
            case 2: this->case2.~Union11_2(); break; // Lt
        }
        this->tag = 255;
    }
};
struct Tuple21 {
    int v0;
    int v1;
    unsigned char v2;
    __device__ Tuple21() = default;
    __device__ Tuple21(int t0, int t1, unsigned char t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Union12_0 { // None
};
struct Union12_1 { // Some
    static_array<unsigned char,5> v0;
    __device__ Union12_1(static_array<unsigned char,5> t0) : v0(t0) {}
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
struct Tuple22 {
    Union11 v1;
    int v0;
    __device__ Tuple22() = default;
    __device__ Tuple22(int t0, Union11 t1) : v0(t0), v1(t1) {}
};
struct Tuple23 {
    int v0;
    int v1;
    int v2;
    unsigned char v3;
    __device__ Tuple23() = default;
    __device__ Tuple23(int t0, int t1, int t2, unsigned char t3) : v0(t0), v1(t1), v2(t2), v3(t3) {}
};
struct Union13_0 { // None
};
struct Union13_1 { // Some
    static_array<unsigned char,4> v0;
    static_array<unsigned char,3> v1;
    __device__ Union13_1(static_array<unsigned char,4> t0, static_array<unsigned char,3> t1) : v0(t0), v1(t1) {}
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
    static_array<unsigned char,3> v0;
    static_array<unsigned char,4> v1;
    __device__ Union14_1(static_array<unsigned char,3> t0, static_array<unsigned char,4> t1) : v0(t0), v1(t1) {}
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
    static_array<unsigned char,2> v1;
    __device__ Union15_1(static_array<unsigned char,2> t0, static_array<unsigned char,2> t1) : v0(t0), v1(t1) {}
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
    static_array<unsigned char,5> v1;
    __device__ Union16_1(static_array<unsigned char,2> t0, static_array<unsigned char,5> t1) : v0(t0), v1(t1) {}
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
struct Union17_0 { // None
};
struct Union17_1 { // Some
    static_array<unsigned char,2> v0;
    static_array<unsigned char,3> v1;
    __device__ Union17_1(static_array<unsigned char,2> t0, static_array<unsigned char,3> t1) : v0(t0), v1(t1) {}
    __device__ Union17_1() = delete;
};
struct Union17 {
    union {
        Union17_0 case0; // None
        Union17_1 case1; // Some
    };
    unsigned char tag{255};
    __device__ Union17() {}
    __device__ Union17(Union17_0 t) : tag(0), case0(t) {} // None
    __device__ Union17(Union17_1 t) : tag(1), case1(t) {} // Some
    __device__ Union17(Union17 & x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union17_0(x.case0); break; // None
            case 1: new (&this->case1) Union17_1(x.case1); break; // Some
        }
    }
    __device__ Union17(Union17 && x) : tag(x.tag) {
        switch(x.tag){
            case 0: new (&this->case0) Union17_0(std::move(x.case0)); break; // None
            case 1: new (&this->case1) Union17_1(std::move(x.case1)); break; // Some
        }
    }
    __device__ Union17 & operator=(Union17 & x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = x.case0; break; // None
                case 1: this->case1 = x.case1; break; // Some
            }
        } else {
            this->~Union17();
            new (this) Union17{x};
        }
        return *this;
    }
    __device__ Union17 & operator=(Union17 && x) {
        if (this->tag == x.tag) {
            switch(x.tag){
                case 0: this->case0 = std::move(x.case0); break; // None
                case 1: this->case1 = std::move(x.case1); break; // Some
            }
        } else {
            this->~Union17();
            new (this) Union17{std::move(x)};
        }
        return *this;
    }
    __device__ ~Union17() {
        switch(this->tag){
            case 0: this->case0.~Union17_0(); break; // None
            case 1: this->case1.~Union17_1(); break; // Some
        }
        this->tag = 255;
    }
};
__device__ int f_1(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0];
    return v3;
}
__device__ void f_3(unsigned char * v0){
    return ;
}
__device__ Union1 f_2(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    switch (v1) {
        case 0: {
            f_3(v2);
            return Union1{Union1_0{}};
            break;
        }
        case 1: {
            f_3(v2);
            return Union1{Union1_1{}};
            break;
        }
        case 2: {
            f_3(v2);
            return Union1{Union1_2{}};
            break;
        }
        case 3: {
            int v8;
            v8 = f_1(v2);
            return Union1{Union1_3{v8}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
}
__device__ inline bool while_method_0(int v0){
    bool v1;
    v1 = v0 < 2;
    return v1;
}
__device__ Union2 f_5(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+4ull);
    switch (v1) {
        case 0: {
            f_3(v2);
            return Union2{Union2_0{}};
            break;
        }
        case 1: {
            f_3(v2);
            return Union2{Union2_1{}};
            break;
        }
        case 2: {
            f_3(v2);
            return Union2{Union2_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
}
__device__ static_array<Union2,2> f_4(unsigned char * v0){
    static_array<Union2,2> v1;
    int v3;
    v3 = 0;
    while (while_method_0(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned long long v6;
        v6 = v5 * 4ull;
        unsigned char * v7;
        v7 = (unsigned char *)(v0+v6);
        Union2 v9;
        v9 = f_5(v7);
        v1[v3] = v9;
        v3 += 1 ;
    }
    return v1;
}
__device__ Union0 f_0(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+8ull);
    switch (v1) {
        case 0: {
            Union1 v5;
            v5 = f_2(v2);
            return Union0{Union0_0{v5}};
            break;
        }
        case 1: {
            static_array<Union2,2> v7;
            v7 = f_4(v2);
            return Union0{Union0_1{v7}};
            break;
        }
        case 2: {
            f_3(v2);
            return Union0{Union0_2{}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
}
__device__ unsigned long long f_7(unsigned char * v0){
    unsigned long long * v1;
    v1 = (unsigned long long *)(v0+0ull);
    unsigned long long v3;
    v3 = v1[0];
    return v3;
}
__device__ int f_8(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+8ull);
    int v3;
    v3 = v1[0];
    return v3;
}
__device__ unsigned char f_13(unsigned char * v0){
    unsigned char * v1;
    v1 = (unsigned char *)(v0+0ull);
    unsigned char v3;
    v3 = v1[0];
    return v3;
}
__device__ unsigned char f_12(unsigned char * v0){
    unsigned char v1;
    v1 = f_13(v0);
    return v1;
}
__device__ static_array<unsigned char,2> f_11(unsigned char * v0){
    static_array<unsigned char,2> v1;
    int v3;
    v3 = 0;
    while (while_method_0(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        unsigned char v8;
        v8 = f_12(v6);
        v1[v3] = v8;
        v3 += 1 ;
    }
    return v1;
}
__device__ int f_14(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+28ull);
    int v3;
    v3 = v1[0];
    return v3;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 3;
    return v1;
}
__device__ static_array<unsigned char,3> f_15(unsigned char * v0){
    static_array<unsigned char,3> v1;
    int v3;
    v3 = 0;
    while (while_method_1(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        unsigned char v8;
        v8 = f_12(v6);
        v1[v3] = v8;
        v3 += 1 ;
    }
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 5;
    return v1;
}
__device__ static_array<unsigned char,5> f_16(unsigned char * v0){
    static_array<unsigned char,5> v1;
    int v3;
    v3 = 0;
    while (while_method_2(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        unsigned char v8;
        v8 = f_12(v6);
        v1[v3] = v8;
        v3 += 1 ;
    }
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 4;
    return v1;
}
__device__ static_array<unsigned char,4> f_17(unsigned char * v0){
    static_array<unsigned char,4> v1;
    int v3;
    v3 = 0;
    while (while_method_3(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        unsigned char v8;
        v8 = f_12(v6);
        v1[v3] = v8;
        v3 += 1 ;
    }
    return v1;
}
__device__ Tuple2 f_10(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0];
    static_array<static_array<unsigned char,2>,2> v4;
    int v6;
    v6 = 0;
    while (while_method_0(v6)){
        unsigned long long v8;
        v8 = (unsigned long long)v6;
        unsigned long long v9;
        v9 = v8 * 2ull;
        unsigned long long v10;
        v10 = 4ull + v9;
        unsigned char * v11;
        v11 = (unsigned char *)(v0+v10);
        static_array<unsigned char,2> v13;
        v13 = f_11(v11);
        v4[v6] = v13;
        v6 += 1 ;
    }
    static_array<int,2> v14;
    int v16;
    v16 = 0;
    while (while_method_0(v16)){
        unsigned long long v18;
        v18 = (unsigned long long)v16;
        unsigned long long v19;
        v19 = v18 * 4ull;
        unsigned long long v20;
        v20 = 8ull + v19;
        unsigned char * v21;
        v21 = (unsigned char *)(v0+v20);
        int v23;
        v23 = f_1(v21);
        v14[v16] = v23;
        v16 += 1 ;
    }
    int * v24;
    v24 = (int *)(v0+16ull);
    int v26;
    v26 = v24[0];
    static_array<int,2> v27;
    int v29;
    v29 = 0;
    while (while_method_0(v29)){
        unsigned long long v31;
        v31 = (unsigned long long)v29;
        unsigned long long v32;
        v32 = v31 * 4ull;
        unsigned long long v33;
        v33 = 20ull + v32;
        unsigned char * v34;
        v34 = (unsigned char *)(v0+v33);
        int v36;
        v36 = f_1(v34);
        v27[v29] = v36;
        v29 += 1 ;
    }
    int v37;
    v37 = f_14(v0);
    unsigned char * v38;
    v38 = (unsigned char *)(v0+32ull);
    Union5 v48;
    switch (v37) {
        case 0: {
            static_array<unsigned char,3> v41;
            v41 = f_15(v38);
            v48 = Union5{Union5_0{v41}};
            break;
        }
        case 1: {
            f_3(v38);
            v48 = Union5{Union5_1{}};
            break;
        }
        case 2: {
            static_array<unsigned char,5> v44;
            v44 = f_16(v38);
            v48 = Union5{Union5_2{v44}};
            break;
        }
        case 3: {
            static_array<unsigned char,4> v46;
            v46 = f_17(v38);
            v48 = Union5{Union5_3{v46}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
    return Tuple2{v3, v4, v14, v26, v27, v48};
}
__device__ int f_19(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+40ull);
    int v3;
    v3 = v1[0];
    return v3;
}
__device__ Tuple3 f_18(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0];
    static_array<static_array<unsigned char,2>,2> v4;
    int v6;
    v6 = 0;
    while (while_method_0(v6)){
        unsigned long long v8;
        v8 = (unsigned long long)v6;
        unsigned long long v9;
        v9 = v8 * 2ull;
        unsigned long long v10;
        v10 = 4ull + v9;
        unsigned char * v11;
        v11 = (unsigned char *)(v0+v10);
        static_array<unsigned char,2> v13;
        v13 = f_11(v11);
        v4[v6] = v13;
        v6 += 1 ;
    }
    static_array<int,2> v14;
    int v16;
    v16 = 0;
    while (while_method_0(v16)){
        unsigned long long v18;
        v18 = (unsigned long long)v16;
        unsigned long long v19;
        v19 = v18 * 4ull;
        unsigned long long v20;
        v20 = 8ull + v19;
        unsigned char * v21;
        v21 = (unsigned char *)(v0+v20);
        int v23;
        v23 = f_1(v21);
        v14[v16] = v23;
        v16 += 1 ;
    }
    int * v24;
    v24 = (int *)(v0+16ull);
    int v26;
    v26 = v24[0];
    static_array<int,2> v27;
    int v29;
    v29 = 0;
    while (while_method_0(v29)){
        unsigned long long v31;
        v31 = (unsigned long long)v29;
        unsigned long long v32;
        v32 = v31 * 4ull;
        unsigned long long v33;
        v33 = 20ull + v32;
        unsigned char * v34;
        v34 = (unsigned char *)(v0+v33);
        int v36;
        v36 = f_1(v34);
        v27[v29] = v36;
        v29 += 1 ;
    }
    int v37;
    v37 = f_14(v0);
    unsigned char * v38;
    v38 = (unsigned char *)(v0+32ull);
    Union5 v48;
    switch (v37) {
        case 0: {
            static_array<unsigned char,3> v41;
            v41 = f_15(v38);
            v48 = Union5{Union5_0{v41}};
            break;
        }
        case 1: {
            f_3(v38);
            v48 = Union5{Union5_1{}};
            break;
        }
        case 2: {
            static_array<unsigned char,5> v44;
            v44 = f_16(v38);
            v48 = Union5{Union5_2{v44}};
            break;
        }
        case 3: {
            static_array<unsigned char,4> v46;
            v46 = f_17(v38);
            v48 = Union5{Union5_3{v46}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
    int v49;
    v49 = f_19(v0);
    unsigned char * v50;
    v50 = (unsigned char *)(v0+44ull);
    Union1 v58;
    switch (v49) {
        case 0: {
            f_3(v50);
            v58 = Union1{Union1_0{}};
            break;
        }
        case 1: {
            f_3(v50);
            v58 = Union1{Union1_1{}};
            break;
        }
        case 2: {
            f_3(v50);
            v58 = Union1{Union1_2{}};
            break;
        }
        case 3: {
            int v56;
            v56 = f_1(v50);
            v58 = Union1{Union1_3{v56}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
    return Tuple3{v3, v4, v14, v26, v27, v48, v58};
}
__device__ Union4 f_9(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+16ull);
    switch (v1) {
        case 0: {
            int v5; static_array<static_array<unsigned char,2>,2> v6; static_array<int,2> v7; int v8; static_array<int,2> v9; Union5 v10;
            Tuple2 tmp0 = f_10(v2);
            v5 = tmp0.v0; v6 = tmp0.v1; v7 = tmp0.v2; v8 = tmp0.v3; v9 = tmp0.v4; v10 = tmp0.v5;
            return Union4{Union4_0{v5, v6, v7, v8, v9, v10}};
            break;
        }
        case 1: {
            int v12; static_array<static_array<unsigned char,2>,2> v13; static_array<int,2> v14; int v15; static_array<int,2> v16; Union5 v17;
            Tuple2 tmp1 = f_10(v2);
            v12 = tmp1.v0; v13 = tmp1.v1; v14 = tmp1.v2; v15 = tmp1.v3; v16 = tmp1.v4; v17 = tmp1.v5;
            return Union4{Union4_1{v12, v13, v14, v15, v16, v17}};
            break;
        }
        case 2: {
            f_3(v2);
            return Union4{Union4_2{}};
            break;
        }
        case 3: {
            int v20; static_array<static_array<unsigned char,2>,2> v21; static_array<int,2> v22; int v23; static_array<int,2> v24; Union5 v25;
            Tuple2 tmp2 = f_10(v2);
            v20 = tmp2.v0; v21 = tmp2.v1; v22 = tmp2.v2; v23 = tmp2.v3; v24 = tmp2.v4; v25 = tmp2.v5;
            return Union4{Union4_3{v20, v21, v22, v23, v24, v25}};
            break;
        }
        case 4: {
            int v27; static_array<static_array<unsigned char,2>,2> v28; static_array<int,2> v29; int v30; static_array<int,2> v31; Union5 v32;
            Tuple2 tmp3 = f_10(v2);
            v27 = tmp3.v0; v28 = tmp3.v1; v29 = tmp3.v2; v30 = tmp3.v3; v31 = tmp3.v4; v32 = tmp3.v5;
            return Union4{Union4_4{v27, v28, v29, v30, v31, v32}};
            break;
        }
        case 5: {
            int v34; static_array<static_array<unsigned char,2>,2> v35; static_array<int,2> v36; int v37; static_array<int,2> v38; Union5 v39; Union1 v40;
            Tuple3 tmp4 = f_18(v2);
            v34 = tmp4.v0; v35 = tmp4.v1; v36 = tmp4.v2; v37 = tmp4.v3; v38 = tmp4.v4; v39 = tmp4.v5; v40 = tmp4.v6;
            return Union4{Union4_5{v34, v35, v36, v37, v38, v39, v40}};
            break;
        }
        case 6: {
            int v42; static_array<static_array<unsigned char,2>,2> v43; static_array<int,2> v44; int v45; static_array<int,2> v46; Union5 v47;
            Tuple2 tmp5 = f_10(v2);
            v42 = tmp5.v0; v43 = tmp5.v1; v44 = tmp5.v2; v45 = tmp5.v3; v46 = tmp5.v4; v47 = tmp5.v5;
            return Union4{Union4_6{v42, v43, v44, v45, v46, v47}};
            break;
        }
        case 7: {
            int v49; static_array<static_array<unsigned char,2>,2> v50; static_array<int,2> v51; int v52; static_array<int,2> v53; Union5 v54;
            Tuple2 tmp6 = f_10(v2);
            v49 = tmp6.v0; v50 = tmp6.v1; v51 = tmp6.v2; v52 = tmp6.v3; v53 = tmp6.v4; v54 = tmp6.v5;
            return Union4{Union4_7{v49, v50, v51, v52, v53, v54}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
}
__device__ int f_20(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+80ull);
    int v3;
    v3 = v1[0];
    return v3;
}
__device__ inline bool while_method_4(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ static_array_list<unsigned char,5> f_22(unsigned char * v0){
    static_array_list<unsigned char,5> v1;
    v1 = static_array_list<unsigned char,5>{};
    int v3;
    v3 = f_1(v0);
    v1.unsafe_set_length(v3);
    int v4;
    v4 = v1.length;
    int v5;
    v5 = 0;
    while (while_method_4(v4, v5)){
        unsigned long long v7;
        v7 = (unsigned long long)v5;
        unsigned long long v8;
        v8 = 4ull + v7;
        unsigned char * v9;
        v9 = (unsigned char *)(v0+v8);
        unsigned char v11;
        v11 = f_12(v9);
        v1[v5] = v11;
        v5 += 1 ;
    }
    return v1;
}
__device__ Tuple4 f_23(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0];
    int * v4;
    v4 = (int *)(v0+4ull);
    int v6;
    v6 = v4[0];
    return Tuple4{v3, v6};
}
__device__ int f_25(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+4ull);
    int v3;
    v3 = v1[0];
    return v3;
}
__device__ Tuple5 f_24(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0];
    int v4;
    v4 = f_25(v0);
    unsigned char * v5;
    v5 = (unsigned char *)(v0+8ull);
    Union1 v13;
    switch (v4) {
        case 0: {
            f_3(v5);
            v13 = Union1{Union1_0{}};
            break;
        }
        case 1: {
            f_3(v5);
            v13 = Union1{Union1_1{}};
            break;
        }
        case 2: {
            f_3(v5);
            v13 = Union1{Union1_2{}};
            break;
        }
        case 3: {
            int v11;
            v11 = f_1(v5);
            v13 = Union1{Union1_3{v11}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
    return Tuple5{v3, v13};
}
__device__ Tuple6 f_26(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0];
    static_array<unsigned char,2> v4;
    int v6;
    v6 = 0;
    while (while_method_0(v6)){
        unsigned long long v8;
        v8 = (unsigned long long)v6;
        unsigned long long v9;
        v9 = 4ull + v8;
        unsigned char * v10;
        v10 = (unsigned char *)(v0+v9);
        unsigned char v12;
        v12 = f_12(v10);
        v4[v6] = v12;
        v6 += 1 ;
    }
    return Tuple6{v3, v4};
}
__device__ Tuple0 f_29(unsigned char * v0){
    static_array<unsigned char,5> v1;
    int v3;
    v3 = 0;
    while (while_method_2(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        unsigned char v8;
        v8 = f_12(v6);
        v1[v3] = v8;
        v3 += 1 ;
    }
    char * v9;
    v9 = (char *)(v0+5ull);
    char v11;
    v11 = v9[0];
    return Tuple0{v1, v11};
}
__device__ Tuple0 f_28(unsigned char * v0){
    static_array<unsigned char,5> v1; char v2;
    Tuple0 tmp10 = f_29(v0);
    v1 = tmp10.v0; v2 = tmp10.v1;
    return Tuple0{v1, v2};
}
__device__ Tuple7 f_27(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0];
    static_array<Tuple0,2> v4;
    int v6;
    v6 = 0;
    while (while_method_0(v6)){
        unsigned long long v8;
        v8 = (unsigned long long)v6;
        unsigned long long v9;
        v9 = v8 * 8ull;
        unsigned long long v10;
        v10 = 8ull + v9;
        unsigned char * v11;
        v11 = (unsigned char *)(v0+v10);
        static_array<unsigned char,5> v13; char v14;
        Tuple0 tmp11 = f_28(v11);
        v13 = tmp11.v0; v14 = tmp11.v1;
        v4[v6] = Tuple0{v13, v14};
        v6 += 1 ;
    }
    int * v15;
    v15 = (int *)(v0+24ull);
    int v17;
    v17 = v15[0];
    return Tuple7{v3, v4, v17};
}
__device__ Union6 f_21(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+16ull);
    switch (v1) {
        case 0: {
            static_array_list<unsigned char,5> v5;
            v5 = f_22(v2);
            return Union6{Union6_0{v5}};
            break;
        }
        case 1: {
            int v7; int v8;
            Tuple4 tmp7 = f_23(v2);
            v7 = tmp7.v0; v8 = tmp7.v1;
            return Union6{Union6_1{v7, v8}};
            break;
        }
        case 2: {
            int v10; Union1 v11;
            Tuple5 tmp8 = f_24(v2);
            v10 = tmp8.v0; v11 = tmp8.v1;
            return Union6{Union6_2{v10, v11}};
            break;
        }
        case 3: {
            int v13; static_array<unsigned char,2> v14;
            Tuple6 tmp9 = f_26(v2);
            v13 = tmp9.v0; v14 = tmp9.v1;
            return Union6{Union6_3{v13, v14}};
            break;
        }
        case 4: {
            int v16; static_array<Tuple0,2> v17; int v18;
            Tuple7 tmp12 = f_27(v2);
            v16 = tmp12.v0; v17 = tmp12.v1; v18 = tmp12.v2;
            return Union6{Union6_4{v16, v17, v18}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
}
__device__ int f_30(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+6248ull);
    int v3;
    v3 = v1[0];
    return v3;
}
__device__ Tuple1 f_6(unsigned char * v0){
    unsigned long long v1;
    v1 = f_7(v0);
    int v2;
    v2 = f_8(v0);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    Union3 v9;
    switch (v2) {
        case 0: {
            f_3(v3);
            v9 = Union3{Union3_0{}};
            break;
        }
        case 1: {
            Union4 v7;
            v7 = f_9(v3);
            v9 = Union3{Union3_1{v7}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
    dynamic_array_list<Union6,128> v10{0};
    int v12;
    v12 = f_20(v0);
    v10.unsafe_set_length(v12);
    int v13;
    v13 = v10.length_();
    int v14;
    v14 = 0;
    while (while_method_4(v13, v14)){
        unsigned long long v16;
        v16 = (unsigned long long)v14;
        unsigned long long v17;
        v17 = v16 * 48ull;
        unsigned long long v18;
        v18 = 96ull + v17;
        unsigned char * v19;
        v19 = (unsigned char *)(v0+v18);
        Union6 v21;
        v21 = f_21(v19);
        v10[v14] = v21;
        v14 += 1 ;
    }
    static_array<Union2,2> v22;
    int v24;
    v24 = 0;
    while (while_method_0(v24)){
        unsigned long long v26;
        v26 = (unsigned long long)v24;
        unsigned long long v27;
        v27 = v26 * 4ull;
        unsigned long long v28;
        v28 = 6240ull + v27;
        unsigned char * v29;
        v29 = (unsigned char *)(v0+v28);
        Union2 v31;
        v31 = f_5(v29);
        v22[v24] = v31;
        v24 += 1 ;
    }
    int v32;
    v32 = f_30(v0);
    unsigned char * v33;
    v33 = (unsigned char *)(v0+6256ull);
    Union7 v51;
    switch (v32) {
        case 0: {
            f_3(v33);
            v51 = Union7{Union7_0{}};
            break;
        }
        case 1: {
            int v37; static_array<static_array<unsigned char,2>,2> v38; static_array<int,2> v39; int v40; static_array<int,2> v41; Union5 v42;
            Tuple2 tmp13 = f_10(v33);
            v37 = tmp13.v0; v38 = tmp13.v1; v39 = tmp13.v2; v40 = tmp13.v3; v41 = tmp13.v4; v42 = tmp13.v5;
            v51 = Union7{Union7_1{v37, v38, v39, v40, v41, v42}};
            break;
        }
        case 2: {
            int v44; static_array<static_array<unsigned char,2>,2> v45; static_array<int,2> v46; int v47; static_array<int,2> v48; Union5 v49;
            Tuple2 tmp14 = f_10(v33);
            v44 = tmp14.v0; v45 = tmp14.v1; v46 = tmp14.v2; v47 = tmp14.v3; v48 = tmp14.v4; v49 = tmp14.v5;
            v51 = Union7{Union7_2{v44, v45, v46, v47, v48, v49}};
            break;
        }
        default: {
            printf("%s\n", "Invalid tag.");
            __trap();
        }
    }
    return Tuple1{v1, v9, v10, v22, v51};
}
__device__ inline bool while_method_5(Union3 v0){
    switch (v0.tag) {
        case 0: { // None
            return false;
            break;
        }
        case 1: { // Some
            Union4 v1 = v0.case1.v0;
            return true;
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ unsigned int loop_34(unsigned int v0, curandStatePhilox4_32_10_t & v1){
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
        return loop_34(v0, v1);
    }
}
__device__ Tuple10 draw_card_33(curandStatePhilox4_32_10_t & v0, unsigned long long v1){
    int v2;
    v2 = __popcll(v1);
    unsigned int v3;
    v3 = (unsigned int)v2;
    unsigned int v4;
    v4 = loop_34(v3, v0);
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
    return Tuple10{v23, v26};
}
__device__ Tuple8 draw_cards_32(curandStatePhilox4_32_10_t & v0, unsigned long long v1){
    static_array<unsigned char,3> v2;
    int v4; unsigned long long v5;
    Tuple9 tmp16 = Tuple9{0, v1};
    v4 = tmp16.v0; v5 = tmp16.v1;
    while (while_method_1(v4)){
        unsigned char v7; unsigned long long v8;
        Tuple10 tmp17 = draw_card_33(v0, v5);
        v7 = tmp17.v0; v8 = tmp17.v1;
        v2[v4] = v7;
        v5 = v8;
        v4 += 1 ;
    }
    return Tuple8{v2, v5};
}
__device__ static_array_list<unsigned char,5> get_community_cards_35(Union5 v0, static_array<unsigned char,3> v1){
    static_array_list<unsigned char,5> v2;
    v2 = static_array_list<unsigned char,5>{};
    switch (v0.tag) {
        case 0: { // Flop
            static_array<unsigned char,3> v4 = v0.case0.v0;
            int v5;
            v5 = 0;
            while (while_method_1(v5)){
                unsigned char v7;
                v7 = v4[v5];
                v2.push(v7);
                v5 += 1 ;
            }
            break;
        }
        case 1: { // Preflop
            break;
        }
        case 2: { // River
            static_array<unsigned char,5> v14 = v0.case2.v0;
            int v15;
            v15 = 0;
            while (while_method_2(v15)){
                unsigned char v17;
                v17 = v14[v15];
                v2.push(v17);
                v15 += 1 ;
            }
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4> v9 = v0.case3.v0;
            int v10;
            v10 = 0;
            while (while_method_3(v10)){
                unsigned char v12;
                v12 = v9[v10];
                v2.push(v12);
                v10 += 1 ;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v19;
    v19 = 0;
    while (while_method_1(v19)){
        unsigned char v21;
        v21 = v1[v19];
        v2.push(v21);
        v19 += 1 ;
    }
    return v2;
}
__device__ bool player_can_act_37(int v0, static_array<static_array<unsigned char,2>,2> v1, static_array<int,2> v2, int v3, static_array<int,2> v4, Union5 v5){
    int v6;
    v6 = v3 % 2;
    int v7;
    v7 = v4[v6];
    bool v9;
    v9 = v7 > 0;
    int v10;
    v10 = v2[v6];
    int v12;
    v12 = v2[0];
    int v14; int v15;
    Tuple4 tmp19 = Tuple4{1, v12};
    v14 = tmp19.v0; v15 = tmp19.v1;
    while (while_method_0(v14)){
        int v17;
        v17 = v2[v14];
        bool v19;
        v19 = v15 >= v17;
        int v20;
        if (v19){
            v20 = v15;
        } else {
            v20 = v17;
        }
        v15 = v20;
        v14 += 1 ;
    }
    bool v21;
    v21 = v10 < v15;
    int v22; int v23;
    Tuple4 tmp20 = Tuple4{0, 0};
    v22 = tmp20.v0; v23 = tmp20.v1;
    while (while_method_0(v22)){
        int v25;
        v25 = v4[v22];
        bool v27;
        v27 = 0 < v25;
        int v28;
        if (v27){
            v28 = 1;
        } else {
            v28 = 0;
        }
        int v29;
        v29 = v23 + v28;
        v23 = v29;
        v22 += 1 ;
    }
    if (v9){
        if (v21){
            return true;
        } else {
            bool v30;
            v30 = v3 < 2;
            if (v30){
                bool v31;
                v31 = 0 < v23;
                return v31;
            } else {
                return false;
            }
        }
    } else {
        return false;
    }
}
__device__ Union4 go_next_street_38(int v0, static_array<static_array<unsigned char,2>,2> v1, static_array<int,2> v2, int v3, static_array<int,2> v4, Union5 v5){
    switch (v5.tag) {
        case 0: { // Flop
            static_array<unsigned char,3> v7 = v5.case0.v0;
            return Union4{Union4_7{v0, v1, v2, v3, v4, v5}};
            break;
        }
        case 1: { // Preflop
            return Union4{Union4_0{v0, v1, v2, v3, v4, v5}};
            break;
        }
        case 2: { // River
            static_array<unsigned char,5> v11 = v5.case2.v0;
            return Union4{Union4_6{v0, v1, v2, v3, v4, v5}};
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4> v9 = v5.case3.v0;
            return Union4{Union4_3{v0, v1, v2, v3, v4, v5}};
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ Union4 try_round_36(int v0, static_array<static_array<unsigned char,2>,2> v1, static_array<int,2> v2, int v3, static_array<int,2> v4, Union5 v5){
    int v6;
    v6 = v3 + 1;
    bool v7;
    v7 = player_can_act_37(v0, v1, v2, v3, v4, v5);
    if (v7){
        return Union4{Union4_4{v0, v1, v2, v3, v4, v5}};
    } else {
        bool v9;
        v9 = player_can_act_37(v0, v1, v2, v6, v4, v5);
        if (v9){
            return Union4{Union4_4{v0, v1, v2, v6, v4, v5}};
        } else {
            return go_next_street_38(v0, v1, v2, v3, v4, v5);
        }
    }
}
__device__ Tuple11 draw_cards_39(curandStatePhilox4_32_10_t & v0, unsigned long long v1){
    static_array<unsigned char,2> v2;
    int v4; unsigned long long v5;
    Tuple9 tmp21 = Tuple9{0, v1};
    v4 = tmp21.v0; v5 = tmp21.v1;
    while (while_method_0(v4)){
        unsigned char v7; unsigned long long v8;
        Tuple10 tmp22 = draw_card_33(v0, v5);
        v7 = tmp22.v0; v8 = tmp22.v1;
        v2[v4] = v7;
        v5 = v8;
        v4 += 1 ;
    }
    return Tuple11{v2, v5};
}
__device__ inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 1;
    return v1;
}
__device__ Tuple12 draw_cards_40(curandStatePhilox4_32_10_t & v0, unsigned long long v1){
    static_array<unsigned char,1> v2;
    int v4; unsigned long long v5;
    Tuple9 tmp25 = Tuple9{0, v1};
    v4 = tmp25.v0; v5 = tmp25.v1;
    while (while_method_6(v4)){
        unsigned char v7; unsigned long long v8;
        Tuple10 tmp26 = draw_card_33(v0, v5);
        v7 = tmp26.v0; v8 = tmp26.v1;
        v2[v4] = v7;
        v5 = v8;
        v4 += 1 ;
    }
    return Tuple12{v2, v5};
}
__device__ static_array_list<unsigned char,5> get_community_cards_41(Union5 v0, static_array<unsigned char,1> v1){
    static_array_list<unsigned char,5> v2;
    v2 = static_array_list<unsigned char,5>{};
    switch (v0.tag) {
        case 0: { // Flop
            static_array<unsigned char,3> v4 = v0.case0.v0;
            int v5;
            v5 = 0;
            while (while_method_1(v5)){
                unsigned char v7;
                v7 = v4[v5];
                v2.push(v7);
                v5 += 1 ;
            }
            break;
        }
        case 1: { // Preflop
            break;
        }
        case 2: { // River
            static_array<unsigned char,5> v14 = v0.case2.v0;
            int v15;
            v15 = 0;
            while (while_method_2(v15)){
                unsigned char v17;
                v17 = v14[v15];
                v2.push(v17);
                v15 += 1 ;
            }
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4> v9 = v0.case3.v0;
            int v10;
            v10 = 0;
            while (while_method_3(v10)){
                unsigned char v12;
                v12 = v9[v10];
                v2.push(v12);
                v10 += 1 ;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v19;
    v19 = 0;
    while (while_method_6(v19)){
        unsigned char v21;
        v21 = v1[v19];
        v2.push(v21);
        v19 += 1 ;
    }
    return v2;
}
__device__ inline bool while_method_7(int v0){
    bool v1;
    v1 = v0 < 524288;
    return v1;
}
__device__ inline bool while_method_8(int v0){
    bool v1;
    v1 = v0 < 10;
    return v1;
}
__device__ void method_42(unsigned int v0, float * v1, int v2){
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
    Tuple13 tmp28 = Tuple13{0, v3};
    v8 = tmp28.v0; v9 = tmp28.v1;
    while (while_method_8(v8)){
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
__device__ inline bool while_method_9(int v0){
    bool v1;
    v1 = v0 < 11;
    return v1;
}
__device__ void method_43(unsigned int v0, float * v1, int v2){
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
    Tuple13 tmp29 = Tuple13{0, v3};
    v8 = tmp29.v0; v9 = tmp29.v1;
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
__device__ int int_range_44(int v0, int v1, curandStatePhilox4_32_10_t & v2){
    int v3;
    v3 = v0 - v1;
    unsigned int v4;
    v4 = (unsigned int)v3;
    unsigned int v5;
    v5 = loop_34(v4, v2);
    unsigned int v6;
    v6 = (unsigned int)v1;
    unsigned int v7;
    v7 = v5 + v6;
    int v8;
    v8 = (int)v7;
    return v8;
}
__device__ inline bool while_method_10(int v0){
    bool v1;
    v1 = v0 < 8;
    return v1;
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
__device__ void method_45(unsigned int * v0, int v1, float * v2){
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
            while (while_method_3(v37)){
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
            while (while_method_3(v74)){
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
            while (while_method_3(v86)){
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
__device__ Tuple14 method_46(curandStatePhilox4_32_10_t & v0, int * v1, float * v2, float * v3, float * v4, float * v5, float * v6, float * v7, float * v8, int v9, int v10){
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
    while (while_method_3(v60)){
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
            while (while_method_3(v95)){
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
            while (while_method_3(v121)){
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
            while (while_method_3(v131)){
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
            while (while_method_3(v143)){
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
            while (while_method_3(v158)){
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
            while (while_method_3(v167)){
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
            while (while_method_3(v184)){
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
            Tuple15 tmp30 = Tuple15{0, 0.0f};
            v201 = tmp30.v0; v202 = tmp30.v1;
            while (while_method_3(v201)){
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
            Tuple15 tmp31 = Tuple15{0, v217};
            v218 = tmp31.v0; v219 = tmp31.v1;
            while (while_method_3(v218)){
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
            while (while_method_3(v229)){
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
        Tuple16 tmp32 = Tuple16{-1.0f / 0.0f, false};
        v236 = tmp32.v0; v237 = tmp32.v1;
        int v238;
        v238 = 0;
        while (while_method_6(v238)){
            int v240;
            v240 = 0;
            while (while_method_3(v240)){
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
        Tuple16 tmp33 = cooperative_groups::reduce(v257, Tuple16{v236, v237}, v258);
        v259 = tmp33.v0; v260 = tmp33.v1;
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
            while (while_method_3(v267)){
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
        Tuple14 tmp34 = Tuple14{0.0f, 2147483647};
        v273 = tmp34.v0; v274 = tmp34.v1;
        int v275;
        v275 = 0;
        while (while_method_6(v275)){
            int v277;
            v277 = 0;
            while (while_method_3(v277)){
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
        Tuple14 tmp35 = cooperative_groups::reduce(v289, Tuple14{v273, v274}, v290);
        v291 = tmp35.v0; v292 = tmp35.v1;
        float v293;
        v293 = v259 * v291;
        int v294[4];
        bool v295[4];
        int v296;
        v296 = 0;
        while (while_method_6(v296)){
            int v298;
            v298 = 0;
            while (while_method_3(v298)){
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
        Tuple17 tmp36 = Tuple17{2147483647, false};
        v309 = tmp36.v0; v310 = tmp36.v1;
        int v311;
        v311 = 0;
        while (while_method_6(v311)){
            int v313;
            v313 = 0;
            while (while_method_3(v313)){
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
        Tuple17 tmp37 = cooperative_groups::reduce(v330, Tuple17{v309, v310}, v331);
        v332 = tmp37.v0; v333 = tmp37.v1;
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
            while (while_method_3(v339)){
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
            while (while_method_3(v349)){
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
            while (while_method_3(v361)){
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
            while (while_method_3(v375)){
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
            while (while_method_3(v384)){
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
            while (while_method_3(v400)){
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
        Tuple14 tmp38 = Tuple14{0.0f, 2147483647};
        v412 = tmp38.v0; v413 = tmp38.v1;
        int v414;
        v414 = 0;
        while (while_method_6(v414)){
            int v416;
            v416 = 0;
            while (while_method_3(v416)){
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
        Tuple14 tmp39 = cooperative_groups::reduce(v431, Tuple14{v412, v413}, v432);
        v433 = tmp39.v0; v434 = tmp39.v1;
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
    return Tuple14{v441, v442};
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
__device__ int loop_50(static_array<float,6> v0, float v1, int v2){
    bool v3;
    v3 = v2 < 6;
    if (v3){
        float v4;
        v4 = v0[v2];
        bool v6;
        v6 = v1 <= v4;
        if (v6){
            return v2;
        } else {
            int v7;
            v7 = v2 + 1;
            return loop_50(v0, v1, v7);
        }
    } else {
        return 5;
    }
}
__device__ int pick_discrete__49(static_array<float,6> v0, float v1){
    static_array<float,6> v2;
    int v4;
    v4 = 0;
    while (while_method_14(v4)){
        float v6;
        v6 = v0[v4];
        v2[v4] = v6;
        v4 += 1 ;
    }
    int v8;
    v8 = 1;
    while (while_method_15(v2, v8)){
        int v10;
        v10 = 6;
        while (while_method_16(v8, v10)){
            v10 -= 1 ;
            int v12;
            v12 = v10 - v8;
            float v13;
            v13 = v2[v12];
            float v15;
            v15 = v2[v10];
            float v17;
            v17 = v13 + v15;
            v2[v10] = v17;
        }
        int v18;
        v18 = v8 * 2;
        v8 = v18;
    }
    float v19;
    v19 = v2[5];
    float v21;
    v21 = v1 * v19;
    int v22;
    v22 = 0;
    return loop_50(v2, v21, v22);
}
__device__ int sample_discrete__48(static_array<float,6> v0, curandStatePhilox4_32_10_t & v1){
    float v2;
    v2 = curand_uniform(&v1);
    return pick_discrete__49(v0, v2);
}
__device__ Union1 sample_discrete_47(static_array<Tuple18,6> v0, curandStatePhilox4_32_10_t & v1){
    static_array<float,6> v2;
    int v4;
    v4 = 0;
    while (while_method_14(v4)){
        Union1 v6; float v7;
        Tuple18 tmp49 = v0[v4];
        v6 = tmp49.v0; v7 = tmp49.v1;
        v2[v4] = v7;
        v4 += 1 ;
    }
    int v10;
    v10 = sample_discrete__48(v2, v1);
    Union1 v11; float v12;
    Tuple18 tmp50 = v0[v10];
    v11 = tmp50.v0; v12 = tmp50.v1;
    return v11;
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
__device__ Tuple0 score_51(static_array<unsigned char,7> v0){
    static_array<unsigned char,7> v1;
    int v3;
    v3 = 0;
    while (while_method_17(v3)){
        unsigned char v5;
        v5 = v0[v3];
        v1[v3] = v5;
        v3 += 1 ;
    }
    static_array<unsigned char,7> v7;
    bool v9; int v10;
    Tuple19 tmp57 = Tuple19{true, 1};
    v9 = tmp57.v0; v10 = tmp57.v1;
    while (while_method_18(v1, v9, v10)){
        int v12;
        v12 = 0;
        while (while_method_19(v1, v12)){
            int v14;
            v14 = v12 + v10;
            bool v15;
            v15 = v14 < 7;
            int v16;
            if (v15){
                v16 = v14;
            } else {
                v16 = 7;
            }
            int v17;
            v17 = v10 * 2;
            int v18;
            v18 = v12 + v17;
            bool v19;
            v19 = v18 < 7;
            int v20;
            if (v19){
                v20 = v18;
            } else {
                v20 = 7;
            }
            int v21; int v22; int v23;
            Tuple20 tmp58 = Tuple20{v12, v16, v12};
            v21 = tmp58.v0; v22 = tmp58.v1; v23 = tmp58.v2;
            while (while_method_20(v20, v21, v22, v23)){
                bool v25;
                v25 = v21 < v16;
                bool v27;
                if (v25){
                    bool v26;
                    v26 = v22 < v20;
                    v27 = v26;
                } else {
                    v27 = false;
                }
                unsigned char v77; int v78; int v79;
                if (v27){
                    unsigned char v32;
                    if (v9){
                        unsigned char v28;
                        v28 = v1[v21];
                        v32 = v28;
                    } else {
                        unsigned char v30;
                        v30 = v7[v21];
                        v32 = v30;
                    }
                    unsigned char v37;
                    if (v9){
                        unsigned char v33;
                        v33 = v1[v22];
                        v37 = v33;
                    } else {
                        unsigned char v35;
                        v35 = v7[v22];
                        v37 = v35;
                    }
                    unsigned char v38;
                    v38 = v37 / 4u;
                    unsigned char v39;
                    v39 = v32 / 4u;
                    bool v40;
                    v40 = v38 < v39;
                    Union11 v46;
                    if (v40){
                        v46 = Union11{Union11_2{}};
                    } else {
                        bool v42;
                        v42 = v38 > v39;
                        if (v42){
                            v46 = Union11{Union11_1{}};
                        } else {
                            v46 = Union11{Union11_0{}};
                        }
                    }
                    Union11 v56;
                    switch (v46.tag) {
                        case 0: { // Eq
                            unsigned char v47;
                            v47 = v32 % 4u;
                            unsigned char v48;
                            v48 = v37 % 4u;
                            bool v49;
                            v49 = v47 < v48;
                            if (v49){
                                v56 = Union11{Union11_2{}};
                            } else {
                                bool v51;
                                v51 = v47 > v48;
                                if (v51){
                                    v56 = Union11{Union11_1{}};
                                } else {
                                    v56 = Union11{Union11_0{}};
                                }
                            }
                            break;
                        }
                        default: {
                            v56 = v46;
                        }
                    }
                    switch (v56.tag) {
                        case 1: { // Gt
                            int v57;
                            v57 = v22 + 1;
                            v77 = v37; v78 = v21; v79 = v57;
                            break;
                        }
                        default: {
                            int v58;
                            v58 = v21 + 1;
                            v77 = v32; v78 = v58; v79 = v22;
                        }
                    }
                } else {
                    if (v25){
                        unsigned char v66;
                        if (v9){
                            unsigned char v62;
                            v62 = v1[v21];
                            v66 = v62;
                        } else {
                            unsigned char v64;
                            v64 = v7[v21];
                            v66 = v64;
                        }
                        int v67;
                        v67 = v21 + 1;
                        v77 = v66; v78 = v67; v79 = v22;
                    } else {
                        unsigned char v72;
                        if (v9){
                            unsigned char v68;
                            v68 = v1[v22];
                            v72 = v68;
                        } else {
                            unsigned char v70;
                            v70 = v7[v22];
                            v72 = v70;
                        }
                        int v73;
                        v73 = v22 + 1;
                        v77 = v72; v78 = v21; v79 = v73;
                    }
                }
                if (v9){
                    v7[v23] = v77;
                } else {
                    v1[v23] = v77;
                }
                int v80;
                v80 = v23 + 1;
                v21 = v78;
                v22 = v79;
                v23 = v80;
            }
            v12 = v18;
        }
        bool v81;
        v81 = v9 == false;
        int v82;
        v82 = v10 * 2;
        v9 = v81;
        v10 = v82;
    }
    bool v83;
    v83 = v9 == false;
    static_array<unsigned char,7> v84;
    if (v83){
        v84 = v7;
    } else {
        v84 = v1;
    }
    static_array<unsigned char,5> v85;
    int v87; int v88; unsigned char v89;
    Tuple21 tmp59 = Tuple21{0, 0, 12u};
    v87 = tmp59.v0; v88 = tmp59.v1; v89 = tmp59.v2;
    while (while_method_17(v87)){
        unsigned char v91;
        v91 = v84[v87];
        bool v93;
        v93 = v88 < 5;
        int v105; unsigned char v106;
        if (v93){
            unsigned char v94;
            v94 = v91 % 4u;
            bool v95;
            v95 = 0u == v94;
            if (v95){
                unsigned char v96;
                v96 = v91 / 4u;
                bool v97;
                v97 = v89 == v96;
                int v98;
                if (v97){
                    v98 = v88;
                } else {
                    v98 = 0;
                }
                v85[v98] = v91;
                int v99;
                v99 = v98 + 1;
                unsigned char v100;
                v100 = v96 - 1u;
                v105 = v99; v106 = v100;
            } else {
                v105 = v88; v106 = v89;
            }
        } else {
            break;
        }
        v88 = v105;
        v89 = v106;
        v87 += 1 ;
    }
    bool v107;
    v107 = v88 == 4;
    bool v146;
    if (v107){
        unsigned char v108;
        v108 = v89 + 1u;
        bool v109;
        v109 = v108 == 0u;
        if (v109){
            unsigned char v110;
            v110 = v84[0];
            unsigned char v112;
            v112 = v110 % 4u;
            bool v113;
            v113 = 0u == v112;
            bool v117;
            if (v113){
                unsigned char v114;
                v114 = v110 / 4u;
                bool v115;
                v115 = v114 == 12u;
                if (v115){
                    v85[4] = v110;
                    v117 = true;
                } else {
                    v117 = false;
                }
            } else {
                v117 = false;
            }
            if (v117){
                v146 = true;
            } else {
                unsigned char v118;
                v118 = v84[1];
                unsigned char v120;
                v120 = v118 % 4u;
                bool v121;
                v121 = 0u == v120;
                bool v125;
                if (v121){
                    unsigned char v122;
                    v122 = v118 / 4u;
                    bool v123;
                    v123 = v122 == 12u;
                    if (v123){
                        v85[4] = v118;
                        v125 = true;
                    } else {
                        v125 = false;
                    }
                } else {
                    v125 = false;
                }
                if (v125){
                    v146 = true;
                } else {
                    unsigned char v126;
                    v126 = v84[2];
                    unsigned char v128;
                    v128 = v126 % 4u;
                    bool v129;
                    v129 = 0u == v128;
                    bool v133;
                    if (v129){
                        unsigned char v130;
                        v130 = v126 / 4u;
                        bool v131;
                        v131 = v130 == 12u;
                        if (v131){
                            v85[4] = v126;
                            v133 = true;
                        } else {
                            v133 = false;
                        }
                    } else {
                        v133 = false;
                    }
                    if (v133){
                        v146 = true;
                    } else {
                        unsigned char v134;
                        v134 = v84[3];
                        unsigned char v136;
                        v136 = v134 % 4u;
                        bool v137;
                        v137 = 0u == v136;
                        if (v137){
                            unsigned char v138;
                            v138 = v134 / 4u;
                            bool v139;
                            v139 = v138 == 12u;
                            if (v139){
                                v85[4] = v134;
                                v146 = true;
                            } else {
                                v146 = false;
                            }
                        } else {
                            v146 = false;
                        }
                    }
                }
            }
        } else {
            v146 = false;
        }
    } else {
        v146 = false;
    }
    Union12 v152;
    if (v146){
        v152 = Union12{Union12_1{v85}};
    } else {
        bool v148;
        v148 = v88 == 5;
        if (v148){
            v152 = Union12{Union12_1{v85}};
        } else {
            v152 = Union12{Union12_0{}};
        }
    }
    static_array<unsigned char,5> v153;
    int v155; int v156; unsigned char v157;
    Tuple21 tmp60 = Tuple21{0, 0, 12u};
    v155 = tmp60.v0; v156 = tmp60.v1; v157 = tmp60.v2;
    while (while_method_17(v155)){
        unsigned char v159;
        v159 = v84[v155];
        bool v161;
        v161 = v156 < 5;
        int v173; unsigned char v174;
        if (v161){
            unsigned char v162;
            v162 = v159 % 4u;
            bool v163;
            v163 = 1u == v162;
            if (v163){
                unsigned char v164;
                v164 = v159 / 4u;
                bool v165;
                v165 = v157 == v164;
                int v166;
                if (v165){
                    v166 = v156;
                } else {
                    v166 = 0;
                }
                v153[v166] = v159;
                int v167;
                v167 = v166 + 1;
                unsigned char v168;
                v168 = v164 - 1u;
                v173 = v167; v174 = v168;
            } else {
                v173 = v156; v174 = v157;
            }
        } else {
            break;
        }
        v156 = v173;
        v157 = v174;
        v155 += 1 ;
    }
    bool v175;
    v175 = v156 == 4;
    bool v214;
    if (v175){
        unsigned char v176;
        v176 = v157 + 1u;
        bool v177;
        v177 = v176 == 0u;
        if (v177){
            unsigned char v178;
            v178 = v84[0];
            unsigned char v180;
            v180 = v178 % 4u;
            bool v181;
            v181 = 1u == v180;
            bool v185;
            if (v181){
                unsigned char v182;
                v182 = v178 / 4u;
                bool v183;
                v183 = v182 == 12u;
                if (v183){
                    v153[4] = v178;
                    v185 = true;
                } else {
                    v185 = false;
                }
            } else {
                v185 = false;
            }
            if (v185){
                v214 = true;
            } else {
                unsigned char v186;
                v186 = v84[1];
                unsigned char v188;
                v188 = v186 % 4u;
                bool v189;
                v189 = 1u == v188;
                bool v193;
                if (v189){
                    unsigned char v190;
                    v190 = v186 / 4u;
                    bool v191;
                    v191 = v190 == 12u;
                    if (v191){
                        v153[4] = v186;
                        v193 = true;
                    } else {
                        v193 = false;
                    }
                } else {
                    v193 = false;
                }
                if (v193){
                    v214 = true;
                } else {
                    unsigned char v194;
                    v194 = v84[2];
                    unsigned char v196;
                    v196 = v194 % 4u;
                    bool v197;
                    v197 = 1u == v196;
                    bool v201;
                    if (v197){
                        unsigned char v198;
                        v198 = v194 / 4u;
                        bool v199;
                        v199 = v198 == 12u;
                        if (v199){
                            v153[4] = v194;
                            v201 = true;
                        } else {
                            v201 = false;
                        }
                    } else {
                        v201 = false;
                    }
                    if (v201){
                        v214 = true;
                    } else {
                        unsigned char v202;
                        v202 = v84[3];
                        unsigned char v204;
                        v204 = v202 % 4u;
                        bool v205;
                        v205 = 1u == v204;
                        if (v205){
                            unsigned char v206;
                            v206 = v202 / 4u;
                            bool v207;
                            v207 = v206 == 12u;
                            if (v207){
                                v153[4] = v202;
                                v214 = true;
                            } else {
                                v214 = false;
                            }
                        } else {
                            v214 = false;
                        }
                    }
                }
            }
        } else {
            v214 = false;
        }
    } else {
        v214 = false;
    }
    Union12 v220;
    if (v214){
        v220 = Union12{Union12_1{v153}};
    } else {
        bool v216;
        v216 = v156 == 5;
        if (v216){
            v220 = Union12{Union12_1{v153}};
        } else {
            v220 = Union12{Union12_0{}};
        }
    }
    Union12 v248;
    switch (v152.tag) {
        case 0: { // None
            v248 = v220;
            break;
        }
        case 1: { // Some
            static_array<unsigned char,5> v221 = v152.case1.v0;
            switch (v220.tag) {
                case 0: { // None
                    v248 = v152;
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,5> v222 = v220.case1.v0;
                    Union11 v223;
                    v223 = Union11{Union11_0{}};
                    int v224; Union11 v225;
                    Tuple22 tmp61 = Tuple22{0, v223};
                    v224 = tmp61.v0; v225 = tmp61.v1;
                    while (while_method_2(v224)){
                        unsigned char v227;
                        v227 = v221[v224];
                        unsigned char v229;
                        v229 = v222[v224];
                        Union11 v241;
                        switch (v225.tag) {
                            case 0: { // Eq
                                unsigned char v231;
                                v231 = v227 / 4u;
                                unsigned char v232;
                                v232 = v229 / 4u;
                                bool v233;
                                v233 = v231 < v232;
                                if (v233){
                                    v241 = Union11{Union11_2{}};
                                } else {
                                    bool v235;
                                    v235 = v231 > v232;
                                    if (v235){
                                        v241 = Union11{Union11_1{}};
                                    } else {
                                        v241 = Union11{Union11_0{}};
                                    }
                                }
                                break;
                            }
                            default: {
                                break;
                            }
                        }
                        v225 = v241;
                        v224 += 1 ;
                    }
                    bool v242;
                    switch (v225.tag) {
                        case 1: { // Gt
                            v242 = true;
                            break;
                        }
                        default: {
                            v242 = false;
                        }
                    }
                    static_array<unsigned char,5> v243;
                    if (v242){
                        v243 = v221;
                    } else {
                        v243 = v222;
                    }
                    v248 = Union12{Union12_1{v243}};
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
    static_array<unsigned char,5> v249;
    int v251; int v252; unsigned char v253;
    Tuple21 tmp62 = Tuple21{0, 0, 12u};
    v251 = tmp62.v0; v252 = tmp62.v1; v253 = tmp62.v2;
    while (while_method_17(v251)){
        unsigned char v255;
        v255 = v84[v251];
        bool v257;
        v257 = v252 < 5;
        int v269; unsigned char v270;
        if (v257){
            unsigned char v258;
            v258 = v255 % 4u;
            bool v259;
            v259 = 2u == v258;
            if (v259){
                unsigned char v260;
                v260 = v255 / 4u;
                bool v261;
                v261 = v253 == v260;
                int v262;
                if (v261){
                    v262 = v252;
                } else {
                    v262 = 0;
                }
                v249[v262] = v255;
                int v263;
                v263 = v262 + 1;
                unsigned char v264;
                v264 = v260 - 1u;
                v269 = v263; v270 = v264;
            } else {
                v269 = v252; v270 = v253;
            }
        } else {
            break;
        }
        v252 = v269;
        v253 = v270;
        v251 += 1 ;
    }
    bool v271;
    v271 = v252 == 4;
    bool v310;
    if (v271){
        unsigned char v272;
        v272 = v253 + 1u;
        bool v273;
        v273 = v272 == 0u;
        if (v273){
            unsigned char v274;
            v274 = v84[0];
            unsigned char v276;
            v276 = v274 % 4u;
            bool v277;
            v277 = 2u == v276;
            bool v281;
            if (v277){
                unsigned char v278;
                v278 = v274 / 4u;
                bool v279;
                v279 = v278 == 12u;
                if (v279){
                    v249[4] = v274;
                    v281 = true;
                } else {
                    v281 = false;
                }
            } else {
                v281 = false;
            }
            if (v281){
                v310 = true;
            } else {
                unsigned char v282;
                v282 = v84[1];
                unsigned char v284;
                v284 = v282 % 4u;
                bool v285;
                v285 = 2u == v284;
                bool v289;
                if (v285){
                    unsigned char v286;
                    v286 = v282 / 4u;
                    bool v287;
                    v287 = v286 == 12u;
                    if (v287){
                        v249[4] = v282;
                        v289 = true;
                    } else {
                        v289 = false;
                    }
                } else {
                    v289 = false;
                }
                if (v289){
                    v310 = true;
                } else {
                    unsigned char v290;
                    v290 = v84[2];
                    unsigned char v292;
                    v292 = v290 % 4u;
                    bool v293;
                    v293 = 2u == v292;
                    bool v297;
                    if (v293){
                        unsigned char v294;
                        v294 = v290 / 4u;
                        bool v295;
                        v295 = v294 == 12u;
                        if (v295){
                            v249[4] = v290;
                            v297 = true;
                        } else {
                            v297 = false;
                        }
                    } else {
                        v297 = false;
                    }
                    if (v297){
                        v310 = true;
                    } else {
                        unsigned char v298;
                        v298 = v84[3];
                        unsigned char v300;
                        v300 = v298 % 4u;
                        bool v301;
                        v301 = 2u == v300;
                        if (v301){
                            unsigned char v302;
                            v302 = v298 / 4u;
                            bool v303;
                            v303 = v302 == 12u;
                            if (v303){
                                v249[4] = v298;
                                v310 = true;
                            } else {
                                v310 = false;
                            }
                        } else {
                            v310 = false;
                        }
                    }
                }
            }
        } else {
            v310 = false;
        }
    } else {
        v310 = false;
    }
    Union12 v316;
    if (v310){
        v316 = Union12{Union12_1{v249}};
    } else {
        bool v312;
        v312 = v252 == 5;
        if (v312){
            v316 = Union12{Union12_1{v249}};
        } else {
            v316 = Union12{Union12_0{}};
        }
    }
    Union12 v344;
    switch (v248.tag) {
        case 0: { // None
            v344 = v316;
            break;
        }
        case 1: { // Some
            static_array<unsigned char,5> v317 = v248.case1.v0;
            switch (v316.tag) {
                case 0: { // None
                    v344 = v248;
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,5> v318 = v316.case1.v0;
                    Union11 v319;
                    v319 = Union11{Union11_0{}};
                    int v320; Union11 v321;
                    Tuple22 tmp63 = Tuple22{0, v319};
                    v320 = tmp63.v0; v321 = tmp63.v1;
                    while (while_method_2(v320)){
                        unsigned char v323;
                        v323 = v317[v320];
                        unsigned char v325;
                        v325 = v318[v320];
                        Union11 v337;
                        switch (v321.tag) {
                            case 0: { // Eq
                                unsigned char v327;
                                v327 = v323 / 4u;
                                unsigned char v328;
                                v328 = v325 / 4u;
                                bool v329;
                                v329 = v327 < v328;
                                if (v329){
                                    v337 = Union11{Union11_2{}};
                                } else {
                                    bool v331;
                                    v331 = v327 > v328;
                                    if (v331){
                                        v337 = Union11{Union11_1{}};
                                    } else {
                                        v337 = Union11{Union11_0{}};
                                    }
                                }
                                break;
                            }
                            default: {
                                break;
                            }
                        }
                        v321 = v337;
                        v320 += 1 ;
                    }
                    bool v338;
                    switch (v321.tag) {
                        case 1: { // Gt
                            v338 = true;
                            break;
                        }
                        default: {
                            v338 = false;
                        }
                    }
                    static_array<unsigned char,5> v339;
                    if (v338){
                        v339 = v317;
                    } else {
                        v339 = v318;
                    }
                    v344 = Union12{Union12_1{v339}};
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
    static_array<unsigned char,5> v345;
    int v347; int v348; unsigned char v349;
    Tuple21 tmp64 = Tuple21{0, 0, 12u};
    v347 = tmp64.v0; v348 = tmp64.v1; v349 = tmp64.v2;
    while (while_method_17(v347)){
        unsigned char v351;
        v351 = v84[v347];
        bool v353;
        v353 = v348 < 5;
        int v365; unsigned char v366;
        if (v353){
            unsigned char v354;
            v354 = v351 % 4u;
            bool v355;
            v355 = 3u == v354;
            if (v355){
                unsigned char v356;
                v356 = v351 / 4u;
                bool v357;
                v357 = v349 == v356;
                int v358;
                if (v357){
                    v358 = v348;
                } else {
                    v358 = 0;
                }
                v345[v358] = v351;
                int v359;
                v359 = v358 + 1;
                unsigned char v360;
                v360 = v356 - 1u;
                v365 = v359; v366 = v360;
            } else {
                v365 = v348; v366 = v349;
            }
        } else {
            break;
        }
        v348 = v365;
        v349 = v366;
        v347 += 1 ;
    }
    bool v367;
    v367 = v348 == 4;
    bool v406;
    if (v367){
        unsigned char v368;
        v368 = v349 + 1u;
        bool v369;
        v369 = v368 == 0u;
        if (v369){
            unsigned char v370;
            v370 = v84[0];
            unsigned char v372;
            v372 = v370 % 4u;
            bool v373;
            v373 = 3u == v372;
            bool v377;
            if (v373){
                unsigned char v374;
                v374 = v370 / 4u;
                bool v375;
                v375 = v374 == 12u;
                if (v375){
                    v345[4] = v370;
                    v377 = true;
                } else {
                    v377 = false;
                }
            } else {
                v377 = false;
            }
            if (v377){
                v406 = true;
            } else {
                unsigned char v378;
                v378 = v84[1];
                unsigned char v380;
                v380 = v378 % 4u;
                bool v381;
                v381 = 3u == v380;
                bool v385;
                if (v381){
                    unsigned char v382;
                    v382 = v378 / 4u;
                    bool v383;
                    v383 = v382 == 12u;
                    if (v383){
                        v345[4] = v378;
                        v385 = true;
                    } else {
                        v385 = false;
                    }
                } else {
                    v385 = false;
                }
                if (v385){
                    v406 = true;
                } else {
                    unsigned char v386;
                    v386 = v84[2];
                    unsigned char v388;
                    v388 = v386 % 4u;
                    bool v389;
                    v389 = 3u == v388;
                    bool v393;
                    if (v389){
                        unsigned char v390;
                        v390 = v386 / 4u;
                        bool v391;
                        v391 = v390 == 12u;
                        if (v391){
                            v345[4] = v386;
                            v393 = true;
                        } else {
                            v393 = false;
                        }
                    } else {
                        v393 = false;
                    }
                    if (v393){
                        v406 = true;
                    } else {
                        unsigned char v394;
                        v394 = v84[3];
                        unsigned char v396;
                        v396 = v394 % 4u;
                        bool v397;
                        v397 = 3u == v396;
                        if (v397){
                            unsigned char v398;
                            v398 = v394 / 4u;
                            bool v399;
                            v399 = v398 == 12u;
                            if (v399){
                                v345[4] = v394;
                                v406 = true;
                            } else {
                                v406 = false;
                            }
                        } else {
                            v406 = false;
                        }
                    }
                }
            }
        } else {
            v406 = false;
        }
    } else {
        v406 = false;
    }
    Union12 v412;
    if (v406){
        v412 = Union12{Union12_1{v345}};
    } else {
        bool v408;
        v408 = v348 == 5;
        if (v408){
            v412 = Union12{Union12_1{v345}};
        } else {
            v412 = Union12{Union12_0{}};
        }
    }
    Union12 v440;
    switch (v344.tag) {
        case 0: { // None
            v440 = v412;
            break;
        }
        case 1: { // Some
            static_array<unsigned char,5> v413 = v344.case1.v0;
            switch (v412.tag) {
                case 0: { // None
                    v440 = v344;
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,5> v414 = v412.case1.v0;
                    Union11 v415;
                    v415 = Union11{Union11_0{}};
                    int v416; Union11 v417;
                    Tuple22 tmp65 = Tuple22{0, v415};
                    v416 = tmp65.v0; v417 = tmp65.v1;
                    while (while_method_2(v416)){
                        unsigned char v419;
                        v419 = v413[v416];
                        unsigned char v421;
                        v421 = v414[v416];
                        Union11 v433;
                        switch (v417.tag) {
                            case 0: { // Eq
                                unsigned char v423;
                                v423 = v419 / 4u;
                                unsigned char v424;
                                v424 = v421 / 4u;
                                bool v425;
                                v425 = v423 < v424;
                                if (v425){
                                    v433 = Union11{Union11_2{}};
                                } else {
                                    bool v427;
                                    v427 = v423 > v424;
                                    if (v427){
                                        v433 = Union11{Union11_1{}};
                                    } else {
                                        v433 = Union11{Union11_0{}};
                                    }
                                }
                                break;
                            }
                            default: {
                                break;
                            }
                        }
                        v417 = v433;
                        v416 += 1 ;
                    }
                    bool v434;
                    switch (v417.tag) {
                        case 1: { // Gt
                            v434 = true;
                            break;
                        }
                        default: {
                            v434 = false;
                        }
                    }
                    static_array<unsigned char,5> v435;
                    if (v434){
                        v435 = v413;
                    } else {
                        v435 = v414;
                    }
                    v440 = Union12{Union12_1{v435}};
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
    static_array<unsigned char,5> v1037; char v1038;
    switch (v440.tag) {
        case 0: { // None
            static_array<unsigned char,4> v442;
            static_array<unsigned char,3> v444;
            int v446; int v447; int v448; unsigned char v449;
            Tuple23 tmp66 = Tuple23{0, 0, 0, 12u};
            v446 = tmp66.v0; v447 = tmp66.v1; v448 = tmp66.v2; v449 = tmp66.v3;
            while (while_method_17(v446)){
                unsigned char v451;
                v451 = v84[v446];
                bool v453;
                v453 = v448 < 4;
                int v461; int v462; unsigned char v463;
                if (v453){
                    unsigned char v454;
                    v454 = v451 / 4u;
                    bool v455;
                    v455 = v449 == v454;
                    int v456;
                    if (v455){
                        v456 = v448;
                    } else {
                        v456 = 0;
                    }
                    v442[v456] = v451;
                    int v457;
                    v457 = v456 + 1;
                    v461 = v446; v462 = v457; v463 = v454;
                } else {
                    break;
                }
                v447 = v461;
                v448 = v462;
                v449 = v463;
                v446 += 1 ;
            }
            bool v464;
            v464 = v448 == 4;
            Union13 v475;
            if (v464){
                int v465;
                v465 = 0;
                while (while_method_1(v465)){
                    int v467;
                    v467 = v447 + -3;
                    bool v468;
                    v468 = v465 < v467;
                    int v469;
                    if (v468){
                        v469 = 0;
                    } else {
                        v469 = 4;
                    }
                    int v470;
                    v470 = v469 + v465;
                    unsigned char v471;
                    v471 = v84[v470];
                    v444[v465] = v471;
                    v465 += 1 ;
                }
                v475 = Union13{Union13_1{v442, v444}};
            } else {
                v475 = Union13{Union13_0{}};
            }
            Union12 v498;
            switch (v475.tag) {
                case 0: { // None
                    v498 = Union12{Union12_0{}};
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,4> v476 = v475.case1.v0; static_array<unsigned char,3> v477 = v475.case1.v1;
                    static_array<unsigned char,1> v478;
                    int v480;
                    v480 = 0;
                    while (while_method_6(v480)){
                        unsigned char v482;
                        v482 = v477[v480];
                        v478[v480] = v482;
                        v480 += 1 ;
                    }
                    static_array<unsigned char,5> v484;
                    int v486;
                    v486 = 0;
                    while (while_method_3(v486)){
                        unsigned char v488;
                        v488 = v476[v486];
                        v484[v486] = v488;
                        v486 += 1 ;
                    }
                    int v490;
                    v490 = 0;
                    while (while_method_6(v490)){
                        unsigned char v492;
                        v492 = v478[v490];
                        int v494;
                        v494 = 4 + v490;
                        v484[v494] = v492;
                        v490 += 1 ;
                    }
                    v498 = Union12{Union12_1{v484}};
                    break;
                }
                default: {
                    assert("Invalid tag." && false); __trap();
                }
            }
            switch (v498.tag) {
                case 0: { // None
                    static_array<unsigned char,3> v500;
                    static_array<unsigned char,4> v502;
                    int v504; int v505; int v506; unsigned char v507;
                    Tuple23 tmp67 = Tuple23{0, 0, 0, 12u};
                    v504 = tmp67.v0; v505 = tmp67.v1; v506 = tmp67.v2; v507 = tmp67.v3;
                    while (while_method_17(v504)){
                        unsigned char v509;
                        v509 = v84[v504];
                        bool v511;
                        v511 = v506 < 3;
                        int v519; int v520; unsigned char v521;
                        if (v511){
                            unsigned char v512;
                            v512 = v509 / 4u;
                            bool v513;
                            v513 = v507 == v512;
                            int v514;
                            if (v513){
                                v514 = v506;
                            } else {
                                v514 = 0;
                            }
                            v500[v514] = v509;
                            int v515;
                            v515 = v514 + 1;
                            v519 = v504; v520 = v515; v521 = v512;
                        } else {
                            break;
                        }
                        v505 = v519;
                        v506 = v520;
                        v507 = v521;
                        v504 += 1 ;
                    }
                    bool v522;
                    v522 = v506 == 3;
                    Union14 v533;
                    if (v522){
                        int v523;
                        v523 = 0;
                        while (while_method_3(v523)){
                            int v525;
                            v525 = v505 + -2;
                            bool v526;
                            v526 = v523 < v525;
                            int v527;
                            if (v526){
                                v527 = 0;
                            } else {
                                v527 = 3;
                            }
                            int v528;
                            v528 = v527 + v523;
                            unsigned char v529;
                            v529 = v84[v528];
                            v502[v523] = v529;
                            v523 += 1 ;
                        }
                        v533 = Union14{Union14_1{v500, v502}};
                    } else {
                        v533 = Union14{Union14_0{}};
                    }
                    Union12 v589;
                    switch (v533.tag) {
                        case 0: { // None
                            v589 = Union12{Union12_0{}};
                            break;
                        }
                        case 1: { // Some
                            static_array<unsigned char,3> v534 = v533.case1.v0; static_array<unsigned char,4> v535 = v533.case1.v1;
                            static_array<unsigned char,2> v536;
                            static_array<unsigned char,2> v538;
                            int v540; int v541; int v542; unsigned char v543;
                            Tuple23 tmp68 = Tuple23{0, 0, 0, 12u};
                            v540 = tmp68.v0; v541 = tmp68.v1; v542 = tmp68.v2; v543 = tmp68.v3;
                            while (while_method_3(v540)){
                                unsigned char v545;
                                v545 = v535[v540];
                                bool v547;
                                v547 = v542 < 2;
                                int v555; int v556; unsigned char v557;
                                if (v547){
                                    unsigned char v548;
                                    v548 = v545 / 4u;
                                    bool v549;
                                    v549 = v543 == v548;
                                    int v550;
                                    if (v549){
                                        v550 = v542;
                                    } else {
                                        v550 = 0;
                                    }
                                    v536[v550] = v545;
                                    int v551;
                                    v551 = v550 + 1;
                                    v555 = v540; v556 = v551; v557 = v548;
                                } else {
                                    break;
                                }
                                v541 = v555;
                                v542 = v556;
                                v543 = v557;
                                v540 += 1 ;
                            }
                            bool v558;
                            v558 = v542 == 2;
                            Union15 v569;
                            if (v558){
                                int v559;
                                v559 = 0;
                                while (while_method_0(v559)){
                                    int v561;
                                    v561 = v541 + -1;
                                    bool v562;
                                    v562 = v559 < v561;
                                    int v563;
                                    if (v562){
                                        v563 = 0;
                                    } else {
                                        v563 = 2;
                                    }
                                    int v564;
                                    v564 = v563 + v559;
                                    unsigned char v565;
                                    v565 = v535[v564];
                                    v538[v559] = v565;
                                    v559 += 1 ;
                                }
                                v569 = Union15{Union15_1{v536, v538}};
                            } else {
                                v569 = Union15{Union15_0{}};
                            }
                            switch (v569.tag) {
                                case 0: { // None
                                    v589 = Union12{Union12_0{}};
                                    break;
                                }
                                case 1: { // Some
                                    static_array<unsigned char,2> v570 = v569.case1.v0; static_array<unsigned char,2> v571 = v569.case1.v1;
                                    static_array<unsigned char,5> v572;
                                    int v574;
                                    v574 = 0;
                                    while (while_method_1(v574)){
                                        unsigned char v576;
                                        v576 = v534[v574];
                                        v572[v574] = v576;
                                        v574 += 1 ;
                                    }
                                    int v578;
                                    v578 = 0;
                                    while (while_method_0(v578)){
                                        unsigned char v580;
                                        v580 = v570[v578];
                                        int v582;
                                        v582 = 3 + v578;
                                        v572[v582] = v580;
                                        v578 += 1 ;
                                    }
                                    v589 = Union12{Union12_1{v572}};
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
                    switch (v589.tag) {
                        case 0: { // None
                            static_array<unsigned char,5> v591;
                            int v593; int v594;
                            Tuple4 tmp69 = Tuple4{0, 0};
                            v593 = tmp69.v0; v594 = tmp69.v1;
                            while (while_method_17(v593)){
                                unsigned char v596;
                                v596 = v84[v593];
                                unsigned char v598;
                                v598 = v596 % 4u;
                                bool v599;
                                v599 = v598 == 0u;
                                bool v601;
                                if (v599){
                                    bool v600;
                                    v600 = v594 < 5;
                                    v601 = v600;
                                } else {
                                    v601 = false;
                                }
                                int v603;
                                if (v601){
                                    v591[v594] = v596;
                                    int v602;
                                    v602 = v594 + 1;
                                    v603 = v602;
                                } else {
                                    v603 = v594;
                                }
                                v594 = v603;
                                v593 += 1 ;
                            }
                            bool v604;
                            v604 = v594 == 5;
                            Union12 v607;
                            if (v604){
                                v607 = Union12{Union12_1{v591}};
                            } else {
                                v607 = Union12{Union12_0{}};
                            }
                            static_array<unsigned char,5> v608;
                            int v610; int v611;
                            Tuple4 tmp70 = Tuple4{0, 0};
                            v610 = tmp70.v0; v611 = tmp70.v1;
                            while (while_method_17(v610)){
                                unsigned char v613;
                                v613 = v84[v610];
                                unsigned char v615;
                                v615 = v613 % 4u;
                                bool v616;
                                v616 = v615 == 1u;
                                bool v618;
                                if (v616){
                                    bool v617;
                                    v617 = v611 < 5;
                                    v618 = v617;
                                } else {
                                    v618 = false;
                                }
                                int v620;
                                if (v618){
                                    v608[v611] = v613;
                                    int v619;
                                    v619 = v611 + 1;
                                    v620 = v619;
                                } else {
                                    v620 = v611;
                                }
                                v611 = v620;
                                v610 += 1 ;
                            }
                            bool v621;
                            v621 = v611 == 5;
                            Union12 v624;
                            if (v621){
                                v624 = Union12{Union12_1{v608}};
                            } else {
                                v624 = Union12{Union12_0{}};
                            }
                            Union12 v652;
                            switch (v607.tag) {
                                case 0: { // None
                                    v652 = v624;
                                    break;
                                }
                                case 1: { // Some
                                    static_array<unsigned char,5> v625 = v607.case1.v0;
                                    switch (v624.tag) {
                                        case 0: { // None
                                            v652 = v607;
                                            break;
                                        }
                                        case 1: { // Some
                                            static_array<unsigned char,5> v626 = v624.case1.v0;
                                            Union11 v627;
                                            v627 = Union11{Union11_0{}};
                                            int v628; Union11 v629;
                                            Tuple22 tmp71 = Tuple22{0, v627};
                                            v628 = tmp71.v0; v629 = tmp71.v1;
                                            while (while_method_2(v628)){
                                                unsigned char v631;
                                                v631 = v625[v628];
                                                unsigned char v633;
                                                v633 = v626[v628];
                                                Union11 v645;
                                                switch (v629.tag) {
                                                    case 0: { // Eq
                                                        unsigned char v635;
                                                        v635 = v631 / 4u;
                                                        unsigned char v636;
                                                        v636 = v633 / 4u;
                                                        bool v637;
                                                        v637 = v635 < v636;
                                                        if (v637){
                                                            v645 = Union11{Union11_2{}};
                                                        } else {
                                                            bool v639;
                                                            v639 = v635 > v636;
                                                            if (v639){
                                                                v645 = Union11{Union11_1{}};
                                                            } else {
                                                                v645 = Union11{Union11_0{}};
                                                            }
                                                        }
                                                        break;
                                                    }
                                                    default: {
                                                        break;
                                                    }
                                                }
                                                v629 = v645;
                                                v628 += 1 ;
                                            }
                                            bool v646;
                                            switch (v629.tag) {
                                                case 1: { // Gt
                                                    v646 = true;
                                                    break;
                                                }
                                                default: {
                                                    v646 = false;
                                                }
                                            }
                                            static_array<unsigned char,5> v647;
                                            if (v646){
                                                v647 = v625;
                                            } else {
                                                v647 = v626;
                                            }
                                            v652 = Union12{Union12_1{v647}};
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
                            static_array<unsigned char,5> v653;
                            int v655; int v656;
                            Tuple4 tmp72 = Tuple4{0, 0};
                            v655 = tmp72.v0; v656 = tmp72.v1;
                            while (while_method_17(v655)){
                                unsigned char v658;
                                v658 = v84[v655];
                                unsigned char v660;
                                v660 = v658 % 4u;
                                bool v661;
                                v661 = v660 == 2u;
                                bool v663;
                                if (v661){
                                    bool v662;
                                    v662 = v656 < 5;
                                    v663 = v662;
                                } else {
                                    v663 = false;
                                }
                                int v665;
                                if (v663){
                                    v653[v656] = v658;
                                    int v664;
                                    v664 = v656 + 1;
                                    v665 = v664;
                                } else {
                                    v665 = v656;
                                }
                                v656 = v665;
                                v655 += 1 ;
                            }
                            bool v666;
                            v666 = v656 == 5;
                            Union12 v669;
                            if (v666){
                                v669 = Union12{Union12_1{v653}};
                            } else {
                                v669 = Union12{Union12_0{}};
                            }
                            Union12 v697;
                            switch (v652.tag) {
                                case 0: { // None
                                    v697 = v669;
                                    break;
                                }
                                case 1: { // Some
                                    static_array<unsigned char,5> v670 = v652.case1.v0;
                                    switch (v669.tag) {
                                        case 0: { // None
                                            v697 = v652;
                                            break;
                                        }
                                        case 1: { // Some
                                            static_array<unsigned char,5> v671 = v669.case1.v0;
                                            Union11 v672;
                                            v672 = Union11{Union11_0{}};
                                            int v673; Union11 v674;
                                            Tuple22 tmp73 = Tuple22{0, v672};
                                            v673 = tmp73.v0; v674 = tmp73.v1;
                                            while (while_method_2(v673)){
                                                unsigned char v676;
                                                v676 = v670[v673];
                                                unsigned char v678;
                                                v678 = v671[v673];
                                                Union11 v690;
                                                switch (v674.tag) {
                                                    case 0: { // Eq
                                                        unsigned char v680;
                                                        v680 = v676 / 4u;
                                                        unsigned char v681;
                                                        v681 = v678 / 4u;
                                                        bool v682;
                                                        v682 = v680 < v681;
                                                        if (v682){
                                                            v690 = Union11{Union11_2{}};
                                                        } else {
                                                            bool v684;
                                                            v684 = v680 > v681;
                                                            if (v684){
                                                                v690 = Union11{Union11_1{}};
                                                            } else {
                                                                v690 = Union11{Union11_0{}};
                                                            }
                                                        }
                                                        break;
                                                    }
                                                    default: {
                                                        break;
                                                    }
                                                }
                                                v674 = v690;
                                                v673 += 1 ;
                                            }
                                            bool v691;
                                            switch (v674.tag) {
                                                case 1: { // Gt
                                                    v691 = true;
                                                    break;
                                                }
                                                default: {
                                                    v691 = false;
                                                }
                                            }
                                            static_array<unsigned char,5> v692;
                                            if (v691){
                                                v692 = v670;
                                            } else {
                                                v692 = v671;
                                            }
                                            v697 = Union12{Union12_1{v692}};
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
                            static_array<unsigned char,5> v698;
                            int v700; int v701;
                            Tuple4 tmp74 = Tuple4{0, 0};
                            v700 = tmp74.v0; v701 = tmp74.v1;
                            while (while_method_17(v700)){
                                unsigned char v703;
                                v703 = v84[v700];
                                unsigned char v705;
                                v705 = v703 % 4u;
                                bool v706;
                                v706 = v705 == 3u;
                                bool v708;
                                if (v706){
                                    bool v707;
                                    v707 = v701 < 5;
                                    v708 = v707;
                                } else {
                                    v708 = false;
                                }
                                int v710;
                                if (v708){
                                    v698[v701] = v703;
                                    int v709;
                                    v709 = v701 + 1;
                                    v710 = v709;
                                } else {
                                    v710 = v701;
                                }
                                v701 = v710;
                                v700 += 1 ;
                            }
                            bool v711;
                            v711 = v701 == 5;
                            Union12 v714;
                            if (v711){
                                v714 = Union12{Union12_1{v698}};
                            } else {
                                v714 = Union12{Union12_0{}};
                            }
                            Union12 v742;
                            switch (v697.tag) {
                                case 0: { // None
                                    v742 = v714;
                                    break;
                                }
                                case 1: { // Some
                                    static_array<unsigned char,5> v715 = v697.case1.v0;
                                    switch (v714.tag) {
                                        case 0: { // None
                                            v742 = v697;
                                            break;
                                        }
                                        case 1: { // Some
                                            static_array<unsigned char,5> v716 = v714.case1.v0;
                                            Union11 v717;
                                            v717 = Union11{Union11_0{}};
                                            int v718; Union11 v719;
                                            Tuple22 tmp75 = Tuple22{0, v717};
                                            v718 = tmp75.v0; v719 = tmp75.v1;
                                            while (while_method_2(v718)){
                                                unsigned char v721;
                                                v721 = v715[v718];
                                                unsigned char v723;
                                                v723 = v716[v718];
                                                Union11 v735;
                                                switch (v719.tag) {
                                                    case 0: { // Eq
                                                        unsigned char v725;
                                                        v725 = v721 / 4u;
                                                        unsigned char v726;
                                                        v726 = v723 / 4u;
                                                        bool v727;
                                                        v727 = v725 < v726;
                                                        if (v727){
                                                            v735 = Union11{Union11_2{}};
                                                        } else {
                                                            bool v729;
                                                            v729 = v725 > v726;
                                                            if (v729){
                                                                v735 = Union11{Union11_1{}};
                                                            } else {
                                                                v735 = Union11{Union11_0{}};
                                                            }
                                                        }
                                                        break;
                                                    }
                                                    default: {
                                                        break;
                                                    }
                                                }
                                                v719 = v735;
                                                v718 += 1 ;
                                            }
                                            bool v736;
                                            switch (v719.tag) {
                                                case 1: { // Gt
                                                    v736 = true;
                                                    break;
                                                }
                                                default: {
                                                    v736 = false;
                                                }
                                            }
                                            static_array<unsigned char,5> v737;
                                            if (v736){
                                                v737 = v715;
                                            } else {
                                                v737 = v716;
                                            }
                                            v742 = Union12{Union12_1{v737}};
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
                            switch (v742.tag) {
                                case 0: { // None
                                    static_array<unsigned char,5> v744;
                                    int v746; int v747; unsigned char v748;
                                    Tuple21 tmp76 = Tuple21{0, 0, 12u};
                                    v746 = tmp76.v0; v747 = tmp76.v1; v748 = tmp76.v2;
                                    while (while_method_17(v746)){
                                        unsigned char v750;
                                        v750 = v84[v746];
                                        bool v752;
                                        v752 = v747 < 5;
                                        int v764; unsigned char v765;
                                        if (v752){
                                            unsigned char v753;
                                            v753 = v750 / 4u;
                                            unsigned char v754;
                                            v754 = v753 - 1u;
                                            bool v755;
                                            v755 = v748 == v754;
                                            bool v756;
                                            v756 = v755 != true;
                                            if (v756){
                                                bool v757;
                                                v757 = v748 == v753;
                                                int v758;
                                                if (v757){
                                                    v758 = v747;
                                                } else {
                                                    v758 = 0;
                                                }
                                                v744[v758] = v750;
                                                int v759;
                                                v759 = v758 + 1;
                                                v764 = v759; v765 = v754;
                                            } else {
                                                v764 = v747; v765 = v748;
                                            }
                                        } else {
                                            break;
                                        }
                                        v747 = v764;
                                        v748 = v765;
                                        v746 += 1 ;
                                    }
                                    bool v766;
                                    v766 = v747 == 4;
                                    bool v775;
                                    if (v766){
                                        unsigned char v767;
                                        v767 = v748 + 1u;
                                        bool v768;
                                        v768 = v767 == 0u;
                                        if (v768){
                                            unsigned char v769;
                                            v769 = v84[0];
                                            unsigned char v771;
                                            v771 = v769 / 4u;
                                            bool v772;
                                            v772 = v771 == 12u;
                                            if (v772){
                                                v744[4] = v769;
                                                v775 = true;
                                            } else {
                                                v775 = false;
                                            }
                                        } else {
                                            v775 = false;
                                        }
                                    } else {
                                        v775 = false;
                                    }
                                    Union12 v781;
                                    if (v775){
                                        v781 = Union12{Union12_1{v744}};
                                    } else {
                                        bool v777;
                                        v777 = v747 == 5;
                                        if (v777){
                                            v781 = Union12{Union12_1{v744}};
                                        } else {
                                            v781 = Union12{Union12_0{}};
                                        }
                                    }
                                    switch (v781.tag) {
                                        case 0: { // None
                                            static_array<unsigned char,3> v783;
                                            static_array<unsigned char,4> v785;
                                            int v787; int v788; int v789; unsigned char v790;
                                            Tuple23 tmp77 = Tuple23{0, 0, 0, 12u};
                                            v787 = tmp77.v0; v788 = tmp77.v1; v789 = tmp77.v2; v790 = tmp77.v3;
                                            while (while_method_17(v787)){
                                                unsigned char v792;
                                                v792 = v84[v787];
                                                bool v794;
                                                v794 = v789 < 3;
                                                int v802; int v803; unsigned char v804;
                                                if (v794){
                                                    unsigned char v795;
                                                    v795 = v792 / 4u;
                                                    bool v796;
                                                    v796 = v790 == v795;
                                                    int v797;
                                                    if (v796){
                                                        v797 = v789;
                                                    } else {
                                                        v797 = 0;
                                                    }
                                                    v783[v797] = v792;
                                                    int v798;
                                                    v798 = v797 + 1;
                                                    v802 = v787; v803 = v798; v804 = v795;
                                                } else {
                                                    break;
                                                }
                                                v788 = v802;
                                                v789 = v803;
                                                v790 = v804;
                                                v787 += 1 ;
                                            }
                                            bool v805;
                                            v805 = v789 == 3;
                                            Union14 v816;
                                            if (v805){
                                                int v806;
                                                v806 = 0;
                                                while (while_method_3(v806)){
                                                    int v808;
                                                    v808 = v788 + -2;
                                                    bool v809;
                                                    v809 = v806 < v808;
                                                    int v810;
                                                    if (v809){
                                                        v810 = 0;
                                                    } else {
                                                        v810 = 3;
                                                    }
                                                    int v811;
                                                    v811 = v810 + v806;
                                                    unsigned char v812;
                                                    v812 = v84[v811];
                                                    v785[v806] = v812;
                                                    v806 += 1 ;
                                                }
                                                v816 = Union14{Union14_1{v783, v785}};
                                            } else {
                                                v816 = Union14{Union14_0{}};
                                            }
                                            Union12 v839;
                                            switch (v816.tag) {
                                                case 0: { // None
                                                    v839 = Union12{Union12_0{}};
                                                    break;
                                                }
                                                case 1: { // Some
                                                    static_array<unsigned char,3> v817 = v816.case1.v0; static_array<unsigned char,4> v818 = v816.case1.v1;
                                                    static_array<unsigned char,2> v819;
                                                    int v821;
                                                    v821 = 0;
                                                    while (while_method_0(v821)){
                                                        unsigned char v823;
                                                        v823 = v818[v821];
                                                        v819[v821] = v823;
                                                        v821 += 1 ;
                                                    }
                                                    static_array<unsigned char,5> v825;
                                                    int v827;
                                                    v827 = 0;
                                                    while (while_method_1(v827)){
                                                        unsigned char v829;
                                                        v829 = v817[v827];
                                                        v825[v827] = v829;
                                                        v827 += 1 ;
                                                    }
                                                    int v831;
                                                    v831 = 0;
                                                    while (while_method_0(v831)){
                                                        unsigned char v833;
                                                        v833 = v819[v831];
                                                        int v835;
                                                        v835 = 3 + v831;
                                                        v825[v835] = v833;
                                                        v831 += 1 ;
                                                    }
                                                    v839 = Union12{Union12_1{v825}};
                                                    break;
                                                }
                                                default: {
                                                    assert("Invalid tag." && false); __trap();
                                                }
                                            }
                                            switch (v839.tag) {
                                                case 0: { // None
                                                    static_array<unsigned char,2> v841;
                                                    static_array<unsigned char,5> v843;
                                                    int v845; int v846; int v847; unsigned char v848;
                                                    Tuple23 tmp78 = Tuple23{0, 0, 0, 12u};
                                                    v845 = tmp78.v0; v846 = tmp78.v1; v847 = tmp78.v2; v848 = tmp78.v3;
                                                    while (while_method_17(v845)){
                                                        unsigned char v850;
                                                        v850 = v84[v845];
                                                        bool v852;
                                                        v852 = v847 < 2;
                                                        int v860; int v861; unsigned char v862;
                                                        if (v852){
                                                            unsigned char v853;
                                                            v853 = v850 / 4u;
                                                            bool v854;
                                                            v854 = v848 == v853;
                                                            int v855;
                                                            if (v854){
                                                                v855 = v847;
                                                            } else {
                                                                v855 = 0;
                                                            }
                                                            v841[v855] = v850;
                                                            int v856;
                                                            v856 = v855 + 1;
                                                            v860 = v845; v861 = v856; v862 = v853;
                                                        } else {
                                                            break;
                                                        }
                                                        v846 = v860;
                                                        v847 = v861;
                                                        v848 = v862;
                                                        v845 += 1 ;
                                                    }
                                                    bool v863;
                                                    v863 = v847 == 2;
                                                    Union16 v874;
                                                    if (v863){
                                                        int v864;
                                                        v864 = 0;
                                                        while (while_method_2(v864)){
                                                            int v866;
                                                            v866 = v846 + -1;
                                                            bool v867;
                                                            v867 = v864 < v866;
                                                            int v868;
                                                            if (v867){
                                                                v868 = 0;
                                                            } else {
                                                                v868 = 2;
                                                            }
                                                            int v869;
                                                            v869 = v868 + v864;
                                                            unsigned char v870;
                                                            v870 = v84[v869];
                                                            v843[v864] = v870;
                                                            v864 += 1 ;
                                                        }
                                                        v874 = Union16{Union16_1{v841, v843}};
                                                    } else {
                                                        v874 = Union16{Union16_0{}};
                                                    }
                                                    Union12 v941;
                                                    switch (v874.tag) {
                                                        case 0: { // None
                                                            v941 = Union12{Union12_0{}};
                                                            break;
                                                        }
                                                        case 1: { // Some
                                                            static_array<unsigned char,2> v875 = v874.case1.v0; static_array<unsigned char,5> v876 = v874.case1.v1;
                                                            static_array<unsigned char,2> v877;
                                                            static_array<unsigned char,3> v879;
                                                            int v881; int v882; int v883; unsigned char v884;
                                                            Tuple23 tmp79 = Tuple23{0, 0, 0, 12u};
                                                            v881 = tmp79.v0; v882 = tmp79.v1; v883 = tmp79.v2; v884 = tmp79.v3;
                                                            while (while_method_2(v881)){
                                                                unsigned char v886;
                                                                v886 = v876[v881];
                                                                bool v888;
                                                                v888 = v883 < 2;
                                                                int v896; int v897; unsigned char v898;
                                                                if (v888){
                                                                    unsigned char v889;
                                                                    v889 = v886 / 4u;
                                                                    bool v890;
                                                                    v890 = v884 == v889;
                                                                    int v891;
                                                                    if (v890){
                                                                        v891 = v883;
                                                                    } else {
                                                                        v891 = 0;
                                                                    }
                                                                    v877[v891] = v886;
                                                                    int v892;
                                                                    v892 = v891 + 1;
                                                                    v896 = v881; v897 = v892; v898 = v889;
                                                                } else {
                                                                    break;
                                                                }
                                                                v882 = v896;
                                                                v883 = v897;
                                                                v884 = v898;
                                                                v881 += 1 ;
                                                            }
                                                            bool v899;
                                                            v899 = v883 == 2;
                                                            Union17 v910;
                                                            if (v899){
                                                                int v900;
                                                                v900 = 0;
                                                                while (while_method_1(v900)){
                                                                    int v902;
                                                                    v902 = v882 + -1;
                                                                    bool v903;
                                                                    v903 = v900 < v902;
                                                                    int v904;
                                                                    if (v903){
                                                                        v904 = 0;
                                                                    } else {
                                                                        v904 = 2;
                                                                    }
                                                                    int v905;
                                                                    v905 = v904 + v900;
                                                                    unsigned char v906;
                                                                    v906 = v876[v905];
                                                                    v879[v900] = v906;
                                                                    v900 += 1 ;
                                                                }
                                                                v910 = Union17{Union17_1{v877, v879}};
                                                            } else {
                                                                v910 = Union17{Union17_0{}};
                                                            }
                                                            switch (v910.tag) {
                                                                case 0: { // None
                                                                    v941 = Union12{Union12_0{}};
                                                                    break;
                                                                }
                                                                case 1: { // Some
                                                                    static_array<unsigned char,2> v911 = v910.case1.v0; static_array<unsigned char,3> v912 = v910.case1.v1;
                                                                    static_array<unsigned char,1> v913;
                                                                    int v915;
                                                                    v915 = 0;
                                                                    while (while_method_6(v915)){
                                                                        unsigned char v917;
                                                                        v917 = v912[v915];
                                                                        v913[v915] = v917;
                                                                        v915 += 1 ;
                                                                    }
                                                                    static_array<unsigned char,5> v919;
                                                                    int v921;
                                                                    v921 = 0;
                                                                    while (while_method_0(v921)){
                                                                        unsigned char v923;
                                                                        v923 = v875[v921];
                                                                        v919[v921] = v923;
                                                                        v921 += 1 ;
                                                                    }
                                                                    int v925;
                                                                    v925 = 0;
                                                                    while (while_method_0(v925)){
                                                                        unsigned char v927;
                                                                        v927 = v911[v925];
                                                                        int v929;
                                                                        v929 = 2 + v925;
                                                                        v919[v929] = v927;
                                                                        v925 += 1 ;
                                                                    }
                                                                    int v930;
                                                                    v930 = 0;
                                                                    while (while_method_6(v930)){
                                                                        unsigned char v932;
                                                                        v932 = v913[v930];
                                                                        int v934;
                                                                        v934 = 4 + v930;
                                                                        v919[v934] = v932;
                                                                        v930 += 1 ;
                                                                    }
                                                                    v941 = Union12{Union12_1{v919}};
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
                                                    switch (v941.tag) {
                                                        case 0: { // None
                                                            static_array<unsigned char,2> v943;
                                                            static_array<unsigned char,5> v945;
                                                            int v947; int v948; int v949; unsigned char v950;
                                                            Tuple23 tmp80 = Tuple23{0, 0, 0, 12u};
                                                            v947 = tmp80.v0; v948 = tmp80.v1; v949 = tmp80.v2; v950 = tmp80.v3;
                                                            while (while_method_17(v947)){
                                                                unsigned char v952;
                                                                v952 = v84[v947];
                                                                bool v954;
                                                                v954 = v949 < 2;
                                                                int v962; int v963; unsigned char v964;
                                                                if (v954){
                                                                    unsigned char v955;
                                                                    v955 = v952 / 4u;
                                                                    bool v956;
                                                                    v956 = v950 == v955;
                                                                    int v957;
                                                                    if (v956){
                                                                        v957 = v949;
                                                                    } else {
                                                                        v957 = 0;
                                                                    }
                                                                    v943[v957] = v952;
                                                                    int v958;
                                                                    v958 = v957 + 1;
                                                                    v962 = v947; v963 = v958; v964 = v955;
                                                                } else {
                                                                    break;
                                                                }
                                                                v948 = v962;
                                                                v949 = v963;
                                                                v950 = v964;
                                                                v947 += 1 ;
                                                            }
                                                            bool v965;
                                                            v965 = v949 == 2;
                                                            Union16 v976;
                                                            if (v965){
                                                                int v966;
                                                                v966 = 0;
                                                                while (while_method_2(v966)){
                                                                    int v968;
                                                                    v968 = v948 + -1;
                                                                    bool v969;
                                                                    v969 = v966 < v968;
                                                                    int v970;
                                                                    if (v969){
                                                                        v970 = 0;
                                                                    } else {
                                                                        v970 = 2;
                                                                    }
                                                                    int v971;
                                                                    v971 = v970 + v966;
                                                                    unsigned char v972;
                                                                    v972 = v84[v971];
                                                                    v945[v966] = v972;
                                                                    v966 += 1 ;
                                                                }
                                                                v976 = Union16{Union16_1{v943, v945}};
                                                            } else {
                                                                v976 = Union16{Union16_0{}};
                                                            }
                                                            Union12 v999;
                                                            switch (v976.tag) {
                                                                case 0: { // None
                                                                    v999 = Union12{Union12_0{}};
                                                                    break;
                                                                }
                                                                case 1: { // Some
                                                                    static_array<unsigned char,2> v977 = v976.case1.v0; static_array<unsigned char,5> v978 = v976.case1.v1;
                                                                    static_array<unsigned char,3> v979;
                                                                    int v981;
                                                                    v981 = 0;
                                                                    while (while_method_1(v981)){
                                                                        unsigned char v983;
                                                                        v983 = v978[v981];
                                                                        v979[v981] = v983;
                                                                        v981 += 1 ;
                                                                    }
                                                                    static_array<unsigned char,5> v985;
                                                                    int v987;
                                                                    v987 = 0;
                                                                    while (while_method_0(v987)){
                                                                        unsigned char v989;
                                                                        v989 = v977[v987];
                                                                        v985[v987] = v989;
                                                                        v987 += 1 ;
                                                                    }
                                                                    int v991;
                                                                    v991 = 0;
                                                                    while (while_method_1(v991)){
                                                                        unsigned char v993;
                                                                        v993 = v979[v991];
                                                                        int v995;
                                                                        v995 = 2 + v991;
                                                                        v985[v995] = v993;
                                                                        v991 += 1 ;
                                                                    }
                                                                    v999 = Union12{Union12_1{v985}};
                                                                    break;
                                                                }
                                                                default: {
                                                                    assert("Invalid tag." && false); __trap();
                                                                }
                                                            }
                                                            switch (v999.tag) {
                                                                case 0: { // None
                                                                    static_array<unsigned char,5> v1001;
                                                                    int v1003;
                                                                    v1003 = 0;
                                                                    while (while_method_2(v1003)){
                                                                        unsigned char v1005;
                                                                        v1005 = v84[v1003];
                                                                        v1001[v1003] = v1005;
                                                                        v1003 += 1 ;
                                                                    }
                                                                    v1037 = v1001; v1038 = 0;
                                                                    break;
                                                                }
                                                                case 1: { // Some
                                                                    static_array<unsigned char,5> v1000 = v999.case1.v0;
                                                                    v1037 = v1000; v1038 = 1;
                                                                    break;
                                                                }
                                                                default: {
                                                                    assert("Invalid tag." && false); __trap();
                                                                }
                                                            }
                                                            break;
                                                        }
                                                        case 1: { // Some
                                                            static_array<unsigned char,5> v942 = v941.case1.v0;
                                                            v1037 = v942; v1038 = 2;
                                                            break;
                                                        }
                                                        default: {
                                                            assert("Invalid tag." && false); __trap();
                                                        }
                                                    }
                                                    break;
                                                }
                                                case 1: { // Some
                                                    static_array<unsigned char,5> v840 = v839.case1.v0;
                                                    v1037 = v840; v1038 = 3;
                                                    break;
                                                }
                                                default: {
                                                    assert("Invalid tag." && false); __trap();
                                                }
                                            }
                                            break;
                                        }
                                        case 1: { // Some
                                            static_array<unsigned char,5> v782 = v781.case1.v0;
                                            v1037 = v782; v1038 = 4;
                                            break;
                                        }
                                        default: {
                                            assert("Invalid tag." && false); __trap();
                                        }
                                    }
                                    break;
                                }
                                case 1: { // Some
                                    static_array<unsigned char,5> v743 = v742.case1.v0;
                                    v1037 = v743; v1038 = 5;
                                    break;
                                }
                                default: {
                                    assert("Invalid tag." && false); __trap();
                                }
                            }
                            break;
                        }
                        case 1: { // Some
                            static_array<unsigned char,5> v590 = v589.case1.v0;
                            v1037 = v590; v1038 = 6;
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false); __trap();
                        }
                    }
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,5> v499 = v498.case1.v0;
                    v1037 = v499; v1038 = 7;
                    break;
                }
                default: {
                    assert("Invalid tag." && false); __trap();
                }
            }
            break;
        }
        case 1: { // Some
            static_array<unsigned char,5> v441 = v440.case1.v0;
            v1037 = v441; v1038 = 8;
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    return Tuple0{v1037, v1038};
}
__device__ void play_loop_31(unsigned char * v0, unsigned long long v1, unsigned char * v2, unsigned long long v3, StackMut0 & v4, Union4 v5){
    dynamic_array_list<Union6,128> & v6 = v4.v2;
    unsigned long long & v7 = v4.v0;
    Union3 v8;
    v8 = Union3{Union3_1{v5}};
    Union3 v9;
    v9 = v8;
    while (while_method_5(v9)){
        Union3 v1621;
        switch (v9.tag) {
            case 0: { // None
                v1621 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v11 = v9.case1.v0;
                switch (v11.tag) {
                    case 0: { // G_Flop
                        int v1504 = v11.case0.v0; static_array<static_array<unsigned char,2>,2> v1505 = v11.case0.v1; static_array<int,2> v1506 = v11.case0.v2; int v1507 = v11.case0.v3; static_array<int,2> v1508 = v11.case0.v4; Union5 v1509 = v11.case0.v5;
                        curandStatePhilox4_32_10_t & v1510 = v4.v4;
                        curandStatePhilox4_32_10_t & v1511 = v1510;
                        static_array<unsigned char,3> v1512; unsigned long long v1513;
                        Tuple8 tmp18 = draw_cards_32(v1511, v7);
                        v1512 = tmp18.v0; v1513 = tmp18.v1;
                        v4.v0 = v1513;
                        static_array_list<unsigned char,5> v1514;
                        v1514 = get_community_cards_35(v1509, v1512);
                        Union6 v1515;
                        v1515 = Union6{Union6_0{v1514}};
                        v6.push(v1515);
                        Union5 v1518;
                        switch (v1509.tag) {
                            case 1: { // Preflop
                                v1518 = Union5{Union5_0{v1512}};
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in flop.");
                                __trap();
                            }
                        }
                        int v1519;
                        v1519 = 2;
                        int v1520;
                        v1520 = 0;
                        Union4 v1521;
                        v1521 = try_round_36(v1519, v1505, v1506, v1520, v1508, v1518);
                        v1621 = Union3{Union3_1{v1521}};
                        break;
                    }
                    case 1: { // G_Fold
                        int v12 = v11.case1.v0; static_array<static_array<unsigned char,2>,2> v13 = v11.case1.v1; static_array<int,2> v14 = v11.case1.v2; int v15 = v11.case1.v3; static_array<int,2> v16 = v11.case1.v4; Union5 v17 = v11.case1.v5;
                        int v18;
                        v18 = v15 % 2;
                        int v19;
                        v19 = v14[v18];
                        int v21;
                        v21 = v15 + 1;
                        int v22;
                        v22 = v21 % 2;
                        Union6 v23;
                        v23 = Union6{Union6_1{v19, v22}};
                        v6.push(v23);
                        Union7 v24;
                        v24 = Union7{Union7_1{v12, v13, v14, v15, v16, v17}};
                        v4.v5 = v24;
                        Union3 v25;
                        v25 = Union3{Union3_0{}};
                        v4.v1 = v25;
                        v1621 = Union3{Union3_0{}};
                        break;
                    }
                    case 2: { // G_Preflop
                        curandStatePhilox4_32_10_t & v1585 = v4.v4;
                        curandStatePhilox4_32_10_t & v1586 = v1585;
                        static_array<unsigned char,2> v1587; unsigned long long v1588;
                        Tuple11 tmp23 = draw_cards_39(v1586, v7);
                        v1587 = tmp23.v0; v1588 = tmp23.v1;
                        v4.v0 = v1588;
                        curandStatePhilox4_32_10_t & v1589 = v4.v4;
                        curandStatePhilox4_32_10_t & v1590 = v1589;
                        static_array<unsigned char,2> v1591; unsigned long long v1592;
                        Tuple11 tmp24 = draw_cards_39(v1590, v7);
                        v1591 = tmp24.v0; v1592 = tmp24.v1;
                        v4.v0 = v1592;
                        Union6 v1593;
                        v1593 = Union6{Union6_3{0, v1587}};
                        v6.push(v1593);
                        Union6 v1594;
                        v1594 = Union6{Union6_3{1, v1591}};
                        v6.push(v1594);
                        static_array<static_array<unsigned char,2>,2> v1595;
                        v1595[0] = v1587;
                        v1595[1] = v1591;
                        static_array<int,2> v1597;
                        v1597[0] = 2;
                        v1597[1] = 1;
                        static_array<int,2> v1599;
                        int v1601;
                        v1601 = 0;
                        while (while_method_0(v1601)){
                            int v1603;
                            v1603 = v1597[v1601];
                            int v1605;
                            v1605 = 100 - v1603;
                            v1599[v1601] = v1605;
                            v1601 += 1 ;
                        }
                        int v1606;
                        v1606 = 2;
                        int v1607;
                        v1607 = 0;
                        Union5 v1608;
                        v1608 = Union5{Union5_1{}};
                        Union4 v1609;
                        v1609 = try_round_36(v1606, v1595, v1597, v1607, v1599, v1608);
                        v1621 = Union3{Union3_1{v1609}};
                        break;
                    }
                    case 3: { // G_River
                        int v1554 = v11.case3.v0; static_array<static_array<unsigned char,2>,2> v1555 = v11.case3.v1; static_array<int,2> v1556 = v11.case3.v2; int v1557 = v11.case3.v3; static_array<int,2> v1558 = v11.case3.v4; Union5 v1559 = v11.case3.v5;
                        curandStatePhilox4_32_10_t & v1560 = v4.v4;
                        curandStatePhilox4_32_10_t & v1561 = v1560;
                        static_array<unsigned char,1> v1562; unsigned long long v1563;
                        Tuple12 tmp27 = draw_cards_40(v1561, v7);
                        v1562 = tmp27.v0; v1563 = tmp27.v1;
                        v4.v0 = v1563;
                        static_array_list<unsigned char,5> v1564;
                        v1564 = get_community_cards_41(v1559, v1562);
                        Union6 v1565;
                        v1565 = Union6{Union6_0{v1564}};
                        v6.push(v1565);
                        Union5 v1580;
                        switch (v1559.tag) {
                            case 3: { // Turn
                                static_array<unsigned char,4> v1566 = v1559.case3.v0;
                                static_array<unsigned char,5> v1567;
                                int v1569;
                                v1569 = 0;
                                while (while_method_3(v1569)){
                                    unsigned char v1571;
                                    v1571 = v1566[v1569];
                                    v1567[v1569] = v1571;
                                    v1569 += 1 ;
                                }
                                int v1573;
                                v1573 = 0;
                                while (while_method_6(v1573)){
                                    unsigned char v1575;
                                    v1575 = v1562[v1573];
                                    int v1577;
                                    v1577 = 4 + v1573;
                                    v1567[v1577] = v1575;
                                    v1573 += 1 ;
                                }
                                v1580 = Union5{Union5_2{v1567}};
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in river.");
                                __trap();
                            }
                        }
                        int v1581;
                        v1581 = 2;
                        int v1582;
                        v1582 = 0;
                        Union4 v1583;
                        v1583 = try_round_36(v1581, v1555, v1556, v1582, v1558, v1580);
                        v1621 = Union3{Union3_1{v1583}};
                        break;
                    }
                    case 4: { // G_Round
                        int v107 = v11.case4.v0; static_array<static_array<unsigned char,2>,2> v108 = v11.case4.v1; static_array<int,2> v109 = v11.case4.v2; int v110 = v11.case4.v3; static_array<int,2> v111 = v11.case4.v4; Union5 v112 = v11.case4.v5;
                        int v113;
                        v113 = v110 % 2;
                        static_array<Union2,2> & v114 = v4.v3;
                        Union2 v115;
                        v115 = v114[v113];
                        switch (v115.tag) {
                            case 0: { // Computer
                                bool v117;
                                v117 = 12419088ull == v3;
                                bool v118;
                                v118 = v117 == false;
                                if (v118){
                                    assert("The params needs to have matching offsets." && v117);
                                } else {
                                }
                                bool v120;
                                v120 = 204570624ull == v1;
                                bool v121;
                                v121 = v120 == false;
                                if (v121){
                                    assert("The outputs needs to have matching offsets." && v120);
                                } else {
                                }
                                dynamic_array_list<Union6,128> & v123 = v4.v2;
                                curandStatePhilox4_32_10_t & v124 = v4.v4;
                                curandStatePhilox4_32_10_t & v125 = v124;
                                unsigned int * v126;
                                v126 = reinterpret_cast<unsigned int *>(&v0[53477376ull]);
                                float * v128;
                                v128 = reinterpret_cast<float *>(&v0[0ull]);
                                int v130;
                                v130 = threadIdx.x;
                                int v131;
                                v131 = blockIdx.x;
                                int v132;
                                v132 = v131 * 256;
                                int v133;
                                v133 = v130 + v132;
                                unsigned long long v134;
                                v134 = (unsigned long long)v133;
                                curandStatePhilox4_32_10_t v135;
                                curand_init(12344321ull,v134,0ull,&v135);
                                float * v136;
                                v136 = reinterpret_cast<float *>(&v0[0ull]);
                                int v138;
                                v138 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v138 && v138 < 24);
                                int v139;
                                v139 = 524288 * v138;
                                int v140;
                                v140 = threadIdx.x;
                                int v141;
                                v141 = blockIdx.x;
                                int v142;
                                v142 = v141 * 256;
                                int v143;
                                v143 = v140 + v142;
                                unsigned long long v144;
                                v144 = (unsigned long long)v143;
                                curandStatePhilox4_32_10_t v145;
                                curand_init(12344321ull,v144,0ull,&v145);
                                int v146;
                                v146 = threadIdx.x;
                                int v147;
                                v147 = v146;
                                while (while_method_7(v147)){
                                    bool v149;
                                    v149 = 0 <= v147;
                                    bool v150;
                                    v150 = v149 == false;
                                    if (v150){
                                        assert("The index needs to be zero or positive." && v149);
                                    } else {
                                    }
                                    int v152;
                                    v152 = v147 % 2048;
                                    int v153;
                                    v153 = v147 / 2048;
                                    bool v154;
                                    v154 = v153 < 256;
                                    bool v155;
                                    v155 = v154 == false;
                                    if (v155){
                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v154);
                                    } else {
                                    }
                                    assert("Tensor range check" && 0 <= v153 && v153 < 256);
                                    assert("Tensor range check" && 0 <= v152 && v152 < 2048);
                                    int v157;
                                    v157 = v152 + v139;
                                    int v158;
                                    v158 = 2048 * v153;
                                    int v159;
                                    v159 = v158 + v157;
                                    v136[v159] = 0.0f;
                                    v147 += 256 ;
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                int v160;
                                v160 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v160 && v160 < 256);
                                int v161;
                                v161 = 2048 * v160;
                                int v162;
                                v162 = v161 + v139;
                                int v163;
                                v163 = v123.length_();
                                bool v164;
                                v164 = 128 >= v163;
                                bool v165;
                                v165 = v164 == false;
                                if (v165){
                                    assert("The type level dimension has to equal the value passed at runtime into create." && v164);
                                } else {
                                }
                                dynamic_array_list<Union8,128> v167{0};
                                v167.unsafe_set_length(v163);
                                int v169;
                                v169 = 0;
                                while (while_method_4(v163, v169)){
                                    Union6 v171;
                                    v171 = v123[v169];
                                    Union8 v177;
                                    switch (v171.tag) {
                                        case 2: { // PlayerAction
                                            int v173 = v171.case2.v0; Union1 v174 = v171.case2.v1;
                                            v177 = Union8{Union8_1{v174}};
                                            break;
                                        }
                                        default: {
                                            v177 = Union8{Union8_0{}};
                                        }
                                    }
                                    v167[v169] = v177;
                                    v169 += 1 ;
                                }
                                static_array<int,2> v178;
                                int v180;
                                v180 = 0;
                                while (while_method_0(v180)){
                                    int v182;
                                    v182 = v180 + v113;
                                    int v183;
                                    v183 = v109[v182];
                                    v178[v180] = v183;
                                    v180 += 1 ;
                                }
                                static_array<int,2> v185;
                                int v187;
                                v187 = 0;
                                while (while_method_0(v187)){
                                    int v189;
                                    v189 = v187 + v113;
                                    int v190;
                                    v190 = v111[v189];
                                    v185[v187] = v190;
                                    v187 += 1 ;
                                }
                                static_array<unsigned char,2> v192;
                                v192 = v108[v113];
                                static_array_list<unsigned char,5> v194;
                                v194 = static_array_list<unsigned char,5>{};
                                switch (v112.tag) {
                                    case 0: { // Flop
                                        static_array<unsigned char,3> v196 = v112.case0.v0;
                                        int v197;
                                        v197 = 0;
                                        while (while_method_1(v197)){
                                            unsigned char v199;
                                            v199 = v196[v197];
                                            v194.push(v199);
                                            v197 += 1 ;
                                        }
                                        break;
                                    }
                                    case 1: { // Preflop
                                        break;
                                    }
                                    case 2: { // River
                                        static_array<unsigned char,5> v206 = v112.case2.v0;
                                        int v207;
                                        v207 = 0;
                                        while (while_method_2(v207)){
                                            unsigned char v209;
                                            v209 = v206[v207];
                                            v194.push(v209);
                                            v207 += 1 ;
                                        }
                                        break;
                                    }
                                    case 3: { // Turn
                                        static_array<unsigned char,4> v201 = v112.case3.v0;
                                        int v202;
                                        v202 = 0;
                                        while (while_method_3(v202)){
                                            unsigned char v204;
                                            v204 = v201[v202];
                                            v194.push(v204);
                                            v202 += 1 ;
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                float * v211;
                                v211 = v136+v162;
                                int v213;
                                v213 = v167.length_();
                                bool v214;
                                v214 = v213 == 0;
                                if (v214){
                                    v211[0] = 1.0f;
                                } else {
                                }
                                int v215;
                                v215 = v167.length_();
                                int v216;
                                v216 = 0;
                                while (while_method_4(v215, v216)){
                                    Union8 v218;
                                    v218 = v167[v216];
                                    int v220;
                                    v220 = v216 * 14;
                                    int v221;
                                    v221 = 1 + v220;
                                    switch (v218.tag) {
                                        case 0: { // None
                                            v211[v221] = 1.0f;
                                            break;
                                        }
                                        case 1: { // Some
                                            Union1 v222 = v218.case1.v0;
                                            int v223;
                                            v223 = v221 + 1;
                                            switch (v222.tag) {
                                                case 0: { // A_All_In
                                                    v211[v223] = 1.0f;
                                                    break;
                                                }
                                                case 1: { // A_Call
                                                    int v224;
                                                    v224 = v223 + 1;
                                                    v211[v224] = 1.0f;
                                                    break;
                                                }
                                                case 2: { // A_Fold
                                                    int v225;
                                                    v225 = v223 + 2;
                                                    v211[v225] = 1.0f;
                                                    break;
                                                }
                                                case 3: { // A_Raise
                                                    int v226 = v222.case3.v0;
                                                    int v227;
                                                    v227 = v223 + 3;
                                                    bool v228;
                                                    v228 = 1 <= v226;
                                                    bool v230;
                                                    if (v228){
                                                        bool v229;
                                                        v229 = v226 < 1023;
                                                        v230 = v229;
                                                    } else {
                                                        v230 = false;
                                                    }
                                                    bool v231;
                                                    v231 = v230 == false;
                                                    if (v231){
                                                        assert("Pickle failure. The input is out of the bounds of the given range." && v230);
                                                    } else {
                                                    }
                                                    int v233;
                                                    v233 = v226 - 1;
                                                    unsigned int v234;
                                                    v234 = (unsigned int)v233;
                                                    method_42(v234, v211, v227);
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
                                    v216 += 1 ;
                                }
                                int v235;
                                v235 = 0;
                                while (while_method_0(v235)){
                                    int v237;
                                    v237 = v178[v235];
                                    int v239;
                                    v239 = v235 * 11;
                                    int v240;
                                    v240 = 1794 + v239;
                                    bool v241;
                                    v241 = 0 <= v237;
                                    bool v243;
                                    if (v241){
                                        bool v242;
                                        v242 = v237 < 1023;
                                        v243 = v242;
                                    } else {
                                        v243 = false;
                                    }
                                    bool v244;
                                    v244 = v243 == false;
                                    if (v244){
                                        assert("Pickle failure. The input is out of the bounds of the given range." && v243);
                                    } else {
                                    }
                                    unsigned int v246;
                                    v246 = (unsigned int)v237;
                                    method_43(v246, v211, v240);
                                    v235 += 1 ;
                                }
                                int v247;
                                v247 = 0;
                                while (while_method_0(v247)){
                                    int v249;
                                    v249 = v185[v247];
                                    int v251;
                                    v251 = v247 * 11;
                                    int v252;
                                    v252 = 1817 + v251;
                                    bool v253;
                                    v253 = 0 <= v249;
                                    bool v255;
                                    if (v253){
                                        bool v254;
                                        v254 = v249 < 1023;
                                        v255 = v254;
                                    } else {
                                        v255 = false;
                                    }
                                    bool v256;
                                    v256 = v255 == false;
                                    if (v256){
                                        assert("Pickle failure. The input is out of the bounds of the given range." && v255);
                                    } else {
                                    }
                                    unsigned int v258;
                                    v258 = (unsigned int)v249;
                                    method_43(v258, v211, v252);
                                    v247 += 1 ;
                                }
                                int v259;
                                v259 = 0;
                                while (while_method_0(v259)){
                                    unsigned char v261;
                                    v261 = v192[v259];
                                    int v263;
                                    v263 = v259 * 17;
                                    int v264;
                                    v264 = 1840 + v263;
                                    unsigned char v265;
                                    v265 = v261 % 4u;
                                    int v266;
                                    v266 = (int)v265;
                                    unsigned char v267;
                                    v267 = v261 / 4u;
                                    int v268;
                                    v268 = (int)v267;
                                    unsigned int v269;
                                    v269 = (unsigned int)v266;
                                    int v270;
                                    v270 = (int)v269;
                                    bool v271;
                                    v271 = v270 < 4;
                                    bool v272;
                                    v272 = v271 == false;
                                    if (v272){
                                        assert("Pickle failure. Int value out of bounds." && v271);
                                    } else {
                                    }
                                    int v274;
                                    v274 = v264 + v270;
                                    v211[v274] = 1.0f;
                                    int v275;
                                    v275 = v264 + 4;
                                    unsigned int v276;
                                    v276 = (unsigned int)v268;
                                    int v277;
                                    v277 = (int)v276;
                                    bool v278;
                                    v278 = v277 < 13;
                                    bool v279;
                                    v279 = v278 == false;
                                    if (v279){
                                        assert("Pickle failure. Int value out of bounds." && v278);
                                    } else {
                                    }
                                    int v281;
                                    v281 = v275 + v277;
                                    v211[v281] = 1.0f;
                                    v259 += 1 ;
                                }
                                int v282;
                                v282 = v194.length;
                                bool v283;
                                v283 = v282 == 0;
                                if (v283){
                                    v211[1874] = 1.0f;
                                } else {
                                }
                                int v284;
                                v284 = v194.length;
                                int v285;
                                v285 = 0;
                                while (while_method_4(v284, v285)){
                                    unsigned char v287;
                                    v287 = v194[v285];
                                    int v289;
                                    v289 = v285 * 17;
                                    int v290;
                                    v290 = 1875 + v289;
                                    unsigned char v291;
                                    v291 = v287 % 4u;
                                    int v292;
                                    v292 = (int)v291;
                                    unsigned char v293;
                                    v293 = v287 / 4u;
                                    int v294;
                                    v294 = (int)v293;
                                    unsigned int v295;
                                    v295 = (unsigned int)v292;
                                    int v296;
                                    v296 = (int)v295;
                                    bool v297;
                                    v297 = v296 < 4;
                                    bool v298;
                                    v298 = v297 == false;
                                    if (v298){
                                        assert("Pickle failure. Int value out of bounds." && v297);
                                    } else {
                                    }
                                    int v300;
                                    v300 = v290 + v296;
                                    v211[v300] = 1.0f;
                                    int v301;
                                    v301 = v290 + 4;
                                    unsigned int v302;
                                    v302 = (unsigned int)v294;
                                    int v303;
                                    v303 = (int)v302;
                                    bool v304;
                                    v304 = v303 < 13;
                                    bool v305;
                                    v305 = v304 == false;
                                    if (v305){
                                        assert("Pickle failure. Int value out of bounds." && v304);
                                    } else {
                                    }
                                    int v307;
                                    v307 = v301 + v303;
                                    v211[v307] = 1.0f;
                                    v285 += 1 ;
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                int v308;
                                v308 = 0;
                                int v309;
                                v309 = 4;
                                int v310;
                                v310 = int_range_44(v309, v308, v145);
                                extern __shared__ unsigned char v311[];
                                int * v312;
                                v312 = reinterpret_cast<int *>(&v311[0ull]);
                                int v314;
                                v314 = threadIdx.x;
                                bool v315;
                                v315 = v314 == 0;
                                if (v315){
                                    v312[0] = v310;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                int v316;
                                v316 = v312[0];
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                float * v317;
                                v317 = reinterpret_cast<float *>(&v0[0ull]);
                                float * v319;
                                v319 = reinterpret_cast<float *>(&v2[0ull]);
                                assert("Tensor range check" && 0 <= v316 && v316 < 4);
                                int v321;
                                v321 = 262144 * v316;
                                float * v322;
                                v322 = reinterpret_cast<float *>(&v0[50331648ull]);
                                int v324;
                                v324 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v324 && v324 < 24);
                                int v325;
                                v325 = 524288 * v324;
                                int v326;
                                v326 = blockIdx.x;
                                assert("Tensor range check" && 0 <= v326 && v326 < 24);
                                int v327;
                                v327 = 32768 * v326;
                                cuda::pipeline<cuda::thread_scope_thread> v328 = cuda::make_pipeline();
                                extern __shared__ unsigned char v329[];
                                float * v330;
                                v330 = reinterpret_cast<float *>(&v329[0ull]);
                                float * v332;
                                v332 = reinterpret_cast<float *>(&v329[34816ull]);
                                float * v334;
                                v334 = reinterpret_cast<float *>(&v329[0ull]);
                                int v336;
                                v336 = threadIdx.x;
                                int v337;
                                v337 = v336 / 32;
                                bool v338;
                                v338 = 0 <= v337;
                                bool v339;
                                v339 = v338 == false;
                                if (v339){
                                    assert("The index needs to be zero or positive." && v338);
                                } else {
                                }
                                int v341;
                                v341 = v337 % 8;
                                int v342;
                                v342 = v337 / 8;
                                bool v343;
                                v343 = v342 < 1;
                                bool v344;
                                v344 = v343 == false;
                                if (v344){
                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v343);
                                } else {
                                }
                                assert("Tensor range check" && 0 <= v342 && v342 < 1);
                                assert("Tensor range check" && 0 <= v341 && v341 < 8);
                                int v346;
                                v346 = 16 * v341;
                                int v347;
                                v347 = 17408 * v342;
                                int v348;
                                v348 = v347 + v346;
                                float * v349;
                                v349 = v334+v348;
                                assert("Tensor range check" && 0 <= v342 && v342 < 1);
                                int v351;
                                v351 = 8704 * v342;
                                int v352;
                                v352 = threadIdx.x;
                                int v353;
                                v353 = v352 % 32;
                                bool v354;
                                v354 = 0 <= v353;
                                bool v355;
                                v355 = v354 == false;
                                if (v355){
                                    assert("The index needs to be zero or positive." && v354);
                                } else {
                                }
                                int v357;
                                v357 = v353 % 4;
                                int v358;
                                v358 = v353 / 4;
                                bool v359;
                                v359 = v358 < 8;
                                bool v360;
                                v360 = v359 == false;
                                if (v360){
                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v359);
                                } else {
                                }
                                assert("Tensor range check" && 0 <= v358 && v358 < 8);
                                assert("Tensor range check" && 0 <= v357 && v357 < 4);
                                int v362;
                                v362 = v357 + v351;
                                int v363;
                                v363 = 68 * v358;
                                int v364;
                                v364 = v363 + v362;
                                float * v365;
                                v365 = v330+v364;
                                assert("Tensor range check" && 0 <= v341 && v341 < 8);
                                int v367;
                                v367 = 1088 * v341;
                                int v368;
                                v368 = threadIdx.x;
                                int v369;
                                v369 = v368 % 32;
                                bool v370;
                                v370 = 0 <= v369;
                                bool v371;
                                v371 = v370 == false;
                                if (v371){
                                    assert("The index needs to be zero or positive." && v370);
                                } else {
                                }
                                int v373;
                                v373 = v369 % 4;
                                int v374;
                                v374 = v369 / 4;
                                bool v375;
                                v375 = v374 < 8;
                                bool v376;
                                v376 = v375 == false;
                                if (v376){
                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v375);
                                } else {
                                }
                                assert("Tensor range check" && 0 <= v374 && v374 < 8);
                                assert("Tensor range check" && 0 <= v373 && v373 < 4);
                                int v378;
                                v378 = v373 + v367;
                                int v379;
                                v379 = 68 * v374;
                                int v380;
                                v380 = v379 + v378;
                                float * v381;
                                v381 = v332+v380;
                                wmma::fragment<wmma::accumulator, 16, 16, 8, float> v383[8];
                                int v384;
                                v384 = 0;
                                while (while_method_0(v384)){
                                    int v386;
                                    v386 = 0;
                                    while (while_method_6(v386)){
                                        assert("Tensor range check" && 0 <= v384 && v384 < 2);
                                        assert("Tensor range check" && 0 <= v386 && v386 < 1);
                                        int v388;
                                        v388 = 128 * v386;
                                        int v389;
                                        v389 = v388 + v327;
                                        int v390;
                                        v390 = 16384 * v384;
                                        int v391;
                                        v391 = v390 + v389;
                                        float * v392;
                                        v392 = v322+v391;
                                        // Pushing the loop unrolling to: 0
                                        int v394;
                                        v394 = 0;
                                        #pragma unroll
                                        while (while_method_10(v394)){
                                            int v396;
                                            v396 = 0;
                                            #pragma unroll
                                            while (while_method_6(v396)){
                                                assert("Tensor range check" && 0 <= v394 && v394 < 8);
                                                assert("Tensor range check" && 0 <= v396 && v396 < 1);
                                                int v398;
                                                v398 = v394 + v396;
                                                wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v399 = v383[v398];
                                                wmma::fill_fragment(v399, 0.0f);
                                                v396 += 1 ;
                                            }
                                            v394 += 1 ;
                                        }
                                        // Poping the loop unrolling to: 0
                                        int v400;
                                        v400 = 0;
                                        while (while_method_11(v400)){
                                            int v402;
                                            v402 = v400 + 1;
                                            bool v403;
                                            v403 = v400 == 0;
                                            int v404;
                                            v404 = v400 % 2;
                                            bool v405;
                                            v405 = 0 <= v400;
                                            bool v406;
                                            v406 = v405 == false;
                                            if (v406){
                                                assert("The index needs to be zero or positive." && v405);
                                            } else {
                                            }
                                            bool v408;
                                            v408 = v400 < 32;
                                            bool v409;
                                            v409 = v408 == false;
                                            if (v409){
                                                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v408);
                                            } else {
                                            }
                                            bool v411;
                                            v411 = v402 < 32;
                                            Union9 v417;
                                            if (v411){
                                                bool v412;
                                                v412 = 0 <= v402;
                                                bool v413;
                                                v413 = v412 == false;
                                                if (v413){
                                                    assert("The index needs to be zero or positive." && v412);
                                                } else {
                                                }
                                                v417 = Union9{Union9_1{v402}};
                                            } else {
                                                v417 = Union9{Union9_0{}};
                                            }
                                            assert("Tensor range check" && 0 <= v384 && v384 < 2);
                                            int v418;
                                            v418 = 262144 * v384;
                                            int v419;
                                            v419 = v418 + v325;
                                            assert("Tensor range check" && 0 <= v400 && v400 < 32);
                                            int v420;
                                            v420 = 64 * v400;
                                            int v421;
                                            v421 = v420 + v419;
                                            float * v422;
                                            v422 = v317+v421;
                                            assert("Tensor range check" && 0 <= v386 && v386 < 1);
                                            int v424;
                                            v424 = 262144 * v386;
                                            int v425;
                                            v425 = v424 + v321;
                                            if (v403){
                                                assert("Tensor range check" && 0 <= v400 && v400 < 32);
                                                int v426;
                                                v426 = v420 + v425;
                                                float * v427;
                                                v427 = v319+v426;
                                                // Pushing the loop unrolling to: 0
                                                v328.producer_acquire();
                                                int v429;
                                                v429 = threadIdx.x;
                                                bool v430;
                                                v430 = 0 <= v429;
                                                bool v431;
                                                v431 = v430 == false;
                                                if (v431){
                                                    assert("The index needs to be zero or positive." && v430);
                                                } else {
                                                }
                                                int v433;
                                                v433 = v429 % 16;
                                                int v434;
                                                v434 = v429 / 16;
                                                bool v435;
                                                v435 = v434 < 16;
                                                bool v436;
                                                v436 = v435 == false;
                                                if (v436){
                                                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v435);
                                                } else {
                                                }
                                                assert("Tensor range check" && 0 <= v434 && v434 < 16);
                                                assert("Tensor range check" && 0 <= v433 && v433 < 16);
                                                int v438;
                                                v438 = 4 * v433;
                                                int v439;
                                                v439 = 68 * v434;
                                                int v440;
                                                v440 = v439 + v438;
                                                int v441;
                                                v441 = 2048 * v434;
                                                int v442;
                                                v442 = v441 + v438;
                                                float * v443;
                                                v443 = v332+v440;
                                                float * v445;
                                                v445 = v427+v442;
                                                int v447;
                                                v447 = 0;
                                                #pragma unroll
                                                while (while_method_10(v447)){
                                                    int v449;
                                                    v449 = 0;
                                                    #pragma unroll
                                                    while (while_method_6(v449)){
                                                        assert("Tensor range check" && 0 <= v447 && v447 < 8);
                                                        assert("Tensor range check" && 0 <= v449 && v449 < 1);
                                                        int v451;
                                                        v451 = 64 * v449;
                                                        int v452;
                                                        v452 = 1088 * v447;
                                                        int v453;
                                                        v453 = v452 + v451;
                                                        int v454;
                                                        v454 = 32768 * v447;
                                                        int v455;
                                                        v455 = v454 + v451;
                                                        constexpr int v456 = sizeof(float) * 4;
                                                        assert("Pointer alignment check" && (unsigned long long)(v445 + v455) % v456 == 0 && (unsigned long long)(v443 + v453) % v456 == 0);
                                                        cuda::memcpy_async(v443 + v453, v445 + v455, cuda::aligned_size_t<v456>(v456), v328);
                                                        v449 += 1 ;
                                                    }
                                                    v447 += 1 ;
                                                }
                                                v328.producer_commit();
                                                // Poping the loop unrolling to: 0
                                            } else {
                                            }
                                            // Pushing the loop unrolling to: 0
                                            int v457;
                                            v457 = threadIdx.x;
                                            bool v458;
                                            v458 = 0 <= v457;
                                            bool v459;
                                            v459 = v458 == false;
                                            if (v459){
                                                assert("The index needs to be zero or positive." && v458);
                                            } else {
                                            }
                                            int v461;
                                            v461 = v457 % 16;
                                            int v462;
                                            v462 = v457 / 16;
                                            bool v463;
                                            v463 = v462 < 16;
                                            bool v464;
                                            v464 = v463 == false;
                                            if (v464){
                                                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v463);
                                            } else {
                                            }
                                            assert("Tensor range check" && 0 <= v462 && v462 < 16);
                                            assert("Tensor range check" && 0 <= v461 && v461 < 16);
                                            int v466;
                                            v466 = 4 * v461;
                                            int v467;
                                            v467 = 68 * v462;
                                            int v468;
                                            v468 = v467 + v466;
                                            int v469;
                                            v469 = 2048 * v462;
                                            int v470;
                                            v470 = v469 + v466;
                                            float * v471;
                                            v471 = v330+v468;
                                            float * v473;
                                            v473 = v422+v470;
                                            int v475;
                                            v475 = 0;
                                            #pragma unroll
                                            while (while_method_10(v475)){
                                                int v477;
                                                v477 = 0;
                                                #pragma unroll
                                                while (while_method_6(v477)){
                                                    assert("Tensor range check" && 0 <= v475 && v475 < 8);
                                                    assert("Tensor range check" && 0 <= v477 && v477 < 1);
                                                    int v479;
                                                    v479 = 64 * v477;
                                                    int v480;
                                                    v480 = 1088 * v475;
                                                    int v481;
                                                    v481 = v480 + v479;
                                                    int v482;
                                                    v482 = 32768 * v475;
                                                    int v483;
                                                    v483 = v482 + v479;
                                                    int4* v484;
                                                    v484 = reinterpret_cast<int4*>(v473 + v483);
                                                    int4* v485;
                                                    v485 = reinterpret_cast<int4*>(v471 + v481);
                                                    assert("Pointer alignment check" && (unsigned long long)(v484) % 4 == 0 && (unsigned long long)(v485) % 4 == 0);
                                                    *v485 = *v484;
                                                    v477 += 1 ;
                                                }
                                                v475 += 1 ;
                                            }
                                            // Poping the loop unrolling to: 0
                                            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v486[1];
                                            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> v487[8];
                                            cuda::pipeline_consumer_wait_prior<0>(v328);;
                                            asm("barrier.cta.sync %0;" :: "r"(0));
                                            // Pushing the loop unrolling to: 0
                                            int v488;
                                            v488 = 0;
                                            #pragma unroll
                                            while (while_method_6(v488)){
                                                int v490;
                                                v490 = 0;
                                                #pragma unroll
                                                while (while_method_10(v490)){
                                                    assert("Tensor range check" && 0 <= v488 && v488 < 1);
                                                    assert("Tensor range check" && 0 <= v490 && v490 < 8);
                                                    int v492;
                                                    v492 = 8 * v488;
                                                    int v493;
                                                    v493 = v492 + v490;
                                                    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v494 = v487[v493];
                                                    assert("Tensor range check" && 0 <= v488 && v488 < 1);
                                                    int v495;
                                                    v495 = 1088 * v488;
                                                    assert("Tensor range check" && 0 <= v490 && v490 < 8);
                                                    int v496;
                                                    v496 = 8 * v490;
                                                    int v497;
                                                    v497 = v496 + v495;
                                                    int v498;
                                                    v498 = 0;
                                                    #pragma unroll
                                                    while (while_method_0(v498)){
                                                        int v500;
                                                        v500 = 0;
                                                        #pragma unroll
                                                        while (while_method_0(v500)){
                                                            assert("Tensor range check" && 0 <= v498 && v498 < 2);
                                                            assert("Tensor range check" && 0 <= v500 && v500 < 2);
                                                            int v502;
                                                            v502 = 4 * v500;
                                                            int v503;
                                                            v503 = v502 + v497;
                                                            int v504;
                                                            v504 = 544 * v498;
                                                            int v505;
                                                            v505 = v504 + v503;
                                                            float v506;
                                                            v506 = v381[v505];
                                                            bool v507;
                                                            v507 = 0 <= v500;
                                                            bool v509;
                                                            if (v507){
                                                                bool v508;
                                                                v508 = v500 < 2;
                                                                v509 = v508;
                                                            } else {
                                                                v509 = false;
                                                            }
                                                            bool v510;
                                                            v510 = v509 == false;
                                                            if (v510){
                                                                assert("The indices should be inside the range of the dimension." && v509);
                                                            } else {
                                                            }
                                                            bool v512;
                                                            v512 = 0 <= v498;
                                                            bool v514;
                                                            if (v512){
                                                                bool v513;
                                                                v513 = v498 < 2;
                                                                v514 = v513;
                                                            } else {
                                                                v514 = false;
                                                            }
                                                            bool v515;
                                                            v515 = v514 == false;
                                                            if (v515){
                                                                assert("The indices should be inside the range of the dimension." && v514);
                                                            } else {
                                                            }
                                                            int v517;
                                                            v517 = v498 * 2;
                                                            int v518;
                                                            v518 = v500 + v517;
                                                            v494.x[v518] = wmma::__float_to_tf32(v506);
                                                            v500 += 1 ;
                                                        }
                                                        v498 += 1 ;
                                                    }
                                                    v490 += 1 ;
                                                }
                                                v488 += 1 ;
                                            }
                                            // Poping the loop unrolling to: 0
                                            v328.consumer_release();
                                            switch (v417.tag) {
                                                case 0: { // None
                                                    break;
                                                }
                                                case 1: { // Some
                                                    int v519 = v417.case1.v0;
                                                    assert("Tensor range check" && 0 <= v519 && v519 < 32);
                                                    int v520;
                                                    v520 = 64 * v519;
                                                    int v521;
                                                    v521 = v520 + v425;
                                                    float * v522;
                                                    v522 = v319+v521;
                                                    asm("barrier.cta.sync %0;" :: "r"(0));
                                                    // Pushing the loop unrolling to: 0
                                                    v328.producer_acquire();
                                                    int v524;
                                                    v524 = threadIdx.x;
                                                    bool v525;
                                                    v525 = 0 <= v524;
                                                    bool v526;
                                                    v526 = v525 == false;
                                                    if (v526){
                                                        assert("The index needs to be zero or positive." && v525);
                                                    } else {
                                                    }
                                                    int v528;
                                                    v528 = v524 % 16;
                                                    int v529;
                                                    v529 = v524 / 16;
                                                    bool v530;
                                                    v530 = v529 < 16;
                                                    bool v531;
                                                    v531 = v530 == false;
                                                    if (v531){
                                                        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v530);
                                                    } else {
                                                    }
                                                    assert("Tensor range check" && 0 <= v529 && v529 < 16);
                                                    assert("Tensor range check" && 0 <= v528 && v528 < 16);
                                                    int v533;
                                                    v533 = 4 * v528;
                                                    int v534;
                                                    v534 = 68 * v529;
                                                    int v535;
                                                    v535 = v534 + v533;
                                                    int v536;
                                                    v536 = 2048 * v529;
                                                    int v537;
                                                    v537 = v536 + v533;
                                                    float * v538;
                                                    v538 = v332+v535;
                                                    float * v540;
                                                    v540 = v522+v537;
                                                    int v542;
                                                    v542 = 0;
                                                    #pragma unroll
                                                    while (while_method_10(v542)){
                                                        int v544;
                                                        v544 = 0;
                                                        #pragma unroll
                                                        while (while_method_6(v544)){
                                                            assert("Tensor range check" && 0 <= v542 && v542 < 8);
                                                            assert("Tensor range check" && 0 <= v544 && v544 < 1);
                                                            int v546;
                                                            v546 = 64 * v544;
                                                            int v547;
                                                            v547 = 1088 * v542;
                                                            int v548;
                                                            v548 = v547 + v546;
                                                            int v549;
                                                            v549 = 32768 * v542;
                                                            int v550;
                                                            v550 = v549 + v546;
                                                            constexpr int v551 = sizeof(float) * 4;
                                                            assert("Pointer alignment check" && (unsigned long long)(v540 + v550) % v551 == 0 && (unsigned long long)(v538 + v548) % v551 == 0);
                                                            cuda::memcpy_async(v538 + v548, v540 + v550, cuda::aligned_size_t<v551>(v551), v328);
                                                            v544 += 1 ;
                                                        }
                                                        v542 += 1 ;
                                                    }
                                                    v328.producer_commit();
                                                    // Poping the loop unrolling to: 0
                                                    break;
                                                }
                                                default: {
                                                    assert("Invalid tag." && false); __trap();
                                                }
                                            }
                                            // Pushing the loop unrolling to: 0
                                            int v552;
                                            v552 = 0;
                                            #pragma unroll
                                            while (while_method_10(v552)){
                                                int v554;
                                                v554 = 0;
                                                #pragma unroll
                                                while (while_method_10(v554)){
                                                    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> & v556 = v486[0];
                                                    assert("Tensor range check" && 0 <= v552 && v552 < 8);
                                                    int v557;
                                                    v557 = 1088 * v552;
                                                    assert("Tensor range check" && 0 <= v554 && v554 < 8);
                                                    int v558;
                                                    v558 = 8 * v554;
                                                    int v559;
                                                    v559 = v558 + v557;
                                                    int v560;
                                                    v560 = 0;
                                                    #pragma unroll
                                                    while (while_method_0(v560)){
                                                        int v562;
                                                        v562 = 0;
                                                        #pragma unroll
                                                        while (while_method_0(v562)){
                                                            assert("Tensor range check" && 0 <= v560 && v560 < 2);
                                                            assert("Tensor range check" && 0 <= v562 && v562 < 2);
                                                            int v564;
                                                            v564 = 544 * v562;
                                                            int v565;
                                                            v565 = v564 + v559;
                                                            int v566;
                                                            v566 = 4 * v560;
                                                            int v567;
                                                            v567 = v566 + v565;
                                                            float v568;
                                                            v568 = v365[v567];
                                                            bool v569;
                                                            v569 = 0 <= v562;
                                                            bool v571;
                                                            if (v569){
                                                                bool v570;
                                                                v570 = v562 < 2;
                                                                v571 = v570;
                                                            } else {
                                                                v571 = false;
                                                            }
                                                            bool v572;
                                                            v572 = v571 == false;
                                                            if (v572){
                                                                assert("The indices should be inside the range of the dimension." && v571);
                                                            } else {
                                                            }
                                                            bool v574;
                                                            v574 = 0 <= v560;
                                                            bool v576;
                                                            if (v574){
                                                                bool v575;
                                                                v575 = v560 < 2;
                                                                v576 = v575;
                                                            } else {
                                                                v576 = false;
                                                            }
                                                            bool v577;
                                                            v577 = v576 == false;
                                                            if (v577){
                                                                assert("The indices should be inside the range of the dimension." && v576);
                                                            } else {
                                                            }
                                                            int v579;
                                                            v579 = v560 * 2;
                                                            int v580;
                                                            v580 = v562 + v579;
                                                            v556.x[v580] = wmma::__float_to_tf32(v568);
                                                            v562 += 1 ;
                                                        }
                                                        v560 += 1 ;
                                                    }
                                                    int v581;
                                                    v581 = 0;
                                                    #pragma unroll
                                                    while (while_method_6(v581)){
                                                        assert("Tensor range check" && 0 <= v552 && v552 < 8);
                                                        assert("Tensor range check" && 0 <= v581 && v581 < 1);
                                                        int v583;
                                                        v583 = v552 + v581;
                                                        wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v584 = v383[v583];
                                                        assert("Tensor range check" && 0 <= v581 && v581 < 1);
                                                        assert("Tensor range check" && 0 <= v554 && v554 < 8);
                                                        int v585;
                                                        v585 = 8 * v581;
                                                        int v586;
                                                        v586 = v585 + v554;
                                                        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> & v587 = v487[v586];
                                                        wmma::mma_sync(v584, v556, v587, v584);
                                                        v581 += 1 ;
                                                    }
                                                    v554 += 1 ;
                                                }
                                                v552 += 1 ;
                                            }
                                            // Poping the loop unrolling to: 0
                                            asm("barrier.cta.sync %0;" :: "r"(0));
                                            v400 = v402;
                                        }
                                        // Pushing the loop unrolling to: 0
                                        int v588;
                                        v588 = 0;
                                        #pragma unroll
                                        while (while_method_10(v588)){
                                            int v590;
                                            v590 = 0;
                                            #pragma unroll
                                            while (while_method_6(v590)){
                                                assert("Tensor range check" && 0 <= v588 && v588 < 8);
                                                assert("Tensor range check" && 0 <= v590 && v590 < 1);
                                                int v592;
                                                v592 = v588 + v590;
                                                wmma::fragment<wmma::accumulator, 16, 16, 8, float> & v593 = v383[v592];
                                                assert("Tensor range check" && 0 <= v588 && v588 < 8);
                                                assert("Tensor range check" && 0 <= v590 && v590 < 1);
                                                int v594;
                                                v594 = 16 * v590;
                                                int v595;
                                                v595 = 2176 * v588;
                                                int v596;
                                                v596 = v595 + v594;
                                                float * v597;
                                                v597 = v349+v596;
                                                wmma::store_matrix_sync(v597, v593, 136, wmma::mem_row_major);
                                                v590 += 1 ;
                                            }
                                            v588 += 1 ;
                                        }
                                        // Poping the loop unrolling to: 0
                                        asm("barrier.cta.sync %0;" :: "r"(0));
                                        // Pushing the loop unrolling to: 0
                                        int v599;
                                        v599 = threadIdx.x;
                                        bool v600;
                                        v600 = 0 <= v599;
                                        bool v601;
                                        v601 = v600 == false;
                                        if (v601){
                                            assert("The index needs to be zero or positive." && v600);
                                        } else {
                                        }
                                        int v603;
                                        v603 = v599 % 32;
                                        int v604;
                                        v604 = v599 / 32;
                                        bool v605;
                                        v605 = v604 < 8;
                                        bool v606;
                                        v606 = v605 == false;
                                        if (v606){
                                            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v605);
                                        } else {
                                        }
                                        assert("Tensor range check" && 0 <= v604 && v604 < 8);
                                        assert("Tensor range check" && 0 <= v603 && v603 < 32);
                                        int v608;
                                        v608 = 4 * v603;
                                        int v609;
                                        v609 = 128 * v604;
                                        int v610;
                                        v610 = v609 + v608;
                                        int v611;
                                        v611 = 136 * v604;
                                        int v612;
                                        v612 = v611 + v608;
                                        float * v613;
                                        v613 = v392+v610;
                                        float * v615;
                                        v615 = v334+v612;
                                        int v617;
                                        v617 = 0;
                                        #pragma unroll
                                        while (while_method_12(v617)){
                                            int v619;
                                            v619 = 0;
                                            #pragma unroll
                                            while (while_method_6(v619)){
                                                assert("Tensor range check" && 0 <= v617 && v617 < 16);
                                                assert("Tensor range check" && 0 <= v619 && v619 < 1);
                                                int v621;
                                                v621 = 128 * v619;
                                                int v622;
                                                v622 = 1024 * v617;
                                                int v623;
                                                v623 = v622 + v621;
                                                int v624;
                                                v624 = 1088 * v617;
                                                int v625;
                                                v625 = v624 + v621;
                                                int4* v626;
                                                v626 = reinterpret_cast<int4*>(v615 + v625);
                                                int4* v627;
                                                v627 = reinterpret_cast<int4*>(v613 + v623);
                                                assert("Pointer alignment check" && (unsigned long long)(v626) % 4 == 0 && (unsigned long long)(v627) % 4 == 0);
                                                *v627 = *v626;
                                                v619 += 1 ;
                                            }
                                            v617 += 1 ;
                                        }
                                        // Poping the loop unrolling to: 0
                                        asm("barrier.cta.sync %0;" :: "r"(0));
                                        v386 += 1 ;
                                    }
                                    v384 += 1 ;
                                }
                                unsigned int * v628;
                                v628 = reinterpret_cast<unsigned int *>(&v0[53477376ull]);
                                assert("Tensor range check" && 0 <= v316 && v316 < 4);
                                int v630;
                                v630 = 6144 * v316;
                                method_45(v628, v630, v322);
                                int * v631;
                                v631 = reinterpret_cast<int *>(&v2[4194304ull]);
                                float * v633;
                                v633 = reinterpret_cast<float *>(&v2[4194320ull]);
                                float * v635;
                                v635 = reinterpret_cast<float *>(&v2[5242896ull]);
                                float * v637;
                                v637 = reinterpret_cast<float *>(&v2[6291472ull]);
                                float * v639;
                                v639 = reinterpret_cast<float *>(&v2[7340048ull]);
                                float * v641;
                                v641 = reinterpret_cast<float *>(&v2[8388624ull]);
                                float * v643;
                                v643 = reinterpret_cast<float *>(&v2[9437200ull]);
                                float * v645;
                                v645 = reinterpret_cast<float *>(&v2[10485776ull]);
                                int * v647;
                                v647 = reinterpret_cast<int *>(&v0[53575680ull]);
                                float * v649;
                                v649 = reinterpret_cast<float *>(&v0[66158592ull]);
                                int * v651;
                                v651 = reinterpret_cast<int *>(&v0[78741504ull]);
                                int * v653;
                                v653 = reinterpret_cast<int *>(&v0[91324416ull]);
                                double * v655;
                                v655 = reinterpret_cast<double *>(&v0[103907328ull]);
                                double * v657;
                                v657 = reinterpret_cast<double *>(&v0[154238976ull]);
                                double * v659;
                                v659 = reinterpret_cast<double *>(&v2[11534352ull]);
                                double * v661;
                                v661 = reinterpret_cast<double *>(&v2[11927568ull]);
                                int * v663;
                                v663 = reinterpret_cast<int *>(&v2[12320784ull]);
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                unsigned int * v665;
                                v665 = reinterpret_cast<unsigned int *>(&v0[53477376ull]);
                                int v667;
                                v667 = blockIdx.x;
                                int v668;
                                v668 = threadIdx.x;
                                assert("Tensor range check" && 0 <= v316 && v316 < 4);
                                assert("Tensor range check" && 0 <= v667 && v667 < 24);
                                assert("Tensor range check" && 0 <= v668 && v668 < 256);
                                int v669;
                                v669 = 256 * v667;
                                int v670;
                                v670 = v669 + v668;
                                int v671;
                                v671 = v630 + v670;
                                unsigned int v672;
                                v672 = v665[v671];
                                int * v673;
                                v673 = reinterpret_cast<int *>(&v2[4194304ull]);
                                float * v675;
                                v675 = reinterpret_cast<float *>(&v2[4194320ull]);
                                float * v677;
                                v677 = reinterpret_cast<float *>(&v2[5242896ull]);
                                float * v679;
                                v679 = reinterpret_cast<float *>(&v2[6291472ull]);
                                float * v681;
                                v681 = reinterpret_cast<float *>(&v2[7340048ull]);
                                float * v683;
                                v683 = reinterpret_cast<float *>(&v2[8388624ull]);
                                float * v685;
                                v685 = reinterpret_cast<float *>(&v2[9437200ull]);
                                float * v687;
                                v687 = reinterpret_cast<float *>(&v2[10485776ull]);
                                int v689;
                                v689 = (int)v672;
                                float v690; int v691;
                                Tuple14 tmp40 = method_46(v125, v673, v675, v677, v679, v681, v683, v685, v687, v689, v316);
                                v690 = tmp40.v0; v691 = tmp40.v1;
                                bool v692;
                                v692 = 0 == v691;
                                Union10 v725;
                                if (v692){
                                    v725 = Union10{Union10_1{}};
                                } else {
                                    bool v694;
                                    v694 = 1 == v691;
                                    if (v694){
                                        v725 = Union10{Union10_0{}};
                                    } else {
                                        bool v696;
                                        v696 = 2 == v691;
                                        if (v696){
                                            v725 = Union10{Union10_2{1, 3}};
                                        } else {
                                            bool v698;
                                            v698 = 3 == v691;
                                            if (v698){
                                                v725 = Union10{Union10_2{1, 2}};
                                            } else {
                                                bool v700;
                                                v700 = 4 == v691;
                                                if (v700){
                                                    v725 = Union10{Union10_2{2, 3}};
                                                } else {
                                                    bool v702;
                                                    v702 = 5 == v691;
                                                    if (v702){
                                                        v725 = Union10{Union10_2{3, 4}};
                                                    } else {
                                                        bool v704;
                                                        v704 = 6 == v691;
                                                        if (v704){
                                                            v725 = Union10{Union10_2{1, 1}};
                                                        } else {
                                                            bool v706;
                                                            v706 = 7 == v691;
                                                            if (v706){
                                                                v725 = Union10{Union10_2{3, 2}};
                                                            } else {
                                                                bool v708;
                                                                v708 = 8 == v691;
                                                                if (v708){
                                                                    v725 = Union10{Union10_2{2, 1}};
                                                                } else {
                                                                    bool v710;
                                                                    v710 = 9 == v691;
                                                                    if (v710){
                                                                        v725 = Union10{Union10_2{3, 1}};
                                                                    } else {
                                                                        bool v712;
                                                                        v712 = 10 == v691;
                                                                        if (v712){
                                                                            v725 = Union10{Union10_2{2147483647, 1}};
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
                                Union1 v803;
                                switch (v725.tag) {
                                    case 0: { // AA_Call
                                        v803 = Union1{Union1_1{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v726;
                                        v726 = v109[0];
                                        int v728; int v729;
                                        Tuple4 tmp41 = Tuple4{1, v726};
                                        v728 = tmp41.v0; v729 = tmp41.v1;
                                        while (while_method_0(v728)){
                                            int v731;
                                            v731 = v109[v728];
                                            bool v733;
                                            v733 = v729 >= v731;
                                            int v734;
                                            if (v733){
                                                v734 = v729;
                                            } else {
                                                v734 = v731;
                                            }
                                            v729 = v734;
                                            v728 += 1 ;
                                        }
                                        int v735;
                                        v735 = v109[v113];
                                        bool v737;
                                        v737 = v735 == v729;
                                        if (v737){
                                            v803 = Union1{Union1_1{}};
                                        } else {
                                            v803 = Union1{Union1_2{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        int v742 = v725.case2.v0; int v743 = v725.case2.v1;
                                        static_array<int,2> v744;
                                        int v746;
                                        v746 = 0;
                                        while (while_method_0(v746)){
                                            int v748;
                                            v748 = v111[v746];
                                            int v750;
                                            v750 = v109[v746];
                                            int v752;
                                            v752 = v748 + v750;
                                            v744[v746] = v752;
                                            v746 += 1 ;
                                        }
                                        int v753;
                                        v753 = v109[0];
                                        int v755; int v756;
                                        Tuple4 tmp42 = Tuple4{1, v753};
                                        v755 = tmp42.v0; v756 = tmp42.v1;
                                        while (while_method_0(v755)){
                                            int v758;
                                            v758 = v109[v755];
                                            bool v760;
                                            v760 = v756 >= v758;
                                            int v761;
                                            if (v760){
                                                v761 = v756;
                                            } else {
                                                v761 = v758;
                                            }
                                            v756 = v761;
                                            v755 += 1 ;
                                        }
                                        int v762;
                                        v762 = v744[v113];
                                        bool v764;
                                        v764 = v756 < v762;
                                        int v765;
                                        if (v764){
                                            v765 = v756;
                                        } else {
                                            v765 = v762;
                                        }
                                        static_array<int,2> v766;
                                        int v768;
                                        v768 = 0;
                                        while (while_method_0(v768)){
                                            int v770;
                                            v770 = v109[v768];
                                            bool v772;
                                            v772 = v113 == v768;
                                            int v773;
                                            if (v772){
                                                v773 = v765;
                                            } else {
                                                v773 = v770;
                                            }
                                            v766[v768] = v773;
                                            v768 += 1 ;
                                        }
                                        int v774;
                                        v774 = v766[0];
                                        int v776; int v777;
                                        Tuple4 tmp43 = Tuple4{1, v774};
                                        v776 = tmp43.v0; v777 = tmp43.v1;
                                        while (while_method_0(v776)){
                                            int v779;
                                            v779 = v766[v776];
                                            int v781;
                                            v781 = v777 + v779;
                                            v777 = v781;
                                            v776 += 1 ;
                                        }
                                        static_array<int,2> v782;
                                        int v784;
                                        v784 = 0;
                                        while (while_method_0(v784)){
                                            int v786;
                                            v786 = v744[v784];
                                            int v788;
                                            v788 = v766[v784];
                                            int v790;
                                            v790 = v786 - v788;
                                            v782[v784] = v790;
                                            v784 += 1 ;
                                        }
                                        int v791;
                                        v791 = v742 * v777;
                                        int v792;
                                        v792 = v791 / v743;
                                        bool v793;
                                        v793 = v107 >= v792;
                                        int v794;
                                        if (v793){
                                            v794 = v107;
                                        } else {
                                            v794 = v792;
                                        }
                                        int v795;
                                        v795 = v782[v113];
                                        bool v797;
                                        v797 = v794 >= v795;
                                        if (v797){
                                            v803 = Union1{Union1_0{}};
                                        } else {
                                            v803 = Union1{Union1_3{v794}};
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                int v804;
                                v804 = sizeof(Union1);
                                unsigned long long v805;
                                v805 = (unsigned long long)v804;
                                bool v806;
                                v806 = v805 <= 98304ull;
                                bool v807;
                                v807 = v806 == false;
                                if (v807){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v806);
                                } else {
                                }
                                extern __shared__ unsigned char v809[];
                                bool v810;
                                v810 = v805 <= v805;
                                bool v811;
                                v811 = v810 == false;
                                if (v811){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v810);
                                } else {
                                }
                                Union1 * v813;
                                v813 = reinterpret_cast<Union1 *>(&v809[0ull]);
                                int v815;
                                v815 = threadIdx.x;
                                bool v816;
                                v816 = v815 == 0;
                                if (v816){
                                    v813[0] = v803;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                Union1 v817;
                                v817 = v813[0];
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                Union6 v818;
                                v818 = Union6{Union6_2{v113, v817}};
                                v6.push(v818);
                                Union4 v1006;
                                switch (v817.tag) {
                                    case 0: { // A_All_In
                                        static_array<int,2> v936;
                                        int v938;
                                        v938 = 0;
                                        while (while_method_0(v938)){
                                            int v940;
                                            v940 = v111[v938];
                                            int v942;
                                            v942 = v109[v938];
                                            int v944;
                                            v944 = v940 + v942;
                                            v936[v938] = v944;
                                            v938 += 1 ;
                                        }
                                        int v945;
                                        v945 = v109[0];
                                        int v947; int v948;
                                        Tuple4 tmp44 = Tuple4{1, v945};
                                        v947 = tmp44.v0; v948 = tmp44.v1;
                                        while (while_method_0(v947)){
                                            int v950;
                                            v950 = v109[v947];
                                            bool v952;
                                            v952 = v948 >= v950;
                                            int v953;
                                            if (v952){
                                                v953 = v948;
                                            } else {
                                                v953 = v950;
                                            }
                                            v948 = v953;
                                            v947 += 1 ;
                                        }
                                        int v954;
                                        v954 = v936[v113];
                                        bool v956;
                                        v956 = v948 < v954;
                                        int v957;
                                        if (v956){
                                            v957 = v948;
                                        } else {
                                            v957 = v954;
                                        }
                                        static_array<int,2> v958;
                                        int v960;
                                        v960 = 0;
                                        while (while_method_0(v960)){
                                            int v962;
                                            v962 = v109[v960];
                                            bool v964;
                                            v964 = v113 == v960;
                                            int v965;
                                            if (v964){
                                                v965 = v957;
                                            } else {
                                                v965 = v962;
                                            }
                                            v958[v960] = v965;
                                            v960 += 1 ;
                                        }
                                        static_array<int,2> v966;
                                        int v968;
                                        v968 = 0;
                                        while (while_method_0(v968)){
                                            int v970;
                                            v970 = v936[v968];
                                            int v972;
                                            v972 = v958[v968];
                                            int v974;
                                            v974 = v970 - v972;
                                            v966[v968] = v974;
                                            v968 += 1 ;
                                        }
                                        int v975;
                                        v975 = v966[v113];
                                        int v977;
                                        v977 = v948 + v975;
                                        int v978;
                                        v978 = v936[v113];
                                        bool v980;
                                        v980 = v977 < v978;
                                        int v981;
                                        if (v980){
                                            v981 = v977;
                                        } else {
                                            v981 = v978;
                                        }
                                        static_array<int,2> v982;
                                        int v984;
                                        v984 = 0;
                                        while (while_method_0(v984)){
                                            int v986;
                                            v986 = v109[v984];
                                            bool v988;
                                            v988 = v113 == v984;
                                            int v989;
                                            if (v988){
                                                v989 = v981;
                                            } else {
                                                v989 = v986;
                                            }
                                            v982[v984] = v989;
                                            v984 += 1 ;
                                        }
                                        static_array<int,2> v990;
                                        int v992;
                                        v992 = 0;
                                        while (while_method_0(v992)){
                                            int v994;
                                            v994 = v936[v992];
                                            int v996;
                                            v996 = v982[v992];
                                            int v998;
                                            v998 = v994 - v996;
                                            v990[v992] = v998;
                                            v992 += 1 ;
                                        }
                                        bool v999;
                                        v999 = v975 >= v107;
                                        int v1000;
                                        if (v999){
                                            v1000 = v975;
                                        } else {
                                            v1000 = v107;
                                        }
                                        int v1001;
                                        v1001 = v110 + 1;
                                        v1006 = try_round_36(v1000, v108, v982, v1001, v990, v112);
                                        break;
                                    }
                                    case 1: { // A_Call
                                        static_array<int,2> v820;
                                        int v822;
                                        v822 = 0;
                                        while (while_method_0(v822)){
                                            int v824;
                                            v824 = v111[v822];
                                            int v826;
                                            v826 = v109[v822];
                                            int v828;
                                            v828 = v824 + v826;
                                            v820[v822] = v828;
                                            v822 += 1 ;
                                        }
                                        int v829;
                                        v829 = v109[0];
                                        int v831; int v832;
                                        Tuple4 tmp45 = Tuple4{1, v829};
                                        v831 = tmp45.v0; v832 = tmp45.v1;
                                        while (while_method_0(v831)){
                                            int v834;
                                            v834 = v109[v831];
                                            bool v836;
                                            v836 = v832 >= v834;
                                            int v837;
                                            if (v836){
                                                v837 = v832;
                                            } else {
                                                v837 = v834;
                                            }
                                            v832 = v837;
                                            v831 += 1 ;
                                        }
                                        int v838;
                                        v838 = v820[v113];
                                        bool v840;
                                        v840 = v832 < v838;
                                        int v841;
                                        if (v840){
                                            v841 = v832;
                                        } else {
                                            v841 = v838;
                                        }
                                        static_array<int,2> v842;
                                        int v844;
                                        v844 = 0;
                                        while (while_method_0(v844)){
                                            int v846;
                                            v846 = v109[v844];
                                            bool v848;
                                            v848 = v113 == v844;
                                            int v849;
                                            if (v848){
                                                v849 = v841;
                                            } else {
                                                v849 = v846;
                                            }
                                            v842[v844] = v849;
                                            v844 += 1 ;
                                        }
                                        static_array<int,2> v850;
                                        int v852;
                                        v852 = 0;
                                        while (while_method_0(v852)){
                                            int v854;
                                            v854 = v820[v852];
                                            int v856;
                                            v856 = v842[v852];
                                            int v858;
                                            v858 = v854 - v856;
                                            v850[v852] = v858;
                                            v852 += 1 ;
                                        }
                                        bool v859;
                                        v859 = v113 < 2;
                                        if (v859){
                                            int v860;
                                            v860 = v110 + 1;
                                            v1006 = try_round_36(v107, v108, v842, v860, v850, v112);
                                        } else {
                                            v1006 = go_next_street_38(v107, v108, v842, v110, v850, v112);
                                        }
                                        break;
                                    }
                                    case 2: { // A_Fold
                                        v1006 = Union4{Union4_1{v107, v108, v109, v110, v111, v112}};
                                        break;
                                    }
                                    case 3: { // A_Raise
                                        int v864 = v817.case3.v0;
                                        bool v865;
                                        v865 = v107 <= v864;
                                        bool v866;
                                        v866 = v865 == false;
                                        if (v866){
                                            assert("The raise amount must match the minimum." && v865);
                                        } else {
                                        }
                                        static_array<int,2> v868;
                                        int v870;
                                        v870 = 0;
                                        while (while_method_0(v870)){
                                            int v872;
                                            v872 = v111[v870];
                                            int v874;
                                            v874 = v109[v870];
                                            int v876;
                                            v876 = v872 + v874;
                                            v868[v870] = v876;
                                            v870 += 1 ;
                                        }
                                        int v877;
                                        v877 = v109[0];
                                        int v879; int v880;
                                        Tuple4 tmp46 = Tuple4{1, v877};
                                        v879 = tmp46.v0; v880 = tmp46.v1;
                                        while (while_method_0(v879)){
                                            int v882;
                                            v882 = v109[v879];
                                            bool v884;
                                            v884 = v880 >= v882;
                                            int v885;
                                            if (v884){
                                                v885 = v880;
                                            } else {
                                                v885 = v882;
                                            }
                                            v880 = v885;
                                            v879 += 1 ;
                                        }
                                        int v886;
                                        v886 = v868[v113];
                                        bool v888;
                                        v888 = v880 < v886;
                                        int v889;
                                        if (v888){
                                            v889 = v880;
                                        } else {
                                            v889 = v886;
                                        }
                                        static_array<int,2> v890;
                                        int v892;
                                        v892 = 0;
                                        while (while_method_0(v892)){
                                            int v894;
                                            v894 = v109[v892];
                                            bool v896;
                                            v896 = v113 == v892;
                                            int v897;
                                            if (v896){
                                                v897 = v889;
                                            } else {
                                                v897 = v894;
                                            }
                                            v890[v892] = v897;
                                            v892 += 1 ;
                                        }
                                        static_array<int,2> v898;
                                        int v900;
                                        v900 = 0;
                                        while (while_method_0(v900)){
                                            int v902;
                                            v902 = v868[v900];
                                            int v904;
                                            v904 = v890[v900];
                                            int v906;
                                            v906 = v902 - v904;
                                            v898[v900] = v906;
                                            v900 += 1 ;
                                        }
                                        int v907;
                                        v907 = v898[v113];
                                        bool v909;
                                        v909 = v864 < v907;
                                        bool v910;
                                        v910 = v909 == false;
                                        if (v910){
                                            assert("The raise amount must be less than the stack size after calling." && v909);
                                        } else {
                                        }
                                        int v912;
                                        v912 = v880 + v864;
                                        int v913;
                                        v913 = v868[v113];
                                        bool v915;
                                        v915 = v912 < v913;
                                        int v916;
                                        if (v915){
                                            v916 = v912;
                                        } else {
                                            v916 = v913;
                                        }
                                        static_array<int,2> v917;
                                        int v919;
                                        v919 = 0;
                                        while (while_method_0(v919)){
                                            int v921;
                                            v921 = v109[v919];
                                            bool v923;
                                            v923 = v113 == v919;
                                            int v924;
                                            if (v923){
                                                v924 = v916;
                                            } else {
                                                v924 = v921;
                                            }
                                            v917[v919] = v924;
                                            v919 += 1 ;
                                        }
                                        static_array<int,2> v925;
                                        int v927;
                                        v927 = 0;
                                        while (while_method_0(v927)){
                                            int v929;
                                            v929 = v868[v927];
                                            int v931;
                                            v931 = v917[v927];
                                            int v933;
                                            v933 = v929 - v931;
                                            v925[v927] = v933;
                                            v927 += 1 ;
                                        }
                                        int v934;
                                        v934 = v110 + 1;
                                        v1006 = try_round_36(v864, v108, v917, v934, v925, v112);
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                v1621 = Union3{Union3_1{v1006}};
                                break;
                            }
                            case 1: { // Human
                                Union7 v1008;
                                v1008 = Union7{Union7_2{v107, v108, v109, v110, v111, v112}};
                                v4.v5 = v1008;
                                Union3 v1009;
                                v1009 = Union3{Union3_1{v11}};
                                v4.v1 = v1009;
                                v1621 = Union3{Union3_0{}};
                                break;
                            }
                            case 2: { // Random
                                curandStatePhilox4_32_10_t & v1011 = v4.v4;
                                curandStatePhilox4_32_10_t & v1012 = v1011;
                                static_array<int,2> v1013;
                                int v1015;
                                v1015 = 0;
                                while (while_method_0(v1015)){
                                    int v1017;
                                    v1017 = v111[v1015];
                                    int v1019;
                                    v1019 = v109[v1015];
                                    int v1021;
                                    v1021 = v1017 + v1019;
                                    v1013[v1015] = v1021;
                                    v1015 += 1 ;
                                }
                                int v1022;
                                v1022 = v109[0];
                                int v1024; int v1025;
                                Tuple4 tmp47 = Tuple4{1, v1022};
                                v1024 = tmp47.v0; v1025 = tmp47.v1;
                                while (while_method_0(v1024)){
                                    int v1027;
                                    v1027 = v109[v1024];
                                    bool v1029;
                                    v1029 = v1025 >= v1027;
                                    int v1030;
                                    if (v1029){
                                        v1030 = v1025;
                                    } else {
                                        v1030 = v1027;
                                    }
                                    v1025 = v1030;
                                    v1024 += 1 ;
                                }
                                int v1031;
                                v1031 = v1013[v113];
                                bool v1033;
                                v1033 = v1025 < v1031;
                                int v1034;
                                if (v1033){
                                    v1034 = v1025;
                                } else {
                                    v1034 = v1031;
                                }
                                static_array<int,2> v1035;
                                int v1037;
                                v1037 = 0;
                                while (while_method_0(v1037)){
                                    int v1039;
                                    v1039 = v109[v1037];
                                    bool v1041;
                                    v1041 = v113 == v1037;
                                    int v1042;
                                    if (v1041){
                                        v1042 = v1034;
                                    } else {
                                        v1042 = v1039;
                                    }
                                    v1035[v1037] = v1042;
                                    v1037 += 1 ;
                                }
                                int v1043;
                                v1043 = v1035[0];
                                int v1045; int v1046;
                                Tuple4 tmp48 = Tuple4{1, v1043};
                                v1045 = tmp48.v0; v1046 = tmp48.v1;
                                while (while_method_0(v1045)){
                                    int v1048;
                                    v1048 = v1035[v1045];
                                    int v1050;
                                    v1050 = v1046 + v1048;
                                    v1046 = v1050;
                                    v1045 += 1 ;
                                }
                                static_array<int,2> v1051;
                                int v1053;
                                v1053 = 0;
                                while (while_method_0(v1053)){
                                    int v1055;
                                    v1055 = v1013[v1053];
                                    int v1057;
                                    v1057 = v1035[v1053];
                                    int v1059;
                                    v1059 = v1055 - v1057;
                                    v1051[v1053] = v1059;
                                    v1053 += 1 ;
                                }
                                int v1060;
                                v1060 = v109[v113];
                                bool v1062;
                                v1062 = v1060 < v1025;
                                float v1063;
                                if (v1062){
                                    v1063 = 1.0f;
                                } else {
                                    v1063 = 0.0f;
                                }
                                int v1064;
                                v1064 = v1046 / 3;
                                bool v1065;
                                v1065 = v107 <= v1064;
                                bool v1069;
                                if (v1065){
                                    int v1066;
                                    v1066 = v1051[v113];
                                    bool v1068;
                                    v1068 = v1064 < v1066;
                                    v1069 = v1068;
                                } else {
                                    v1069 = false;
                                }
                                float v1070;
                                if (v1069){
                                    v1070 = 1.0f;
                                } else {
                                    v1070 = 0.0f;
                                }
                                int v1071;
                                v1071 = v1046 / 2;
                                bool v1072;
                                v1072 = v107 <= v1071;
                                bool v1076;
                                if (v1072){
                                    int v1073;
                                    v1073 = v1051[v113];
                                    bool v1075;
                                    v1075 = v1071 < v1073;
                                    v1076 = v1075;
                                } else {
                                    v1076 = false;
                                }
                                float v1077;
                                if (v1076){
                                    v1077 = 1.0f;
                                } else {
                                    v1077 = 0.0f;
                                }
                                bool v1078;
                                v1078 = v107 <= v1046;
                                bool v1082;
                                if (v1078){
                                    int v1079;
                                    v1079 = v1051[v113];
                                    bool v1081;
                                    v1081 = v1046 < v1079;
                                    v1082 = v1081;
                                } else {
                                    v1082 = false;
                                }
                                float v1083;
                                if (v1082){
                                    v1083 = 1.0f;
                                } else {
                                    v1083 = 0.0f;
                                }
                                static_array<Tuple18,6> v1084;
                                Union1 v1086;
                                v1086 = Union1{Union1_2{}};
                                v1084[0] = Tuple18{v1086, v1063};
                                Union1 v1088;
                                v1088 = Union1{Union1_1{}};
                                v1084[1] = Tuple18{v1088, 4.0f};
                                Union1 v1090;
                                v1090 = Union1{Union1_3{v1064}};
                                v1084[2] = Tuple18{v1090, v1070};
                                Union1 v1092;
                                v1092 = Union1{Union1_3{v1071}};
                                v1084[3] = Tuple18{v1092, v1077};
                                Union1 v1094;
                                v1094 = Union1{Union1_3{v1046}};
                                v1084[4] = Tuple18{v1094, v1083};
                                Union1 v1096;
                                v1096 = Union1{Union1_0{}};
                                v1084[5] = Tuple18{v1096, 1.0f};
                                Union1 v1098;
                                v1098 = sample_discrete_47(v1084, v1012);
                                int v1099;
                                v1099 = sizeof(Union1);
                                unsigned long long v1100;
                                v1100 = (unsigned long long)v1099;
                                bool v1101;
                                v1101 = v1100 <= 98304ull;
                                bool v1102;
                                v1102 = v1101 == false;
                                if (v1102){
                                    assert("The dynamic shared memory is insufficient to allocate the tensor." && v1101);
                                } else {
                                }
                                extern __shared__ unsigned char v1104[];
                                bool v1105;
                                v1105 = v1100 <= v1100;
                                bool v1106;
                                v1106 = v1105 == false;
                                if (v1106){
                                    assert("The length of the partition has to be less than or equal to the length of the base array." && v1105);
                                } else {
                                }
                                Union1 * v1108;
                                v1108 = reinterpret_cast<Union1 *>(&v1104[0ull]);
                                int v1110;
                                v1110 = threadIdx.x;
                                bool v1111;
                                v1111 = v1110 == 0;
                                if (v1111){
                                    v1108[0] = v1098;
                                } else {
                                }
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                Union1 v1112;
                                v1112 = v1108[0];
                                asm("barrier.cta.sync %0;" :: "r"(0));
                                Union6 v1113;
                                v1113 = Union6{Union6_2{v113, v1112}};
                                v6.push(v1113);
                                Union4 v1301;
                                switch (v1112.tag) {
                                    case 0: { // A_All_In
                                        static_array<int,2> v1231;
                                        int v1233;
                                        v1233 = 0;
                                        while (while_method_0(v1233)){
                                            int v1235;
                                            v1235 = v111[v1233];
                                            int v1237;
                                            v1237 = v109[v1233];
                                            int v1239;
                                            v1239 = v1235 + v1237;
                                            v1231[v1233] = v1239;
                                            v1233 += 1 ;
                                        }
                                        int v1240;
                                        v1240 = v109[0];
                                        int v1242; int v1243;
                                        Tuple4 tmp51 = Tuple4{1, v1240};
                                        v1242 = tmp51.v0; v1243 = tmp51.v1;
                                        while (while_method_0(v1242)){
                                            int v1245;
                                            v1245 = v109[v1242];
                                            bool v1247;
                                            v1247 = v1243 >= v1245;
                                            int v1248;
                                            if (v1247){
                                                v1248 = v1243;
                                            } else {
                                                v1248 = v1245;
                                            }
                                            v1243 = v1248;
                                            v1242 += 1 ;
                                        }
                                        int v1249;
                                        v1249 = v1231[v113];
                                        bool v1251;
                                        v1251 = v1243 < v1249;
                                        int v1252;
                                        if (v1251){
                                            v1252 = v1243;
                                        } else {
                                            v1252 = v1249;
                                        }
                                        static_array<int,2> v1253;
                                        int v1255;
                                        v1255 = 0;
                                        while (while_method_0(v1255)){
                                            int v1257;
                                            v1257 = v109[v1255];
                                            bool v1259;
                                            v1259 = v113 == v1255;
                                            int v1260;
                                            if (v1259){
                                                v1260 = v1252;
                                            } else {
                                                v1260 = v1257;
                                            }
                                            v1253[v1255] = v1260;
                                            v1255 += 1 ;
                                        }
                                        static_array<int,2> v1261;
                                        int v1263;
                                        v1263 = 0;
                                        while (while_method_0(v1263)){
                                            int v1265;
                                            v1265 = v1231[v1263];
                                            int v1267;
                                            v1267 = v1253[v1263];
                                            int v1269;
                                            v1269 = v1265 - v1267;
                                            v1261[v1263] = v1269;
                                            v1263 += 1 ;
                                        }
                                        int v1270;
                                        v1270 = v1261[v113];
                                        int v1272;
                                        v1272 = v1243 + v1270;
                                        int v1273;
                                        v1273 = v1231[v113];
                                        bool v1275;
                                        v1275 = v1272 < v1273;
                                        int v1276;
                                        if (v1275){
                                            v1276 = v1272;
                                        } else {
                                            v1276 = v1273;
                                        }
                                        static_array<int,2> v1277;
                                        int v1279;
                                        v1279 = 0;
                                        while (while_method_0(v1279)){
                                            int v1281;
                                            v1281 = v109[v1279];
                                            bool v1283;
                                            v1283 = v113 == v1279;
                                            int v1284;
                                            if (v1283){
                                                v1284 = v1276;
                                            } else {
                                                v1284 = v1281;
                                            }
                                            v1277[v1279] = v1284;
                                            v1279 += 1 ;
                                        }
                                        static_array<int,2> v1285;
                                        int v1287;
                                        v1287 = 0;
                                        while (while_method_0(v1287)){
                                            int v1289;
                                            v1289 = v1231[v1287];
                                            int v1291;
                                            v1291 = v1277[v1287];
                                            int v1293;
                                            v1293 = v1289 - v1291;
                                            v1285[v1287] = v1293;
                                            v1287 += 1 ;
                                        }
                                        bool v1294;
                                        v1294 = v1270 >= v107;
                                        int v1295;
                                        if (v1294){
                                            v1295 = v1270;
                                        } else {
                                            v1295 = v107;
                                        }
                                        int v1296;
                                        v1296 = v110 + 1;
                                        v1301 = try_round_36(v1295, v108, v1277, v1296, v1285, v112);
                                        break;
                                    }
                                    case 1: { // A_Call
                                        static_array<int,2> v1115;
                                        int v1117;
                                        v1117 = 0;
                                        while (while_method_0(v1117)){
                                            int v1119;
                                            v1119 = v111[v1117];
                                            int v1121;
                                            v1121 = v109[v1117];
                                            int v1123;
                                            v1123 = v1119 + v1121;
                                            v1115[v1117] = v1123;
                                            v1117 += 1 ;
                                        }
                                        int v1124;
                                        v1124 = v109[0];
                                        int v1126; int v1127;
                                        Tuple4 tmp52 = Tuple4{1, v1124};
                                        v1126 = tmp52.v0; v1127 = tmp52.v1;
                                        while (while_method_0(v1126)){
                                            int v1129;
                                            v1129 = v109[v1126];
                                            bool v1131;
                                            v1131 = v1127 >= v1129;
                                            int v1132;
                                            if (v1131){
                                                v1132 = v1127;
                                            } else {
                                                v1132 = v1129;
                                            }
                                            v1127 = v1132;
                                            v1126 += 1 ;
                                        }
                                        int v1133;
                                        v1133 = v1115[v113];
                                        bool v1135;
                                        v1135 = v1127 < v1133;
                                        int v1136;
                                        if (v1135){
                                            v1136 = v1127;
                                        } else {
                                            v1136 = v1133;
                                        }
                                        static_array<int,2> v1137;
                                        int v1139;
                                        v1139 = 0;
                                        while (while_method_0(v1139)){
                                            int v1141;
                                            v1141 = v109[v1139];
                                            bool v1143;
                                            v1143 = v113 == v1139;
                                            int v1144;
                                            if (v1143){
                                                v1144 = v1136;
                                            } else {
                                                v1144 = v1141;
                                            }
                                            v1137[v1139] = v1144;
                                            v1139 += 1 ;
                                        }
                                        static_array<int,2> v1145;
                                        int v1147;
                                        v1147 = 0;
                                        while (while_method_0(v1147)){
                                            int v1149;
                                            v1149 = v1115[v1147];
                                            int v1151;
                                            v1151 = v1137[v1147];
                                            int v1153;
                                            v1153 = v1149 - v1151;
                                            v1145[v1147] = v1153;
                                            v1147 += 1 ;
                                        }
                                        bool v1154;
                                        v1154 = v113 < 2;
                                        if (v1154){
                                            int v1155;
                                            v1155 = v110 + 1;
                                            v1301 = try_round_36(v107, v108, v1137, v1155, v1145, v112);
                                        } else {
                                            v1301 = go_next_street_38(v107, v108, v1137, v110, v1145, v112);
                                        }
                                        break;
                                    }
                                    case 2: { // A_Fold
                                        v1301 = Union4{Union4_1{v107, v108, v109, v110, v111, v112}};
                                        break;
                                    }
                                    case 3: { // A_Raise
                                        int v1159 = v1112.case3.v0;
                                        bool v1160;
                                        v1160 = v107 <= v1159;
                                        bool v1161;
                                        v1161 = v1160 == false;
                                        if (v1161){
                                            assert("The raise amount must match the minimum." && v1160);
                                        } else {
                                        }
                                        static_array<int,2> v1163;
                                        int v1165;
                                        v1165 = 0;
                                        while (while_method_0(v1165)){
                                            int v1167;
                                            v1167 = v111[v1165];
                                            int v1169;
                                            v1169 = v109[v1165];
                                            int v1171;
                                            v1171 = v1167 + v1169;
                                            v1163[v1165] = v1171;
                                            v1165 += 1 ;
                                        }
                                        int v1172;
                                        v1172 = v109[0];
                                        int v1174; int v1175;
                                        Tuple4 tmp53 = Tuple4{1, v1172};
                                        v1174 = tmp53.v0; v1175 = tmp53.v1;
                                        while (while_method_0(v1174)){
                                            int v1177;
                                            v1177 = v109[v1174];
                                            bool v1179;
                                            v1179 = v1175 >= v1177;
                                            int v1180;
                                            if (v1179){
                                                v1180 = v1175;
                                            } else {
                                                v1180 = v1177;
                                            }
                                            v1175 = v1180;
                                            v1174 += 1 ;
                                        }
                                        int v1181;
                                        v1181 = v1163[v113];
                                        bool v1183;
                                        v1183 = v1175 < v1181;
                                        int v1184;
                                        if (v1183){
                                            v1184 = v1175;
                                        } else {
                                            v1184 = v1181;
                                        }
                                        static_array<int,2> v1185;
                                        int v1187;
                                        v1187 = 0;
                                        while (while_method_0(v1187)){
                                            int v1189;
                                            v1189 = v109[v1187];
                                            bool v1191;
                                            v1191 = v113 == v1187;
                                            int v1192;
                                            if (v1191){
                                                v1192 = v1184;
                                            } else {
                                                v1192 = v1189;
                                            }
                                            v1185[v1187] = v1192;
                                            v1187 += 1 ;
                                        }
                                        static_array<int,2> v1193;
                                        int v1195;
                                        v1195 = 0;
                                        while (while_method_0(v1195)){
                                            int v1197;
                                            v1197 = v1163[v1195];
                                            int v1199;
                                            v1199 = v1185[v1195];
                                            int v1201;
                                            v1201 = v1197 - v1199;
                                            v1193[v1195] = v1201;
                                            v1195 += 1 ;
                                        }
                                        int v1202;
                                        v1202 = v1193[v113];
                                        bool v1204;
                                        v1204 = v1159 < v1202;
                                        bool v1205;
                                        v1205 = v1204 == false;
                                        if (v1205){
                                            assert("The raise amount must be less than the stack size after calling." && v1204);
                                        } else {
                                        }
                                        int v1207;
                                        v1207 = v1175 + v1159;
                                        int v1208;
                                        v1208 = v1163[v113];
                                        bool v1210;
                                        v1210 = v1207 < v1208;
                                        int v1211;
                                        if (v1210){
                                            v1211 = v1207;
                                        } else {
                                            v1211 = v1208;
                                        }
                                        static_array<int,2> v1212;
                                        int v1214;
                                        v1214 = 0;
                                        while (while_method_0(v1214)){
                                            int v1216;
                                            v1216 = v109[v1214];
                                            bool v1218;
                                            v1218 = v113 == v1214;
                                            int v1219;
                                            if (v1218){
                                                v1219 = v1211;
                                            } else {
                                                v1219 = v1216;
                                            }
                                            v1212[v1214] = v1219;
                                            v1214 += 1 ;
                                        }
                                        static_array<int,2> v1220;
                                        int v1222;
                                        v1222 = 0;
                                        while (while_method_0(v1222)){
                                            int v1224;
                                            v1224 = v1163[v1222];
                                            int v1226;
                                            v1226 = v1212[v1222];
                                            int v1228;
                                            v1228 = v1224 - v1226;
                                            v1220[v1222] = v1228;
                                            v1222 += 1 ;
                                        }
                                        int v1229;
                                        v1229 = v110 + 1;
                                        v1301 = try_round_36(v1159, v108, v1212, v1229, v1220, v112);
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                v1621 = Union3{Union3_1{v1301}};
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        break;
                    }
                    case 5: { // G_Round'
                        int v1306 = v11.case5.v0; static_array<static_array<unsigned char,2>,2> v1307 = v11.case5.v1; static_array<int,2> v1308 = v11.case5.v2; int v1309 = v11.case5.v3; static_array<int,2> v1310 = v11.case5.v4; Union5 v1311 = v11.case5.v5; Union1 v1312 = v11.case5.v6;
                        int v1313;
                        v1313 = v1309 % 2;
                        Union6 v1314;
                        v1314 = Union6{Union6_2{v1313, v1312}};
                        v6.push(v1314);
                        Union4 v1502;
                        switch (v1312.tag) {
                            case 0: { // A_All_In
                                static_array<int,2> v1432;
                                int v1434;
                                v1434 = 0;
                                while (while_method_0(v1434)){
                                    int v1436;
                                    v1436 = v1310[v1434];
                                    int v1438;
                                    v1438 = v1308[v1434];
                                    int v1440;
                                    v1440 = v1436 + v1438;
                                    v1432[v1434] = v1440;
                                    v1434 += 1 ;
                                }
                                int v1441;
                                v1441 = v1308[0];
                                int v1443; int v1444;
                                Tuple4 tmp54 = Tuple4{1, v1441};
                                v1443 = tmp54.v0; v1444 = tmp54.v1;
                                while (while_method_0(v1443)){
                                    int v1446;
                                    v1446 = v1308[v1443];
                                    bool v1448;
                                    v1448 = v1444 >= v1446;
                                    int v1449;
                                    if (v1448){
                                        v1449 = v1444;
                                    } else {
                                        v1449 = v1446;
                                    }
                                    v1444 = v1449;
                                    v1443 += 1 ;
                                }
                                int v1450;
                                v1450 = v1432[v1313];
                                bool v1452;
                                v1452 = v1444 < v1450;
                                int v1453;
                                if (v1452){
                                    v1453 = v1444;
                                } else {
                                    v1453 = v1450;
                                }
                                static_array<int,2> v1454;
                                int v1456;
                                v1456 = 0;
                                while (while_method_0(v1456)){
                                    int v1458;
                                    v1458 = v1308[v1456];
                                    bool v1460;
                                    v1460 = v1313 == v1456;
                                    int v1461;
                                    if (v1460){
                                        v1461 = v1453;
                                    } else {
                                        v1461 = v1458;
                                    }
                                    v1454[v1456] = v1461;
                                    v1456 += 1 ;
                                }
                                static_array<int,2> v1462;
                                int v1464;
                                v1464 = 0;
                                while (while_method_0(v1464)){
                                    int v1466;
                                    v1466 = v1432[v1464];
                                    int v1468;
                                    v1468 = v1454[v1464];
                                    int v1470;
                                    v1470 = v1466 - v1468;
                                    v1462[v1464] = v1470;
                                    v1464 += 1 ;
                                }
                                int v1471;
                                v1471 = v1462[v1313];
                                int v1473;
                                v1473 = v1444 + v1471;
                                int v1474;
                                v1474 = v1432[v1313];
                                bool v1476;
                                v1476 = v1473 < v1474;
                                int v1477;
                                if (v1476){
                                    v1477 = v1473;
                                } else {
                                    v1477 = v1474;
                                }
                                static_array<int,2> v1478;
                                int v1480;
                                v1480 = 0;
                                while (while_method_0(v1480)){
                                    int v1482;
                                    v1482 = v1308[v1480];
                                    bool v1484;
                                    v1484 = v1313 == v1480;
                                    int v1485;
                                    if (v1484){
                                        v1485 = v1477;
                                    } else {
                                        v1485 = v1482;
                                    }
                                    v1478[v1480] = v1485;
                                    v1480 += 1 ;
                                }
                                static_array<int,2> v1486;
                                int v1488;
                                v1488 = 0;
                                while (while_method_0(v1488)){
                                    int v1490;
                                    v1490 = v1432[v1488];
                                    int v1492;
                                    v1492 = v1478[v1488];
                                    int v1494;
                                    v1494 = v1490 - v1492;
                                    v1486[v1488] = v1494;
                                    v1488 += 1 ;
                                }
                                bool v1495;
                                v1495 = v1471 >= v1306;
                                int v1496;
                                if (v1495){
                                    v1496 = v1471;
                                } else {
                                    v1496 = v1306;
                                }
                                int v1497;
                                v1497 = v1309 + 1;
                                v1502 = try_round_36(v1496, v1307, v1478, v1497, v1486, v1311);
                                break;
                            }
                            case 1: { // A_Call
                                static_array<int,2> v1316;
                                int v1318;
                                v1318 = 0;
                                while (while_method_0(v1318)){
                                    int v1320;
                                    v1320 = v1310[v1318];
                                    int v1322;
                                    v1322 = v1308[v1318];
                                    int v1324;
                                    v1324 = v1320 + v1322;
                                    v1316[v1318] = v1324;
                                    v1318 += 1 ;
                                }
                                int v1325;
                                v1325 = v1308[0];
                                int v1327; int v1328;
                                Tuple4 tmp55 = Tuple4{1, v1325};
                                v1327 = tmp55.v0; v1328 = tmp55.v1;
                                while (while_method_0(v1327)){
                                    int v1330;
                                    v1330 = v1308[v1327];
                                    bool v1332;
                                    v1332 = v1328 >= v1330;
                                    int v1333;
                                    if (v1332){
                                        v1333 = v1328;
                                    } else {
                                        v1333 = v1330;
                                    }
                                    v1328 = v1333;
                                    v1327 += 1 ;
                                }
                                int v1334;
                                v1334 = v1316[v1313];
                                bool v1336;
                                v1336 = v1328 < v1334;
                                int v1337;
                                if (v1336){
                                    v1337 = v1328;
                                } else {
                                    v1337 = v1334;
                                }
                                static_array<int,2> v1338;
                                int v1340;
                                v1340 = 0;
                                while (while_method_0(v1340)){
                                    int v1342;
                                    v1342 = v1308[v1340];
                                    bool v1344;
                                    v1344 = v1313 == v1340;
                                    int v1345;
                                    if (v1344){
                                        v1345 = v1337;
                                    } else {
                                        v1345 = v1342;
                                    }
                                    v1338[v1340] = v1345;
                                    v1340 += 1 ;
                                }
                                static_array<int,2> v1346;
                                int v1348;
                                v1348 = 0;
                                while (while_method_0(v1348)){
                                    int v1350;
                                    v1350 = v1316[v1348];
                                    int v1352;
                                    v1352 = v1338[v1348];
                                    int v1354;
                                    v1354 = v1350 - v1352;
                                    v1346[v1348] = v1354;
                                    v1348 += 1 ;
                                }
                                bool v1355;
                                v1355 = v1313 < 2;
                                if (v1355){
                                    int v1356;
                                    v1356 = v1309 + 1;
                                    v1502 = try_round_36(v1306, v1307, v1338, v1356, v1346, v1311);
                                } else {
                                    v1502 = go_next_street_38(v1306, v1307, v1338, v1309, v1346, v1311);
                                }
                                break;
                            }
                            case 2: { // A_Fold
                                v1502 = Union4{Union4_1{v1306, v1307, v1308, v1309, v1310, v1311}};
                                break;
                            }
                            case 3: { // A_Raise
                                int v1360 = v1312.case3.v0;
                                bool v1361;
                                v1361 = v1306 <= v1360;
                                bool v1362;
                                v1362 = v1361 == false;
                                if (v1362){
                                    assert("The raise amount must match the minimum." && v1361);
                                } else {
                                }
                                static_array<int,2> v1364;
                                int v1366;
                                v1366 = 0;
                                while (while_method_0(v1366)){
                                    int v1368;
                                    v1368 = v1310[v1366];
                                    int v1370;
                                    v1370 = v1308[v1366];
                                    int v1372;
                                    v1372 = v1368 + v1370;
                                    v1364[v1366] = v1372;
                                    v1366 += 1 ;
                                }
                                int v1373;
                                v1373 = v1308[0];
                                int v1375; int v1376;
                                Tuple4 tmp56 = Tuple4{1, v1373};
                                v1375 = tmp56.v0; v1376 = tmp56.v1;
                                while (while_method_0(v1375)){
                                    int v1378;
                                    v1378 = v1308[v1375];
                                    bool v1380;
                                    v1380 = v1376 >= v1378;
                                    int v1381;
                                    if (v1380){
                                        v1381 = v1376;
                                    } else {
                                        v1381 = v1378;
                                    }
                                    v1376 = v1381;
                                    v1375 += 1 ;
                                }
                                int v1382;
                                v1382 = v1364[v1313];
                                bool v1384;
                                v1384 = v1376 < v1382;
                                int v1385;
                                if (v1384){
                                    v1385 = v1376;
                                } else {
                                    v1385 = v1382;
                                }
                                static_array<int,2> v1386;
                                int v1388;
                                v1388 = 0;
                                while (while_method_0(v1388)){
                                    int v1390;
                                    v1390 = v1308[v1388];
                                    bool v1392;
                                    v1392 = v1313 == v1388;
                                    int v1393;
                                    if (v1392){
                                        v1393 = v1385;
                                    } else {
                                        v1393 = v1390;
                                    }
                                    v1386[v1388] = v1393;
                                    v1388 += 1 ;
                                }
                                static_array<int,2> v1394;
                                int v1396;
                                v1396 = 0;
                                while (while_method_0(v1396)){
                                    int v1398;
                                    v1398 = v1364[v1396];
                                    int v1400;
                                    v1400 = v1386[v1396];
                                    int v1402;
                                    v1402 = v1398 - v1400;
                                    v1394[v1396] = v1402;
                                    v1396 += 1 ;
                                }
                                int v1403;
                                v1403 = v1394[v1313];
                                bool v1405;
                                v1405 = v1360 < v1403;
                                bool v1406;
                                v1406 = v1405 == false;
                                if (v1406){
                                    assert("The raise amount must be less than the stack size after calling." && v1405);
                                } else {
                                }
                                int v1408;
                                v1408 = v1376 + v1360;
                                int v1409;
                                v1409 = v1364[v1313];
                                bool v1411;
                                v1411 = v1408 < v1409;
                                int v1412;
                                if (v1411){
                                    v1412 = v1408;
                                } else {
                                    v1412 = v1409;
                                }
                                static_array<int,2> v1413;
                                int v1415;
                                v1415 = 0;
                                while (while_method_0(v1415)){
                                    int v1417;
                                    v1417 = v1308[v1415];
                                    bool v1419;
                                    v1419 = v1313 == v1415;
                                    int v1420;
                                    if (v1419){
                                        v1420 = v1412;
                                    } else {
                                        v1420 = v1417;
                                    }
                                    v1413[v1415] = v1420;
                                    v1415 += 1 ;
                                }
                                static_array<int,2> v1421;
                                int v1423;
                                v1423 = 0;
                                while (while_method_0(v1423)){
                                    int v1425;
                                    v1425 = v1364[v1423];
                                    int v1427;
                                    v1427 = v1413[v1423];
                                    int v1429;
                                    v1429 = v1425 - v1427;
                                    v1421[v1423] = v1429;
                                    v1423 += 1 ;
                                }
                                int v1430;
                                v1430 = v1309 + 1;
                                v1502 = try_round_36(v1360, v1307, v1413, v1430, v1421, v1311);
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        v1621 = Union3{Union3_1{v1502}};
                        break;
                    }
                    case 6: { // G_Showdown
                        int v27 = v11.case6.v0; static_array<static_array<unsigned char,2>,2> v28 = v11.case6.v1; static_array<int,2> v29 = v11.case6.v2; int v30 = v11.case6.v3; static_array<int,2> v31 = v11.case6.v4; Union5 v32 = v11.case6.v5;
                        static_array<unsigned char,5> v35;
                        switch (v32.tag) {
                            case 2: { // River
                                static_array<unsigned char,5> v33 = v32.case2.v0;
                                v35 = v33;
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in showdown.");
                                __trap();
                            }
                        }
                        static_array<unsigned char,2> v36;
                        v36 = v28[0];
                        static_array<unsigned char,7> v38;
                        int v40;
                        v40 = 0;
                        while (while_method_0(v40)){
                            unsigned char v42;
                            v42 = v36[v40];
                            v38[v40] = v42;
                            v40 += 1 ;
                        }
                        int v44;
                        v44 = 0;
                        while (while_method_2(v44)){
                            unsigned char v46;
                            v46 = v35[v44];
                            int v48;
                            v48 = 2 + v44;
                            v38[v48] = v46;
                            v44 += 1 ;
                        }
                        static_array<unsigned char,5> v49; char v50;
                        Tuple0 tmp81 = score_51(v38);
                        v49 = tmp81.v0; v50 = tmp81.v1;
                        static_array<unsigned char,2> v51;
                        v51 = v28[1];
                        static_array<unsigned char,7> v53;
                        int v55;
                        v55 = 0;
                        while (while_method_0(v55)){
                            unsigned char v57;
                            v57 = v51[v55];
                            v53[v55] = v57;
                            v55 += 1 ;
                        }
                        int v59;
                        v59 = 0;
                        while (while_method_2(v59)){
                            unsigned char v61;
                            v61 = v35[v59];
                            int v63;
                            v63 = 2 + v59;
                            v53[v63] = v61;
                            v59 += 1 ;
                        }
                        static_array<unsigned char,5> v64; char v65;
                        Tuple0 tmp82 = score_51(v53);
                        v64 = tmp82.v0; v65 = tmp82.v1;
                        int v66;
                        v66 = v30 % 2;
                        int v67;
                        v67 = v29[v66];
                        bool v69;
                        v69 = v50 < v65;
                        Union11 v75;
                        if (v69){
                            v75 = Union11{Union11_2{}};
                        } else {
                            bool v71;
                            v71 = v50 > v65;
                            if (v71){
                                v75 = Union11{Union11_1{}};
                            } else {
                                v75 = Union11{Union11_0{}};
                            }
                        }
                        Union11 v94;
                        switch (v75.tag) {
                            case 0: { // Eq
                                Union11 v76;
                                v76 = Union11{Union11_0{}};
                                int v77;
                                v77 = 0;
                                while (while_method_2(v77)){
                                    unsigned char v79;
                                    v79 = v49[v77];
                                    unsigned char v81;
                                    v81 = v64[v77];
                                    unsigned char v83;
                                    v83 = v79 / 4u;
                                    unsigned char v84;
                                    v84 = v81 / 4u;
                                    bool v85;
                                    v85 = v83 < v84;
                                    Union11 v91;
                                    if (v85){
                                        v91 = Union11{Union11_2{}};
                                    } else {
                                        bool v87;
                                        v87 = v83 > v84;
                                        if (v87){
                                            v91 = Union11{Union11_1{}};
                                        } else {
                                            v91 = Union11{Union11_0{}};
                                        }
                                    }
                                    bool v92;
                                    switch (v91.tag) {
                                        case 0: { // Eq
                                            v92 = true;
                                            break;
                                        }
                                        default: {
                                            v92 = false;
                                        }
                                    }
                                    bool v93;
                                    v93 = v92 == false;
                                    if (v93){
                                        v76 = v91;
                                        break;
                                    } else {
                                    }
                                    v77 += 1 ;
                                }
                                v94 = v76;
                                break;
                            }
                            default: {
                                v94 = v75;
                            }
                        }
                        int v99; int v100;
                        switch (v94.tag) {
                            case 0: { // Eq
                                v99 = 0; v100 = -1;
                                break;
                            }
                            case 1: { // Gt
                                v99 = v67; v100 = 0;
                                break;
                            }
                            case 2: { // Lt
                                v99 = v67; v100 = 1;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        static_array<Tuple0,2> v101;
                        v101[0] = Tuple0{v49, v50};
                        v101[1] = Tuple0{v64, v65};
                        Union6 v103;
                        v103 = Union6{Union6_4{v99, v101, v100}};
                        v6.push(v103);
                        Union7 v104;
                        v104 = Union7{Union7_1{v27, v28, v29, v30, v31, v32}};
                        v4.v5 = v104;
                        Union3 v105;
                        v105 = Union3{Union3_0{}};
                        v4.v1 = v105;
                        v1621 = Union3{Union3_0{}};
                        break;
                    }
                    case 7: { // G_Turn
                        int v1523 = v11.case7.v0; static_array<static_array<unsigned char,2>,2> v1524 = v11.case7.v1; static_array<int,2> v1525 = v11.case7.v2; int v1526 = v11.case7.v3; static_array<int,2> v1527 = v11.case7.v4; Union5 v1528 = v11.case7.v5;
                        curandStatePhilox4_32_10_t & v1529 = v4.v4;
                        curandStatePhilox4_32_10_t & v1530 = v1529;
                        static_array<unsigned char,1> v1531; unsigned long long v1532;
                        Tuple12 tmp83 = draw_cards_40(v1530, v7);
                        v1531 = tmp83.v0; v1532 = tmp83.v1;
                        v4.v0 = v1532;
                        static_array_list<unsigned char,5> v1533;
                        v1533 = get_community_cards_41(v1528, v1531);
                        Union6 v1534;
                        v1534 = Union6{Union6_0{v1533}};
                        v6.push(v1534);
                        Union5 v1549;
                        switch (v1528.tag) {
                            case 0: { // Flop
                                static_array<unsigned char,3> v1535 = v1528.case0.v0;
                                static_array<unsigned char,4> v1536;
                                int v1538;
                                v1538 = 0;
                                while (while_method_1(v1538)){
                                    unsigned char v1540;
                                    v1540 = v1535[v1538];
                                    v1536[v1538] = v1540;
                                    v1538 += 1 ;
                                }
                                int v1542;
                                v1542 = 0;
                                while (while_method_6(v1542)){
                                    unsigned char v1544;
                                    v1544 = v1531[v1542];
                                    int v1546;
                                    v1546 = 3 + v1542;
                                    v1536[v1546] = v1544;
                                    v1542 += 1 ;
                                }
                                v1549 = Union5{Union5_3{v1536}};
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in turn.");
                                __trap();
                            }
                        }
                        int v1550;
                        v1550 = 2;
                        int v1551;
                        v1551 = 0;
                        Union4 v1552;
                        v1552 = try_round_36(v1550, v1524, v1525, v1551, v1527, v1549);
                        v1621 = Union3{Union3_1{v1552}};
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
        v9 = v1621;
    }
    return ;
}
__device__ void f_53(unsigned char * v0, unsigned long long v1){
    unsigned long long * v2;
    v2 = (unsigned long long *)(v0+0ull);
    v2[0] = v1;
    return ;
}
__device__ void f_54(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+8ull);
    v2[0] = v1;
    return ;
}
__device__ void f_55(unsigned char * v0){
    return ;
}
__device__ void f_57(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+0ull);
    v2[0] = v1;
    return ;
}
__device__ void f_61(unsigned char * v0, unsigned char v1){
    unsigned char * v2;
    v2 = (unsigned char *)(v0+0ull);
    v2[0] = v1;
    return ;
}
__device__ void f_60(unsigned char * v0, unsigned char v1){
    return f_61(v0, v1);
}
__device__ void f_59(unsigned char * v0, static_array<unsigned char,2> v1){
    int v2;
    v2 = 0;
    while (while_method_0(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned char * v5;
        v5 = (unsigned char *)(v0+v4);
        unsigned char v7;
        v7 = v1[v2];
        f_60(v5, v7);
        v2 += 1 ;
    }
    return ;
}
__device__ void f_62(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+28ull);
    v2[0] = v1;
    return ;
}
__device__ void f_63(unsigned char * v0, static_array<unsigned char,3> v1){
    int v2;
    v2 = 0;
    while (while_method_1(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned char * v5;
        v5 = (unsigned char *)(v0+v4);
        unsigned char v7;
        v7 = v1[v2];
        f_60(v5, v7);
        v2 += 1 ;
    }
    return ;
}
__device__ void f_64(unsigned char * v0, static_array<unsigned char,5> v1){
    int v2;
    v2 = 0;
    while (while_method_2(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned char * v5;
        v5 = (unsigned char *)(v0+v4);
        unsigned char v7;
        v7 = v1[v2];
        f_60(v5, v7);
        v2 += 1 ;
    }
    return ;
}
__device__ void f_65(unsigned char * v0, static_array<unsigned char,4> v1){
    int v2;
    v2 = 0;
    while (while_method_3(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned char * v5;
        v5 = (unsigned char *)(v0+v4);
        unsigned char v7;
        v7 = v1[v2];
        f_60(v5, v7);
        v2 += 1 ;
    }
    return ;
}
__device__ void f_58(unsigned char * v0, int v1, static_array<static_array<unsigned char,2>,2> v2, static_array<int,2> v3, int v4, static_array<int,2> v5, Union5 v6){
    int * v7;
    v7 = (int *)(v0+0ull);
    v7[0] = v1;
    int v9;
    v9 = 0;
    while (while_method_0(v9)){
        unsigned long long v11;
        v11 = (unsigned long long)v9;
        unsigned long long v12;
        v12 = v11 * 2ull;
        unsigned long long v13;
        v13 = 4ull + v12;
        unsigned char * v14;
        v14 = (unsigned char *)(v0+v13);
        static_array<unsigned char,2> v16;
        v16 = v2[v9];
        f_59(v14, v16);
        v9 += 1 ;
    }
    int v18;
    v18 = 0;
    while (while_method_0(v18)){
        unsigned long long v20;
        v20 = (unsigned long long)v18;
        unsigned long long v21;
        v21 = v20 * 4ull;
        unsigned long long v22;
        v22 = 8ull + v21;
        unsigned char * v23;
        v23 = (unsigned char *)(v0+v22);
        int v25;
        v25 = v3[v18];
        f_57(v23, v25);
        v18 += 1 ;
    }
    int * v27;
    v27 = (int *)(v0+16ull);
    v27[0] = v4;
    int v29;
    v29 = 0;
    while (while_method_0(v29)){
        unsigned long long v31;
        v31 = (unsigned long long)v29;
        unsigned long long v32;
        v32 = v31 * 4ull;
        unsigned long long v33;
        v33 = 20ull + v32;
        unsigned char * v34;
        v34 = (unsigned char *)(v0+v33);
        int v36;
        v36 = v5[v29];
        f_57(v34, v36);
        v29 += 1 ;
    }
    int v38;
    v38 = v6.tag;
    f_62(v0, v38);
    unsigned char * v39;
    v39 = (unsigned char *)(v0+32ull);
    switch (v6.tag) {
        case 0: { // Flop
            static_array<unsigned char,3> v41 = v6.case0.v0;
            return f_63(v39, v41);
            break;
        }
        case 1: { // Preflop
            return f_55(v39);
            break;
        }
        case 2: { // River
            static_array<unsigned char,5> v42 = v6.case2.v0;
            return f_64(v39, v42);
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4> v43 = v6.case3.v0;
            return f_65(v39, v43);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_67(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+40ull);
    v2[0] = v1;
    return ;
}
__device__ void f_66(unsigned char * v0, int v1, static_array<static_array<unsigned char,2>,2> v2, static_array<int,2> v3, int v4, static_array<int,2> v5, Union5 v6, Union1 v7){
    int * v8;
    v8 = (int *)(v0+0ull);
    v8[0] = v1;
    int v10;
    v10 = 0;
    while (while_method_0(v10)){
        unsigned long long v12;
        v12 = (unsigned long long)v10;
        unsigned long long v13;
        v13 = v12 * 2ull;
        unsigned long long v14;
        v14 = 4ull + v13;
        unsigned char * v15;
        v15 = (unsigned char *)(v0+v14);
        static_array<unsigned char,2> v17;
        v17 = v2[v10];
        f_59(v15, v17);
        v10 += 1 ;
    }
    int v19;
    v19 = 0;
    while (while_method_0(v19)){
        unsigned long long v21;
        v21 = (unsigned long long)v19;
        unsigned long long v22;
        v22 = v21 * 4ull;
        unsigned long long v23;
        v23 = 8ull + v22;
        unsigned char * v24;
        v24 = (unsigned char *)(v0+v23);
        int v26;
        v26 = v3[v19];
        f_57(v24, v26);
        v19 += 1 ;
    }
    int * v28;
    v28 = (int *)(v0+16ull);
    v28[0] = v4;
    int v30;
    v30 = 0;
    while (while_method_0(v30)){
        unsigned long long v32;
        v32 = (unsigned long long)v30;
        unsigned long long v33;
        v33 = v32 * 4ull;
        unsigned long long v34;
        v34 = 20ull + v33;
        unsigned char * v35;
        v35 = (unsigned char *)(v0+v34);
        int v37;
        v37 = v5[v30];
        f_57(v35, v37);
        v30 += 1 ;
    }
    int v39;
    v39 = v6.tag;
    f_62(v0, v39);
    unsigned char * v40;
    v40 = (unsigned char *)(v0+32ull);
    switch (v6.tag) {
        case 0: { // Flop
            static_array<unsigned char,3> v42 = v6.case0.v0;
            f_63(v40, v42);
            break;
        }
        case 1: { // Preflop
            f_55(v40);
            break;
        }
        case 2: { // River
            static_array<unsigned char,5> v43 = v6.case2.v0;
            f_64(v40, v43);
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4> v44 = v6.case3.v0;
            f_65(v40, v44);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v45;
    v45 = v7.tag;
    f_67(v0, v45);
    unsigned char * v46;
    v46 = (unsigned char *)(v0+44ull);
    switch (v7.tag) {
        case 0: { // A_All_In
            return f_55(v46);
            break;
        }
        case 1: { // A_Call
            return f_55(v46);
            break;
        }
        case 2: { // A_Fold
            return f_55(v46);
            break;
        }
        case 3: { // A_Raise
            int v48 = v7.case3.v0;
            return f_57(v46, v48);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_56(unsigned char * v0, Union4 v1){
    int v2;
    v2 = v1.tag;
    f_57(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // G_Flop
            int v5 = v1.case0.v0; static_array<static_array<unsigned char,2>,2> v6 = v1.case0.v1; static_array<int,2> v7 = v1.case0.v2; int v8 = v1.case0.v3; static_array<int,2> v9 = v1.case0.v4; Union5 v10 = v1.case0.v5;
            return f_58(v3, v5, v6, v7, v8, v9, v10);
            break;
        }
        case 1: { // G_Fold
            int v11 = v1.case1.v0; static_array<static_array<unsigned char,2>,2> v12 = v1.case1.v1; static_array<int,2> v13 = v1.case1.v2; int v14 = v1.case1.v3; static_array<int,2> v15 = v1.case1.v4; Union5 v16 = v1.case1.v5;
            return f_58(v3, v11, v12, v13, v14, v15, v16);
            break;
        }
        case 2: { // G_Preflop
            return f_55(v3);
            break;
        }
        case 3: { // G_River
            int v17 = v1.case3.v0; static_array<static_array<unsigned char,2>,2> v18 = v1.case3.v1; static_array<int,2> v19 = v1.case3.v2; int v20 = v1.case3.v3; static_array<int,2> v21 = v1.case3.v4; Union5 v22 = v1.case3.v5;
            return f_58(v3, v17, v18, v19, v20, v21, v22);
            break;
        }
        case 4: { // G_Round
            int v23 = v1.case4.v0; static_array<static_array<unsigned char,2>,2> v24 = v1.case4.v1; static_array<int,2> v25 = v1.case4.v2; int v26 = v1.case4.v3; static_array<int,2> v27 = v1.case4.v4; Union5 v28 = v1.case4.v5;
            return f_58(v3, v23, v24, v25, v26, v27, v28);
            break;
        }
        case 5: { // G_Round'
            int v29 = v1.case5.v0; static_array<static_array<unsigned char,2>,2> v30 = v1.case5.v1; static_array<int,2> v31 = v1.case5.v2; int v32 = v1.case5.v3; static_array<int,2> v33 = v1.case5.v4; Union5 v34 = v1.case5.v5; Union1 v35 = v1.case5.v6;
            return f_66(v3, v29, v30, v31, v32, v33, v34, v35);
            break;
        }
        case 6: { // G_Showdown
            int v36 = v1.case6.v0; static_array<static_array<unsigned char,2>,2> v37 = v1.case6.v1; static_array<int,2> v38 = v1.case6.v2; int v39 = v1.case6.v3; static_array<int,2> v40 = v1.case6.v4; Union5 v41 = v1.case6.v5;
            return f_58(v3, v36, v37, v38, v39, v40, v41);
            break;
        }
        case 7: { // G_Turn
            int v42 = v1.case7.v0; static_array<static_array<unsigned char,2>,2> v43 = v1.case7.v1; static_array<int,2> v44 = v1.case7.v2; int v45 = v1.case7.v3; static_array<int,2> v46 = v1.case7.v4; Union5 v47 = v1.case7.v5;
            return f_58(v3, v42, v43, v44, v45, v46, v47);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_68(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+80ull);
    v2[0] = v1;
    return ;
}
__device__ void f_70(unsigned char * v0, static_array_list<unsigned char,5> v1){
    int v2;
    v2 = v1.length;
    f_57(v0, v2);
    int v3;
    v3 = v1.length;
    int v4;
    v4 = 0;
    while (while_method_4(v3, v4)){
        unsigned long long v6;
        v6 = (unsigned long long)v4;
        unsigned long long v7;
        v7 = 4ull + v6;
        unsigned char * v8;
        v8 = (unsigned char *)(v0+v7);
        unsigned char v10;
        v10 = v1[v4];
        f_60(v8, v10);
        v4 += 1 ;
    }
    return ;
}
__device__ void f_71(unsigned char * v0, int v1, int v2){
    int * v3;
    v3 = (int *)(v0+0ull);
    v3[0] = v1;
    int * v5;
    v5 = (int *)(v0+4ull);
    v5[0] = v2;
    return ;
}
__device__ void f_73(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+4ull);
    v2[0] = v1;
    return ;
}
__device__ void f_72(unsigned char * v0, int v1, Union1 v2){
    int * v3;
    v3 = (int *)(v0+0ull);
    v3[0] = v1;
    int v5;
    v5 = v2.tag;
    f_73(v0, v5);
    unsigned char * v6;
    v6 = (unsigned char *)(v0+8ull);
    switch (v2.tag) {
        case 0: { // A_All_In
            return f_55(v6);
            break;
        }
        case 1: { // A_Call
            return f_55(v6);
            break;
        }
        case 2: { // A_Fold
            return f_55(v6);
            break;
        }
        case 3: { // A_Raise
            int v8 = v2.case3.v0;
            return f_57(v6, v8);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_74(unsigned char * v0, int v1, static_array<unsigned char,2> v2){
    int * v3;
    v3 = (int *)(v0+0ull);
    v3[0] = v1;
    int v5;
    v5 = 0;
    while (while_method_0(v5)){
        unsigned long long v7;
        v7 = (unsigned long long)v5;
        unsigned long long v8;
        v8 = 4ull + v7;
        unsigned char * v9;
        v9 = (unsigned char *)(v0+v8);
        unsigned char v11;
        v11 = v2[v5];
        f_60(v9, v11);
        v5 += 1 ;
    }
    return ;
}
__device__ void f_77(unsigned char * v0, static_array<unsigned char,5> v1, char v2){
    int v3;
    v3 = 0;
    while (while_method_2(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        unsigned char v8;
        v8 = v1[v3];
        f_60(v6, v8);
        v3 += 1 ;
    }
    char * v10;
    v10 = (char *)(v0+5ull);
    v10[0] = v2;
    return ;
}
__device__ void f_76(unsigned char * v0, static_array<unsigned char,5> v1, char v2){
    return f_77(v0, v1, v2);
}
__device__ void f_75(unsigned char * v0, int v1, static_array<Tuple0,2> v2, int v3){
    int * v4;
    v4 = (int *)(v0+0ull);
    v4[0] = v1;
    int v6;
    v6 = 0;
    while (while_method_0(v6)){
        unsigned long long v8;
        v8 = (unsigned long long)v6;
        unsigned long long v9;
        v9 = v8 * 8ull;
        unsigned long long v10;
        v10 = 8ull + v9;
        unsigned char * v11;
        v11 = (unsigned char *)(v0+v10);
        static_array<unsigned char,5> v13; char v14;
        Tuple0 tmp84 = v2[v6];
        v13 = tmp84.v0; v14 = tmp84.v1;
        f_76(v11, v13, v14);
        v6 += 1 ;
    }
    int * v17;
    v17 = (int *)(v0+24ull);
    v17[0] = v3;
    return ;
}
__device__ void f_69(unsigned char * v0, Union6 v1){
    int v2;
    v2 = v1.tag;
    f_57(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // CommunityCardsAre
            static_array_list<unsigned char,5> v5 = v1.case0.v0;
            return f_70(v3, v5);
            break;
        }
        case 1: { // Fold
            int v6 = v1.case1.v0; int v7 = v1.case1.v1;
            return f_71(v3, v6, v7);
            break;
        }
        case 2: { // PlayerAction
            int v8 = v1.case2.v0; Union1 v9 = v1.case2.v1;
            return f_72(v3, v8, v9);
            break;
        }
        case 3: { // PlayerGotCards
            int v10 = v1.case3.v0; static_array<unsigned char,2> v11 = v1.case3.v1;
            return f_74(v3, v10, v11);
            break;
        }
        case 4: { // Showdown
            int v12 = v1.case4.v0; static_array<Tuple0,2> v13 = v1.case4.v1; int v14 = v1.case4.v2;
            return f_75(v3, v12, v13, v14);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_78(unsigned char * v0, Union2 v1){
    int v2;
    v2 = v1.tag;
    f_57(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // Computer
            return f_55(v3);
            break;
        }
        case 1: { // Human
            return f_55(v3);
            break;
        }
        case 2: { // Random
            return f_55(v3);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_79(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+6248ull);
    v2[0] = v1;
    return ;
}
__device__ void f_52(unsigned char * v0, unsigned long long v1, Union3 v2, dynamic_array_list<Union6,128> v3, static_array<Union2,2> v4, Union7 v5){
    f_53(v0, v1);
    int v6;
    v6 = v2.tag;
    f_54(v0, v6);
    unsigned char * v7;
    v7 = (unsigned char *)(v0+16ull);
    switch (v2.tag) {
        case 0: { // None
            f_55(v7);
            break;
        }
        case 1: { // Some
            Union4 v9 = v2.case1.v0;
            f_56(v7, v9);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v10;
    v10 = v3.length_();
    f_68(v0, v10);
    int v11;
    v11 = v3.length_();
    int v12;
    v12 = 0;
    while (while_method_4(v11, v12)){
        unsigned long long v14;
        v14 = (unsigned long long)v12;
        unsigned long long v15;
        v15 = v14 * 48ull;
        unsigned long long v16;
        v16 = 96ull + v15;
        unsigned char * v17;
        v17 = (unsigned char *)(v0+v16);
        Union6 v19;
        v19 = v3[v12];
        f_69(v17, v19);
        v12 += 1 ;
    }
    int v21;
    v21 = 0;
    while (while_method_0(v21)){
        unsigned long long v23;
        v23 = (unsigned long long)v21;
        unsigned long long v24;
        v24 = v23 * 4ull;
        unsigned long long v25;
        v25 = 6240ull + v24;
        unsigned char * v26;
        v26 = (unsigned char *)(v0+v25);
        Union2 v28;
        v28 = v4[v21];
        f_78(v26, v28);
        v21 += 1 ;
    }
    int v30;
    v30 = v5.tag;
    f_79(v0, v30);
    unsigned char * v31;
    v31 = (unsigned char *)(v0+6256ull);
    switch (v5.tag) {
        case 0: { // GameNotStarted
            return f_55(v31);
            break;
        }
        case 1: { // GameOver
            int v33 = v5.case1.v0; static_array<static_array<unsigned char,2>,2> v34 = v5.case1.v1; static_array<int,2> v35 = v5.case1.v2; int v36 = v5.case1.v3; static_array<int,2> v37 = v5.case1.v4; Union5 v38 = v5.case1.v5;
            return f_58(v31, v33, v34, v35, v36, v37, v38);
            break;
        }
        case 2: { // WaitingForActionFromPlayerId
            int v39 = v5.case2.v0; static_array<static_array<unsigned char,2>,2> v40 = v5.case2.v1; static_array<int,2> v41 = v5.case2.v2; int v42 = v5.case2.v3; static_array<int,2> v43 = v5.case2.v4; Union5 v44 = v5.case2.v5;
            return f_58(v31, v39, v40, v41, v42, v43, v44);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1, unsigned char * v2, unsigned long long v3, unsigned char * v4, unsigned long long v5) {
    Union0 v6;
    v6 = f_0(v1);
    unsigned long long v7; Union3 v8; dynamic_array_list<Union6,128> v9; static_array<Union2,2> v10; Union7 v11;
    Tuple1 tmp15 = f_6(v0);
    v7 = tmp15.v0; v8 = tmp15.v1; v9 = tmp15.v2; v10 = tmp15.v3; v11 = tmp15.v4;
    unsigned long long v12;
    v12 = clock64();
    int v13;
    v13 = threadIdx.x;
    int v14;
    v14 = blockIdx.x;
    int v15;
    v15 = v14 * 256;
    int v16;
    v16 = v13 + v15;
    unsigned long long v17;
    v17 = (unsigned long long)v16;
    curandStatePhilox4_32_10_t v18;
    curand_init(v12,v17,0ull,&v18);
    curandStatePhilox4_32_10_t & v19 = v18;
    StackMut0 v20{v7, v8, v9, v10, v19, v11};
    Union3 v53;
    switch (v6.tag) {
        case 0: { // ActionSelected
            Union1 v35 = v6.case0.v0;
            Union3 & v36 = v20.v1;
            switch (v36.tag) {
                case 0: { // None
                    printf("%s\n", "The game hasn't been started in ActionSelected.");
                    __trap();
                    break;
                }
                case 1: { // Some
                    Union4 v37 = v36.case1.v0;
                    switch (v37.tag) {
                        case 4: { // G_Round
                            int v38 = v37.case4.v0; static_array<static_array<unsigned char,2>,2> v39 = v37.case4.v1; static_array<int,2> v40 = v37.case4.v2; int v41 = v37.case4.v3; static_array<int,2> v42 = v37.case4.v4; Union5 v43 = v37.case4.v5;
                            Union4 v44;
                            v44 = Union4{Union4_5{v38, v39, v40, v41, v42, v43, v35}};
                            v53 = Union3{Union3_1{v44}};
                            break;
                        }
                        default: {
                            printf("%s\n", "Unexpected game node in ActionSelected.");
                            __trap();
                        }
                    }
                    break;
                }
                default: {
                    assert("Invalid tag." && false); __trap();
                }
            }
            break;
        }
        case 1: { // PlayerChanged
            static_array<Union2,2> v33 = v6.case1.v0;
            v20.v3 = v33;
            v53 = Union3{Union3_0{}};
            break;
        }
        case 2: { // StartGame
            static_array<Union2,2> v21;
            Union2 v23;
            v23 = Union2{Union2_0{}};
            v21[0] = v23;
            Union2 v25;
            v25 = Union2{Union2_1{}};
            v21[1] = v25;
            dynamic_array_list<Union6,128> v27{0};
            Union7 v29;
            v29 = Union7{Union7_0{}};
            v20.v5 = v29;
            v20.v3 = v21;
            Union3 v30;
            v30 = Union3{Union3_0{}};
            v20.v1 = v30;
            v20.v0 = 4503599627370495ull;
            v20.v2 = v27;
            Union4 v31;
            v31 = Union4{Union4_2{}};
            v53 = Union3{Union3_1{v31}};
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    switch (v53.tag) {
        case 0: { // None
            break;
        }
        case 1: { // Some
            Union4 v54 = v53.case1.v0;
            play_loop_31(v2, v3, v4, v5, v20, v54);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v55;
    v55 = threadIdx.x;
    int v56;
    v56 = blockIdx.x;
    int v57;
    v57 = v56 * 256;
    int v58;
    v58 = v55 + v57;
    bool v59;
    v59 = v58 == 0;
    if (v59){
        Union7 & v60 = v20.v5;
        static_array<Union2,2> & v61 = v20.v3;
        dynamic_array_list<Union6,128> & v62 = v20.v2;
        Union3 & v63 = v20.v1;
        unsigned long long & v64 = v20.v0;
        return f_52(v0, v64, v63, v62, v61, v60);
    } else {
        return ;
    }
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
US0 = Union[US0_0, US0_1, US0_2]
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
            case US0_0(_): # ActionSelected
                method78(v12, v2)
                v34 = cp.cuda.Device().attributes['MultiProcessorCount']
                v35 = v34 == 24
                del v34
                v36 = v35 == False
                if v36:
                    v37 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
                    assert v35, v37
                    del v37
                else:
                    pass
                del v35, v36
                v38 = 0
                v39 = raw_module.get_function(f"entry{v38}")
                del v38
                v39.max_dynamic_shared_size_bytes = 98304 
                print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
                v39((24,),(256,),(v13, v12, v8, v9, v10, v11),shared_mem=98304)
                del v39
            case US0_1(_): # PlayerChanged
                method78(v12, v2)
                v27 = cp.cuda.Device().attributes['MultiProcessorCount']
                v28 = v27 == 24
                del v27
                v29 = v28 == False
                if v29:
                    v30 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
                    assert v28, v30
                    del v30
                else:
                    pass
                del v28, v29
                v31 = 0
                v32 = raw_module.get_function(f"entry{v31}")
                del v31
                v32.max_dynamic_shared_size_bytes = 98304 
                print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
                v32((24,),(256,),(v13, v12, v8, v9, v10, v11),shared_mem=98304)
                del v32
            case US0_2(): # StartGame
                method78(v12, v2)
                v20 = cp.cuda.Device().attributes['MultiProcessorCount']
                v21 = v20 == 24
                del v20
                v22 = v21 == False
                if v22:
                    v23 = "The number of SMs per GPU at runtime must much that what is declared atop of corecuda.base. Make sure to use the correct constant so it can be propagated at compile time."
                    assert v21, v23
                    del v23
                else:
                    pass
                del v21, v22
                v24 = 0
                v25 = raw_module.get_function(f"entry{v24}")
                del v24
                v25.max_dynamic_shared_size_bytes = 98304 
                print(f'DEBUG MODE. Threads per block, blocks per grid: {256}, {24}')
                v25((24,),(256,),(v13, v12, v8, v9, v10, v11),shared_mem=98304)
                del v25
            case t:
                raise Exception(f'Pattern matching miss. Got: {t}')
        del v2, v12
        cp.cuda.get_current_stream().synchronize()
        v40 = time.perf_counter()
        v43 = "{}"
        v44 = "The time it took to run the kernel (in seconds) is: "
        print(v43.format(v44),end="")
        del v43, v44
        v45 = v40 - v18
        del v18, v40
        v48 = "{:.6f}\n"
        print(v48.format(v45),end="")
        del v45, v48
        v49, v50, v51, v52, v53 = method81(v13)
        del v13
        return method109(v49, v50, v51, v52, v53, v8, v9, v10, v11, v19)
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
        v7 = dynamic_array_list(128)
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
        return method158(v35, v36, v7, v1, v37, v9, v38, v8, v39)
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
                del v2, v9
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
def method28(v0 : object) -> dynamic_array_list:
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
    v7 = dynamic_array_list(128)
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
def method27(v0 : object) -> Tuple[dynamic_array_list, static_array, US6]:
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
def method10(v0 : object) -> Tuple[u64, US3, dynamic_array_list, static_array, US6]:
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
def method9(v0 : object) -> Tuple[u64, US3, dynamic_array_list, static_array, US6, cp.ndarray, u64, cp.ndarray, u64]:
    v1 = v0["game"] # type: ignore
    v2, v3, v4, v5, v6 = method10(v1)
    del v1
    v7 = v0["neural"] # type: ignore
    del v0
    v8, v9, v10, v11 = method40(v7)
    del v7
    return v2, v3, v4, v5, v6, v8, v9, v10, v11
def method8(v0 : object) -> Tuple[u64, US3, dynamic_array_list, static_array, US6, cp.ndarray, u64, cp.ndarray, u64]:
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
        v8 = v1[v2]
        method55(v6, v8)
        del v6, v8
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
        v8 = v1[v2]
        method55(v6, v8)
        del v6, v8
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
        v8 = v1[v2]
        method55(v6, v8)
        del v6, v8
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
        v8 = v1[v2]
        method55(v6, v8)
        del v6, v8
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
        v17 = v2[v9]
        method54(v15, v17)
        del v15, v17
        v9 += 1 
    del v2, v9
    v18 = 0
    while method53(v18):
        v20 = u64(v18)
        v21 = v20 * 4
        del v20
        v22 = 8 + v21
        del v21
        v24 = v0[v22:].view(cp.uint8)
        del v22
        v26 = v3[v18]
        method51(v24, v26)
        del v24, v26
        v18 += 1 
    del v3, v18
    v28 = v0[16:].view(cp.int32)
    v28[0] = v4
    del v4, v28
    v29 = 0
    while method53(v29):
        v31 = u64(v29)
        v32 = v31 * 4
        del v31
        v33 = 20 + v32
        del v32
        v35 = v0[v33:].view(cp.uint8)
        del v33
        v37 = v5[v29]
        method51(v35, v37)
        del v35, v37
        v29 += 1 
    del v5, v29
    v38 = v6.tag
    method57(v0, v38)
    del v38
    v40 = v0[32:].view(cp.uint8)
    del v0
    match v6:
        case US5_0(v41): # Flop
            del v6
            return method58(v40, v41)
        case US5_1(): # Preflop
            del v6
            return method49(v40)
        case US5_2(v42): # River
            del v6
            return method60(v40, v42)
        case US5_3(v43): # Turn
            del v6
            return method62(v40, v43)
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
        v18 = v2[v10]
        method54(v16, v18)
        del v16, v18
        v10 += 1 
    del v2, v10
    v19 = 0
    while method53(v19):
        v21 = u64(v19)
        v22 = v21 * 4
        del v21
        v23 = 8 + v22
        del v22
        v25 = v0[v23:].view(cp.uint8)
        del v23
        v27 = v3[v19]
        method51(v25, v27)
        del v25, v27
        v19 += 1 
    del v3, v19
    v29 = v0[16:].view(cp.int32)
    v29[0] = v4
    del v4, v29
    v30 = 0
    while method53(v30):
        v32 = u64(v30)
        v33 = v32 * 4
        del v32
        v34 = 20 + v33
        del v33
        v36 = v0[v34:].view(cp.uint8)
        del v34
        v38 = v5[v30]
        method51(v36, v38)
        del v36, v38
        v30 += 1 
    del v5, v30
    v39 = v6.tag
    method57(v0, v39)
    del v39
    v41 = v0[32:].view(cp.uint8)
    match v6:
        case US5_0(v42): # Flop
            method58(v41, v42)
        case US5_1(): # Preflop
            method49(v41)
        case US5_2(v43): # River
            method60(v41, v43)
        case US5_3(v44): # Turn
            method62(v41, v44)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v6, v41
    v45 = v7.tag
    method65(v0, v45)
    del v45
    v47 = v0[44:].view(cp.uint8)
    del v0
    match v7:
        case US1_0(): # A_All_In
            del v7
            return method49(v47)
        case US1_1(): # A_Call
            del v7
            return method49(v47)
        case US1_2(): # A_Fold
            del v7
            return method49(v47)
        case US1_3(v48): # A_Raise
            del v7
            return method51(v47, v48)
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
        v12 = v2[v5]
        method55(v10, v12)
        del v10, v12
        v5 += 1 
    del v0, v2, v5
    return 
def method75(v0 : cp.ndarray, v1 : static_array, v2 : i8) -> None:
    v3 = 0
    while method61(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v9 = v1[v3]
        method55(v7, v9)
        del v7, v9
        v3 += 1 
    del v1, v3
    v11 = v0[5:].view(cp.int8)
    del v0
    v11[0] = v2
    del v2, v11
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
        v15, v16 = v2[v6]
        method74(v12, v15, v16)
        del v12, v15, v16
        v6 += 1 
    del v2, v6
    v18 = v0[24:].view(cp.int32)
    del v0
    v18[0] = v3
    del v3, v18
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
def method46(v0 : cp.ndarray, v1 : u64, v2 : US3, v3 : dynamic_array_list, v4 : static_array, v5 : US6) -> None:
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
    v10 = v3.length_()
    method66(v0, v10)
    del v10
    v11 = v3.length_()
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
        v29 = v4[v21]
        method76(v27, v29)
        del v27, v29
        v21 += 1 
    del v4, v21
    v30 = v5.tag
    method77(v0, v30)
    del v30
    v32 = v0[6256:].view(cp.uint8)
    del v0
    match v5:
        case US6_0(): # GameNotStarted
            del v5
            return method49(v32)
        case US6_1(v33, v34, v35, v36, v37, v38): # GameOver
            del v5
            return method52(v32, v33, v34, v35, v36, v37, v38)
        case US6_2(v39, v40, v41, v42, v43, v44): # WaitingForActionFromPlayerId
            del v5
            return method52(v32, v39, v40, v41, v42, v43, v44)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method79(v0 : cp.ndarray, v1 : US1) -> None:
    v2 = v1.tag
    method51(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US1_0(): # A_All_In
            del v1
            return method49(v4)
        case US1_1(): # A_Call
            del v1
            return method49(v4)
        case US1_2(): # A_Fold
            del v1
            return method49(v4)
        case US1_3(v5): # A_Raise
            del v1
            return method51(v4, v5)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method80(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method53(v2):
        v4 = u64(v2)
        v5 = v4 * 4
        del v4
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v9 = v1[v2]
        method76(v7, v9)
        del v7, v9
        v2 += 1 
    del v0, v1, v2
    return 
def method78(v0 : cp.ndarray, v1 : US0) -> None:
    v2 = v1.tag
    method51(v0, v2)
    del v2
    v4 = v0[8:].view(cp.uint8)
    del v0
    match v1:
        case US0_0(v5): # ActionSelected
            del v1
            return method79(v4, v5)
        case US0_1(v6): # PlayerChanged
            del v1
            return method80(v4, v6)
        case US0_2(): # StartGame
            del v1
            return method49(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method82(v0 : cp.ndarray) -> u64:
    v2 = v0[0:].view(cp.uint64)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method83(v0 : cp.ndarray) -> i32:
    v2 = v0[8:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method84(v0 : cp.ndarray) -> None:
    del v0
    return 
def method86(v0 : cp.ndarray) -> i32:
    v2 = v0[0:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method90(v0 : cp.ndarray) -> u8:
    v2 = v0[0:].view(cp.uint8)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method89(v0 : cp.ndarray) -> u8:
    v1 = method90(v0)
    del v0
    return v1
def method88(v0 : cp.ndarray) -> static_array:
    v2 = static_array(2)
    v3 = 0
    while method53(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v8 = method89(v7)
        del v7
        v2[v3] = v8
        del v8
        v3 += 1 
    del v0, v3
    return v2
def method91(v0 : cp.ndarray) -> i32:
    v2 = v0[28:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method92(v0 : cp.ndarray) -> static_array:
    v2 = static_array(3)
    v3 = 0
    while method59(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v8 = method89(v7)
        del v7
        v2[v3] = v8
        del v8
        v3 += 1 
    del v0, v3
    return v2
def method93(v0 : cp.ndarray) -> static_array:
    v2 = static_array(5)
    v3 = 0
    while method61(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v8 = method89(v7)
        del v7
        v2[v3] = v8
        del v8
        v3 += 1 
    del v0, v3
    return v2
def method94(v0 : cp.ndarray) -> static_array:
    v2 = static_array(4)
    v3 = 0
    while method63(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v8 = method89(v7)
        del v7
        v2[v3] = v8
        del v8
        v3 += 1 
    del v0, v3
    return v2
def method87(v0 : cp.ndarray) -> Tuple[i32, static_array, static_array, i32, static_array, US5]:
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
        v13 = method88(v12)
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
        v23 = method86(v22)
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
        v36 = method86(v35)
        del v35
        v28[v29] = v36
        del v36
        v29 += 1 
    del v29
    v37 = method91(v0)
    v39 = v0[32:].view(cp.uint8)
    del v0
    if v37 == 0:
        v41 = method92(v39)
        v48 = US5_0(v41)
    elif v37 == 1:
        method84(v39)
        v48 = US5_1()
    elif v37 == 2:
        v44 = method93(v39)
        v48 = US5_2(v44)
    elif v37 == 3:
        v46 = method94(v39)
        v48 = US5_3(v46)
    else:
        raise Exception("Invalid tag.")
    del v37, v39
    return v3, v5, v15, v26, v28, v48
def method96(v0 : cp.ndarray) -> i32:
    v2 = v0[40:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method95(v0 : cp.ndarray) -> Tuple[i32, static_array, static_array, i32, static_array, US5, US1]:
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
        v13 = method88(v12)
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
        v23 = method86(v22)
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
        v36 = method86(v35)
        del v35
        v28[v29] = v36
        del v36
        v29 += 1 
    del v29
    v37 = method91(v0)
    v39 = v0[32:].view(cp.uint8)
    if v37 == 0:
        v41 = method92(v39)
        v48 = US5_0(v41)
    elif v37 == 1:
        method84(v39)
        v48 = US5_1()
    elif v37 == 2:
        v44 = method93(v39)
        v48 = US5_2(v44)
    elif v37 == 3:
        v46 = method94(v39)
        v48 = US5_3(v46)
    else:
        raise Exception("Invalid tag.")
    del v37, v39
    v49 = method96(v0)
    v51 = v0[44:].view(cp.uint8)
    del v0
    if v49 == 0:
        method84(v51)
        v58 = US1_0()
    elif v49 == 1:
        method84(v51)
        v58 = US1_1()
    elif v49 == 2:
        method84(v51)
        v58 = US1_2()
    elif v49 == 3:
        v56 = method86(v51)
        v58 = US1_3(v56)
    else:
        raise Exception("Invalid tag.")
    del v49, v51
    return v3, v5, v15, v26, v28, v48, v58
def method85(v0 : cp.ndarray) -> US4:
    v1 = method86(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5, v6, v7, v8, v9, v10 = method87(v3)
        del v3
        return US4_0(v5, v6, v7, v8, v9, v10)
    elif v1 == 1:
        del v1
        v12, v13, v14, v15, v16, v17 = method87(v3)
        del v3
        return US4_1(v12, v13, v14, v15, v16, v17)
    elif v1 == 2:
        del v1
        method84(v3)
        del v3
        return US4_2()
    elif v1 == 3:
        del v1
        v20, v21, v22, v23, v24, v25 = method87(v3)
        del v3
        return US4_3(v20, v21, v22, v23, v24, v25)
    elif v1 == 4:
        del v1
        v27, v28, v29, v30, v31, v32 = method87(v3)
        del v3
        return US4_4(v27, v28, v29, v30, v31, v32)
    elif v1 == 5:
        del v1
        v34, v35, v36, v37, v38, v39, v40 = method95(v3)
        del v3
        return US4_5(v34, v35, v36, v37, v38, v39, v40)
    elif v1 == 6:
        del v1
        v42, v43, v44, v45, v46, v47 = method87(v3)
        del v3
        return US4_6(v42, v43, v44, v45, v46, v47)
    elif v1 == 7:
        del v1
        v49, v50, v51, v52, v53, v54 = method87(v3)
        del v3
        return US4_7(v49, v50, v51, v52, v53, v54)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method97(v0 : cp.ndarray) -> i32:
    v2 = v0[80:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method99(v0 : cp.ndarray) -> static_array_list:
    v2 = static_array_list(5)
    v3 = method86(v0)
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
        v11 = method89(v10)
        del v10
        v2[v5] = v11
        del v11
        v5 += 1 
    del v0, v4, v5
    return v2
def method100(v0 : cp.ndarray) -> Tuple[i32, i32]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v5 = v0[4:].view(cp.int32)
    del v0
    v6 = v5[0].item()
    del v5
    return v3, v6
def method102(v0 : cp.ndarray) -> i32:
    v2 = v0[4:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method101(v0 : cp.ndarray) -> Tuple[i32, US1]:
    v2 = v0[0:].view(cp.int32)
    v3 = v2[0].item()
    del v2
    v4 = method102(v0)
    v6 = v0[8:].view(cp.uint8)
    del v0
    if v4 == 0:
        method84(v6)
        v13 = US1_0()
    elif v4 == 1:
        method84(v6)
        v13 = US1_1()
    elif v4 == 2:
        method84(v6)
        v13 = US1_2()
    elif v4 == 3:
        v11 = method86(v6)
        v13 = US1_3(v11)
    else:
        raise Exception("Invalid tag.")
    del v4, v6
    return v3, v13
def method103(v0 : cp.ndarray) -> Tuple[i32, static_array]:
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
        v12 = method89(v11)
        del v11
        v5[v6] = v12
        del v12
        v6 += 1 
    del v0, v6
    return v3, v5
def method106(v0 : cp.ndarray) -> Tuple[static_array, i8]:
    v2 = static_array(5)
    v3 = 0
    while method61(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v8 = method89(v7)
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
def method105(v0 : cp.ndarray) -> Tuple[static_array, i8]:
    v1, v2 = method106(v0)
    del v0
    return v1, v2
def method104(v0 : cp.ndarray) -> Tuple[i32, static_array, i32]:
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
        v13, v14 = method105(v12)
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
def method98(v0 : cp.ndarray) -> US7:
    v1 = method86(v0)
    v3 = v0[16:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        v5 = method99(v3)
        del v3
        return US7_0(v5)
    elif v1 == 1:
        del v1
        v7, v8 = method100(v3)
        del v3
        return US7_1(v7, v8)
    elif v1 == 2:
        del v1
        v10, v11 = method101(v3)
        del v3
        return US7_2(v10, v11)
    elif v1 == 3:
        del v1
        v13, v14 = method103(v3)
        del v3
        return US7_3(v13, v14)
    elif v1 == 4:
        del v1
        v16, v17, v18 = method104(v3)
        del v3
        return US7_4(v16, v17, v18)
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method107(v0 : cp.ndarray) -> US2:
    v1 = method86(v0)
    v3 = v0[4:].view(cp.uint8)
    del v0
    if v1 == 0:
        del v1
        method84(v3)
        del v3
        return US2_0()
    elif v1 == 1:
        del v1
        method84(v3)
        del v3
        return US2_1()
    elif v1 == 2:
        del v1
        method84(v3)
        del v3
        return US2_2()
    else:
        del v1, v3
        raise Exception("Invalid tag.")
def method108(v0 : cp.ndarray) -> i32:
    v2 = v0[6248:].view(cp.int32)
    del v0
    v3 = v2[0].item()
    del v2
    return v3
def method81(v0 : cp.ndarray) -> Tuple[u64, US3, dynamic_array_list, static_array, US6]:
    v1 = method82(v0)
    v2 = method83(v0)
    v4 = v0[16:].view(cp.uint8)
    if v2 == 0:
        method84(v4)
        v9 = US3_0()
    elif v2 == 1:
        v7 = method85(v4)
        v9 = US3_1(v7)
    else:
        raise Exception("Invalid tag.")
    del v2, v4
    v11 = dynamic_array_list(128)
    v12 = method97(v0)
    v11.unsafe_set_length(v12)
    del v12
    v13 = v11.length_()
    v14 = 0
    while method6(v13, v14):
        v16 = u64(v14)
        v17 = v16 * 48
        del v16
        v18 = 96 + v17
        del v17
        v20 = v0[v18:].view(cp.uint8)
        del v18
        v21 = method98(v20)
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
        v31 = method107(v30)
        del v30
        v23[v24] = v31
        del v31
        v24 += 1 
    del v24
    v32 = method108(v0)
    v34 = v0[6256:].view(cp.uint8)
    del v0
    if v32 == 0:
        method84(v34)
        v51 = US6_0()
    elif v32 == 1:
        v37, v38, v39, v40, v41, v42 = method87(v34)
        v51 = US6_1(v37, v38, v39, v40, v41, v42)
    elif v32 == 2:
        v44, v45, v46, v47, v48, v49 = method87(v34)
        v51 = US6_2(v44, v45, v46, v47, v48, v49)
    else:
        raise Exception("Invalid tag.")
    del v32, v34
    return v1, v9, v11, v23, v51
def method115(v0 : u64) -> object:
    v1 = v0
    del v0
    return v1
def method114(v0 : u64) -> object:
    return method115(v0)
def method117() -> object:
    v0 = []
    return v0
def method120(v0 : i32) -> object:
    v1 = v0
    del v0
    return v1
def method124(v0 : u8) -> object:
    v1 = v0
    del v0
    return v1
def method123(v0 : u8) -> object:
    return method124(v0)
def method122(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method53(v2):
        v5 = v0[v2]
        v6 = method123(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method121(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method53(v2):
        v5 = v0[v2]
        v6 = method122(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method125(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method53(v2):
        v5 = v0[v2]
        v6 = method120(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method127(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method59(v2):
        v5 = v0[v2]
        v6 = method123(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method128(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method61(v2):
        v5 = v0[v2]
        v6 = method123(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method129(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method63(v2):
        v5 = v0[v2]
        v6 = method123(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method126(v0 : US5) -> object:
    match v0:
        case US5_0(v1): # Flop
            del v0
            v2 = method127(v1)
            del v1
            v3 = "Flop"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US5_1(): # Preflop
            del v0
            v5 = method117()
            v6 = "Preflop"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case US5_2(v8): # River
            del v0
            v9 = method128(v8)
            del v8
            v10 = "River"
            v11 = [v10,v9]
            del v9, v10
            return v11
        case US5_3(v12): # Turn
            del v0
            v13 = method129(v12)
            del v12
            v14 = "Turn"
            v15 = [v14,v13]
            del v13, v14
            return v15
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method119(v0 : i32, v1 : static_array, v2 : static_array, v3 : i32, v4 : static_array, v5 : US5) -> object:
    v6 = method120(v0)
    del v0
    v7 = method121(v1)
    del v1
    v8 = method125(v2)
    del v2
    v9 = method120(v3)
    del v3
    v10 = method125(v4)
    del v4
    v11 = method126(v5)
    del v5
    v12 = {'min_raise': v6, 'pl_card': v7, 'pot': v8, 'round_turn': v9, 'stack': v10, 'street': v11}
    del v6, v7, v8, v9, v10, v11
    return v12
def method131(v0 : US1) -> object:
    match v0:
        case US1_0(): # A_All_In
            del v0
            v1 = method117()
            v2 = "A_All_In"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US1_1(): # A_Call
            del v0
            v4 = method117()
            v5 = "A_Call"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US1_2(): # A_Fold
            del v0
            v7 = method117()
            v8 = "A_Fold"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US1_3(v10): # A_Raise
            del v0
            v11 = method120(v10)
            del v10
            v12 = "A_Raise"
            v13 = [v12,v11]
            del v11, v12
            return v13
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method130(v0 : i32, v1 : static_array, v2 : static_array, v3 : i32, v4 : static_array, v5 : US5, v6 : US1) -> object:
    v7 = []
    v8 = method119(v0, v1, v2, v3, v4, v5)
    del v0, v1, v2, v3, v4, v5
    v7.append(v8)
    del v8
    v9 = method131(v6)
    del v6
    v7.append(v9)
    del v9
    v10 = v7
    del v7
    return v10
def method118(v0 : US4) -> object:
    match v0:
        case US4_0(v1, v2, v3, v4, v5, v6): # G_Flop
            del v0
            v7 = method119(v1, v2, v3, v4, v5, v6)
            del v1, v2, v3, v4, v5, v6
            v8 = "G_Flop"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US4_1(v10, v11, v12, v13, v14, v15): # G_Fold
            del v0
            v16 = method119(v10, v11, v12, v13, v14, v15)
            del v10, v11, v12, v13, v14, v15
            v17 = "G_Fold"
            v18 = [v17,v16]
            del v16, v17
            return v18
        case US4_2(): # G_Preflop
            del v0
            v19 = method117()
            v20 = "G_Preflop"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case US4_3(v22, v23, v24, v25, v26, v27): # G_River
            del v0
            v28 = method119(v22, v23, v24, v25, v26, v27)
            del v22, v23, v24, v25, v26, v27
            v29 = "G_River"
            v30 = [v29,v28]
            del v28, v29
            return v30
        case US4_4(v31, v32, v33, v34, v35, v36): # G_Round
            del v0
            v37 = method119(v31, v32, v33, v34, v35, v36)
            del v31, v32, v33, v34, v35, v36
            v38 = "G_Round"
            v39 = [v38,v37]
            del v37, v38
            return v39
        case US4_5(v40, v41, v42, v43, v44, v45, v46): # G_Round'
            del v0
            v47 = method130(v40, v41, v42, v43, v44, v45, v46)
            del v40, v41, v42, v43, v44, v45, v46
            v48 = "G_Round'"
            v49 = [v48,v47]
            del v47, v48
            return v49
        case US4_6(v50, v51, v52, v53, v54, v55): # G_Showdown
            del v0
            v56 = method119(v50, v51, v52, v53, v54, v55)
            del v50, v51, v52, v53, v54, v55
            v57 = "G_Showdown"
            v58 = [v57,v56]
            del v56, v57
            return v58
        case US4_7(v59, v60, v61, v62, v63, v64): # G_Turn
            del v0
            v65 = method119(v59, v60, v61, v62, v63, v64)
            del v59, v60, v61, v62, v63, v64
            v66 = "G_Turn"
            v67 = [v66,v65]
            del v65, v66
            return v67
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method116(v0 : US3) -> object:
    match v0:
        case US3_0(): # None
            del v0
            v1 = method117()
            v2 = "None"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US3_1(v4): # Some
            del v0
            v5 = method118(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method113(v0 : u64, v1 : US3) -> object:
    v2 = method114(v0)
    del v0
    v3 = method116(v1)
    del v1
    v4 = {'deck': v2, 'game': v3}
    del v2, v3
    return v4
def method135(v0 : static_array_list) -> object:
    v1 = []
    v2 = v0.length
    v3 = 0
    while method6(v2, v3):
        v6 = v0[v3]
        v7 = method123(v6)
        del v6
        v1.append(v7)
        del v7
        v3 += 1 
    del v0, v2, v3
    return v1
def method136(v0 : i32, v1 : i32) -> object:
    v2 = method120(v0)
    del v0
    v3 = method120(v1)
    del v1
    v4 = {'chips_won': v2, 'winner_id': v3}
    del v2, v3
    return v4
def method137(v0 : i32, v1 : US1) -> object:
    v2 = []
    v3 = method120(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method131(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method138(v0 : i32, v1 : static_array) -> object:
    v2 = []
    v3 = method120(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method122(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method143(v0 : i8) -> object:
    v1 = v0
    del v0
    return v1
def method142(v0 : static_array, v1 : i8) -> object:
    v2 = method128(v0)
    del v0
    v3 = method143(v1)
    del v1
    v4 = {'hand': v2, 'score': v3}
    del v2, v3
    return v4
def method141(v0 : static_array, v1 : i8) -> object:
    return method142(v0, v1)
def method140(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method53(v2):
        v6, v7 = v0[v2]
        v8 = method141(v6, v7)
        del v6, v7
        v1.append(v8)
        del v8
        v2 += 1 
    del v0, v2
    return v1
def method139(v0 : i32, v1 : static_array, v2 : i32) -> object:
    v3 = method120(v0)
    del v0
    v4 = method140(v1)
    del v1
    v5 = method120(v2)
    del v2
    v6 = {'chips_won': v3, 'hands_shown': v4, 'winner_id': v5}
    del v3, v4, v5
    return v6
def method134(v0 : US7) -> object:
    match v0:
        case US7_0(v1): # CommunityCardsAre
            del v0
            v2 = method135(v1)
            del v1
            v3 = "CommunityCardsAre"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US7_1(v5, v6): # Fold
            del v0
            v7 = method136(v5, v6)
            del v5, v6
            v8 = "Fold"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US7_2(v10, v11): # PlayerAction
            del v0
            v12 = method137(v10, v11)
            del v10, v11
            v13 = "PlayerAction"
            v14 = [v13,v12]
            del v12, v13
            return v14
        case US7_3(v15, v16): # PlayerGotCards
            del v0
            v17 = method138(v15, v16)
            del v15, v16
            v18 = "PlayerGotCards"
            v19 = [v18,v17]
            del v17, v18
            return v19
        case US7_4(v20, v21, v22): # Showdown
            del v0
            v23 = method139(v20, v21, v22)
            del v20, v21, v22
            v24 = "Showdown"
            v25 = [v24,v23]
            del v23, v24
            return v25
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method133(v0 : dynamic_array_list) -> object:
    v1 = []
    v2 = v0.length_()
    v3 = 0
    while method6(v2, v3):
        v6 = v0[v3]
        v7 = method134(v6)
        del v6
        v1.append(v7)
        del v7
        v3 += 1 
    del v0, v2, v3
    return v1
def method145(v0 : US2) -> object:
    match v0:
        case US2_0(): # Computer
            del v0
            v1 = method117()
            v2 = "Computer"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US2_1(): # Human
            del v0
            v4 = method117()
            v5 = "Human"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US2_2(): # Random
            del v0
            v7 = method117()
            v8 = "Random"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method144(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method53(v2):
        v5 = v0[v2]
        v6 = method145(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method146(v0 : US6) -> object:
    match v0:
        case US6_0(): # GameNotStarted
            del v0
            v1 = method117()
            v2 = "GameNotStarted"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US6_1(v4, v5, v6, v7, v8, v9): # GameOver
            del v0
            v10 = method119(v4, v5, v6, v7, v8, v9)
            del v4, v5, v6, v7, v8, v9
            v11 = "GameOver"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case US6_2(v13, v14, v15, v16, v17, v18): # WaitingForActionFromPlayerId
            del v0
            v19 = method119(v13, v14, v15, v16, v17, v18)
            del v13, v14, v15, v16, v17, v18
            v20 = "WaitingForActionFromPlayerId"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method132(v0 : dynamic_array_list, v1 : static_array, v2 : US6) -> object:
    v3 = method133(v0)
    del v0
    v4 = method144(v1)
    del v1
    v5 = method146(v2)
    del v2
    v6 = {'messages': v3, 'pl_type': v4, 'ui_game_state': v5}
    del v3, v4, v5
    return v6
def method112(v0 : u64, v1 : US3, v2 : dynamic_array_list, v3 : static_array, v4 : US6) -> object:
    v5 = method113(v0, v1)
    del v0, v1
    v6 = method132(v2, v3, v4)
    del v2, v3, v4
    v7 = {'private': v5, 'public': v6}
    del v5, v6
    return v7
def method152(v0 : cp.ndarray) -> object:
    v1 = v0
    del v0
    return v1
def method151(v0 : cp.ndarray) -> object:
    return method152(v0)
def method150(v0 : cp.ndarray, v1 : u64) -> object:
    v2 = []
    v3 = method151(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method115(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method149(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method150(v0, v1)
    del v0, v1
    v5 = method150(v2, v3)
    del v2, v3
    v6 = {'output': v4, 'param': v5}
    del v4, v5
    return v6
def method148(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    return method149(v0, v1, v2, v3)
def method147(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method148(v0, v1, v2, v3)
    del v0, v1, v2, v3
    v5 = {'model_data': v4}
    del v4
    return v5
def method111(v0 : u64, v1 : US3, v2 : dynamic_array_list, v3 : static_array, v4 : US6, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64) -> object:
    v9 = method112(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    v10 = method147(v5, v6, v7, v8)
    del v5, v6, v7, v8
    v11 = {'game': v9, 'neural': v10}
    del v9, v10
    return v11
def method157(v0 : f32) -> object:
    v1 = v0
    del v0
    return v1
def method156(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method6(v2, v3):
        v5 = v0[v3]
        v6 = method157(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method155(v0 : list) -> object:
    v1 = []
    v2 = len(v0)
    v3 = 0
    while method6(v2, v3):
        v5 = v0[v3]
        v6 = method156(v5)
        del v5
        v1.append(v6)
        del v6
        v3 += 1 
    del v0, v2, v3
    return v1
def method154(v0 : US8) -> object:
    match v0:
        case US8_0(v1): # AddRewardsRando
            del v0
            v2 = method155(v1)
            del v1
            v3 = "AddRewardsRando"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US8_1(v5): # AddRewardsSelf
            del v0
            v6 = method155(v5)
            del v5
            v7 = "AddRewardsSelf"
            v8 = [v7,v6]
            del v6, v7
            return v8
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
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
def method110(v0 : u64, v1 : US3, v2 : dynamic_array_list, v3 : static_array, v4 : US6, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64, v9 : list) -> object:
    v10 = []
    v11 = method111(v0, v1, v2, v3, v4, v5, v6, v7, v8)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8
    v10.append(v11)
    del v11
    v12 = method153(v9)
    del v9
    v10.append(v12)
    del v12
    v13 = v10
    del v10
    return v13
def method109(v0 : u64, v1 : US3, v2 : dynamic_array_list, v3 : static_array, v4 : US6, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64, v9 : list) -> object:
    v10 = method110(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8, v9
    return v10
def method158(v0 : u64, v1 : US3, v2 : dynamic_array_list, v3 : static_array, v4 : US6, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64) -> object:
    v9 = method111(v0, v1, v2, v3, v4, v5, v6, v7, v8)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8
    return v9
def main_body():
    v0 = Closure0()
    v1 = Closure1()
    v2 = collections.namedtuple("Leduc_Full",['event_loop_gpu', 'init'])(v0, v1)
    del v0, v1
    return v2

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
