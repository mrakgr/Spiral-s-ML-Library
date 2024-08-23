kernel = r"""
#include <new>
#include <assert.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cuda/semaphore>
__device__ cuda::binary_semaphore<cuda::thread_scope_system> console_lock(1);
#include <mma.h>
using namespace nvcuda;
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
__device__ static_array<Union2,2l> f_4(unsigned char * v0);
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
__device__ static_array<unsigned char,2l> f_11(unsigned char * v0);
__device__ int f_14(unsigned char * v0);
__device__ static_array<unsigned char,3l> f_15(unsigned char * v0);
__device__ static_array<unsigned char,5l> f_16(unsigned char * v0);
__device__ static_array<unsigned char,4l> f_17(unsigned char * v0);
__device__ Tuple2 f_10(unsigned char * v0);
struct Tuple3;
__device__ int f_19(unsigned char * v0);
__device__ Tuple3 f_18(unsigned char * v0);
__device__ Union4 f_9(unsigned char * v0);
__device__ int f_20(unsigned char * v0);
__device__ static_array_list<unsigned char,5l> f_22(unsigned char * v0);
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
struct Tuple8;
struct Tuple9;
struct Tuple10;
__device__ unsigned int loop_34(unsigned int v0, curandStatePhilox4_32_10_t & v1);
__device__ Tuple10 draw_card_33(curandStatePhilox4_32_10_t & v0, unsigned long long v1);
__device__ Tuple8 draw_cards_32(curandStatePhilox4_32_10_t & v0, unsigned long long v1);
__device__ static_array_list<unsigned char,5l> get_community_cards_35(Union5 v0, static_array<unsigned char,3l> v1);
__device__ bool player_can_act_37(int v0, static_array<static_array<unsigned char,2l>,2l> v1, static_array<int,2l> v2, int v3, static_array<int,2l> v4, Union5 v5);
__device__ Union4 go_next_street_38(int v0, static_array<static_array<unsigned char,2l>,2l> v1, static_array<int,2l> v2, int v3, static_array<int,2l> v4, Union5 v5);
__device__ Union4 try_round_36(int v0, static_array<static_array<unsigned char,2l>,2l> v1, static_array<int,2l> v2, int v3, static_array<int,2l> v4, Union5 v5);
struct Tuple11;
__device__ Tuple11 draw_cards_39(curandStatePhilox4_32_10_t & v0, unsigned long long v1);
struct Tuple12;
__device__ Tuple12 draw_cards_40(curandStatePhilox4_32_10_t & v0, unsigned long long v1);
__device__ static_array_list<unsigned char,5l> get_community_cards_41(Union5 v0, static_array<unsigned char,1l> v1);
struct Union8;
struct Union9;
struct Union10;
struct Tuple13;
__device__ void method_43(unsigned int v0, float * v1, int v2);
__device__ void method_44(unsigned int v0, float * v1, int v2);
__device__ int int_range_45(int v0, int v1, curandStatePhilox4_32_10_t & v2);
__device__ void method_46(float * v0, int v1, float * v2, int v3, float * v4, int v5);
__device__ void method_47(unsigned int * v0, int v1, float * v2);
struct Tuple14;
struct Tuple15;
struct Tuple16;
struct Tuple17;
struct Tuple18;
__device__ Tuple14 method_48(float * v0, float * v1, float * v2, float * v3, float * v4, int v5, int v6);
__device__ Union9 noinline_run_42(unsigned char * v0, unsigned char * v1, Union8 v2);
__device__ void method_49(Union1 v0);
struct Tuple19;
__device__ int loop_53(static_array<float,6l> v0, float v1, int v2);
__device__ int pick_discrete__52(static_array<float,6l> v0, float v1);
__device__ int sample_discrete__51(static_array<float,6l> v0, curandStatePhilox4_32_10_t & v1);
__device__ Union1 sample_discrete_50(static_array<Tuple19,6l> v0, curandStatePhilox4_32_10_t & v1);
struct Tuple20;
struct Tuple21;
struct Union11;
struct Tuple22;
struct Union12;
struct Tuple23;
struct Tuple24;
struct Union13;
struct Union14;
struct Union15;
struct Union16;
struct Union17;
__device__ Tuple0 score_54(static_array<unsigned char,7l> v0);
__device__ void play_loop_31(curandStatePhilox4_32_10_t & v0, unsigned char * v1, unsigned long long v2, unsigned char * v3, unsigned long long v4, unsigned long long & v5, Union3 & v6, dynamic_array_list<Union6,128l> & v7, static_array<Union2,2l> & v8, Union7 & v9, Union4 v10);
__device__ void f_56(unsigned char * v0, unsigned long long v1);
__device__ void f_57(unsigned char * v0, int v1);
__device__ void f_58(unsigned char * v0);
__device__ void f_60(unsigned char * v0, int v1);
__device__ void f_64(unsigned char * v0, unsigned char v1);
__device__ void f_63(unsigned char * v0, unsigned char v1);
__device__ void f_62(unsigned char * v0, static_array<unsigned char,2l> v1);
__device__ void f_65(unsigned char * v0, int v1);
__device__ void f_66(unsigned char * v0, static_array<unsigned char,3l> v1);
__device__ void f_67(unsigned char * v0, static_array<unsigned char,5l> v1);
__device__ void f_68(unsigned char * v0, static_array<unsigned char,4l> v1);
__device__ void f_61(unsigned char * v0, int v1, static_array<static_array<unsigned char,2l>,2l> v2, static_array<int,2l> v3, int v4, static_array<int,2l> v5, Union5 v6);
__device__ void f_70(unsigned char * v0, int v1);
__device__ void f_69(unsigned char * v0, int v1, static_array<static_array<unsigned char,2l>,2l> v2, static_array<int,2l> v3, int v4, static_array<int,2l> v5, Union5 v6, Union1 v7);
__device__ void f_59(unsigned char * v0, Union4 v1);
__device__ void f_71(unsigned char * v0, int v1);
__device__ void f_73(unsigned char * v0, static_array_list<unsigned char,5l> v1);
__device__ void f_74(unsigned char * v0, int v1, int v2);
__device__ void f_76(unsigned char * v0, int v1);
__device__ void f_75(unsigned char * v0, int v1, Union1 v2);
__device__ void f_77(unsigned char * v0, int v1, static_array<unsigned char,2l> v2);
__device__ void f_80(unsigned char * v0, static_array<unsigned char,5l> v1, char v2);
__device__ void f_79(unsigned char * v0, static_array<unsigned char,5l> v1, char v2);
__device__ void f_78(unsigned char * v0, int v1, static_array<Tuple0,2l> v2, int v3);
__device__ void f_72(unsigned char * v0, Union6 v1);
__device__ void f_81(unsigned char * v0, Union2 v1);
__device__ void f_82(unsigned char * v0, int v1);
__device__ void f_55(unsigned char * v0, unsigned long long v1, Union3 v2, dynamic_array_list<Union6,128l> v3, static_array<Union2,2l> v4, Union7 v5);
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
    static_array<Union2,2l> v0;
    __device__ Union0_1(static_array<Union2,2l> t0) : v0(t0) {}
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
    static_array<unsigned char,3l> v0;
    __device__ Union5_0(static_array<unsigned char,3l> t0) : v0(t0) {}
    __device__ Union5_0() = delete;
};
struct Union5_1 { // Preflop
};
struct Union5_2 { // River
    static_array<unsigned char,5l> v0;
    __device__ Union5_2(static_array<unsigned char,5l> t0) : v0(t0) {}
    __device__ Union5_2() = delete;
};
struct Union5_3 { // Turn
    static_array<unsigned char,4l> v0;
    __device__ Union5_3(static_array<unsigned char,4l> t0) : v0(t0) {}
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
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union5 v5;
    int v0;
    int v3;
    __device__ Union4_0(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union5 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_0() = delete;
};
struct Union4_1 { // G_Fold
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union5 v5;
    int v0;
    int v3;
    __device__ Union4_1(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union5 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_1() = delete;
};
struct Union4_2 { // G_Preflop
};
struct Union4_3 { // G_River
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union5 v5;
    int v0;
    int v3;
    __device__ Union4_3(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union5 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_3() = delete;
};
struct Union4_4 { // G_Round
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union5 v5;
    int v0;
    int v3;
    __device__ Union4_4(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union5 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_4() = delete;
};
struct Union4_5 { // G_Round'
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union5 v5;
    Union1 v6;
    int v0;
    int v3;
    __device__ Union4_5(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union5 t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
    __device__ Union4_5() = delete;
};
struct Union4_6 { // G_Showdown
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union5 v5;
    int v0;
    int v3;
    __device__ Union4_6(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union5 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union4_6() = delete;
};
struct Union4_7 { // G_Turn
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union5 v5;
    int v0;
    int v3;
    __device__ Union4_7(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union5 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
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
    static_array<unsigned char,5l> v0;
    char v1;
    __device__ Tuple0() = default;
    __device__ Tuple0(static_array<unsigned char,5l> t0, char t1) : v0(t0), v1(t1) {}
};
struct Union6_0 { // CommunityCardsAre
    static_array_list<unsigned char,5l> v0;
    __device__ Union6_0(static_array_list<unsigned char,5l> t0) : v0(t0) {}
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
    static_array<unsigned char,2l> v1;
    int v0;
    __device__ Union6_3(int t0, static_array<unsigned char,2l> t1) : v0(t0), v1(t1) {}
    __device__ Union6_3() = delete;
};
struct Union6_4 { // Showdown
    static_array<Tuple0,2l> v1;
    int v0;
    int v2;
    __device__ Union6_4(int t0, static_array<Tuple0,2l> t1, int t2) : v0(t0), v1(t1), v2(t2) {}
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
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union5 v5;
    int v0;
    int v3;
    __device__ Union7_1(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union5 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
    __device__ Union7_1() = delete;
};
struct Union7_2 { // WaitingForActionFromPlayerId
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union5 v5;
    int v0;
    int v3;
    __device__ Union7_2(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union5 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
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
    dynamic_array_list<Union6,128l> v2;
    static_array<Union2,2l> v3;
    Union7 v4;
    __device__ Tuple1() = default;
    __device__ Tuple1(unsigned long long t0, Union3 t1, dynamic_array_list<Union6,128l> t2, static_array<Union2,2l> t3, Union7 t4) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4) {}
};
struct Tuple2 {
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union5 v5;
    int v0;
    int v3;
    __device__ Tuple2() = default;
    __device__ Tuple2(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union5 t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
};
struct Tuple3 {
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union5 v5;
    Union1 v6;
    int v0;
    int v3;
    __device__ Tuple3() = default;
    __device__ Tuple3(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union5 t5, Union1 t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
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
    static_array<unsigned char,2l> v1;
    int v0;
    __device__ Tuple6() = default;
    __device__ Tuple6(int t0, static_array<unsigned char,2l> t1) : v0(t0), v1(t1) {}
};
struct Tuple7 {
    static_array<Tuple0,2l> v1;
    int v0;
    int v2;
    __device__ Tuple7() = default;
    __device__ Tuple7(int t0, static_array<Tuple0,2l> t1, int t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Tuple8 {
    static_array<unsigned char,3l> v0;
    unsigned long long v1;
    __device__ Tuple8() = default;
    __device__ Tuple8(static_array<unsigned char,3l> t0, unsigned long long t1) : v0(t0), v1(t1) {}
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
    static_array<unsigned char,2l> v0;
    unsigned long long v1;
    __device__ Tuple11() = default;
    __device__ Tuple11(static_array<unsigned char,2l> t0, unsigned long long t1) : v0(t0), v1(t1) {}
};
struct Tuple12 {
    static_array<unsigned char,1l> v0;
    unsigned long long v1;
    __device__ Tuple12() = default;
    __device__ Tuple12(static_array<unsigned char,1l> t0, unsigned long long t1) : v0(t0), v1(t1) {}
};
struct Union8_0 { // None
};
struct Union8_1 { // Some
    static_array<static_array<unsigned char,2l>,2l> v1;
    static_array<int,2l> v2;
    static_array<int,2l> v4;
    Union5 v5;
    dynamic_array_list<Union6,128l> v6;
    int v0;
    int v3;
    __device__ Union8_1(int t0, static_array<static_array<unsigned char,2l>,2l> t1, static_array<int,2l> t2, int t3, static_array<int,2l> t4, Union5 t5, dynamic_array_list<Union6,128l> t6) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5), v6(t6) {}
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
struct Union10_0 { // None
};
struct Union10_1 { // Some
    Union1 v0;
    __device__ Union10_1(Union1 t0) : v0(t0) {}
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
struct Tuple13 {
    int v0;
    unsigned int v1;
    __device__ Tuple13() = default;
    __device__ Tuple13(int t0, unsigned int t1) : v0(t0), v1(t1) {}
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
    float v1;
    int v2;
    __device__ Tuple14() = default;
    __device__ Tuple14(float t0, float t1, int t2) : v0(t0), v1(t1), v2(t2) {}
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
struct Tuple17 {
    float v0;
    int v1;
    __device__ Tuple17() = default;
    __device__ Tuple17(float t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure5 {
    __device__ Tuple17 operator()(Tuple17 tup0, Tuple17 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v1 < v3;
        if (v4){
            return Tuple17{v0, v1};
        } else {
            return Tuple17{v2, v3};
        }
    }
};
struct Tuple18 {
    int v0;
    bool v1;
    __device__ Tuple18() = default;
    __device__ Tuple18(int t0, bool t1) : v0(t0), v1(t1) {}
};
struct Closure6 {
    __device__ Tuple18 operator()(Tuple18 tup0, Tuple18 tup1){
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
                return Tuple18{v5, true};
            } else {
                return Tuple18{v0, v1};
            }
        } else {
            if (v3){
                return Tuple18{v2, v3};
            } else {
                return Tuple18{v0, v1};
            }
        }
    }
};
struct Closure7 {
    int v0;
    __device__ Tuple17 operator()(Tuple17 tup0, Tuple17 tup1){
        int & v0 = this->v0;
        float v1 = tup0.v0; int v2 = tup0.v1; float v3 = tup1.v0; int v4 = tup1.v1;
        bool v5;
        v5 = v2 == v0;
        if (v5){
            return Tuple17{v1, v2};
        } else {
            bool v6;
            v6 = v4 == v0;
            if (v6){
                return Tuple17{v3, v4};
            } else {
                return Tuple17{v1, v2};
            }
        }
    }
    __device__ Closure7(int _v0) : v0(_v0) { }
};
struct Tuple19 {
    Union1 v0;
    float v1;
    __device__ Tuple19() = default;
    __device__ Tuple19(Union1 t0, float t1) : v0(t0), v1(t1) {}
};
struct Tuple20 {
    int v1;
    bool v0;
    __device__ Tuple20() = default;
    __device__ Tuple20(bool t0, int t1) : v0(t0), v1(t1) {}
};
struct Tuple21 {
    int v0;
    int v1;
    int v2;
    __device__ Tuple21() = default;
    __device__ Tuple21(int t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
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
struct Tuple22 {
    int v0;
    int v1;
    unsigned char v2;
    __device__ Tuple22() = default;
    __device__ Tuple22(int t0, int t1, unsigned char t2) : v0(t0), v1(t1), v2(t2) {}
};
struct Union12_0 { // None
};
struct Union12_1 { // Some
    static_array<unsigned char,5l> v0;
    __device__ Union12_1(static_array<unsigned char,5l> t0) : v0(t0) {}
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
struct Tuple23 {
    Union11 v1;
    int v0;
    __device__ Tuple23() = default;
    __device__ Tuple23(int t0, Union11 t1) : v0(t0), v1(t1) {}
};
struct Tuple24 {
    int v0;
    int v1;
    int v2;
    unsigned char v3;
    __device__ Tuple24() = default;
    __device__ Tuple24(int t0, int t1, int t2, unsigned char t3) : v0(t0), v1(t1), v2(t2), v3(t3) {}
};
struct Union13_0 { // None
};
struct Union13_1 { // Some
    static_array<unsigned char,4l> v0;
    static_array<unsigned char,3l> v1;
    __device__ Union13_1(static_array<unsigned char,4l> t0, static_array<unsigned char,3l> t1) : v0(t0), v1(t1) {}
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
    static_array<unsigned char,3l> v0;
    static_array<unsigned char,4l> v1;
    __device__ Union14_1(static_array<unsigned char,3l> t0, static_array<unsigned char,4l> t1) : v0(t0), v1(t1) {}
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
    static_array<unsigned char,2l> v0;
    static_array<unsigned char,2l> v1;
    __device__ Union15_1(static_array<unsigned char,2l> t0, static_array<unsigned char,2l> t1) : v0(t0), v1(t1) {}
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
    static_array<unsigned char,2l> v0;
    static_array<unsigned char,5l> v1;
    __device__ Union16_1(static_array<unsigned char,2l> t0, static_array<unsigned char,5l> t1) : v0(t0), v1(t1) {}
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
    static_array<unsigned char,2l> v0;
    static_array<unsigned char,3l> v1;
    __device__ Union17_1(static_array<unsigned char,2l> t0, static_array<unsigned char,3l> t1) : v0(t0), v1(t1) {}
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
    v3 = v1[0l];
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
    v1 = v0 < 2l;
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
__device__ static_array<Union2,2l> f_4(unsigned char * v0){
    static_array<Union2,2l> v1;
    int v3;
    v3 = 0l;
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
        v3 += 1l ;
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
            static_array<Union2,2l> v7;
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
    v3 = v1[0l];
    return v3;
}
__device__ int f_8(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+8ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ unsigned char f_13(unsigned char * v0){
    unsigned char * v1;
    v1 = (unsigned char *)(v0+0ull);
    unsigned char v3;
    v3 = v1[0l];
    return v3;
}
__device__ unsigned char f_12(unsigned char * v0){
    unsigned char v1;
    v1 = f_13(v0);
    return v1;
}
__device__ static_array<unsigned char,2l> f_11(unsigned char * v0){
    static_array<unsigned char,2l> v1;
    int v3;
    v3 = 0l;
    while (while_method_0(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        unsigned char v8;
        v8 = f_12(v6);
        v1[v3] = v8;
        v3 += 1l ;
    }
    return v1;
}
__device__ int f_14(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+28ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ inline bool while_method_1(int v0){
    bool v1;
    v1 = v0 < 3l;
    return v1;
}
__device__ static_array<unsigned char,3l> f_15(unsigned char * v0){
    static_array<unsigned char,3l> v1;
    int v3;
    v3 = 0l;
    while (while_method_1(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        unsigned char v8;
        v8 = f_12(v6);
        v1[v3] = v8;
        v3 += 1l ;
    }
    return v1;
}
__device__ inline bool while_method_2(int v0){
    bool v1;
    v1 = v0 < 5l;
    return v1;
}
__device__ static_array<unsigned char,5l> f_16(unsigned char * v0){
    static_array<unsigned char,5l> v1;
    int v3;
    v3 = 0l;
    while (while_method_2(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        unsigned char v8;
        v8 = f_12(v6);
        v1[v3] = v8;
        v3 += 1l ;
    }
    return v1;
}
__device__ inline bool while_method_3(int v0){
    bool v1;
    v1 = v0 < 4l;
    return v1;
}
__device__ static_array<unsigned char,4l> f_17(unsigned char * v0){
    static_array<unsigned char,4l> v1;
    int v3;
    v3 = 0l;
    while (while_method_3(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        unsigned char v8;
        v8 = f_12(v6);
        v1[v3] = v8;
        v3 += 1l ;
    }
    return v1;
}
__device__ Tuple2 f_10(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0l];
    static_array<static_array<unsigned char,2l>,2l> v4;
    int v6;
    v6 = 0l;
    while (while_method_0(v6)){
        unsigned long long v8;
        v8 = (unsigned long long)v6;
        unsigned long long v9;
        v9 = v8 * 2ull;
        unsigned long long v10;
        v10 = 4ull + v9;
        unsigned char * v11;
        v11 = (unsigned char *)(v0+v10);
        static_array<unsigned char,2l> v13;
        v13 = f_11(v11);
        v4[v6] = v13;
        v6 += 1l ;
    }
    static_array<int,2l> v14;
    int v16;
    v16 = 0l;
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
        v16 += 1l ;
    }
    int * v24;
    v24 = (int *)(v0+16ull);
    int v26;
    v26 = v24[0l];
    static_array<int,2l> v27;
    int v29;
    v29 = 0l;
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
        v29 += 1l ;
    }
    int v37;
    v37 = f_14(v0);
    unsigned char * v38;
    v38 = (unsigned char *)(v0+32ull);
    Union5 v48;
    switch (v37) {
        case 0: {
            static_array<unsigned char,3l> v41;
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
            static_array<unsigned char,5l> v44;
            v44 = f_16(v38);
            v48 = Union5{Union5_2{v44}};
            break;
        }
        case 3: {
            static_array<unsigned char,4l> v46;
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
    v3 = v1[0l];
    return v3;
}
__device__ Tuple3 f_18(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0l];
    static_array<static_array<unsigned char,2l>,2l> v4;
    int v6;
    v6 = 0l;
    while (while_method_0(v6)){
        unsigned long long v8;
        v8 = (unsigned long long)v6;
        unsigned long long v9;
        v9 = v8 * 2ull;
        unsigned long long v10;
        v10 = 4ull + v9;
        unsigned char * v11;
        v11 = (unsigned char *)(v0+v10);
        static_array<unsigned char,2l> v13;
        v13 = f_11(v11);
        v4[v6] = v13;
        v6 += 1l ;
    }
    static_array<int,2l> v14;
    int v16;
    v16 = 0l;
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
        v16 += 1l ;
    }
    int * v24;
    v24 = (int *)(v0+16ull);
    int v26;
    v26 = v24[0l];
    static_array<int,2l> v27;
    int v29;
    v29 = 0l;
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
        v29 += 1l ;
    }
    int v37;
    v37 = f_14(v0);
    unsigned char * v38;
    v38 = (unsigned char *)(v0+32ull);
    Union5 v48;
    switch (v37) {
        case 0: {
            static_array<unsigned char,3l> v41;
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
            static_array<unsigned char,5l> v44;
            v44 = f_16(v38);
            v48 = Union5{Union5_2{v44}};
            break;
        }
        case 3: {
            static_array<unsigned char,4l> v46;
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
            int v5; static_array<static_array<unsigned char,2l>,2l> v6; static_array<int,2l> v7; int v8; static_array<int,2l> v9; Union5 v10;
            Tuple2 tmp0 = f_10(v2);
            v5 = tmp0.v0; v6 = tmp0.v1; v7 = tmp0.v2; v8 = tmp0.v3; v9 = tmp0.v4; v10 = tmp0.v5;
            return Union4{Union4_0{v5, v6, v7, v8, v9, v10}};
            break;
        }
        case 1: {
            int v12; static_array<static_array<unsigned char,2l>,2l> v13; static_array<int,2l> v14; int v15; static_array<int,2l> v16; Union5 v17;
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
            int v20; static_array<static_array<unsigned char,2l>,2l> v21; static_array<int,2l> v22; int v23; static_array<int,2l> v24; Union5 v25;
            Tuple2 tmp2 = f_10(v2);
            v20 = tmp2.v0; v21 = tmp2.v1; v22 = tmp2.v2; v23 = tmp2.v3; v24 = tmp2.v4; v25 = tmp2.v5;
            return Union4{Union4_3{v20, v21, v22, v23, v24, v25}};
            break;
        }
        case 4: {
            int v27; static_array<static_array<unsigned char,2l>,2l> v28; static_array<int,2l> v29; int v30; static_array<int,2l> v31; Union5 v32;
            Tuple2 tmp3 = f_10(v2);
            v27 = tmp3.v0; v28 = tmp3.v1; v29 = tmp3.v2; v30 = tmp3.v3; v31 = tmp3.v4; v32 = tmp3.v5;
            return Union4{Union4_4{v27, v28, v29, v30, v31, v32}};
            break;
        }
        case 5: {
            int v34; static_array<static_array<unsigned char,2l>,2l> v35; static_array<int,2l> v36; int v37; static_array<int,2l> v38; Union5 v39; Union1 v40;
            Tuple3 tmp4 = f_18(v2);
            v34 = tmp4.v0; v35 = tmp4.v1; v36 = tmp4.v2; v37 = tmp4.v3; v38 = tmp4.v4; v39 = tmp4.v5; v40 = tmp4.v6;
            return Union4{Union4_5{v34, v35, v36, v37, v38, v39, v40}};
            break;
        }
        case 6: {
            int v42; static_array<static_array<unsigned char,2l>,2l> v43; static_array<int,2l> v44; int v45; static_array<int,2l> v46; Union5 v47;
            Tuple2 tmp5 = f_10(v2);
            v42 = tmp5.v0; v43 = tmp5.v1; v44 = tmp5.v2; v45 = tmp5.v3; v46 = tmp5.v4; v47 = tmp5.v5;
            return Union4{Union4_6{v42, v43, v44, v45, v46, v47}};
            break;
        }
        case 7: {
            int v49; static_array<static_array<unsigned char,2l>,2l> v50; static_array<int,2l> v51; int v52; static_array<int,2l> v53; Union5 v54;
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
    v3 = v1[0l];
    return v3;
}
__device__ inline bool while_method_4(int v0, int v1){
    bool v2;
    v2 = v1 < v0;
    return v2;
}
__device__ static_array_list<unsigned char,5l> f_22(unsigned char * v0){
    static_array_list<unsigned char,5l> v1;
    v1 = static_array_list<unsigned char,5l>{};
    int v3;
    v3 = f_1(v0);
    v1.unsafe_set_length(v3);
    int v4;
    v4 = v1.length;
    int v5;
    v5 = 0l;
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
        v5 += 1l ;
    }
    return v1;
}
__device__ Tuple4 f_23(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0l];
    int * v4;
    v4 = (int *)(v0+4ull);
    int v6;
    v6 = v4[0l];
    return Tuple4{v3, v6};
}
__device__ int f_25(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+4ull);
    int v3;
    v3 = v1[0l];
    return v3;
}
__device__ Tuple5 f_24(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0l];
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
    v3 = v1[0l];
    static_array<unsigned char,2l> v4;
    int v6;
    v6 = 0l;
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
        v6 += 1l ;
    }
    return Tuple6{v3, v4};
}
__device__ Tuple0 f_29(unsigned char * v0){
    static_array<unsigned char,5l> v1;
    int v3;
    v3 = 0l;
    while (while_method_2(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        unsigned char v8;
        v8 = f_12(v6);
        v1[v3] = v8;
        v3 += 1l ;
    }
    char * v9;
    v9 = (char *)(v0+5ull);
    char v11;
    v11 = v9[0l];
    return Tuple0{v1, v11};
}
__device__ Tuple0 f_28(unsigned char * v0){
    static_array<unsigned char,5l> v1; char v2;
    Tuple0 tmp10 = f_29(v0);
    v1 = tmp10.v0; v2 = tmp10.v1;
    return Tuple0{v1, v2};
}
__device__ Tuple7 f_27(unsigned char * v0){
    int * v1;
    v1 = (int *)(v0+0ull);
    int v3;
    v3 = v1[0l];
    static_array<Tuple0,2l> v4;
    int v6;
    v6 = 0l;
    while (while_method_0(v6)){
        unsigned long long v8;
        v8 = (unsigned long long)v6;
        unsigned long long v9;
        v9 = v8 * 8ull;
        unsigned long long v10;
        v10 = 8ull + v9;
        unsigned char * v11;
        v11 = (unsigned char *)(v0+v10);
        static_array<unsigned char,5l> v13; char v14;
        Tuple0 tmp11 = f_28(v11);
        v13 = tmp11.v0; v14 = tmp11.v1;
        v4[v6] = Tuple0{v13, v14};
        v6 += 1l ;
    }
    int * v15;
    v15 = (int *)(v0+24ull);
    int v17;
    v17 = v15[0l];
    return Tuple7{v3, v4, v17};
}
__device__ Union6 f_21(unsigned char * v0){
    int v1;
    v1 = f_1(v0);
    unsigned char * v2;
    v2 = (unsigned char *)(v0+16ull);
    switch (v1) {
        case 0: {
            static_array_list<unsigned char,5l> v5;
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
            int v13; static_array<unsigned char,2l> v14;
            Tuple6 tmp9 = f_26(v2);
            v13 = tmp9.v0; v14 = tmp9.v1;
            return Union6{Union6_3{v13, v14}};
            break;
        }
        case 4: {
            int v16; static_array<Tuple0,2l> v17; int v18;
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
    v3 = v1[0l];
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
    dynamic_array_list<Union6,128l> v10{0};
    int v12;
    v12 = f_20(v0);
    v10.unsafe_set_length(v12);
    int v13;
    v13 = v10.length_();
    int v14;
    v14 = 0l;
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
        v14 += 1l ;
    }
    static_array<Union2,2l> v22;
    int v24;
    v24 = 0l;
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
        v24 += 1l ;
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
            int v37; static_array<static_array<unsigned char,2l>,2l> v38; static_array<int,2l> v39; int v40; static_array<int,2l> v41; Union5 v42;
            Tuple2 tmp13 = f_10(v33);
            v37 = tmp13.v0; v38 = tmp13.v1; v39 = tmp13.v2; v40 = tmp13.v3; v41 = tmp13.v4; v42 = tmp13.v5;
            v51 = Union7{Union7_1{v37, v38, v39, v40, v41, v42}};
            break;
        }
        case 2: {
            int v44; static_array<static_array<unsigned char,2l>,2l> v45; static_array<int,2l> v46; int v47; static_array<int,2l> v48; Union5 v49;
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
    v5 = 0ul - v0;
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
    v7 = v1 >> 32l;
    unsigned int v8;
    v8 = (unsigned int)v7;
    int v9;
    v9 = __popc(v6);
    bool v10;
    v10 = v5 < v9;
    unsigned int v17;
    if (v10){
        int v11;
        v11 = v5 + 1l;
        unsigned int v12;
        v12 = __fns(v6,0ul,v11);
        v17 = v12;
    } else {
        int v13;
        v13 = v5 - v9;
        int v14;
        v14 = v13 + 1l;
        unsigned int v15;
        v15 = __fns(v8,0ul,v14);
        unsigned int v16;
        v16 = v15 + 32ul;
        v17 = v16;
    }
    unsigned char v18;
    v18 = (unsigned char)v17;
    int v19;
    v19 = (int)v17;
    unsigned long long v20;
    v20 = 1ull << v19;
    unsigned long long v21;
    v21 = v1 ^ v20;
    return Tuple10{v18, v21};
}
__device__ Tuple8 draw_cards_32(curandStatePhilox4_32_10_t & v0, unsigned long long v1){
    static_array<unsigned char,3l> v2;
    int v4; unsigned long long v5;
    Tuple9 tmp16 = Tuple9{0l, v1};
    v4 = tmp16.v0; v5 = tmp16.v1;
    while (while_method_1(v4)){
        unsigned char v7; unsigned long long v8;
        Tuple10 tmp17 = draw_card_33(v0, v5);
        v7 = tmp17.v0; v8 = tmp17.v1;
        v2[v4] = v7;
        v5 = v8;
        v4 += 1l ;
    }
    return Tuple8{v2, v5};
}
__device__ static_array_list<unsigned char,5l> get_community_cards_35(Union5 v0, static_array<unsigned char,3l> v1){
    static_array_list<unsigned char,5l> v2;
    v2 = static_array_list<unsigned char,5l>{};
    switch (v0.tag) {
        case 0: { // Flop
            static_array<unsigned char,3l> v4 = v0.case0.v0;
            int v5;
            v5 = 0l;
            while (while_method_1(v5)){
                unsigned char v7;
                v7 = v4[v5];
                v2.push(v7);
                v5 += 1l ;
            }
            break;
        }
        case 1: { // Preflop
            break;
        }
        case 2: { // River
            static_array<unsigned char,5l> v14 = v0.case2.v0;
            int v15;
            v15 = 0l;
            while (while_method_2(v15)){
                unsigned char v17;
                v17 = v14[v15];
                v2.push(v17);
                v15 += 1l ;
            }
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4l> v9 = v0.case3.v0;
            int v10;
            v10 = 0l;
            while (while_method_3(v10)){
                unsigned char v12;
                v12 = v9[v10];
                v2.push(v12);
                v10 += 1l ;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v19;
    v19 = 0l;
    while (while_method_1(v19)){
        unsigned char v21;
        v21 = v1[v19];
        v2.push(v21);
        v19 += 1l ;
    }
    return v2;
}
__device__ bool player_can_act_37(int v0, static_array<static_array<unsigned char,2l>,2l> v1, static_array<int,2l> v2, int v3, static_array<int,2l> v4, Union5 v5){
    int v6;
    v6 = v3 % 2l;
    int v7;
    v7 = v4[v6];
    bool v9;
    v9 = v7 > 0l;
    int v10;
    v10 = v2[v6];
    int v12;
    v12 = v2[0l];
    int v14; int v15;
    Tuple4 tmp19 = Tuple4{1l, v12};
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
        v14 += 1l ;
    }
    bool v21;
    v21 = v10 < v15;
    int v22; int v23;
    Tuple4 tmp20 = Tuple4{0l, 0l};
    v22 = tmp20.v0; v23 = tmp20.v1;
    while (while_method_0(v22)){
        int v25;
        v25 = v4[v22];
        bool v27;
        v27 = 0l < v25;
        int v28;
        if (v27){
            v28 = 1l;
        } else {
            v28 = 0l;
        }
        int v29;
        v29 = v23 + v28;
        v23 = v29;
        v22 += 1l ;
    }
    if (v9){
        if (v21){
            return true;
        } else {
            bool v30;
            v30 = v3 < 2l;
            if (v30){
                bool v31;
                v31 = 0l < v23;
                return v31;
            } else {
                return false;
            }
        }
    } else {
        return false;
    }
}
__device__ Union4 go_next_street_38(int v0, static_array<static_array<unsigned char,2l>,2l> v1, static_array<int,2l> v2, int v3, static_array<int,2l> v4, Union5 v5){
    switch (v5.tag) {
        case 0: { // Flop
            static_array<unsigned char,3l> v7 = v5.case0.v0;
            return Union4{Union4_7{v0, v1, v2, v3, v4, v5}};
            break;
        }
        case 1: { // Preflop
            return Union4{Union4_0{v0, v1, v2, v3, v4, v5}};
            break;
        }
        case 2: { // River
            static_array<unsigned char,5l> v11 = v5.case2.v0;
            return Union4{Union4_6{v0, v1, v2, v3, v4, v5}};
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4l> v9 = v5.case3.v0;
            return Union4{Union4_3{v0, v1, v2, v3, v4, v5}};
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ Union4 try_round_36(int v0, static_array<static_array<unsigned char,2l>,2l> v1, static_array<int,2l> v2, int v3, static_array<int,2l> v4, Union5 v5){
    int v6;
    v6 = v3 + 1l;
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
    static_array<unsigned char,2l> v2;
    int v4; unsigned long long v5;
    Tuple9 tmp21 = Tuple9{0l, v1};
    v4 = tmp21.v0; v5 = tmp21.v1;
    while (while_method_0(v4)){
        unsigned char v7; unsigned long long v8;
        Tuple10 tmp22 = draw_card_33(v0, v5);
        v7 = tmp22.v0; v8 = tmp22.v1;
        v2[v4] = v7;
        v5 = v8;
        v4 += 1l ;
    }
    return Tuple11{v2, v5};
}
__device__ inline bool while_method_6(int v0){
    bool v1;
    v1 = v0 < 1l;
    return v1;
}
__device__ Tuple12 draw_cards_40(curandStatePhilox4_32_10_t & v0, unsigned long long v1){
    static_array<unsigned char,1l> v2;
    int v4; unsigned long long v5;
    Tuple9 tmp25 = Tuple9{0l, v1};
    v4 = tmp25.v0; v5 = tmp25.v1;
    while (while_method_6(v4)){
        unsigned char v7; unsigned long long v8;
        Tuple10 tmp26 = draw_card_33(v0, v5);
        v7 = tmp26.v0; v8 = tmp26.v1;
        v2[v4] = v7;
        v5 = v8;
        v4 += 1l ;
    }
    return Tuple12{v2, v5};
}
__device__ static_array_list<unsigned char,5l> get_community_cards_41(Union5 v0, static_array<unsigned char,1l> v1){
    static_array_list<unsigned char,5l> v2;
    v2 = static_array_list<unsigned char,5l>{};
    switch (v0.tag) {
        case 0: { // Flop
            static_array<unsigned char,3l> v4 = v0.case0.v0;
            int v5;
            v5 = 0l;
            while (while_method_1(v5)){
                unsigned char v7;
                v7 = v4[v5];
                v2.push(v7);
                v5 += 1l ;
            }
            break;
        }
        case 1: { // Preflop
            break;
        }
        case 2: { // River
            static_array<unsigned char,5l> v14 = v0.case2.v0;
            int v15;
            v15 = 0l;
            while (while_method_2(v15)){
                unsigned char v17;
                v17 = v14[v15];
                v2.push(v17);
                v15 += 1l ;
            }
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4l> v9 = v0.case3.v0;
            int v10;
            v10 = 0l;
            while (while_method_3(v10)){
                unsigned char v12;
                v12 = v9[v10];
                v2.push(v12);
                v10 += 1l ;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v19;
    v19 = 0l;
    while (while_method_6(v19)){
        unsigned char v21;
        v21 = v1[v19];
        v2.push(v21);
        v19 += 1l ;
    }
    return v2;
}
__device__ inline bool while_method_7(int v0){
    bool v1;
    v1 = v0 < 65536l;
    return v1;
}
__device__ inline bool while_method_8(int v0){
    bool v1;
    v1 = v0 < 10l;
    return v1;
}
__device__ void method_43(unsigned int v0, float * v1, int v2){
    unsigned int v3;
    v3 = v0 + 1ul;
    bool v4;
    v4 = v3 == 0ul;
    bool v5;
    v5 = v4 != true;
    bool v6;
    v6 = v5 == false;
    if (v6){
        assert("Pickle failure. The input is too large in the binary serializer." && v5);
    } else {
    }
    int v8; unsigned int v9;
    Tuple13 tmp28 = Tuple13{0l, v3};
    v8 = tmp28.v0; v9 = tmp28.v1;
    while (while_method_8(v8)){
        unsigned int v11;
        v11 = v9 & 1ul;
        int v12;
        v12 = v2 + v8;
        float v13;
        v13 = (float)v11;
        v1[v12] = v13;
        unsigned int v14;
        v14 = v9 >> 1l;
        v9 = v14;
        v8 += 1l ;
    }
    bool v15;
    v15 = v9 == 0ul;
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
    v1 = v0 < 11l;
    return v1;
}
__device__ void method_44(unsigned int v0, float * v1, int v2){
    unsigned int v3;
    v3 = v0 + 1ul;
    bool v4;
    v4 = v3 == 0ul;
    bool v5;
    v5 = v4 != true;
    bool v6;
    v6 = v5 == false;
    if (v6){
        assert("Pickle failure. The input is too large in the binary serializer." && v5);
    } else {
    }
    int v8; unsigned int v9;
    Tuple13 tmp29 = Tuple13{0l, v3};
    v8 = tmp29.v0; v9 = tmp29.v1;
    while (while_method_9(v8)){
        unsigned int v11;
        v11 = v9 & 1ul;
        int v12;
        v12 = v2 + v8;
        float v13;
        v13 = (float)v11;
        v1[v12] = v13;
        unsigned int v14;
        v14 = v9 >> 1l;
        v9 = v14;
        v8 += 1l ;
    }
    bool v15;
    v15 = v9 == 0ul;
    bool v16;
    v16 = v15 == false;
    if (v16){
        assert("Picke failure. The remains of the input has to equal zero in the binary pickler." && v15);
        return ;
    } else {
        return ;
    }
}
__device__ int int_range_45(int v0, int v1, curandStatePhilox4_32_10_t & v2){
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
    v1 = v0 < 8l;
    return v1;
}
__device__ inline bool while_method_11(int v0){
    bool v1;
    v1 = v0 < 256l;
    return v1;
}
__device__ void method_46(float * v0, int v1, float * v2, int v3, float * v4, int v5){
    unsigned int v6;
    v6 = 0ul;
    asm("mov.u32 %0, %dynamic_smem_size;" : "=r"(v6));
    unsigned long long v7;
    v7 = (unsigned long long)v6;
    bool v8;
    v8 = 1536ull <= v7;
    bool v9;
    v9 = v8 == false;
    if (v9){
        assert("The shared memory used in the matmult node is lower than the allocated amount." && v8);
    } else {
    }
    extern __shared__ unsigned char v11[];
    float * v12;
    v12 = reinterpret_cast<float *>(&v11[0ull]);
    float * v14;
    v14 = reinterpret_cast<float *>(&v11[768ull]);
    float * v16;
    v16 = reinterpret_cast<float *>(&v11[0ull]);
    int v18;
    v18 = threadIdx.x;
    int v19;
    v19 = v18 / 32l;
    bool v20;
    v20 = 0l <= v19;
    bool v21;
    v21 = v20 == false;
    if (v21){
        assert("The index needs to be zero or positive." && v20);
    } else {
    }
    int v23;
    v23 = v19 % 1l;
    bool v24;
    v24 = v19 < 1l;
    bool v25;
    v25 = v24 == false;
    if (v25){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v24);
    } else {
    }
    assert("Tensor range check" && 0 <= v19 && v19 < 1l);
    assert("Tensor range check" && 0 <= v23 && v23 < 1l);
    int v27;
    v27 = 16l * v23;
    int v28;
    v28 = 384l * v19;
    int v29;
    v29 = v28 + v27;
    float * v30;
    v30 = v16+v29;
    assert("Tensor range check" && 0 <= v19 && v19 < 1l);
    int v32;
    v32 = 192l * v19;
    int v33;
    v33 = threadIdx.x;
    int v34;
    v34 = v33 % 32l;
    bool v35;
    v35 = 0l <= v34;
    bool v36;
    v36 = v35 == false;
    if (v36){
        assert("The index needs to be zero or positive." && v35);
    } else {
    }
    int v38;
    v38 = v34 % 4l;
    int v39;
    v39 = v34 / 4l;
    bool v40;
    v40 = v39 < 8l;
    bool v41;
    v41 = v40 == false;
    if (v41){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v40);
    } else {
    }
    assert("Tensor range check" && 0 <= v39 && v39 < 8l);
    assert("Tensor range check" && 0 <= v38 && v38 < 4l);
    int v43;
    v43 = v38 + v32;
    int v44;
    v44 = 12l * v39;
    int v45;
    v45 = v44 + v43;
    float * v46;
    v46 = v12+v45;
    assert("Tensor range check" && 0 <= v23 && v23 < 1l);
    int v48;
    v48 = 192l * v23;
    int v49;
    v49 = threadIdx.x;
    int v50;
    v50 = v49 % 32l;
    bool v51;
    v51 = 0l <= v50;
    bool v52;
    v52 = v51 == false;
    if (v52){
        assert("The index needs to be zero or positive." && v51);
    } else {
    }
    int v54;
    v54 = v50 % 4l;
    int v55;
    v55 = v50 / 4l;
    bool v56;
    v56 = v55 < 8l;
    bool v57;
    v57 = v56 == false;
    if (v57){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v56);
    } else {
    }
    assert("Tensor range check" && 0 <= v55 && v55 < 8l);
    assert("Tensor range check" && 0 <= v54 && v54 < 4l);
    int v59;
    v59 = v54 + v48;
    int v60;
    v60 = 12l * v55;
    int v61;
    v61 = v60 + v59;
    float * v62;
    v62 = v14+v61;
    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> v64[1l];
    int v65;
    v65 = 0l;
    while (while_method_0(v65)){
        int v67;
        v67 = 0l;
        while (while_method_10(v67)){
            assert("Tensor range check" && 0 <= v65 && v65 < 2l);
            assert("Tensor range check" && 0 <= v67 && v67 < 8l);
            int v69;
            v69 = 16l * v67;
            int v70;
            v70 = v69 + v3;
            int v71;
            v71 = 2048l * v65;
            int v72;
            v72 = v71 + v70;
            float * v73;
            v73 = v2+v72;
            // Pushing the loop unrolling to: 0
            int v75;
            v75 = 0l;
            #pragma unroll
            while (while_method_6(v75)){
                int v77;
                v77 = 0l;
                #pragma unroll
                while (while_method_6(v77)){
                    assert("Tensor range check" && 0 <= v75 && v75 < 1l);
                    assert("Tensor range check" && 0 <= v77 && v77 < 1l);
                    int v79;
                    v79 = v75 + v77;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v80 = v64[v79];
                    wmma::fill_fragment(v80, 0.0f);
                    v77 += 1l ;
                }
                v75 += 1l ;
            }
            int v81;
            v81 = 0l;
            #pragma unroll
            while (while_method_11(v81)){
                assert("Tensor range check" && 0 <= v65 && v65 < 2l);
                int v83;
                v83 = 32768l * v65;
                int v84;
                v84 = v83 + v5;
                assert("Tensor range check" && 0 <= v81 && v81 < 256l);
                int v85;
                v85 = 8l * v81;
                int v86;
                v86 = v85 + v84;
                float * v87;
                v87 = v4+v86;
                assert("Tensor range check" && 0 <= v67 && v67 < 8l);
                int v89;
                v89 = 32768l * v67;
                int v90;
                v90 = v89 + v1;
                assert("Tensor range check" && 0 <= v81 && v81 < 256l);
                int v91;
                v91 = v85 + v90;
                float * v92;
                v92 = v0+v91;
                int v94;
                v94 = threadIdx.x;
                bool v95;
                v95 = 0l <= v94;
                bool v96;
                v96 = v95 == false;
                if (v96){
                    assert("The index needs to be zero or positive." && v95);
                } else {
                }
                int v98;
                v98 = v94 % 2l;
                int v99;
                v99 = v94 / 2l;
                bool v100;
                v100 = v99 < 16l;
                bool v101;
                v101 = v100 == false;
                if (v101){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v100);
                } else {
                }
                assert("Tensor range check" && 0 <= v99 && v99 < 16l);
                assert("Tensor range check" && 0 <= v98 && v98 < 2l);
                int v103;
                v103 = 4l * v98;
                int v104;
                v104 = 12l * v99;
                int v105;
                v105 = v104 + v103;
                int v106;
                v106 = 2048l * v99;
                int v107;
                v107 = v106 + v103;
                float * v108;
                v108 = v14+v105;
                float * v110;
                v110 = v92+v107;
                int v112;
                v112 = 0l;
                #pragma unroll
                while (while_method_6(v112)){
                    int v114;
                    v114 = 0l;
                    #pragma unroll
                    while (while_method_6(v114)){
                        assert("Tensor range check" && 0 <= v112 && v112 < 1l);
                        assert("Tensor range check" && 0 <= v114 && v114 < 1l);
                        int v116;
                        v116 = 8l * v114;
                        int v117;
                        v117 = 192l * v112;
                        int v118;
                        v118 = v117 + v116;
                        int v119;
                        v119 = 32768l * v112;
                        int v120;
                        v120 = v119 + v116;
                        float v121[4l];
                        int v122;
                        v122 = 0l;
                        #pragma unroll
                        while (while_method_3(v122)){
                            assert("Tensor range check" && 0 <= v122 && v122 < 4l);
                            int v124;
                            v124 = v122 + v120;
                            float v125;
                            v125 = v110[v124];
                            float v126;
                            v126 = wmma::__float_to_tf32(v125);
                            assert("Tensor range check" && 0 <= v122 && v122 < 4l);
                            v121[v122] = v126;
                            v122 += 1l ;
                        }
                        int4* v127;
                        v127 = reinterpret_cast<int4*>(v121 + 0l);
                        int4* v128;
                        v128 = reinterpret_cast<int4*>(v108 + v118);
                        assert("Pointer alignment check" && (unsigned long long)(v127) % 4l == 0 && (unsigned long long)(v128) % 4l == 0);
                        *v128 = *v127;
                        v114 += 1l ;
                    }
                    v112 += 1l ;
                }
                int v129;
                v129 = threadIdx.x;
                bool v130;
                v130 = 0l <= v129;
                bool v131;
                v131 = v130 == false;
                if (v131){
                    assert("The index needs to be zero or positive." && v130);
                } else {
                }
                int v133;
                v133 = v129 % 2l;
                int v134;
                v134 = v129 / 2l;
                bool v135;
                v135 = v134 < 16l;
                bool v136;
                v136 = v135 == false;
                if (v136){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v135);
                } else {
                }
                assert("Tensor range check" && 0 <= v134 && v134 < 16l);
                assert("Tensor range check" && 0 <= v133 && v133 < 2l);
                int v138;
                v138 = 4l * v133;
                int v139;
                v139 = 12l * v134;
                int v140;
                v140 = v139 + v138;
                int v141;
                v141 = 2048l * v134;
                int v142;
                v142 = v141 + v138;
                float * v143;
                v143 = v12+v140;
                float * v145;
                v145 = v87+v142;
                int v147;
                v147 = 0l;
                #pragma unroll
                while (while_method_6(v147)){
                    int v149;
                    v149 = 0l;
                    #pragma unroll
                    while (while_method_6(v149)){
                        assert("Tensor range check" && 0 <= v147 && v147 < 1l);
                        assert("Tensor range check" && 0 <= v149 && v149 < 1l);
                        int v151;
                        v151 = 8l * v149;
                        int v152;
                        v152 = 192l * v147;
                        int v153;
                        v153 = v152 + v151;
                        int v154;
                        v154 = 32768l * v147;
                        int v155;
                        v155 = v154 + v151;
                        float v156[4l];
                        int v157;
                        v157 = 0l;
                        #pragma unroll
                        while (while_method_3(v157)){
                            assert("Tensor range check" && 0 <= v157 && v157 < 4l);
                            int v159;
                            v159 = v157 + v155;
                            float v160;
                            v160 = v145[v159];
                            float v161;
                            v161 = wmma::__float_to_tf32(v160);
                            assert("Tensor range check" && 0 <= v157 && v157 < 4l);
                            v156[v157] = v161;
                            v157 += 1l ;
                        }
                        int4* v162;
                        v162 = reinterpret_cast<int4*>(v156 + 0l);
                        int4* v163;
                        v163 = reinterpret_cast<int4*>(v143 + v153);
                        assert("Pointer alignment check" && (unsigned long long)(v162) % 4l == 0 && (unsigned long long)(v163) % 4l == 0);
                        *v163 = *v162;
                        v149 += 1l ;
                    }
                    v147 += 1l ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0l));
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v164[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v165[1l];
                int v166;
                v166 = 0l;
                #pragma unroll
                while (while_method_6(v166)){
                    int v168;
                    v168 = 0l;
                    #pragma unroll
                    while (while_method_6(v168)){
                        assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                        assert("Tensor range check" && 0 <= v168 && v168 < 1l);
                        int v170;
                        v170 = v166 + v168;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v171 = v164[v170];
                        assert("Tensor range check" && 0 <= v166 && v166 < 1l);
                        int v172;
                        v172 = 192l * v166;
                        assert("Tensor range check" && 0 <= v168 && v168 < 1l);
                        int v173;
                        v173 = 8l * v168;
                        int v174;
                        v174 = v173 + v172;
                        int v175;
                        v175 = 0l;
                        #pragma unroll
                        while (while_method_0(v175)){
                            int v177;
                            v177 = 0l;
                            #pragma unroll
                            while (while_method_0(v177)){
                                assert("Tensor range check" && 0 <= v175 && v175 < 2l);
                                assert("Tensor range check" && 0 <= v177 && v177 < 2l);
                                int v179;
                                v179 = 96l * v177;
                                int v180;
                                v180 = v179 + v174;
                                int v181;
                                v181 = 4l * v175;
                                int v182;
                                v182 = v181 + v180;
                                float v183;
                                v183 = v46[v182];
                                bool v184;
                                v184 = 0l <= v177;
                                bool v186;
                                if (v184){
                                    bool v185;
                                    v185 = v177 < 2l;
                                    v186 = v185;
                                } else {
                                    v186 = false;
                                }
                                bool v187;
                                v187 = v186 == false;
                                if (v187){
                                    assert("The indices should be inside the range of the dimension." && v186);
                                } else {
                                }
                                bool v189;
                                v189 = 0l <= v175;
                                bool v191;
                                if (v189){
                                    bool v190;
                                    v190 = v175 < 2l;
                                    v191 = v190;
                                } else {
                                    v191 = false;
                                }
                                bool v192;
                                v192 = v191 == false;
                                if (v192){
                                    assert("The indices should be inside the range of the dimension." && v191);
                                } else {
                                }
                                int v194;
                                v194 = v175 * 2l;
                                int v195;
                                v195 = v177 + v194;
                                v171.x[v195] = v183;
                                v177 += 1l ;
                            }
                            v175 += 1l ;
                        }
                        v168 += 1l ;
                    }
                    v166 += 1l ;
                }
                int v196;
                v196 = 0l;
                #pragma unroll
                while (while_method_6(v196)){
                    int v198;
                    v198 = 0l;
                    #pragma unroll
                    while (while_method_6(v198)){
                        assert("Tensor range check" && 0 <= v196 && v196 < 1l);
                        assert("Tensor range check" && 0 <= v198 && v198 < 1l);
                        int v200;
                        v200 = v196 + v198;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v201 = v165[v200];
                        assert("Tensor range check" && 0 <= v196 && v196 < 1l);
                        int v202;
                        v202 = 192l * v196;
                        assert("Tensor range check" && 0 <= v198 && v198 < 1l);
                        int v203;
                        v203 = 8l * v198;
                        int v204;
                        v204 = v203 + v202;
                        int v205;
                        v205 = 0l;
                        #pragma unroll
                        while (while_method_0(v205)){
                            int v207;
                            v207 = 0l;
                            #pragma unroll
                            while (while_method_0(v207)){
                                assert("Tensor range check" && 0 <= v205 && v205 < 2l);
                                assert("Tensor range check" && 0 <= v207 && v207 < 2l);
                                int v209;
                                v209 = 4l * v207;
                                int v210;
                                v210 = v209 + v204;
                                int v211;
                                v211 = 96l * v205;
                                int v212;
                                v212 = v211 + v210;
                                float v213;
                                v213 = v62[v212];
                                bool v214;
                                v214 = 0l <= v207;
                                bool v216;
                                if (v214){
                                    bool v215;
                                    v215 = v207 < 2l;
                                    v216 = v215;
                                } else {
                                    v216 = false;
                                }
                                bool v217;
                                v217 = v216 == false;
                                if (v217){
                                    assert("The indices should be inside the range of the dimension." && v216);
                                } else {
                                }
                                bool v219;
                                v219 = 0l <= v205;
                                bool v221;
                                if (v219){
                                    bool v220;
                                    v220 = v205 < 2l;
                                    v221 = v220;
                                } else {
                                    v221 = false;
                                }
                                bool v222;
                                v222 = v221 == false;
                                if (v222){
                                    assert("The indices should be inside the range of the dimension." && v221);
                                } else {
                                }
                                int v224;
                                v224 = v205 * 2l;
                                int v225;
                                v225 = v207 + v224;
                                v201.x[v225] = v213;
                                v207 += 1l ;
                            }
                            v205 += 1l ;
                        }
                        v198 += 1l ;
                    }
                    v196 += 1l ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0l));
                int v226;
                v226 = 0l;
                #pragma unroll
                while (while_method_6(v226)){
                    int v228;
                    v228 = 0l;
                    #pragma unroll
                    while (while_method_6(v228)){
                        int v230;
                        v230 = 0l;
                        #pragma unroll
                        while (while_method_6(v230)){
                            assert("Tensor range check" && 0 <= v226 && v226 < 1l);
                            assert("Tensor range check" && 0 <= v228 && v228 < 1l);
                            int v232;
                            v232 = v226 + v228;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v233 = v64[v232];
                            assert("Tensor range check" && 0 <= v226 && v226 < 1l);
                            assert("Tensor range check" && 0 <= v230 && v230 < 1l);
                            int v234;
                            v234 = v226 + v230;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v235 = v164[v234];
                            assert("Tensor range check" && 0 <= v228 && v228 < 1l);
                            assert("Tensor range check" && 0 <= v230 && v230 < 1l);
                            int v236;
                            v236 = v228 + v230;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v237 = v165[v236];
                            wmma::mma_sync(v233, v235, v237, v233);
                            v230 += 1l ;
                        }
                        v228 += 1l ;
                    }
                    v226 += 1l ;
                }
                v81 += 1l ;
            }
            int v238;
            v238 = 0l;
            #pragma unroll
            while (while_method_6(v238)){
                int v240;
                v240 = 0l;
                #pragma unroll
                while (while_method_6(v240)){
                    assert("Tensor range check" && 0 <= v238 && v238 < 1l);
                    assert("Tensor range check" && 0 <= v240 && v240 < 1l);
                    int v242;
                    v242 = v238 + v240;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v243 = v64[v242];
                    assert("Tensor range check" && 0 <= v238 && v238 < 1l);
                    assert("Tensor range check" && 0 <= v240 && v240 < 1l);
                    int v244;
                    v244 = 16l * v240;
                    int v245;
                    v245 = 384l * v238;
                    int v246;
                    v246 = v245 + v244;
                    float * v247;
                    v247 = v30+v246;
                    wmma::store_matrix_sync(v247, v243, 24l, wmma::mem_row_major);
                    v240 += 1l ;
                }
                v238 += 1l ;
            }
            asm("barrier.cta.sync %0;" :: "r"(0l));
            int v249;
            v249 = threadIdx.x;
            bool v250;
            v250 = 0l <= v249;
            bool v251;
            v251 = v250 == false;
            if (v251){
                assert("The index needs to be zero or positive." && v250);
            } else {
            }
            int v253;
            v253 = v249 % 4l;
            int v254;
            v254 = v249 / 4l;
            bool v255;
            v255 = v254 < 8l;
            bool v256;
            v256 = v255 == false;
            if (v256){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v255);
            } else {
            }
            assert("Tensor range check" && 0 <= v254 && v254 < 8l);
            assert("Tensor range check" && 0 <= v253 && v253 < 4l);
            int v258;
            v258 = 4l * v253;
            int v259;
            v259 = 128l * v254;
            int v260;
            v260 = v259 + v258;
            int v261;
            v261 = 24l * v254;
            int v262;
            v262 = v261 + v258;
            float * v263;
            v263 = v73+v260;
            float * v265;
            v265 = v16+v262;
            int v267;
            v267 = 0l;
            #pragma unroll
            while (while_method_0(v267)){
                int v269;
                v269 = 0l;
                #pragma unroll
                while (while_method_6(v269)){
                    assert("Tensor range check" && 0 <= v267 && v267 < 2l);
                    assert("Tensor range check" && 0 <= v269 && v269 < 1l);
                    int v271;
                    v271 = 16l * v269;
                    int v272;
                    v272 = 1024l * v267;
                    int v273;
                    v273 = v272 + v271;
                    int v274;
                    v274 = 192l * v267;
                    int v275;
                    v275 = v274 + v271;
                    int4* v276;
                    v276 = reinterpret_cast<int4*>(v265 + v275);
                    int4* v277;
                    v277 = reinterpret_cast<int4*>(v263 + v273);
                    assert("Pointer alignment check" && (unsigned long long)(v276) % 4l == 0 && (unsigned long long)(v277) % 4l == 0);
                    *v277 = *v276;
                    v269 += 1l ;
                }
                v267 += 1l ;
            }
            asm("barrier.cta.sync %0;" :: "r"(0l));
            // Poping the loop unrolling to: 0
            v67 += 1l ;
        }
        v65 += 1l ;
    }
    return ;
}
__device__ inline bool while_method_12(int v0){
    bool v1;
    v1 = v0 < 32l;
    return v1;
}
__device__ void method_47(unsigned int * v0, int v1, float * v2){
    int v3;
    v3 = blockIdx.x;
    assert("Tensor range check" && 0 <= v3 && v3 < 1l);
    int v4;
    v4 = 4096l * v3;
    int v5;
    v5 = blockIdx.x;
    assert("Tensor range check" && 0 <= v5 && v5 < 1l);
    int v6;
    v6 = 32l * v5;
    int v7;
    v7 = v6 + v1;
    int v8;
    v8 = threadIdx.x;
    bool v9;
    v9 = 0l <= v8;
    bool v10;
    v10 = v9 == false;
    if (v10){
        assert("The index needs to be zero or positive." && v9);
    } else {
    }
    int v12;
    v12 = v8 % 32l;
    int v13;
    v13 = v8 / 32l;
    bool v14;
    v14 = v13 < 1l;
    bool v15;
    v15 = v14 == false;
    if (v15){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v14);
    } else {
    }
    assert("Tensor range check" && 0 <= v13 && v13 < 1l);
    assert("Tensor range check" && 0 <= v12 && v12 < 32l);
    int v17;
    v17 = 4l * v12;
    int v18;
    v18 = v17 + v4;
    int v19;
    v19 = 128l * v13;
    int v20;
    v20 = v19 + v18;
    assert("Tensor range check" && 0 <= v13 && v13 < 1l);
    int v21;
    v21 = v13 + v7;
    int v22;
    v22 = 0l;
    while (while_method_12(v22)){
        assert("Tensor range check" && 0 <= v22 && v22 < 32l);
        int v24;
        v24 = 128l * v22;
        int v25;
        v25 = v24 + v20;
        float v26[4l];
        int v27[4l];
        int v28;
        v28 = 0l;
        while (while_method_6(v28)){
            assert("Tensor range check" && 0 <= v28 && v28 < 1l);
            int v30;
            v30 = 4l * v28;
            assert("Tensor range check" && 0 <= v28 && v28 < 1l);
            int v31;
            v31 = 128l * v28;
            int v32;
            v32 = v31 + v25;
            int4* v33;
            v33 = reinterpret_cast<int4*>(v2 + v32);
            int4* v34;
            v34 = reinterpret_cast<int4*>(v26 + v30);
            assert("Pointer alignment check" && (unsigned long long)(v33) % 4l == 0 && (unsigned long long)(v34) % 4l == 0);
            *v34 = *v33;
            v28 += 1l ;
        }
        int v35;
        v35 = 0l;
        while (while_method_6(v35)){
            int v37;
            v37 = 0l;
            while (while_method_3(v37)){
                bool v39;
                v39 = 0l <= v37;
                bool v41;
                if (v39){
                    bool v40;
                    v40 = v37 < 4l;
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
                v44 = 0l <= v12;
                bool v46;
                if (v44){
                    bool v45;
                    v45 = v12 < 32l;
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
                v49 = v12 * 4l;
                int v50;
                v50 = v37 + v49;
                bool v51;
                v51 = 0l <= v35;
                bool v53;
                if (v51){
                    bool v52;
                    v52 = v35 < 1l;
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
                v56 = v35 * 128l;
                int v57;
                v57 = v50 + v56;
                assert("Tensor range check" && 0 <= v35 && v35 < 1l);
                assert("Tensor range check" && 0 <= v37 && v37 < 4l);
                int v58;
                v58 = 4l * v35;
                int v59;
                v59 = v58 + v37;
                v27[v59] = v57;
                v37 += 1l ;
            }
            v35 += 1l ;
        }
        bool v60;
        v60 = 0l <= v13;
        bool v61;
        v61 = v60 && v14;
        bool v62;
        v62 = v61 == false;
        if (v62){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v61);
        } else {
        }
        bool v64;
        v64 = 0l <= v22;
        bool v66;
        if (v64){
            bool v65;
            v65 = v22 < 32l;
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
        v69 = v22 + v13;
        unsigned int v70[4l];
        int v71;
        v71 = 0l;
        while (while_method_6(v71)){
            int v73;
            v73 = 0l;
            while (while_method_3(v73)){
                assert("Tensor range check" && 0 <= v71 && v71 < 1l);
                assert("Tensor range check" && 0 <= v73 && v73 < 4l);
                int v75;
                v75 = 4l * v71;
                int v76;
                v76 = v75 + v73;
                float v77;
                v77 = v26[v76];
                int v78;
                v78 = v27[v76];
                bool v79;
                v79 = v77 <= 0.0f;
                unsigned int v81;
                if (v79){
                    v81 = 0ul;
                } else {
                    unsigned int v80;
                    v80 = 1ul << v78;
                    v81 = v80;
                }
                assert("Tensor range check" && 0 <= v71 && v71 < 1l);
                assert("Tensor range check" && 0 <= v73 && v73 < 4l);
                v70[v76] = v81;
                v73 += 1l ;
            }
            v71 += 1l ;
        }
        unsigned int v82;
        v82 = 0ul;
        int v83;
        v83 = 0l;
        while (while_method_6(v83)){
            int v85;
            v85 = 0l;
            while (while_method_3(v85)){
                assert("Tensor range check" && 0 <= v83 && v83 < 1l);
                assert("Tensor range check" && 0 <= v85 && v85 < 4l);
                int v87;
                v87 = 4l * v83;
                int v88;
                v88 = v87 + v85;
                unsigned int v89;
                v89 = v70[v88];
                unsigned int v90;
                v90 = v82 | v89;
                v82 = v90;
                v85 += 1l ;
            }
            v83 += 1l ;
        }
        auto v91 = cooperative_groups::coalesced_threads();
        int v92;
        v92 = threadIdx.x;
        int v93;
        v93 = v92 / 32l;
        auto v94 = cooperative_groups::labeled_partition(v91,v93);
        Closure0 v95{};
        unsigned int v96;
        v96 = cooperative_groups::reduce(v94, v82, v95);
        unsigned int v97;
        v97 = v96 % 4096ul;
        assert("Tensor range check" && 0 <= v22 && v22 < 32l);
        int v98;
        v98 = v22 + v21;
        v0[v98] = v97;
        v22 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return ;
}
__device__ Tuple14 method_48(float * v0, float * v1, float * v2, float * v3, float * v4, int v5, int v6){
    assert("Tensor range check" && 0 <= v6 && v6 < 4l);
    int v7;
    v7 = 16384l * v6;
    assert("Tensor range check" && 0 <= v5 && v5 < 4096l);
    int v8;
    v8 = 4l * v5;
    int v9;
    v9 = v8 + v7;
    float * v10;
    v10 = v0+v9;
    float * v12;
    v12 = v1+v9;
    __shared__ float * v14[32l];
    __shared__ float * v15[32l];
    /* void shared array create v16 */;
    __shared__ float v17[32l];
    __shared__ float v18[32l];
    __shared__ int v19[32l];
    int v20;
    v20 = threadIdx.x;
    bool v21;
    v21 = v20 < 32l;
    if (v21){
        assert("Tensor range check" && 0 <= v20 && v20 < 32l);
        v14[v20] = v10;
        v15[v20] = v12;
        /* void array set */;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    bool v22;
    v22 = 0l <= v20;
    bool v23;
    v23 = v22 == false;
    if (v23){
        assert("The index needs to be zero or positive." && v22);
    } else {
    }
    int v25;
    v25 = v20 % 1l;
    bool v26;
    v26 = v21 == false;
    if (v26){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v21);
    } else {
    }
    assert("Tensor range check" && 0 <= v20 && v20 < 32l);
    int v28;
    v28 = 0l;
    while (while_method_6(v28)){
        bool v30;
        v30 = v22 && v21;
        bool v31;
        v31 = v30 == false;
        if (v31){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v30);
        } else {
        }
        bool v33;
        v33 = 0l <= v28;
        bool v35;
        if (v33){
            bool v34;
            v34 = v28 < 1l;
            v35 = v34;
        } else {
            v35 = false;
        }
        bool v36;
        v36 = v35 == false;
        if (v36){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v35);
        } else {
        }
        int v38;
        v38 = v28 * 32l;
        int v39;
        v39 = v38 + v20;
        assert("Tensor range check" && 0 <= v28 && v28 < 1l);
        int v40;
        v40 = 32l * v28;
        int v41;
        v41 = v40 + v20;
        float * v42;
        v42 = v14[v41];
        float * v43;
        v43 = v15[v41];
        /* void array index */;
        assert("Tensor range check" && 0 <= v25 && v25 < 1l);
        int v44;
        v44 = 4l * v25;
        float v45[4l];
        float v46[4l];
        int v47[4l];
        int v48;
        v48 = 0l;
        while (while_method_6(v48)){
            assert("Tensor range check" && 0 <= v48 && v48 < 1l);
            int v50;
            v50 = 4l * v48;
            assert("Tensor range check" && 0 <= v48 && v48 < 1l);
            int v51;
            v51 = v50 + v44;
            int4* v52;
            v52 = reinterpret_cast<int4*>(v42 + v51);
            int4* v53;
            v53 = reinterpret_cast<int4*>(v45 + v50);
            assert("Pointer alignment check" && (unsigned long long)(v52) % 4l == 0 && (unsigned long long)(v53) % 4l == 0);
            *v53 = *v52;
            int4* v54;
            v54 = reinterpret_cast<int4*>(v43 + v51);
            int4* v55;
            v55 = reinterpret_cast<int4*>(v46 + v50);
            assert("Pointer alignment check" && (unsigned long long)(v54) % 4l == 0 && (unsigned long long)(v55) % 4l == 0);
            *v55 = *v54;
            v48 += 1l ;
        }
        int v56;
        v56 = 0l;
        while (while_method_6(v56)){
            int v58;
            v58 = 0l;
            while (while_method_3(v58)){
                bool v60;
                v60 = 0l <= v58;
                bool v62;
                if (v60){
                    bool v61;
                    v61 = v58 < 4l;
                    v62 = v61;
                } else {
                    v62 = false;
                }
                bool v63;
                v63 = v62 == false;
                if (v63){
                    assert("The indices should be inside the range of the dimension." && v62);
                } else {
                }
                bool v65;
                v65 = 0l <= v25;
                bool v67;
                if (v65){
                    bool v66;
                    v66 = v25 < 1l;
                    v67 = v66;
                } else {
                    v67 = false;
                }
                bool v68;
                v68 = v67 == false;
                if (v68){
                    assert("The indices should be inside the range of the dimension." && v67);
                } else {
                }
                int v70;
                v70 = v25 * 4l;
                int v71;
                v71 = v58 + v70;
                bool v72;
                v72 = 0l <= v56;
                bool v74;
                if (v72){
                    bool v73;
                    v73 = v56 < 1l;
                    v74 = v73;
                } else {
                    v74 = false;
                }
                bool v75;
                v75 = v74 == false;
                if (v75){
                    assert("The indices should be inside the range of the dimension." && v74);
                } else {
                }
                int v77;
                v77 = v56 * 4l;
                int v78;
                v78 = v71 + v77;
                assert("Tensor range check" && 0 <= v56 && v56 < 1l);
                assert("Tensor range check" && 0 <= v58 && v58 < 4l);
                int v79;
                v79 = 4l * v56;
                int v80;
                v80 = v79 + v58;
                v47[v80] = v78;
                v58 += 1l ;
            }
            v56 += 1l ;
        }
        unsigned long long v81;
        v81 = clock64();
        int v82;
        v82 = threadIdx.x;
        unsigned long long v83;
        v83 = (unsigned long long)v82;
        curandStatePhilox4_32_10_t v84;
        curand_init(v81,v83,0ull,&v84);
        bool v85[4l];
        int v86;
        v86 = 0l;
        while (while_method_6(v86)){
            int v88;
            v88 = 0l;
            while (while_method_3(v88)){
                assert("Tensor range check" && 0 <= v86 && v86 < 1l);
                assert("Tensor range check" && 0 <= v88 && v88 < 4l);
                int v90;
                v90 = 4l * v86;
                int v91;
                v91 = v90 + v88;
                float v92;
                v92 = v45[v91];
                int v93;
                v93 = v47[v91];
                bool v94;
                v94 = v93 < 11l;
                assert("Tensor range check" && 0 <= v86 && v86 < 1l);
                assert("Tensor range check" && 0 <= v88 && v88 < 4l);
                v85[v91] = v94;
                v88 += 1l ;
            }
            v86 += 1l ;
        }
        float v95[4l];
        int v96;
        v96 = 0l;
        while (while_method_6(v96)){
            int v98;
            v98 = 0l;
            while (while_method_3(v98)){
                assert("Tensor range check" && 0 <= v96 && v96 < 1l);
                assert("Tensor range check" && 0 <= v98 && v98 < 4l);
                int v100;
                v100 = 4l * v96;
                int v101;
                v101 = v100 + v98;
                float v102;
                v102 = v45[v101];
                bool v103;
                v103 = v85[v101];
                float v106;
                if (v103){
                    bool v104;
                    v104 = 0.0f >= v102;
                    if (v104){
                        v106 = 0.0f;
                    } else {
                        v106 = v102;
                    }
                } else {
                    v106 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v96 && v96 < 1l);
                assert("Tensor range check" && 0 <= v98 && v98 < 4l);
                v95[v101] = v106;
                v98 += 1l ;
            }
            v96 += 1l ;
        }
        float v107;
        v107 = 0.0f;
        int v108;
        v108 = 0l;
        while (while_method_6(v108)){
            int v110;
            v110 = 0l;
            while (while_method_3(v110)){
                assert("Tensor range check" && 0 <= v108 && v108 < 1l);
                assert("Tensor range check" && 0 <= v110 && v110 < 4l);
                int v112;
                v112 = 4l * v108;
                int v113;
                v113 = v112 + v110;
                float v114;
                v114 = v95[v113];
                float v115;
                v115 = v107 + v114;
                v107 = v115;
                v110 += 1l ;
            }
            v108 += 1l ;
        }
        auto v116 = cooperative_groups::coalesced_threads();
        int v117;
        v117 = threadIdx.x;
        auto v118 = cooperative_groups::labeled_partition(v116,v117);
        Closure1 v119{};
        float v120;
        v120 = cooperative_groups::reduce(v118, v107, v119);
        int v121[4l];
        int v122;
        v122 = 0l;
        while (while_method_6(v122)){
            int v124;
            v124 = 0l;
            while (while_method_3(v124)){
                assert("Tensor range check" && 0 <= v122 && v122 < 1l);
                assert("Tensor range check" && 0 <= v124 && v124 < 4l);
                int v126;
                v126 = 4l * v122;
                int v127;
                v127 = v126 + v124;
                bool v128;
                v128 = v85[v127];
                int v129;
                if (v128){
                    v129 = 1l;
                } else {
                    v129 = 0l;
                }
                assert("Tensor range check" && 0 <= v122 && v122 < 1l);
                assert("Tensor range check" && 0 <= v124 && v124 < 4l);
                v121[v127] = v129;
                v124 += 1l ;
            }
            v122 += 1l ;
        }
        int v130;
        v130 = 0l;
        int v131;
        v131 = 0l;
        while (while_method_6(v131)){
            int v133;
            v133 = 0l;
            while (while_method_3(v133)){
                assert("Tensor range check" && 0 <= v131 && v131 < 1l);
                assert("Tensor range check" && 0 <= v133 && v133 < 4l);
                int v135;
                v135 = 4l * v131;
                int v136;
                v136 = v135 + v133;
                int v137;
                v137 = v121[v136];
                int v138;
                v138 = v130 + v137;
                v130 = v138;
                v133 += 1l ;
            }
            v131 += 1l ;
        }
        auto v139 = cooperative_groups::coalesced_threads();
        int v140;
        v140 = threadIdx.x;
        auto v141 = cooperative_groups::labeled_partition(v139,v140);
        Closure2 v142{};
        int v143;
        v143 = cooperative_groups::reduce(v141, v130, v142);
        float v144;
        v144 = (float)v143;
        float v145;
        v145 = 1.0f / v144;
        float v146[4l];
        int v147;
        v147 = 0l;
        while (while_method_6(v147)){
            int v149;
            v149 = 0l;
            while (while_method_3(v149)){
                assert("Tensor range check" && 0 <= v147 && v147 < 1l);
                assert("Tensor range check" && 0 <= v149 && v149 < 4l);
                int v151;
                v151 = 4l * v147;
                int v152;
                v152 = v151 + v149;
                float v153;
                v153 = v95[v152];
                bool v154;
                v154 = v85[v152];
                bool v155;
                v155 = v154 == false;
                float v160;
                if (v155){
                    v160 = 0.0f;
                } else {
                    bool v156;
                    v156 = v120 == 0.0f;
                    bool v157;
                    v157 = v156 != true;
                    if (v157){
                        float v158;
                        v158 = v153 / v120;
                        v160 = v158;
                    } else {
                        v160 = v145;
                    }
                }
                assert("Tensor range check" && 0 <= v147 && v147 < 1l);
                assert("Tensor range check" && 0 <= v149 && v149 < 4l);
                v146[v152] = v160;
                v149 += 1l ;
            }
            v147 += 1l ;
        }
        float v161[4l];
        float v162;
        v162 = 0.0f;
        int v163;
        v163 = 0l;
        while (while_method_6(v163)){
            assert("Tensor range check" && 0 <= v163 && v163 < 1l);
            int v165;
            v165 = 4l * v163;
            assert("Tensor range check" && 0 <= v163 && v163 < 1l);
            int v166; float v167;
            Tuple15 tmp30 = Tuple15{0l, 0.0f};
            v166 = tmp30.v0; v167 = tmp30.v1;
            while (while_method_3(v166)){
                assert("Tensor range check" && 0 <= v166 && v166 < 4l);
                int v169;
                v169 = v166 + v165;
                float v170;
                v170 = v146[v169];
                float v171;
                v171 = v167 + v170;
                v167 = v171;
                v166 += 1l ;
            }
            auto v172 = cooperative_groups::coalesced_threads();
            int v173;
            v173 = threadIdx.x;
            auto v174 = cooperative_groups::labeled_partition(v172,v173);
            Closure3 v175{};
            float v176;
            v176 = cooperative_groups::inclusive_scan(v174, v167, v175);
            float v177;
            v177 = v174.shfl_up(v176,1);
            bool v178;
            v178 = v174.thread_rank() == 0;
            float v179;
            if (v178){
                v179 = 0.0f;
            } else {
                v179 = v177;
            }
            float v180;
            v180 = v174.shfl(v176,v174.num_threads()-1);
            float v181;
            v181 = v162 + v179;
            int v182; float v183;
            Tuple15 tmp31 = Tuple15{0l, v181};
            v182 = tmp31.v0; v183 = tmp31.v1;
            while (while_method_3(v182)){
                assert("Tensor range check" && 0 <= v182 && v182 < 4l);
                int v185;
                v185 = v182 + v165;
                float v186;
                v186 = v146[v185];
                float v187;
                v187 = v183 + v186;
                assert("Tensor range check" && 0 <= v182 && v182 < 4l);
                v161[v185] = v187;
                v183 = v187;
                v182 += 1l ;
            }
            float v188;
            v188 = v162 + v180;
            v162 = v188;
            v163 += 1l ;
        }
        float v189[4l];
        bool v190[4l];
        int v191;
        v191 = 0l;
        while (while_method_6(v191)){
            int v193;
            v193 = 0l;
            while (while_method_3(v193)){
                assert("Tensor range check" && 0 <= v191 && v191 < 1l);
                assert("Tensor range check" && 0 <= v193 && v193 < 4l);
                int v195;
                v195 = 4l * v191;
                int v196;
                v196 = v195 + v193;
                float v197;
                v197 = v161[v196];
                float v198;
                v198 = v146[v196];
                bool v199;
                v199 = v198 > 0.0f;
                assert("Tensor range check" && 0 <= v191 && v191 < 1l);
                assert("Tensor range check" && 0 <= v193 && v193 < 4l);
                v189[v196] = v197;
                v190[v196] = v199;
                v193 += 1l ;
            }
            v191 += 1l ;
        }
        float v200; bool v201;
        Tuple16 tmp32 = Tuple16{-1.0f / 0.0f, false};
        v200 = tmp32.v0; v201 = tmp32.v1;
        int v202;
        v202 = 0l;
        while (while_method_6(v202)){
            int v204;
            v204 = 0l;
            while (while_method_3(v204)){
                assert("Tensor range check" && 0 <= v202 && v202 < 1l);
                assert("Tensor range check" && 0 <= v204 && v204 < 4l);
                int v206;
                v206 = 4l * v202;
                int v207;
                v207 = v206 + v204;
                float v208;
                v208 = v189[v207];
                bool v209;
                v209 = v190[v207];
                float v216; bool v217;
                if (v201){
                    if (v209){
                        bool v210;
                        v210 = v200 >= v208;
                        float v211;
                        if (v210){
                            v211 = v200;
                        } else {
                            v211 = v208;
                        }
                        v216 = v211; v217 = true;
                    } else {
                        v216 = v200; v217 = v201;
                    }
                } else {
                    if (v209){
                        v216 = v208; v217 = v209;
                    } else {
                        v216 = v200; v217 = v201;
                    }
                }
                v200 = v216;
                v201 = v217;
                v204 += 1l ;
            }
            v202 += 1l ;
        }
        auto v218 = cooperative_groups::coalesced_threads();
        int v219;
        v219 = threadIdx.x;
        auto v220 = cooperative_groups::labeled_partition(v218,v219);
        Closure4 v221{};
        float v222; bool v223;
        Tuple16 tmp33 = cooperative_groups::reduce(v220, Tuple16{v200, v201}, v221);
        v222 = tmp33.v0; v223 = tmp33.v1;
        bool v224;
        v224 = v223 == false;
        if (v224){
            assert("The local reduce must be true." && v223);
        } else {
        }
        float v226[4l];
        int v227[4l];
        int v228;
        v228 = 0l;
        while (while_method_6(v228)){
            int v230;
            v230 = 0l;
            while (while_method_3(v230)){
                assert("Tensor range check" && 0 <= v228 && v228 < 1l);
                assert("Tensor range check" && 0 <= v230 && v230 < 4l);
                int v232;
                v232 = 4l * v228;
                int v233;
                v233 = v232 + v230;
                int v234;
                v234 = v47[v233];
                float v235;
                v235 = curand_uniform(&v84);
                assert("Tensor range check" && 0 <= v228 && v228 < 1l);
                assert("Tensor range check" && 0 <= v230 && v230 < 4l);
                v226[v233] = v235;
                v227[v233] = v234;
                v230 += 1l ;
            }
            v228 += 1l ;
        }
        float v236; int v237;
        Tuple17 tmp34 = Tuple17{0.0f, 2147483647l};
        v236 = tmp34.v0; v237 = tmp34.v1;
        int v238;
        v238 = 0l;
        while (while_method_6(v238)){
            int v240;
            v240 = 0l;
            while (while_method_3(v240)){
                assert("Tensor range check" && 0 <= v238 && v238 < 1l);
                assert("Tensor range check" && 0 <= v240 && v240 < 4l);
                int v242;
                v242 = 4l * v238;
                int v243;
                v243 = v242 + v240;
                float v244;
                v244 = v226[v243];
                int v245;
                v245 = v227[v243];
                bool v246;
                v246 = v237 < v245;
                float v247; int v248;
                if (v246){
                    v247 = v236; v248 = v237;
                } else {
                    v247 = v244; v248 = v245;
                }
                v236 = v247;
                v237 = v248;
                v240 += 1l ;
            }
            v238 += 1l ;
        }
        auto v249 = cooperative_groups::coalesced_threads();
        int v250;
        v250 = threadIdx.x;
        auto v251 = cooperative_groups::labeled_partition(v249,v250);
        Closure5 v252{};
        float v253; int v254;
        Tuple17 tmp35 = cooperative_groups::reduce(v251, Tuple17{v236, v237}, v252);
        v253 = tmp35.v0; v254 = tmp35.v1;
        float v255;
        v255 = v222 * v253;
        int v256[4l];
        bool v257[4l];
        int v258;
        v258 = 0l;
        while (while_method_6(v258)){
            int v260;
            v260 = 0l;
            while (while_method_3(v260)){
                assert("Tensor range check" && 0 <= v258 && v258 < 1l);
                assert("Tensor range check" && 0 <= v260 && v260 < 4l);
                int v262;
                v262 = 4l * v258;
                int v263;
                v263 = v262 + v260;
                float v264;
                v264 = v189[v263];
                bool v265;
                v265 = v190[v263];
                int v266;
                v266 = v47[v263];
                int v269; bool v270;
                if (v265){
                    float v267;
                    v267 = v264 - v255;
                    bool v268;
                    v268 = v267 >= 0.0f;
                    v269 = v266; v270 = v268;
                } else {
                    v269 = 2147483647l; v270 = false;
                }
                assert("Tensor range check" && 0 <= v258 && v258 < 1l);
                assert("Tensor range check" && 0 <= v260 && v260 < 4l);
                v256[v263] = v269;
                v257[v263] = v270;
                v260 += 1l ;
            }
            v258 += 1l ;
        }
        int v271; bool v272;
        Tuple18 tmp36 = Tuple18{2147483647l, false};
        v271 = tmp36.v0; v272 = tmp36.v1;
        int v273;
        v273 = 0l;
        while (while_method_6(v273)){
            int v275;
            v275 = 0l;
            while (while_method_3(v275)){
                assert("Tensor range check" && 0 <= v273 && v273 < 1l);
                assert("Tensor range check" && 0 <= v275 && v275 < 4l);
                int v277;
                v277 = 4l * v273;
                int v278;
                v278 = v277 + v275;
                int v279;
                v279 = v256[v278];
                bool v280;
                v280 = v257[v278];
                int v287; bool v288;
                if (v272){
                    if (v280){
                        bool v281;
                        v281 = v271 < v279;
                        int v282;
                        if (v281){
                            v282 = v271;
                        } else {
                            v282 = v279;
                        }
                        v287 = v282; v288 = true;
                    } else {
                        v287 = v271; v288 = v272;
                    }
                } else {
                    if (v280){
                        v287 = v279; v288 = v280;
                    } else {
                        v287 = v271; v288 = v272;
                    }
                }
                v271 = v287;
                v272 = v288;
                v275 += 1l ;
            }
            v273 += 1l ;
        }
        auto v289 = cooperative_groups::coalesced_threads();
        int v290;
        v290 = threadIdx.x;
        auto v291 = cooperative_groups::labeled_partition(v289,v290);
        Closure6 v292{};
        int v293; bool v294;
        Tuple18 tmp37 = cooperative_groups::reduce(v291, Tuple18{v271, v272}, v292);
        v293 = tmp37.v0; v294 = tmp37.v1;
        bool v295;
        v295 = v294 == false;
        if (v295){
            assert("The local reduce must be true." && v294);
        } else {
        }
        bool v297[4l];
        int v298;
        v298 = 0l;
        while (while_method_6(v298)){
            int v300;
            v300 = 0l;
            while (while_method_3(v300)){
                assert("Tensor range check" && 0 <= v298 && v298 < 1l);
                assert("Tensor range check" && 0 <= v300 && v300 < 4l);
                int v302;
                v302 = 4l * v298;
                int v303;
                v303 = v302 + v300;
                float v304;
                v304 = v46[v303];
                int v305;
                v305 = v47[v303];
                bool v306;
                v306 = v305 < 11l;
                assert("Tensor range check" && 0 <= v298 && v298 < 1l);
                assert("Tensor range check" && 0 <= v300 && v300 < 4l);
                v297[v303] = v306;
                v300 += 1l ;
            }
            v298 += 1l ;
        }
        float v307[4l];
        int v308;
        v308 = 0l;
        while (while_method_6(v308)){
            int v310;
            v310 = 0l;
            while (while_method_3(v310)){
                assert("Tensor range check" && 0 <= v308 && v308 < 1l);
                assert("Tensor range check" && 0 <= v310 && v310 < 4l);
                int v312;
                v312 = 4l * v308;
                int v313;
                v313 = v312 + v310;
                float v314;
                v314 = v46[v313];
                bool v315;
                v315 = v297[v313];
                float v318;
                if (v315){
                    bool v316;
                    v316 = 0.0f >= v314;
                    if (v316){
                        v318 = 0.0f;
                    } else {
                        v318 = v314;
                    }
                } else {
                    v318 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v308 && v308 < 1l);
                assert("Tensor range check" && 0 <= v310 && v310 < 4l);
                v307[v313] = v318;
                v310 += 1l ;
            }
            v308 += 1l ;
        }
        float v319;
        v319 = 0.0f;
        int v320;
        v320 = 0l;
        while (while_method_6(v320)){
            int v322;
            v322 = 0l;
            while (while_method_3(v322)){
                assert("Tensor range check" && 0 <= v320 && v320 < 1l);
                assert("Tensor range check" && 0 <= v322 && v322 < 4l);
                int v324;
                v324 = 4l * v320;
                int v325;
                v325 = v324 + v322;
                float v326;
                v326 = v307[v325];
                float v327;
                v327 = v319 + v326;
                v319 = v327;
                v322 += 1l ;
            }
            v320 += 1l ;
        }
        auto v328 = cooperative_groups::coalesced_threads();
        int v329;
        v329 = threadIdx.x;
        auto v330 = cooperative_groups::labeled_partition(v328,v329);
        float v331;
        v331 = cooperative_groups::reduce(v330, v319, v119);
        int v332[4l];
        int v333;
        v333 = 0l;
        while (while_method_6(v333)){
            int v335;
            v335 = 0l;
            while (while_method_3(v335)){
                assert("Tensor range check" && 0 <= v333 && v333 < 1l);
                assert("Tensor range check" && 0 <= v335 && v335 < 4l);
                int v337;
                v337 = 4l * v333;
                int v338;
                v338 = v337 + v335;
                bool v339;
                v339 = v297[v338];
                int v340;
                if (v339){
                    v340 = 1l;
                } else {
                    v340 = 0l;
                }
                assert("Tensor range check" && 0 <= v333 && v333 < 1l);
                assert("Tensor range check" && 0 <= v335 && v335 < 4l);
                v332[v338] = v340;
                v335 += 1l ;
            }
            v333 += 1l ;
        }
        int v341;
        v341 = 0l;
        int v342;
        v342 = 0l;
        while (while_method_6(v342)){
            int v344;
            v344 = 0l;
            while (while_method_3(v344)){
                assert("Tensor range check" && 0 <= v342 && v342 < 1l);
                assert("Tensor range check" && 0 <= v344 && v344 < 4l);
                int v346;
                v346 = 4l * v342;
                int v347;
                v347 = v346 + v344;
                int v348;
                v348 = v332[v347];
                int v349;
                v349 = v341 + v348;
                v341 = v349;
                v344 += 1l ;
            }
            v342 += 1l ;
        }
        auto v350 = cooperative_groups::coalesced_threads();
        int v351;
        v351 = threadIdx.x;
        auto v352 = cooperative_groups::labeled_partition(v350,v351);
        int v353;
        v353 = cooperative_groups::reduce(v352, v341, v142);
        float v354;
        v354 = (float)v353;
        float v355;
        v355 = 1.0f / v354;
        float v356[4l];
        int v357;
        v357 = 0l;
        while (while_method_6(v357)){
            int v359;
            v359 = 0l;
            while (while_method_3(v359)){
                assert("Tensor range check" && 0 <= v357 && v357 < 1l);
                assert("Tensor range check" && 0 <= v359 && v359 < 4l);
                int v361;
                v361 = 4l * v357;
                int v362;
                v362 = v361 + v359;
                float v363;
                v363 = v307[v362];
                bool v364;
                v364 = v297[v362];
                bool v365;
                v365 = v364 == false;
                float v370;
                if (v365){
                    v370 = 0.0f;
                } else {
                    bool v366;
                    v366 = v331 == 0.0f;
                    bool v367;
                    v367 = v366 != true;
                    if (v367){
                        float v368;
                        v368 = v363 / v331;
                        v370 = v368;
                    } else {
                        v370 = v355;
                    }
                }
                assert("Tensor range check" && 0 <= v357 && v357 < 1l);
                assert("Tensor range check" && 0 <= v359 && v359 < 4l);
                v356[v362] = v370;
                v359 += 1l ;
            }
            v357 += 1l ;
        }
        float v371; int v372;
        Tuple17 tmp38 = Tuple17{0.0f, 2147483647l};
        v371 = tmp38.v0; v372 = tmp38.v1;
        int v373;
        v373 = 0l;
        while (while_method_6(v373)){
            int v375;
            v375 = 0l;
            while (while_method_3(v375)){
                assert("Tensor range check" && 0 <= v373 && v373 < 1l);
                assert("Tensor range check" && 0 <= v375 && v375 < 4l);
                int v377;
                v377 = 4l * v373;
                int v378;
                v378 = v377 + v375;
                float v379;
                v379 = v146[v378];
                int v380;
                v380 = v47[v378];
                bool v381;
                v381 = v372 == v293;
                float v385; int v386;
                if (v381){
                    v385 = v371; v386 = v372;
                } else {
                    bool v382;
                    v382 = v380 == v293;
                    if (v382){
                        v385 = v379; v386 = v380;
                    } else {
                        v385 = v371; v386 = v372;
                    }
                }
                v371 = v385;
                v372 = v386;
                v375 += 1l ;
            }
            v373 += 1l ;
        }
        auto v387 = cooperative_groups::coalesced_threads();
        int v388;
        v388 = threadIdx.x;
        auto v389 = cooperative_groups::labeled_partition(v387,v388);
        Closure7 v390{v293};
        float v391; int v392;
        Tuple17 tmp39 = cooperative_groups::reduce(v389, Tuple17{v371, v372}, v390);
        v391 = tmp39.v0; v392 = tmp39.v1;
        bool v393;
        v393 = v392 == 2147483647l;
        bool v394;
        v394 = v393 != true;
        bool v395;
        v395 = v394 == false;
        if (v395){
            assert("Expected a valid action id in get_action." && v394);
        } else {
        }
        float v397; int v398;
        Tuple17 tmp40 = Tuple17{0.0f, 2147483647l};
        v397 = tmp40.v0; v398 = tmp40.v1;
        int v399;
        v399 = 0l;
        while (while_method_6(v399)){
            int v401;
            v401 = 0l;
            while (while_method_3(v401)){
                assert("Tensor range check" && 0 <= v399 && v399 < 1l);
                assert("Tensor range check" && 0 <= v401 && v401 < 4l);
                int v403;
                v403 = 4l * v399;
                int v404;
                v404 = v403 + v401;
                float v405;
                v405 = v356[v404];
                int v406;
                v406 = v47[v404];
                bool v407;
                v407 = v398 == v293;
                float v411; int v412;
                if (v407){
                    v411 = v397; v412 = v398;
                } else {
                    bool v408;
                    v408 = v406 == v293;
                    if (v408){
                        v411 = v405; v412 = v406;
                    } else {
                        v411 = v397; v412 = v398;
                    }
                }
                v397 = v411;
                v398 = v412;
                v401 += 1l ;
            }
            v399 += 1l ;
        }
        auto v413 = cooperative_groups::coalesced_threads();
        int v414;
        v414 = threadIdx.x;
        auto v415 = cooperative_groups::labeled_partition(v413,v414);
        float v416; int v417;
        Tuple17 tmp41 = cooperative_groups::reduce(v415, Tuple17{v397, v398}, v390);
        v416 = tmp41.v0; v417 = tmp41.v1;
        bool v418;
        v418 = v417 == 2147483647l;
        bool v419;
        v419 = v418 != true;
        bool v420;
        v420 = v419 == false;
        if (v420){
            assert("Expected a valid action id in get_action." && v419);
        } else {
        }
        int v422;
        v422 = 0l;
        while (while_method_6(v422)){
            assert("Tensor range check" && 0 <= v422 && v422 < 1l);
            assert("Tensor range check" && 0 <= v422 && v422 < 1l);
            v422 += 1l ;
        }
        assert("Tensor range check" && 0 <= v39 && v39 < 32l);
        v17[v39] = v416;
        v18[v39] = v391;
        v19[v39] = v293;
        v28 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float v431; float v432; int v433;
    if (v21){
        assert("Tensor range check" && 0 <= v20 && v20 < 32l);
        float v424;
        v424 = v17[v20];
        float v425;
        v425 = v18[v20];
        int v426;
        v426 = v19[v20];
        v431 = v424; v432 = v425; v433 = v426;
    } else {
        Tuple14 v427[1l];
        float v428; float v429; int v430;
        Tuple14 tmp42 = v427[0l];
        v428 = tmp42.v0; v429 = tmp42.v1; v430 = tmp42.v2;
        v431 = v428; v432 = v429; v433 = v430;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return Tuple14{v431, v432, v433};
}
__device__ __noinline__ Union9 noinline_run_42(unsigned char * v0, unsigned char * v1, Union8 v2){
    asm("barrier.cta.sync %0;" :: "r"(0l));
    unsigned int * v3;
    v3 = reinterpret_cast<unsigned int *>(&v0[278528ull]);
    float * v5;
    v5 = reinterpret_cast<float *>(&v0[0ull]);
    unsigned long long v7;
    v7 = clock64();
    int v8;
    v8 = threadIdx.x;
    int v9;
    v9 = blockIdx.x;
    int v10;
    v10 = v9 * 32l;
    int v11;
    v11 = v8 + v10;
    unsigned long long v12;
    v12 = (unsigned long long)v11;
    curandStatePhilox4_32_10_t v13;
    curand_init(v7,v12,0ull,&v13);
    float * v14;
    v14 = reinterpret_cast<float *>(&v0[0ull]);
    int v16;
    v16 = blockIdx.x;
    assert("Tensor range check" && 0 <= v16 && v16 < 1l);
    int v17;
    v17 = 65536l * v16;
    unsigned long long v18;
    v18 = clock64();
    int v19;
    v19 = threadIdx.x;
    int v20;
    v20 = blockIdx.x;
    int v21;
    v21 = v20 * 32l;
    int v22;
    v22 = v19 + v21;
    unsigned long long v23;
    v23 = (unsigned long long)v22;
    curandStatePhilox4_32_10_t v24;
    curand_init(v18,v23,0ull,&v24);
    int v25;
    v25 = threadIdx.x;
    int v26;
    v26 = v25;
    while (while_method_7(v26)){
        bool v28;
        v28 = 0l <= v26;
        bool v29;
        v29 = v28 == false;
        if (v29){
            assert("The index needs to be zero or positive." && v28);
        } else {
        }
        int v31;
        v31 = v26 % 2048l;
        int v32;
        v32 = v26 / 2048l;
        bool v33;
        v33 = v32 < 32l;
        bool v34;
        v34 = v33 == false;
        if (v34){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v33);
        } else {
        }
        assert("Tensor range check" && 0 <= v32 && v32 < 32l);
        assert("Tensor range check" && 0 <= v31 && v31 < 2048l);
        int v36;
        v36 = v31 + v17;
        int v37;
        v37 = 2048l * v32;
        int v38;
        v38 = v37 + v36;
        v14[v38] = 0.0f;
        v26 += 32l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    switch (v2.tag) {
        case 0: { // None
            break;
        }
        case 1: { // Some
            int v39 = v2.case1.v0; static_array<static_array<unsigned char,2l>,2l> v40 = v2.case1.v1; static_array<int,2l> v41 = v2.case1.v2; int v42 = v2.case1.v3; static_array<int,2l> v43 = v2.case1.v4; Union5 v44 = v2.case1.v5; dynamic_array_list<Union6,128l> v45 = v2.case1.v6;
            int v46;
            v46 = threadIdx.x;
            assert("Tensor range check" && 0 <= v46 && v46 < 32l);
            int v47;
            v47 = 2048l * v46;
            int v48;
            v48 = v47 + v17;
            int v49;
            v49 = v45.length_();
            bool v50;
            v50 = 128l >= v49;
            bool v51;
            v51 = v50 == false;
            if (v51){
                assert("The type level dimension has to equal the value passed at runtime into create." && v50);
            } else {
            }
            dynamic_array_list<Union10,128l> v53{0};
            v53.unsafe_set_length(v49);
            int v55;
            v55 = 0l;
            while (while_method_4(v49, v55)){
                Union6 v57;
                v57 = v45[v55];
                Union10 v63;
                switch (v57.tag) {
                    case 2: { // PlayerAction
                        int v59 = v57.case2.v0; Union1 v60 = v57.case2.v1;
                        v63 = Union10{Union10_1{v60}};
                        break;
                    }
                    default: {
                        v63 = Union10{Union10_0{}};
                    }
                }
                v53[v55] = v63;
                v55 += 1l ;
            }
            static_array<int,2l> v64;
            int v66;
            v66 = 0l;
            while (while_method_0(v66)){
                int v68;
                v68 = v42 % 2l;
                int v69;
                v69 = v66 + v68;
                int v70;
                v70 = v41[v69];
                v64[v66] = v70;
                v66 += 1l ;
            }
            static_array<int,2l> v72;
            int v74;
            v74 = 0l;
            while (while_method_0(v74)){
                int v76;
                v76 = v42 % 2l;
                int v77;
                v77 = v74 + v76;
                int v78;
                v78 = v43[v77];
                v72[v74] = v78;
                v74 += 1l ;
            }
            int v80;
            v80 = v42 % 2l;
            static_array<unsigned char,2l> v81;
            v81 = v40[v80];
            static_array_list<unsigned char,5l> v83;
            v83 = static_array_list<unsigned char,5l>{};
            switch (v44.tag) {
                case 0: { // Flop
                    static_array<unsigned char,3l> v85 = v44.case0.v0;
                    int v86;
                    v86 = 0l;
                    while (while_method_1(v86)){
                        unsigned char v88;
                        v88 = v85[v86];
                        v83.push(v88);
                        v86 += 1l ;
                    }
                    break;
                }
                case 1: { // Preflop
                    break;
                }
                case 2: { // River
                    static_array<unsigned char,5l> v95 = v44.case2.v0;
                    int v96;
                    v96 = 0l;
                    while (while_method_2(v96)){
                        unsigned char v98;
                        v98 = v95[v96];
                        v83.push(v98);
                        v96 += 1l ;
                    }
                    break;
                }
                case 3: { // Turn
                    static_array<unsigned char,4l> v90 = v44.case3.v0;
                    int v91;
                    v91 = 0l;
                    while (while_method_3(v91)){
                        unsigned char v93;
                        v93 = v90[v91];
                        v83.push(v93);
                        v91 += 1l ;
                    }
                    break;
                }
                default: {
                    assert("Invalid tag." && false); __trap();
                }
            }
            float * v100;
            v100 = v14+v48;
            int v102;
            v102 = v53.length_();
            bool v103;
            v103 = v102 == 0l;
            if (v103){
                v100[0l] = 1.0f;
            } else {
            }
            int v104;
            v104 = v53.length_();
            int v105;
            v105 = 0l;
            while (while_method_4(v104, v105)){
                Union10 v107;
                v107 = v53[v105];
                int v109;
                v109 = v105 * 14l;
                int v110;
                v110 = 1l + v109;
                switch (v107.tag) {
                    case 0: { // None
                        v100[v110] = 1.0f;
                        break;
                    }
                    case 1: { // Some
                        Union1 v111 = v107.case1.v0;
                        int v112;
                        v112 = v110 + 1l;
                        switch (v111.tag) {
                            case 0: { // A_All_In
                                v100[v112] = 1.0f;
                                break;
                            }
                            case 1: { // A_Call
                                int v113;
                                v113 = v112 + 1l;
                                v100[v113] = 1.0f;
                                break;
                            }
                            case 2: { // A_Fold
                                int v114;
                                v114 = v112 + 2l;
                                v100[v114] = 1.0f;
                                break;
                            }
                            case 3: { // A_Raise
                                int v115 = v111.case3.v0;
                                int v116;
                                v116 = v112 + 3l;
                                bool v117;
                                v117 = 1l <= v115;
                                bool v119;
                                if (v117){
                                    bool v118;
                                    v118 = v115 < 1023l;
                                    v119 = v118;
                                } else {
                                    v119 = false;
                                }
                                bool v120;
                                v120 = v119 == false;
                                if (v120){
                                    assert("Pickle failure. The input is out of the bounds of the given range." && v119);
                                } else {
                                }
                                int v122;
                                v122 = v115 - 1l;
                                unsigned int v123;
                                v123 = (unsigned int)v122;
                                method_43(v123, v100, v116);
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
                v105 += 1l ;
            }
            int v124;
            v124 = 0l;
            while (while_method_0(v124)){
                int v126;
                v126 = v64[v124];
                int v128;
                v128 = v124 * 11l;
                int v129;
                v129 = 1794l + v128;
                bool v130;
                v130 = 0l <= v126;
                bool v132;
                if (v130){
                    bool v131;
                    v131 = v126 < 1023l;
                    v132 = v131;
                } else {
                    v132 = false;
                }
                bool v133;
                v133 = v132 == false;
                if (v133){
                    assert("Pickle failure. The input is out of the bounds of the given range." && v132);
                } else {
                }
                unsigned int v135;
                v135 = (unsigned int)v126;
                method_44(v135, v100, v129);
                v124 += 1l ;
            }
            int v136;
            v136 = 0l;
            while (while_method_0(v136)){
                int v138;
                v138 = v72[v136];
                int v140;
                v140 = v136 * 11l;
                int v141;
                v141 = 1817l + v140;
                bool v142;
                v142 = 0l <= v138;
                bool v144;
                if (v142){
                    bool v143;
                    v143 = v138 < 1023l;
                    v144 = v143;
                } else {
                    v144 = false;
                }
                bool v145;
                v145 = v144 == false;
                if (v145){
                    assert("Pickle failure. The input is out of the bounds of the given range." && v144);
                } else {
                }
                unsigned int v147;
                v147 = (unsigned int)v138;
                method_44(v147, v100, v141);
                v136 += 1l ;
            }
            int v148;
            v148 = 0l;
            while (while_method_0(v148)){
                unsigned char v150;
                v150 = v81[v148];
                int v152;
                v152 = v148 * 17l;
                int v153;
                v153 = 1840l + v152;
                unsigned char v154;
                v154 = v150 % 4u;
                int v155;
                v155 = (int)v154;
                unsigned char v156;
                v156 = v150 / 4u;
                int v157;
                v157 = (int)v156;
                unsigned int v158;
                v158 = (unsigned int)v155;
                int v159;
                v159 = (int)v158;
                bool v160;
                v160 = v159 < 4l;
                bool v161;
                v161 = v160 == false;
                if (v161){
                    assert("Pickle failure. Int value out of bounds." && v160);
                } else {
                }
                int v163;
                v163 = v153 + v159;
                v100[v163] = 1.0f;
                int v164;
                v164 = v153 + 4l;
                unsigned int v165;
                v165 = (unsigned int)v157;
                int v166;
                v166 = (int)v165;
                bool v167;
                v167 = v166 < 13l;
                bool v168;
                v168 = v167 == false;
                if (v168){
                    assert("Pickle failure. Int value out of bounds." && v167);
                } else {
                }
                int v170;
                v170 = v164 + v166;
                v100[v170] = 1.0f;
                v148 += 1l ;
            }
            int v171;
            v171 = v83.length;
            bool v172;
            v172 = v171 == 0l;
            if (v172){
                v100[1874l] = 1.0f;
            } else {
            }
            int v173;
            v173 = v83.length;
            int v174;
            v174 = 0l;
            while (while_method_4(v173, v174)){
                unsigned char v176;
                v176 = v83[v174];
                int v178;
                v178 = v174 * 17l;
                int v179;
                v179 = 1875l + v178;
                unsigned char v180;
                v180 = v176 % 4u;
                int v181;
                v181 = (int)v180;
                unsigned char v182;
                v182 = v176 / 4u;
                int v183;
                v183 = (int)v182;
                unsigned int v184;
                v184 = (unsigned int)v181;
                int v185;
                v185 = (int)v184;
                bool v186;
                v186 = v185 < 4l;
                bool v187;
                v187 = v186 == false;
                if (v187){
                    assert("Pickle failure. Int value out of bounds." && v186);
                } else {
                }
                int v189;
                v189 = v179 + v185;
                v100[v189] = 1.0f;
                int v190;
                v190 = v179 + 4l;
                unsigned int v191;
                v191 = (unsigned int)v183;
                int v192;
                v192 = (int)v191;
                bool v193;
                v193 = v192 < 13l;
                bool v194;
                v194 = v193 == false;
                if (v194){
                    assert("Pickle failure. Int value out of bounds." && v193);
                } else {
                }
                int v196;
                v196 = v190 + v192;
                v100[v196] = 1.0f;
                v174 += 1l ;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v197;
    v197 = 0l;
    int v198;
    v198 = 4l;
    int v199;
    v199 = int_range_45(v198, v197, v24);
    __shared__ int v200[1l];
    int v201;
    v201 = threadIdx.x;
    bool v202;
    v202 = v201 == 0l;
    if (v202){
        v200[0l] = v199;
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v203;
    v203 = v200[0l];
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v204;
    v204 = reinterpret_cast<float *>(&v0[0ull]);
    float * v206;
    v206 = reinterpret_cast<float *>(&v1[0ull]);
    assert("Tensor range check" && 0 <= v203 && v203 < 4l);
    int v208;
    v208 = 262144l * v203;
    float * v209;
    v209 = reinterpret_cast<float *>(&v0[262144ull]);
    int v211;
    v211 = blockIdx.x;
    assert("Tensor range check" && 0 <= v211 && v211 < 1l);
    int v212;
    v212 = 65536l * v211;
    int v213;
    v213 = blockIdx.x;
    assert("Tensor range check" && 0 <= v213 && v213 < 1l);
    int v214;
    v214 = 4096l * v213;
    method_46(v206, v208, v209, v214, v204, v212);
    unsigned int * v215;
    v215 = reinterpret_cast<unsigned int *>(&v0[278528ull]);
    assert("Tensor range check" && 0 <= v203 && v203 < 4l);
    int v217;
    v217 = 32l * v203;
    method_47(v215, v217, v209);
    float * v218;
    v218 = reinterpret_cast<float *>(&v1[4194304ull]);
    float * v220;
    v220 = reinterpret_cast<float *>(&v1[4456448ull]);
    float * v222;
    v222 = reinterpret_cast<float *>(&v1[4718592ull]);
    float * v224;
    v224 = reinterpret_cast<float *>(&v1[4980736ull]);
    float * v226;
    v226 = reinterpret_cast<float *>(&v1[5242880ull]);
    int * v228;
    v228 = reinterpret_cast<int *>(&v1[5505024ull]);
    float * v230;
    v230 = reinterpret_cast<float *>(&v1[5513216ull]);
    int * v232;
    v232 = reinterpret_cast<int *>(&v1[5521408ull]);
    int * v234;
    v234 = reinterpret_cast<int *>(&v1[5529600ull]);
    double * v236;
    v236 = reinterpret_cast<double *>(&v1[5537792ull]);
    double * v238;
    v238 = reinterpret_cast<double *>(&v1[5570560ull]);
    float * v240;
    v240 = reinterpret_cast<float *>(&v1[5603328ull]);
    float * v242;
    v242 = reinterpret_cast<float *>(&v1[5636096ull]);
    float * v244;
    v244 = reinterpret_cast<float *>(&v1[5644288ull]);
    asm("barrier.cta.sync %0;" :: "r"(0l));
    unsigned int * v246;
    v246 = reinterpret_cast<unsigned int *>(&v0[278528ull]);
    int v248;
    v248 = blockIdx.x;
    int v249;
    v249 = threadIdx.x;
    assert("Tensor range check" && 0 <= v203 && v203 < 4l);
    assert("Tensor range check" && 0 <= v248 && v248 < 1l);
    assert("Tensor range check" && 0 <= v249 && v249 < 32l);
    int v250;
    v250 = 32l * v248;
    int v251;
    v251 = v250 + v249;
    int v252;
    v252 = v217 + v251;
    unsigned int v253;
    v253 = v246[v252];
    float * v254;
    v254 = reinterpret_cast<float *>(&v1[4194304ull]);
    float * v256;
    v256 = reinterpret_cast<float *>(&v1[4456448ull]);
    float * v258;
    v258 = reinterpret_cast<float *>(&v1[4718592ull]);
    float * v260;
    v260 = reinterpret_cast<float *>(&v1[4980736ull]);
    float * v262;
    v262 = reinterpret_cast<float *>(&v1[5242880ull]);
    int * v264;
    v264 = reinterpret_cast<int *>(&v1[5505024ull]);
    float * v266;
    v266 = reinterpret_cast<float *>(&v1[5513216ull]);
    int * v268;
    v268 = reinterpret_cast<int *>(&v1[5521408ull]);
    int * v270;
    v270 = reinterpret_cast<int *>(&v1[5529600ull]);
    double * v272;
    v272 = reinterpret_cast<double *>(&v1[5537792ull]);
    double * v274;
    v274 = reinterpret_cast<double *>(&v1[5570560ull]);
    float * v276;
    v276 = reinterpret_cast<float *>(&v1[5603328ull]);
    float * v278;
    v278 = reinterpret_cast<float *>(&v1[5636096ull]);
    float * v280;
    v280 = reinterpret_cast<float *>(&v1[5644288ull]);
    int v282;
    v282 = (int)v253;
    float v283; float v284; int v285;
    Tuple14 tmp43 = method_48(v254, v256, v258, v260, v262, v282, v203);
    v283 = tmp43.v0; v284 = tmp43.v1; v285 = tmp43.v2;
    bool v286;
    v286 = 0l == v285;
    if (v286){
        return Union9{Union9_1{}};
    } else {
        bool v288;
        v288 = 1l == v285;
        if (v288){
            return Union9{Union9_0{}};
        } else {
            bool v290;
            v290 = 2l == v285;
            if (v290){
                return Union9{Union9_2{1l, 3l}};
            } else {
                bool v292;
                v292 = 3l == v285;
                if (v292){
                    return Union9{Union9_2{1l, 2l}};
                } else {
                    bool v294;
                    v294 = 4l == v285;
                    if (v294){
                        return Union9{Union9_2{2l, 3l}};
                    } else {
                        bool v296;
                        v296 = 5l == v285;
                        if (v296){
                            return Union9{Union9_2{3l, 4l}};
                        } else {
                            bool v298;
                            v298 = 6l == v285;
                            if (v298){
                                return Union9{Union9_2{1l, 1l}};
                            } else {
                                bool v300;
                                v300 = 7l == v285;
                                if (v300){
                                    return Union9{Union9_2{3l, 2l}};
                                } else {
                                    bool v302;
                                    v302 = 8l == v285;
                                    if (v302){
                                        return Union9{Union9_2{2l, 1l}};
                                    } else {
                                        bool v304;
                                        v304 = 9l == v285;
                                        if (v304){
                                            return Union9{Union9_2{3l, 1l}};
                                        } else {
                                            bool v306;
                                            v306 = 10l == v285;
                                            if (v306){
                                                return Union9{Union9_2{2147483647l, 1l}};
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
}
__device__ void method_49(Union1 v0){
    switch (v0.tag) {
        case 0: { // A_All_In
            printf("%s","A_All_In");
            return ;
            break;
        }
        case 1: { // A_Call
            printf("%s","A_Call");
            return ;
            break;
        }
        case 2: { // A_Fold
            printf("%s","A_Fold");
            return ;
            break;
        }
        case 3: { // A_Raise
            int v1 = v0.case3.v0;
            printf("%s(%d)","A_Raise", v1);
            return ;
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ inline bool while_method_13(int v0){
    bool v1;
    v1 = v0 < 6l;
    return v1;
}
__device__ inline bool while_method_14(static_array<float,6l> v0, int v1){
    bool v2;
    v2 = v1 < 6l;
    return v2;
}
__device__ inline bool while_method_15(int v0, int v1){
    bool v2;
    v2 = v1 > v0;
    return v2;
}
__device__ int loop_53(static_array<float,6l> v0, float v1, int v2){
    bool v3;
    v3 = v2 < 6l;
    if (v3){
        float v4;
        v4 = v0[v2];
        bool v6;
        v6 = v1 <= v4;
        if (v6){
            return v2;
        } else {
            int v7;
            v7 = v2 + 1l;
            return loop_53(v0, v1, v7);
        }
    } else {
        return 5l;
    }
}
__device__ int pick_discrete__52(static_array<float,6l> v0, float v1){
    static_array<float,6l> v2;
    int v4;
    v4 = 0l;
    while (while_method_13(v4)){
        float v6;
        v6 = v0[v4];
        v2[v4] = v6;
        v4 += 1l ;
    }
    int v8;
    v8 = 1l;
    while (while_method_14(v2, v8)){
        int v10;
        v10 = 6l;
        while (while_method_15(v8, v10)){
            v10 -= 1l ;
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
        v18 = v8 * 2l;
        v8 = v18;
    }
    float v19;
    v19 = v2[5l];
    float v21;
    v21 = v1 * v19;
    int v22;
    v22 = 0l;
    return loop_53(v2, v21, v22);
}
__device__ int sample_discrete__51(static_array<float,6l> v0, curandStatePhilox4_32_10_t & v1){
    float v2;
    v2 = curand_uniform(&v1);
    return pick_discrete__52(v0, v2);
}
__device__ Union1 sample_discrete_50(static_array<Tuple19,6l> v0, curandStatePhilox4_32_10_t & v1){
    static_array<float,6l> v2;
    int v4;
    v4 = 0l;
    while (while_method_13(v4)){
        Union1 v6; float v7;
        Tuple19 tmp52 = v0[v4];
        v6 = tmp52.v0; v7 = tmp52.v1;
        v2[v4] = v7;
        v4 += 1l ;
    }
    int v10;
    v10 = sample_discrete__51(v2, v1);
    Union1 v11; float v12;
    Tuple19 tmp53 = v0[v10];
    v11 = tmp53.v0; v12 = tmp53.v1;
    return v11;
}
__device__ inline bool while_method_16(int v0){
    bool v1;
    v1 = v0 < 7l;
    return v1;
}
__device__ inline bool while_method_17(static_array<unsigned char,7l> v0, bool v1, int v2){
    bool v3;
    v3 = v2 < 7l;
    return v3;
}
__device__ inline bool while_method_18(static_array<unsigned char,7l> v0, int v1){
    bool v2;
    v2 = v1 < 7l;
    return v2;
}
__device__ inline bool while_method_19(int v0, int v1, int v2, int v3){
    bool v4;
    v4 = v3 < v0;
    return v4;
}
__device__ Tuple0 score_54(static_array<unsigned char,7l> v0){
    static_array<unsigned char,7l> v1;
    int v3;
    v3 = 0l;
    while (while_method_16(v3)){
        unsigned char v5;
        v5 = v0[v3];
        v1[v3] = v5;
        v3 += 1l ;
    }
    static_array<unsigned char,7l> v7;
    bool v9; int v10;
    Tuple20 tmp60 = Tuple20{true, 1l};
    v9 = tmp60.v0; v10 = tmp60.v1;
    while (while_method_17(v1, v9, v10)){
        int v12;
        v12 = 0l;
        while (while_method_18(v1, v12)){
            int v14;
            v14 = v12 + v10;
            bool v15;
            v15 = v14 < 7l;
            int v16;
            if (v15){
                v16 = v14;
            } else {
                v16 = 7l;
            }
            int v17;
            v17 = v10 * 2l;
            int v18;
            v18 = v12 + v17;
            bool v19;
            v19 = v18 < 7l;
            int v20;
            if (v19){
                v20 = v18;
            } else {
                v20 = 7l;
            }
            int v21; int v22; int v23;
            Tuple21 tmp61 = Tuple21{v12, v16, v12};
            v21 = tmp61.v0; v22 = tmp61.v1; v23 = tmp61.v2;
            while (while_method_19(v20, v21, v22, v23)){
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
                            v57 = v22 + 1l;
                            v77 = v37; v78 = v21; v79 = v57;
                            break;
                        }
                        default: {
                            int v58;
                            v58 = v21 + 1l;
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
                        v67 = v21 + 1l;
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
                        v73 = v22 + 1l;
                        v77 = v72; v78 = v21; v79 = v73;
                    }
                }
                if (v9){
                    v7[v23] = v77;
                } else {
                    v1[v23] = v77;
                }
                int v80;
                v80 = v23 + 1l;
                v21 = v78;
                v22 = v79;
                v23 = v80;
            }
            v12 = v18;
        }
        bool v81;
        v81 = v9 == false;
        int v82;
        v82 = v10 * 2l;
        v9 = v81;
        v10 = v82;
    }
    bool v83;
    v83 = v9 == false;
    static_array<unsigned char,7l> v84;
    if (v83){
        v84 = v7;
    } else {
        v84 = v1;
    }
    static_array<unsigned char,5l> v85;
    int v87; int v88; unsigned char v89;
    Tuple22 tmp62 = Tuple22{0l, 0l, 12u};
    v87 = tmp62.v0; v88 = tmp62.v1; v89 = tmp62.v2;
    while (while_method_16(v87)){
        unsigned char v91;
        v91 = v84[v87];
        bool v93;
        v93 = v88 < 5l;
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
                    v98 = 0l;
                }
                v85[v98] = v91;
                int v99;
                v99 = v98 + 1l;
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
        v87 += 1l ;
    }
    bool v107;
    v107 = v88 == 4l;
    bool v146;
    if (v107){
        unsigned char v108;
        v108 = v89 + 1u;
        bool v109;
        v109 = v108 == 0u;
        if (v109){
            unsigned char v110;
            v110 = v84[0l];
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
                    v85[4l] = v110;
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
                v118 = v84[1l];
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
                        v85[4l] = v118;
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
                    v126 = v84[2l];
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
                            v85[4l] = v126;
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
                        v134 = v84[3l];
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
                                v85[4l] = v134;
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
        v148 = v88 == 5l;
        if (v148){
            v152 = Union12{Union12_1{v85}};
        } else {
            v152 = Union12{Union12_0{}};
        }
    }
    static_array<unsigned char,5l> v153;
    int v155; int v156; unsigned char v157;
    Tuple22 tmp63 = Tuple22{0l, 0l, 12u};
    v155 = tmp63.v0; v156 = tmp63.v1; v157 = tmp63.v2;
    while (while_method_16(v155)){
        unsigned char v159;
        v159 = v84[v155];
        bool v161;
        v161 = v156 < 5l;
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
                    v166 = 0l;
                }
                v153[v166] = v159;
                int v167;
                v167 = v166 + 1l;
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
        v155 += 1l ;
    }
    bool v175;
    v175 = v156 == 4l;
    bool v214;
    if (v175){
        unsigned char v176;
        v176 = v157 + 1u;
        bool v177;
        v177 = v176 == 0u;
        if (v177){
            unsigned char v178;
            v178 = v84[0l];
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
                    v153[4l] = v178;
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
                v186 = v84[1l];
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
                        v153[4l] = v186;
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
                    v194 = v84[2l];
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
                            v153[4l] = v194;
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
                        v202 = v84[3l];
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
                                v153[4l] = v202;
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
        v216 = v156 == 5l;
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
            static_array<unsigned char,5l> v221 = v152.case1.v0;
            switch (v220.tag) {
                case 0: { // None
                    v248 = v152;
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,5l> v222 = v220.case1.v0;
                    Union11 v223;
                    v223 = Union11{Union11_0{}};
                    int v224; Union11 v225;
                    Tuple23 tmp64 = Tuple23{0l, v223};
                    v224 = tmp64.v0; v225 = tmp64.v1;
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
                        v224 += 1l ;
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
                    static_array<unsigned char,5l> v243;
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
    static_array<unsigned char,5l> v249;
    int v251; int v252; unsigned char v253;
    Tuple22 tmp65 = Tuple22{0l, 0l, 12u};
    v251 = tmp65.v0; v252 = tmp65.v1; v253 = tmp65.v2;
    while (while_method_16(v251)){
        unsigned char v255;
        v255 = v84[v251];
        bool v257;
        v257 = v252 < 5l;
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
                    v262 = 0l;
                }
                v249[v262] = v255;
                int v263;
                v263 = v262 + 1l;
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
        v251 += 1l ;
    }
    bool v271;
    v271 = v252 == 4l;
    bool v310;
    if (v271){
        unsigned char v272;
        v272 = v253 + 1u;
        bool v273;
        v273 = v272 == 0u;
        if (v273){
            unsigned char v274;
            v274 = v84[0l];
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
                    v249[4l] = v274;
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
                v282 = v84[1l];
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
                        v249[4l] = v282;
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
                    v290 = v84[2l];
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
                            v249[4l] = v290;
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
                        v298 = v84[3l];
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
                                v249[4l] = v298;
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
        v312 = v252 == 5l;
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
            static_array<unsigned char,5l> v317 = v248.case1.v0;
            switch (v316.tag) {
                case 0: { // None
                    v344 = v248;
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,5l> v318 = v316.case1.v0;
                    Union11 v319;
                    v319 = Union11{Union11_0{}};
                    int v320; Union11 v321;
                    Tuple23 tmp66 = Tuple23{0l, v319};
                    v320 = tmp66.v0; v321 = tmp66.v1;
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
                        v320 += 1l ;
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
                    static_array<unsigned char,5l> v339;
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
    static_array<unsigned char,5l> v345;
    int v347; int v348; unsigned char v349;
    Tuple22 tmp67 = Tuple22{0l, 0l, 12u};
    v347 = tmp67.v0; v348 = tmp67.v1; v349 = tmp67.v2;
    while (while_method_16(v347)){
        unsigned char v351;
        v351 = v84[v347];
        bool v353;
        v353 = v348 < 5l;
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
                    v358 = 0l;
                }
                v345[v358] = v351;
                int v359;
                v359 = v358 + 1l;
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
        v347 += 1l ;
    }
    bool v367;
    v367 = v348 == 4l;
    bool v406;
    if (v367){
        unsigned char v368;
        v368 = v349 + 1u;
        bool v369;
        v369 = v368 == 0u;
        if (v369){
            unsigned char v370;
            v370 = v84[0l];
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
                    v345[4l] = v370;
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
                v378 = v84[1l];
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
                        v345[4l] = v378;
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
                    v386 = v84[2l];
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
                            v345[4l] = v386;
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
                        v394 = v84[3l];
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
                                v345[4l] = v394;
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
        v408 = v348 == 5l;
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
            static_array<unsigned char,5l> v413 = v344.case1.v0;
            switch (v412.tag) {
                case 0: { // None
                    v440 = v344;
                    break;
                }
                case 1: { // Some
                    static_array<unsigned char,5l> v414 = v412.case1.v0;
                    Union11 v415;
                    v415 = Union11{Union11_0{}};
                    int v416; Union11 v417;
                    Tuple23 tmp68 = Tuple23{0l, v415};
                    v416 = tmp68.v0; v417 = tmp68.v1;
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
                        v416 += 1l ;
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
                    static_array<unsigned char,5l> v435;
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
    static_array<unsigned char,5l> v1037; char v1038;
    switch (v440.tag) {
        case 0: { // None
            static_array<unsigned char,4l> v442;
            static_array<unsigned char,3l> v444;
            int v446; int v447; int v448; unsigned char v449;
            Tuple24 tmp69 = Tuple24{0l, 0l, 0l, 12u};
            v446 = tmp69.v0; v447 = tmp69.v1; v448 = tmp69.v2; v449 = tmp69.v3;
            while (while_method_16(v446)){
                unsigned char v451;
                v451 = v84[v446];
                bool v453;
                v453 = v448 < 4l;
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
                        v456 = 0l;
                    }
                    v442[v456] = v451;
                    int v457;
                    v457 = v456 + 1l;
                    v461 = v446; v462 = v457; v463 = v454;
                } else {
                    break;
                }
                v447 = v461;
                v448 = v462;
                v449 = v463;
                v446 += 1l ;
            }
            bool v464;
            v464 = v448 == 4l;
            Union13 v475;
            if (v464){
                int v465;
                v465 = 0l;
                while (while_method_1(v465)){
                    int v467;
                    v467 = v447 + -3l;
                    bool v468;
                    v468 = v465 < v467;
                    int v469;
                    if (v468){
                        v469 = 0l;
                    } else {
                        v469 = 4l;
                    }
                    int v470;
                    v470 = v469 + v465;
                    unsigned char v471;
                    v471 = v84[v470];
                    v444[v465] = v471;
                    v465 += 1l ;
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
                    static_array<unsigned char,4l> v476 = v475.case1.v0; static_array<unsigned char,3l> v477 = v475.case1.v1;
                    static_array<unsigned char,1l> v478;
                    int v480;
                    v480 = 0l;
                    while (while_method_6(v480)){
                        unsigned char v482;
                        v482 = v477[v480];
                        v478[v480] = v482;
                        v480 += 1l ;
                    }
                    static_array<unsigned char,5l> v484;
                    int v486;
                    v486 = 0l;
                    while (while_method_3(v486)){
                        unsigned char v488;
                        v488 = v476[v486];
                        v484[v486] = v488;
                        v486 += 1l ;
                    }
                    int v490;
                    v490 = 0l;
                    while (while_method_6(v490)){
                        unsigned char v492;
                        v492 = v478[v490];
                        int v494;
                        v494 = 4l + v490;
                        v484[v494] = v492;
                        v490 += 1l ;
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
                    static_array<unsigned char,3l> v500;
                    static_array<unsigned char,4l> v502;
                    int v504; int v505; int v506; unsigned char v507;
                    Tuple24 tmp70 = Tuple24{0l, 0l, 0l, 12u};
                    v504 = tmp70.v0; v505 = tmp70.v1; v506 = tmp70.v2; v507 = tmp70.v3;
                    while (while_method_16(v504)){
                        unsigned char v509;
                        v509 = v84[v504];
                        bool v511;
                        v511 = v506 < 3l;
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
                                v514 = 0l;
                            }
                            v500[v514] = v509;
                            int v515;
                            v515 = v514 + 1l;
                            v519 = v504; v520 = v515; v521 = v512;
                        } else {
                            break;
                        }
                        v505 = v519;
                        v506 = v520;
                        v507 = v521;
                        v504 += 1l ;
                    }
                    bool v522;
                    v522 = v506 == 3l;
                    Union14 v533;
                    if (v522){
                        int v523;
                        v523 = 0l;
                        while (while_method_3(v523)){
                            int v525;
                            v525 = v505 + -2l;
                            bool v526;
                            v526 = v523 < v525;
                            int v527;
                            if (v526){
                                v527 = 0l;
                            } else {
                                v527 = 3l;
                            }
                            int v528;
                            v528 = v527 + v523;
                            unsigned char v529;
                            v529 = v84[v528];
                            v502[v523] = v529;
                            v523 += 1l ;
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
                            static_array<unsigned char,3l> v534 = v533.case1.v0; static_array<unsigned char,4l> v535 = v533.case1.v1;
                            static_array<unsigned char,2l> v536;
                            static_array<unsigned char,2l> v538;
                            int v540; int v541; int v542; unsigned char v543;
                            Tuple24 tmp71 = Tuple24{0l, 0l, 0l, 12u};
                            v540 = tmp71.v0; v541 = tmp71.v1; v542 = tmp71.v2; v543 = tmp71.v3;
                            while (while_method_3(v540)){
                                unsigned char v545;
                                v545 = v535[v540];
                                bool v547;
                                v547 = v542 < 2l;
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
                                        v550 = 0l;
                                    }
                                    v536[v550] = v545;
                                    int v551;
                                    v551 = v550 + 1l;
                                    v555 = v540; v556 = v551; v557 = v548;
                                } else {
                                    break;
                                }
                                v541 = v555;
                                v542 = v556;
                                v543 = v557;
                                v540 += 1l ;
                            }
                            bool v558;
                            v558 = v542 == 2l;
                            Union15 v569;
                            if (v558){
                                int v559;
                                v559 = 0l;
                                while (while_method_0(v559)){
                                    int v561;
                                    v561 = v541 + -1l;
                                    bool v562;
                                    v562 = v559 < v561;
                                    int v563;
                                    if (v562){
                                        v563 = 0l;
                                    } else {
                                        v563 = 2l;
                                    }
                                    int v564;
                                    v564 = v563 + v559;
                                    unsigned char v565;
                                    v565 = v535[v564];
                                    v538[v559] = v565;
                                    v559 += 1l ;
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
                                    static_array<unsigned char,2l> v570 = v569.case1.v0; static_array<unsigned char,2l> v571 = v569.case1.v1;
                                    static_array<unsigned char,5l> v572;
                                    int v574;
                                    v574 = 0l;
                                    while (while_method_1(v574)){
                                        unsigned char v576;
                                        v576 = v534[v574];
                                        v572[v574] = v576;
                                        v574 += 1l ;
                                    }
                                    int v578;
                                    v578 = 0l;
                                    while (while_method_0(v578)){
                                        unsigned char v580;
                                        v580 = v570[v578];
                                        int v582;
                                        v582 = 3l + v578;
                                        v572[v582] = v580;
                                        v578 += 1l ;
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
                            static_array<unsigned char,5l> v591;
                            int v593; int v594;
                            Tuple4 tmp72 = Tuple4{0l, 0l};
                            v593 = tmp72.v0; v594 = tmp72.v1;
                            while (while_method_16(v593)){
                                unsigned char v596;
                                v596 = v84[v593];
                                unsigned char v598;
                                v598 = v596 % 4u;
                                bool v599;
                                v599 = v598 == 0u;
                                bool v601;
                                if (v599){
                                    bool v600;
                                    v600 = v594 < 5l;
                                    v601 = v600;
                                } else {
                                    v601 = false;
                                }
                                int v603;
                                if (v601){
                                    v591[v594] = v596;
                                    int v602;
                                    v602 = v594 + 1l;
                                    v603 = v602;
                                } else {
                                    v603 = v594;
                                }
                                v594 = v603;
                                v593 += 1l ;
                            }
                            bool v604;
                            v604 = v594 == 5l;
                            Union12 v607;
                            if (v604){
                                v607 = Union12{Union12_1{v591}};
                            } else {
                                v607 = Union12{Union12_0{}};
                            }
                            static_array<unsigned char,5l> v608;
                            int v610; int v611;
                            Tuple4 tmp73 = Tuple4{0l, 0l};
                            v610 = tmp73.v0; v611 = tmp73.v1;
                            while (while_method_16(v610)){
                                unsigned char v613;
                                v613 = v84[v610];
                                unsigned char v615;
                                v615 = v613 % 4u;
                                bool v616;
                                v616 = v615 == 1u;
                                bool v618;
                                if (v616){
                                    bool v617;
                                    v617 = v611 < 5l;
                                    v618 = v617;
                                } else {
                                    v618 = false;
                                }
                                int v620;
                                if (v618){
                                    v608[v611] = v613;
                                    int v619;
                                    v619 = v611 + 1l;
                                    v620 = v619;
                                } else {
                                    v620 = v611;
                                }
                                v611 = v620;
                                v610 += 1l ;
                            }
                            bool v621;
                            v621 = v611 == 5l;
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
                                    static_array<unsigned char,5l> v625 = v607.case1.v0;
                                    switch (v624.tag) {
                                        case 0: { // None
                                            v652 = v607;
                                            break;
                                        }
                                        case 1: { // Some
                                            static_array<unsigned char,5l> v626 = v624.case1.v0;
                                            Union11 v627;
                                            v627 = Union11{Union11_0{}};
                                            int v628; Union11 v629;
                                            Tuple23 tmp74 = Tuple23{0l, v627};
                                            v628 = tmp74.v0; v629 = tmp74.v1;
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
                                                v628 += 1l ;
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
                                            static_array<unsigned char,5l> v647;
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
                            static_array<unsigned char,5l> v653;
                            int v655; int v656;
                            Tuple4 tmp75 = Tuple4{0l, 0l};
                            v655 = tmp75.v0; v656 = tmp75.v1;
                            while (while_method_16(v655)){
                                unsigned char v658;
                                v658 = v84[v655];
                                unsigned char v660;
                                v660 = v658 % 4u;
                                bool v661;
                                v661 = v660 == 2u;
                                bool v663;
                                if (v661){
                                    bool v662;
                                    v662 = v656 < 5l;
                                    v663 = v662;
                                } else {
                                    v663 = false;
                                }
                                int v665;
                                if (v663){
                                    v653[v656] = v658;
                                    int v664;
                                    v664 = v656 + 1l;
                                    v665 = v664;
                                } else {
                                    v665 = v656;
                                }
                                v656 = v665;
                                v655 += 1l ;
                            }
                            bool v666;
                            v666 = v656 == 5l;
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
                                    static_array<unsigned char,5l> v670 = v652.case1.v0;
                                    switch (v669.tag) {
                                        case 0: { // None
                                            v697 = v652;
                                            break;
                                        }
                                        case 1: { // Some
                                            static_array<unsigned char,5l> v671 = v669.case1.v0;
                                            Union11 v672;
                                            v672 = Union11{Union11_0{}};
                                            int v673; Union11 v674;
                                            Tuple23 tmp76 = Tuple23{0l, v672};
                                            v673 = tmp76.v0; v674 = tmp76.v1;
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
                                                v673 += 1l ;
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
                                            static_array<unsigned char,5l> v692;
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
                            static_array<unsigned char,5l> v698;
                            int v700; int v701;
                            Tuple4 tmp77 = Tuple4{0l, 0l};
                            v700 = tmp77.v0; v701 = tmp77.v1;
                            while (while_method_16(v700)){
                                unsigned char v703;
                                v703 = v84[v700];
                                unsigned char v705;
                                v705 = v703 % 4u;
                                bool v706;
                                v706 = v705 == 3u;
                                bool v708;
                                if (v706){
                                    bool v707;
                                    v707 = v701 < 5l;
                                    v708 = v707;
                                } else {
                                    v708 = false;
                                }
                                int v710;
                                if (v708){
                                    v698[v701] = v703;
                                    int v709;
                                    v709 = v701 + 1l;
                                    v710 = v709;
                                } else {
                                    v710 = v701;
                                }
                                v701 = v710;
                                v700 += 1l ;
                            }
                            bool v711;
                            v711 = v701 == 5l;
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
                                    static_array<unsigned char,5l> v715 = v697.case1.v0;
                                    switch (v714.tag) {
                                        case 0: { // None
                                            v742 = v697;
                                            break;
                                        }
                                        case 1: { // Some
                                            static_array<unsigned char,5l> v716 = v714.case1.v0;
                                            Union11 v717;
                                            v717 = Union11{Union11_0{}};
                                            int v718; Union11 v719;
                                            Tuple23 tmp78 = Tuple23{0l, v717};
                                            v718 = tmp78.v0; v719 = tmp78.v1;
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
                                                v718 += 1l ;
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
                                            static_array<unsigned char,5l> v737;
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
                                    static_array<unsigned char,5l> v744;
                                    int v746; int v747; unsigned char v748;
                                    Tuple22 tmp79 = Tuple22{0l, 0l, 12u};
                                    v746 = tmp79.v0; v747 = tmp79.v1; v748 = tmp79.v2;
                                    while (while_method_16(v746)){
                                        unsigned char v750;
                                        v750 = v84[v746];
                                        bool v752;
                                        v752 = v747 < 5l;
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
                                                    v758 = 0l;
                                                }
                                                v744[v758] = v750;
                                                int v759;
                                                v759 = v758 + 1l;
                                                v764 = v759; v765 = v754;
                                            } else {
                                                v764 = v747; v765 = v748;
                                            }
                                        } else {
                                            break;
                                        }
                                        v747 = v764;
                                        v748 = v765;
                                        v746 += 1l ;
                                    }
                                    bool v766;
                                    v766 = v747 == 4l;
                                    bool v775;
                                    if (v766){
                                        unsigned char v767;
                                        v767 = v748 + 1u;
                                        bool v768;
                                        v768 = v767 == 0u;
                                        if (v768){
                                            unsigned char v769;
                                            v769 = v84[0l];
                                            unsigned char v771;
                                            v771 = v769 / 4u;
                                            bool v772;
                                            v772 = v771 == 12u;
                                            if (v772){
                                                v744[4l] = v769;
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
                                        v777 = v747 == 5l;
                                        if (v777){
                                            v781 = Union12{Union12_1{v744}};
                                        } else {
                                            v781 = Union12{Union12_0{}};
                                        }
                                    }
                                    switch (v781.tag) {
                                        case 0: { // None
                                            static_array<unsigned char,3l> v783;
                                            static_array<unsigned char,4l> v785;
                                            int v787; int v788; int v789; unsigned char v790;
                                            Tuple24 tmp80 = Tuple24{0l, 0l, 0l, 12u};
                                            v787 = tmp80.v0; v788 = tmp80.v1; v789 = tmp80.v2; v790 = tmp80.v3;
                                            while (while_method_16(v787)){
                                                unsigned char v792;
                                                v792 = v84[v787];
                                                bool v794;
                                                v794 = v789 < 3l;
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
                                                        v797 = 0l;
                                                    }
                                                    v783[v797] = v792;
                                                    int v798;
                                                    v798 = v797 + 1l;
                                                    v802 = v787; v803 = v798; v804 = v795;
                                                } else {
                                                    break;
                                                }
                                                v788 = v802;
                                                v789 = v803;
                                                v790 = v804;
                                                v787 += 1l ;
                                            }
                                            bool v805;
                                            v805 = v789 == 3l;
                                            Union14 v816;
                                            if (v805){
                                                int v806;
                                                v806 = 0l;
                                                while (while_method_3(v806)){
                                                    int v808;
                                                    v808 = v788 + -2l;
                                                    bool v809;
                                                    v809 = v806 < v808;
                                                    int v810;
                                                    if (v809){
                                                        v810 = 0l;
                                                    } else {
                                                        v810 = 3l;
                                                    }
                                                    int v811;
                                                    v811 = v810 + v806;
                                                    unsigned char v812;
                                                    v812 = v84[v811];
                                                    v785[v806] = v812;
                                                    v806 += 1l ;
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
                                                    static_array<unsigned char,3l> v817 = v816.case1.v0; static_array<unsigned char,4l> v818 = v816.case1.v1;
                                                    static_array<unsigned char,2l> v819;
                                                    int v821;
                                                    v821 = 0l;
                                                    while (while_method_0(v821)){
                                                        unsigned char v823;
                                                        v823 = v818[v821];
                                                        v819[v821] = v823;
                                                        v821 += 1l ;
                                                    }
                                                    static_array<unsigned char,5l> v825;
                                                    int v827;
                                                    v827 = 0l;
                                                    while (while_method_1(v827)){
                                                        unsigned char v829;
                                                        v829 = v817[v827];
                                                        v825[v827] = v829;
                                                        v827 += 1l ;
                                                    }
                                                    int v831;
                                                    v831 = 0l;
                                                    while (while_method_0(v831)){
                                                        unsigned char v833;
                                                        v833 = v819[v831];
                                                        int v835;
                                                        v835 = 3l + v831;
                                                        v825[v835] = v833;
                                                        v831 += 1l ;
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
                                                    static_array<unsigned char,2l> v841;
                                                    static_array<unsigned char,5l> v843;
                                                    int v845; int v846; int v847; unsigned char v848;
                                                    Tuple24 tmp81 = Tuple24{0l, 0l, 0l, 12u};
                                                    v845 = tmp81.v0; v846 = tmp81.v1; v847 = tmp81.v2; v848 = tmp81.v3;
                                                    while (while_method_16(v845)){
                                                        unsigned char v850;
                                                        v850 = v84[v845];
                                                        bool v852;
                                                        v852 = v847 < 2l;
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
                                                                v855 = 0l;
                                                            }
                                                            v841[v855] = v850;
                                                            int v856;
                                                            v856 = v855 + 1l;
                                                            v860 = v845; v861 = v856; v862 = v853;
                                                        } else {
                                                            break;
                                                        }
                                                        v846 = v860;
                                                        v847 = v861;
                                                        v848 = v862;
                                                        v845 += 1l ;
                                                    }
                                                    bool v863;
                                                    v863 = v847 == 2l;
                                                    Union16 v874;
                                                    if (v863){
                                                        int v864;
                                                        v864 = 0l;
                                                        while (while_method_2(v864)){
                                                            int v866;
                                                            v866 = v846 + -1l;
                                                            bool v867;
                                                            v867 = v864 < v866;
                                                            int v868;
                                                            if (v867){
                                                                v868 = 0l;
                                                            } else {
                                                                v868 = 2l;
                                                            }
                                                            int v869;
                                                            v869 = v868 + v864;
                                                            unsigned char v870;
                                                            v870 = v84[v869];
                                                            v843[v864] = v870;
                                                            v864 += 1l ;
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
                                                            static_array<unsigned char,2l> v875 = v874.case1.v0; static_array<unsigned char,5l> v876 = v874.case1.v1;
                                                            static_array<unsigned char,2l> v877;
                                                            static_array<unsigned char,3l> v879;
                                                            int v881; int v882; int v883; unsigned char v884;
                                                            Tuple24 tmp82 = Tuple24{0l, 0l, 0l, 12u};
                                                            v881 = tmp82.v0; v882 = tmp82.v1; v883 = tmp82.v2; v884 = tmp82.v3;
                                                            while (while_method_2(v881)){
                                                                unsigned char v886;
                                                                v886 = v876[v881];
                                                                bool v888;
                                                                v888 = v883 < 2l;
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
                                                                        v891 = 0l;
                                                                    }
                                                                    v877[v891] = v886;
                                                                    int v892;
                                                                    v892 = v891 + 1l;
                                                                    v896 = v881; v897 = v892; v898 = v889;
                                                                } else {
                                                                    break;
                                                                }
                                                                v882 = v896;
                                                                v883 = v897;
                                                                v884 = v898;
                                                                v881 += 1l ;
                                                            }
                                                            bool v899;
                                                            v899 = v883 == 2l;
                                                            Union17 v910;
                                                            if (v899){
                                                                int v900;
                                                                v900 = 0l;
                                                                while (while_method_1(v900)){
                                                                    int v902;
                                                                    v902 = v882 + -1l;
                                                                    bool v903;
                                                                    v903 = v900 < v902;
                                                                    int v904;
                                                                    if (v903){
                                                                        v904 = 0l;
                                                                    } else {
                                                                        v904 = 2l;
                                                                    }
                                                                    int v905;
                                                                    v905 = v904 + v900;
                                                                    unsigned char v906;
                                                                    v906 = v876[v905];
                                                                    v879[v900] = v906;
                                                                    v900 += 1l ;
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
                                                                    static_array<unsigned char,2l> v911 = v910.case1.v0; static_array<unsigned char,3l> v912 = v910.case1.v1;
                                                                    static_array<unsigned char,1l> v913;
                                                                    int v915;
                                                                    v915 = 0l;
                                                                    while (while_method_6(v915)){
                                                                        unsigned char v917;
                                                                        v917 = v912[v915];
                                                                        v913[v915] = v917;
                                                                        v915 += 1l ;
                                                                    }
                                                                    static_array<unsigned char,5l> v919;
                                                                    int v921;
                                                                    v921 = 0l;
                                                                    while (while_method_0(v921)){
                                                                        unsigned char v923;
                                                                        v923 = v875[v921];
                                                                        v919[v921] = v923;
                                                                        v921 += 1l ;
                                                                    }
                                                                    int v925;
                                                                    v925 = 0l;
                                                                    while (while_method_0(v925)){
                                                                        unsigned char v927;
                                                                        v927 = v911[v925];
                                                                        int v929;
                                                                        v929 = 2l + v925;
                                                                        v919[v929] = v927;
                                                                        v925 += 1l ;
                                                                    }
                                                                    int v930;
                                                                    v930 = 0l;
                                                                    while (while_method_6(v930)){
                                                                        unsigned char v932;
                                                                        v932 = v913[v930];
                                                                        int v934;
                                                                        v934 = 4l + v930;
                                                                        v919[v934] = v932;
                                                                        v930 += 1l ;
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
                                                            static_array<unsigned char,2l> v943;
                                                            static_array<unsigned char,5l> v945;
                                                            int v947; int v948; int v949; unsigned char v950;
                                                            Tuple24 tmp83 = Tuple24{0l, 0l, 0l, 12u};
                                                            v947 = tmp83.v0; v948 = tmp83.v1; v949 = tmp83.v2; v950 = tmp83.v3;
                                                            while (while_method_16(v947)){
                                                                unsigned char v952;
                                                                v952 = v84[v947];
                                                                bool v954;
                                                                v954 = v949 < 2l;
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
                                                                        v957 = 0l;
                                                                    }
                                                                    v943[v957] = v952;
                                                                    int v958;
                                                                    v958 = v957 + 1l;
                                                                    v962 = v947; v963 = v958; v964 = v955;
                                                                } else {
                                                                    break;
                                                                }
                                                                v948 = v962;
                                                                v949 = v963;
                                                                v950 = v964;
                                                                v947 += 1l ;
                                                            }
                                                            bool v965;
                                                            v965 = v949 == 2l;
                                                            Union16 v976;
                                                            if (v965){
                                                                int v966;
                                                                v966 = 0l;
                                                                while (while_method_2(v966)){
                                                                    int v968;
                                                                    v968 = v948 + -1l;
                                                                    bool v969;
                                                                    v969 = v966 < v968;
                                                                    int v970;
                                                                    if (v969){
                                                                        v970 = 0l;
                                                                    } else {
                                                                        v970 = 2l;
                                                                    }
                                                                    int v971;
                                                                    v971 = v970 + v966;
                                                                    unsigned char v972;
                                                                    v972 = v84[v971];
                                                                    v945[v966] = v972;
                                                                    v966 += 1l ;
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
                                                                    static_array<unsigned char,2l> v977 = v976.case1.v0; static_array<unsigned char,5l> v978 = v976.case1.v1;
                                                                    static_array<unsigned char,3l> v979;
                                                                    int v981;
                                                                    v981 = 0l;
                                                                    while (while_method_1(v981)){
                                                                        unsigned char v983;
                                                                        v983 = v978[v981];
                                                                        v979[v981] = v983;
                                                                        v981 += 1l ;
                                                                    }
                                                                    static_array<unsigned char,5l> v985;
                                                                    int v987;
                                                                    v987 = 0l;
                                                                    while (while_method_0(v987)){
                                                                        unsigned char v989;
                                                                        v989 = v977[v987];
                                                                        v985[v987] = v989;
                                                                        v987 += 1l ;
                                                                    }
                                                                    int v991;
                                                                    v991 = 0l;
                                                                    while (while_method_1(v991)){
                                                                        unsigned char v993;
                                                                        v993 = v979[v991];
                                                                        int v995;
                                                                        v995 = 2l + v991;
                                                                        v985[v995] = v993;
                                                                        v991 += 1l ;
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
                                                                    static_array<unsigned char,5l> v1001;
                                                                    int v1003;
                                                                    v1003 = 0l;
                                                                    while (while_method_2(v1003)){
                                                                        unsigned char v1005;
                                                                        v1005 = v84[v1003];
                                                                        v1001[v1003] = v1005;
                                                                        v1003 += 1l ;
                                                                    }
                                                                    v1037 = v1001; v1038 = 0;
                                                                    break;
                                                                }
                                                                case 1: { // Some
                                                                    static_array<unsigned char,5l> v1000 = v999.case1.v0;
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
                                                            static_array<unsigned char,5l> v942 = v941.case1.v0;
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
                                                    static_array<unsigned char,5l> v840 = v839.case1.v0;
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
                                            static_array<unsigned char,5l> v782 = v781.case1.v0;
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
                                    static_array<unsigned char,5l> v743 = v742.case1.v0;
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
                            static_array<unsigned char,5l> v590 = v589.case1.v0;
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
                    static_array<unsigned char,5l> v499 = v498.case1.v0;
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
            static_array<unsigned char,5l> v441 = v440.case1.v0;
            v1037 = v441; v1038 = 8;
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    return Tuple0{v1037, v1038};
}
__device__ void play_loop_31(curandStatePhilox4_32_10_t & v0, unsigned char * v1, unsigned long long v2, unsigned char * v3, unsigned long long v4, unsigned long long & v5, Union3 & v6, dynamic_array_list<Union6,128l> & v7, static_array<Union2,2l> & v8, Union7 & v9, Union4 v10){
    dynamic_array_list<Union6,128l> & v11 = v7;
    unsigned long long & v12 = v5;
    Union3 v13;
    v13 = Union3{Union3_1{v10}};
    Union3 v14;
    v14 = v13;
    while (while_method_5(v14)){
        Union3 v998;
        switch (v14.tag) {
            case 0: { // None
                v998 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v16 = v14.case1.v0;
                switch (v16.tag) {
                    case 0: { // G_Flop
                        int v891 = v16.case0.v0; static_array<static_array<unsigned char,2l>,2l> v892 = v16.case0.v1; static_array<int,2l> v893 = v16.case0.v2; int v894 = v16.case0.v3; static_array<int,2l> v895 = v16.case0.v4; Union5 v896 = v16.case0.v5;
                        static_array<unsigned char,3l> v897; unsigned long long v898;
                        Tuple8 tmp18 = draw_cards_32(v0, v12);
                        v897 = tmp18.v0; v898 = tmp18.v1;
                        v5 = v898;
                        static_array_list<unsigned char,5l> v899;
                        v899 = get_community_cards_35(v896, v897);
                        Union6 v900;
                        v900 = Union6{Union6_0{v899}};
                        v11.push(v900);
                        Union5 v903;
                        switch (v896.tag) {
                            case 1: { // Preflop
                                v903 = Union5{Union5_0{v897}};
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in flop.");
                                __trap();
                            }
                        }
                        int v904;
                        v904 = 2l;
                        int v905;
                        v905 = 0l;
                        Union4 v906;
                        v906 = try_round_36(v904, v892, v893, v905, v895, v903);
                        v998 = Union3{Union3_1{v906}};
                        break;
                    }
                    case 1: { // G_Fold
                        int v17 = v16.case1.v0; static_array<static_array<unsigned char,2l>,2l> v18 = v16.case1.v1; static_array<int,2l> v19 = v16.case1.v2; int v20 = v16.case1.v3; static_array<int,2l> v21 = v16.case1.v4; Union5 v22 = v16.case1.v5;
                        int v23;
                        v23 = v20 % 2l;
                        int v24;
                        v24 = v19[v23];
                        int v26;
                        v26 = v20 + 1l;
                        int v27;
                        v27 = v26 % 2l;
                        Union6 v28;
                        v28 = Union6{Union6_1{v24, v27}};
                        v11.push(v28);
                        Union7 v29;
                        v29 = Union7{Union7_1{v17, v18, v19, v20, v21, v22}};
                        v9 = v29;
                        Union3 v30;
                        v30 = Union3{Union3_0{}};
                        v6 = v30;
                        v998 = Union3{Union3_0{}};
                        break;
                    }
                    case 2: { // G_Preflop
                        static_array<unsigned char,2l> v966; unsigned long long v967;
                        Tuple11 tmp23 = draw_cards_39(v0, v12);
                        v966 = tmp23.v0; v967 = tmp23.v1;
                        v5 = v967;
                        static_array<unsigned char,2l> v968; unsigned long long v969;
                        Tuple11 tmp24 = draw_cards_39(v0, v12);
                        v968 = tmp24.v0; v969 = tmp24.v1;
                        v5 = v969;
                        Union6 v970;
                        v970 = Union6{Union6_3{0l, v966}};
                        v11.push(v970);
                        Union6 v971;
                        v971 = Union6{Union6_3{1l, v968}};
                        v11.push(v971);
                        static_array<static_array<unsigned char,2l>,2l> v972;
                        v972[0l] = v966;
                        v972[1l] = v968;
                        static_array<int,2l> v974;
                        v974[0l] = 2l;
                        v974[1l] = 1l;
                        static_array<int,2l> v976;
                        int v978;
                        v978 = 0l;
                        while (while_method_0(v978)){
                            int v980;
                            v980 = v974[v978];
                            int v982;
                            v982 = 100l - v980;
                            v976[v978] = v982;
                            v978 += 1l ;
                        }
                        int v983;
                        v983 = 2l;
                        int v984;
                        v984 = 0l;
                        Union5 v985;
                        v985 = Union5{Union5_1{}};
                        Union4 v986;
                        v986 = try_round_36(v983, v972, v974, v984, v976, v985);
                        v998 = Union3{Union3_1{v986}};
                        break;
                    }
                    case 3: { // G_River
                        int v937 = v16.case3.v0; static_array<static_array<unsigned char,2l>,2l> v938 = v16.case3.v1; static_array<int,2l> v939 = v16.case3.v2; int v940 = v16.case3.v3; static_array<int,2l> v941 = v16.case3.v4; Union5 v942 = v16.case3.v5;
                        static_array<unsigned char,1l> v943; unsigned long long v944;
                        Tuple12 tmp27 = draw_cards_40(v0, v12);
                        v943 = tmp27.v0; v944 = tmp27.v1;
                        v5 = v944;
                        static_array_list<unsigned char,5l> v945;
                        v945 = get_community_cards_41(v942, v943);
                        Union6 v946;
                        v946 = Union6{Union6_0{v945}};
                        v11.push(v946);
                        Union5 v961;
                        switch (v942.tag) {
                            case 3: { // Turn
                                static_array<unsigned char,4l> v947 = v942.case3.v0;
                                static_array<unsigned char,5l> v948;
                                int v950;
                                v950 = 0l;
                                while (while_method_3(v950)){
                                    unsigned char v952;
                                    v952 = v947[v950];
                                    v948[v950] = v952;
                                    v950 += 1l ;
                                }
                                int v954;
                                v954 = 0l;
                                while (while_method_6(v954)){
                                    unsigned char v956;
                                    v956 = v943[v954];
                                    int v958;
                                    v958 = 4l + v954;
                                    v948[v958] = v956;
                                    v954 += 1l ;
                                }
                                v961 = Union5{Union5_2{v948}};
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in river.");
                                __trap();
                            }
                        }
                        int v962;
                        v962 = 2l;
                        int v963;
                        v963 = 0l;
                        Union4 v964;
                        v964 = try_round_36(v962, v938, v939, v963, v941, v961);
                        v998 = Union3{Union3_1{v964}};
                        break;
                    }
                    case 4: { // G_Round
                        int v112 = v16.case4.v0; static_array<static_array<unsigned char,2l>,2l> v113 = v16.case4.v1; static_array<int,2l> v114 = v16.case4.v2; int v115 = v16.case4.v3; static_array<int,2l> v116 = v16.case4.v4; Union5 v117 = v16.case4.v5;
                        int v118;
                        v118 = v115 % 2l;
                        static_array<Union2,2l> v119 = v8;
                        Union2 v120;
                        v120 = v119[v118];
                        switch (v120.tag) {
                            case 0: { // Computer
                                bool v122;
                                v122 = 5652480ull == v4;
                                bool v123;
                                v123 = v122 == false;
                                if (v123){
                                    assert("The params needs to have matching offsets." && v122);
                                } else {
                                }
                                bool v125;
                                v125 = 279040ull == v2;
                                bool v126;
                                v126 = v125 == false;
                                if (v126){
                                    assert("The outputs needs to have matching offsets." && v125);
                                } else {
                                }
                                dynamic_array_list<Union6,128l> & v128 = v7;
                                cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v129 = console_lock;
                                auto v130 = cooperative_groups::coalesced_threads();
                                v129.acquire();
                                printf("%s\n","Running the GPU model.");
                                v129.release();
                                v130.sync() ;
                                Union8 v133;
                                v133 = Union8{Union8_1{v112, v113, v114, v115, v116, v117, v128}};
                                Union9 v134;
                                v134 = noinline_run_42(v1, v3, v133);
                                Union1 v212;
                                switch (v134.tag) {
                                    case 0: { // AA_Call
                                        v212 = Union1{Union1_1{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v135;
                                        v135 = v114[0l];
                                        int v137; int v138;
                                        Tuple4 tmp44 = Tuple4{1l, v135};
                                        v137 = tmp44.v0; v138 = tmp44.v1;
                                        while (while_method_0(v137)){
                                            int v140;
                                            v140 = v114[v137];
                                            bool v142;
                                            v142 = v138 >= v140;
                                            int v143;
                                            if (v142){
                                                v143 = v138;
                                            } else {
                                                v143 = v140;
                                            }
                                            v138 = v143;
                                            v137 += 1l ;
                                        }
                                        int v144;
                                        v144 = v114[v118];
                                        bool v146;
                                        v146 = v144 == v138;
                                        if (v146){
                                            v212 = Union1{Union1_1{}};
                                        } else {
                                            v212 = Union1{Union1_2{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        int v151 = v134.case2.v0; int v152 = v134.case2.v1;
                                        static_array<int,2l> v153;
                                        int v155;
                                        v155 = 0l;
                                        while (while_method_0(v155)){
                                            int v157;
                                            v157 = v116[v155];
                                            int v159;
                                            v159 = v114[v155];
                                            int v161;
                                            v161 = v157 + v159;
                                            v153[v155] = v161;
                                            v155 += 1l ;
                                        }
                                        int v162;
                                        v162 = v114[0l];
                                        int v164; int v165;
                                        Tuple4 tmp45 = Tuple4{1l, v162};
                                        v164 = tmp45.v0; v165 = tmp45.v1;
                                        while (while_method_0(v164)){
                                            int v167;
                                            v167 = v114[v164];
                                            bool v169;
                                            v169 = v165 >= v167;
                                            int v170;
                                            if (v169){
                                                v170 = v165;
                                            } else {
                                                v170 = v167;
                                            }
                                            v165 = v170;
                                            v164 += 1l ;
                                        }
                                        int v171;
                                        v171 = v153[v118];
                                        bool v173;
                                        v173 = v165 < v171;
                                        int v174;
                                        if (v173){
                                            v174 = v165;
                                        } else {
                                            v174 = v171;
                                        }
                                        static_array<int,2l> v175;
                                        int v177;
                                        v177 = 0l;
                                        while (while_method_0(v177)){
                                            int v179;
                                            v179 = v114[v177];
                                            bool v181;
                                            v181 = v118 == v177;
                                            int v182;
                                            if (v181){
                                                v182 = v174;
                                            } else {
                                                v182 = v179;
                                            }
                                            v175[v177] = v182;
                                            v177 += 1l ;
                                        }
                                        int v183;
                                        v183 = v175[0l];
                                        int v185; int v186;
                                        Tuple4 tmp46 = Tuple4{1l, v183};
                                        v185 = tmp46.v0; v186 = tmp46.v1;
                                        while (while_method_0(v185)){
                                            int v188;
                                            v188 = v175[v185];
                                            int v190;
                                            v190 = v186 + v188;
                                            v186 = v190;
                                            v185 += 1l ;
                                        }
                                        static_array<int,2l> v191;
                                        int v193;
                                        v193 = 0l;
                                        while (while_method_0(v193)){
                                            int v195;
                                            v195 = v153[v193];
                                            int v197;
                                            v197 = v175[v193];
                                            int v199;
                                            v199 = v195 - v197;
                                            v191[v193] = v199;
                                            v193 += 1l ;
                                        }
                                        int v200;
                                        v200 = v151 * v186;
                                        int v201;
                                        v201 = v200 / v152;
                                        bool v202;
                                        v202 = v112 >= v201;
                                        int v203;
                                        if (v202){
                                            v203 = v112;
                                        } else {
                                            v203 = v201;
                                        }
                                        int v204;
                                        v204 = v191[v118];
                                        bool v206;
                                        v206 = v203 >= v204;
                                        if (v206){
                                            v212 = Union1{Union1_0{}};
                                        } else {
                                            v212 = Union1{Union1_3{v203}};
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v213 = console_lock;
                                auto v214 = cooperative_groups::coalesced_threads();
                                v213.acquire();
                                printf("%s","The action is: ");
                                v213.release();
                                v214.sync() ;
                                cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v217 = console_lock;
                                auto v218 = cooperative_groups::coalesced_threads();
                                v217.acquire();
                                printf("");
                                method_49(v212);
                                printf("\n");
                                v217.release();
                                v218.sync() ;
                                Union6 v221;
                                v221 = Union6{Union6_2{v118, v212}};
                                v11.push(v221);
                                Union4 v409;
                                switch (v212.tag) {
                                    case 0: { // A_All_In
                                        static_array<int,2l> v339;
                                        int v341;
                                        v341 = 0l;
                                        while (while_method_0(v341)){
                                            int v343;
                                            v343 = v116[v341];
                                            int v345;
                                            v345 = v114[v341];
                                            int v347;
                                            v347 = v343 + v345;
                                            v339[v341] = v347;
                                            v341 += 1l ;
                                        }
                                        int v348;
                                        v348 = v114[0l];
                                        int v350; int v351;
                                        Tuple4 tmp47 = Tuple4{1l, v348};
                                        v350 = tmp47.v0; v351 = tmp47.v1;
                                        while (while_method_0(v350)){
                                            int v353;
                                            v353 = v114[v350];
                                            bool v355;
                                            v355 = v351 >= v353;
                                            int v356;
                                            if (v355){
                                                v356 = v351;
                                            } else {
                                                v356 = v353;
                                            }
                                            v351 = v356;
                                            v350 += 1l ;
                                        }
                                        int v357;
                                        v357 = v339[v118];
                                        bool v359;
                                        v359 = v351 < v357;
                                        int v360;
                                        if (v359){
                                            v360 = v351;
                                        } else {
                                            v360 = v357;
                                        }
                                        static_array<int,2l> v361;
                                        int v363;
                                        v363 = 0l;
                                        while (while_method_0(v363)){
                                            int v365;
                                            v365 = v114[v363];
                                            bool v367;
                                            v367 = v118 == v363;
                                            int v368;
                                            if (v367){
                                                v368 = v360;
                                            } else {
                                                v368 = v365;
                                            }
                                            v361[v363] = v368;
                                            v363 += 1l ;
                                        }
                                        static_array<int,2l> v369;
                                        int v371;
                                        v371 = 0l;
                                        while (while_method_0(v371)){
                                            int v373;
                                            v373 = v339[v371];
                                            int v375;
                                            v375 = v361[v371];
                                            int v377;
                                            v377 = v373 - v375;
                                            v369[v371] = v377;
                                            v371 += 1l ;
                                        }
                                        int v378;
                                        v378 = v369[v118];
                                        int v380;
                                        v380 = v351 + v378;
                                        int v381;
                                        v381 = v339[v118];
                                        bool v383;
                                        v383 = v380 < v381;
                                        int v384;
                                        if (v383){
                                            v384 = v380;
                                        } else {
                                            v384 = v381;
                                        }
                                        static_array<int,2l> v385;
                                        int v387;
                                        v387 = 0l;
                                        while (while_method_0(v387)){
                                            int v389;
                                            v389 = v114[v387];
                                            bool v391;
                                            v391 = v118 == v387;
                                            int v392;
                                            if (v391){
                                                v392 = v384;
                                            } else {
                                                v392 = v389;
                                            }
                                            v385[v387] = v392;
                                            v387 += 1l ;
                                        }
                                        static_array<int,2l> v393;
                                        int v395;
                                        v395 = 0l;
                                        while (while_method_0(v395)){
                                            int v397;
                                            v397 = v339[v395];
                                            int v399;
                                            v399 = v385[v395];
                                            int v401;
                                            v401 = v397 - v399;
                                            v393[v395] = v401;
                                            v395 += 1l ;
                                        }
                                        bool v402;
                                        v402 = v378 >= v112;
                                        int v403;
                                        if (v402){
                                            v403 = v378;
                                        } else {
                                            v403 = v112;
                                        }
                                        int v404;
                                        v404 = v115 + 1l;
                                        v409 = try_round_36(v403, v113, v385, v404, v393, v117);
                                        break;
                                    }
                                    case 1: { // A_Call
                                        static_array<int,2l> v223;
                                        int v225;
                                        v225 = 0l;
                                        while (while_method_0(v225)){
                                            int v227;
                                            v227 = v116[v225];
                                            int v229;
                                            v229 = v114[v225];
                                            int v231;
                                            v231 = v227 + v229;
                                            v223[v225] = v231;
                                            v225 += 1l ;
                                        }
                                        int v232;
                                        v232 = v114[0l];
                                        int v234; int v235;
                                        Tuple4 tmp48 = Tuple4{1l, v232};
                                        v234 = tmp48.v0; v235 = tmp48.v1;
                                        while (while_method_0(v234)){
                                            int v237;
                                            v237 = v114[v234];
                                            bool v239;
                                            v239 = v235 >= v237;
                                            int v240;
                                            if (v239){
                                                v240 = v235;
                                            } else {
                                                v240 = v237;
                                            }
                                            v235 = v240;
                                            v234 += 1l ;
                                        }
                                        int v241;
                                        v241 = v223[v118];
                                        bool v243;
                                        v243 = v235 < v241;
                                        int v244;
                                        if (v243){
                                            v244 = v235;
                                        } else {
                                            v244 = v241;
                                        }
                                        static_array<int,2l> v245;
                                        int v247;
                                        v247 = 0l;
                                        while (while_method_0(v247)){
                                            int v249;
                                            v249 = v114[v247];
                                            bool v251;
                                            v251 = v118 == v247;
                                            int v252;
                                            if (v251){
                                                v252 = v244;
                                            } else {
                                                v252 = v249;
                                            }
                                            v245[v247] = v252;
                                            v247 += 1l ;
                                        }
                                        static_array<int,2l> v253;
                                        int v255;
                                        v255 = 0l;
                                        while (while_method_0(v255)){
                                            int v257;
                                            v257 = v223[v255];
                                            int v259;
                                            v259 = v245[v255];
                                            int v261;
                                            v261 = v257 - v259;
                                            v253[v255] = v261;
                                            v255 += 1l ;
                                        }
                                        bool v262;
                                        v262 = v118 < 2l;
                                        if (v262){
                                            int v263;
                                            v263 = v115 + 1l;
                                            v409 = try_round_36(v112, v113, v245, v263, v253, v117);
                                        } else {
                                            v409 = go_next_street_38(v112, v113, v245, v115, v253, v117);
                                        }
                                        break;
                                    }
                                    case 2: { // A_Fold
                                        v409 = Union4{Union4_1{v112, v113, v114, v115, v116, v117}};
                                        break;
                                    }
                                    case 3: { // A_Raise
                                        int v267 = v212.case3.v0;
                                        bool v268;
                                        v268 = v112 <= v267;
                                        bool v269;
                                        v269 = v268 == false;
                                        if (v269){
                                            assert("The raise amount must match the minimum." && v268);
                                        } else {
                                        }
                                        static_array<int,2l> v271;
                                        int v273;
                                        v273 = 0l;
                                        while (while_method_0(v273)){
                                            int v275;
                                            v275 = v116[v273];
                                            int v277;
                                            v277 = v114[v273];
                                            int v279;
                                            v279 = v275 + v277;
                                            v271[v273] = v279;
                                            v273 += 1l ;
                                        }
                                        int v280;
                                        v280 = v114[0l];
                                        int v282; int v283;
                                        Tuple4 tmp49 = Tuple4{1l, v280};
                                        v282 = tmp49.v0; v283 = tmp49.v1;
                                        while (while_method_0(v282)){
                                            int v285;
                                            v285 = v114[v282];
                                            bool v287;
                                            v287 = v283 >= v285;
                                            int v288;
                                            if (v287){
                                                v288 = v283;
                                            } else {
                                                v288 = v285;
                                            }
                                            v283 = v288;
                                            v282 += 1l ;
                                        }
                                        int v289;
                                        v289 = v271[v118];
                                        bool v291;
                                        v291 = v283 < v289;
                                        int v292;
                                        if (v291){
                                            v292 = v283;
                                        } else {
                                            v292 = v289;
                                        }
                                        static_array<int,2l> v293;
                                        int v295;
                                        v295 = 0l;
                                        while (while_method_0(v295)){
                                            int v297;
                                            v297 = v114[v295];
                                            bool v299;
                                            v299 = v118 == v295;
                                            int v300;
                                            if (v299){
                                                v300 = v292;
                                            } else {
                                                v300 = v297;
                                            }
                                            v293[v295] = v300;
                                            v295 += 1l ;
                                        }
                                        static_array<int,2l> v301;
                                        int v303;
                                        v303 = 0l;
                                        while (while_method_0(v303)){
                                            int v305;
                                            v305 = v271[v303];
                                            int v307;
                                            v307 = v293[v303];
                                            int v309;
                                            v309 = v305 - v307;
                                            v301[v303] = v309;
                                            v303 += 1l ;
                                        }
                                        int v310;
                                        v310 = v301[v118];
                                        bool v312;
                                        v312 = v267 < v310;
                                        bool v313;
                                        v313 = v312 == false;
                                        if (v313){
                                            assert("The raise amount must be less than the stack size after calling." && v312);
                                        } else {
                                        }
                                        int v315;
                                        v315 = v283 + v267;
                                        int v316;
                                        v316 = v271[v118];
                                        bool v318;
                                        v318 = v315 < v316;
                                        int v319;
                                        if (v318){
                                            v319 = v315;
                                        } else {
                                            v319 = v316;
                                        }
                                        static_array<int,2l> v320;
                                        int v322;
                                        v322 = 0l;
                                        while (while_method_0(v322)){
                                            int v324;
                                            v324 = v114[v322];
                                            bool v326;
                                            v326 = v118 == v322;
                                            int v327;
                                            if (v326){
                                                v327 = v319;
                                            } else {
                                                v327 = v324;
                                            }
                                            v320[v322] = v327;
                                            v322 += 1l ;
                                        }
                                        static_array<int,2l> v328;
                                        int v330;
                                        v330 = 0l;
                                        while (while_method_0(v330)){
                                            int v332;
                                            v332 = v271[v330];
                                            int v334;
                                            v334 = v320[v330];
                                            int v336;
                                            v336 = v332 - v334;
                                            v328[v330] = v336;
                                            v330 += 1l ;
                                        }
                                        int v337;
                                        v337 = v115 + 1l;
                                        v409 = try_round_36(v267, v113, v320, v337, v328, v117);
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                v998 = Union3{Union3_1{v409}};
                                break;
                            }
                            case 1: { // Human
                                Union7 v411;
                                v411 = Union7{Union7_2{v112, v113, v114, v115, v116, v117}};
                                v9 = v411;
                                Union3 v412;
                                v412 = Union3{Union3_1{v16}};
                                v6 = v412;
                                v998 = Union3{Union3_0{}};
                                break;
                            }
                            case 2: { // Random
                                static_array<int,2l> v414;
                                int v416;
                                v416 = 0l;
                                while (while_method_0(v416)){
                                    int v418;
                                    v418 = v116[v416];
                                    int v420;
                                    v420 = v114[v416];
                                    int v422;
                                    v422 = v418 + v420;
                                    v414[v416] = v422;
                                    v416 += 1l ;
                                }
                                int v423;
                                v423 = v114[0l];
                                int v425; int v426;
                                Tuple4 tmp50 = Tuple4{1l, v423};
                                v425 = tmp50.v0; v426 = tmp50.v1;
                                while (while_method_0(v425)){
                                    int v428;
                                    v428 = v114[v425];
                                    bool v430;
                                    v430 = v426 >= v428;
                                    int v431;
                                    if (v430){
                                        v431 = v426;
                                    } else {
                                        v431 = v428;
                                    }
                                    v426 = v431;
                                    v425 += 1l ;
                                }
                                int v432;
                                v432 = v414[v118];
                                bool v434;
                                v434 = v426 < v432;
                                int v435;
                                if (v434){
                                    v435 = v426;
                                } else {
                                    v435 = v432;
                                }
                                static_array<int,2l> v436;
                                int v438;
                                v438 = 0l;
                                while (while_method_0(v438)){
                                    int v440;
                                    v440 = v114[v438];
                                    bool v442;
                                    v442 = v118 == v438;
                                    int v443;
                                    if (v442){
                                        v443 = v435;
                                    } else {
                                        v443 = v440;
                                    }
                                    v436[v438] = v443;
                                    v438 += 1l ;
                                }
                                int v444;
                                v444 = v436[0l];
                                int v446; int v447;
                                Tuple4 tmp51 = Tuple4{1l, v444};
                                v446 = tmp51.v0; v447 = tmp51.v1;
                                while (while_method_0(v446)){
                                    int v449;
                                    v449 = v436[v446];
                                    int v451;
                                    v451 = v447 + v449;
                                    v447 = v451;
                                    v446 += 1l ;
                                }
                                static_array<int,2l> v452;
                                int v454;
                                v454 = 0l;
                                while (while_method_0(v454)){
                                    int v456;
                                    v456 = v414[v454];
                                    int v458;
                                    v458 = v436[v454];
                                    int v460;
                                    v460 = v456 - v458;
                                    v452[v454] = v460;
                                    v454 += 1l ;
                                }
                                int v461;
                                v461 = v114[v118];
                                bool v463;
                                v463 = v461 < v426;
                                float v464;
                                if (v463){
                                    v464 = 1.0f;
                                } else {
                                    v464 = 0.0f;
                                }
                                int v465;
                                v465 = v447 / 3l;
                                bool v466;
                                v466 = v112 <= v465;
                                bool v470;
                                if (v466){
                                    int v467;
                                    v467 = v452[v118];
                                    bool v469;
                                    v469 = v465 < v467;
                                    v470 = v469;
                                } else {
                                    v470 = false;
                                }
                                float v471;
                                if (v470){
                                    v471 = 1.0f;
                                } else {
                                    v471 = 0.0f;
                                }
                                int v472;
                                v472 = v447 / 2l;
                                bool v473;
                                v473 = v112 <= v472;
                                bool v477;
                                if (v473){
                                    int v474;
                                    v474 = v452[v118];
                                    bool v476;
                                    v476 = v472 < v474;
                                    v477 = v476;
                                } else {
                                    v477 = false;
                                }
                                float v478;
                                if (v477){
                                    v478 = 1.0f;
                                } else {
                                    v478 = 0.0f;
                                }
                                bool v479;
                                v479 = v112 <= v447;
                                bool v483;
                                if (v479){
                                    int v480;
                                    v480 = v452[v118];
                                    bool v482;
                                    v482 = v447 < v480;
                                    v483 = v482;
                                } else {
                                    v483 = false;
                                }
                                float v484;
                                if (v483){
                                    v484 = 1.0f;
                                } else {
                                    v484 = 0.0f;
                                }
                                static_array<Tuple19,6l> v485;
                                Union1 v487;
                                v487 = Union1{Union1_2{}};
                                v485[0l] = Tuple19{v487, v464};
                                Union1 v489;
                                v489 = Union1{Union1_1{}};
                                v485[1l] = Tuple19{v489, 4.0f};
                                Union1 v491;
                                v491 = Union1{Union1_3{v465}};
                                v485[2l] = Tuple19{v491, v471};
                                Union1 v493;
                                v493 = Union1{Union1_3{v472}};
                                v485[3l] = Tuple19{v493, v478};
                                Union1 v495;
                                v495 = Union1{Union1_3{v447}};
                                v485[4l] = Tuple19{v495, v484};
                                Union1 v497;
                                v497 = Union1{Union1_0{}};
                                v485[5l] = Tuple19{v497, 1.0f};
                                Union1 v499;
                                v499 = sample_discrete_50(v485, v0);
                                Union6 v500;
                                v500 = Union6{Union6_2{v118, v499}};
                                v11.push(v500);
                                Union4 v688;
                                switch (v499.tag) {
                                    case 0: { // A_All_In
                                        static_array<int,2l> v618;
                                        int v620;
                                        v620 = 0l;
                                        while (while_method_0(v620)){
                                            int v622;
                                            v622 = v116[v620];
                                            int v624;
                                            v624 = v114[v620];
                                            int v626;
                                            v626 = v622 + v624;
                                            v618[v620] = v626;
                                            v620 += 1l ;
                                        }
                                        int v627;
                                        v627 = v114[0l];
                                        int v629; int v630;
                                        Tuple4 tmp54 = Tuple4{1l, v627};
                                        v629 = tmp54.v0; v630 = tmp54.v1;
                                        while (while_method_0(v629)){
                                            int v632;
                                            v632 = v114[v629];
                                            bool v634;
                                            v634 = v630 >= v632;
                                            int v635;
                                            if (v634){
                                                v635 = v630;
                                            } else {
                                                v635 = v632;
                                            }
                                            v630 = v635;
                                            v629 += 1l ;
                                        }
                                        int v636;
                                        v636 = v618[v118];
                                        bool v638;
                                        v638 = v630 < v636;
                                        int v639;
                                        if (v638){
                                            v639 = v630;
                                        } else {
                                            v639 = v636;
                                        }
                                        static_array<int,2l> v640;
                                        int v642;
                                        v642 = 0l;
                                        while (while_method_0(v642)){
                                            int v644;
                                            v644 = v114[v642];
                                            bool v646;
                                            v646 = v118 == v642;
                                            int v647;
                                            if (v646){
                                                v647 = v639;
                                            } else {
                                                v647 = v644;
                                            }
                                            v640[v642] = v647;
                                            v642 += 1l ;
                                        }
                                        static_array<int,2l> v648;
                                        int v650;
                                        v650 = 0l;
                                        while (while_method_0(v650)){
                                            int v652;
                                            v652 = v618[v650];
                                            int v654;
                                            v654 = v640[v650];
                                            int v656;
                                            v656 = v652 - v654;
                                            v648[v650] = v656;
                                            v650 += 1l ;
                                        }
                                        int v657;
                                        v657 = v648[v118];
                                        int v659;
                                        v659 = v630 + v657;
                                        int v660;
                                        v660 = v618[v118];
                                        bool v662;
                                        v662 = v659 < v660;
                                        int v663;
                                        if (v662){
                                            v663 = v659;
                                        } else {
                                            v663 = v660;
                                        }
                                        static_array<int,2l> v664;
                                        int v666;
                                        v666 = 0l;
                                        while (while_method_0(v666)){
                                            int v668;
                                            v668 = v114[v666];
                                            bool v670;
                                            v670 = v118 == v666;
                                            int v671;
                                            if (v670){
                                                v671 = v663;
                                            } else {
                                                v671 = v668;
                                            }
                                            v664[v666] = v671;
                                            v666 += 1l ;
                                        }
                                        static_array<int,2l> v672;
                                        int v674;
                                        v674 = 0l;
                                        while (while_method_0(v674)){
                                            int v676;
                                            v676 = v618[v674];
                                            int v678;
                                            v678 = v664[v674];
                                            int v680;
                                            v680 = v676 - v678;
                                            v672[v674] = v680;
                                            v674 += 1l ;
                                        }
                                        bool v681;
                                        v681 = v657 >= v112;
                                        int v682;
                                        if (v681){
                                            v682 = v657;
                                        } else {
                                            v682 = v112;
                                        }
                                        int v683;
                                        v683 = v115 + 1l;
                                        v688 = try_round_36(v682, v113, v664, v683, v672, v117);
                                        break;
                                    }
                                    case 1: { // A_Call
                                        static_array<int,2l> v502;
                                        int v504;
                                        v504 = 0l;
                                        while (while_method_0(v504)){
                                            int v506;
                                            v506 = v116[v504];
                                            int v508;
                                            v508 = v114[v504];
                                            int v510;
                                            v510 = v506 + v508;
                                            v502[v504] = v510;
                                            v504 += 1l ;
                                        }
                                        int v511;
                                        v511 = v114[0l];
                                        int v513; int v514;
                                        Tuple4 tmp55 = Tuple4{1l, v511};
                                        v513 = tmp55.v0; v514 = tmp55.v1;
                                        while (while_method_0(v513)){
                                            int v516;
                                            v516 = v114[v513];
                                            bool v518;
                                            v518 = v514 >= v516;
                                            int v519;
                                            if (v518){
                                                v519 = v514;
                                            } else {
                                                v519 = v516;
                                            }
                                            v514 = v519;
                                            v513 += 1l ;
                                        }
                                        int v520;
                                        v520 = v502[v118];
                                        bool v522;
                                        v522 = v514 < v520;
                                        int v523;
                                        if (v522){
                                            v523 = v514;
                                        } else {
                                            v523 = v520;
                                        }
                                        static_array<int,2l> v524;
                                        int v526;
                                        v526 = 0l;
                                        while (while_method_0(v526)){
                                            int v528;
                                            v528 = v114[v526];
                                            bool v530;
                                            v530 = v118 == v526;
                                            int v531;
                                            if (v530){
                                                v531 = v523;
                                            } else {
                                                v531 = v528;
                                            }
                                            v524[v526] = v531;
                                            v526 += 1l ;
                                        }
                                        static_array<int,2l> v532;
                                        int v534;
                                        v534 = 0l;
                                        while (while_method_0(v534)){
                                            int v536;
                                            v536 = v502[v534];
                                            int v538;
                                            v538 = v524[v534];
                                            int v540;
                                            v540 = v536 - v538;
                                            v532[v534] = v540;
                                            v534 += 1l ;
                                        }
                                        bool v541;
                                        v541 = v118 < 2l;
                                        if (v541){
                                            int v542;
                                            v542 = v115 + 1l;
                                            v688 = try_round_36(v112, v113, v524, v542, v532, v117);
                                        } else {
                                            v688 = go_next_street_38(v112, v113, v524, v115, v532, v117);
                                        }
                                        break;
                                    }
                                    case 2: { // A_Fold
                                        v688 = Union4{Union4_1{v112, v113, v114, v115, v116, v117}};
                                        break;
                                    }
                                    case 3: { // A_Raise
                                        int v546 = v499.case3.v0;
                                        bool v547;
                                        v547 = v112 <= v546;
                                        bool v548;
                                        v548 = v547 == false;
                                        if (v548){
                                            assert("The raise amount must match the minimum." && v547);
                                        } else {
                                        }
                                        static_array<int,2l> v550;
                                        int v552;
                                        v552 = 0l;
                                        while (while_method_0(v552)){
                                            int v554;
                                            v554 = v116[v552];
                                            int v556;
                                            v556 = v114[v552];
                                            int v558;
                                            v558 = v554 + v556;
                                            v550[v552] = v558;
                                            v552 += 1l ;
                                        }
                                        int v559;
                                        v559 = v114[0l];
                                        int v561; int v562;
                                        Tuple4 tmp56 = Tuple4{1l, v559};
                                        v561 = tmp56.v0; v562 = tmp56.v1;
                                        while (while_method_0(v561)){
                                            int v564;
                                            v564 = v114[v561];
                                            bool v566;
                                            v566 = v562 >= v564;
                                            int v567;
                                            if (v566){
                                                v567 = v562;
                                            } else {
                                                v567 = v564;
                                            }
                                            v562 = v567;
                                            v561 += 1l ;
                                        }
                                        int v568;
                                        v568 = v550[v118];
                                        bool v570;
                                        v570 = v562 < v568;
                                        int v571;
                                        if (v570){
                                            v571 = v562;
                                        } else {
                                            v571 = v568;
                                        }
                                        static_array<int,2l> v572;
                                        int v574;
                                        v574 = 0l;
                                        while (while_method_0(v574)){
                                            int v576;
                                            v576 = v114[v574];
                                            bool v578;
                                            v578 = v118 == v574;
                                            int v579;
                                            if (v578){
                                                v579 = v571;
                                            } else {
                                                v579 = v576;
                                            }
                                            v572[v574] = v579;
                                            v574 += 1l ;
                                        }
                                        static_array<int,2l> v580;
                                        int v582;
                                        v582 = 0l;
                                        while (while_method_0(v582)){
                                            int v584;
                                            v584 = v550[v582];
                                            int v586;
                                            v586 = v572[v582];
                                            int v588;
                                            v588 = v584 - v586;
                                            v580[v582] = v588;
                                            v582 += 1l ;
                                        }
                                        int v589;
                                        v589 = v580[v118];
                                        bool v591;
                                        v591 = v546 < v589;
                                        bool v592;
                                        v592 = v591 == false;
                                        if (v592){
                                            assert("The raise amount must be less than the stack size after calling." && v591);
                                        } else {
                                        }
                                        int v594;
                                        v594 = v562 + v546;
                                        int v595;
                                        v595 = v550[v118];
                                        bool v597;
                                        v597 = v594 < v595;
                                        int v598;
                                        if (v597){
                                            v598 = v594;
                                        } else {
                                            v598 = v595;
                                        }
                                        static_array<int,2l> v599;
                                        int v601;
                                        v601 = 0l;
                                        while (while_method_0(v601)){
                                            int v603;
                                            v603 = v114[v601];
                                            bool v605;
                                            v605 = v118 == v601;
                                            int v606;
                                            if (v605){
                                                v606 = v598;
                                            } else {
                                                v606 = v603;
                                            }
                                            v599[v601] = v606;
                                            v601 += 1l ;
                                        }
                                        static_array<int,2l> v607;
                                        int v609;
                                        v609 = 0l;
                                        while (while_method_0(v609)){
                                            int v611;
                                            v611 = v550[v609];
                                            int v613;
                                            v613 = v599[v609];
                                            int v615;
                                            v615 = v611 - v613;
                                            v607[v609] = v615;
                                            v609 += 1l ;
                                        }
                                        int v616;
                                        v616 = v115 + 1l;
                                        v688 = try_round_36(v546, v113, v599, v616, v607, v117);
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false); __trap();
                                    }
                                }
                                v998 = Union3{Union3_1{v688}};
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        break;
                    }
                    case 5: { // G_Round'
                        int v693 = v16.case5.v0; static_array<static_array<unsigned char,2l>,2l> v694 = v16.case5.v1; static_array<int,2l> v695 = v16.case5.v2; int v696 = v16.case5.v3; static_array<int,2l> v697 = v16.case5.v4; Union5 v698 = v16.case5.v5; Union1 v699 = v16.case5.v6;
                        int v700;
                        v700 = v696 % 2l;
                        Union6 v701;
                        v701 = Union6{Union6_2{v700, v699}};
                        v11.push(v701);
                        Union4 v889;
                        switch (v699.tag) {
                            case 0: { // A_All_In
                                static_array<int,2l> v819;
                                int v821;
                                v821 = 0l;
                                while (while_method_0(v821)){
                                    int v823;
                                    v823 = v697[v821];
                                    int v825;
                                    v825 = v695[v821];
                                    int v827;
                                    v827 = v823 + v825;
                                    v819[v821] = v827;
                                    v821 += 1l ;
                                }
                                int v828;
                                v828 = v695[0l];
                                int v830; int v831;
                                Tuple4 tmp57 = Tuple4{1l, v828};
                                v830 = tmp57.v0; v831 = tmp57.v1;
                                while (while_method_0(v830)){
                                    int v833;
                                    v833 = v695[v830];
                                    bool v835;
                                    v835 = v831 >= v833;
                                    int v836;
                                    if (v835){
                                        v836 = v831;
                                    } else {
                                        v836 = v833;
                                    }
                                    v831 = v836;
                                    v830 += 1l ;
                                }
                                int v837;
                                v837 = v819[v700];
                                bool v839;
                                v839 = v831 < v837;
                                int v840;
                                if (v839){
                                    v840 = v831;
                                } else {
                                    v840 = v837;
                                }
                                static_array<int,2l> v841;
                                int v843;
                                v843 = 0l;
                                while (while_method_0(v843)){
                                    int v845;
                                    v845 = v695[v843];
                                    bool v847;
                                    v847 = v700 == v843;
                                    int v848;
                                    if (v847){
                                        v848 = v840;
                                    } else {
                                        v848 = v845;
                                    }
                                    v841[v843] = v848;
                                    v843 += 1l ;
                                }
                                static_array<int,2l> v849;
                                int v851;
                                v851 = 0l;
                                while (while_method_0(v851)){
                                    int v853;
                                    v853 = v819[v851];
                                    int v855;
                                    v855 = v841[v851];
                                    int v857;
                                    v857 = v853 - v855;
                                    v849[v851] = v857;
                                    v851 += 1l ;
                                }
                                int v858;
                                v858 = v849[v700];
                                int v860;
                                v860 = v831 + v858;
                                int v861;
                                v861 = v819[v700];
                                bool v863;
                                v863 = v860 < v861;
                                int v864;
                                if (v863){
                                    v864 = v860;
                                } else {
                                    v864 = v861;
                                }
                                static_array<int,2l> v865;
                                int v867;
                                v867 = 0l;
                                while (while_method_0(v867)){
                                    int v869;
                                    v869 = v695[v867];
                                    bool v871;
                                    v871 = v700 == v867;
                                    int v872;
                                    if (v871){
                                        v872 = v864;
                                    } else {
                                        v872 = v869;
                                    }
                                    v865[v867] = v872;
                                    v867 += 1l ;
                                }
                                static_array<int,2l> v873;
                                int v875;
                                v875 = 0l;
                                while (while_method_0(v875)){
                                    int v877;
                                    v877 = v819[v875];
                                    int v879;
                                    v879 = v865[v875];
                                    int v881;
                                    v881 = v877 - v879;
                                    v873[v875] = v881;
                                    v875 += 1l ;
                                }
                                bool v882;
                                v882 = v858 >= v693;
                                int v883;
                                if (v882){
                                    v883 = v858;
                                } else {
                                    v883 = v693;
                                }
                                int v884;
                                v884 = v696 + 1l;
                                v889 = try_round_36(v883, v694, v865, v884, v873, v698);
                                break;
                            }
                            case 1: { // A_Call
                                static_array<int,2l> v703;
                                int v705;
                                v705 = 0l;
                                while (while_method_0(v705)){
                                    int v707;
                                    v707 = v697[v705];
                                    int v709;
                                    v709 = v695[v705];
                                    int v711;
                                    v711 = v707 + v709;
                                    v703[v705] = v711;
                                    v705 += 1l ;
                                }
                                int v712;
                                v712 = v695[0l];
                                int v714; int v715;
                                Tuple4 tmp58 = Tuple4{1l, v712};
                                v714 = tmp58.v0; v715 = tmp58.v1;
                                while (while_method_0(v714)){
                                    int v717;
                                    v717 = v695[v714];
                                    bool v719;
                                    v719 = v715 >= v717;
                                    int v720;
                                    if (v719){
                                        v720 = v715;
                                    } else {
                                        v720 = v717;
                                    }
                                    v715 = v720;
                                    v714 += 1l ;
                                }
                                int v721;
                                v721 = v703[v700];
                                bool v723;
                                v723 = v715 < v721;
                                int v724;
                                if (v723){
                                    v724 = v715;
                                } else {
                                    v724 = v721;
                                }
                                static_array<int,2l> v725;
                                int v727;
                                v727 = 0l;
                                while (while_method_0(v727)){
                                    int v729;
                                    v729 = v695[v727];
                                    bool v731;
                                    v731 = v700 == v727;
                                    int v732;
                                    if (v731){
                                        v732 = v724;
                                    } else {
                                        v732 = v729;
                                    }
                                    v725[v727] = v732;
                                    v727 += 1l ;
                                }
                                static_array<int,2l> v733;
                                int v735;
                                v735 = 0l;
                                while (while_method_0(v735)){
                                    int v737;
                                    v737 = v703[v735];
                                    int v739;
                                    v739 = v725[v735];
                                    int v741;
                                    v741 = v737 - v739;
                                    v733[v735] = v741;
                                    v735 += 1l ;
                                }
                                bool v742;
                                v742 = v700 < 2l;
                                if (v742){
                                    int v743;
                                    v743 = v696 + 1l;
                                    v889 = try_round_36(v693, v694, v725, v743, v733, v698);
                                } else {
                                    v889 = go_next_street_38(v693, v694, v725, v696, v733, v698);
                                }
                                break;
                            }
                            case 2: { // A_Fold
                                v889 = Union4{Union4_1{v693, v694, v695, v696, v697, v698}};
                                break;
                            }
                            case 3: { // A_Raise
                                int v747 = v699.case3.v0;
                                bool v748;
                                v748 = v693 <= v747;
                                bool v749;
                                v749 = v748 == false;
                                if (v749){
                                    assert("The raise amount must match the minimum." && v748);
                                } else {
                                }
                                static_array<int,2l> v751;
                                int v753;
                                v753 = 0l;
                                while (while_method_0(v753)){
                                    int v755;
                                    v755 = v697[v753];
                                    int v757;
                                    v757 = v695[v753];
                                    int v759;
                                    v759 = v755 + v757;
                                    v751[v753] = v759;
                                    v753 += 1l ;
                                }
                                int v760;
                                v760 = v695[0l];
                                int v762; int v763;
                                Tuple4 tmp59 = Tuple4{1l, v760};
                                v762 = tmp59.v0; v763 = tmp59.v1;
                                while (while_method_0(v762)){
                                    int v765;
                                    v765 = v695[v762];
                                    bool v767;
                                    v767 = v763 >= v765;
                                    int v768;
                                    if (v767){
                                        v768 = v763;
                                    } else {
                                        v768 = v765;
                                    }
                                    v763 = v768;
                                    v762 += 1l ;
                                }
                                int v769;
                                v769 = v751[v700];
                                bool v771;
                                v771 = v763 < v769;
                                int v772;
                                if (v771){
                                    v772 = v763;
                                } else {
                                    v772 = v769;
                                }
                                static_array<int,2l> v773;
                                int v775;
                                v775 = 0l;
                                while (while_method_0(v775)){
                                    int v777;
                                    v777 = v695[v775];
                                    bool v779;
                                    v779 = v700 == v775;
                                    int v780;
                                    if (v779){
                                        v780 = v772;
                                    } else {
                                        v780 = v777;
                                    }
                                    v773[v775] = v780;
                                    v775 += 1l ;
                                }
                                static_array<int,2l> v781;
                                int v783;
                                v783 = 0l;
                                while (while_method_0(v783)){
                                    int v785;
                                    v785 = v751[v783];
                                    int v787;
                                    v787 = v773[v783];
                                    int v789;
                                    v789 = v785 - v787;
                                    v781[v783] = v789;
                                    v783 += 1l ;
                                }
                                int v790;
                                v790 = v781[v700];
                                bool v792;
                                v792 = v747 < v790;
                                bool v793;
                                v793 = v792 == false;
                                if (v793){
                                    assert("The raise amount must be less than the stack size after calling." && v792);
                                } else {
                                }
                                int v795;
                                v795 = v763 + v747;
                                int v796;
                                v796 = v751[v700];
                                bool v798;
                                v798 = v795 < v796;
                                int v799;
                                if (v798){
                                    v799 = v795;
                                } else {
                                    v799 = v796;
                                }
                                static_array<int,2l> v800;
                                int v802;
                                v802 = 0l;
                                while (while_method_0(v802)){
                                    int v804;
                                    v804 = v695[v802];
                                    bool v806;
                                    v806 = v700 == v802;
                                    int v807;
                                    if (v806){
                                        v807 = v799;
                                    } else {
                                        v807 = v804;
                                    }
                                    v800[v802] = v807;
                                    v802 += 1l ;
                                }
                                static_array<int,2l> v808;
                                int v810;
                                v810 = 0l;
                                while (while_method_0(v810)){
                                    int v812;
                                    v812 = v751[v810];
                                    int v814;
                                    v814 = v800[v810];
                                    int v816;
                                    v816 = v812 - v814;
                                    v808[v810] = v816;
                                    v810 += 1l ;
                                }
                                int v817;
                                v817 = v696 + 1l;
                                v889 = try_round_36(v747, v694, v800, v817, v808, v698);
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        v998 = Union3{Union3_1{v889}};
                        break;
                    }
                    case 6: { // G_Showdown
                        int v32 = v16.case6.v0; static_array<static_array<unsigned char,2l>,2l> v33 = v16.case6.v1; static_array<int,2l> v34 = v16.case6.v2; int v35 = v16.case6.v3; static_array<int,2l> v36 = v16.case6.v4; Union5 v37 = v16.case6.v5;
                        static_array<unsigned char,5l> v40;
                        switch (v37.tag) {
                            case 2: { // River
                                static_array<unsigned char,5l> v38 = v37.case2.v0;
                                v40 = v38;
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in showdown.");
                                __trap();
                            }
                        }
                        static_array<unsigned char,2l> v41;
                        v41 = v33[0l];
                        static_array<unsigned char,7l> v43;
                        int v45;
                        v45 = 0l;
                        while (while_method_0(v45)){
                            unsigned char v47;
                            v47 = v41[v45];
                            v43[v45] = v47;
                            v45 += 1l ;
                        }
                        int v49;
                        v49 = 0l;
                        while (while_method_2(v49)){
                            unsigned char v51;
                            v51 = v40[v49];
                            int v53;
                            v53 = 2l + v49;
                            v43[v53] = v51;
                            v49 += 1l ;
                        }
                        static_array<unsigned char,5l> v54; char v55;
                        Tuple0 tmp84 = score_54(v43);
                        v54 = tmp84.v0; v55 = tmp84.v1;
                        static_array<unsigned char,2l> v56;
                        v56 = v33[1l];
                        static_array<unsigned char,7l> v58;
                        int v60;
                        v60 = 0l;
                        while (while_method_0(v60)){
                            unsigned char v62;
                            v62 = v56[v60];
                            v58[v60] = v62;
                            v60 += 1l ;
                        }
                        int v64;
                        v64 = 0l;
                        while (while_method_2(v64)){
                            unsigned char v66;
                            v66 = v40[v64];
                            int v68;
                            v68 = 2l + v64;
                            v58[v68] = v66;
                            v64 += 1l ;
                        }
                        static_array<unsigned char,5l> v69; char v70;
                        Tuple0 tmp85 = score_54(v58);
                        v69 = tmp85.v0; v70 = tmp85.v1;
                        int v71;
                        v71 = v35 % 2l;
                        int v72;
                        v72 = v34[v71];
                        bool v74;
                        v74 = v55 < v70;
                        Union11 v80;
                        if (v74){
                            v80 = Union11{Union11_2{}};
                        } else {
                            bool v76;
                            v76 = v55 > v70;
                            if (v76){
                                v80 = Union11{Union11_1{}};
                            } else {
                                v80 = Union11{Union11_0{}};
                            }
                        }
                        Union11 v99;
                        switch (v80.tag) {
                            case 0: { // Eq
                                Union11 v81;
                                v81 = Union11{Union11_0{}};
                                int v82;
                                v82 = 0l;
                                while (while_method_2(v82)){
                                    unsigned char v84;
                                    v84 = v54[v82];
                                    unsigned char v86;
                                    v86 = v69[v82];
                                    unsigned char v88;
                                    v88 = v84 / 4u;
                                    unsigned char v89;
                                    v89 = v86 / 4u;
                                    bool v90;
                                    v90 = v88 < v89;
                                    Union11 v96;
                                    if (v90){
                                        v96 = Union11{Union11_2{}};
                                    } else {
                                        bool v92;
                                        v92 = v88 > v89;
                                        if (v92){
                                            v96 = Union11{Union11_1{}};
                                        } else {
                                            v96 = Union11{Union11_0{}};
                                        }
                                    }
                                    bool v97;
                                    switch (v96.tag) {
                                        case 0: { // Eq
                                            v97 = true;
                                            break;
                                        }
                                        default: {
                                            v97 = false;
                                        }
                                    }
                                    bool v98;
                                    v98 = v97 == false;
                                    if (v98){
                                        v81 = v96;
                                        break;
                                    } else {
                                    }
                                    v82 += 1l ;
                                }
                                v99 = v81;
                                break;
                            }
                            default: {
                                v99 = v80;
                            }
                        }
                        int v104; int v105;
                        switch (v99.tag) {
                            case 0: { // Eq
                                v104 = 0l; v105 = -1l;
                                break;
                            }
                            case 1: { // Gt
                                v104 = v72; v105 = 0l;
                                break;
                            }
                            case 2: { // Lt
                                v104 = v72; v105 = 1l;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false); __trap();
                            }
                        }
                        static_array<Tuple0,2l> v106;
                        v106[0l] = Tuple0{v54, v55};
                        v106[1l] = Tuple0{v69, v70};
                        Union6 v108;
                        v108 = Union6{Union6_4{v104, v106, v105}};
                        v11.push(v108);
                        Union7 v109;
                        v109 = Union7{Union7_1{v32, v33, v34, v35, v36, v37}};
                        v9 = v109;
                        Union3 v110;
                        v110 = Union3{Union3_0{}};
                        v6 = v110;
                        v998 = Union3{Union3_0{}};
                        break;
                    }
                    case 7: { // G_Turn
                        int v908 = v16.case7.v0; static_array<static_array<unsigned char,2l>,2l> v909 = v16.case7.v1; static_array<int,2l> v910 = v16.case7.v2; int v911 = v16.case7.v3; static_array<int,2l> v912 = v16.case7.v4; Union5 v913 = v16.case7.v5;
                        static_array<unsigned char,1l> v914; unsigned long long v915;
                        Tuple12 tmp86 = draw_cards_40(v0, v12);
                        v914 = tmp86.v0; v915 = tmp86.v1;
                        v5 = v915;
                        static_array_list<unsigned char,5l> v916;
                        v916 = get_community_cards_41(v913, v914);
                        Union6 v917;
                        v917 = Union6{Union6_0{v916}};
                        v11.push(v917);
                        Union5 v932;
                        switch (v913.tag) {
                            case 0: { // Flop
                                static_array<unsigned char,3l> v918 = v913.case0.v0;
                                static_array<unsigned char,4l> v919;
                                int v921;
                                v921 = 0l;
                                while (while_method_1(v921)){
                                    unsigned char v923;
                                    v923 = v918[v921];
                                    v919[v921] = v923;
                                    v921 += 1l ;
                                }
                                int v925;
                                v925 = 0l;
                                while (while_method_6(v925)){
                                    unsigned char v927;
                                    v927 = v914[v925];
                                    int v929;
                                    v929 = 3l + v925;
                                    v919[v929] = v927;
                                    v925 += 1l ;
                                }
                                v932 = Union5{Union5_3{v919}};
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in turn.");
                                __trap();
                            }
                        }
                        int v933;
                        v933 = 2l;
                        int v934;
                        v934 = 0l;
                        Union4 v935;
                        v935 = try_round_36(v933, v909, v910, v934, v912, v932);
                        v998 = Union3{Union3_1{v935}};
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
        v14 = v998;
    }
    return ;
}
__device__ void f_56(unsigned char * v0, unsigned long long v1){
    unsigned long long * v2;
    v2 = (unsigned long long *)(v0+0ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_57(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+8ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_58(unsigned char * v0){
    return ;
}
__device__ void f_60(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+0ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_64(unsigned char * v0, unsigned char v1){
    unsigned char * v2;
    v2 = (unsigned char *)(v0+0ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_63(unsigned char * v0, unsigned char v1){
    return f_64(v0, v1);
}
__device__ void f_62(unsigned char * v0, static_array<unsigned char,2l> v1){
    int v2;
    v2 = 0l;
    while (while_method_0(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned char * v5;
        v5 = (unsigned char *)(v0+v4);
        unsigned char v7;
        v7 = v1[v2];
        f_63(v5, v7);
        v2 += 1l ;
    }
    return ;
}
__device__ void f_65(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+28ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_66(unsigned char * v0, static_array<unsigned char,3l> v1){
    int v2;
    v2 = 0l;
    while (while_method_1(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned char * v5;
        v5 = (unsigned char *)(v0+v4);
        unsigned char v7;
        v7 = v1[v2];
        f_63(v5, v7);
        v2 += 1l ;
    }
    return ;
}
__device__ void f_67(unsigned char * v0, static_array<unsigned char,5l> v1){
    int v2;
    v2 = 0l;
    while (while_method_2(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned char * v5;
        v5 = (unsigned char *)(v0+v4);
        unsigned char v7;
        v7 = v1[v2];
        f_63(v5, v7);
        v2 += 1l ;
    }
    return ;
}
__device__ void f_68(unsigned char * v0, static_array<unsigned char,4l> v1){
    int v2;
    v2 = 0l;
    while (while_method_3(v2)){
        unsigned long long v4;
        v4 = (unsigned long long)v2;
        unsigned char * v5;
        v5 = (unsigned char *)(v0+v4);
        unsigned char v7;
        v7 = v1[v2];
        f_63(v5, v7);
        v2 += 1l ;
    }
    return ;
}
__device__ void f_61(unsigned char * v0, int v1, static_array<static_array<unsigned char,2l>,2l> v2, static_array<int,2l> v3, int v4, static_array<int,2l> v5, Union5 v6){
    int * v7;
    v7 = (int *)(v0+0ull);
    v7[0l] = v1;
    int v9;
    v9 = 0l;
    while (while_method_0(v9)){
        unsigned long long v11;
        v11 = (unsigned long long)v9;
        unsigned long long v12;
        v12 = v11 * 2ull;
        unsigned long long v13;
        v13 = 4ull + v12;
        unsigned char * v14;
        v14 = (unsigned char *)(v0+v13);
        static_array<unsigned char,2l> v16;
        v16 = v2[v9];
        f_62(v14, v16);
        v9 += 1l ;
    }
    int v18;
    v18 = 0l;
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
        f_60(v23, v25);
        v18 += 1l ;
    }
    int * v27;
    v27 = (int *)(v0+16ull);
    v27[0l] = v4;
    int v29;
    v29 = 0l;
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
        f_60(v34, v36);
        v29 += 1l ;
    }
    int v38;
    v38 = v6.tag;
    f_65(v0, v38);
    unsigned char * v39;
    v39 = (unsigned char *)(v0+32ull);
    switch (v6.tag) {
        case 0: { // Flop
            static_array<unsigned char,3l> v41 = v6.case0.v0;
            return f_66(v39, v41);
            break;
        }
        case 1: { // Preflop
            return f_58(v39);
            break;
        }
        case 2: { // River
            static_array<unsigned char,5l> v42 = v6.case2.v0;
            return f_67(v39, v42);
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4l> v43 = v6.case3.v0;
            return f_68(v39, v43);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_70(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+40ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_69(unsigned char * v0, int v1, static_array<static_array<unsigned char,2l>,2l> v2, static_array<int,2l> v3, int v4, static_array<int,2l> v5, Union5 v6, Union1 v7){
    int * v8;
    v8 = (int *)(v0+0ull);
    v8[0l] = v1;
    int v10;
    v10 = 0l;
    while (while_method_0(v10)){
        unsigned long long v12;
        v12 = (unsigned long long)v10;
        unsigned long long v13;
        v13 = v12 * 2ull;
        unsigned long long v14;
        v14 = 4ull + v13;
        unsigned char * v15;
        v15 = (unsigned char *)(v0+v14);
        static_array<unsigned char,2l> v17;
        v17 = v2[v10];
        f_62(v15, v17);
        v10 += 1l ;
    }
    int v19;
    v19 = 0l;
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
        f_60(v24, v26);
        v19 += 1l ;
    }
    int * v28;
    v28 = (int *)(v0+16ull);
    v28[0l] = v4;
    int v30;
    v30 = 0l;
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
        f_60(v35, v37);
        v30 += 1l ;
    }
    int v39;
    v39 = v6.tag;
    f_65(v0, v39);
    unsigned char * v40;
    v40 = (unsigned char *)(v0+32ull);
    switch (v6.tag) {
        case 0: { // Flop
            static_array<unsigned char,3l> v42 = v6.case0.v0;
            f_66(v40, v42);
            break;
        }
        case 1: { // Preflop
            f_58(v40);
            break;
        }
        case 2: { // River
            static_array<unsigned char,5l> v43 = v6.case2.v0;
            f_67(v40, v43);
            break;
        }
        case 3: { // Turn
            static_array<unsigned char,4l> v44 = v6.case3.v0;
            f_68(v40, v44);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v45;
    v45 = v7.tag;
    f_70(v0, v45);
    unsigned char * v46;
    v46 = (unsigned char *)(v0+44ull);
    switch (v7.tag) {
        case 0: { // A_All_In
            return f_58(v46);
            break;
        }
        case 1: { // A_Call
            return f_58(v46);
            break;
        }
        case 2: { // A_Fold
            return f_58(v46);
            break;
        }
        case 3: { // A_Raise
            int v48 = v7.case3.v0;
            return f_60(v46, v48);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_59(unsigned char * v0, Union4 v1){
    int v2;
    v2 = v1.tag;
    f_60(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // G_Flop
            int v5 = v1.case0.v0; static_array<static_array<unsigned char,2l>,2l> v6 = v1.case0.v1; static_array<int,2l> v7 = v1.case0.v2; int v8 = v1.case0.v3; static_array<int,2l> v9 = v1.case0.v4; Union5 v10 = v1.case0.v5;
            return f_61(v3, v5, v6, v7, v8, v9, v10);
            break;
        }
        case 1: { // G_Fold
            int v11 = v1.case1.v0; static_array<static_array<unsigned char,2l>,2l> v12 = v1.case1.v1; static_array<int,2l> v13 = v1.case1.v2; int v14 = v1.case1.v3; static_array<int,2l> v15 = v1.case1.v4; Union5 v16 = v1.case1.v5;
            return f_61(v3, v11, v12, v13, v14, v15, v16);
            break;
        }
        case 2: { // G_Preflop
            return f_58(v3);
            break;
        }
        case 3: { // G_River
            int v17 = v1.case3.v0; static_array<static_array<unsigned char,2l>,2l> v18 = v1.case3.v1; static_array<int,2l> v19 = v1.case3.v2; int v20 = v1.case3.v3; static_array<int,2l> v21 = v1.case3.v4; Union5 v22 = v1.case3.v5;
            return f_61(v3, v17, v18, v19, v20, v21, v22);
            break;
        }
        case 4: { // G_Round
            int v23 = v1.case4.v0; static_array<static_array<unsigned char,2l>,2l> v24 = v1.case4.v1; static_array<int,2l> v25 = v1.case4.v2; int v26 = v1.case4.v3; static_array<int,2l> v27 = v1.case4.v4; Union5 v28 = v1.case4.v5;
            return f_61(v3, v23, v24, v25, v26, v27, v28);
            break;
        }
        case 5: { // G_Round'
            int v29 = v1.case5.v0; static_array<static_array<unsigned char,2l>,2l> v30 = v1.case5.v1; static_array<int,2l> v31 = v1.case5.v2; int v32 = v1.case5.v3; static_array<int,2l> v33 = v1.case5.v4; Union5 v34 = v1.case5.v5; Union1 v35 = v1.case5.v6;
            return f_69(v3, v29, v30, v31, v32, v33, v34, v35);
            break;
        }
        case 6: { // G_Showdown
            int v36 = v1.case6.v0; static_array<static_array<unsigned char,2l>,2l> v37 = v1.case6.v1; static_array<int,2l> v38 = v1.case6.v2; int v39 = v1.case6.v3; static_array<int,2l> v40 = v1.case6.v4; Union5 v41 = v1.case6.v5;
            return f_61(v3, v36, v37, v38, v39, v40, v41);
            break;
        }
        case 7: { // G_Turn
            int v42 = v1.case7.v0; static_array<static_array<unsigned char,2l>,2l> v43 = v1.case7.v1; static_array<int,2l> v44 = v1.case7.v2; int v45 = v1.case7.v3; static_array<int,2l> v46 = v1.case7.v4; Union5 v47 = v1.case7.v5;
            return f_61(v3, v42, v43, v44, v45, v46, v47);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_71(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+80ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_73(unsigned char * v0, static_array_list<unsigned char,5l> v1){
    int v2;
    v2 = v1.length;
    f_60(v0, v2);
    int v3;
    v3 = v1.length;
    int v4;
    v4 = 0l;
    while (while_method_4(v3, v4)){
        unsigned long long v6;
        v6 = (unsigned long long)v4;
        unsigned long long v7;
        v7 = 4ull + v6;
        unsigned char * v8;
        v8 = (unsigned char *)(v0+v7);
        unsigned char v10;
        v10 = v1[v4];
        f_63(v8, v10);
        v4 += 1l ;
    }
    return ;
}
__device__ void f_74(unsigned char * v0, int v1, int v2){
    int * v3;
    v3 = (int *)(v0+0ull);
    v3[0l] = v1;
    int * v5;
    v5 = (int *)(v0+4ull);
    v5[0l] = v2;
    return ;
}
__device__ void f_76(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+4ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_75(unsigned char * v0, int v1, Union1 v2){
    int * v3;
    v3 = (int *)(v0+0ull);
    v3[0l] = v1;
    int v5;
    v5 = v2.tag;
    f_76(v0, v5);
    unsigned char * v6;
    v6 = (unsigned char *)(v0+8ull);
    switch (v2.tag) {
        case 0: { // A_All_In
            return f_58(v6);
            break;
        }
        case 1: { // A_Call
            return f_58(v6);
            break;
        }
        case 2: { // A_Fold
            return f_58(v6);
            break;
        }
        case 3: { // A_Raise
            int v8 = v2.case3.v0;
            return f_60(v6, v8);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_77(unsigned char * v0, int v1, static_array<unsigned char,2l> v2){
    int * v3;
    v3 = (int *)(v0+0ull);
    v3[0l] = v1;
    int v5;
    v5 = 0l;
    while (while_method_0(v5)){
        unsigned long long v7;
        v7 = (unsigned long long)v5;
        unsigned long long v8;
        v8 = 4ull + v7;
        unsigned char * v9;
        v9 = (unsigned char *)(v0+v8);
        unsigned char v11;
        v11 = v2[v5];
        f_63(v9, v11);
        v5 += 1l ;
    }
    return ;
}
__device__ void f_80(unsigned char * v0, static_array<unsigned char,5l> v1, char v2){
    int v3;
    v3 = 0l;
    while (while_method_2(v3)){
        unsigned long long v5;
        v5 = (unsigned long long)v3;
        unsigned char * v6;
        v6 = (unsigned char *)(v0+v5);
        unsigned char v8;
        v8 = v1[v3];
        f_63(v6, v8);
        v3 += 1l ;
    }
    char * v10;
    v10 = (char *)(v0+5ull);
    v10[0l] = v2;
    return ;
}
__device__ void f_79(unsigned char * v0, static_array<unsigned char,5l> v1, char v2){
    return f_80(v0, v1, v2);
}
__device__ void f_78(unsigned char * v0, int v1, static_array<Tuple0,2l> v2, int v3){
    int * v4;
    v4 = (int *)(v0+0ull);
    v4[0l] = v1;
    int v6;
    v6 = 0l;
    while (while_method_0(v6)){
        unsigned long long v8;
        v8 = (unsigned long long)v6;
        unsigned long long v9;
        v9 = v8 * 8ull;
        unsigned long long v10;
        v10 = 8ull + v9;
        unsigned char * v11;
        v11 = (unsigned char *)(v0+v10);
        static_array<unsigned char,5l> v13; char v14;
        Tuple0 tmp87 = v2[v6];
        v13 = tmp87.v0; v14 = tmp87.v1;
        f_79(v11, v13, v14);
        v6 += 1l ;
    }
    int * v17;
    v17 = (int *)(v0+24ull);
    v17[0l] = v3;
    return ;
}
__device__ void f_72(unsigned char * v0, Union6 v1){
    int v2;
    v2 = v1.tag;
    f_60(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+16ull);
    switch (v1.tag) {
        case 0: { // CommunityCardsAre
            static_array_list<unsigned char,5l> v5 = v1.case0.v0;
            return f_73(v3, v5);
            break;
        }
        case 1: { // Fold
            int v6 = v1.case1.v0; int v7 = v1.case1.v1;
            return f_74(v3, v6, v7);
            break;
        }
        case 2: { // PlayerAction
            int v8 = v1.case2.v0; Union1 v9 = v1.case2.v1;
            return f_75(v3, v8, v9);
            break;
        }
        case 3: { // PlayerGotCards
            int v10 = v1.case3.v0; static_array<unsigned char,2l> v11 = v1.case3.v1;
            return f_77(v3, v10, v11);
            break;
        }
        case 4: { // Showdown
            int v12 = v1.case4.v0; static_array<Tuple0,2l> v13 = v1.case4.v1; int v14 = v1.case4.v2;
            return f_78(v3, v12, v13, v14);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_81(unsigned char * v0, Union2 v1){
    int v2;
    v2 = v1.tag;
    f_60(v0, v2);
    unsigned char * v3;
    v3 = (unsigned char *)(v0+4ull);
    switch (v1.tag) {
        case 0: { // Computer
            return f_58(v3);
            break;
        }
        case 1: { // Human
            return f_58(v3);
            break;
        }
        case 2: { // Random
            return f_58(v3);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ void f_82(unsigned char * v0, int v1){
    int * v2;
    v2 = (int *)(v0+6248ull);
    v2[0l] = v1;
    return ;
}
__device__ void f_55(unsigned char * v0, unsigned long long v1, Union3 v2, dynamic_array_list<Union6,128l> v3, static_array<Union2,2l> v4, Union7 v5){
    f_56(v0, v1);
    int v6;
    v6 = v2.tag;
    f_57(v0, v6);
    unsigned char * v7;
    v7 = (unsigned char *)(v0+16ull);
    switch (v2.tag) {
        case 0: { // None
            f_58(v7);
            break;
        }
        case 1: { // Some
            Union4 v9 = v2.case1.v0;
            f_59(v7, v9);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
    int v10;
    v10 = v3.length_();
    f_71(v0, v10);
    int v11;
    v11 = v3.length_();
    int v12;
    v12 = 0l;
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
        f_72(v17, v19);
        v12 += 1l ;
    }
    int v21;
    v21 = 0l;
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
        f_81(v26, v28);
        v21 += 1l ;
    }
    int v30;
    v30 = v5.tag;
    f_82(v0, v30);
    unsigned char * v31;
    v31 = (unsigned char *)(v0+6256ull);
    switch (v5.tag) {
        case 0: { // GameNotStarted
            return f_58(v31);
            break;
        }
        case 1: { // GameOver
            int v33 = v5.case1.v0; static_array<static_array<unsigned char,2l>,2l> v34 = v5.case1.v1; static_array<int,2l> v35 = v5.case1.v2; int v36 = v5.case1.v3; static_array<int,2l> v37 = v5.case1.v4; Union5 v38 = v5.case1.v5;
            return f_61(v31, v33, v34, v35, v36, v37, v38);
            break;
        }
        case 2: { // WaitingForActionFromPlayerId
            int v39 = v5.case2.v0; static_array<static_array<unsigned char,2l>,2l> v40 = v5.case2.v1; static_array<int,2l> v41 = v5.case2.v2; int v42 = v5.case2.v3; static_array<int,2l> v43 = v5.case2.v4; Union5 v44 = v5.case2.v5;
            return f_61(v31, v39, v40, v41, v42, v43, v44);
            break;
        }
        default: {
            assert("Invalid tag." && false); __trap();
        }
    }
}
__device__ inline bool while_method_20(){
    return true;
}
extern "C" __global__ void entry0(unsigned char * v0, unsigned char * v1, unsigned char * v2, unsigned long long v3, unsigned char * v4, unsigned long long v5) {
    __shared__ int v6;
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v7;
    v7 = threadIdx.x;
    bool v8;
    v8 = v7 == 0l;
    if (v8){
        new(&v6) int{32l};
    } else {
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v9;
    v9 = threadIdx.x;
    bool v10;
    v10 = v9 == 0l;
    if (v10){
        Union0 v11;
        v11 = f_0(v1);
        unsigned long long v12; Union3 v13; dynamic_array_list<Union6,128l> v14; static_array<Union2,2l> v15; Union7 v16;
        Tuple1 tmp15 = f_6(v0);
        v12 = tmp15.v0; v13 = tmp15.v1; v14 = tmp15.v2; v15 = tmp15.v3; v16 = tmp15.v4;
        Union7 & v17 = v16;
        static_array<Union2,2l> & v18 = v15;
        Union3 & v19 = v13;
        unsigned long long & v20 = v12;
        dynamic_array_list<Union6,128l> & v21 = v14;
        unsigned long long v22;
        v22 = clock64();
        int v23;
        v23 = threadIdx.x;
        int v24;
        v24 = blockIdx.x;
        int v25;
        v25 = v24 * 32l;
        int v26;
        v26 = v23 + v25;
        unsigned long long v27;
        v27 = (unsigned long long)v26;
        curandStatePhilox4_32_10_t v28;
        curand_init(v22,v27,0ull,&v28);
        switch (v11.tag) {
            case 0: { // ActionSelected
                Union1 v41 = v11.case0.v0;
                Union3 v42 = v19;
                switch (v42.tag) {
                    case 0: { // None
                        printf("%s\n", "The game hasn't been started in ActionSelected.");
                        __trap();
                        break;
                    }
                    case 1: { // Some
                        Union4 v43 = v42.case1.v0;
                        switch (v43.tag) {
                            case 4: { // G_Round
                                int v44 = v43.case4.v0; static_array<static_array<unsigned char,2l>,2l> v45 = v43.case4.v1; static_array<int,2l> v46 = v43.case4.v2; int v47 = v43.case4.v3; static_array<int,2l> v48 = v43.case4.v4; Union5 v49 = v43.case4.v5;
                                Union4 v50;
                                v50 = Union4{Union4_5{v44, v45, v46, v47, v48, v49, v41}};
                                play_loop_31(v28, v2, v3, v4, v5, v20, v19, v21, v18, v17, v50);
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
                static_array<Union2,2l> v40 = v11.case1.v0;
                v18 = v40;
                break;
            }
            case 2: { // StartGame
                static_array<Union2,2l> v29;
                Union2 v31;
                v31 = Union2{Union2_0{}};
                v29[0l] = v31;
                Union2 v33;
                v33 = Union2{Union2_1{}};
                v29[1l] = v33;
                dynamic_array_list<Union6,128l> v35{0};
                Union7 v37;
                v37 = Union7{Union7_0{}};
                v17 = v37;
                v18 = v29;
                Union3 v38;
                v38 = Union3{Union3_0{}};
                v19 = v38;
                v20 = 4503599627370495ull;
                v21 = v35;
                Union4 v39;
                v39 = Union4{Union4_2{}};
                play_loop_31(v28, v2, v3, v4, v5, v20, v19, v21, v18, v17, v39);
                break;
            }
            default: {
                assert("Invalid tag." && false); __trap();
            }
        }
        f_55(v0, v12, v13, v14, v15, v16);
    } else {
    }
    int v51;
    v51 = atomicAdd(&v6,-1l);
    while (while_method_20()){
        bool v53;
        v53 = 5652480ull == v5;
        bool v54;
        v54 = v53 == false;
        if (v54){
            assert("The params needs to have matching offsets." && v53);
        } else {
        }
        bool v56;
        v56 = 279040ull == v3;
        bool v57;
        v57 = v56 == false;
        if (v57){
            assert("The outputs needs to have matching offsets." && v56);
        } else {
        }
        Union8 v59;
        v59 = Union8{Union8_0{}};
        Union9 v60;
        v60 = noinline_run_42(v2, v4, v59);
        int v61 = v6;
        bool v62;
        v62 = v61 == 0l;
        if (v62){
            break;
        } else {
        }
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
i8 = i16 = i32 = i64 = u8 = u16 = u32 = u64 = int; f32 = f64 = float; char = string = str

import time
options = []
options.append('--dopt=on')
options.append('--diag-suppress=550,20012,68,39')
options.append('--restrict')
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
def Closure0():
    def inner(v0 : object, v1 : object) -> object:
        v2 = cp.empty(16,dtype=cp.uint8)
        v3 = cp.empty(6304,dtype=cp.uint8)
        v4 = method0(v0)
        method8(v2, v4)
        del v4
        v5, v6, v7, v8, v9, v10, v11, v12, v13 = method15(v1)
        method53(v3, v5, v6, v7, v8, v9)
        del v5, v6, v7, v8, v9
        v16 = "{}\n"
        v17 = "Going to run the NL Holdem game kernel."
        print(v16.format(v17),end="")
        del v16, v17
        v18 = time.perf_counter()
        v19 = 0
        v20 = raw_module.get_function(f"entry{v19}")
        del v19
        v20.max_dynamic_shared_size_bytes = 1536 
        v20((1,),(32,),(v3, v2, v10, v11, v12, v13),shared_mem=1536)
        del v2, v20
        v21 = time.perf_counter()
        v24 = "{}"
        v25 = "The time it took to run the kernel (in seconds) is: "
        print(v24.format(v25),end="")
        del v24, v25
        v26 = v21 - v18
        del v18, v21
        v29 = "{:.6f}\n"
        print(v29.format(v26),end="")
        del v26, v29
        v30, v31, v32, v33, v34 = method81(v3)
        del v3
        return method109(v30, v31, v32, v33, v34, v10, v11, v12, v13)
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
        v8 = cp.empty(5652480,dtype=cp.uint8)
        v9 = cp.empty(279040,dtype=cp.uint8)
        v11 = v8[0:0+4*1048576].view(cp.float32)
        v12 = cp.random.normal(0.0,0.0009765625,1048576,dtype=cp.float32) # type: ignore
        cp.copyto(v11[0:0+1048576],v12[0:0+1048576])
        del v11, v12
        v14 = v8[4194304:4194304+4*65536].view(cp.float32)
        v16 = v8[4456448:4456448+4*65536].view(cp.float32)
        v18 = v8[4718592:4718592+4*65536].view(cp.float32)
        v20 = v8[4980736:4980736+4*65536].view(cp.float32)
        v22 = v8[5242880:5242880+4*65536].view(cp.float32)
        v24 = v8[5505024:5505024+4*2048].view(cp.int32)
        v26 = v8[5513216:5513216+4*2048].view(cp.float32)
        v28 = v8[5521408:5521408+4*2048].view(cp.int32)
        v30 = v8[5529600:5529600+4*2048].view(cp.int32)
        v32 = v8[5537792:5537792+8*4096].view(cp.float64)
        v34 = v8[5570560:5570560+8*4096].view(cp.float64)
        v36 = v8[5603328:5603328+4*8192].view(cp.float32)
        v38 = v8[5636096:5636096+4*2048].view(cp.float32)
        v40 = v8[5644288:5644288+4*2048].view(cp.float32)
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
        v30[:] = 0
        del v30
        v32[:] = 0
        del v32
        v34[:] = 0
        del v34
        v36[:] = 0
        del v36
        v38[:] = 0
        del v38
        v40[:] = 0
        del v40
        v41 = 4503599627370495
        v42 = US3_0()
        v43 = US6_0()
        v44 = 279040
        v45 = 5652480
        return method109(v41, v42, v7, v1, v43, v9, v44, v8, v45)
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
def method9(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[0:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method11(v0 : cp.ndarray) -> None:
    del v0
    return 
def method10(v0 : cp.ndarray, v1 : US1) -> None:
    v2 = v1.tag
    method9(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US1_0(): # A_All_In
            del v1
            return method11(v4)
        case US1_1(): # A_Call
            del v1
            return method11(v4)
        case US1_2(): # A_Fold
            del v1
            return method11(v4)
        case US1_3(v5): # A_Raise
            del v1
            return method9(v4, v5)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method13(v0 : i32) -> bool:
    v1 = v0 < 2
    del v0
    return v1
def method14(v0 : cp.ndarray, v1 : US2) -> None:
    v2 = v1.tag
    method9(v0, v2)
    del v2
    v4 = v0[4:].view(cp.uint8)
    del v0
    match v1:
        case US2_0(): # Computer
            del v1
            return method11(v4)
        case US2_1(): # Human
            del v1
            return method11(v4)
        case US2_2(): # Random
            del v1
            return method11(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method12(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method13(v2):
        v4 = u64(v2)
        v5 = v4 * 4
        del v4
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v9 = v1[v2]
        method14(v7, v9)
        del v7, v9
        v2 += 1 
    del v0, v1, v2
    return 
def method8(v0 : cp.ndarray, v1 : US0) -> None:
    v2 = v1.tag
    method9(v0, v2)
    del v2
    v4 = v0[8:].view(cp.uint8)
    del v0
    match v1:
        case US0_0(v5): # ActionSelected
            del v1
            return method10(v4, v5)
        case US0_1(v6): # PlayerChanged
            del v1
            return method12(v4, v6)
        case US0_2(): # StartGame
            del v1
            return method11(v4)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method20(v0 : object) -> u64:
    assert isinstance(v0,u64), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method19(v0 : object) -> u64:
    v1 = method20(v0)
    del v0
    return v1
def method27(v0 : object) -> u8:
    assert isinstance(v0,u8), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method26(v0 : object) -> u8:
    v1 = method27(v0)
    del v0
    return v1
def method25(v0 : object) -> static_array:
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
        v10 = method26(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
def method24(v0 : object) -> static_array:
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
        v10 = method25(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
def method28(v0 : object) -> static_array:
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
def method30(v0 : object) -> static_array:
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
        v10 = method26(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
def method31(v0 : object) -> static_array:
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
        v10 = method26(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
def method32(v0 : object) -> static_array:
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
        v10 = method26(v9)
        del v9
        v6[v7] = v10
        del v10
        v7 += 1 
    del v0, v1, v7
    return v6
def method29(v0 : object) -> US5:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "Flop" == v1
    if v3:
        del v1, v3
        v4 = method30(v2)
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
                v9 = method31(v2)
                del v2
                return US5_2(v9)
            else:
                del v8
                v11 = "Turn" == v1
                if v11:
                    del v1, v11
                    v12 = method32(v2)
                    del v2
                    return US5_3(v12)
                else:
                    del v2, v11
                    raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                    del v1
                    raise Exception("Error")
def method23(v0 : object) -> Tuple[i32, static_array, static_array, i32, static_array, US5]:
    v1 = v0["min_raise"] # type: ignore
    v2 = method4(v1)
    del v1
    v3 = v0["pl_card"] # type: ignore
    v4 = method24(v3)
    del v3
    v5 = v0["pot"] # type: ignore
    v6 = method28(v5)
    del v5
    v7 = v0["round_turn"] # type: ignore
    v8 = method4(v7)
    del v7
    v9 = v0["stack"] # type: ignore
    v10 = method28(v9)
    del v9
    v11 = v0["street"] # type: ignore
    del v0
    v12 = method29(v11)
    del v11
    return v2, v4, v6, v8, v10, v12
def method33(v0 : object) -> Tuple[i32, static_array, static_array, i32, static_array, US5, US1]:
    v1 = v0[0] # type: ignore
    v2, v3, v4, v5, v6, v7 = method23(v1)
    del v1
    v8 = v0[1] # type: ignore
    del v0
    v9 = method2(v8)
    del v8
    return v2, v3, v4, v5, v6, v7, v9
def method22(v0 : object) -> US4:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "G_Flop" == v1
    if v3:
        del v1, v3
        v4, v5, v6, v7, v8, v9 = method23(v2)
        del v2
        return US4_0(v4, v5, v6, v7, v8, v9)
    else:
        del v3
        v11 = "G_Fold" == v1
        if v11:
            del v1, v11
            v12, v13, v14, v15, v16, v17 = method23(v2)
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
                    v22, v23, v24, v25, v26, v27 = method23(v2)
                    del v2
                    return US4_3(v22, v23, v24, v25, v26, v27)
                else:
                    del v21
                    v29 = "G_Round" == v1
                    if v29:
                        del v1, v29
                        v30, v31, v32, v33, v34, v35 = method23(v2)
                        del v2
                        return US4_4(v30, v31, v32, v33, v34, v35)
                    else:
                        del v29
                        v37 = "G_Round'" == v1
                        if v37:
                            del v1, v37
                            v38, v39, v40, v41, v42, v43, v44 = method33(v2)
                            del v2
                            return US4_5(v38, v39, v40, v41, v42, v43, v44)
                        else:
                            del v37
                            v46 = "G_Showdown" == v1
                            if v46:
                                del v1, v46
                                v47, v48, v49, v50, v51, v52 = method23(v2)
                                del v2
                                return US4_6(v47, v48, v49, v50, v51, v52)
                            else:
                                del v46
                                v54 = "G_Turn" == v1
                                if v54:
                                    del v1, v54
                                    v55, v56, v57, v58, v59, v60 = method23(v2)
                                    del v2
                                    return US4_7(v55, v56, v57, v58, v59, v60)
                                else:
                                    del v2, v54
                                    raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                                    del v1
                                    raise Exception("Error")
def method21(v0 : object) -> US3:
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
            v6 = method22(v2)
            del v2
            return US3_1(v6)
        else:
            del v2, v5
            raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
            del v1
            raise Exception("Error")
def method18(v0 : object) -> Tuple[u64, US3]:
    v1 = v0["deck"] # type: ignore
    v2 = method19(v1)
    del v1
    v3 = v0["game"] # type: ignore
    del v0
    v4 = method21(v3)
    del v3
    return v2, v4
def method37(v0 : object) -> static_array_list:
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
        v11 = method26(v10)
        del v10
        v7[v8] = v11
        del v11
        v8 += 1 
    del v0, v2, v8
    return v7
def method38(v0 : object) -> Tuple[i32, i32]:
    v1 = v0["chips_won"] # type: ignore
    v2 = method4(v1)
    del v1
    v3 = v0["winner_id"] # type: ignore
    del v0
    v4 = method4(v3)
    del v3
    return v2, v4
def method39(v0 : object) -> Tuple[i32, US1]:
    v1 = v0[0] # type: ignore
    v2 = method4(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method2(v3)
    del v3
    return v2, v4
def method40(v0 : object) -> Tuple[i32, static_array]:
    v1 = v0[0] # type: ignore
    v2 = method4(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method25(v3)
    del v3
    return v2, v4
def method45(v0 : object) -> i8:
    assert isinstance(v0,i8), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method44(v0 : object) -> Tuple[static_array, i8]:
    v1 = v0["hand"] # type: ignore
    v2 = method31(v1)
    del v1
    v3 = v0["score"] # type: ignore
    del v0
    v4 = method45(v3)
    del v3
    return v2, v4
def method43(v0 : object) -> Tuple[static_array, i8]:
    v1, v2 = method44(v0)
    del v0
    return v1, v2
def method42(v0 : object) -> static_array:
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
        v10, v11 = method43(v9)
        del v9
        v6[v7] = (v10, v11)
        del v10, v11
        v7 += 1 
    del v0, v1, v7
    return v6
def method41(v0 : object) -> Tuple[i32, static_array, i32]:
    v1 = v0["chips_won"] # type: ignore
    v2 = method4(v1)
    del v1
    v3 = v0["hands_shown"] # type: ignore
    v4 = method42(v3)
    del v3
    v5 = v0["winner_id"] # type: ignore
    del v0
    v6 = method4(v5)
    del v5
    return v2, v4, v6
def method36(v0 : object) -> US7:
    v1 = v0[0] # type: ignore
    v2 = v0[1] # type: ignore
    del v0
    v3 = "CommunityCardsAre" == v1
    if v3:
        del v1, v3
        v4 = method37(v2)
        del v2
        return US7_0(v4)
    else:
        del v3
        v6 = "Fold" == v1
        if v6:
            del v1, v6
            v7, v8 = method38(v2)
            del v2
            return US7_1(v7, v8)
        else:
            del v6
            v10 = "PlayerAction" == v1
            if v10:
                del v1, v10
                v11, v12 = method39(v2)
                del v2
                return US7_2(v11, v12)
            else:
                del v10
                v14 = "PlayerGotCards" == v1
                if v14:
                    del v1, v14
                    v15, v16 = method40(v2)
                    del v2
                    return US7_3(v15, v16)
                else:
                    del v14
                    v18 = "Showdown" == v1
                    if v18:
                        del v1, v18
                        v19, v20, v21 = method41(v2)
                        del v2
                        return US7_4(v19, v20, v21)
                    else:
                        del v2, v18
                        raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                        del v1
                        raise Exception("Error")
def method35(v0 : object) -> dynamic_array_list:
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
        v11 = method36(v10)
        del v10
        v7[v8] = v11
        del v11
        v8 += 1 
    del v0, v2, v8
    return v7
def method46(v0 : object) -> US6:
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
            v6, v7, v8, v9, v10, v11 = method23(v2)
            del v2
            return US6_1(v6, v7, v8, v9, v10, v11)
        else:
            del v5
            v13 = "WaitingForActionFromPlayerId" == v1
            if v13:
                del v1, v13
                v14, v15, v16, v17, v18, v19 = method23(v2)
                del v2
                return US6_2(v14, v15, v16, v17, v18, v19)
            else:
                del v2, v13
                raise TypeError(f"Cannot convert the Python object into a Spiral union type. Invalid string tag. Got: {v1}")
                del v1
                raise Exception("Error")
def method34(v0 : object) -> Tuple[dynamic_array_list, static_array, US6]:
    v1 = v0["messages"] # type: ignore
    v2 = method35(v1)
    del v1
    v3 = v0["pl_type"] # type: ignore
    v4 = method5(v3)
    del v3
    v5 = v0["ui_game_state"] # type: ignore
    del v0
    v6 = method46(v5)
    del v5
    return v2, v4, v6
def method17(v0 : object) -> Tuple[u64, US3, dynamic_array_list, static_array, US6]:
    v1 = v0["private"] # type: ignore
    v2, v3 = method18(v1)
    del v1
    v4 = v0["public"] # type: ignore
    del v0
    v5, v6, v7 = method34(v4)
    del v4
    return v2, v3, v5, v6, v7
def method52(v0 : object) -> cp.ndarray:
    assert isinstance(v0,cp.ndarray), f'The object needs to be the right primitive type. Got: {v0}'
    v1 = v0
    del v0
    return v1
def method51(v0 : object) -> cp.ndarray:
    v1 = method52(v0)
    del v0
    return v1
def method50(v0 : object) -> Tuple[cp.ndarray, u64]:
    v1 = v0[0] # type: ignore
    v2 = method51(v1)
    del v1
    v3 = v0[1] # type: ignore
    del v0
    v4 = method20(v3)
    del v3
    return v2, v4
def method49(v0 : object) -> Tuple[cp.ndarray, u64, cp.ndarray, u64]:
    v1 = v0["output"] # type: ignore
    v2, v3 = method50(v1)
    del v1
    v4 = v0["param"] # type: ignore
    del v0
    v5, v6 = method50(v4)
    del v4
    return v2, v3, v5, v6
def method48(v0 : object) -> Tuple[cp.ndarray, u64, cp.ndarray, u64]:
    v1, v2, v3, v4 = method49(v0)
    del v0
    return v1, v2, v3, v4
def method47(v0 : object) -> Tuple[cp.ndarray, u64, cp.ndarray, u64]:
    v1 = v0["model_data"] # type: ignore
    del v0
    v2, v3, v4, v5 = method48(v1)
    del v1
    return v2, v3, v4, v5
def method16(v0 : object) -> Tuple[u64, US3, dynamic_array_list, static_array, US6, cp.ndarray, u64, cp.ndarray, u64]:
    v1 = v0["game"] # type: ignore
    v2, v3, v4, v5, v6 = method17(v1)
    del v1
    v7 = v0["neural"] # type: ignore
    del v0
    v8, v9, v10, v11 = method47(v7)
    del v7
    return v2, v3, v4, v5, v6, v8, v9, v10, v11
def method15(v0 : object) -> Tuple[u64, US3, dynamic_array_list, static_array, US6, cp.ndarray, u64, cp.ndarray, u64]:
    return method16(v0)
def method54(v0 : cp.ndarray, v1 : u64) -> None:
    v3 = v0[0:].view(cp.uint64)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method55(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[8:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method60(v0 : cp.ndarray, v1 : u8) -> None:
    v3 = v0[0:].view(cp.uint8)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method59(v0 : cp.ndarray, v1 : u8) -> None:
    return method60(v0, v1)
def method58(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method13(v2):
        v4 = u64(v2)
        v6 = v0[v4:].view(cp.uint8)
        del v4
        v8 = v1[v2]
        method59(v6, v8)
        del v6, v8
        v2 += 1 
    del v0, v1, v2
    return 
def method61(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[28:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method63(v0 : i32) -> bool:
    v1 = v0 < 3
    del v0
    return v1
def method62(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method63(v2):
        v4 = u64(v2)
        v6 = v0[v4:].view(cp.uint8)
        del v4
        v8 = v1[v2]
        method59(v6, v8)
        del v6, v8
        v2 += 1 
    del v0, v1, v2
    return 
def method65(v0 : i32) -> bool:
    v1 = v0 < 5
    del v0
    return v1
def method64(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method65(v2):
        v4 = u64(v2)
        v6 = v0[v4:].view(cp.uint8)
        del v4
        v8 = v1[v2]
        method59(v6, v8)
        del v6, v8
        v2 += 1 
    del v0, v1, v2
    return 
def method67(v0 : i32) -> bool:
    v1 = v0 < 4
    del v0
    return v1
def method66(v0 : cp.ndarray, v1 : static_array) -> None:
    v2 = 0
    while method67(v2):
        v4 = u64(v2)
        v6 = v0[v4:].view(cp.uint8)
        del v4
        v8 = v1[v2]
        method59(v6, v8)
        del v6, v8
        v2 += 1 
    del v0, v1, v2
    return 
def method57(v0 : cp.ndarray, v1 : i32, v2 : static_array, v3 : static_array, v4 : i32, v5 : static_array, v6 : US5) -> None:
    v8 = v0[0:].view(cp.int32)
    v8[0] = v1
    del v1, v8
    v9 = 0
    while method13(v9):
        v11 = u64(v9)
        v12 = v11 * 2
        del v11
        v13 = 4 + v12
        del v12
        v15 = v0[v13:].view(cp.uint8)
        del v13
        v17 = v2[v9]
        method58(v15, v17)
        del v15, v17
        v9 += 1 
    del v2, v9
    v18 = 0
    while method13(v18):
        v20 = u64(v18)
        v21 = v20 * 4
        del v20
        v22 = 8 + v21
        del v21
        v24 = v0[v22:].view(cp.uint8)
        del v22
        v26 = v3[v18]
        method9(v24, v26)
        del v24, v26
        v18 += 1 
    del v3, v18
    v28 = v0[16:].view(cp.int32)
    v28[0] = v4
    del v4, v28
    v29 = 0
    while method13(v29):
        v31 = u64(v29)
        v32 = v31 * 4
        del v31
        v33 = 20 + v32
        del v32
        v35 = v0[v33:].view(cp.uint8)
        del v33
        v37 = v5[v29]
        method9(v35, v37)
        del v35, v37
        v29 += 1 
    del v5, v29
    v38 = v6.tag
    method61(v0, v38)
    del v38
    v40 = v0[32:].view(cp.uint8)
    del v0
    match v6:
        case US5_0(v41): # Flop
            del v6
            return method62(v40, v41)
        case US5_1(): # Preflop
            del v6
            return method11(v40)
        case US5_2(v42): # River
            del v6
            return method64(v40, v42)
        case US5_3(v43): # Turn
            del v6
            return method66(v40, v43)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method69(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[40:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method68(v0 : cp.ndarray, v1 : i32, v2 : static_array, v3 : static_array, v4 : i32, v5 : static_array, v6 : US5, v7 : US1) -> None:
    v9 = v0[0:].view(cp.int32)
    v9[0] = v1
    del v1, v9
    v10 = 0
    while method13(v10):
        v12 = u64(v10)
        v13 = v12 * 2
        del v12
        v14 = 4 + v13
        del v13
        v16 = v0[v14:].view(cp.uint8)
        del v14
        v18 = v2[v10]
        method58(v16, v18)
        del v16, v18
        v10 += 1 
    del v2, v10
    v19 = 0
    while method13(v19):
        v21 = u64(v19)
        v22 = v21 * 4
        del v21
        v23 = 8 + v22
        del v22
        v25 = v0[v23:].view(cp.uint8)
        del v23
        v27 = v3[v19]
        method9(v25, v27)
        del v25, v27
        v19 += 1 
    del v3, v19
    v29 = v0[16:].view(cp.int32)
    v29[0] = v4
    del v4, v29
    v30 = 0
    while method13(v30):
        v32 = u64(v30)
        v33 = v32 * 4
        del v32
        v34 = 20 + v33
        del v33
        v36 = v0[v34:].view(cp.uint8)
        del v34
        v38 = v5[v30]
        method9(v36, v38)
        del v36, v38
        v30 += 1 
    del v5, v30
    v39 = v6.tag
    method61(v0, v39)
    del v39
    v41 = v0[32:].view(cp.uint8)
    match v6:
        case US5_0(v42): # Flop
            method62(v41, v42)
        case US5_1(): # Preflop
            method11(v41)
        case US5_2(v43): # River
            method64(v41, v43)
        case US5_3(v44): # Turn
            method66(v41, v44)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v6, v41
    v45 = v7.tag
    method69(v0, v45)
    del v45
    v47 = v0[44:].view(cp.uint8)
    del v0
    match v7:
        case US1_0(): # A_All_In
            del v7
            return method11(v47)
        case US1_1(): # A_Call
            del v7
            return method11(v47)
        case US1_2(): # A_Fold
            del v7
            return method11(v47)
        case US1_3(v48): # A_Raise
            del v7
            return method9(v47, v48)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method56(v0 : cp.ndarray, v1 : US4) -> None:
    v2 = v1.tag
    method9(v0, v2)
    del v2
    v4 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US4_0(v5, v6, v7, v8, v9, v10): # G_Flop
            del v1
            return method57(v4, v5, v6, v7, v8, v9, v10)
        case US4_1(v11, v12, v13, v14, v15, v16): # G_Fold
            del v1
            return method57(v4, v11, v12, v13, v14, v15, v16)
        case US4_2(): # G_Preflop
            del v1
            return method11(v4)
        case US4_3(v17, v18, v19, v20, v21, v22): # G_River
            del v1
            return method57(v4, v17, v18, v19, v20, v21, v22)
        case US4_4(v23, v24, v25, v26, v27, v28): # G_Round
            del v1
            return method57(v4, v23, v24, v25, v26, v27, v28)
        case US4_5(v29, v30, v31, v32, v33, v34, v35): # G_Round'
            del v1
            return method68(v4, v29, v30, v31, v32, v33, v34, v35)
        case US4_6(v36, v37, v38, v39, v40, v41): # G_Showdown
            del v1
            return method57(v4, v36, v37, v38, v39, v40, v41)
        case US4_7(v42, v43, v44, v45, v46, v47): # G_Turn
            del v1
            return method57(v4, v42, v43, v44, v45, v46, v47)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method70(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[80:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method72(v0 : cp.ndarray, v1 : static_array_list) -> None:
    v2 = v1.length
    method9(v0, v2)
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
        method59(v9, v11)
        del v9, v11
        v4 += 1 
    del v0, v1, v3, v4
    return 
def method73(v0 : cp.ndarray, v1 : i32, v2 : i32) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v6 = v0[4:].view(cp.int32)
    del v0
    v6[0] = v2
    del v2, v6
    return 
def method75(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[4:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method74(v0 : cp.ndarray, v1 : i32, v2 : US1) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v5 = v2.tag
    method75(v0, v5)
    del v5
    v7 = v0[8:].view(cp.uint8)
    del v0
    match v2:
        case US1_0(): # A_All_In
            del v2
            return method11(v7)
        case US1_1(): # A_Call
            del v2
            return method11(v7)
        case US1_2(): # A_Fold
            del v2
            return method11(v7)
        case US1_3(v8): # A_Raise
            del v2
            return method9(v7, v8)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method76(v0 : cp.ndarray, v1 : i32, v2 : static_array) -> None:
    v4 = v0[0:].view(cp.int32)
    v4[0] = v1
    del v1, v4
    v5 = 0
    while method13(v5):
        v7 = u64(v5)
        v8 = 4 + v7
        del v7
        v10 = v0[v8:].view(cp.uint8)
        del v8
        v12 = v2[v5]
        method59(v10, v12)
        del v10, v12
        v5 += 1 
    del v0, v2, v5
    return 
def method79(v0 : cp.ndarray, v1 : static_array, v2 : i8) -> None:
    v3 = 0
    while method65(v3):
        v5 = u64(v3)
        v7 = v0[v5:].view(cp.uint8)
        del v5
        v9 = v1[v3]
        method59(v7, v9)
        del v7, v9
        v3 += 1 
    del v1, v3
    v11 = v0[5:].view(cp.int8)
    del v0
    v11[0] = v2
    del v2, v11
    return 
def method78(v0 : cp.ndarray, v1 : static_array, v2 : i8) -> None:
    return method79(v0, v1, v2)
def method77(v0 : cp.ndarray, v1 : i32, v2 : static_array, v3 : i32) -> None:
    v5 = v0[0:].view(cp.int32)
    v5[0] = v1
    del v1, v5
    v6 = 0
    while method13(v6):
        v8 = u64(v6)
        v9 = v8 * 8
        del v8
        v10 = 8 + v9
        del v9
        v12 = v0[v10:].view(cp.uint8)
        del v10
        v15, v16 = v2[v6]
        method78(v12, v15, v16)
        del v12, v15, v16
        v6 += 1 
    del v2, v6
    v18 = v0[24:].view(cp.int32)
    del v0
    v18[0] = v3
    del v3, v18
    return 
def method71(v0 : cp.ndarray, v1 : US7) -> None:
    v2 = v1.tag
    method9(v0, v2)
    del v2
    v4 = v0[16:].view(cp.uint8)
    del v0
    match v1:
        case US7_0(v5): # CommunityCardsAre
            del v1
            return method72(v4, v5)
        case US7_1(v6, v7): # Fold
            del v1
            return method73(v4, v6, v7)
        case US7_2(v8, v9): # PlayerAction
            del v1
            return method74(v4, v8, v9)
        case US7_3(v10, v11): # PlayerGotCards
            del v1
            return method76(v4, v10, v11)
        case US7_4(v12, v13, v14): # Showdown
            del v1
            return method77(v4, v12, v13, v14)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method80(v0 : cp.ndarray, v1 : i32) -> None:
    v3 = v0[6248:].view(cp.int32)
    del v0
    v3[0] = v1
    del v1, v3
    return 
def method53(v0 : cp.ndarray, v1 : u64, v2 : US3, v3 : dynamic_array_list, v4 : static_array, v5 : US6) -> None:
    method54(v0, v1)
    del v1
    v6 = v2.tag
    method55(v0, v6)
    del v6
    v8 = v0[16:].view(cp.uint8)
    match v2:
        case US3_0(): # None
            method11(v8)
        case US3_1(v9): # Some
            method56(v8, v9)
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
    del v2, v8
    v10 = v3.length_()
    method70(v0, v10)
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
        method71(v18, v20)
        del v18, v20
        v12 += 1 
    del v3, v11, v12
    v21 = 0
    while method13(v21):
        v23 = u64(v21)
        v24 = v23 * 4
        del v23
        v25 = 6240 + v24
        del v24
        v27 = v0[v25:].view(cp.uint8)
        del v25
        v29 = v4[v21]
        method14(v27, v29)
        del v27, v29
        v21 += 1 
    del v4, v21
    v30 = v5.tag
    method80(v0, v30)
    del v30
    v32 = v0[6256:].view(cp.uint8)
    del v0
    match v5:
        case US6_0(): # GameNotStarted
            del v5
            return method11(v32)
        case US6_1(v33, v34, v35, v36, v37, v38): # GameOver
            del v5
            return method57(v32, v33, v34, v35, v36, v37, v38)
        case US6_2(v39, v40, v41, v42, v43, v44): # WaitingForActionFromPlayerId
            del v5
            return method57(v32, v39, v40, v41, v42, v43, v44)
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
    while method13(v3):
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
def method93(v0 : cp.ndarray) -> static_array:
    v2 = static_array(5)
    v3 = 0
    while method65(v3):
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
    while method67(v3):
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
    while method13(v6):
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
    while method13(v16):
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
    while method13(v29):
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
    while method13(v6):
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
    while method13(v16):
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
    while method13(v29):
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
    while method13(v6):
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
    while method65(v3):
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
    while method13(v6):
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
    while method13(v24):
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
def method114(v0 : u64) -> object:
    v1 = v0
    del v0
    return v1
def method113(v0 : u64) -> object:
    return method114(v0)
def method116() -> object:
    v0 = []
    return v0
def method119(v0 : i32) -> object:
    v1 = v0
    del v0
    return v1
def method123(v0 : u8) -> object:
    v1 = v0
    del v0
    return v1
def method122(v0 : u8) -> object:
    return method123(v0)
def method121(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method13(v2):
        v5 = v0[v2]
        v6 = method122(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method120(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method13(v2):
        v5 = v0[v2]
        v6 = method121(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method124(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method13(v2):
        v5 = v0[v2]
        v6 = method119(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method126(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method63(v2):
        v5 = v0[v2]
        v6 = method122(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method127(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method65(v2):
        v5 = v0[v2]
        v6 = method122(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method128(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method67(v2):
        v5 = v0[v2]
        v6 = method122(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method125(v0 : US5) -> object:
    match v0:
        case US5_0(v1): # Flop
            del v0
            v2 = method126(v1)
            del v1
            v3 = "Flop"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US5_1(): # Preflop
            del v0
            v5 = method116()
            v6 = "Preflop"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case US5_2(v8): # River
            del v0
            v9 = method127(v8)
            del v8
            v10 = "River"
            v11 = [v10,v9]
            del v9, v10
            return v11
        case US5_3(v12): # Turn
            del v0
            v13 = method128(v12)
            del v12
            v14 = "Turn"
            v15 = [v14,v13]
            del v13, v14
            return v15
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method118(v0 : i32, v1 : static_array, v2 : static_array, v3 : i32, v4 : static_array, v5 : US5) -> object:
    v6 = method119(v0)
    del v0
    v7 = method120(v1)
    del v1
    v8 = method124(v2)
    del v2
    v9 = method119(v3)
    del v3
    v10 = method124(v4)
    del v4
    v11 = method125(v5)
    del v5
    v12 = {'min_raise': v6, 'pl_card': v7, 'pot': v8, 'round_turn': v9, 'stack': v10, 'street': v11}
    del v6, v7, v8, v9, v10, v11
    return v12
def method130(v0 : US1) -> object:
    match v0:
        case US1_0(): # A_All_In
            del v0
            v1 = method116()
            v2 = "A_All_In"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US1_1(): # A_Call
            del v0
            v4 = method116()
            v5 = "A_Call"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US1_2(): # A_Fold
            del v0
            v7 = method116()
            v8 = "A_Fold"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US1_3(v10): # A_Raise
            del v0
            v11 = method119(v10)
            del v10
            v12 = "A_Raise"
            v13 = [v12,v11]
            del v11, v12
            return v13
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method129(v0 : i32, v1 : static_array, v2 : static_array, v3 : i32, v4 : static_array, v5 : US5, v6 : US1) -> object:
    v7 = []
    v8 = method118(v0, v1, v2, v3, v4, v5)
    del v0, v1, v2, v3, v4, v5
    v7.append(v8)
    del v8
    v9 = method130(v6)
    del v6
    v7.append(v9)
    del v9
    v10 = v7
    del v7
    return v10
def method117(v0 : US4) -> object:
    match v0:
        case US4_0(v1, v2, v3, v4, v5, v6): # G_Flop
            del v0
            v7 = method118(v1, v2, v3, v4, v5, v6)
            del v1, v2, v3, v4, v5, v6
            v8 = "G_Flop"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US4_1(v10, v11, v12, v13, v14, v15): # G_Fold
            del v0
            v16 = method118(v10, v11, v12, v13, v14, v15)
            del v10, v11, v12, v13, v14, v15
            v17 = "G_Fold"
            v18 = [v17,v16]
            del v16, v17
            return v18
        case US4_2(): # G_Preflop
            del v0
            v19 = method116()
            v20 = "G_Preflop"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case US4_3(v22, v23, v24, v25, v26, v27): # G_River
            del v0
            v28 = method118(v22, v23, v24, v25, v26, v27)
            del v22, v23, v24, v25, v26, v27
            v29 = "G_River"
            v30 = [v29,v28]
            del v28, v29
            return v30
        case US4_4(v31, v32, v33, v34, v35, v36): # G_Round
            del v0
            v37 = method118(v31, v32, v33, v34, v35, v36)
            del v31, v32, v33, v34, v35, v36
            v38 = "G_Round"
            v39 = [v38,v37]
            del v37, v38
            return v39
        case US4_5(v40, v41, v42, v43, v44, v45, v46): # G_Round'
            del v0
            v47 = method129(v40, v41, v42, v43, v44, v45, v46)
            del v40, v41, v42, v43, v44, v45, v46
            v48 = "G_Round'"
            v49 = [v48,v47]
            del v47, v48
            return v49
        case US4_6(v50, v51, v52, v53, v54, v55): # G_Showdown
            del v0
            v56 = method118(v50, v51, v52, v53, v54, v55)
            del v50, v51, v52, v53, v54, v55
            v57 = "G_Showdown"
            v58 = [v57,v56]
            del v56, v57
            return v58
        case US4_7(v59, v60, v61, v62, v63, v64): # G_Turn
            del v0
            v65 = method118(v59, v60, v61, v62, v63, v64)
            del v59, v60, v61, v62, v63, v64
            v66 = "G_Turn"
            v67 = [v66,v65]
            del v65, v66
            return v67
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method115(v0 : US3) -> object:
    match v0:
        case US3_0(): # None
            del v0
            v1 = method116()
            v2 = "None"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US3_1(v4): # Some
            del v0
            v5 = method117(v4)
            del v4
            v6 = "Some"
            v7 = [v6,v5]
            del v5, v6
            return v7
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method112(v0 : u64, v1 : US3) -> object:
    v2 = method113(v0)
    del v0
    v3 = method115(v1)
    del v1
    v4 = {'deck': v2, 'game': v3}
    del v2, v3
    return v4
def method134(v0 : static_array_list) -> object:
    v1 = []
    v2 = v0.length
    v3 = 0
    while method6(v2, v3):
        v6 = v0[v3]
        v7 = method122(v6)
        del v6
        v1.append(v7)
        del v7
        v3 += 1 
    del v0, v2, v3
    return v1
def method135(v0 : i32, v1 : i32) -> object:
    v2 = method119(v0)
    del v0
    v3 = method119(v1)
    del v1
    v4 = {'chips_won': v2, 'winner_id': v3}
    del v2, v3
    return v4
def method136(v0 : i32, v1 : US1) -> object:
    v2 = []
    v3 = method119(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method130(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method137(v0 : i32, v1 : static_array) -> object:
    v2 = []
    v3 = method119(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method121(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method142(v0 : i8) -> object:
    v1 = v0
    del v0
    return v1
def method141(v0 : static_array, v1 : i8) -> object:
    v2 = method127(v0)
    del v0
    v3 = method142(v1)
    del v1
    v4 = {'hand': v2, 'score': v3}
    del v2, v3
    return v4
def method140(v0 : static_array, v1 : i8) -> object:
    return method141(v0, v1)
def method139(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method13(v2):
        v6, v7 = v0[v2]
        v8 = method140(v6, v7)
        del v6, v7
        v1.append(v8)
        del v8
        v2 += 1 
    del v0, v2
    return v1
def method138(v0 : i32, v1 : static_array, v2 : i32) -> object:
    v3 = method119(v0)
    del v0
    v4 = method139(v1)
    del v1
    v5 = method119(v2)
    del v2
    v6 = {'chips_won': v3, 'hands_shown': v4, 'winner_id': v5}
    del v3, v4, v5
    return v6
def method133(v0 : US7) -> object:
    match v0:
        case US7_0(v1): # CommunityCardsAre
            del v0
            v2 = method134(v1)
            del v1
            v3 = "CommunityCardsAre"
            v4 = [v3,v2]
            del v2, v3
            return v4
        case US7_1(v5, v6): # Fold
            del v0
            v7 = method135(v5, v6)
            del v5, v6
            v8 = "Fold"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case US7_2(v10, v11): # PlayerAction
            del v0
            v12 = method136(v10, v11)
            del v10, v11
            v13 = "PlayerAction"
            v14 = [v13,v12]
            del v12, v13
            return v14
        case US7_3(v15, v16): # PlayerGotCards
            del v0
            v17 = method137(v15, v16)
            del v15, v16
            v18 = "PlayerGotCards"
            v19 = [v18,v17]
            del v17, v18
            return v19
        case US7_4(v20, v21, v22): # Showdown
            del v0
            v23 = method138(v20, v21, v22)
            del v20, v21, v22
            v24 = "Showdown"
            v25 = [v24,v23]
            del v23, v24
            return v25
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method132(v0 : dynamic_array_list) -> object:
    v1 = []
    v2 = v0.length_()
    v3 = 0
    while method6(v2, v3):
        v6 = v0[v3]
        v7 = method133(v6)
        del v6
        v1.append(v7)
        del v7
        v3 += 1 
    del v0, v2, v3
    return v1
def method144(v0 : US2) -> object:
    match v0:
        case US2_0(): # Computer
            del v0
            v1 = method116()
            v2 = "Computer"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US2_1(): # Human
            del v0
            v4 = method116()
            v5 = "Human"
            v6 = [v5,v4]
            del v4, v5
            return v6
        case US2_2(): # Random
            del v0
            v7 = method116()
            v8 = "Random"
            v9 = [v8,v7]
            del v7, v8
            return v9
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method143(v0 : static_array) -> object:
    v1 = []
    v2 = 0
    while method13(v2):
        v5 = v0[v2]
        v6 = method144(v5)
        del v5
        v1.append(v6)
        del v6
        v2 += 1 
    del v0, v2
    return v1
def method145(v0 : US6) -> object:
    match v0:
        case US6_0(): # GameNotStarted
            del v0
            v1 = method116()
            v2 = "GameNotStarted"
            v3 = [v2,v1]
            del v1, v2
            return v3
        case US6_1(v4, v5, v6, v7, v8, v9): # GameOver
            del v0
            v10 = method118(v4, v5, v6, v7, v8, v9)
            del v4, v5, v6, v7, v8, v9
            v11 = "GameOver"
            v12 = [v11,v10]
            del v10, v11
            return v12
        case US6_2(v13, v14, v15, v16, v17, v18): # WaitingForActionFromPlayerId
            del v0
            v19 = method118(v13, v14, v15, v16, v17, v18)
            del v13, v14, v15, v16, v17, v18
            v20 = "WaitingForActionFromPlayerId"
            v21 = [v20,v19]
            del v19, v20
            return v21
        case t:
            raise Exception(f'Pattern matching miss. Got: {t}')
def method131(v0 : dynamic_array_list, v1 : static_array, v2 : US6) -> object:
    v3 = method132(v0)
    del v0
    v4 = method143(v1)
    del v1
    v5 = method145(v2)
    del v2
    v6 = {'messages': v3, 'pl_type': v4, 'ui_game_state': v5}
    del v3, v4, v5
    return v6
def method111(v0 : u64, v1 : US3, v2 : dynamic_array_list, v3 : static_array, v4 : US6) -> object:
    v5 = method112(v0, v1)
    del v0, v1
    v6 = method131(v2, v3, v4)
    del v2, v3, v4
    v7 = {'private': v5, 'public': v6}
    del v5, v6
    return v7
def method151(v0 : cp.ndarray) -> object:
    v1 = v0
    del v0
    return v1
def method150(v0 : cp.ndarray) -> object:
    return method151(v0)
def method149(v0 : cp.ndarray, v1 : u64) -> object:
    v2 = []
    v3 = method150(v0)
    del v0
    v2.append(v3)
    del v3
    v4 = method114(v1)
    del v1
    v2.append(v4)
    del v4
    v5 = v2
    del v2
    return v5
def method148(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method149(v0, v1)
    del v0, v1
    v5 = method149(v2, v3)
    del v2, v3
    v6 = {'output': v4, 'param': v5}
    del v4, v5
    return v6
def method147(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    return method148(v0, v1, v2, v3)
def method146(v0 : cp.ndarray, v1 : u64, v2 : cp.ndarray, v3 : u64) -> object:
    v4 = method147(v0, v1, v2, v3)
    del v0, v1, v2, v3
    v5 = {'model_data': v4}
    del v4
    return v5
def method110(v0 : u64, v1 : US3, v2 : dynamic_array_list, v3 : static_array, v4 : US6, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64) -> object:
    v9 = method111(v0, v1, v2, v3, v4)
    del v0, v1, v2, v3, v4
    v10 = method146(v5, v6, v7, v8)
    del v5, v6, v7, v8
    v11 = {'game': v9, 'neural': v10}
    del v9, v10
    return v11
def method109(v0 : u64, v1 : US3, v2 : dynamic_array_list, v3 : static_array, v4 : US6, v5 : cp.ndarray, v6 : u64, v7 : cp.ndarray, v8 : u64) -> object:
    v9 = method110(v0, v1, v2, v3, v4, v5, v6, v7, v8)
    del v0, v1, v2, v3, v4, v5, v6, v7, v8
    return v9
def main_body():
    v0 = Closure0()
    v1 = Closure1()
    v2 = collections.namedtuple("HU_Holdem_Game",['event_loop_gpu', 'init'])(v0, v1)
    del v0, v1
    return v2

def main():
    r = main_body()
    cp.cuda.get_current_stream().synchronize() # This line is here so the `__trap()` calls on the kernel aren't missed.
    return r

if __name__ == '__main__': print(main())
