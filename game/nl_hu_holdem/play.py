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
struct Tuple13;
struct Union10;
struct Tuple14;
__device__ void method_43(unsigned int v0, float * v1, int v2);
__device__ void method_44(unsigned int v0, float * v1, int v2);
__device__ void method_45(float * v0, int v1, float * v2, int v3, float * v4, int v5);
struct Tuple15;
struct Tuple16;
__device__ Tuple16 method_47(float v0, int v1, float v2, int v3);
__device__ void method_46(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5);
__device__ int int_range_48(int v0, int v1, curandStatePhilox4_32_10_t & v2);
__device__ Tuple13 noinline_run_42(unsigned char * v0, unsigned char * v1, Union8 v2);
__device__ void method_49(Union1 v0);
struct Tuple17;
__device__ int loop_53(static_array<float,6l> v0, float v1, int v2);
__device__ int pick_discrete__52(static_array<float,6l> v0, float v1);
__device__ int sample_discrete__51(static_array<float,6l> v0, curandStatePhilox4_32_10_t & v1);
__device__ Union1 sample_discrete_50(static_array<Tuple17,6l> v0, curandStatePhilox4_32_10_t & v1);
struct Tuple18;
struct Tuple19;
struct Union11;
struct Tuple20;
struct Union12;
struct Tuple21;
struct Tuple22;
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
struct Tuple13 {
    Union9 v0;
    float * v1;
    int v2;
    int v3;
    int v4;
    float v5;
    __device__ Tuple13() = default;
    __device__ Tuple13(Union9 t0, float * t1, int t2, int t3, int t4, float t5) : v0(t0), v1(t1), v2(t2), v3(t3), v4(t4), v5(t5) {}
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
struct Tuple14 {
    int v0;
    unsigned int v1;
    __device__ Tuple14() = default;
    __device__ Tuple14(int t0, unsigned int t1) : v0(t0), v1(t1) {}
};
struct Closure0 {
    __device__ int operator()(int tup0, int tup1){
        int v0 = tup0; int v1 = tup1;
        int v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Closure1 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
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
struct Closure2 {
    __device__ float operator()(float tup0, float tup1){
        float v0 = tup0; float v1 = tup1;
        float v2;
        v2 = v0 + v1;
        return v2;
    }
};
struct Tuple16 {
    float v0;
    int v1;
    __device__ Tuple16() = default;
    __device__ Tuple16(float t0, int t1) : v0(t0), v1(t1) {}
};
struct Closure3 {
    __device__ Tuple16 operator()(Tuple16 tup0, Tuple16 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        bool v4;
        v4 = v1 < v3;
        if (v4){
            return Tuple16{v0, v1};
        } else {
            return Tuple16{v2, v3};
        }
    }
};
struct Closure4 {
    __device__ Tuple16 operator()(Tuple16 tup0, Tuple16 tup1){
        float v0 = tup0.v0; int v1 = tup0.v1; float v2 = tup1.v0; int v3 = tup1.v1;
        return method_47(v0, v1, v2, v3);
    }
};
struct Tuple17 {
    Union1 v0;
    float v1;
    __device__ Tuple17() = default;
    __device__ Tuple17(Union1 t0, float t1) : v0(t0), v1(t1) {}
};
struct Tuple18 {
    int v1;
    bool v0;
    __device__ Tuple18() = default;
    __device__ Tuple18(bool t0, int t1) : v0(t0), v1(t1) {}
};
struct Tuple19 {
    int v0;
    int v1;
    int v2;
    __device__ Tuple19() = default;
    __device__ Tuple19(int t0, int t1, int t2) : v0(t0), v1(t1), v2(t2) {}
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
struct Tuple20 {
    int v0;
    int v1;
    unsigned char v2;
    __device__ Tuple20() = default;
    __device__ Tuple20(int t0, int t1, unsigned char t2) : v0(t0), v1(t1), v2(t2) {}
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
struct Tuple21 {
    Union11 v1;
    int v0;
    __device__ Tuple21() = default;
    __device__ Tuple21(int t0, Union11 t1) : v0(t0), v1(t1) {}
};
struct Tuple22 {
    int v0;
    int v1;
    int v2;
    unsigned char v3;
    __device__ Tuple22() = default;
    __device__ Tuple22(int t0, int t1, int t2, unsigned char t3) : v0(t0), v1(t1), v2(t2), v3(t3) {}
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
            asm("exit;");
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
            asm("exit;");
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
            asm("exit;");
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
            asm("exit;");
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
            asm("exit;");
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
            asm("exit;");
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
            asm("exit;");
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
            asm("exit;");
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
            asm("exit;");
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
            asm("exit;");
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
            asm("exit;");
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
            assert("Invalid tag." && false);
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
            assert("Invalid tag." && false);
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
            assert("Invalid tag." && false);
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
            assert("Invalid tag." && false);
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
    Tuple14 tmp28 = Tuple14{0l, v3};
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
    Tuple14 tmp29 = Tuple14{0l, v3};
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
__device__ inline bool while_method_10(int v0){
    bool v1;
    v1 = v0 < 128l;
    return v1;
}
__device__ inline bool while_method_11(int v0){
    bool v1;
    v1 = v0 < 256l;
    return v1;
}
__device__ void method_45(float * v0, int v1, float * v2, int v3, float * v4, int v5){
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
            assert("Tensor range check" && 0 <= v67 && v67 < 128l);
            int v69;
            v69 = 16l * v67;
            int v70;
            v70 = v69 + v3;
            int v71;
            v71 = 32768l * v65;
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
                v83 = v71 + v5;
                assert("Tensor range check" && 0 <= v81 && v81 < 256l);
                int v84;
                v84 = 8l * v81;
                int v85;
                v85 = v84 + v83;
                float * v86;
                v86 = v4+v85;
                assert("Tensor range check" && 0 <= v67 && v67 < 128l);
                int v88;
                v88 = 32768l * v67;
                int v89;
                v89 = v88 + v1;
                assert("Tensor range check" && 0 <= v81 && v81 < 256l);
                int v90;
                v90 = v84 + v89;
                float * v91;
                v91 = v0+v90;
                int v93;
                v93 = threadIdx.x;
                bool v94;
                v94 = 0l <= v93;
                bool v95;
                v95 = v94 == false;
                if (v95){
                    assert("The index needs to be zero or positive." && v94);
                } else {
                }
                int v97;
                v97 = v93 % 2l;
                int v98;
                v98 = v93 / 2l;
                bool v99;
                v99 = v98 < 16l;
                bool v100;
                v100 = v99 == false;
                if (v100){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v99);
                } else {
                }
                assert("Tensor range check" && 0 <= v98 && v98 < 16l);
                assert("Tensor range check" && 0 <= v97 && v97 < 2l);
                int v102;
                v102 = 4l * v97;
                int v103;
                v103 = 12l * v98;
                int v104;
                v104 = v103 + v102;
                int v105;
                v105 = 2048l * v98;
                int v106;
                v106 = v105 + v102;
                float * v107;
                v107 = v14+v104;
                float * v109;
                v109 = v91+v106;
                int v111;
                v111 = 0l;
                #pragma unroll
                while (while_method_6(v111)){
                    int v113;
                    v113 = 0l;
                    #pragma unroll
                    while (while_method_6(v113)){
                        assert("Tensor range check" && 0 <= v111 && v111 < 1l);
                        assert("Tensor range check" && 0 <= v113 && v113 < 1l);
                        int v115;
                        v115 = 8l * v113;
                        int v116;
                        v116 = 192l * v111;
                        int v117;
                        v117 = v116 + v115;
                        int v118;
                        v118 = 32768l * v111;
                        int v119;
                        v119 = v118 + v115;
                        float v120[4l];
                        int v121;
                        v121 = 0l;
                        #pragma unroll
                        while (while_method_3(v121)){
                            assert("Tensor range check" && 0 <= v121 && v121 < 4l);
                            int v123;
                            v123 = v121 + v119;
                            float v124;
                            v124 = v109[v123];
                            float v125;
                            v125 = wmma::__float_to_tf32(v124);
                            assert("Tensor range check" && 0 <= v121 && v121 < 4l);
                            v120[v121] = v125;
                            v121 += 1l ;
                        }
                        int4* v126;
                        v126 = reinterpret_cast<int4*>(v120 + 0l);
                        int4* v127;
                        v127 = reinterpret_cast<int4*>(v107 + v117);
                        assert("Pointer alignment check" && (unsigned long long)(v126) % 4l == 0 && (unsigned long long)(v127) % 4l == 0);
                        *v127 = *v126;
                        v113 += 1l ;
                    }
                    v111 += 1l ;
                }
                int v128;
                v128 = threadIdx.x;
                bool v129;
                v129 = 0l <= v128;
                bool v130;
                v130 = v129 == false;
                if (v130){
                    assert("The index needs to be zero or positive." && v129);
                } else {
                }
                int v132;
                v132 = v128 % 2l;
                int v133;
                v133 = v128 / 2l;
                bool v134;
                v134 = v133 < 16l;
                bool v135;
                v135 = v134 == false;
                if (v135){
                    assert("The last element of the projection dimensions needs to be greater than the index remainder." && v134);
                } else {
                }
                assert("Tensor range check" && 0 <= v133 && v133 < 16l);
                assert("Tensor range check" && 0 <= v132 && v132 < 2l);
                int v137;
                v137 = 4l * v132;
                int v138;
                v138 = 12l * v133;
                int v139;
                v139 = v138 + v137;
                int v140;
                v140 = 2048l * v133;
                int v141;
                v141 = v140 + v137;
                float * v142;
                v142 = v12+v139;
                float * v144;
                v144 = v86+v141;
                int v146;
                v146 = 0l;
                #pragma unroll
                while (while_method_6(v146)){
                    int v148;
                    v148 = 0l;
                    #pragma unroll
                    while (while_method_6(v148)){
                        assert("Tensor range check" && 0 <= v146 && v146 < 1l);
                        assert("Tensor range check" && 0 <= v148 && v148 < 1l);
                        int v150;
                        v150 = 8l * v148;
                        int v151;
                        v151 = 192l * v146;
                        int v152;
                        v152 = v151 + v150;
                        int v153;
                        v153 = 32768l * v146;
                        int v154;
                        v154 = v153 + v150;
                        float v155[4l];
                        int v156;
                        v156 = 0l;
                        #pragma unroll
                        while (while_method_3(v156)){
                            assert("Tensor range check" && 0 <= v156 && v156 < 4l);
                            int v158;
                            v158 = v156 + v154;
                            float v159;
                            v159 = v144[v158];
                            float v160;
                            v160 = wmma::__float_to_tf32(v159);
                            assert("Tensor range check" && 0 <= v156 && v156 < 4l);
                            v155[v156] = v160;
                            v156 += 1l ;
                        }
                        int4* v161;
                        v161 = reinterpret_cast<int4*>(v155 + 0l);
                        int4* v162;
                        v162 = reinterpret_cast<int4*>(v142 + v152);
                        assert("Pointer alignment check" && (unsigned long long)(v161) % 4l == 0 && (unsigned long long)(v162) % 4l == 0);
                        *v162 = *v161;
                        v148 += 1l ;
                    }
                    v146 += 1l ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0l));
                wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> v163[1l];
                wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> v164[1l];
                int v165;
                v165 = 0l;
                #pragma unroll
                while (while_method_6(v165)){
                    int v167;
                    v167 = 0l;
                    #pragma unroll
                    while (while_method_6(v167)){
                        assert("Tensor range check" && 0 <= v165 && v165 < 1l);
                        assert("Tensor range check" && 0 <= v167 && v167 < 1l);
                        int v169;
                        v169 = v165 + v167;
                        wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v170 = v163[v169];
                        assert("Tensor range check" && 0 <= v165 && v165 < 1l);
                        int v171;
                        v171 = 192l * v165;
                        assert("Tensor range check" && 0 <= v167 && v167 < 1l);
                        int v172;
                        v172 = 8l * v167;
                        int v173;
                        v173 = v172 + v171;
                        int v174;
                        v174 = 0l;
                        #pragma unroll
                        while (while_method_0(v174)){
                            int v176;
                            v176 = 0l;
                            #pragma unroll
                            while (while_method_0(v176)){
                                assert("Tensor range check" && 0 <= v174 && v174 < 2l);
                                assert("Tensor range check" && 0 <= v176 && v176 < 2l);
                                int v178;
                                v178 = 96l * v176;
                                int v179;
                                v179 = v178 + v173;
                                int v180;
                                v180 = 4l * v174;
                                int v181;
                                v181 = v180 + v179;
                                float v182;
                                v182 = v46[v181];
                                bool v183;
                                v183 = 0l <= v176;
                                bool v185;
                                if (v183){
                                    bool v184;
                                    v184 = v176 < 2l;
                                    v185 = v184;
                                } else {
                                    v185 = false;
                                }
                                bool v186;
                                v186 = v185 == false;
                                if (v186){
                                    assert("The indices should be inside the range of the dimension." && v185);
                                } else {
                                }
                                bool v188;
                                v188 = 0l <= v174;
                                bool v190;
                                if (v188){
                                    bool v189;
                                    v189 = v174 < 2l;
                                    v190 = v189;
                                } else {
                                    v190 = false;
                                }
                                bool v191;
                                v191 = v190 == false;
                                if (v191){
                                    assert("The indices should be inside the range of the dimension." && v190);
                                } else {
                                }
                                int v193;
                                v193 = v174 * 2l;
                                int v194;
                                v194 = v176 + v193;
                                v170.x[v194] = v182;
                                v176 += 1l ;
                            }
                            v174 += 1l ;
                        }
                        v167 += 1l ;
                    }
                    v165 += 1l ;
                }
                int v195;
                v195 = 0l;
                #pragma unroll
                while (while_method_6(v195)){
                    int v197;
                    v197 = 0l;
                    #pragma unroll
                    while (while_method_6(v197)){
                        assert("Tensor range check" && 0 <= v195 && v195 < 1l);
                        assert("Tensor range check" && 0 <= v197 && v197 < 1l);
                        int v199;
                        v199 = v195 + v197;
                        wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v200 = v164[v199];
                        assert("Tensor range check" && 0 <= v195 && v195 < 1l);
                        int v201;
                        v201 = 192l * v195;
                        assert("Tensor range check" && 0 <= v197 && v197 < 1l);
                        int v202;
                        v202 = 8l * v197;
                        int v203;
                        v203 = v202 + v201;
                        int v204;
                        v204 = 0l;
                        #pragma unroll
                        while (while_method_0(v204)){
                            int v206;
                            v206 = 0l;
                            #pragma unroll
                            while (while_method_0(v206)){
                                assert("Tensor range check" && 0 <= v204 && v204 < 2l);
                                assert("Tensor range check" && 0 <= v206 && v206 < 2l);
                                int v208;
                                v208 = 4l * v206;
                                int v209;
                                v209 = v208 + v203;
                                int v210;
                                v210 = 96l * v204;
                                int v211;
                                v211 = v210 + v209;
                                float v212;
                                v212 = v62[v211];
                                bool v213;
                                v213 = 0l <= v206;
                                bool v215;
                                if (v213){
                                    bool v214;
                                    v214 = v206 < 2l;
                                    v215 = v214;
                                } else {
                                    v215 = false;
                                }
                                bool v216;
                                v216 = v215 == false;
                                if (v216){
                                    assert("The indices should be inside the range of the dimension." && v215);
                                } else {
                                }
                                bool v218;
                                v218 = 0l <= v204;
                                bool v220;
                                if (v218){
                                    bool v219;
                                    v219 = v204 < 2l;
                                    v220 = v219;
                                } else {
                                    v220 = false;
                                }
                                bool v221;
                                v221 = v220 == false;
                                if (v221){
                                    assert("The indices should be inside the range of the dimension." && v220);
                                } else {
                                }
                                int v223;
                                v223 = v204 * 2l;
                                int v224;
                                v224 = v206 + v223;
                                v200.x[v224] = v212;
                                v206 += 1l ;
                            }
                            v204 += 1l ;
                        }
                        v197 += 1l ;
                    }
                    v195 += 1l ;
                }
                asm("barrier.cta.sync %0;" :: "r"(0l));
                int v225;
                v225 = 0l;
                #pragma unroll
                while (while_method_6(v225)){
                    int v227;
                    v227 = 0l;
                    #pragma unroll
                    while (while_method_6(v227)){
                        int v229;
                        v229 = 0l;
                        #pragma unroll
                        while (while_method_6(v229)){
                            assert("Tensor range check" && 0 <= v225 && v225 < 1l);
                            assert("Tensor range check" && 0 <= v227 && v227 < 1l);
                            int v231;
                            v231 = v225 + v227;
                            wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v232 = v64[v231];
                            assert("Tensor range check" && 0 <= v225 && v225 < 1l);
                            assert("Tensor range check" && 0 <= v229 && v229 < 1l);
                            int v233;
                            v233 = v225 + v229;
                            wmma::fragment<wmma::matrix_a, 16l, 16l, 8l, wmma::precision::tf32, wmma::row_major> & v234 = v163[v233];
                            assert("Tensor range check" && 0 <= v227 && v227 < 1l);
                            assert("Tensor range check" && 0 <= v229 && v229 < 1l);
                            int v235;
                            v235 = v227 + v229;
                            wmma::fragment<wmma::matrix_b, 16l, 16l, 8l, wmma::precision::tf32, wmma::col_major> & v236 = v164[v235];
                            wmma::mma_sync(v232, v234, v236, v232);
                            v229 += 1l ;
                        }
                        v227 += 1l ;
                    }
                    v225 += 1l ;
                }
                v81 += 1l ;
            }
            int v237;
            v237 = 0l;
            #pragma unroll
            while (while_method_6(v237)){
                int v239;
                v239 = 0l;
                #pragma unroll
                while (while_method_6(v239)){
                    assert("Tensor range check" && 0 <= v237 && v237 < 1l);
                    assert("Tensor range check" && 0 <= v239 && v239 < 1l);
                    int v241;
                    v241 = v237 + v239;
                    wmma::fragment<wmma::accumulator, 16l, 16l, 8l, float> & v242 = v64[v241];
                    assert("Tensor range check" && 0 <= v237 && v237 < 1l);
                    assert("Tensor range check" && 0 <= v239 && v239 < 1l);
                    int v243;
                    v243 = 16l * v239;
                    int v244;
                    v244 = 384l * v237;
                    int v245;
                    v245 = v244 + v243;
                    float * v246;
                    v246 = v30+v245;
                    wmma::store_matrix_sync(v246, v242, 24l, wmma::mem_row_major);
                    v239 += 1l ;
                }
                v237 += 1l ;
            }
            asm("barrier.cta.sync %0;" :: "r"(0l));
            int v248;
            v248 = threadIdx.x;
            bool v249;
            v249 = 0l <= v248;
            bool v250;
            v250 = v249 == false;
            if (v250){
                assert("The index needs to be zero or positive." && v249);
            } else {
            }
            int v252;
            v252 = v248 % 4l;
            int v253;
            v253 = v248 / 4l;
            bool v254;
            v254 = v253 < 8l;
            bool v255;
            v255 = v254 == false;
            if (v255){
                assert("The last element of the projection dimensions needs to be greater than the index remainder." && v254);
            } else {
            }
            assert("Tensor range check" && 0 <= v253 && v253 < 8l);
            assert("Tensor range check" && 0 <= v252 && v252 < 4l);
            int v257;
            v257 = 4l * v252;
            int v258;
            v258 = 2048l * v253;
            int v259;
            v259 = v258 + v257;
            int v260;
            v260 = 24l * v253;
            int v261;
            v261 = v260 + v257;
            float * v262;
            v262 = v73+v259;
            float * v264;
            v264 = v16+v261;
            int v266;
            v266 = 0l;
            #pragma unroll
            while (while_method_0(v266)){
                int v268;
                v268 = 0l;
                #pragma unroll
                while (while_method_6(v268)){
                    assert("Tensor range check" && 0 <= v266 && v266 < 2l);
                    assert("Tensor range check" && 0 <= v268 && v268 < 1l);
                    int v270;
                    v270 = 16l * v268;
                    int v271;
                    v271 = 16384l * v266;
                    int v272;
                    v272 = v271 + v270;
                    int v273;
                    v273 = 192l * v266;
                    int v274;
                    v274 = v273 + v270;
                    int4* v275;
                    v275 = reinterpret_cast<int4*>(v264 + v274);
                    int4* v276;
                    v276 = reinterpret_cast<int4*>(v262 + v272);
                    assert("Pointer alignment check" && (unsigned long long)(v275) % 4l == 0 && (unsigned long long)(v276) % 4l == 0);
                    *v276 = *v275;
                    v268 += 1l ;
                }
                v266 += 1l ;
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
__device__ inline bool while_method_13(int v0){
    bool v1;
    v1 = v0 < 16l;
    return v1;
}
__device__ Tuple16 method_47(float v0, int v1, float v2, int v3){
    bool v4;
    v4 = v1 < v3;
    float v5; int v6; float v7; int v8;
    if (v4){
        v5 = v0; v6 = v1; v7 = v2; v8 = v3;
    } else {
        v5 = v2; v6 = v3; v7 = v0; v8 = v1;
    }
    bool v9;
    v9 = v5 >= 0.0f;
    bool v11;
    if (v9){
        bool v10;
        v10 = v7 >= 0.0f;
        v11 = v10;
    } else {
        v11 = false;
    }
    if (v11){
        bool v12;
        v12 = v5 <= v7;
        if (v12){
            return Tuple16{v5, v6};
        } else {
            return Tuple16{v7, v8};
        }
    } else {
        if (v9){
            return Tuple16{v5, v6};
        } else {
            bool v15;
            v15 = v7 >= 0.0f;
            if (v15){
                return Tuple16{v7, v8};
            } else {
                return Tuple16{v5, v6};
            }
        }
    }
}
__device__ void method_46(int * v0, int v1, float * v2, int v3, float * v4, curandStatePhilox4_32_10_t & v5){
    int v6;
    v6 = blockIdx.x;
    assert("Tensor range check" && 0 <= v6 && v6 < 1l);
    int v7;
    v7 = 65536l * v6;
    int v8;
    v8 = blockIdx.x;
    assert("Tensor range check" && 0 <= v8 && v8 < 1l);
    int v9;
    v9 = 65536l * v8;
    int v10;
    v10 = v9 + v3;
    int v11;
    v11 = blockIdx.x;
    assert("Tensor range check" && 0 <= v11 && v11 < 1l);
    int v12;
    v12 = 32l * v11;
    int v13;
    v13 = v12 + v1;
    int v14;
    v14 = threadIdx.x;
    bool v15;
    v15 = 0l <= v14;
    bool v16;
    v16 = v15 == false;
    if (v16){
        assert("The index needs to be zero or positive." && v15);
    } else {
    }
    int v18;
    v18 = v14 % 32l;
    int v19;
    v19 = v14 / 32l;
    bool v20;
    v20 = v19 < 1l;
    bool v21;
    v21 = v20 == false;
    if (v21){
        assert("The last element of the projection dimensions needs to be greater than the index remainder." && v20);
    } else {
    }
    assert("Tensor range check" && 0 <= v19 && v19 < 1l);
    assert("Tensor range check" && 0 <= v18 && v18 < 32l);
    int v23;
    v23 = 4l * v18;
    int v24;
    v24 = v23 + v7;
    int v25;
    v25 = 2048l * v19;
    int v26;
    v26 = v25 + v24;
    assert("Tensor range check" && 0 <= v19 && v19 < 1l);
    assert("Tensor range check" && 0 <= v18 && v18 < 32l);
    int v27;
    v27 = v23 + v10;
    int v28;
    v28 = v25 + v27;
    assert("Tensor range check" && 0 <= v19 && v19 < 1l);
    int v29;
    v29 = v19 + v13;
    int v30;
    v30 = 0l;
    while (while_method_12(v30)){
        assert("Tensor range check" && 0 <= v30 && v30 < 32l);
        int v32;
        v32 = 2048l * v30;
        int v33;
        v33 = v32 + v26;
        float v34[64l];
        int v35[64l];
        int v36;
        v36 = 0l;
        while (while_method_13(v36)){
            assert("Tensor range check" && 0 <= v36 && v36 < 16l);
            int v38;
            v38 = 4l * v36;
            assert("Tensor range check" && 0 <= v36 && v36 < 16l);
            int v39;
            v39 = 128l * v36;
            int v40;
            v40 = v39 + v33;
            int4* v41;
            v41 = reinterpret_cast<int4*>(v4 + v40);
            int4* v42;
            v42 = reinterpret_cast<int4*>(v34 + v38);
            assert("Pointer alignment check" && (unsigned long long)(v41) % 4l == 0 && (unsigned long long)(v42) % 4l == 0);
            *v42 = *v41;
            v36 += 1l ;
        }
        int v43;
        v43 = 0l;
        while (while_method_13(v43)){
            int v45;
            v45 = 0l;
            while (while_method_3(v45)){
                bool v47;
                v47 = 0l <= v45;
                bool v49;
                if (v47){
                    bool v48;
                    v48 = v45 < 4l;
                    v49 = v48;
                } else {
                    v49 = false;
                }
                bool v50;
                v50 = v49 == false;
                if (v50){
                    assert("The indices should be inside the range of the dimension." && v49);
                } else {
                }
                bool v52;
                v52 = 0l <= v18;
                bool v54;
                if (v52){
                    bool v53;
                    v53 = v18 < 32l;
                    v54 = v53;
                } else {
                    v54 = false;
                }
                bool v55;
                v55 = v54 == false;
                if (v55){
                    assert("The indices should be inside the range of the dimension." && v54);
                } else {
                }
                int v57;
                v57 = v18 * 4l;
                int v58;
                v58 = v45 + v57;
                bool v59;
                v59 = 0l <= v43;
                bool v61;
                if (v59){
                    bool v60;
                    v60 = v43 < 16l;
                    v61 = v60;
                } else {
                    v61 = false;
                }
                bool v62;
                v62 = v61 == false;
                if (v62){
                    assert("The indices should be inside the range of the dimension." && v61);
                } else {
                }
                int v64;
                v64 = v43 * 128l;
                int v65;
                v65 = v58 + v64;
                assert("Tensor range check" && 0 <= v43 && v43 < 16l);
                assert("Tensor range check" && 0 <= v45 && v45 < 4l);
                int v66;
                v66 = 4l * v43;
                int v67;
                v67 = v66 + v45;
                v35[v67] = v65;
                v45 += 1l ;
            }
            v43 += 1l ;
        }
        bool v68;
        v68 = 0l <= v19;
        bool v69;
        v69 = v68 && v20;
        bool v70;
        v70 = v69 == false;
        if (v70){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v69);
        } else {
        }
        bool v72;
        v72 = 0l <= v30;
        bool v74;
        if (v72){
            bool v73;
            v73 = v30 < 32l;
            v74 = v73;
        } else {
            v74 = false;
        }
        bool v75;
        v75 = v74 == false;
        if (v75){
            assert("The rigid merge indices have to be greater than or equal to 0 and less than the dimensions." && v74);
        } else {
        }
        int v77;
        v77 = v30 + v19;
        bool v78[64l];
        int v79;
        v79 = 0l;
        while (while_method_13(v79)){
            int v81;
            v81 = 0l;
            while (while_method_3(v81)){
                assert("Tensor range check" && 0 <= v79 && v79 < 16l);
                assert("Tensor range check" && 0 <= v81 && v81 < 4l);
                int v83;
                v83 = 4l * v79;
                int v84;
                v84 = v83 + v81;
                float v85;
                v85 = v34[v84];
                int v86;
                v86 = v35[v84];
                bool v87;
                v87 = v86 < 11l;
                assert("Tensor range check" && 0 <= v79 && v79 < 16l);
                assert("Tensor range check" && 0 <= v81 && v81 < 4l);
                v78[v84] = v87;
                v81 += 1l ;
            }
            v79 += 1l ;
        }
        int v88[64l];
        int v89;
        v89 = 0l;
        while (while_method_13(v89)){
            int v91;
            v91 = 0l;
            while (while_method_3(v91)){
                assert("Tensor range check" && 0 <= v89 && v89 < 16l);
                assert("Tensor range check" && 0 <= v91 && v91 < 4l);
                int v93;
                v93 = 4l * v89;
                int v94;
                v94 = v93 + v91;
                bool v95;
                v95 = v78[v94];
                int v96;
                if (v95){
                    v96 = 1l;
                } else {
                    v96 = 0l;
                }
                assert("Tensor range check" && 0 <= v89 && v89 < 16l);
                assert("Tensor range check" && 0 <= v91 && v91 < 4l);
                v88[v94] = v96;
                v91 += 1l ;
            }
            v89 += 1l ;
        }
        int v97;
        v97 = 0l;
        int v98;
        v98 = 0l;
        while (while_method_13(v98)){
            int v100;
            v100 = 0l;
            while (while_method_3(v100)){
                assert("Tensor range check" && 0 <= v98 && v98 < 16l);
                assert("Tensor range check" && 0 <= v100 && v100 < 4l);
                int v102;
                v102 = 4l * v98;
                int v103;
                v103 = v102 + v100;
                int v104;
                v104 = v88[v103];
                int v105;
                v105 = v97 + v104;
                v97 = v105;
                v100 += 1l ;
            }
            v98 += 1l ;
        }
        auto v106 = cooperative_groups::coalesced_threads();
        int v107;
        v107 = threadIdx.x;
        int v108;
        v108 = v107 / 32l;
        auto v109 = cooperative_groups::labeled_partition(v106,v108);
        Closure0 v110{};
        int v111;
        v111 = cooperative_groups::reduce(v109, v97, v110);
        int v112;
        v112 = threadIdx.x;
        cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v113 = console_lock;
        auto v114 = cooperative_groups::coalesced_threads();
        v113.acquire();
        printf("{%s = %d; %s = %d; %s = %d}\n","result", v97, "tid", v112, "x", v111);
        v113.release();
        v114.sync() ;
        float v119[64l];
        int v120;
        v120 = 0l;
        while (while_method_13(v120)){
            int v122;
            v122 = 0l;
            while (while_method_3(v122)){
                assert("Tensor range check" && 0 <= v120 && v120 < 16l);
                assert("Tensor range check" && 0 <= v122 && v122 < 4l);
                int v124;
                v124 = 4l * v120;
                int v125;
                v125 = v124 + v122;
                float v126;
                v126 = v34[v125];
                bool v127;
                v127 = v78[v125];
                float v128;
                if (v127){
                    v128 = v126;
                } else {
                    v128 = 0.0f;
                }
                assert("Tensor range check" && 0 <= v120 && v120 < 16l);
                assert("Tensor range check" && 0 <= v122 && v122 < 4l);
                v119[v125] = v128;
                v122 += 1l ;
            }
            v120 += 1l ;
        }
        float v129;
        v129 = 0.0f;
        int v130;
        v130 = 0l;
        while (while_method_13(v130)){
            int v132;
            v132 = 0l;
            while (while_method_3(v132)){
                assert("Tensor range check" && 0 <= v130 && v130 < 16l);
                assert("Tensor range check" && 0 <= v132 && v132 < 4l);
                int v134;
                v134 = 4l * v130;
                int v135;
                v135 = v134 + v132;
                float v136;
                v136 = v119[v135];
                float v137;
                v137 = v129 + v136;
                v129 = v137;
                v132 += 1l ;
            }
            v130 += 1l ;
        }
        auto v138 = cooperative_groups::coalesced_threads();
        int v139;
        v139 = threadIdx.x;
        int v140;
        v140 = v139 / 32l;
        auto v141 = cooperative_groups::labeled_partition(v138,v140);
        Closure1 v142{};
        float v143;
        v143 = cooperative_groups::reduce(v141, v129, v142);
        int v144;
        v144 = threadIdx.x;
        cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v145 = console_lock;
        auto v146 = cooperative_groups::coalesced_threads();
        v145.acquire();
        printf("{%s = %f; %s = %d; %s = %f}\n","result", v129, "tid", v144, "x", v143);
        v145.release();
        v146.sync() ;
        float v151;
        v151 = (float)v111;
        float v152;
        v152 = v143 / v151;
        float v153[64l];
        int v154;
        v154 = 0l;
        while (while_method_13(v154)){
            int v156;
            v156 = 0l;
            while (while_method_3(v156)){
                assert("Tensor range check" && 0 <= v154 && v154 < 16l);
                assert("Tensor range check" && 0 <= v156 && v156 < 4l);
                int v158;
                v158 = 4l * v154;
                int v159;
                v159 = v158 + v156;
                float v160;
                v160 = v34[v159];
                bool v161;
                v161 = v78[v159];
                float v162;
                if (v161){
                    v162 = v160;
                } else {
                    v162 = -1.0f / 0.0f;
                }
                float v163;
                v163 = v162 - v152;
                float v164;
                v164 = exp(v163);
                assert("Tensor range check" && 0 <= v154 && v154 < 16l);
                assert("Tensor range check" && 0 <= v156 && v156 < 4l);
                v153[v159] = v164;
                v156 += 1l ;
            }
            v154 += 1l ;
        }
        float v165;
        v165 = 0.0f;
        int v166;
        v166 = 0l;
        while (while_method_13(v166)){
            int v168;
            v168 = 0l;
            while (while_method_3(v168)){
                assert("Tensor range check" && 0 <= v166 && v166 < 16l);
                assert("Tensor range check" && 0 <= v168 && v168 < 4l);
                int v170;
                v170 = 4l * v166;
                int v171;
                v171 = v170 + v168;
                float v172;
                v172 = v153[v171];
                float v173;
                v173 = v165 + v172;
                v165 = v173;
                v168 += 1l ;
            }
            v166 += 1l ;
        }
        auto v174 = cooperative_groups::coalesced_threads();
        int v175;
        v175 = threadIdx.x;
        int v176;
        v176 = v175 / 32l;
        auto v177 = cooperative_groups::labeled_partition(v174,v176);
        float v178;
        v178 = cooperative_groups::reduce(v177, v165, v142);
        int v179;
        v179 = threadIdx.x;
        cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v180 = console_lock;
        auto v181 = cooperative_groups::coalesced_threads();
        v180.acquire();
        printf("{%s = %f; %s = %d; %s = %f}\n","result", v165, "tid", v179, "x", v178);
        v180.release();
        v181.sync() ;
        float v186[64l];
        int v187;
        v187 = 0l;
        while (while_method_13(v187)){
            int v189;
            v189 = 0l;
            while (while_method_3(v189)){
                assert("Tensor range check" && 0 <= v187 && v187 < 16l);
                assert("Tensor range check" && 0 <= v189 && v189 < 4l);
                int v191;
                v191 = 4l * v187;
                int v192;
                v192 = v191 + v189;
                float v193;
                v193 = v153[v192];
                bool v194;
                v194 = v178 == 0.0f;
                bool v195;
                v195 = v194 != true;
                float v197;
                if (v195){
                    float v196;
                    v196 = v193 / v178;
                    v197 = v196;
                } else {
                    v197 = 0.00048828125f;
                }
                assert("Tensor range check" && 0 <= v187 && v187 < 16l);
                assert("Tensor range check" && 0 <= v189 && v189 < 4l);
                v186[v192] = v197;
                v189 += 1l ;
            }
            v187 += 1l ;
        }
        float v198[64l];
        float v199;
        v199 = 0.0f;
        int v200;
        v200 = 0l;
        while (while_method_13(v200)){
            assert("Tensor range check" && 0 <= v200 && v200 < 16l);
            int v202;
            v202 = 4l * v200;
            assert("Tensor range check" && 0 <= v200 && v200 < 16l);
            int v203; float v204;
            Tuple15 tmp30 = Tuple15{0l, 0.0f};
            v203 = tmp30.v0; v204 = tmp30.v1;
            while (while_method_3(v203)){
                assert("Tensor range check" && 0 <= v203 && v203 < 4l);
                int v206;
                v206 = v203 + v202;
                float v207;
                v207 = v186[v206];
                float v208;
                v208 = v204 + v207;
                v204 = v208;
                v203 += 1l ;
            }
            auto v209 = cooperative_groups::coalesced_threads();
            int v210;
            v210 = threadIdx.x;
            int v211;
            v211 = v210 / 32l;
            auto v212 = cooperative_groups::labeled_partition(v209,v211);
            Closure2 v213{};
            float v214;
            v214 = cooperative_groups::inclusive_scan(v212, v204, v213);
            float v215;
            v215 = v212.shfl_up(v214,1);
            bool v216;
            v216 = v212.thread_rank() == 0;
            float v217;
            if (v216){
                v217 = 0.0f;
            } else {
                v217 = v215;
            }
            float v218;
            v218 = v212.shfl(v214,v212.num_threads()-1);
            float v219;
            v219 = v199 + v217;
            int v220; float v221;
            Tuple15 tmp31 = Tuple15{0l, v219};
            v220 = tmp31.v0; v221 = tmp31.v1;
            while (while_method_3(v220)){
                assert("Tensor range check" && 0 <= v220 && v220 < 4l);
                int v223;
                v223 = v220 + v202;
                float v224;
                v224 = v186[v223];
                float v225;
                v225 = v221 + v224;
                assert("Tensor range check" && 0 <= v220 && v220 < 4l);
                v198[v223] = v225;
                v221 = v225;
                v220 += 1l ;
            }
            float v226;
            v226 = v199 + v218;
            v199 = v226;
            v200 += 1l ;
        }
        float v227[64l];
        int v228[64l];
        int v229;
        v229 = 0l;
        while (while_method_13(v229)){
            int v231;
            v231 = 0l;
            while (while_method_3(v231)){
                assert("Tensor range check" && 0 <= v229 && v229 < 16l);
                assert("Tensor range check" && 0 <= v231 && v231 < 4l);
                int v233;
                v233 = 4l * v229;
                int v234;
                v234 = v233 + v231;
                int v235;
                v235 = v35[v234];
                float v236;
                v236 = curand_uniform(&v5);
                assert("Tensor range check" && 0 <= v229 && v229 < 16l);
                assert("Tensor range check" && 0 <= v231 && v231 < 4l);
                v227[v234] = v236;
                v228[v234] = v235;
                v231 += 1l ;
            }
            v229 += 1l ;
        }
        float v237; int v238;
        Tuple16 tmp32 = Tuple16{0.0f, 2147483647l};
        v237 = tmp32.v0; v238 = tmp32.v1;
        int v239;
        v239 = 0l;
        while (while_method_13(v239)){
            int v241;
            v241 = 0l;
            while (while_method_3(v241)){
                assert("Tensor range check" && 0 <= v239 && v239 < 16l);
                assert("Tensor range check" && 0 <= v241 && v241 < 4l);
                int v243;
                v243 = 4l * v239;
                int v244;
                v244 = v243 + v241;
                float v245;
                v245 = v227[v244];
                int v246;
                v246 = v228[v244];
                bool v247;
                v247 = v238 < v246;
                float v248; int v249;
                if (v247){
                    v248 = v237; v249 = v238;
                } else {
                    v248 = v245; v249 = v246;
                }
                v237 = v248;
                v238 = v249;
                v241 += 1l ;
            }
            v239 += 1l ;
        }
        auto v250 = cooperative_groups::coalesced_threads();
        int v251;
        v251 = threadIdx.x;
        int v252;
        v252 = v251 / 32l;
        auto v253 = cooperative_groups::labeled_partition(v250,v252);
        Closure3 v254{};
        float v255; int v256;
        Tuple16 tmp33 = cooperative_groups::reduce(v253, Tuple16{v237, v238}, v254);
        v255 = tmp33.v0; v256 = tmp33.v1;
        int v257;
        v257 = threadIdx.x;
        cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v258 = console_lock;
        auto v259 = cooperative_groups::coalesced_threads();
        v258.acquire();
        printf("{%s = %f, %d; %s = %d; %s = %f, %d}\n","result", v237, v238, "tid", v257, "x", v255, v256);
        v258.release();
        v259.sync() ;
        float v264[64l];
        int v265;
        v265 = 0l;
        while (while_method_13(v265)){
            int v267;
            v267 = 0l;
            while (while_method_3(v267)){
                assert("Tensor range check" && 0 <= v265 && v265 < 16l);
                assert("Tensor range check" && 0 <= v267 && v267 < 4l);
                int v269;
                v269 = 4l * v265;
                int v270;
                v270 = v269 + v267;
                float v271;
                v271 = v198[v270];
                float v272;
                v272 = v271 - v255;
                assert("Tensor range check" && 0 <= v265 && v265 < 16l);
                assert("Tensor range check" && 0 <= v267 && v267 < 4l);
                v264[v270] = v272;
                v267 += 1l ;
            }
            v265 += 1l ;
        }
        float v273; int v274;
        Tuple16 tmp34 = Tuple16{-1.0f / 0.0f, 2147483647l};
        v273 = tmp34.v0; v274 = tmp34.v1;
        int v275;
        v275 = 0l;
        while (while_method_13(v275)){
            int v277;
            v277 = 0l;
            while (while_method_3(v277)){
                assert("Tensor range check" && 0 <= v275 && v275 < 16l);
                assert("Tensor range check" && 0 <= v277 && v277 < 4l);
                int v279;
                v279 = 4l * v275;
                int v280;
                v280 = v279 + v277;
                float v281;
                v281 = v264[v280];
                int v282;
                v282 = v35[v280];
                float v283; int v284;
                Tuple16 tmp35 = method_47(v273, v274, v281, v282);
                v283 = tmp35.v0; v284 = tmp35.v1;
                v273 = v283;
                v274 = v284;
                v277 += 1l ;
            }
            v275 += 1l ;
        }
        auto v285 = cooperative_groups::coalesced_threads();
        int v286;
        v286 = threadIdx.x;
        int v287;
        v287 = v286 / 32l;
        auto v288 = cooperative_groups::labeled_partition(v285,v287);
        Closure4 v289{};
        float v290; int v291;
        Tuple16 tmp36 = cooperative_groups::reduce(v288, Tuple16{v273, v274}, v289);
        v290 = tmp36.v0; v291 = tmp36.v1;
        int v292;
        v292 = threadIdx.x;
        cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v293 = console_lock;
        auto v294 = cooperative_groups::coalesced_threads();
        v293.acquire();
        printf("{%s = %f, %d; %s = %d; %s = %f, %d}\n","result", v273, v274, "tid", v292, "x", v290, v291);
        v293.release();
        v294.sync() ;
        bool v299;
        v299 = v291 < 11l;
        bool v300;
        v300 = v299 == false;
        if (v300){
            int v301;
            v301 = threadIdx.x;
            cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v302 = console_lock;
            auto v303 = cooperative_groups::coalesced_threads();
            v302.acquire();
            int v304;
            v304 = 0l;
            printf("{%s = %d; %s = %d; %s = %c","output_id", v291, "tid", v301, "x", '[');
            int v305;
            v305 = 0l;
            while (while_method_13(v305)){
                int v307;
                v307 = v304;
                bool v308;
                v308 = v307 >= 100l;
                if (v308){
                    printf("%s"," ...");
                    break;
                } else {
                }
                bool v309;
                v309 = v305 == 0l;
                bool v310;
                v310 = v309 != true;
                if (v310){
                    printf("%s","; ");
                } else {
                }
                printf("%c",'[');
                int v311;
                v311 = 0l;
                while (while_method_3(v311)){
                    int v313;
                    v313 = v304;
                    bool v314;
                    v314 = v313 >= 100l;
                    if (v314){
                        printf("%s"," ...");
                        break;
                    } else {
                    }
                    bool v315;
                    v315 = v311 == 0l;
                    bool v316;
                    v316 = v315 != true;
                    if (v316){
                        printf("%s","; ");
                    } else {
                    }
                    int v317;
                    v317 = v304 + 1l;
                    v304 = v317;
                    int v318;
                    v318 = v305 * 4l;
                    int v319;
                    v319 = v318 + v311;
                    float v320;
                    v320 = v264[v319];
                    printf("%f",v320);
                    v311 += 1l ;
                }
                printf("%c",']');
                v305 += 1l ;
            }
            printf("%c",']');
            printf("}\n");
            v302.release();
            v303.sync() ;
        } else {
        }
        if (v300){
            assert("The masking requirement is violated in masked_softmax_and_discrete_sample_." && v299);
        } else {
        }
        assert("Tensor range check" && 0 <= v30 && v30 < 32l);
        int v353;
        v353 = v32 + v28;
        int v354;
        v354 = 0l;
        while (while_method_13(v354)){
            assert("Tensor range check" && 0 <= v354 && v354 < 16l);
            int v356;
            v356 = 128l * v354;
            int v357;
            v357 = v356 + v353;
            assert("Tensor range check" && 0 <= v354 && v354 < 16l);
            int v358;
            v358 = 4l * v354;
            int4* v359;
            v359 = reinterpret_cast<int4*>(v186 + v358);
            int4* v360;
            v360 = reinterpret_cast<int4*>(v2 + v357);
            assert("Pointer alignment check" && (unsigned long long)(v359) % 4l == 0 && (unsigned long long)(v360) % 4l == 0);
            *v360 = *v359;
            v354 += 1l ;
        }
        assert("Tensor range check" && 0 <= v30 && v30 < 32l);
        int v361;
        v361 = v30 + v29;
        v0[v361] = v291;
        v30 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    return ;
}
__device__ int int_range_48(int v0, int v1, curandStatePhilox4_32_10_t & v2){
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
__device__ __noinline__ Tuple13 noinline_run_42(unsigned char * v0, unsigned char * v1, Union8 v2){
    asm("barrier.cta.sync %0;" :: "r"(0l));
    float * v3;
    v3 = reinterpret_cast<float *>(&v0[524288ull]);
    float * v5;
    v5 = reinterpret_cast<float *>(&v0[0ull]);
    int * v7;
    v7 = reinterpret_cast<int *>(&v0[1572864ull]);
    unsigned long long v9;
    v9 = clock64();
    int v10;
    v10 = threadIdx.x;
    int v11;
    v11 = blockIdx.x;
    int v12;
    v12 = v11 * 32l;
    int v13;
    v13 = v10 + v12;
    unsigned long long v14;
    v14 = (unsigned long long)v13;
    curandStatePhilox4_32_10_t v15;
    curand_init(v9,v14,0ull,&v15);
    float * v16;
    v16 = reinterpret_cast<float *>(&v0[0ull]);
    int v18;
    v18 = blockIdx.x;
    assert("Tensor range check" && 0 <= v18 && v18 < 1l);
    int v19;
    v19 = 65536l * v18;
    unsigned long long v20;
    v20 = clock64();
    int v21;
    v21 = threadIdx.x;
    int v22;
    v22 = blockIdx.x;
    int v23;
    v23 = v22 * 32l;
    int v24;
    v24 = v21 + v23;
    unsigned long long v25;
    v25 = (unsigned long long)v24;
    curandStatePhilox4_32_10_t v26;
    curand_init(v20,v25,0ull,&v26);
    int v27;
    v27 = threadIdx.x;
    int v28;
    v28 = v27;
    while (while_method_7(v28)){
        bool v30;
        v30 = 0l <= v28;
        bool v31;
        v31 = v30 == false;
        if (v31){
            assert("The index needs to be zero or positive." && v30);
        } else {
        }
        int v33;
        v33 = v28 % 2048l;
        int v34;
        v34 = v28 / 2048l;
        bool v35;
        v35 = v34 < 32l;
        bool v36;
        v36 = v35 == false;
        if (v36){
            assert("The last element of the projection dimensions needs to be greater than the index remainder." && v35);
        } else {
        }
        assert("Tensor range check" && 0 <= v34 && v34 < 32l);
        assert("Tensor range check" && 0 <= v33 && v33 < 2048l);
        int v38;
        v38 = v33 + v19;
        int v39;
        v39 = 2048l * v34;
        int v40;
        v40 = v39 + v38;
        v16[v40] = 0.0f;
        v28 += 32l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    switch (v2.tag) {
        case 0: { // None
            break;
        }
        case 1: { // Some
            int v41 = v2.case1.v0; static_array<static_array<unsigned char,2l>,2l> v42 = v2.case1.v1; static_array<int,2l> v43 = v2.case1.v2; int v44 = v2.case1.v3; static_array<int,2l> v45 = v2.case1.v4; Union5 v46 = v2.case1.v5; dynamic_array_list<Union6,128l> v47 = v2.case1.v6;
            int v48;
            v48 = threadIdx.x;
            assert("Tensor range check" && 0 <= v48 && v48 < 32l);
            int v49;
            v49 = 2048l * v48;
            int v50;
            v50 = v49 + v19;
            int v51;
            v51 = v47.length_();
            bool v52;
            v52 = 128l >= v51;
            bool v53;
            v53 = v52 == false;
            if (v53){
                assert("The type level dimension has to equal the value passed at runtime into create." && v52);
            } else {
            }
            dynamic_array_list<Union10,128l> v55{0};
            v55.unsafe_set_length(v51);
            int v57;
            v57 = 0l;
            while (while_method_4(v51, v57)){
                Union6 v59;
                v59 = v47[v57];
                Union10 v65;
                switch (v59.tag) {
                    case 2: { // PlayerAction
                        int v61 = v59.case2.v0; Union1 v62 = v59.case2.v1;
                        v65 = Union10{Union10_1{v62}};
                        break;
                    }
                    default: {
                        v65 = Union10{Union10_0{}};
                    }
                }
                v55[v57] = v65;
                v57 += 1l ;
            }
            static_array<int,2l> v66;
            int v68;
            v68 = 0l;
            while (while_method_0(v68)){
                int v70;
                v70 = v44 % 2l;
                int v71;
                v71 = v68 + v70;
                int v72;
                v72 = v43[v71];
                v66[v68] = v72;
                v68 += 1l ;
            }
            static_array<int,2l> v74;
            int v76;
            v76 = 0l;
            while (while_method_0(v76)){
                int v78;
                v78 = v44 % 2l;
                int v79;
                v79 = v76 + v78;
                int v80;
                v80 = v45[v79];
                v74[v76] = v80;
                v76 += 1l ;
            }
            int v82;
            v82 = v44 % 2l;
            static_array<unsigned char,2l> v83;
            v83 = v42[v82];
            static_array_list<unsigned char,5l> v85;
            v85 = static_array_list<unsigned char,5l>{};
            switch (v46.tag) {
                case 0: { // Flop
                    static_array<unsigned char,3l> v87 = v46.case0.v0;
                    int v88;
                    v88 = 0l;
                    while (while_method_1(v88)){
                        unsigned char v90;
                        v90 = v87[v88];
                        v85.push(v90);
                        v88 += 1l ;
                    }
                    break;
                }
                case 1: { // Preflop
                    break;
                }
                case 2: { // River
                    static_array<unsigned char,5l> v97 = v46.case2.v0;
                    int v98;
                    v98 = 0l;
                    while (while_method_2(v98)){
                        unsigned char v100;
                        v100 = v97[v98];
                        v85.push(v100);
                        v98 += 1l ;
                    }
                    break;
                }
                case 3: { // Turn
                    static_array<unsigned char,4l> v92 = v46.case3.v0;
                    int v93;
                    v93 = 0l;
                    while (while_method_3(v93)){
                        unsigned char v95;
                        v95 = v92[v93];
                        v85.push(v95);
                        v93 += 1l ;
                    }
                    break;
                }
                default: {
                    assert("Invalid tag." && false);
                }
            }
            float * v102;
            v102 = v16+v50;
            int v104;
            v104 = v55.length_();
            bool v105;
            v105 = v104 == 0l;
            if (v105){
                v102[0l] = 1.0f;
            } else {
            }
            int v106;
            v106 = v55.length_();
            int v107;
            v107 = 0l;
            while (while_method_4(v106, v107)){
                Union10 v109;
                v109 = v55[v107];
                int v111;
                v111 = v107 * 14l;
                int v112;
                v112 = 1l + v111;
                switch (v109.tag) {
                    case 0: { // None
                        v102[v112] = 1.0f;
                        break;
                    }
                    case 1: { // Some
                        Union1 v113 = v109.case1.v0;
                        int v114;
                        v114 = v112 + 1l;
                        switch (v113.tag) {
                            case 0: { // A_All_In
                                v102[v114] = 1.0f;
                                break;
                            }
                            case 1: { // A_Call
                                int v115;
                                v115 = v114 + 1l;
                                v102[v115] = 1.0f;
                                break;
                            }
                            case 2: { // A_Fold
                                int v116;
                                v116 = v114 + 2l;
                                v102[v116] = 1.0f;
                                break;
                            }
                            case 3: { // A_Raise
                                int v117 = v113.case3.v0;
                                int v118;
                                v118 = v114 + 3l;
                                bool v119;
                                v119 = 1l <= v117;
                                bool v121;
                                if (v119){
                                    bool v120;
                                    v120 = v117 < 1023l;
                                    v121 = v120;
                                } else {
                                    v121 = false;
                                }
                                bool v122;
                                v122 = v121 == false;
                                if (v122){
                                    assert("Pickle failure. The input is out of the bounds of the given range." && v121);
                                } else {
                                }
                                int v124;
                                v124 = v117 - 1l;
                                unsigned int v125;
                                v125 = (unsigned int)v124;
                                method_43(v125, v102, v118);
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                            }
                        }
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                    }
                }
                v107 += 1l ;
            }
            int v126;
            v126 = 0l;
            while (while_method_0(v126)){
                int v128;
                v128 = v66[v126];
                int v130;
                v130 = v126 * 11l;
                int v131;
                v131 = 1794l + v130;
                bool v132;
                v132 = 0l <= v128;
                bool v134;
                if (v132){
                    bool v133;
                    v133 = v128 < 1023l;
                    v134 = v133;
                } else {
                    v134 = false;
                }
                bool v135;
                v135 = v134 == false;
                if (v135){
                    assert("Pickle failure. The input is out of the bounds of the given range." && v134);
                } else {
                }
                unsigned int v137;
                v137 = (unsigned int)v128;
                method_44(v137, v102, v131);
                v126 += 1l ;
            }
            int v138;
            v138 = 0l;
            while (while_method_0(v138)){
                int v140;
                v140 = v74[v138];
                int v142;
                v142 = v138 * 11l;
                int v143;
                v143 = 1817l + v142;
                bool v144;
                v144 = 0l <= v140;
                bool v146;
                if (v144){
                    bool v145;
                    v145 = v140 < 1023l;
                    v146 = v145;
                } else {
                    v146 = false;
                }
                bool v147;
                v147 = v146 == false;
                if (v147){
                    assert("Pickle failure. The input is out of the bounds of the given range." && v146);
                } else {
                }
                unsigned int v149;
                v149 = (unsigned int)v140;
                method_44(v149, v102, v143);
                v138 += 1l ;
            }
            int v150;
            v150 = 0l;
            while (while_method_0(v150)){
                unsigned char v152;
                v152 = v83[v150];
                int v154;
                v154 = v150 * 17l;
                int v155;
                v155 = 1840l + v154;
                unsigned char v156;
                v156 = v152 % 4u;
                int v157;
                v157 = (int)v156;
                unsigned char v158;
                v158 = v152 / 4u;
                int v159;
                v159 = (int)v158;
                unsigned int v160;
                v160 = (unsigned int)v157;
                int v161;
                v161 = (int)v160;
                bool v162;
                v162 = v161 < 4l;
                bool v163;
                v163 = v162 == false;
                if (v163){
                    assert("Pickle failure. Int value out of bounds." && v162);
                } else {
                }
                int v165;
                v165 = v155 + v161;
                v102[v165] = 1.0f;
                int v166;
                v166 = v155 + 4l;
                unsigned int v167;
                v167 = (unsigned int)v159;
                int v168;
                v168 = (int)v167;
                bool v169;
                v169 = v168 < 13l;
                bool v170;
                v170 = v169 == false;
                if (v170){
                    assert("Pickle failure. Int value out of bounds." && v169);
                } else {
                }
                int v172;
                v172 = v166 + v168;
                v102[v172] = 1.0f;
                v150 += 1l ;
            }
            int v173;
            v173 = v85.length;
            bool v174;
            v174 = v173 == 0l;
            if (v174){
                v102[1874l] = 1.0f;
            } else {
            }
            int v175;
            v175 = v85.length;
            int v176;
            v176 = 0l;
            while (while_method_4(v175, v176)){
                unsigned char v178;
                v178 = v85[v176];
                int v180;
                v180 = v176 * 17l;
                int v181;
                v181 = 1875l + v180;
                unsigned char v182;
                v182 = v178 % 4u;
                int v183;
                v183 = (int)v182;
                unsigned char v184;
                v184 = v178 / 4u;
                int v185;
                v185 = (int)v184;
                unsigned int v186;
                v186 = (unsigned int)v183;
                int v187;
                v187 = (int)v186;
                bool v188;
                v188 = v187 < 4l;
                bool v189;
                v189 = v188 == false;
                if (v189){
                    assert("Pickle failure. Int value out of bounds." && v188);
                } else {
                }
                int v191;
                v191 = v181 + v187;
                v102[v191] = 1.0f;
                int v192;
                v192 = v181 + 4l;
                unsigned int v193;
                v193 = (unsigned int)v185;
                int v194;
                v194 = (int)v193;
                bool v195;
                v195 = v194 < 13l;
                bool v196;
                v196 = v195 == false;
                if (v196){
                    assert("Pickle failure. Int value out of bounds." && v195);
                } else {
                }
                int v198;
                v198 = v192 + v194;
                v102[v198] = 1.0f;
                v176 += 1l ;
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v199;
    v199 = 0l;
    while (while_method_3(v199)){
        float * v201;
        v201 = reinterpret_cast<float *>(&v0[0ull]);
        float * v203;
        v203 = reinterpret_cast<float *>(&v1[0ull]);
        assert("Tensor range check" && 0 <= v199 && v199 < 4l);
        int v205;
        v205 = 4194304l * v199;
        float * v206;
        v206 = reinterpret_cast<float *>(&v0[262144ull]);
        int v208;
        v208 = blockIdx.x;
        assert("Tensor range check" && 0 <= v208 && v208 < 1l);
        int v209;
        v209 = 65536l * v208;
        int v210;
        v210 = blockIdx.x;
        assert("Tensor range check" && 0 <= v210 && v210 < 1l);
        int v211;
        v211 = 65536l * v210;
        method_45(v203, v205, v206, v211, v201, v209);
        float * v212;
        v212 = reinterpret_cast<float *>(&v0[524288ull]);
        assert("Tensor range check" && 0 <= v199 && v199 < 4l);
        int v214;
        v214 = 65536l * v199;
        int * v215;
        v215 = reinterpret_cast<int *>(&v0[1572864ull]);
        assert("Tensor range check" && 0 <= v199 && v199 < 4l);
        int v217;
        v217 = 32l * v199;
        method_46(v215, v217, v212, v214, v206, v26);
        v199 += 1l ;
    }
    asm("barrier.cta.sync %0;" :: "r"(0l));
    int v218;
    v218 = 0l;
    int v219;
    v219 = 4l;
    int v220;
    v220 = int_range_48(v219, v218, v26);
    float * v221;
    v221 = reinterpret_cast<float *>(&v0[524288ull]);
    int * v223;
    v223 = reinterpret_cast<int *>(&v0[1572864ull]);
    assert("Tensor range check" && 0 <= v220 && v220 < 4l);
    int v225;
    v225 = 32l * v220;
    int v226;
    v226 = blockIdx.x;
    assert("Tensor range check" && 0 <= v226 && v226 < 1l);
    int v227;
    v227 = 32l * v226;
    int v228;
    v228 = v227 + v225;
    int v229;
    v229 = threadIdx.x;
    assert("Tensor range check" && 0 <= v229 && v229 < 32l);
    int v230;
    v230 = v229 + v228;
    int v231;
    v231 = v223[v230];
    bool v232;
    v232 = 0l == v231;
    Union9 v265;
    if (v232){
        v265 = Union9{Union9_1{}};
    } else {
        bool v234;
        v234 = 1l == v231;
        if (v234){
            v265 = Union9{Union9_0{}};
        } else {
            bool v236;
            v236 = 2l == v231;
            if (v236){
                v265 = Union9{Union9_2{1l, 3l}};
            } else {
                bool v238;
                v238 = 3l == v231;
                if (v238){
                    v265 = Union9{Union9_2{1l, 2l}};
                } else {
                    bool v240;
                    v240 = 4l == v231;
                    if (v240){
                        v265 = Union9{Union9_2{2l, 3l}};
                    } else {
                        bool v242;
                        v242 = 5l == v231;
                        if (v242){
                            v265 = Union9{Union9_2{3l, 4l}};
                        } else {
                            bool v244;
                            v244 = 6l == v231;
                            if (v244){
                                v265 = Union9{Union9_2{1l, 1l}};
                            } else {
                                bool v246;
                                v246 = 7l == v231;
                                if (v246){
                                    v265 = Union9{Union9_2{3l, 2l}};
                                } else {
                                    bool v248;
                                    v248 = 8l == v231;
                                    if (v248){
                                        v265 = Union9{Union9_2{2l, 1l}};
                                    } else {
                                        bool v250;
                                        v250 = 9l == v231;
                                        if (v250){
                                            v265 = Union9{Union9_2{3l, 1l}};
                                        } else {
                                            bool v252;
                                            v252 = 10l == v231;
                                            if (v252){
                                                v265 = Union9{Union9_2{2147483647l, 1l}};
                                            } else {
                                                printf("%s\n", "Invalid output id in the NL Holdem model.");
                                                asm("exit;");
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
    int v266;
    v266 = blockIdx.x;
    int v267;
    v267 = threadIdx.x;
    assert("Tensor range check" && 0 <= v220 && v220 < 4l);
    assert("Tensor range check" && 0 <= v266 && v266 < 1l);
    assert("Tensor range check" && 0 <= v267 && v267 < 32l);
    assert("Tensor range check" && 0 <= v231 && v231 < 2048l);
    int v268;
    v268 = 2048l * v267;
    int v269;
    v269 = v268 + v231;
    int v270;
    v270 = 65536l * v266;
    int v271;
    v271 = v270 + v269;
    int v272;
    v272 = 65536l * v220;
    int v273;
    v273 = v272 + v271;
    float v274;
    v274 = v221[v273];
    int v275;
    v275 = blockIdx.x;
    assert("Tensor range check" && 0 <= v275 && v275 < 1l);
    int v276;
    v276 = 65536l * v275;
    int v277;
    v277 = threadIdx.x;
    assert("Tensor range check" && 0 <= v277 && v277 < 32l);
    int v278;
    v278 = 2048l * v277;
    int v279;
    v279 = v278 + v276;
    assert("Tensor range check" && 0 <= v231 && v231 < 2048l);
    int v280;
    v280 = v231 + v279;
    return Tuple13{v265, v221, v280, 65536l, 4l, v274};
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
            assert("Invalid tag." && false);
        }
    }
}
__device__ inline bool while_method_14(int v0){
    bool v1;
    v1 = v0 < 6l;
    return v1;
}
__device__ inline bool while_method_15(static_array<float,6l> v0, int v1){
    bool v2;
    v2 = v1 < 6l;
    return v2;
}
__device__ inline bool while_method_16(int v0, int v1){
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
    while (while_method_14(v4)){
        float v6;
        v6 = v0[v4];
        v2[v4] = v6;
        v4 += 1l ;
    }
    int v8;
    v8 = 1l;
    while (while_method_15(v2, v8)){
        int v10;
        v10 = 6l;
        while (while_method_16(v8, v10)){
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
__device__ Union1 sample_discrete_50(static_array<Tuple17,6l> v0, curandStatePhilox4_32_10_t & v1){
    static_array<float,6l> v2;
    int v4;
    v4 = 0l;
    while (while_method_14(v4)){
        Union1 v6; float v7;
        Tuple17 tmp46 = v0[v4];
        v6 = tmp46.v0; v7 = tmp46.v1;
        v2[v4] = v7;
        v4 += 1l ;
    }
    int v10;
    v10 = sample_discrete__51(v2, v1);
    Union1 v11; float v12;
    Tuple17 tmp47 = v0[v10];
    v11 = tmp47.v0; v12 = tmp47.v1;
    return v11;
}
__device__ inline bool while_method_17(int v0){
    bool v1;
    v1 = v0 < 7l;
    return v1;
}
__device__ inline bool while_method_18(static_array<unsigned char,7l> v0, bool v1, int v2){
    bool v3;
    v3 = v2 < 7l;
    return v3;
}
__device__ inline bool while_method_19(static_array<unsigned char,7l> v0, int v1){
    bool v2;
    v2 = v1 < 7l;
    return v2;
}
__device__ inline bool while_method_20(int v0, int v1, int v2, int v3){
    bool v4;
    v4 = v3 < v0;
    return v4;
}
__device__ Tuple0 score_54(static_array<unsigned char,7l> v0){
    static_array<unsigned char,7l> v1;
    int v3;
    v3 = 0l;
    while (while_method_17(v3)){
        unsigned char v5;
        v5 = v0[v3];
        v1[v3] = v5;
        v3 += 1l ;
    }
    static_array<unsigned char,7l> v7;
    bool v9; int v10;
    Tuple18 tmp54 = Tuple18{true, 1l};
    v9 = tmp54.v0; v10 = tmp54.v1;
    while (while_method_18(v1, v9, v10)){
        int v12;
        v12 = 0l;
        while (while_method_19(v1, v12)){
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
            Tuple19 tmp55 = Tuple19{v12, v16, v12};
            v21 = tmp55.v0; v22 = tmp55.v1; v23 = tmp55.v2;
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
    Tuple20 tmp56 = Tuple20{0l, 0l, 12u};
    v87 = tmp56.v0; v88 = tmp56.v1; v89 = tmp56.v2;
    while (while_method_17(v87)){
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
    Tuple20 tmp57 = Tuple20{0l, 0l, 12u};
    v155 = tmp57.v0; v156 = tmp57.v1; v157 = tmp57.v2;
    while (while_method_17(v155)){
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
                    Tuple21 tmp58 = Tuple21{0l, v223};
                    v224 = tmp58.v0; v225 = tmp58.v1;
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
                    assert("Invalid tag." && false);
                }
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    static_array<unsigned char,5l> v249;
    int v251; int v252; unsigned char v253;
    Tuple20 tmp59 = Tuple20{0l, 0l, 12u};
    v251 = tmp59.v0; v252 = tmp59.v1; v253 = tmp59.v2;
    while (while_method_17(v251)){
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
                    Tuple21 tmp60 = Tuple21{0l, v319};
                    v320 = tmp60.v0; v321 = tmp60.v1;
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
                    assert("Invalid tag." && false);
                }
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    static_array<unsigned char,5l> v345;
    int v347; int v348; unsigned char v349;
    Tuple20 tmp61 = Tuple20{0l, 0l, 12u};
    v347 = tmp61.v0; v348 = tmp61.v1; v349 = tmp61.v2;
    while (while_method_17(v347)){
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
                    Tuple21 tmp62 = Tuple21{0l, v415};
                    v416 = tmp62.v0; v417 = tmp62.v1;
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
                    assert("Invalid tag." && false);
                }
            }
            break;
        }
        default: {
            assert("Invalid tag." && false);
        }
    }
    static_array<unsigned char,5l> v1037; char v1038;
    switch (v440.tag) {
        case 0: { // None
            static_array<unsigned char,4l> v442;
            static_array<unsigned char,3l> v444;
            int v446; int v447; int v448; unsigned char v449;
            Tuple22 tmp63 = Tuple22{0l, 0l, 0l, 12u};
            v446 = tmp63.v0; v447 = tmp63.v1; v448 = tmp63.v2; v449 = tmp63.v3;
            while (while_method_17(v446)){
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
                    assert("Invalid tag." && false);
                }
            }
            switch (v498.tag) {
                case 0: { // None
                    static_array<unsigned char,3l> v500;
                    static_array<unsigned char,4l> v502;
                    int v504; int v505; int v506; unsigned char v507;
                    Tuple22 tmp64 = Tuple22{0l, 0l, 0l, 12u};
                    v504 = tmp64.v0; v505 = tmp64.v1; v506 = tmp64.v2; v507 = tmp64.v3;
                    while (while_method_17(v504)){
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
                            Tuple22 tmp65 = Tuple22{0l, 0l, 0l, 12u};
                            v540 = tmp65.v0; v541 = tmp65.v1; v542 = tmp65.v2; v543 = tmp65.v3;
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
                                    assert("Invalid tag." && false);
                                }
                            }
                            break;
                        }
                        default: {
                            assert("Invalid tag." && false);
                        }
                    }
                    switch (v589.tag) {
                        case 0: { // None
                            static_array<unsigned char,5l> v591;
                            int v593; int v594;
                            Tuple4 tmp66 = Tuple4{0l, 0l};
                            v593 = tmp66.v0; v594 = tmp66.v1;
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
                            Tuple4 tmp67 = Tuple4{0l, 0l};
                            v610 = tmp67.v0; v611 = tmp67.v1;
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
                                            Tuple21 tmp68 = Tuple21{0l, v627};
                                            v628 = tmp68.v0; v629 = tmp68.v1;
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
                                            assert("Invalid tag." && false);
                                        }
                                    }
                                    break;
                                }
                                default: {
                                    assert("Invalid tag." && false);
                                }
                            }
                            static_array<unsigned char,5l> v653;
                            int v655; int v656;
                            Tuple4 tmp69 = Tuple4{0l, 0l};
                            v655 = tmp69.v0; v656 = tmp69.v1;
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
                                            Tuple21 tmp70 = Tuple21{0l, v672};
                                            v673 = tmp70.v0; v674 = tmp70.v1;
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
                                            assert("Invalid tag." && false);
                                        }
                                    }
                                    break;
                                }
                                default: {
                                    assert("Invalid tag." && false);
                                }
                            }
                            static_array<unsigned char,5l> v698;
                            int v700; int v701;
                            Tuple4 tmp71 = Tuple4{0l, 0l};
                            v700 = tmp71.v0; v701 = tmp71.v1;
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
                                            Tuple21 tmp72 = Tuple21{0l, v717};
                                            v718 = tmp72.v0; v719 = tmp72.v1;
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
                                            assert("Invalid tag." && false);
                                        }
                                    }
                                    break;
                                }
                                default: {
                                    assert("Invalid tag." && false);
                                }
                            }
                            switch (v742.tag) {
                                case 0: { // None
                                    static_array<unsigned char,5l> v744;
                                    int v746; int v747; unsigned char v748;
                                    Tuple20 tmp73 = Tuple20{0l, 0l, 12u};
                                    v746 = tmp73.v0; v747 = tmp73.v1; v748 = tmp73.v2;
                                    while (while_method_17(v746)){
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
                                            Tuple22 tmp74 = Tuple22{0l, 0l, 0l, 12u};
                                            v787 = tmp74.v0; v788 = tmp74.v1; v789 = tmp74.v2; v790 = tmp74.v3;
                                            while (while_method_17(v787)){
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
                                                    assert("Invalid tag." && false);
                                                }
                                            }
                                            switch (v839.tag) {
                                                case 0: { // None
                                                    static_array<unsigned char,2l> v841;
                                                    static_array<unsigned char,5l> v843;
                                                    int v845; int v846; int v847; unsigned char v848;
                                                    Tuple22 tmp75 = Tuple22{0l, 0l, 0l, 12u};
                                                    v845 = tmp75.v0; v846 = tmp75.v1; v847 = tmp75.v2; v848 = tmp75.v3;
                                                    while (while_method_17(v845)){
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
                                                            Tuple22 tmp76 = Tuple22{0l, 0l, 0l, 12u};
                                                            v881 = tmp76.v0; v882 = tmp76.v1; v883 = tmp76.v2; v884 = tmp76.v3;
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
                                                                    assert("Invalid tag." && false);
                                                                }
                                                            }
                                                            break;
                                                        }
                                                        default: {
                                                            assert("Invalid tag." && false);
                                                        }
                                                    }
                                                    switch (v941.tag) {
                                                        case 0: { // None
                                                            static_array<unsigned char,2l> v943;
                                                            static_array<unsigned char,5l> v945;
                                                            int v947; int v948; int v949; unsigned char v950;
                                                            Tuple22 tmp77 = Tuple22{0l, 0l, 0l, 12u};
                                                            v947 = tmp77.v0; v948 = tmp77.v1; v949 = tmp77.v2; v950 = tmp77.v3;
                                                            while (while_method_17(v947)){
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
                                                                    assert("Invalid tag." && false);
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
                                                                    assert("Invalid tag." && false);
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
                                                            assert("Invalid tag." && false);
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
                                                    assert("Invalid tag." && false);
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
                                            assert("Invalid tag." && false);
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
                                    assert("Invalid tag." && false);
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
                            assert("Invalid tag." && false);
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
                    assert("Invalid tag." && false);
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
            assert("Invalid tag." && false);
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
        Union3 v1001;
        switch (v14.tag) {
            case 0: { // None
                v1001 = Union3{Union3_0{}};
                break;
            }
            case 1: { // Some
                Union4 v16 = v14.case1.v0;
                switch (v16.tag) {
                    case 0: { // G_Flop
                        int v894 = v16.case0.v0; static_array<static_array<unsigned char,2l>,2l> v895 = v16.case0.v1; static_array<int,2l> v896 = v16.case0.v2; int v897 = v16.case0.v3; static_array<int,2l> v898 = v16.case0.v4; Union5 v899 = v16.case0.v5;
                        static_array<unsigned char,3l> v900; unsigned long long v901;
                        Tuple8 tmp18 = draw_cards_32(v0, v12);
                        v900 = tmp18.v0; v901 = tmp18.v1;
                        v5 = v901;
                        static_array_list<unsigned char,5l> v902;
                        v902 = get_community_cards_35(v899, v900);
                        Union6 v903;
                        v903 = Union6{Union6_0{v902}};
                        v11.push(v903);
                        Union5 v906;
                        switch (v899.tag) {
                            case 1: { // Preflop
                                v906 = Union5{Union5_0{v900}};
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in flop.");
                                asm("exit;");
                            }
                        }
                        int v907;
                        v907 = 2l;
                        int v908;
                        v908 = 0l;
                        Union4 v909;
                        v909 = try_round_36(v907, v895, v896, v908, v898, v906);
                        v1001 = Union3{Union3_1{v909}};
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
                        v1001 = Union3{Union3_0{}};
                        break;
                    }
                    case 2: { // G_Preflop
                        static_array<unsigned char,2l> v969; unsigned long long v970;
                        Tuple11 tmp23 = draw_cards_39(v0, v12);
                        v969 = tmp23.v0; v970 = tmp23.v1;
                        v5 = v970;
                        static_array<unsigned char,2l> v971; unsigned long long v972;
                        Tuple11 tmp24 = draw_cards_39(v0, v12);
                        v971 = tmp24.v0; v972 = tmp24.v1;
                        v5 = v972;
                        Union6 v973;
                        v973 = Union6{Union6_3{0l, v969}};
                        v11.push(v973);
                        Union6 v974;
                        v974 = Union6{Union6_3{1l, v971}};
                        v11.push(v974);
                        static_array<static_array<unsigned char,2l>,2l> v975;
                        v975[0l] = v969;
                        v975[1l] = v971;
                        static_array<int,2l> v977;
                        v977[0l] = 2l;
                        v977[1l] = 1l;
                        static_array<int,2l> v979;
                        int v981;
                        v981 = 0l;
                        while (while_method_0(v981)){
                            int v983;
                            v983 = v977[v981];
                            int v985;
                            v985 = 100l - v983;
                            v979[v981] = v985;
                            v981 += 1l ;
                        }
                        int v986;
                        v986 = 2l;
                        int v987;
                        v987 = 0l;
                        Union5 v988;
                        v988 = Union5{Union5_1{}};
                        Union4 v989;
                        v989 = try_round_36(v986, v975, v977, v987, v979, v988);
                        v1001 = Union3{Union3_1{v989}};
                        break;
                    }
                    case 3: { // G_River
                        int v940 = v16.case3.v0; static_array<static_array<unsigned char,2l>,2l> v941 = v16.case3.v1; static_array<int,2l> v942 = v16.case3.v2; int v943 = v16.case3.v3; static_array<int,2l> v944 = v16.case3.v4; Union5 v945 = v16.case3.v5;
                        static_array<unsigned char,1l> v946; unsigned long long v947;
                        Tuple12 tmp27 = draw_cards_40(v0, v12);
                        v946 = tmp27.v0; v947 = tmp27.v1;
                        v5 = v947;
                        static_array_list<unsigned char,5l> v948;
                        v948 = get_community_cards_41(v945, v946);
                        Union6 v949;
                        v949 = Union6{Union6_0{v948}};
                        v11.push(v949);
                        Union5 v964;
                        switch (v945.tag) {
                            case 3: { // Turn
                                static_array<unsigned char,4l> v950 = v945.case3.v0;
                                static_array<unsigned char,5l> v951;
                                int v953;
                                v953 = 0l;
                                while (while_method_3(v953)){
                                    unsigned char v955;
                                    v955 = v950[v953];
                                    v951[v953] = v955;
                                    v953 += 1l ;
                                }
                                int v957;
                                v957 = 0l;
                                while (while_method_6(v957)){
                                    unsigned char v959;
                                    v959 = v946[v957];
                                    int v961;
                                    v961 = 4l + v957;
                                    v951[v961] = v959;
                                    v957 += 1l ;
                                }
                                v964 = Union5{Union5_2{v951}};
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in river.");
                                asm("exit;");
                            }
                        }
                        int v965;
                        v965 = 2l;
                        int v966;
                        v966 = 0l;
                        Union4 v967;
                        v967 = try_round_36(v965, v941, v942, v966, v944, v964);
                        v1001 = Union3{Union3_1{v967}};
                        break;
                    }
                    case 4: { // G_Round
                        int v110 = v16.case4.v0; static_array<static_array<unsigned char,2l>,2l> v111 = v16.case4.v1; static_array<int,2l> v112 = v16.case4.v2; int v113 = v16.case4.v3; static_array<int,2l> v114 = v16.case4.v4; Union5 v115 = v16.case4.v5;
                        int v116;
                        v116 = v113 % 2l;
                        static_array<Union2,2l> v117 = v8;
                        Union2 v118;
                        v118 = v117[v116];
                        switch (v118.tag) {
                            case 0: { // Computer
                                bool v120;
                                v120 = 67108864ull == v4;
                                bool v121;
                                v121 = v120 == false;
                                if (v121){
                                    assert("The params needs to have matching offsets." && v120);
                                } else {
                                }
                                bool v123;
                                v123 = 1573376ull == v2;
                                bool v124;
                                v124 = v123 == false;
                                if (v124){
                                    assert("The outputs needs to have matching offsets." && v123);
                                } else {
                                }
                                dynamic_array_list<Union6,128l> & v126 = v7;
                                cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v127 = console_lock;
                                auto v128 = cooperative_groups::coalesced_threads();
                                v127.acquire();
                                printf("%s\n","Running the GPU model.");
                                v127.release();
                                v128.sync() ;
                                Union8 v131;
                                v131 = Union8{Union8_1{v110, v111, v112, v113, v114, v115, v126}};
                                Union9 v132; float * v133; int v134; int v135; int v136; float v137;
                                Tuple13 tmp37 = noinline_run_42(v1, v3, v131);
                                v132 = tmp37.v0; v133 = tmp37.v1; v134 = tmp37.v2; v135 = tmp37.v3; v136 = tmp37.v4; v137 = tmp37.v5;
                                Union1 v215;
                                switch (v132.tag) {
                                    case 0: { // AA_Call
                                        v215 = Union1{Union1_1{}};
                                        break;
                                    }
                                    case 1: { // AA_Fold
                                        int v138;
                                        v138 = v112[0l];
                                        int v140; int v141;
                                        Tuple4 tmp38 = Tuple4{1l, v138};
                                        v140 = tmp38.v0; v141 = tmp38.v1;
                                        while (while_method_0(v140)){
                                            int v143;
                                            v143 = v112[v140];
                                            bool v145;
                                            v145 = v141 >= v143;
                                            int v146;
                                            if (v145){
                                                v146 = v141;
                                            } else {
                                                v146 = v143;
                                            }
                                            v141 = v146;
                                            v140 += 1l ;
                                        }
                                        int v147;
                                        v147 = v112[v116];
                                        bool v149;
                                        v149 = v147 == v141;
                                        if (v149){
                                            v215 = Union1{Union1_1{}};
                                        } else {
                                            v215 = Union1{Union1_2{}};
                                        }
                                        break;
                                    }
                                    case 2: { // AA_Raise
                                        int v154 = v132.case2.v0; int v155 = v132.case2.v1;
                                        static_array<int,2l> v156;
                                        int v158;
                                        v158 = 0l;
                                        while (while_method_0(v158)){
                                            int v160;
                                            v160 = v114[v158];
                                            int v162;
                                            v162 = v112[v158];
                                            int v164;
                                            v164 = v160 + v162;
                                            v156[v158] = v164;
                                            v158 += 1l ;
                                        }
                                        int v165;
                                        v165 = v112[0l];
                                        int v167; int v168;
                                        Tuple4 tmp39 = Tuple4{1l, v165};
                                        v167 = tmp39.v0; v168 = tmp39.v1;
                                        while (while_method_0(v167)){
                                            int v170;
                                            v170 = v112[v167];
                                            bool v172;
                                            v172 = v168 >= v170;
                                            int v173;
                                            if (v172){
                                                v173 = v168;
                                            } else {
                                                v173 = v170;
                                            }
                                            v168 = v173;
                                            v167 += 1l ;
                                        }
                                        int v174;
                                        v174 = v156[v116];
                                        bool v176;
                                        v176 = v168 < v174;
                                        int v177;
                                        if (v176){
                                            v177 = v168;
                                        } else {
                                            v177 = v174;
                                        }
                                        static_array<int,2l> v178;
                                        int v180;
                                        v180 = 0l;
                                        while (while_method_0(v180)){
                                            int v182;
                                            v182 = v112[v180];
                                            bool v184;
                                            v184 = v116 == v180;
                                            int v185;
                                            if (v184){
                                                v185 = v177;
                                            } else {
                                                v185 = v182;
                                            }
                                            v178[v180] = v185;
                                            v180 += 1l ;
                                        }
                                        int v186;
                                        v186 = v178[0l];
                                        int v188; int v189;
                                        Tuple4 tmp40 = Tuple4{1l, v186};
                                        v188 = tmp40.v0; v189 = tmp40.v1;
                                        while (while_method_0(v188)){
                                            int v191;
                                            v191 = v178[v188];
                                            int v193;
                                            v193 = v189 + v191;
                                            v189 = v193;
                                            v188 += 1l ;
                                        }
                                        static_array<int,2l> v194;
                                        int v196;
                                        v196 = 0l;
                                        while (while_method_0(v196)){
                                            int v198;
                                            v198 = v156[v196];
                                            int v200;
                                            v200 = v178[v196];
                                            int v202;
                                            v202 = v198 - v200;
                                            v194[v196] = v202;
                                            v196 += 1l ;
                                        }
                                        int v203;
                                        v203 = v154 * v189;
                                        int v204;
                                        v204 = v203 / v155;
                                        bool v205;
                                        v205 = v110 >= v204;
                                        int v206;
                                        if (v205){
                                            v206 = v110;
                                        } else {
                                            v206 = v204;
                                        }
                                        int v207;
                                        v207 = v194[v116];
                                        bool v209;
                                        v209 = v206 >= v207;
                                        if (v209){
                                            v215 = Union1{Union1_0{}};
                                        } else {
                                            v215 = Union1{Union1_3{v206}};
                                        }
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false);
                                    }
                                }
                                cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v216 = console_lock;
                                auto v217 = cooperative_groups::coalesced_threads();
                                v216.acquire();
                                printf("%s","The action is: ");
                                v216.release();
                                v217.sync() ;
                                cuda::counting_semaphore<cuda::thread_scope_system, 1l> & v220 = console_lock;
                                auto v221 = cooperative_groups::coalesced_threads();
                                v220.acquire();
                                printf("");
                                method_49(v215);
                                printf("\n");
                                v220.release();
                                v221.sync() ;
                                Union6 v224;
                                v224 = Union6{Union6_2{v116, v215}};
                                v11.push(v224);
                                Union4 v412;
                                switch (v215.tag) {
                                    case 0: { // A_All_In
                                        static_array<int,2l> v342;
                                        int v344;
                                        v344 = 0l;
                                        while (while_method_0(v344)){
                                            int v346;
                                            v346 = v114[v344];
                                            int v348;
                                            v348 = v112[v344];
                                            int v350;
                                            v350 = v346 + v348;
                                            v342[v344] = v350;
                                            v344 += 1l ;
                                        }
                                        int v351;
                                        v351 = v112[0l];
                                        int v353; int v354;
                                        Tuple4 tmp41 = Tuple4{1l, v351};
                                        v353 = tmp41.v0; v354 = tmp41.v1;
                                        while (while_method_0(v353)){
                                            int v356;
                                            v356 = v112[v353];
                                            bool v358;
                                            v358 = v354 >= v356;
                                            int v359;
                                            if (v358){
                                                v359 = v354;
                                            } else {
                                                v359 = v356;
                                            }
                                            v354 = v359;
                                            v353 += 1l ;
                                        }
                                        int v360;
                                        v360 = v342[v116];
                                        bool v362;
                                        v362 = v354 < v360;
                                        int v363;
                                        if (v362){
                                            v363 = v354;
                                        } else {
                                            v363 = v360;
                                        }
                                        static_array<int,2l> v364;
                                        int v366;
                                        v366 = 0l;
                                        while (while_method_0(v366)){
                                            int v368;
                                            v368 = v112[v366];
                                            bool v370;
                                            v370 = v116 == v366;
                                            int v371;
                                            if (v370){
                                                v371 = v363;
                                            } else {
                                                v371 = v368;
                                            }
                                            v364[v366] = v371;
                                            v366 += 1l ;
                                        }
                                        static_array<int,2l> v372;
                                        int v374;
                                        v374 = 0l;
                                        while (while_method_0(v374)){
                                            int v376;
                                            v376 = v342[v374];
                                            int v378;
                                            v378 = v364[v374];
                                            int v380;
                                            v380 = v376 - v378;
                                            v372[v374] = v380;
                                            v374 += 1l ;
                                        }
                                        int v381;
                                        v381 = v372[v116];
                                        int v383;
                                        v383 = v354 + v381;
                                        int v384;
                                        v384 = v342[v116];
                                        bool v386;
                                        v386 = v383 < v384;
                                        int v387;
                                        if (v386){
                                            v387 = v383;
                                        } else {
                                            v387 = v384;
                                        }
                                        static_array<int,2l> v388;
                                        int v390;
                                        v390 = 0l;
                                        while (while_method_0(v390)){
                                            int v392;
                                            v392 = v112[v390];
                                            bool v394;
                                            v394 = v116 == v390;
                                            int v395;
                                            if (v394){
                                                v395 = v387;
                                            } else {
                                                v395 = v392;
                                            }
                                            v388[v390] = v395;
                                            v390 += 1l ;
                                        }
                                        static_array<int,2l> v396;
                                        int v398;
                                        v398 = 0l;
                                        while (while_method_0(v398)){
                                            int v400;
                                            v400 = v342[v398];
                                            int v402;
                                            v402 = v388[v398];
                                            int v404;
                                            v404 = v400 - v402;
                                            v396[v398] = v404;
                                            v398 += 1l ;
                                        }
                                        bool v405;
                                        v405 = v381 >= v110;
                                        int v406;
                                        if (v405){
                                            v406 = v381;
                                        } else {
                                            v406 = v110;
                                        }
                                        int v407;
                                        v407 = v113 + 1l;
                                        v412 = try_round_36(v406, v111, v388, v407, v396, v115);
                                        break;
                                    }
                                    case 1: { // A_Call
                                        static_array<int,2l> v226;
                                        int v228;
                                        v228 = 0l;
                                        while (while_method_0(v228)){
                                            int v230;
                                            v230 = v114[v228];
                                            int v232;
                                            v232 = v112[v228];
                                            int v234;
                                            v234 = v230 + v232;
                                            v226[v228] = v234;
                                            v228 += 1l ;
                                        }
                                        int v235;
                                        v235 = v112[0l];
                                        int v237; int v238;
                                        Tuple4 tmp42 = Tuple4{1l, v235};
                                        v237 = tmp42.v0; v238 = tmp42.v1;
                                        while (while_method_0(v237)){
                                            int v240;
                                            v240 = v112[v237];
                                            bool v242;
                                            v242 = v238 >= v240;
                                            int v243;
                                            if (v242){
                                                v243 = v238;
                                            } else {
                                                v243 = v240;
                                            }
                                            v238 = v243;
                                            v237 += 1l ;
                                        }
                                        int v244;
                                        v244 = v226[v116];
                                        bool v246;
                                        v246 = v238 < v244;
                                        int v247;
                                        if (v246){
                                            v247 = v238;
                                        } else {
                                            v247 = v244;
                                        }
                                        static_array<int,2l> v248;
                                        int v250;
                                        v250 = 0l;
                                        while (while_method_0(v250)){
                                            int v252;
                                            v252 = v112[v250];
                                            bool v254;
                                            v254 = v116 == v250;
                                            int v255;
                                            if (v254){
                                                v255 = v247;
                                            } else {
                                                v255 = v252;
                                            }
                                            v248[v250] = v255;
                                            v250 += 1l ;
                                        }
                                        static_array<int,2l> v256;
                                        int v258;
                                        v258 = 0l;
                                        while (while_method_0(v258)){
                                            int v260;
                                            v260 = v226[v258];
                                            int v262;
                                            v262 = v248[v258];
                                            int v264;
                                            v264 = v260 - v262;
                                            v256[v258] = v264;
                                            v258 += 1l ;
                                        }
                                        bool v265;
                                        v265 = v116 < 2l;
                                        if (v265){
                                            int v266;
                                            v266 = v113 + 1l;
                                            v412 = try_round_36(v110, v111, v248, v266, v256, v115);
                                        } else {
                                            v412 = go_next_street_38(v110, v111, v248, v113, v256, v115);
                                        }
                                        break;
                                    }
                                    case 2: { // A_Fold
                                        v412 = Union4{Union4_1{v110, v111, v112, v113, v114, v115}};
                                        break;
                                    }
                                    case 3: { // A_Raise
                                        int v270 = v215.case3.v0;
                                        bool v271;
                                        v271 = v110 <= v270;
                                        bool v272;
                                        v272 = v271 == false;
                                        if (v272){
                                            assert("The raise amount must match the minimum." && v271);
                                        } else {
                                        }
                                        static_array<int,2l> v274;
                                        int v276;
                                        v276 = 0l;
                                        while (while_method_0(v276)){
                                            int v278;
                                            v278 = v114[v276];
                                            int v280;
                                            v280 = v112[v276];
                                            int v282;
                                            v282 = v278 + v280;
                                            v274[v276] = v282;
                                            v276 += 1l ;
                                        }
                                        int v283;
                                        v283 = v112[0l];
                                        int v285; int v286;
                                        Tuple4 tmp43 = Tuple4{1l, v283};
                                        v285 = tmp43.v0; v286 = tmp43.v1;
                                        while (while_method_0(v285)){
                                            int v288;
                                            v288 = v112[v285];
                                            bool v290;
                                            v290 = v286 >= v288;
                                            int v291;
                                            if (v290){
                                                v291 = v286;
                                            } else {
                                                v291 = v288;
                                            }
                                            v286 = v291;
                                            v285 += 1l ;
                                        }
                                        int v292;
                                        v292 = v274[v116];
                                        bool v294;
                                        v294 = v286 < v292;
                                        int v295;
                                        if (v294){
                                            v295 = v286;
                                        } else {
                                            v295 = v292;
                                        }
                                        static_array<int,2l> v296;
                                        int v298;
                                        v298 = 0l;
                                        while (while_method_0(v298)){
                                            int v300;
                                            v300 = v112[v298];
                                            bool v302;
                                            v302 = v116 == v298;
                                            int v303;
                                            if (v302){
                                                v303 = v295;
                                            } else {
                                                v303 = v300;
                                            }
                                            v296[v298] = v303;
                                            v298 += 1l ;
                                        }
                                        static_array<int,2l> v304;
                                        int v306;
                                        v306 = 0l;
                                        while (while_method_0(v306)){
                                            int v308;
                                            v308 = v274[v306];
                                            int v310;
                                            v310 = v296[v306];
                                            int v312;
                                            v312 = v308 - v310;
                                            v304[v306] = v312;
                                            v306 += 1l ;
                                        }
                                        int v313;
                                        v313 = v304[v116];
                                        bool v315;
                                        v315 = v270 < v313;
                                        bool v316;
                                        v316 = v315 == false;
                                        if (v316){
                                            assert("The raise amount must be less than the stack size after calling." && v315);
                                        } else {
                                        }
                                        int v318;
                                        v318 = v286 + v270;
                                        int v319;
                                        v319 = v274[v116];
                                        bool v321;
                                        v321 = v318 < v319;
                                        int v322;
                                        if (v321){
                                            v322 = v318;
                                        } else {
                                            v322 = v319;
                                        }
                                        static_array<int,2l> v323;
                                        int v325;
                                        v325 = 0l;
                                        while (while_method_0(v325)){
                                            int v327;
                                            v327 = v112[v325];
                                            bool v329;
                                            v329 = v116 == v325;
                                            int v330;
                                            if (v329){
                                                v330 = v322;
                                            } else {
                                                v330 = v327;
                                            }
                                            v323[v325] = v330;
                                            v325 += 1l ;
                                        }
                                        static_array<int,2l> v331;
                                        int v333;
                                        v333 = 0l;
                                        while (while_method_0(v333)){
                                            int v335;
                                            v335 = v274[v333];
                                            int v337;
                                            v337 = v323[v333];
                                            int v339;
                                            v339 = v335 - v337;
                                            v331[v333] = v339;
                                            v333 += 1l ;
                                        }
                                        int v340;
                                        v340 = v113 + 1l;
                                        v412 = try_round_36(v270, v111, v323, v340, v331, v115);
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false);
                                    }
                                }
                                v1001 = Union3{Union3_1{v412}};
                                break;
                            }
                            case 1: { // Human
                                Union7 v414;
                                v414 = Union7{Union7_2{v110, v111, v112, v113, v114, v115}};
                                v9 = v414;
                                Union3 v415;
                                v415 = Union3{Union3_1{v16}};
                                v6 = v415;
                                v1001 = Union3{Union3_0{}};
                                break;
                            }
                            case 2: { // Random
                                static_array<int,2l> v417;
                                int v419;
                                v419 = 0l;
                                while (while_method_0(v419)){
                                    int v421;
                                    v421 = v114[v419];
                                    int v423;
                                    v423 = v112[v419];
                                    int v425;
                                    v425 = v421 + v423;
                                    v417[v419] = v425;
                                    v419 += 1l ;
                                }
                                int v426;
                                v426 = v112[0l];
                                int v428; int v429;
                                Tuple4 tmp44 = Tuple4{1l, v426};
                                v428 = tmp44.v0; v429 = tmp44.v1;
                                while (while_method_0(v428)){
                                    int v431;
                                    v431 = v112[v428];
                                    bool v433;
                                    v433 = v429 >= v431;
                                    int v434;
                                    if (v433){
                                        v434 = v429;
                                    } else {
                                        v434 = v431;
                                    }
                                    v429 = v434;
                                    v428 += 1l ;
                                }
                                int v435;
                                v435 = v417[v116];
                                bool v437;
                                v437 = v429 < v435;
                                int v438;
                                if (v437){
                                    v438 = v429;
                                } else {
                                    v438 = v435;
                                }
                                static_array<int,2l> v439;
                                int v441;
                                v441 = 0l;
                                while (while_method_0(v441)){
                                    int v443;
                                    v443 = v112[v441];
                                    bool v445;
                                    v445 = v116 == v441;
                                    int v446;
                                    if (v445){
                                        v446 = v438;
                                    } else {
                                        v446 = v443;
                                    }
                                    v439[v441] = v446;
                                    v441 += 1l ;
                                }
                                int v447;
                                v447 = v439[0l];
                                int v449; int v450;
                                Tuple4 tmp45 = Tuple4{1l, v447};
                                v449 = tmp45.v0; v450 = tmp45.v1;
                                while (while_method_0(v449)){
                                    int v452;
                                    v452 = v439[v449];
                                    int v454;
                                    v454 = v450 + v452;
                                    v450 = v454;
                                    v449 += 1l ;
                                }
                                static_array<int,2l> v455;
                                int v457;
                                v457 = 0l;
                                while (while_method_0(v457)){
                                    int v459;
                                    v459 = v417[v457];
                                    int v461;
                                    v461 = v439[v457];
                                    int v463;
                                    v463 = v459 - v461;
                                    v455[v457] = v463;
                                    v457 += 1l ;
                                }
                                int v464;
                                v464 = v112[v116];
                                bool v466;
                                v466 = v464 < v429;
                                float v467;
                                if (v466){
                                    v467 = 1.0f;
                                } else {
                                    v467 = 0.0f;
                                }
                                int v468;
                                v468 = v450 / 3l;
                                bool v469;
                                v469 = v110 <= v468;
                                bool v473;
                                if (v469){
                                    int v470;
                                    v470 = v455[v116];
                                    bool v472;
                                    v472 = v468 < v470;
                                    v473 = v472;
                                } else {
                                    v473 = false;
                                }
                                float v474;
                                if (v473){
                                    v474 = 1.0f;
                                } else {
                                    v474 = 0.0f;
                                }
                                int v475;
                                v475 = v450 / 2l;
                                bool v476;
                                v476 = v110 <= v475;
                                bool v480;
                                if (v476){
                                    int v477;
                                    v477 = v455[v116];
                                    bool v479;
                                    v479 = v475 < v477;
                                    v480 = v479;
                                } else {
                                    v480 = false;
                                }
                                float v481;
                                if (v480){
                                    v481 = 1.0f;
                                } else {
                                    v481 = 0.0f;
                                }
                                bool v482;
                                v482 = v110 <= v450;
                                bool v486;
                                if (v482){
                                    int v483;
                                    v483 = v455[v116];
                                    bool v485;
                                    v485 = v450 < v483;
                                    v486 = v485;
                                } else {
                                    v486 = false;
                                }
                                float v487;
                                if (v486){
                                    v487 = 1.0f;
                                } else {
                                    v487 = 0.0f;
                                }
                                static_array<Tuple17,6l> v488;
                                Union1 v490;
                                v490 = Union1{Union1_2{}};
                                v488[0l] = Tuple17{v490, v467};
                                Union1 v492;
                                v492 = Union1{Union1_1{}};
                                v488[1l] = Tuple17{v492, 4.0f};
                                Union1 v494;
                                v494 = Union1{Union1_3{v468}};
                                v488[2l] = Tuple17{v494, v474};
                                Union1 v496;
                                v496 = Union1{Union1_3{v475}};
                                v488[3l] = Tuple17{v496, v481};
                                Union1 v498;
                                v498 = Union1{Union1_3{v450}};
                                v488[4l] = Tuple17{v498, v487};
                                Union1 v500;
                                v500 = Union1{Union1_0{}};
                                v488[5l] = Tuple17{v500, 1.0f};
                                Union1 v502;
                                v502 = sample_discrete_50(v488, v0);
                                Union6 v503;
                                v503 = Union6{Union6_2{v116, v502}};
                                v11.push(v503);
                                Union4 v691;
                                switch (v502.tag) {
                                    case 0: { // A_All_In
                                        static_array<int,2l> v621;
                                        int v623;
                                        v623 = 0l;
                                        while (while_method_0(v623)){
                                            int v625;
                                            v625 = v114[v623];
                                            int v627;
                                            v627 = v112[v623];
                                            int v629;
                                            v629 = v625 + v627;
                                            v621[v623] = v629;
                                            v623 += 1l ;
                                        }
                                        int v630;
                                        v630 = v112[0l];
                                        int v632; int v633;
                                        Tuple4 tmp48 = Tuple4{1l, v630};
                                        v632 = tmp48.v0; v633 = tmp48.v1;
                                        while (while_method_0(v632)){
                                            int v635;
                                            v635 = v112[v632];
                                            bool v637;
                                            v637 = v633 >= v635;
                                            int v638;
                                            if (v637){
                                                v638 = v633;
                                            } else {
                                                v638 = v635;
                                            }
                                            v633 = v638;
                                            v632 += 1l ;
                                        }
                                        int v639;
                                        v639 = v621[v116];
                                        bool v641;
                                        v641 = v633 < v639;
                                        int v642;
                                        if (v641){
                                            v642 = v633;
                                        } else {
                                            v642 = v639;
                                        }
                                        static_array<int,2l> v643;
                                        int v645;
                                        v645 = 0l;
                                        while (while_method_0(v645)){
                                            int v647;
                                            v647 = v112[v645];
                                            bool v649;
                                            v649 = v116 == v645;
                                            int v650;
                                            if (v649){
                                                v650 = v642;
                                            } else {
                                                v650 = v647;
                                            }
                                            v643[v645] = v650;
                                            v645 += 1l ;
                                        }
                                        static_array<int,2l> v651;
                                        int v653;
                                        v653 = 0l;
                                        while (while_method_0(v653)){
                                            int v655;
                                            v655 = v621[v653];
                                            int v657;
                                            v657 = v643[v653];
                                            int v659;
                                            v659 = v655 - v657;
                                            v651[v653] = v659;
                                            v653 += 1l ;
                                        }
                                        int v660;
                                        v660 = v651[v116];
                                        int v662;
                                        v662 = v633 + v660;
                                        int v663;
                                        v663 = v621[v116];
                                        bool v665;
                                        v665 = v662 < v663;
                                        int v666;
                                        if (v665){
                                            v666 = v662;
                                        } else {
                                            v666 = v663;
                                        }
                                        static_array<int,2l> v667;
                                        int v669;
                                        v669 = 0l;
                                        while (while_method_0(v669)){
                                            int v671;
                                            v671 = v112[v669];
                                            bool v673;
                                            v673 = v116 == v669;
                                            int v674;
                                            if (v673){
                                                v674 = v666;
                                            } else {
                                                v674 = v671;
                                            }
                                            v667[v669] = v674;
                                            v669 += 1l ;
                                        }
                                        static_array<int,2l> v675;
                                        int v677;
                                        v677 = 0l;
                                        while (while_method_0(v677)){
                                            int v679;
                                            v679 = v621[v677];
                                            int v681;
                                            v681 = v667[v677];
                                            int v683;
                                            v683 = v679 - v681;
                                            v675[v677] = v683;
                                            v677 += 1l ;
                                        }
                                        bool v684;
                                        v684 = v660 >= v110;
                                        int v685;
                                        if (v684){
                                            v685 = v660;
                                        } else {
                                            v685 = v110;
                                        }
                                        int v686;
                                        v686 = v113 + 1l;
                                        v691 = try_round_36(v685, v111, v667, v686, v675, v115);
                                        break;
                                    }
                                    case 1: { // A_Call
                                        static_array<int,2l> v505;
                                        int v507;
                                        v507 = 0l;
                                        while (while_method_0(v507)){
                                            int v509;
                                            v509 = v114[v507];
                                            int v511;
                                            v511 = v112[v507];
                                            int v513;
                                            v513 = v509 + v511;
                                            v505[v507] = v513;
                                            v507 += 1l ;
                                        }
                                        int v514;
                                        v514 = v112[0l];
                                        int v516; int v517;
                                        Tuple4 tmp49 = Tuple4{1l, v514};
                                        v516 = tmp49.v0; v517 = tmp49.v1;
                                        while (while_method_0(v516)){
                                            int v519;
                                            v519 = v112[v516];
                                            bool v521;
                                            v521 = v517 >= v519;
                                            int v522;
                                            if (v521){
                                                v522 = v517;
                                            } else {
                                                v522 = v519;
                                            }
                                            v517 = v522;
                                            v516 += 1l ;
                                        }
                                        int v523;
                                        v523 = v505[v116];
                                        bool v525;
                                        v525 = v517 < v523;
                                        int v526;
                                        if (v525){
                                            v526 = v517;
                                        } else {
                                            v526 = v523;
                                        }
                                        static_array<int,2l> v527;
                                        int v529;
                                        v529 = 0l;
                                        while (while_method_0(v529)){
                                            int v531;
                                            v531 = v112[v529];
                                            bool v533;
                                            v533 = v116 == v529;
                                            int v534;
                                            if (v533){
                                                v534 = v526;
                                            } else {
                                                v534 = v531;
                                            }
                                            v527[v529] = v534;
                                            v529 += 1l ;
                                        }
                                        static_array<int,2l> v535;
                                        int v537;
                                        v537 = 0l;
                                        while (while_method_0(v537)){
                                            int v539;
                                            v539 = v505[v537];
                                            int v541;
                                            v541 = v527[v537];
                                            int v543;
                                            v543 = v539 - v541;
                                            v535[v537] = v543;
                                            v537 += 1l ;
                                        }
                                        bool v544;
                                        v544 = v116 < 2l;
                                        if (v544){
                                            int v545;
                                            v545 = v113 + 1l;
                                            v691 = try_round_36(v110, v111, v527, v545, v535, v115);
                                        } else {
                                            v691 = go_next_street_38(v110, v111, v527, v113, v535, v115);
                                        }
                                        break;
                                    }
                                    case 2: { // A_Fold
                                        v691 = Union4{Union4_1{v110, v111, v112, v113, v114, v115}};
                                        break;
                                    }
                                    case 3: { // A_Raise
                                        int v549 = v502.case3.v0;
                                        bool v550;
                                        v550 = v110 <= v549;
                                        bool v551;
                                        v551 = v550 == false;
                                        if (v551){
                                            assert("The raise amount must match the minimum." && v550);
                                        } else {
                                        }
                                        static_array<int,2l> v553;
                                        int v555;
                                        v555 = 0l;
                                        while (while_method_0(v555)){
                                            int v557;
                                            v557 = v114[v555];
                                            int v559;
                                            v559 = v112[v555];
                                            int v561;
                                            v561 = v557 + v559;
                                            v553[v555] = v561;
                                            v555 += 1l ;
                                        }
                                        int v562;
                                        v562 = v112[0l];
                                        int v564; int v565;
                                        Tuple4 tmp50 = Tuple4{1l, v562};
                                        v564 = tmp50.v0; v565 = tmp50.v1;
                                        while (while_method_0(v564)){
                                            int v567;
                                            v567 = v112[v564];
                                            bool v569;
                                            v569 = v565 >= v567;
                                            int v570;
                                            if (v569){
                                                v570 = v565;
                                            } else {
                                                v570 = v567;
                                            }
                                            v565 = v570;
                                            v564 += 1l ;
                                        }
                                        int v571;
                                        v571 = v553[v116];
                                        bool v573;
                                        v573 = v565 < v571;
                                        int v574;
                                        if (v573){
                                            v574 = v565;
                                        } else {
                                            v574 = v571;
                                        }
                                        static_array<int,2l> v575;
                                        int v577;
                                        v577 = 0l;
                                        while (while_method_0(v577)){
                                            int v579;
                                            v579 = v112[v577];
                                            bool v581;
                                            v581 = v116 == v577;
                                            int v582;
                                            if (v581){
                                                v582 = v574;
                                            } else {
                                                v582 = v579;
                                            }
                                            v575[v577] = v582;
                                            v577 += 1l ;
                                        }
                                        static_array<int,2l> v583;
                                        int v585;
                                        v585 = 0l;
                                        while (while_method_0(v585)){
                                            int v587;
                                            v587 = v553[v585];
                                            int v589;
                                            v589 = v575[v585];
                                            int v591;
                                            v591 = v587 - v589;
                                            v583[v585] = v591;
                                            v585 += 1l ;
                                        }
                                        int v592;
                                        v592 = v583[v116];
                                        bool v594;
                                        v594 = v549 < v592;
                                        bool v595;
                                        v595 = v594 == false;
                                        if (v595){
                                            assert("The raise amount must be less than the stack size after calling." && v594);
                                        } else {
                                        }
                                        int v597;
                                        v597 = v565 + v549;
                                        int v598;
                                        v598 = v553[v116];
                                        bool v600;
                                        v600 = v597 < v598;
                                        int v601;
                                        if (v600){
                                            v601 = v597;
                                        } else {
                                            v601 = v598;
                                        }
                                        static_array<int,2l> v602;
                                        int v604;
                                        v604 = 0l;
                                        while (while_method_0(v604)){
                                            int v606;
                                            v606 = v112[v604];
                                            bool v608;
                                            v608 = v116 == v604;
                                            int v609;
                                            if (v608){
                                                v609 = v601;
                                            } else {
                                                v609 = v606;
                                            }
                                            v602[v604] = v609;
                                            v604 += 1l ;
                                        }
                                        static_array<int,2l> v610;
                                        int v612;
                                        v612 = 0l;
                                        while (while_method_0(v612)){
                                            int v614;
                                            v614 = v553[v612];
                                            int v616;
                                            v616 = v602[v612];
                                            int v618;
                                            v618 = v614 - v616;
                                            v610[v612] = v618;
                                            v612 += 1l ;
                                        }
                                        int v619;
                                        v619 = v113 + 1l;
                                        v691 = try_round_36(v549, v111, v602, v619, v610, v115);
                                        break;
                                    }
                                    default: {
                                        assert("Invalid tag." && false);
                                    }
                                }
                                v1001 = Union3{Union3_1{v691}};
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                            }
                        }
                        break;
                    }
                    case 5: { // G_Round'
                        int v696 = v16.case5.v0; static_array<static_array<unsigned char,2l>,2l> v697 = v16.case5.v1; static_array<int,2l> v698 = v16.case5.v2; int v699 = v16.case5.v3; static_array<int,2l> v700 = v16.case5.v4; Union5 v701 = v16.case5.v5; Union1 v702 = v16.case5.v6;
                        int v703;
                        v703 = v699 % 2l;
                        Union6 v704;
                        v704 = Union6{Union6_2{v703, v702}};
                        v11.push(v704);
                        Union4 v892;
                        switch (v702.tag) {
                            case 0: { // A_All_In
                                static_array<int,2l> v822;
                                int v824;
                                v824 = 0l;
                                while (while_method_0(v824)){
                                    int v826;
                                    v826 = v700[v824];
                                    int v828;
                                    v828 = v698[v824];
                                    int v830;
                                    v830 = v826 + v828;
                                    v822[v824] = v830;
                                    v824 += 1l ;
                                }
                                int v831;
                                v831 = v698[0l];
                                int v833; int v834;
                                Tuple4 tmp51 = Tuple4{1l, v831};
                                v833 = tmp51.v0; v834 = tmp51.v1;
                                while (while_method_0(v833)){
                                    int v836;
                                    v836 = v698[v833];
                                    bool v838;
                                    v838 = v834 >= v836;
                                    int v839;
                                    if (v838){
                                        v839 = v834;
                                    } else {
                                        v839 = v836;
                                    }
                                    v834 = v839;
                                    v833 += 1l ;
                                }
                                int v840;
                                v840 = v822[v703];
                                bool v842;
                                v842 = v834 < v840;
                                int v843;
                                if (v842){
                                    v843 = v834;
                                } else {
                                    v843 = v840;
                                }
                                static_array<int,2l> v844;
                                int v846;
                                v846 = 0l;
                                while (while_method_0(v846)){
                                    int v848;
                                    v848 = v698[v846];
                                    bool v850;
                                    v850 = v703 == v846;
                                    int v851;
                                    if (v850){
                                        v851 = v843;
                                    } else {
                                        v851 = v848;
                                    }
                                    v844[v846] = v851;
                                    v846 += 1l ;
                                }
                                static_array<int,2l> v852;
                                int v854;
                                v854 = 0l;
                                while (while_method_0(v854)){
                                    int v856;
                                    v856 = v822[v854];
                                    int v858;
                                    v858 = v844[v854];
                                    int v860;
                                    v860 = v856 - v858;
                                    v852[v854] = v860;
                                    v854 += 1l ;
                                }
                                int v861;
                                v861 = v852[v703];
                                int v863;
                                v863 = v834 + v861;
                                int v864;
                                v864 = v822[v703];
                                bool v866;
                                v866 = v863 < v864;
                                int v867;
                                if (v866){
                                    v867 = v863;
                                } else {
                                    v867 = v864;
                                }
                                static_array<int,2l> v868;
                                int v870;
                                v870 = 0l;
                                while (while_method_0(v870)){
                                    int v872;
                                    v872 = v698[v870];
                                    bool v874;
                                    v874 = v703 == v870;
                                    int v875;
                                    if (v874){
                                        v875 = v867;
                                    } else {
                                        v875 = v872;
                                    }
                                    v868[v870] = v875;
                                    v870 += 1l ;
                                }
                                static_array<int,2l> v876;
                                int v878;
                                v878 = 0l;
                                while (while_method_0(v878)){
                                    int v880;
                                    v880 = v822[v878];
                                    int v882;
                                    v882 = v868[v878];
                                    int v884;
                                    v884 = v880 - v882;
                                    v876[v878] = v884;
                                    v878 += 1l ;
                                }
                                bool v885;
                                v885 = v861 >= v696;
                                int v886;
                                if (v885){
                                    v886 = v861;
                                } else {
                                    v886 = v696;
                                }
                                int v887;
                                v887 = v699 + 1l;
                                v892 = try_round_36(v886, v697, v868, v887, v876, v701);
                                break;
                            }
                            case 1: { // A_Call
                                static_array<int,2l> v706;
                                int v708;
                                v708 = 0l;
                                while (while_method_0(v708)){
                                    int v710;
                                    v710 = v700[v708];
                                    int v712;
                                    v712 = v698[v708];
                                    int v714;
                                    v714 = v710 + v712;
                                    v706[v708] = v714;
                                    v708 += 1l ;
                                }
                                int v715;
                                v715 = v698[0l];
                                int v717; int v718;
                                Tuple4 tmp52 = Tuple4{1l, v715};
                                v717 = tmp52.v0; v718 = tmp52.v1;
                                while (while_method_0(v717)){
                                    int v720;
                                    v720 = v698[v717];
                                    bool v722;
                                    v722 = v718 >= v720;
                                    int v723;
                                    if (v722){
                                        v723 = v718;
                                    } else {
                                        v723 = v720;
                                    }
                                    v718 = v723;
                                    v717 += 1l ;
                                }
                                int v724;
                                v724 = v706[v703];
                                bool v726;
                                v726 = v718 < v724;
                                int v727;
                                if (v726){
                                    v727 = v718;
                                } else {
                                    v727 = v724;
                                }
                                static_array<int,2l> v728;
                                int v730;
                                v730 = 0l;
                                while (while_method_0(v730)){
                                    int v732;
                                    v732 = v698[v730];
                                    bool v734;
                                    v734 = v703 == v730;
                                    int v735;
                                    if (v734){
                                        v735 = v727;
                                    } else {
                                        v735 = v732;
                                    }
                                    v728[v730] = v735;
                                    v730 += 1l ;
                                }
                                static_array<int,2l> v736;
                                int v738;
                                v738 = 0l;
                                while (while_method_0(v738)){
                                    int v740;
                                    v740 = v706[v738];
                                    int v742;
                                    v742 = v728[v738];
                                    int v744;
                                    v744 = v740 - v742;
                                    v736[v738] = v744;
                                    v738 += 1l ;
                                }
                                bool v745;
                                v745 = v703 < 2l;
                                if (v745){
                                    int v746;
                                    v746 = v699 + 1l;
                                    v892 = try_round_36(v696, v697, v728, v746, v736, v701);
                                } else {
                                    v892 = go_next_street_38(v696, v697, v728, v699, v736, v701);
                                }
                                break;
                            }
                            case 2: { // A_Fold
                                v892 = Union4{Union4_1{v696, v697, v698, v699, v700, v701}};
                                break;
                            }
                            case 3: { // A_Raise
                                int v750 = v702.case3.v0;
                                bool v751;
                                v751 = v696 <= v750;
                                bool v752;
                                v752 = v751 == false;
                                if (v752){
                                    assert("The raise amount must match the minimum." && v751);
                                } else {
                                }
                                static_array<int,2l> v754;
                                int v756;
                                v756 = 0l;
                                while (while_method_0(v756)){
                                    int v758;
                                    v758 = v700[v756];
                                    int v760;
                                    v760 = v698[v756];
                                    int v762;
                                    v762 = v758 + v760;
                                    v754[v756] = v762;
                                    v756 += 1l ;
                                }
                                int v763;
                                v763 = v698[0l];
                                int v765; int v766;
                                Tuple4 tmp53 = Tuple4{1l, v763};
                                v765 = tmp53.v0; v766 = tmp53.v1;
                                while (while_method_0(v765)){
                                    int v768;
                                    v768 = v698[v765];
                                    bool v770;
                                    v770 = v766 >= v768;
                                    int v771;
                                    if (v770){
                                        v771 = v766;
                                    } else {
                                        v771 = v768;
                                    }
                                    v766 = v771;
                                    v765 += 1l ;
                                }
                                int v772;
                                v772 = v754[v703];
                                bool v774;
                                v774 = v766 < v772;
                                int v775;
                                if (v774){
                                    v775 = v766;
                                } else {
                                    v775 = v772;
                                }
                                static_array<int,2l> v776;
                                int v778;
                                v778 = 0l;
                                while (while_method_0(v778)){
                                    int v780;
                                    v780 = v698[v778];
                                    bool v782;
                                    v782 = v703 == v778;
                                    int v783;
                                    if (v782){
                                        v783 = v775;
                                    } else {
                                        v783 = v780;
                                    }
                                    v776[v778] = v783;
                                    v778 += 1l ;
                                }
                                static_array<int,2l> v784;
                                int v786;
                                v786 = 0l;
                                while (while_method_0(v786)){
                                    int v788;
                                    v788 = v754[v786];
                                    int v790;
                                    v790 = v776[v786];
                                    int v792;
                                    v792 = v788 - v790;
                                    v784[v786] = v792;
                                    v786 += 1l ;
                                }
                                int v793;
                                v793 = v784[v703];
                                bool v795;
                                v795 = v750 < v793;
                                bool v796;
                                v796 = v795 == false;
                                if (v796){
                                    assert("The raise amount must be less than the stack size after calling." && v795);
                                } else {
                                }
                                int v798;
                                v798 = v766 + v750;
                                int v799;
                                v799 = v754[v703];
                                bool v801;
                                v801 = v798 < v799;
                                int v802;
                                if (v801){
                                    v802 = v798;
                                } else {
                                    v802 = v799;
                                }
                                static_array<int,2l> v803;
                                int v805;
                                v805 = 0l;
                                while (while_method_0(v805)){
                                    int v807;
                                    v807 = v698[v805];
                                    bool v809;
                                    v809 = v703 == v805;
                                    int v810;
                                    if (v809){
                                        v810 = v802;
                                    } else {
                                        v810 = v807;
                                    }
                                    v803[v805] = v810;
                                    v805 += 1l ;
                                }
                                static_array<int,2l> v811;
                                int v813;
                                v813 = 0l;
                                while (while_method_0(v813)){
                                    int v815;
                                    v815 = v754[v813];
                                    int v817;
                                    v817 = v803[v813];
                                    int v819;
                                    v819 = v815 - v817;
                                    v811[v813] = v819;
                                    v813 += 1l ;
                                }
                                int v820;
                                v820 = v699 + 1l;
                                v892 = try_round_36(v750, v697, v803, v820, v811, v701);
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                            }
                        }
                        v1001 = Union3{Union3_1{v892}};
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
                                asm("exit;");
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
                        Tuple0 tmp78 = score_54(v43);
                        v54 = tmp78.v0; v55 = tmp78.v1;
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
                        Tuple0 tmp79 = score_54(v58);
                        v69 = tmp79.v0; v70 = tmp79.v1;
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
                        Union11 v97;
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
                                    bool v88;
                                    v88 = v84 < v86;
                                    Union11 v94;
                                    if (v88){
                                        v94 = Union11{Union11_2{}};
                                    } else {
                                        bool v90;
                                        v90 = v84 > v86;
                                        if (v90){
                                            v94 = Union11{Union11_1{}};
                                        } else {
                                            v94 = Union11{Union11_0{}};
                                        }
                                    }
                                    bool v95;
                                    switch (v94.tag) {
                                        case 0: { // Eq
                                            v95 = true;
                                            break;
                                        }
                                        default: {
                                            v95 = false;
                                        }
                                    }
                                    bool v96;
                                    v96 = v95 == false;
                                    if (v96){
                                        v81 = v94;
                                        break;
                                    } else {
                                    }
                                    v82 += 1l ;
                                }
                                v97 = v81;
                                break;
                            }
                            default: {
                                v97 = v80;
                            }
                        }
                        int v102; int v103;
                        switch (v97.tag) {
                            case 0: { // Eq
                                v102 = 0l; v103 = -1l;
                                break;
                            }
                            case 1: { // Gt
                                v102 = v72; v103 = 0l;
                                break;
                            }
                            case 2: { // Lt
                                v102 = v72; v103 = 1l;
                                break;
                            }
                            default: {
                                assert("Invalid tag." && false);
                            }
                        }
                        static_array<Tuple0,2l> v104;
                        v104[0l] = Tuple0{v54, v55};
                        v104[1l] = Tuple0{v69, v70};
                        Union6 v106;
                        v106 = Union6{Union6_4{v102, v104, v103}};
                        v11.push(v106);
                        Union7 v107;
                        v107 = Union7{Union7_1{v32, v33, v34, v35, v36, v37}};
                        v9 = v107;
                        Union3 v108;
                        v108 = Union3{Union3_0{}};
                        v6 = v108;
                        v1001 = Union3{Union3_0{}};
                        break;
                    }
                    case 7: { // G_Turn
                        int v911 = v16.case7.v0; static_array<static_array<unsigned char,2l>,2l> v912 = v16.case7.v1; static_array<int,2l> v913 = v16.case7.v2; int v914 = v16.case7.v3; static_array<int,2l> v915 = v16.case7.v4; Union5 v916 = v16.case7.v5;
                        static_array<unsigned char,1l> v917; unsigned long long v918;
                        Tuple12 tmp80 = draw_cards_40(v0, v12);
                        v917 = tmp80.v0; v918 = tmp80.v1;
                        v5 = v918;
                        static_array_list<unsigned char,5l> v919;
                        v919 = get_community_cards_41(v916, v917);
                        Union6 v920;
                        v920 = Union6{Union6_0{v919}};
                        v11.push(v920);
                        Union5 v935;
                        switch (v916.tag) {
                            case 0: { // Flop
                                static_array<unsigned char,3l> v921 = v916.case0.v0;
                                static_array<unsigned char,4l> v922;
                                int v924;
                                v924 = 0l;
                                while (while_method_1(v924)){
                                    unsigned char v926;
                                    v926 = v921[v924];
                                    v922[v924] = v926;
                                    v924 += 1l ;
                                }
                                int v928;
                                v928 = 0l;
                                while (while_method_6(v928)){
                                    unsigned char v930;
                                    v930 = v917[v928];
                                    int v932;
                                    v932 = 3l + v928;
                                    v922[v932] = v930;
                                    v928 += 1l ;
                                }
                                v935 = Union5{Union5_3{v922}};
                                break;
                            }
                            default: {
                                printf("%s\n", "Invalid street in turn.");
                                asm("exit;");
                            }
                        }
                        int v936;
                        v936 = 2l;
                        int v937;
                        v937 = 0l;
                        Union4 v938;
                        v938 = try_round_36(v936, v912, v913, v937, v915, v935);
                        v1001 = Union3{Union3_1{v938}};
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
                    }
                }
                break;
            }
            default: {
                assert("Invalid tag." && false);
            }
        }
        v14 = v1001;
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
            assert("Invalid tag." && false);
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
            assert("Invalid tag." && false);
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
            assert("Invalid tag." && false);
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
            assert("Invalid tag." && false);
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
            assert("Invalid tag." && false);
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
        Tuple0 tmp81 = v2[v6];
        v13 = tmp81.v0; v14 = tmp81.v1;
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
            assert("Invalid tag." && false);
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
            assert("Invalid tag." && false);
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
            assert("Invalid tag." && false);
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
            assert("Invalid tag." && false);
        }
    }
}
__device__ inline bool while_method_21(){
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
                        asm("exit;");
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
                                asm("exit;");
                            }
                        }
                        break;
                    }
                    default: {
                        assert("Invalid tag." && false);
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
                assert("Invalid tag." && false);
            }
        }
        f_55(v0, v12, v13, v14, v15, v16);
    } else {
    }
    atomicAdd(&v6,-1l);
    while (while_method_21()){
        bool v52;
        v52 = 67108864ull == v5;
        bool v53;
        v53 = v52 == false;
        if (v53){
            assert("The params needs to have matching offsets." && v52);
        } else {
        }
        bool v55;
        v55 = 1573376ull == v3;
        bool v56;
        v56 = v55 == false;
        if (v56){
            assert("The outputs needs to have matching offsets." && v55);
        } else {
        }
        Union8 v58;
        v58 = Union8{Union8_0{}};
        Union9 v59; float * v60; int v61; int v62; int v63; float v64;
        Tuple13 tmp82 = noinline_run_42(v2, v4, v58);
        v59 = tmp82.v0; v60 = tmp82.v1; v61 = tmp82.v2; v62 = tmp82.v3; v63 = tmp82.v4; v64 = tmp82.v5;
        int v65 = v6;
        bool v66;
        v66 = v65 == 0l;
        if (v66){
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
options.append('--diag-suppress=550,20012,68')
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
        v8 = cp.empty(67108864,dtype=cp.uint8)
        v9 = cp.empty(1573376,dtype=cp.uint8)
        v11 = v8[0:0+4*16777216].view(cp.float32)
        v12 = cp.random.normal(0.0,1.0,16777216,dtype=cp.float32) # type: ignore
        cp.copyto(v11[0:0+16777216],v12[0:0+16777216])
        del v11, v12
        v13 = 4503599627370495
        v14 = US3_0()
        v15 = US6_0()
        v16 = 1573376
        v17 = 67108864
        return method109(v13, v14, v7, v1, v15, v9, v16, v8, v17)
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
def main():
    v0 = Closure0()
    v1 = Closure1()
    v2 = collections.namedtuple("HU_Holdem_Game",['event_loop_gpu', 'init'])(v0, v1)
    del v0, v1
    return v2

if __name__ == '__main__': print(main())
